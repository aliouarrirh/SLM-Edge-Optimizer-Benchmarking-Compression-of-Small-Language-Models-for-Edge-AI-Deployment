"""
Projet académique : Optimisation et Compression de DistilGPT-2 pour l'Edge AI
Script Modal autonome — lancer avec : modal run modal_distilgpt2.py
Pour un dry run (validation sans dépenser de crédits) :
    modal run modal_distilgpt2.py --dry-run
"""

import modal
import time
import os
import json
import math
import gc
import numpy as np

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
CONFIG = {
    "model_name": "distilgpt2",
    "dataset_name": "roneneldan/TinyStories",
    # Sous-ensembles réduits pour rester dans le budget
    "n_train": 3000,
    "n_val": 500,
    "max_length": 128,
    "n_epochs": 2,
    "batch_size": 8,
    "grad_accum": 4,          # batch effectif = 32
    "warmup_steps": 50,
    "max_grad_norm": 1.0,
    # Taux d'apprentissage adaptés par optimiseur
    "lr": {
        "AdamW":     5e-4,
        "Adafactor": 1e-3,
        "Lion":      1e-4,
        "LAMB":      1e-3,
        "SGD":       1e-3,
    },
    # Dry run : 50 steps, 1 seul optimiseur
    "dry_run_steps": 50,
    "dry_run_n_train": 200,
    "dry_run_n_val": 100,
    # Prompts de génération qualitative
    "prompts": [
        "Once upon a time, there was a little",
        "The dragon lived in a big",
        "Lily found a magic",
    ],
}

# ============================================================
# IMAGE MODAL — toutes les dépendances
# ============================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "torch>=2.3.0",
        "evaluate>=0.4.0",
        "lion-pytorch>=0.1.2",
        "torch_optimizer>=0.3.0",
        "matplotlib>=3.8.0",
        "pandas>=2.0.0",
        "accelerate>=0.30.0",
        "tqdm",
        "scipy",
        "numpy>=1.26.0",
    )
)

app = modal.App("distilgpt2-edge-ai", image=image)

# Volume persistant : survit entre les runs et les sessions
volume = modal.Volume.from_name("distilgpt2-results-v1", create_if_missing=True)
VOL = "/results"


# ============================================================
# UTILITAIRES (exécutés dans le conteneur Modal)
# ============================================================

def _get_optimizer(name: str, params, lr: float):
    """Instancie l'optimiseur demandé."""
    if name == "AdamW":
        from torch.optim import AdamW
        return AdamW(params, lr=lr, weight_decay=0.01)
    elif name == "Adafactor":
        from transformers.optimization import Adafactor
        # relative_step=False : on fournit notre propre LR
        return Adafactor(
            params, lr=lr,
            relative_step=False, scale_parameter=False, warmup_init=False,
        )
    elif name == "Lion":
        from lion_pytorch import Lion
        return Lion(params, lr=lr, weight_decay=0.01)
    elif name == "LAMB":
        import torch_optimizer
        return torch_optimizer.Lamb(params, lr=lr, weight_decay=0.01)
    elif name == "SGD":
        from torch.optim import SGD
        return SGD(params, lr=lr, momentum=0.9, weight_decay=0.01, nesterov=True)
    else:
        raise ValueError(f"Optimiseur inconnu : {name}")


def _compute_perplexity(model, dataloader, device):
    """Perplexité sur un DataLoader (log-loss moyen → exp)."""
    import torch
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
            total_loss += loss.item()
            n += 1
    return math.exp(total_loss / max(n, 1))


def _model_size_mb(model) -> float:
    """Taille du state_dict en Mo."""
    import torch, tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / (1024 ** 2)


def _measure_inference(model, tokenizer, prompt: str, device, n_runs: int = 10):
    """Retourne (latence_moy_s, tokens_par_sec)."""
    import torch
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Warm-up
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    latencies = []
    n_new = 20
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        latencies.append(time.perf_counter() - t0)

    avg = float(np.mean(latencies))
    return avg, n_new / avg


def _generate_texts(model, tokenizer, device) -> dict:
    """Génère des exemples qualitatifs sur les prompts du config."""
    import torch
    model.eval()
    examples = {}
    for prompt in CONFIG["prompts"]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=60,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        examples[prompt] = tokenizer.decode(out[0], skip_special_tokens=True)
    return examples


# ============================================================
# FONCTION PRINCIPALE MODAL
# ============================================================

@app.function(
    gpu="T4",          # T4 = $0.59/h — le plus économique avec GPU
    timeout=5400,      # 90 min max (filet de sécurité budget)
    volumes={VOL: volume},
    memory=8192,
)
def run_full_experiment(dry_run: bool = False):
    """
    Pipeline complet :
      1. Fine-tuning DistilGPT-2 avec 5 optimiseurs sur TinyStories
      2. Évaluation perplexité avant compression
      3. Quantification dynamique (torch.quantization, int8, CPU)
      4. Évaluation après compression
      5. Courbes d'entraînement (PNG)
      6. Tableau comparatif (CSV)
      7. Exemples de génération (JSON)
    """
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset

    os.makedirs(VOL, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  DistilGPT-2 Edge-AI — {'DRY RUN' if dry_run else 'RUN COMPLET'}")
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"{'='*65}\n")

    t_global = time.time()

    # ----------------------------------------------------------
    # 1. CHARGEMENT DU DATASET
    # ----------------------------------------------------------
    n_train = CONFIG["dry_run_n_train"] if dry_run else CONFIG["n_train"]
    n_val   = CONFIG["dry_run_n_val"]   if dry_run else CONFIG["n_val"]
    n_epochs = 1                          if dry_run else CONFIG["n_epochs"]
    max_steps = CONFIG["dry_run_steps"]   if dry_run else None

    print(f"📚 Chargement TinyStories ({n_train} train / {n_val} val)...")
    raw_train = load_dataset("roneneldan/TinyStories", split="train")
    raw_val   = load_dataset("roneneldan/TinyStories", split="validation")
    train_texts = [raw_train[i]["text"] for i in range(n_train)]
    val_texts   = [raw_val[i]["text"]   for i in range(n_val)]

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    class StoryDataset(Dataset):
        def __init__(self, texts):
            self.enc = tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=CONFIG["max_length"], return_tensors="pt",
            )
        def __len__(self): return self.enc["input_ids"].shape[0]
        def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()}

    train_loader = DataLoader(StoryDataset(train_texts), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader   = DataLoader(StoryDataset(val_texts),   batch_size=CONFIG["batch_size"], num_workers=2)
    print(f"✅ Dataset prêt — {len(train_loader)} batches/epoch\n")

    # ----------------------------------------------------------
    # 2. FINE-TUNING — 5 optimiseurs
    # ----------------------------------------------------------
    opt_names = ["AdamW"] if dry_run else list(CONFIG["lr"].keys())
    results: dict = {}
    all_losses: dict = {}

    for opt_name in opt_names:
        print(f"{'─'*65}")
        print(f"🔧  Optimiseur : {opt_name}  (lr={CONFIG['lr'][opt_name]})")
        print(f"{'─'*65}")
        t_opt = time.time()

        model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
        optimizer = _get_optimizer(opt_name, model.parameters(), CONFIG["lr"][opt_name])

        total_steps = len(train_loader) * n_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(CONFIG["warmup_steps"], total_steps // 4),
            num_training_steps=total_steps,
        )

        step_losses, global_step = [], 0

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            for batch in train_loader:
                if max_steps and global_step >= max_steps:
                    break

                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
                (loss / CONFIG["grad_accum"]).backward()

                if (global_step + 1) % CONFIG["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                step_losses.append(loss.item())
                global_step += 1

                if global_step % 100 == 0:
                    elapsed = time.time() - t_global
                    cost_est = elapsed / 3600 * 0.59
                    print(f"  step {global_step:4d} | loss {loss.item():.4f} | ⏱ {elapsed:.0f}s | 💰 ~${cost_est:.3f}")

            print(f"  → Epoch {epoch+1}/{n_epochs} terminée | loss moy: {np.mean(step_losses[-len(train_loader):]):.4f}")

        # Évaluation avant compression (GPU)
        print(f"  📊 Perplexité validation...")
        ppl_before = _compute_perplexity(model, val_loader, device)
        size_before = _model_size_mb(model)
        latency_b, tps_b = _measure_inference(model, tokenizer, CONFIG["prompts"][0], device)

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_mem = 0.0

        train_time = time.time() - t_opt
        elapsed_total = time.time() - t_global
        cost_total = elapsed_total / 3600 * 0.59

        print(f"  ✅ PPL={ppl_before:.2f} | Taille={size_before:.1f}MB | ⏱ train={train_time:.0f}s | 💰 cumulé ~${cost_total:.3f}")

        results[opt_name] = {
            "optimiseur": opt_name,
            "taille_avant_MB": round(size_before, 2),
            "taille_après_MB": None,
            "compression_ratio": None,
            "perplexity_avant": round(ppl_before, 2),
            "perplexity_après": None,
            "latence_avant_s": round(latency_b, 4),
            "latence_après_s": None,
            "tokens_par_sec_avant": round(tps_b, 1),
            "tokens_par_sec_après": None,
            "mémoire_peak_GPU_GB": round(peak_mem, 3),
            "temps_entraînement_s": round(train_time, 1),
        }
        all_losses[opt_name] = step_losses

        # Sauvegarde du modèle fine-tuné
        if not dry_run:
            save_path = f"{VOL}/model_{opt_name}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            volume.commit()
            print(f"  💾 Modèle sauvegardé → {save_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ----------------------------------------------------------
    # 3. QUANTIFICATION DYNAMIQUE (CPU, int8)
    # ----------------------------------------------------------
    best_opt = min(results, key=lambda k: results[k]["perplexity_avant"])
    print(f"\n🏆 Meilleur optimiseur : {best_opt} (PPL={results[best_opt]['perplexity_avant']})\n")

    # On quantifie tous les modèles (ou juste le meilleur en dry run)
    to_quantize = [best_opt] if dry_run else list(results.keys())
    cpu_dev = torch.device("cpu")

    for opt_name in to_quantize:
        print(f"{'─'*65}")
        print(f"🗜️  Quantification dynamique — {opt_name}")

        if dry_run:
            base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        else:
            base_model = AutoModelForCausalLM.from_pretrained(f"{VOL}/model_{opt_name}")

        # Dynamic quantization : remplace nn.Linear par versions int8
        q_model = torch.quantization.quantize_dynamic(
            base_model.cpu(),
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

        size_after = _model_size_mb(q_model)
        size_before = results[opt_name]["taille_avant_MB"]
        ratio = size_before / size_after

        ppl_after = _compute_perplexity(q_model, val_loader, cpu_dev)
        lat_after, tps_after = _measure_inference(q_model, tokenizer, CONFIG["prompts"][0], cpu_dev)

        results[opt_name].update({
            "taille_après_MB": round(size_after, 2),
            "compression_ratio": round(ratio, 2),
            "perplexity_après": round(ppl_after, 2),
            "latence_après_s": round(lat_after, 4),
            "tokens_par_sec_après": round(tps_after, 1),
        })

        print(f"  Taille  : {size_before:.1f} → {size_after:.1f} MB  (x{ratio:.2f})")
        print(f"  PPL     : {results[opt_name]['perplexity_avant']} → {ppl_after:.2f}")
        print(f"  Latence : {results[opt_name]['latence_avant_s']:.4f} → {lat_after:.4f} s")
        print(f"  TPS     : {results[opt_name]['tokens_par_sec_avant']} → {tps_after:.1f} tok/s")

        # Exemples de génération qualitatifs
        before_texts = {}
        if dry_run:
            ref_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(cpu_dev)
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(f"{VOL}/model_{opt_name}").to(cpu_dev)

        before_texts = _generate_texts(ref_model, tokenizer, cpu_dev)
        after_texts  = _generate_texts(q_model, tokenizer, cpu_dev)

        with open(f"{VOL}/text_examples_{opt_name}.json", "w", encoding="utf-8") as f:
            json.dump({"avant_compression": before_texts, "après_compression": after_texts}, f,
                      ensure_ascii=False, indent=2)

        del base_model, q_model, ref_model
        gc.collect()

    # ----------------------------------------------------------
    # 4. COURBES D'ENTRAÎNEMENT
    # ----------------------------------------------------------
    print(f"\n📈 Tracé des courbes d'entraînement...")
    n_plots = len(all_losses)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)

    for ax, (opt_name, losses) in zip(axes[0], all_losses.items()):
        xs = list(range(len(losses)))
        window = max(1, len(losses) // 30)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(xs, losses, alpha=0.25, color="#4C72B0")
        ax.plot(range(window - 1, len(losses)), smoothed, color="#4C72B0", linewidth=2)
        ppl = results[opt_name]["perplexity_avant"]
        ax.set_title(f"{opt_name}\nPPL={ppl:.1f}", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (CE)")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Courbes d'entraînement — DistilGPT-2 / TinyStories", fontsize=13, y=1.02)
    plt.tight_layout()
    curve_path = f"{VOL}/training_curves.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé → {curve_path}")

    # ----------------------------------------------------------
    # 5. TABLEAU COMPARATIF
    # ----------------------------------------------------------
    df = pd.DataFrame(list(results.values()))
    # Trier par perplexité avant compression
    df = df.sort_values("perplexity_avant")
    csv_path = f"{VOL}/resultats_comparatifs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📋 Tableau comparatif :")
    print(df.to_string(index=False))

    # ----------------------------------------------------------
    # 6. RÉSUMÉ FINAL
    # ----------------------------------------------------------
    total_time = time.time() - t_global
    total_cost = total_time / 3600 * 0.59  # T4 = ~$0.59/h

    summary = {
        "meilleur_optimiseur": best_opt,
        "meilleur_ppl_avant": results[best_opt]["perplexity_avant"],
        "meilleur_ppl_après": results[best_opt].get("perplexity_après"),
        "meilleur_compression_ratio": results[best_opt].get("compression_ratio"),
        "temps_total_GPU_s": round(total_time, 1),
        "temps_total_GPU_min": round(total_time / 60, 2),
        "coût_estimé_USD": round(total_cost, 3),
        "dry_run": dry_run,
    }
    with open(f"{VOL}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    print(f"\n{'='*65}")
    print(f"  ✅  Expérience terminée !")
    print(f"  ⏱   Temps GPU total : {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  💰  Coût estimé (T4 $0.59/h) : ~${total_cost:.3f}")
    print(f"  🏆  Meilleur optimiseur : {best_opt}")
    print(f"\n  📥  Récupérer les résultats :")
    print(f"      modal volume get distilgpt2-results-v1 / ./results/")
    print(f"{'='*65}\n")

    return summary


# ============================================================
# ENTRYPOINT LOCAL
# ============================================================
@app.local_entrypoint()
def main(dry_run: bool = False):
    """
    Usage :
        modal run modal_distilgpt2.py              # run complet GPU
        modal run modal_distilgpt2.py --dry-run    # dry run rapide
    """
    mode = "DRY RUN (CPU, 50 steps)" if dry_run else "RUN COMPLET (GPU T4)"
    print(f"\n🚀 Lancement — {mode}")
    print(f"   Volume de résultats : distilgpt2-results-v1\n")

    result = run_full_experiment.remote(dry_run=dry_run)

    print("\n📊 Résumé retourné :")
    for k, v in result.items():
        print(f"   {k:35s} : {v}")
