# Rapport — Optimisation et compression des Small Language Models pour Edge AI

**Étudiants :** [Prénom NOM], [Prénom NOM]  
**Module :** Optimisation  
**Date :** Juin 2026

---

## 1. Introduction et objectif

Les Small Language Models (SLM) sont des modèles de langage de taille réduite, adaptés aux environnements à ressources limitées (smartphones, objets connectés, systèmes embarqués). Même allégés, ces modèles restent coûteux en mémoire, en énergie et en temps d'inférence.

**Objectif du projet :** étudier comment les algorithmes d'optimisation et les techniques de compression peuvent être combinés pour obtenir un SLM plus léger, plus rapide et déployable dans un contexte Edge AI.

---

## 2. Choix du modèle et du dataset

### Modèle : DistilGPT-2 (82M paramètres)

DistilGPT-2 est une version distillée de GPT-2, réduite à 82M paramètres contre 117M pour GPT-2 Small et 345M pour GPT-2 Medium. Il est adapté au déploiement Edge (taille FP32 ≈ 320 MB, quantifiable à < 100 MB).

| Modèle | Paramètres | Taille FP32 |
|--------|-----------|-------------|
| DistilGPT-2 | 82M | ~320 MB |
| GPT-2 Small | 117M | ~467 MB |
| GPT-2 Medium | 345M | ~1.38 GB |

### Dataset : TinyStories

TinyStories contient ~2.7M courtes histoires synthétiques (~250 mots, vocabulaire simple). Avantages : tâche de génération simple, dataset propre, rapide à itérer. Sous-ensemble utilisé : **30 000 exemples d'entraînement / 5 000 de validation**.

**Métrique principale :** Perplexité (PPL = exp(val_loss)) — plus basse = meilleure qualité de génération.

---

## 3. Pipeline d'entraînement

### Configuration fixe (comparaison équitable entre optimiseurs)

| Hyperparamètre | Valeur |
|----------------|--------|
| Taille du batch | 16 (train) / 32 (val) |
| Longueur de séquence | 128 tokens |
| Nombre d'epochs | 1 |
| Learning rate par défaut | 3e-4 |
| Gradient accumulation | 2 steps |
| Warmup steps | 100 |
| Gradient clipping | 1.0 |

### Optimiseurs comparés

| Optimiseur | Caractéristique | Mémoire par paramètre |
|-----------|----------------|----------------------|
| AdamW | Adam + weight decay découplé | 8 octets (2 états) |
| Adafactor | Moments factorisés | ~3 octets |
| Lion | Mise à jour par signe | 4 octets (1 état) |
| LAMB | Adaptation par couche | 8 octets (2 états) |
| SGD | Gradient + momentum | 4 octets (1 état) |

---

## 4. Résultats — Comparaison des 5 optimiseurs

### 4.1 Performance à lr = 3e-4 (default)

| Optimiseur | Val Loss | Perplexité ↓ | Énergie (kWh) | Mémoire GPU (MB) | Temps/step (ms) |
|-----------|---------|-------------|---------------|-----------------|----------------|
| AdamW | ___ | ___ | ___ | ___ | ___ |
| Adafactor | ___ | ___ | ___ | ___ | ___ |
| Lion | ___ | ___ | ___ | ___ | ___ |
| LAMB | ___ | ___ | ___ | ___ | ___ |
| SGD | ___ | ___ | ___ | ___ | ___ |

*Remplir avec les valeurs de `results/comparison_results.csv`*

### 4.2 Analyse LR sensitivity (3 valeurs par optimiseur)

*(Insérer figure `results/lr_sensitivity.png`)*

Observations :
- AdamW : ...
- Lion : ...
- Adafactor : ...
- LAMB : ...
- SGD : ...

### 4.3 Normes des gradients

*(Insérer figure `results/gradient_norms.png`)*

---

## 5. Évaluation avant compression (modèle baseline)

**Meilleur optimiseur identifié :** ___

| Métrique | Valeur |
|---------|--------|
| Perplexité (val) | ___ |
| Taille modèle (FP32) | ~320 MB |
| Nombre de paramètres | 82 274 048 |
| Latence p50 (GPU, ms) | ___ |
| Latence p95 (GPU, ms) | ___ |
| Tokens/seconde (GPU) | ___ |
| Latence p50 (CPU, ms) | ___ |
| Mémoire GPU d'inférence (MB) | ___ |
| RAM CPU d'inférence (MB) | ___ |

---

## 6. Techniques de compression appliquées

### 6.1 Quantification dynamique INT8

Quantification post-entraînement des couches `nn.Linear` en INT8 (via `torch.quantization.quantize_dynamic`).

| Métrique | Avant | Après INT8 |
|---------|-------|-----------|
| Taille (MB) | ~320 | ___ |
| Compression ratio | 1× | ___× |
| Perplexité | ___ | ___ |
| Δ PPL | — | +___ |
| Latence CPU p50 (ms) | ___ | ___ |

### 6.2 Pruning non structuré (magnitude)

Suppression des poids de plus faible magnitude dans les couches `weight` (seuil percentile).

| Sparsité | % poids nuls | Perplexité | Δ PPL |
|---------|-------------|-----------|-------|
| 0% (baseline) | 0% | ___ | — |
| 10% | ___ | ___ | +___ |
| 20% | ___ | ___ | +___ |
| 30% | ___ | ___ | +___ |
| 40% | ___ | ___ | +___ |
| 50% | ___ | ___ | +___ |

*(Insérer figure `results/sparsity_vs_perplexity.png`)*

**Note :** le pruning non structuré met les poids à zéro sans modifier la forme des tenseurs denses ; le gain de taille réel est visible uniquement sur un checkpoint compressé (sparse storage).

### 6.3 Combiné : Pruning 20% + INT8

| Métrique | Valeur |
|---------|--------|
| Taille (MB) | ___ |
| Compression ratio | ___× |
| Perplexité | ___ |
| Δ PPL vs original | +___ |

### 6.4 Export ONNX

Export vers ONNX + inférence via ONNX Runtime CPU.

| Métrique | Valeur |
|---------|--------|
| Taille ONNX (MB) | ___ |
| Latence CPU p50 (ms) | ___ |
| Latence CPU p95 (ms) | ___ |

---

## 7. Tableau comparatif final (avant / après compression)

*(Insérer figure `results/optimizer_compression_summary.png`)*

| Variante | Taille (MB) | Ratio | PPL | Δ PPL | p50 CPU (ms) | Tok/s | Edge < 100 MB ? |
|---------|------------|-------|-----|-------|-------------|-------|----------------|
| Original FP32 (GPU) | ~320 | 1× | ___ | — | ___ | ___ | ❌ |
| Original FP32 (CPU) | ~320 | 1× | ___ | — | ___ | ___ | ❌ |
| INT8 Quantifié | ___ | ___× | ___ | +___ | ___ | ___ | ___ |
| Pruning 20% | ___ | ___× | ___ | +___ | ___ | ___ | ___ |
| Pruning 30% | ___ | ___× | ___ | +___ | ___ | ___ | ___ |
| Combiné (Q+Prune) | ___ | ___× | ___ | +___ | ___ | ___ | ✅ |
| ONNX CPU | ___ | ___× | ___ | +___ | ___ | ___ | ___ |

*(Remplir avec `results/comparison_results.csv`)*

*(Insérer figure `results/pareto_front.png`)*

---

## 8. Analyse du compromis compression / performance

### 8.1 Impact de la quantification

La quantification INT8 offre un gain de compression d'environ ___× pour une perte de perplexité de ___ points. C'est la technique offrant le meilleur rapport gain/coût.

### 8.2 Impact du pruning

Au-delà de ___% de sparsité, la perplexité augmente significativement. Le pruning à 20% constitue un bon compromis : ___ points de PPL perdus pour ___% de poids supprimés.

### 8.3 Combinaison quantification + pruning

L'approche combinée atteint une compression de ___× tout en restant sous le seuil de 100 MB requis pour le déploiement Edge AI. La dégradation cumulée de PPL reste acceptable (___ points).

### 8.4 Influence de l'optimiseur sur la compression

| Optimiseur | PPL avant compression | PPL après INT8 | PPL après Q+Prune |
|-----------|----------------------|----------------|-------------------|
| AdamW | ___ | ___ | ___ |
| Lion | ___ | ___ | ___ |
| Adafactor | ___ | ___ | ___ |

Observation : un modèle avec une meilleure perplexité initiale (AdamW) conserve une meilleure qualité après compression.

---

## 9. Extension : DistilGPT-2 vs GPT-2 Small (117M)

*(Résultats du notebook `opti_project10_extension_gpt2.ipynb`)*

| Métrique | DistilGPT-2 (82M) | GPT-2 Small (117M) |
|---------|------------------|-------------------|
| Meilleur optimiseur | ___ | ___ |
| Meilleure PPL | ___ | ___ |
| Taille après INT8 (MB) | ___ | ___ |
| Edge < 100 MB ? | ___ | ___ |

---

## 10. Conclusion et recommandation

### Meilleure combinaison pour Edge AI

D'après nos expériences, la combinaison **[Meilleur optimiseur] + Quantification INT8 + Pruning 20%** offre le meilleur compromis :

- Taille finale : ___ MB (sous le seuil Edge de 100 MB)
- Compression ratio : ___×
- Perplexité : ___ (dégradation de ___ points vs baseline)
- Latence CPU p50 : ___ ms

### Recommandations par contrainte Edge

| Contrainte | Recommandation |
|-----------|---------------|
| Mémoire minimale (training) | Adafactor + INT8 |
| Vitesse d'entraînement | Lion + INT8 |
| Meilleure qualité | AdamW + INT8 |
| Déploiement CPU pur | AdamW + ONNX |
| **Meilleur équilibre global** | **AdamW + Combiné (20% prune + INT8)** |

### Limites et perspectives

- Le pruning non structuré ne réduit pas la taille des tenseurs denses ; un pruning structuré (par têtes d'attention ou filtres entiers) produirait des gains de latence réels supplémentaires.
- La distillation de connaissance (Knowledge Distillation) vers un modèle encore plus petit pourrait offrir une meilleure qualité pour une taille donnée.
- TinyLlama 1.1B avec LoRA serait une extension naturelle pour évaluer ces techniques à plus grande échelle.

---

## Références

- Sanh et al. (2019). DistilBERT, a distilled version of BERT. [arXiv:1910.01108]
- Liu et al. (2023). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.
- Chen et al. (2023). Symbolic Discovery of Optimization Algorithms (Lion). [arXiv:2302.06675]
- You et al. (2020). Large Batch Optimization for Deep Learning: Training BERT in 76 minutes (LAMB).
- Shazeer & Stern (2018). Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.
- Eldan & Li (2023). TinyStories: How Small Can Language Models Be? [arXiv:2305.07759]
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
