# Optimisation et Compression de Small Language Models pour l'Edge AI

**Projet académique — Option 2 : Génération de texte courte**  
**Modèle :** DistilGPT-2 (82M paramètres) — **Dataset :** TinyStories — **Infrastructure :** Modal GPU (T4)

---

## Structure du projet

```
PFM/
├── modal_distilgpt2.py          # Script Modal autonome — pipeline complet GPU
├── opti_project_modal.ipynb     # Notebook de visualisation des résultats
├── results/                     # Résultats générés par le pipeline Modal
│   ├── training_curves.png      # Courbes de loss par optimiseur
│   ├── comparison_charts.png    # 6 graphiques comparatifs
│   ├── resultats_comparatifs.csv
│   ├── summary.json
│   └── text_examples_<OPT>.json
└── story-chatbot/
    ├── backend/main.py          # API FastAPI avec le modèle compressé
    └── mobile/App.tsx           # App React Native (Expo)
```

---

## Pipeline Modal (`modal_distilgpt2.py`)

Script autonome lancé avec `modal run`. Tourne entièrement sur GPU T4 Modal.

### Ce que fait le pipeline

1. **Fine-tuning** de DistilGPT-2 sur TinyStories (3000 exemples, 2 epochs) avec 5 optimiseurs
2. **Évaluation** perplexité sur set de validation (500 exemples) pour chaque optimiseur
3. **Quantification dynamique** int8 (`torch.quantization`) appliquée à tous les modèles
4. **Évaluation post-compression** : perplexité, latence, taille, tokens/sec
5. **Génération** de courbes d'entraînement (PNG) et tableau comparatif (CSV)
6. **Persistance** de tous les résultats dans un Volume Modal (`distilgpt2-results-v1`)

### Optimiseurs comparés

| Optimiseur | LR | Source |
|---|---|---|
| AdamW | 5e-4 | `torch.optim` |
| Adafactor | 1e-3 | `transformers.optimization` |
| Lion | 1e-4 | `lion-pytorch` |
| LAMB | 1e-3 | `torch_optimizer` |
| SGD + Nesterov | 1e-3 | `torch.optim` |

### Configuration

| Paramètre | Valeur |
|---|---|
| GPU | T4 (Modal) |
| Timeout | 90 min (filet de sécurité budget) |
| Batch effectif | 32 (batch=8, grad_accum=4) |
| Longueur max | 128 tokens |
| Coût estimé | ~$0.20 USD (T4 = $0.59/h) |

---

## Résultats obtenus

| Optimiseur | PPL avant | PPL après compression | Dégradation |
|---|---|---|---|
| AdamW | 5.74 | 14.40 | +8.66 |
| Lion | 5.74 | 12.00 | +6.26 |
| Adafactor | 6.60 | **7.79** | **+1.19** ✅ |
| LAMB | 6.94 | 690.78 | effondrement |
| SGD | 10.18 | 119.33 | effondrement |

**Conclusion :** Adafactor est le meilleur optimiseur pour l'Edge AI — légèrement moins performant avant compression, mais de loin le plus robuste à la quantification int8.

---

## Lancer le pipeline

### Prérequis

```bash
pip install modal
modal setup   # crée/connecte le compte Modal
```

### Dry run (validation, 0 crédit)

```bash
modal run modal_distilgpt2.py --dry-run
```

### Pipeline complet (GPU T4)

```bash
modal run modal_distilgpt2.py
```

### Récupérer les résultats en local

```bash
modal volume get distilgpt2-results-v1 / ./results/ --force
```

---

## Visualiser les résultats

Ouvrir `opti_project_modal.ipynb` dans Jupyter. Le notebook lit directement depuis `./results/` et génère :
- Courbes de loss par optimiseur
- 6 graphiques comparatifs (perplexité, taille, latence, débit, compression, temps d'entraînement)
- Tableau comparatif avec mise en forme
- Exemples de génération avant/après compression
- Synthèse automatique et recommandation Edge AI

```bash
jupyter notebook opti_project_modal.ipynb
```

---

## App Story Chatbot

Le modèle fine-tuné et compressé est intégré dans une app mobile React Native via une API FastAPI.

### Backend

```bash
cd story-chatbot/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Le backend charge automatiquement le modèle depuis `./model_compressed/` s'il existe, sinon repli sur distilgpt2 de base. La quantification int8 est appliquée au démarrage (~2s).

### App mobile

```bash
cd story-chatbot/mobile
npx expo start
```

---

## Stack technique

- [Transformers](https://huggingface.co/docs/transformers/) — Hugging Face
- [Modal](https://modal.com/) — infrastructure GPU serverless
- [PyTorch](https://pytorch.org/) + `torch.quantization` — quantification dynamique int8
- [lion-pytorch](https://github.com/lucidrains/lion-pytorch) — optimiseur Lion
- [torch-optimizer](https://github.com/jettify/pytorch-optimizer) — optimiseur LAMB
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — dataset
- [FastAPI](https://fastapi.tiangolo.com/) + [Expo](https://expo.dev/) — app chatbot
