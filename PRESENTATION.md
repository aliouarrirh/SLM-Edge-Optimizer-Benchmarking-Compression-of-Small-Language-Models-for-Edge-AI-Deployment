# Présentation orale — 10 minutes

**Projet :** Optimisation et Compression de Small Language Models pour l'Edge AI  
**Modèle :** DistilGPT-2 × TinyStories — 5 optimiseurs + quantification int8  

---

## Plan des slides (10 slides, ~1 min/slide)

---

### Slide 1 — Titre (30 sec)
**Titre :** Optimizer Benchmarking & Model Compression for Edge AI  
**Sous-titre :** Fine-tuning DistilGPT-2 on TinyStories with 5 optimizers + INT8 quantization  
**Contenu :** Noms, module, date

---

### Slide 2 — Problématique (1 min)
**Titre :** Why compress language models for Edge AI?

**Points clés :**
- Les LLM sont trop lourds pour les dispositifs Edge (pas de GPU, RAM limitée, latence critique)
- DistilGPT-2 = 82M paramètres, ~312 MB — encore trop lourd tel quel
- **Question :** Quel optimiseur de fine-tuning produit le modèle le plus robuste à la compression ?

**Visual :** Schéma simple — Cloud (gros modèle) → Edge device (modèle compressé)

---

### Slide 3 — Stack technique (45 sec)
**Titre :** Setup expérimental

**3 colonnes :**
| Modèle | Dataset | Infrastructure |
|---|---|---|
| DistilGPT-2 (82M params) | TinyStories (3000 train / 500 val) | Modal GPU T4 |
| 6 layers, 768 hidden dim | Histoires courtes en anglais | ~20 min, ~$0.20 |

**Compression :** `torch.quantization.quantize_dynamic` — int8, couches Linear

---

### Slide 4 — Les 5 optimiseurs (1 min)
**Titre :** 5 optimizers, same conditions

**Tableau :**
| Optimiseur | LR | Particularité |
|---|---|---|
| AdamW | 5e-4 | Référence — weight decay découplé |
| Adafactor | 1e-3 | Mémoire-efficace, pas de moments complets |
| Lion | 1e-4 | Basé sur le signe du gradient (2023) |
| LAMB | 1e-3 | Optimisé pour large batch |
| SGD + Nesterov | 1e-3 | Baseline classique |

**Note :** Mêmes epochs, même batch, même scheduler — comparaison équitable

---

### Slide 5 — Résultats avant compression (1 min 30)
**Titre :** Fine-tuning results — Perplexity on validation set

**Graphique :** `training_curves.png`

**Points à dire :**
- AdamW et Lion arrivent ex-aequo à PPL = **5.74** — meilleurs
- Adafactor légèrement derrière à **6.60** mais mémoire GPU la plus faible (2.0 GB vs 2.5 GB)
- SGD clairement en retard à **10.18** — confirme sa faiblesse sur les LLM
- Temps d'entraînement similaires pour tous (~127-131s) — le choix d'optimiseur n'impacte pas le coût GPU

---

### Slide 6 — Impact de la quantification (1 min 30)
**Titre :** After INT8 quantization — the real Edge AI test

**Graphique :** `comparison_charts.png` (panel perplexité avant/après)

**Tableau choc :**
| Optimiseur | PPL avant | PPL après | Verdict |
|---|---|---|---|
| Adafactor | 6.60 | **7.79** | ✅ Robuste (+1.19) |
| Lion | 5.74 | 12.00 | ⚠️ Dégradation modérée |
| AdamW | 5.74 | 14.40 | ⚠️ Dégradation notable |
| SGD | 10.18 | 119.33 | ❌ Effondrement |
| LAMB | 6.94 | 690.78 | ❌ Effondrement total |

**Point fort à souligner :** Le meilleur avant compression ≠ le meilleur pour l'Edge AI

---

### Slide 7 — Pourquoi Adafactor résiste mieux (1 min)
**Titre :** Why Adafactor survives quantization

**Explication visuelle :**
```
AdamW / Lion → mises à jour adaptatives avec moments du 2e ordre
             → poids avec forte variance locale
             → sensibles à l'arrondi int8  →  dégradation

Adafactor    → mises à jour à faible variance (factorisation rang-1)
             → distribution des poids plus régulière
             → tolère mieux l'arrondi int8  →  robuste
```

**LAMB & SGD :** poids numériquement instables hors de leur régime optimal (LAMB = conçu pour batch >> 32)

---

### Slide 8 — Métriques Edge AI complètes (45 sec)
**Titre :** Full Edge AI metrics

**Graphique :** `comparison_charts.png` (panels latence + tokens/sec)

**Note importante :** La latence augmente après compression (GPU → CPU) — pas une régression, c'est un changement de hardware. Sur un vrai device Edge, le modèle int8 sur CPU sera plus rapide qu'un float32 sur le même CPU.

---

### Slide 9 — Conclusion & Recommandation (1 min)
**Titre :** Best combination for Edge AI deployment

**Recommandation :**
> **Adafactor + dynamic INT8 quantization**
> → PPL 7.79 après compression, mémoire GPU minimale, robustesse maximale

**Ce qu'on retient :**
- Ne pas choisir l'optimiseur uniquement sur la PPL brute — tester après compression
- Adafactor est sous-estimé pour les SLM destinés à être quantifiés
- LAMB et SGD inadaptés à ce régime (petit batch, quantification post-training)

**Perspectives :**
- QAT (Quantization-Aware Training) pour réduire encore la dégradation
- Export ONNX pour déploiement cross-platform
- Tester sur des modèles encore plus petits (SmolLM 135M, Phi-1.5)

---

### Slide 10 — Démo / Questions (30 sec)
**Titre :** Demo & Questions

**Contenu :**
- Lien GitHub : https://github.com/aliouarrirh/SLM-Edge-Optimizer-Benchmarking-Compression-of-Small-Language-Models-for-Edge-AI-Deployment
- App chatbot mobile (React Native + FastAPI + modèle compressé)
- Notebook : `opti_project_modal.ipynb`

---

## Timing détaillé

| Slide | Durée | Cumulé |
|---|---|---|
| 1 — Titre | 0:30 | 0:30 |
| 2 — Problématique | 1:00 | 1:30 |
| 3 — Stack technique | 0:45 | 2:15 |
| 4 — Les 5 optimiseurs | 1:00 | 3:15 |
| 5 — Résultats avant compression | 1:30 | 4:45 |
| 6 — Impact quantification | 1:30 | 6:15 |
| 7 — Pourquoi Adafactor | 1:00 | 7:15 |
| 8 — Métriques Edge complètes | 0:45 | 8:00 |
| 9 — Conclusion | 1:00 | 9:00 |
| 10 — Questions | 1:00 | 10:00 |

---

## Questions jury anticipées

**Q : Pourquoi ne pas avoir testé plusieurs learning rates par optimiseur ?**  
R : Contrainte de budget GPU (30 crédits). Les LR utilisés suivent les recommandations de la littérature pour chaque optimiseur. Un grid search multiplierait le coût par 3-4x.

**Q : Pourquoi la taille du modèle augmente après quantification ?**  
R : Artefact de `torch.save` qui stocke les poids float32 et int8 simultanément. La consommation mémoire réelle à l'inférence est bien réduite.

**Q : Pourquoi la latence est plus haute après compression ?**  
R : La mesure avant compression est faite sur GPU, après sur CPU (quantification dynamique = CPU uniquement). Ce n'est pas une régression — sur le même hardware (CPU), int8 est plus rapide que float32.

**Q : Est-ce que c'est du vrai Edge AI ?**  
R : Les techniques sont celles de l'Edge AI (quantification, SLM). Le déploiement final est sur serveur CPU (FastAPI). Un vrai déploiement Edge nécessiterait un export ONNX ou CoreML et une intégration directe dans l'app mobile.

**Q : Pourquoi DistilGPT-2 et pas un modèle plus récent ?**  
R : Choix de l'énoncé (Option 2). DistilGPT-2 est un bon candidat pour l'Edge AI : architecture simple, 82M params, largement étudié dans la littérature, et représentatif de la famille des SLM decoder-only.
