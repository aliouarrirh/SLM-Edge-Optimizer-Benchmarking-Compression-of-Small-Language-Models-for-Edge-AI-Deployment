# Prompt — Génération du rapport académique (Word / PDF)

Copie-colle le bloc ci-dessous dans n'importe quelle IA (Claude, ChatGPT, Gemini, etc.) pour générer le rapport complet du projet.

---

## CONTEXTE DU PROJET

Je travaille sur un projet académique d'optimisation et compression de Small Language Models pour l'Edge AI. Voici les éléments clés :

**Modèle :** DistilGPT-2 (82M paramètres, architecture GPT-2 distillée)  
**Dataset :** TinyStories (roneneldan/TinyStories) — histoires courtes en anglais pour enfants  
**Tâche :** Génération de texte causale (Option 2 de l'énoncé)  
**Infrastructure :** Modal (GPU T4, budget 30 crédits)  
**Compression :** Quantification dynamique post-training (torch.quantization, int8, couches Linear)

**5 optimiseurs comparés :**
| Optimiseur | Learning Rate | Librairie |
|---|---|---|
| AdamW | 5e-4 | torch.optim |
| Adafactor | 1e-3 | transformers.optimization |
| Lion | 1e-4 | lion-pytorch |
| LAMB | 1e-3 | torch_optimizer |
| SGD + Nesterov | 1e-3 | torch.optim |

**Hyperparamètres d'entraînement :**
- 3000 exemples d'entraînement, 500 de validation
- 2 epochs, batch size = 8, gradient accumulation = 4 (batch effectif = 32)
- Longueur max = 128 tokens, warmup = 50 steps, grad clip = 1.0

**Résultats expérimentaux obtenus :**
| Optimiseur | PPL avant compression | PPL après compression | Latence avant (GPU, s) | Latence après (CPU, s) | Taille avant (MB) | Taille après (MB) | Temps entraînement (s) | Mémoire GPU peak (GB) |
|---|---|---|---|---|---|---|---|---|
| AdamW | 5.74 | 14.40 | 0.0936 | 0.5064 | 312.5 | 349.31 | 131.3 | 2.543 |
| Lion | 5.74 | 12.00 | 0.0931 | 0.5357 | 312.5 | 349.31 | 126.6 | 2.213 |
| Adafactor | 6.60 | 7.79 | 0.0935 | 0.5264 | 312.5 | 349.31 | 127.8 | 2.003 |
| LAMB | 6.94 | 690.78 | 0.0925 | 0.5029 | 312.5 | 349.31 | 130.2 | 2.548 |
| SGD | 10.18 | 119.33 | 0.0923 | 0.5117 | 312.5 | 349.31 | 125.4 | 2.215 |

**Note sur la taille :** Le ratio de compression affiché (0.89) est biaisé par `torch.save` qui stocke les poids float32 et int8 simultanément. En mémoire à l'inférence, la quantification divise bien la consommation mémoire des couches Linear par ~4.

**Note sur la latence :** La latence augmente après compression car la mesure avant = GPU, après = CPU (quantification dynamique = CPU uniquement). Ce n'est pas une régression, c'est un changement de hardware.

**Lien GitHub :** https://github.com/aliouarrirh/SLM-Edge-Optimizer-Benchmarking-Compression-of-Small-Language-Models-for-Edge-AI-Deployment  
**Notebook :** `opti_project_modal.ipynb` (disponible dans le dépôt)

---

## CONSIGNE

Génère un rapport académique complet en **anglais**, entre **10 et 15 pages**, au format **Word (.docx)** ou **Markdown convertissable en PDF**, en respectant EXACTEMENT la structure suivante :

---

### 1. Introduction (1 page)
- Présente le contexte général : l'essor des LLM et les contraintes de déploiement sur dispositifs Edge (mémoire limitée, pas de GPU, latence critique)
- Objectif du projet : comparer 5 optimiseurs pour fine-tuner DistilGPT-2 sur TinyStories, puis appliquer une compression par quantification int8, et évaluer l'impact sur les métriques Edge AI
- Brève présentation des 5 optimiseurs (AdamW, Adafactor, Lion, LAMB, SGD) : 1-2 phrases chacun sur leur principe de fonctionnement et leur positionnement dans la littérature
- Brève présentation de DistilGPT-2 : architecture, nombre de paramètres, pourquoi ce choix pour l'Edge AI

### 2. Revue de Littérature (2 pages)
Couvre les thèmes suivants avec des références académiques réelles et citables (format APA ou IEEE) :
- Optimiseurs adaptatifs pour les LLM : Adam (Kingma & Ba, 2014), AdamW (Loshchilov & Hutter, 2019), Adafactor (Shazeer & Stern, 2018), Lion (Chen et al., 2023), LAMB (You et al., 2020)
- Compression de modèles de langage : quantification (Dettmers et al., 2022 — bitsandbytes, LLM.int8()), pruning, distillation
- Small Language Models et Edge AI : contraintes de déploiement, TinyML, DistilBERT (Sanh et al., 2019), DistilGPT-2
- TinyStories dataset (Eldan & Li, 2023) : pourquoi ce dataset est adapté aux SLM

### 3. Méthodologie (2-3 pages)
- Description du modèle : DistilGPT-2, architecture Transformer decoder-only, 82M paramètres, 6 couches, 768 hidden dim
- Description du dataset : TinyStories, sous-ensemble de 3000 exemples train / 500 val, prétraitement (tokenisation, padding, truncation à 128 tokens)
- Infrastructure : Modal (GPU T4 serverless), Volume persistant pour les checkpoints, timeout 90 min
- Protocole expérimental :
  - 5 runs indépendants (1 par optimiseur), modèle réinitialisé à chaque run
  - Hyperparamètres fixes sauf LR (adapté par optimiseur selon recommandations de la littérature)
  - Même seed, même ordre de données, même scheduler (linear warmup)
  - Métriques collectées : loss par step, perplexité validation, taille modèle, latence, tokens/sec, mémoire GPU peak, temps d'entraînement
- Compression : quantification dynamique post-training (torch.quantization.quantize_dynamic, couches nn.Linear, dtype=torch.qint8)
- Évaluation post-compression sur CPU (simulation Edge)

### 4. Résultats (2-3 pages)
- Tableau comparatif complet (reprendre exactement le tableau des résultats ci-dessus)
- Courbes d'entraînement : décris ce que montrent les courbes (convergence, stabilité) pour chaque optimiseur — tu peux générer une description textuelle si tu ne peux pas afficher les images
- Analyse des métriques clés :
  - Perplexité avant compression : AdamW et Lion ex-aequo à 5.74, meilleurs
  - Perplexité après compression : Adafactor champion avec seulement +1.19 de dégradation
  - LAMB et SGD : effondrement complet après quantification (PPL 690 et 119)
  - Temps d'entraînement : très similaires (~126-131s), le choix d'optimiseur n'impacte pas significativement le temps GPU
  - Mémoire GPU : Adafactor le plus économe (2.003 GB peak)

### 5. Discussion (2 pages)

#### 5.1 Interprétation des résultats
- Pourquoi AdamW et Lion convergent mieux (PPL 5.74) : correction du weight decay pour AdamW, mises à jour basées sur le signe du gradient pour Lion
- Pourquoi Adafactor est le plus robuste à la quantification : ses mises à jour à faible variance produisent des poids avec une distribution plus régulière, mieux tolérée par l'arrondi int8
- Pourquoi LAMB et SGD s'effondrent après quantification : LAMB optimisé pour large batch (ici batch effectif = 32 seulement, pas son régime optimal) ; SGD sans adaptation par paramètre génère des poids hétérogènes très sensibles à l'arrondi
- Le critère de sélection du "meilleur" modèle doit être redéfini en contexte Edge AI : PPL après compression > PPL avant compression comme critère principal

#### 5.2 Forces et faiblesses de chaque optimiseur

Pour chacun des 5 optimiseurs, rédige un paragraphe court (3-4 phrases) couvrant :
- Performance brute (PPL avant compression)
- Robustesse à la quantification (PPL après compression)
- Efficacité mémoire (GPU peak)
- Adéquation pour l'Edge AI

#### 5.3 Limites de l'expérience
- 1 seul LR testé par optimiseur (pas de grid search)
- Sous-ensemble réduit de TinyStories (3000/500 exemples vs le dataset complet)
- Quantification dynamique uniquement (pas de QAT ni de quantification statique)
- Métrique de taille sur disque biaisée par torch.save

### 6. Conclusion (1 page)
- Synthèse : Adafactor + quantification dynamique int8 = meilleure combinaison pour l'Edge AI dans ce contexte
- Recommandation pratique : utiliser Adafactor pour fine-tuner des SLM destinés à être quantifiés
- Perspectives : QAT (Quantization-Aware Training) pour réduire la dégradation post-compression, pruning combiné à la quantification, export ONNX pour déploiement cross-platform, test sur modèles plus petits (Phi-1.5, SmolLM)

### 7. Bibliographie
Inclure au minimum ces références (format APA) :
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv:1412.6980*
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*
- Shazeer, N., & Stern, M. (2018). Adafactor: Adaptive learning rates with sublinear memory cost. *ICML 2018*
- Chen, X., et al. (2023). Symbolic discovery of optimization algorithms. *arXiv:2302.06675* (Lion)
- You, Y., et al. (2020). Large batch optimization for deep learning: Training BERT in 76 minutes. *ICLR 2020* (LAMB)
- Dettmers, T., et al. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. *NeurIPS 2022*
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*
- Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still speak coherent English? *arXiv:2305.07759*
- PyTorch. (2024). Quantization — PyTorch documentation. https://pytorch.org/docs/stable/quantization.html

### 8. Annexes
- **Annexe A :** Lien vers le dépôt GitHub : https://github.com/aliouarrirh/SLM-Edge-Optimizer-Benchmarking-Compression-of-Small-Language-Models-for-Edge-AI-Deployment
- **Annexe B :** Description du notebook `opti_project_modal.ipynb` — pipeline complet : fine-tuning, quantification, visualisation des courbes d'entraînement et tableau comparatif. Disponible dans le dépôt GitHub.
- **Annexe C :** Script Modal `modal_distilgpt2.py` — code du pipeline GPU. Disponible dans le dépôt GitHub.

---

## FORMAT DE SORTIE

- **Langue :** Anglais académique
- **Longueur :** 10 à 15 pages
- **Format :** Génère le rapport en Markdown avec titres bien structurés (H1, H2, H3), tableaux Markdown, et sections numérotées. Le Markdown peut ensuite être converti en PDF via Pandoc ou en Word via un outil de conversion.
- **Style :** Académique, troisième personne, phrases complètes. Pas de bullet points dans les sections de contenu (sauf méthodologie et annexes).
- **Tableaux :** Reprendre exactement les données numériques fournies, sans les arrondir ni les modifier.
- **Références :** Citer les sources dans le texte au format (Auteur, année) et les lister complètement en section 7.

---

## COMMANDE FINALE

Génère maintenant le rapport complet en suivant exactement cette structure. Ne résume pas, ne raccourcis pas — chaque section doit être rédigée entièrement. Le rapport doit être autonome et lisible sans connaître ce prompt.
