# 🎯 Projet NLP - Analyse de Sentiment Multi-dimensionnelle

## 📋 Description

Système d'analyse de sentiment avancé utilisant CamemBERT pour détecter :
- **Émotions** (7 classes : Joie, Tristesse, Colère, Peur, Surprise, Dégoût, Neutre)
- **Sentiment** (3 classes : Positif, Négatif, Neutre)
- **Ironie** (2 classes : Ironique, Non-ironique)

## 🏗️ Structure du Projet

```
Antonin_Angela_Manon_Sujet3.3B/
├── data/                      # Données brutes et traitées
├── notebooks/                 # Jupyter notebooks pour exploration
├── src/                       # Code source
│   ├── data/                 # Chargement et preprocessing
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── models/               # Modèles (baseline + CamemBERT)
│   │   ├── baseline.py
│   │   ├── camembert_multitask.py
│   │   └── config.py
│   ├── training/             # Scripts d'entraînement
│   │   ├── train.py
│   │   └── utils.py
│   └── evaluation/           # Métriques et visualisations
│       ├── metrics.py
│       ├── visualization.py
│       └── error_analysis.py
├── models/                    # Modèles sauvegardés
├── results/                   # Résultats et graphiques
├── requirements.txt          # Dépendances Python
└── README.md                 # Ce fichier
```

## 🚀 Installation et Configuration

### 1. Cloner le projet

Si ce n'est pas déjà fait :
```bash
git clone <url_du_repo>
cd Antonin_Angela_Manon_Sujet3.3B
```

### 2. Créer un environnement virtuel

**Windows (PowerShell)** :
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac** :
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Télécharger les ressources NLP (optionnel)

```python
# Dans un terminal Python ou notebook
import nltk
import spacy

# Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Télécharger le modèle spaCy français
# python -m spacy download fr_core_news_sm
```

## 📊 Utilisation

### Phase 1 : Collecte et exploration des données ✅

```bash
# Lancer Jupyter pour l'exploration
jupyter notebook notebooks/
```

### Phase 2 : Baseline ✅

Les modèles baseline sont déjà entraînés et sauvegardés dans `models/`.

### Phase 3 : CamemBERT Multi-tâches ⚡

```bash
# Entraîner le modèle CamemBERT (script simplifié)
python run_training.py

# Ou directement avec le module
python -m src.training.train
```

**Configuration** :
- Batch size : 16 (réduire à 8 si problèmes de mémoire)
- Époques : 5 (avec early stopping)
- Learning rate encodeur : 2e-5
- Learning rate têtes : 1e-4
- Device : Auto-détection (CUDA > MPS > CPU)

Le modèle sera sauvegardé dans `models/best_model.pt`

### Phase 4 : Évaluation

```bash
# Évaluer le modèle sur le test set
python evaluate_model.py
```

Cela génère :
- Rapports de classification détaillés (F1, Precision, Recall)
- Matrices de confusion pour les 3 tâches
- Graphiques sauvegardés dans `results/`

## 🎯 Objectifs de Performance

| Tâche | Métrique | Objectif Minimum | Objectif Optimal |
|-------|----------|------------------|------------------|
| Émotions | F1 (macro) | 0.65 | 0.75+ |
| Sentiment | Accuracy | 0.80 | 0.88+ |
| Ironie | F1 | 0.60 | 0.70+ |

## ⚡ Points Clés à Retenir

✅ **À FAIRE** :
- Fixer les seeds (reproductibilité)
- Stratifier le split train/val/test
- Garder les emojis dans le preprocessing
- Utiliser F1-Score comme métrique principale
- Implémenter early stopping
- Analyser les erreurs

❌ **À NE PAS FAIRE** :
- Prétraiter avant de splitter
- Supprimer emojis/ponctuation
- Se fier uniquement à l'accuracy
- Tuner sur le test set

## 📚 Ressources

- [Documentation CamemBERT](https://huggingface.co/camembert-base)
- [Transformers HuggingFace](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs)

## 👥 Équipe

- Antonin
- Angela
- Manon

## 📝 License

Ce projet est réalisé dans le cadre du cours MSMIN5IN43 - Probabilités & Machine Learning.

---

**Date** : Janvier 2026  
**Version** : 1.0
