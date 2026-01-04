# 🎭 Projet NLP : Analyse de Sentiment Multi-dimensionnelle

## 📌 Vue d'Ensemble

Ce projet implémente un système d'analyse de sentiment avancé capable de détecter simultanément :
1.  **Les Émotions** (7 classes : Joie, Tristesse, Colère, Peur, Surprise, Dégoût, Neutre)
2.  **Le Sentiment** (Positif, Négatif, Neutre)
3.  **L'Ironie** (Ironique, Non-ironique)

Le cœur du système est un modèle **CamemBERT** fine-tuné en mode multi-tâches.

---

## 🛠️ Architecture Technique

### Modèle
- **Base** : `camembert-base` (110M paramètres)
- **Architecture** : Encodeur partagé + 3 têtes de classification indépendantes
- **Loss** : Somme pondérée des CrossEntropyLoss de chaque tâche (avec gestion des labels manquants)

### Données
- **Sentiment** : Dataset **Allociné** (HuggingFace), critiques de films réelles.
- **Émotions & Ironie** : Données synthétiques générées et augmentées pour pallier le manque de datasets français spécialisés libres de droits.
- **Stratégie** : Entraînement mixte où certaines données n'ont que des labels de sentiment (Allociné) et d'autres que des labels d'émotion/ironie.

---

## 📊 Performance

Une baseline (TF-IDF + Logistic Regression) est entraînée pour comparer les performances.
Le modèle Deep Learning (CamemBERT) vise à dépasser cette baseline, notamment sur la compréhension du contexte et de l'ironie.

---

## 📁 Structure du Projet

```
projet-nlp/
├── data/                  # Données (téléchargées automatiquement)
├── models/                # Checkpoints du modèle (.pt)
├── src/                   # Code source
│   ├── app/              # Application Streamlit (Frontend)
│   ├── data/             # Scripts de téléchargement et preprocessing
│   ├── models/           # Définition du modèle CamemBERT et Baseline
│   ├── training/         # Boucles d'entraînement
│   └── evaluation/       # Métriques
├── run_pipeline.py        # Script d'orchestration global
└── README.md             # Guide de démarrage rapide
```
