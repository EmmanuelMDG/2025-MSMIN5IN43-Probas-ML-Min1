# 🚀 Guide de Démarrage Rapide - Projet NLP Multi-tâches

Ce projet a été finalisé pour inclure l'entraînement sur des données réelles (Allociné), la détection d'ironie, et une interface de démonstration.

## � Description du Projet

Ce projet vise à développer un système d'analyse de sentiment avancé pour le français, capable d'aller au-delà de la simple classification positif/négatif. L'objectif est de capturer la richesse émotionnelle des textes (tweets, critiques, commentaires) en détectant simultanément :
1.  **Les Émotions fines** (Joie, Tristesse, Colère, Peur, Surprise, Dégoût).
2.  **Le Sentiment global** (Positif, Négatif, Neutre).
3.  **L'Ironie**, souvent négligée mais cruciale pour comprendre le vrai sens d'un message.

L'idée de départ était de comparer une approche classique (Baseline TF-IDF) avec une approche Deep Learning de pointe (CamemBERT) fine-tunée en mode multi-tâches, permettant au modèle d'apprendre des corrélations entre ces différentes dimensions (ex: l'ironie inverse souvent la polarité du sentiment).

## �📋 Prérequis

- Python 3.8+
- Carte graphique NVIDIA (recommandé) ou CPU

## 🛠️ Installation

1.  Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## ⚡ Lancement Rapide

Un script unique a été créé pour gérer tout le cycle de vie du projet (téléchargement, entraînement, évaluation).

1.  **Lancer le pipeline complet** (Téléchargement données + Entraînement Baseline + Entraînement CamemBERT) :
    ```bash
    python run_pipeline.py
    ```
    *Note : L'entraînement peut prendre 15-30 minutes sur GPU.*

2.  **Lancer la démo interactive** :
    Une fois l'entraînement terminé, lancez l'interface Web :
    ```bash
    streamlit run src/app/app.py
    ```

## 📝 Résumé des Modifications Apportées

- **Données Réelles** : Intégration du dataset `allocine` via la librairie HuggingFace `datasets`.
- **Multi-tâches** : Gestion des labels manquants (ex: Allociné n'a pas d'émotions) via masquage dans la Loss function.
- **Ironie** : Implémentation fonctionnelle de la tête de classification Ironie.
- **Baseline** : Script `train_baseline.py` ajouté pour comparer TF-IDF vs CamemBERT.
- **Frontend** : Application `Streamlit` pour tester le modèle en temps réel.
- **Nettoyage** : Code refactorisé et structure simplifiée.

Pour plus de détails sur l'architecture et le projet, voir [PROJECT_DETAILS.md](PROJECT_DETAILS.md).


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
