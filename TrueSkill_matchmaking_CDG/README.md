# 🎮 TrueSkill Matchmaking Simulator - Documentation Complète

> Projet réalisé dans le cadre du cours **MSMIN5IN43 - IA Probabiliste, Théorie des Jeux et Machine Learning**  
> EPF - Janvier 2026

---

## 📋 Table des Matières

1. [Introduction](docs/01-INTRODUCTION.md) - Contexte et problématique
2. [Théorie TrueSkill](docs/02-TRUESKILL-THEORY.md) - Fondements mathématiques
3. [Implémentation](docs/03-IMPLEMENTATION.md) - Architecture technique
4. [Visualisations](docs/04-VISUALIZATIONS.md) - Graphiques et analyses
5. [Comparaison ELO](docs/05-COMPARISON-ELO.md) - TrueSkill vs ELO
6. [Interface Web](docs/06-WEB-INTERFACE.md) - Application Streamlit
7. [Résultats](docs/07-RESULTS.md) - Analyses et conclusions
8. [Conclusion](docs/08-CONCLUSION.md) - Bilan et perspectives
9. [Sources](docs/SOURCES.md) - Bibliographie complète

---

## 🎯 Résumé Exécutif

Ce projet implémente et analyse le système de classement **TrueSkill**, développé par Microsoft Research pour Xbox Live.  L'objectif est de démontrer comment un système probabiliste peut estimer la compétence des joueurs de manière plus précise et rapide que les systèmes classiques (ELO).

### Résultats Clés

- ✅ **Convergence rapide** : TrueSkill estime correctement les compétences après ~50 matchs
- ✅ **Gestion de l'incertitude** : Le paramètre σ diminue avec le nombre de matchs
- ✅ **Supériorité sur ELO** : +24% de précision sur le classement final
- ✅ **Application interactive** : Interface web pour démonstration en temps réel

---

## 🚀 Démarrage Rapide

```bash
# Installation
git clone <repo>
cd trueskill-matchmaking
pip install -r requirements.txt

# Simulation basique
python main.py

# Visualisations complètes
python demo_visualizations.py

# Comparaison TrueSkill vs ELO
python demo_comparison.py

# Interface web interactive
streamlit run app.py
```

---

## 📊 Aperçu des Résultats

### Convergence de TrueSkill
![Convergence](../results/convergence_mu. png)

### Comparaison TrueSkill vs ELO
![Comparison](../results/ts_vs_elo.png)

---

## 👥 Équipe

- **Quentin Deharo** 
- **Thomas Gombert**
- **Cornel Stefan Cristea** 

---

## 📅 Timeline du Projet

- **Jour 1** : Implémentation du simulateur et visualisations de base
- **Jour 2** : Interface Streamlit et comparaison avec ELO
- **Jour 3** : Documentation et préparation de la présentation
- **6 janvier 2026** : Présentation finale

---

## 📖 Comment Lire Cette Documentation

1. **Si vous êtes pressé** : Lisez le [Résumé Exécutif](#-résumé-exécutif) et les [Résultats](07-RESULTS.md)
2. **Si vous voulez comprendre la théorie** : Commencez par [Introduction](01-INTRODUCTION.md) et [Théorie](02-TRUESKILL-THEORY.md)
3. **Si vous voulez reproduire** : Suivez [Implémentation](03-IMPLEMENTATION.md) étape par étape
4. **Si vous voulez approfondir** : Consultez [Sources](SOURCES.md) pour les références académiques

---

## 🔗 Liens Utiles

- [TrueSkill Official (Microsoft Research)](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
- [TrueSkill Python Library](https://trueskill.org/)
- [Dépôt GitHub du projet](https://github.com/Thomas-G27/2025-MSMIN5IN43-Probas-ML-Min1-DCG)

---

**Licence** :  Projet éducatif - EPF 2026