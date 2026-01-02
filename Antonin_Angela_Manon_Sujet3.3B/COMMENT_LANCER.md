# 🚀 Comment lancer l'entraînement (Mac Intel = problème scipy)

## ⚠️ Problème actuel
Ton Mac Intel avec Python 3.9 a un bug avec scipy qui empêche l'entraînement local.

## ✅ SOLUTION RECOMMANDÉE : Google Colab (GRATUIT + GPU)

### Étape 1 : Préparer les fichiers
1. Compresse le dossier `Antonin_Angela_Manon_Sujet3.3B` en ZIP
2. Ou copie-le directement sur Google Drive

### Étape 2 : Ouvrir Colab
1. Va sur [Google Colab](https://colab.research.google.com/)
2. Clique sur **File → Upload notebook**
3. Upload le fichier `Training_Colab.ipynb`

### Étape 3 : Activer le GPU (IMPORTANT !)
1. Dans Colab : **Runtime → Change runtime type**
2. Sélectionne **T4 GPU**
3. Clique sur **Save**

### Étape 4 : Lancer l'entraînement
1. Exécute toutes les cellules (Runtime → Run all)
2. Autorise l'accès à ton Google Drive
3. Attends 15-30 minutes (avec GPU, c'est rapide !)

### Résultats
- Modèle sauvegardé : `models/best_model.pt`
- Métriques : `models/training_history.json`
- Matrices de confusion : `results/confusion_matrices.png`

---

## 🔄 Alternative : Autre ordinateur

Le code fonctionne parfaitement sur :
- ✅ **Linux** (Ubuntu, Debian, etc.)
- ✅ **Windows** 
- ✅ **Mac M1/M2/M3** (Apple Silicon)
- ✅ **Mac Intel avec Python 3.10+**

### Sur un autre ordi :
```bash
cd Antonin_Angela_Manon_Sujet3.3B

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows

# Installer
pip install -r requirements.txt

# Lancer
python run_training.py
```

---

## 📝 Tout est prêt !

- ✅ Architecture CamemBERT implémentée
- ✅ Script d'entraînement complet
- ✅ Script d'évaluation
- ✅ Notebook Colab prêt
- ✅ Documentation complète

**Il ne reste qu'à lancer sur Colab ou un autre PC !** 🎉

---

## 🆘 Support

Si problème sur Colab, vérifie que :
1. Le GPU est activé (T4)
2. Le chemin vers ton dossier est correct
3. Les fichiers data/processed/*.csv sont présents
