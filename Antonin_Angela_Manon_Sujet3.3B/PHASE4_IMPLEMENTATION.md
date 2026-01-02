# 🎯 Phase 4 : CamemBERT Multi-tâches - IMPLÉMENTÉ ✅

## 📋 Ce qui a été créé

### 1. Configuration (`src/models/config.py`) ✅
- `ModelConfig` : Configuration du modèle (architecture, dropout, etc.)
- `TrainingConfig` : Configuration de l'entraînement (learning rates, batch size, etc.)
- Fonction `set_seed()` pour la reproductibilité
- Mappings des labels (EMOTION_LABELS, SENTIMENT_LABELS, IRONY_LABELS)

### 2. Architecture CamemBERT (`src/models/camembert_multitask.py`) ✅
```
Texte → CamemBERT (encodeur partagé) → [CLS] token
                                            ↓
                                ┌───────────┼───────────┐
                                ↓           ↓           ↓
                            Émotions    Sentiment    Ironie
                            (7 classes) (3 classes)  (2 classes)
```

**Fonctionnalités** :
- Encodeur CamemBERT pré-entraîné
- 3 têtes de classification avec dropout
- Loss combinée pondérée : `1.0×emotion + 0.5×sentiment + 0.3×irony`
- Méthodes `predict()`, `freeze_encoder()`, `unfreeze_encoder()`
- ~110M paramètres

### 3. Script d'entraînement (`src/training/train.py`) ✅
**Classe `MultiTaskDataset`** :
- Dataset PyTorch personnalisé
- Tokenization automatique avec CamemBERT

**Fonction `train_epoch()`** :
- Entraînement sur une époque
- Calcul des métriques (F1-Score, Accuracy)
- Support du gradient accumulation

**Fonction `validate()`** :
- Validation sur val set
- Sans gradient (mode eval)

**Fonction `train_model()` (principale)** :
- Chargement des données (train/val/test)
- Création des DataLoaders
- **Learning rates différenciés** :
  - Encodeur : 2e-5 (fine-tuning doux)
  - Têtes : 1e-4 (entraînement from scratch)
- **Early stopping** : patience de 3 époques
- Sauvegarde du meilleur modèle
- Historique d'entraînement en JSON

### 4. Utilitaires (`src/training/utils.py`) ✅
- `create_optimizer_with_layerwise_lr()` : Optimiseur avec LR différenciés
- `create_scheduler()` : Scheduler avec warmup
- `load_checkpoint()` : Chargement de checkpoints
- `plot_training_history()` : Visualisation de l'entraînement
- `print_model_summary()` : Résumé du modèle

### 5. Scripts prêts à l'emploi ✅

**`run_training.py`** :
```bash
python run_training.py
```
- Lance l'entraînement avec config par défaut
- Batch size 16, 5 époques, early stopping
- Sauvegarde dans `models/best_model.pt`

**`evaluate_model.py`** :
```bash
python evaluate_model.py
```
- Évalue le modèle sur le test set
- Génère rapports de classification complets
- Crée matrices de confusion (3 tâches)
- Sauvegarde graphiques dans `results/`

---

## 🚀 Comment utiliser

### 1. Entraîner le modèle
```bash
cd Antonin_Angela_Manon_Sujet3.3B
python run_training.py
```

**Sortie attendue** :
```
🖥️  Device: mps  # ou cuda ou cpu
📂 Chargement des données...
   ✓ Train: 490 exemples
   ✓ Val: 105 exemples
   ✓ Test: 105 exemples
📥 Chargement de camembert-base...
✅ Modèle créé avec 110,549,767 paramètres

🚀 Début de l'entraînement (5 époques)
================================================================================

📍 Époque 1/5
Training: 100%|████████| 31/31 [00:45<00:00]
Validation: 100%|████████| 7/7 [00:03<00:00]

📊 Résultats Époque 1:
   Train - Loss: 1.2345 | Emotion F1: 0.5234 | Sentiment Acc: 0.7123 | Irony F1: 0.6012
   Val   - Loss: 1.1234 | Emotion F1: 0.5678 | Sentiment Acc: 0.7456 | Irony F1: 0.6234

💾 Nouveau meilleur modèle ! Score: 0.6456
...
```

### 2. Évaluer le modèle
```bash
python evaluate_model.py
```

**Sortie attendue** :
```
🎯 ÉVALUATION DU MODÈLE CAMEMBERT
📥 Chargement du modèle depuis models/best_model.pt...
✅ Modèle chargé avec succès

🧪 Évaluation sur le test set...

📋 RAPPORTS DE CLASSIFICATION
================================================================================

🎭 ÉMOTIONS:
              precision    recall  f1-score   support
        joie     0.7500    0.8000    0.7742        15
   tristesse     0.6923    0.7500    0.7200        12
...

💭 SENTIMENT:
              precision    recall  f1-score   support
    negatif     0.8500    0.8500    0.8500        40
...

😏 IRONIE:
              precision    recall  f1-score   support
non_ironique     0.8750    0.9000    0.8873        80
...

📊 Matrices de confusion sauvegardées: results/confusion_matrices.png
✅ Évaluation terminée !
```

---

## 🎓 Explications pédagogiques

### Pourquoi Learning Rates différenciés ?

```python
optimizer = AdamW([
    {'params': encodeur, 'lr': 2e-5},     # Petit LR
    {'params': têtes, 'lr': 1e-4}         # Grand LR
])
```

**Raison** :
- **Encodeur** : Déjà pré-entraîné sur des milliards de mots
  → On veut juste le "fine-tuner" doucement (petit LR)
- **Têtes** : Entraînées from scratch pour nos tâches
  → On peut apprendre plus vite (grand LR)

### Pourquoi Loss pondérée ?

```python
loss = 1.0×L_emotion + 0.5×L_sentiment + 0.3×L_irony
```

**Raison** :
- **Émotions** (7 classes) : Tâche la plus difficile → poids 1.0
- **Sentiment** (3 classes) : Plus facile → poids 0.5
- **Ironie** (2 classes) : La plus facile mais classes déséquilibrées → poids 0.3

→ On équilibre l'importance des tâches

### Pourquoi Early Stopping ?

```python
if val_score > best_val_score:
    save_model()
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 3:
        stop_training()
```

**Raison** :
- Éviter l'overfitting
- Si le modèle n'améliore plus pendant 3 époques → arrêt
- On garde le meilleur modèle (pas le dernier !)

---

## 📊 Résultats attendus

### Objectifs de performance
- **Émotions** : F1-Score ≥ 0.65 (objectif : 0.75)
- **Sentiment** : Accuracy ≥ 0.80 (objectif : 0.88)
- **Ironie** : F1-Score ≥ 0.60 (objectif : 0.70)

### Comparaison Baseline vs CamemBERT
| Tâche | Baseline TF-IDF | CamemBERT (attendu) |
|-------|-----------------|---------------------|
| Émotions F1 | ~0.50-0.55 | ~0.65-0.75 |
| Sentiment Acc | ~0.70-0.75 | ~0.80-0.88 |
| Ironie F1 | ~0.55-0.60 | ~0.60-0.70 |

**Gain attendu** : +10-15 points sur toutes les métriques 🚀

---

## 🔧 Personnalisation

### Modifier les hyperparamètres

Éditer `run_training.py` :

```python
training_config = TrainingConfig(
    batch_size=8,           # Si problèmes mémoire
    num_epochs=10,          # Plus d'époques
    lr_encoder=1e-5,        # LR plus petit
    lr_classifier=5e-5,     # LR plus petit
    patience=5,             # Plus de patience
    gradient_accumulation_steps=2  # Simuler batch_size=16
)
```

### Changer la pondération des losses

Éditer `run_training.py` :

```python
model_config = ModelConfig(
    loss_weight_emotion=1.0,
    loss_weight_sentiment=1.0,    # Équilibré
    loss_weight_irony=1.0
)
```

---

## ✅ Checklist Phase 4

- [x] Architecture CamemBERT implémentée
- [x] 3 têtes de classification créées
- [x] Loss combinée pondérée
- [x] Dataset PyTorch custom
- [x] Boucle d'entraînement
- [x] Learning rates différenciés
- [x] Early stopping
- [x] Sauvegarde checkpoints
- [x] Script d'évaluation
- [x] Matrices de confusion
- [x] Rapports de classification
- [x] Documentation complète

**→ Phase 4 : COMPLÈTE ! 🎉**

---

## 📝 Prochaines étapes (Phase 5)

1. **Lancer l'entraînement** : `python run_training.py`
2. **Analyser les résultats** : Regarder les courbes d'apprentissage
3. **Évaluer sur test** : `python evaluate_model.py`
4. **Analyse des erreurs** :
   - Identifier 50-100 exemples mal classés
   - Comprendre pourquoi (ironie non détectée, contexte, etc.)
5. **Visualisations avancées** :
   - t-SNE des embeddings
   - Attention weights
6. **Rédiger le rapport final**

---

**Date de création** : 2 janvier 2026  
**Status** : ✅ IMPLÉMENTÉ ET PRÊT À L'EMPLOI
