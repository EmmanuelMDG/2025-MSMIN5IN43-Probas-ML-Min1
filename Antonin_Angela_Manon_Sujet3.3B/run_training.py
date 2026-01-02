#!/usr/bin/env python3
"""
Script principal pour lancer l'entraînement du modèle CamemBERT Multi-tâches

Usage:
    python run_training.py
"""

import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent / "src"))

from training.train import train_model
from models.config import ModelConfig, TrainingConfig


def main():
    """Fonction principale"""
    
    print("=" * 80)
    print("🎯 ENTRAÎNEMENT CAMEMBERT MULTI-TÂCHES")
    print("=" * 80)
    
    # Configuration du modèle
    model_config = ModelConfig(
        model_name="camembert-base",
        dropout=0.3,
        max_length=128,
        # Pondération des losses
        loss_weight_emotion=1.0,
        loss_weight_sentiment=0.5,
        loss_weight_irony=0.3
    )
    
    # Configuration de l'entraînement
    training_config = TrainingConfig(
        batch_size=16,           # Réduire à 8 si problèmes de mémoire
        num_epochs=5,
        lr_encoder=2e-5,         # Learning rate pour l'encodeur
        lr_classifier=1e-4,      # Learning rate pour les têtes
        patience=3,              # Early stopping
        save_dir="models",
        seed=42
    )
    
    print("\n📋 Configuration:")
    print(f"   - Modèle: {model_config.model_name}")
    print(f"   - Batch size: {training_config.batch_size}")
    print(f"   - Époques: {training_config.num_epochs}")
    print(f"   - LR encodeur: {training_config.lr_encoder}")
    print(f"   - LR têtes: {training_config.lr_classifier}")
    print(f"   - Device: {training_config.device}")
    print()
    
    # Lancer l'entraînement
    try:
        model, history = train_model(
            data_path="data/processed",
            model_config=model_config,
            training_config=training_config
        )
        
        print("\n✅ Entraînement terminé avec succès !")
        print("\n📊 Pour visualiser les résultats:")
        print("   - Modèle sauvegardé: models/best_model.pt")
        print("   - Historique: models/training_history.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Entraînement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
