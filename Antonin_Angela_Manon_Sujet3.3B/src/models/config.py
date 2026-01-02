"""
Configuration du modèle et des hyperparamètres
"""

import torch
import numpy as np
import random
from dataclasses import dataclass
from typing import Optional


def set_seed(seed: int = 42):
    """
    Fixe tous les seeds pour la reproductibilité
    
    Args:
        seed: Valeur du seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelConfig:
    """Configuration du modèle CamemBERT Multi-tâches"""
    
    # Modèle de base
    model_name: str = "camembert-base"  # ou "camembert/camembert-base"
    
    # Architecture
    hidden_size: int = 768  # Taille de sortie de CamemBERT
    num_emotions: int = 7   # 7 classes d'émotions
    num_sentiments: int = 3  # 3 classes de sentiment
    num_irony: int = 2      # 2 classes d'ironie
    dropout: float = 0.3
    
    # Tokenization
    max_length: int = 128   # Longueur max des séquences
    
    # Loss weights (pondération des tâches)
    loss_weight_emotion: float = 1.0
    loss_weight_sentiment: float = 0.5
    loss_weight_irony: float = 0.3


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement"""
    
    # Hyperparamètres généraux
    batch_size: int = 16
    num_epochs: int = 5
    seed: int = 42
    
    # Learning rates différenciés
    lr_encoder: float = 2e-5    # Plus petit pour l'encodeur pré-entraîné
    lr_classifier: float = 1e-4  # Plus grand pour les têtes (entraînées from scratch)
    
    # Optimiseur
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    
    # Scheduler
    warmup_steps: int = 100
    
    # Early stopping
    patience: int = 3
    min_delta: float = 0.001  # Amélioration minimale pour considérer un progrès
    
    # Checkpoints
    save_dir: str = "models"
    save_best_only: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Gradient accumulation (si mémoire limitée)
    gradient_accumulation_steps: int = 1  # Mettre à 2 si batch_size=8
    
    # Mixed precision (pour GPU)
    use_amp: bool = False  # Mettre True si GPU compatible


# Instanciation par défaut
model_config = ModelConfig()
training_config = TrainingConfig()


# Mapping des labels
EMOTION_LABELS = {
    0: "joie",
    1: "tristesse", 
    2: "colere",
    3: "peur",
    4: "surprise",
    5: "degout",
    6: "neutre"
}

SENTIMENT_LABELS = {
    0: "negatif",
    1: "neutre",
    2: "positif"
}

IRONY_LABELS = {
    0: "non_ironique",
    1: "ironique"
}
