"""
Utilitaires pour l'entraînement
"""

import torch
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_optimizer_with_layerwise_lr(model, lr_encoder, lr_classifier, weight_decay=0.01):
    """
    Crée un optimiseur avec learning rates différenciés
    
    Args:
        model: Modèle CamemBERT
        lr_encoder: Learning rate pour l'encodeur
        lr_classifier: Learning rate pour les têtes
        weight_decay: Weight decay
    
    Returns:
        Optimiseur AdamW
    """
    optimizer = torch.optim.AdamW([
        {'params': model.get_encoder_params(), 'lr': lr_encoder},
        {'params': model.get_classifier_params(), 'lr': lr_classifier}
    ], weight_decay=weight_decay)
    
    return optimizer


def create_scheduler(optimizer, num_training_steps, num_warmup_steps):
    """
    Crée un scheduler avec warmup linéaire
    
    Args:
        optimizer: Optimiseur
        num_training_steps: Nombre total de steps
        num_warmup_steps: Nombre de steps de warmup
    
    Returns:
        Scheduler
    """
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return scheduler


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Charge un checkpoint
    
    Args:
        model: Modèle
        optimizer: Optimiseur
        checkpoint_path: Chemin vers le checkpoint
    
    Returns:
        Dict avec epoch et métriques
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'val_metrics': checkpoint.get('val_metrics', {})
    }


def plot_training_history(history: Dict[str, Any], save_path: str = None):
    """
    Visualise l'historique d'entraînement
    
    Args:
        history: Dictionnaire avec train et val metrics
        save_path: Chemin de sauvegarde optionnel
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Historique d\'Entraînement', fontsize=16)
    
    epochs = range(1, len(history['train']) + 1)
    
    # Loss totale
    axes[0, 0].plot(epochs, [m['loss'] for m in history['train']], 'b-', label='Train')
    axes[0, 0].plot(epochs, [m['loss'] for m in history['val']], 'r-', label='Val')
    axes[0, 0].set_title('Loss Totale')
    axes[0, 0].set_xlabel('Époque')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Emotion F1
    axes[0, 1].plot(epochs, [m['emotion_f1'] for m in history['train']], 'b-', label='Train')
    axes[0, 1].plot(epochs, [m['emotion_f1'] for m in history['val']], 'r-', label='Val')
    axes[0, 1].set_title('Émotion F1-Score')
    axes[0, 1].set_xlabel('Époque')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Sentiment Accuracy
    axes[1, 0].plot(epochs, [m['sentiment_acc'] for m in history['train']], 'b-', label='Train')
    axes[1, 0].plot(epochs, [m['sentiment_acc'] for m in history['val']], 'r-', label='Val')
    axes[1, 0].set_title('Sentiment Accuracy')
    axes[1, 0].set_xlabel('Époque')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Irony F1
    axes[1, 1].plot(epochs, [m['irony_f1'] for m in history['train']], 'b-', label='Train')
    axes[1, 1].plot(epochs, [m['irony_f1'] for m in history['val']], 'r-', label='Val')
    axes[1, 1].set_title('Ironie F1-Score')
    axes[1, 1].set_xlabel('Époque')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphique sauvegardé : {save_path}")
    
    plt.show()


def print_model_summary(model):
    """
    Affiche un résumé du modèle
    
    Args:
        model: Modèle PyTorch
    """
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ DU MODÈLE")
    print("=" * 80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Paramètres totaux: {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,}")
    print(f"Paramètres gelés: {total_params - trainable_params:,}")
    
    # Taille mémoire
    param_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes par float32
    print(f"Taille mémoire: ~{param_size_mb:.2f} MB")
    
    print("=" * 80 + "\n")


def format_time(seconds: float) -> str:
    """
    Formate un temps en secondes en format lisible
    
    Args:
        seconds: Temps en secondes
    
    Returns:
        String formaté
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

