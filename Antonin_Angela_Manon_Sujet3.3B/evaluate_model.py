#!/usr/bin/env python3
"""
Script d'évaluation du modèle CamemBERT sur le test set

Usage:
    python evaluate_model.py
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent / "src"))

from models.camembert_multitask import CamemBERTMultitask
from models.config import EMOTION_LABELS, SENTIMENT_LABELS, IRONY_LABELS
from training.train import MultiTaskDataset
from transformers import CamembertTokenizer


def load_model(checkpoint_path: str, device: str):
    """
    Charge le modèle depuis un checkpoint
    
    Args:
        checkpoint_path: Chemin vers le checkpoint
        device: Device
    
    Returns:
        Modèle chargé
    """
    print(f"📥 Chargement du modèle depuis {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_config = checkpoint['model_config']
    model = CamemBERTMultitask(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✅ Modèle chargé avec succès")
    return model


def evaluate_on_test(model, test_loader, device):
    """
    Évalue le modèle sur le test set
    
    Args:
        model: Modèle
        test_loader: DataLoader de test
        device: Device
    
    Returns:
        Dictionnaire avec prédictions et labels
    """
    print("\n🧪 Évaluation sur le test set...")
    
    model.eval()
    
    all_emotion_preds = []
    all_emotion_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_irony_preds = []
    all_irony_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            emotion_preds, sentiment_preds, irony_preds = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_labels.extend(batch['emotion_label'].numpy())
            all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
            all_sentiment_labels.extend(batch['sentiment_label'].numpy())
            all_irony_preds.extend(irony_preds.cpu().numpy())
            all_irony_labels.extend(batch['irony_label'].numpy())
    
    return {
        'emotion': {
            'preds': np.array(all_emotion_preds),
            'labels': np.array(all_emotion_labels)
        },
        'sentiment': {
            'preds': np.array(all_sentiment_preds),
            'labels': np.array(all_sentiment_labels)
        },
        'irony': {
            'preds': np.array(all_irony_preds),
            'labels': np.array(all_irony_labels)
        }
    }


def plot_confusion_matrices(results, save_dir="results"):
    """
    Génère les matrices de confusion pour les 3 tâches
    
    Args:
        results: Résultats de l'évaluation
        save_dir: Dossier de sauvegarde
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Matrices de Confusion - Test Set', fontsize=16)
    
    # Émotion
    cm_emotion = confusion_matrix(results['emotion']['labels'], results['emotion']['preds'])
    sns.heatmap(cm_emotion, annot=True, fmt='d', ax=axes[0], cmap='Blues',
                xticklabels=list(EMOTION_LABELS.values()),
                yticklabels=list(EMOTION_LABELS.values()))
    axes[0].set_title('Émotions (7 classes)')
    axes[0].set_ylabel('Vrai')
    axes[0].set_xlabel('Prédit')
    
    # Sentiment
    cm_sentiment = confusion_matrix(results['sentiment']['labels'], results['sentiment']['preds'])
    sns.heatmap(cm_sentiment, annot=True, fmt='d', ax=axes[1], cmap='Greens',
                xticklabels=list(SENTIMENT_LABELS.values()),
                yticklabels=list(SENTIMENT_LABELS.values()))
    axes[1].set_title('Sentiment (3 classes)')
    axes[1].set_ylabel('Vrai')
    axes[1].set_xlabel('Prédit')
    
    # Ironie
    cm_irony = confusion_matrix(results['irony']['labels'], results['irony']['preds'])
    sns.heatmap(cm_irony, annot=True, fmt='d', ax=axes[2], cmap='Oranges',
                xticklabels=list(IRONY_LABELS.values()),
                yticklabels=list(IRONY_LABELS.values()))
    axes[2].set_title('Ironie (2 classes)')
    axes[2].set_ylabel('Vrai')
    axes[2].set_xlabel('Prédit')
    
    plt.tight_layout()
    
    save_path = save_dir / "confusion_matrices.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Matrices de confusion sauvegardées: {save_path}")
    plt.show()


def print_classification_reports(results):
    """
    Affiche les rapports de classification pour les 3 tâches
    
    Args:
        results: Résultats de l'évaluation
    """
    print("\n" + "=" * 80)
    print("📋 RAPPORTS DE CLASSIFICATION")
    print("=" * 80)
    
    # Émotion
    print("\n🎭 ÉMOTIONS:")
    print(classification_report(
        results['emotion']['labels'],
        results['emotion']['preds'],
        target_names=list(EMOTION_LABELS.values()),
        digits=4
    ))
    
    # Sentiment
    print("\n💭 SENTIMENT:")
    print(classification_report(
        results['sentiment']['labels'],
        results['sentiment']['preds'],
        target_names=list(SENTIMENT_LABELS.values()),
        digits=4
    ))
    
    # Ironie
    print("\n😏 IRONIE:")
    print(classification_report(
        results['irony']['labels'],
        results['irony']['preds'],
        target_names=list(IRONY_LABELS.values()),
        digits=4
    ))


def main():
    """Fonction principale"""
    
    print("=" * 80)
    print("🎯 ÉVALUATION DU MODÈLE CAMEMBERT")
    print("=" * 80)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    # Charger le modèle
    checkpoint_path = "models/best_model.pt"
    model = load_model(checkpoint_path, device)
    
    # Charger les données de test
    print("\n📂 Chargement des données de test...")
    test_df = pd.read_csv("data/processed/test.csv")
    print(f"   ✓ Test: {len(test_df)} exemples")
    
    # Tokenizer
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    
    # Dataset et DataLoader
    test_dataset = MultiTaskDataset(
        texts=test_df['text'].tolist(),
        emotion_labels=test_df['emotion_id'].tolist(),
        sentiment_labels=test_df['sentiment'].tolist(),
        irony_labels=test_df['is_ironic'].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Évaluation
    results = evaluate_on_test(model, test_loader, device)
    
    # Afficher les rapports
    print_classification_reports(results)
    
    # Générer les matrices de confusion
    plot_confusion_matrices(results)
    
    print("\n✅ Évaluation terminée !")


if __name__ == "__main__":
    main()
