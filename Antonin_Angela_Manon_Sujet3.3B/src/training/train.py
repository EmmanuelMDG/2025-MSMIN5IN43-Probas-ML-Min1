"""
Script d'entraînement du modèle CamemBERT Multi-tâches
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, Tuple, Optional
from sklearn.metrics import f1_score, accuracy_score, classification_report

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.camembert_multitask import CamemBERTMultitask
from models.config import ModelConfig, TrainingConfig, set_seed
from models.config import EMOTION_LABELS, SENTIMENT_LABELS, IRONY_LABELS


# =============================================================================
# DATASET PYTORCH
# =============================================================================

class MultiTaskDataset(Dataset):
    """Dataset PyTorch pour le multi-tâches"""
    
    def __init__(
        self,
        texts: list,
        emotion_labels: list,
        sentiment_labels: list,
        irony_labels: list,
        tokenizer: CamembertTokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.emotion_labels = emotion_labels
        self.sentiment_labels = sentiment_labels
        self.irony_labels = irony_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenizer le texte
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emotion_label': torch.tensor(self.emotion_labels[idx], dtype=torch.long),
            'sentiment_label': torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
            'irony_label': torch.tensor(self.irony_labels[idx], dtype=torch.long)
        }


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les datasets train/val/test
    
    Args:
        data_path: Chemin vers le dossier data/processed/
    
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    data_dir = Path(data_path)
    
    print("📂 Chargement des données...")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    print(f"   ✓ Train: {len(train_df)} exemples")
    print(f"   ✓ Val: {len(val_df)} exemples")
    print(f"   ✓ Test: {len(test_df)} exemples")
    
    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: CamembertTokenizer,
    config: TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders PyTorch
    
    Args:
        train_df: DataFrame d'entraînement
        val_df: DataFrame de validation
        tokenizer: Tokenizer CamemBERT
        config: Configuration d'entraînement
    
    Returns:
        Tuple (train_loader, val_loader)
    """
    # Créer les datasets
    train_dataset = MultiTaskDataset(
        texts=train_df['text'].tolist(),
        emotion_labels=train_df['emotion_id'].tolist(),
        sentiment_labels=train_df['sentiment'].tolist(),
        irony_labels=train_df['is_ironic'].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )
    
    val_dataset = MultiTaskDataset(
        texts=val_df['text'].tolist(),
        emotion_labels=val_df['emotion_id'].tolist(),
        sentiment_labels=val_df['sentiment'].tolist(),
        irony_labels=val_df['is_ironic'].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # 0 pour éviter des problèmes sur Mac
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_epoch(
    model: CamemBERTMultitask,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    config: TrainingConfig
) -> Dict[str, float]:
    """
    Entraîne le modèle sur une époque
    
    Args:
        model: Modèle à entraîner
        train_loader: DataLoader d'entraînement
        optimizer: Optimiseur
        device: Device (cpu/cuda/mps)
        config: Configuration
    
    Returns:
        Dictionnaire des métriques
    """
    model.train()
    
    total_loss = 0
    emotion_losses = 0
    sentiment_losses = 0
    irony_losses = 0
    
    # Prédictions et vraies valeurs pour les métriques
    all_emotion_preds = []
    all_emotion_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_irony_preds = []
    all_irony_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Déplacer les données sur le device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        emotion_labels = batch['emotion_label'].to(device)
        sentiment_labels = batch['sentiment_label'].to(device)
        irony_labels = batch['irony_label'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            emotion_labels=emotion_labels,
            sentiment_labels=sentiment_labels,
            irony_labels=irony_labels
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Statistiques
        total_loss += loss.item()
        emotion_losses += outputs['emotion_loss'].item()
        sentiment_losses += outputs['sentiment_loss'].item()
        irony_losses += outputs['irony_loss'].item()
        
        # Prédictions
        emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
        sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1)
        irony_preds = torch.argmax(outputs['irony_logits'], dim=1)
        
        all_emotion_preds.extend(emotion_preds.cpu().numpy())
        all_emotion_labels.extend(emotion_labels.cpu().numpy())
        all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
        all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
        all_irony_preds.extend(irony_preds.cpu().numpy())
        all_irony_labels.extend(irony_labels.cpu().numpy())
        
        # Mise à jour de la barre de progression
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculer les métriques
    n_batches = len(train_loader)
    emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    sentiment_acc = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    irony_f1 = f1_score(all_irony_labels, all_irony_preds, average='macro')
    
    return {
        'loss': total_loss / n_batches,
        'emotion_loss': emotion_losses / n_batches,
        'sentiment_loss': sentiment_losses / n_batches,
        'irony_loss': irony_losses / n_batches,
        'emotion_f1': emotion_f1,
        'sentiment_acc': sentiment_acc,
        'irony_f1': irony_f1
    }


def validate(
    model: CamemBERTMultitask,
    val_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Évalue le modèle sur le set de validation
    
    Args:
        model: Modèle à évaluer
        val_loader: DataLoader de validation
        device: Device
    
    Returns:
        Dictionnaire des métriques
    """
    model.eval()
    
    total_loss = 0
    emotion_losses = 0
    sentiment_losses = 0
    irony_losses = 0
    
    all_emotion_preds = []
    all_emotion_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_irony_preds = []
    all_irony_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            irony_labels = batch['irony_label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                emotion_labels=emotion_labels,
                sentiment_labels=sentiment_labels,
                irony_labels=irony_labels
            )
            
            total_loss += outputs['loss'].item()
            emotion_losses += outputs['emotion_loss'].item()
            sentiment_losses += outputs['sentiment_loss'].item()
            irony_losses += outputs['irony_loss'].item()
            
            # Prédictions
            emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1)
            irony_preds = torch.argmax(outputs['irony_logits'], dim=1)
            
            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
            all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
            all_irony_preds.extend(irony_preds.cpu().numpy())
            all_irony_labels.extend(irony_labels.cpu().numpy())
    
    # Métriques
    n_batches = len(val_loader)
    emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    sentiment_acc = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    irony_f1 = f1_score(all_irony_labels, all_irony_preds, average='macro')
    
    return {
        'loss': total_loss / n_batches,
        'emotion_loss': emotion_losses / n_batches,
        'sentiment_loss': sentiment_losses / n_batches,
        'irony_loss': irony_losses / n_batches,
        'emotion_f1': emotion_f1,
        'sentiment_acc': sentiment_acc,
        'irony_f1': irony_f1
    }


# =============================================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# =============================================================================

def train_model(
    data_path: str = "data/processed",
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None
):
    """
    Fonction principale d'entraînement
    
    Args:
        data_path: Chemin vers les données
        model_config: Configuration du modèle
        training_config: Configuration d'entraînement
    """
    # Charger les configs par défaut si non fournies
    if model_config is None:
        from models.config import model_config as default_model_config
        model_config = default_model_config
    
    if training_config is None:
        from models.config import training_config as default_training_config
        training_config = default_training_config
    
    # Fixer les seeds
    set_seed(training_config.seed)
    
    # Device
    device = training_config.device
    print(f"🖥️  Device: {device}")
    
    # Charger les données
    train_df, val_df, test_df = load_data(data_path)
    
    # Tokenizer
    print(f"\n📥 Chargement du tokenizer {model_config.model_name}...")
    tokenizer = CamembertTokenizer.from_pretrained(model_config.model_name)
    
    # DataLoaders
    print("\n🔄 Création des DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, tokenizer, training_config
    )
    
    # Modèle
    print("\n🤖 Initialisation du modèle...")
    model = CamemBERTMultitask(model_config)
    model = model.to(device)
    
    # Optimiseur avec learning rates différenciés
    print("\n⚙️  Configuration de l'optimiseur...")
    optimizer = torch.optim.AdamW([
        {'params': model.get_encoder_params(), 'lr': training_config.lr_encoder},
        {'params': model.get_classifier_params(), 'lr': training_config.lr_classifier}
    ], weight_decay=training_config.weight_decay)
    
    # Early stopping
    best_val_f1 = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    # Créer le dossier de sauvegarde
    save_dir = Path(training_config.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n🚀 Début de l'entraînement ({training_config.num_epochs} époques)")
    print("=" * 80)
    
    # Boucle d'entraînement
    for epoch in range(training_config.num_epochs):
        print(f"\n📍 Époque {epoch + 1}/{training_config.num_epochs}")
        
        # Entraînement
        train_metrics = train_epoch(model, train_loader, optimizer, device, training_config)
        
        # Validation
        val_metrics = validate(model, val_loader, device)
        
        # Sauvegarder l'historique
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Afficher les résultats
        print(f"\n📊 Résultats Époque {epoch + 1}:")
        print(f"   Train - Loss: {train_metrics['loss']:.4f} | "
              f"Emotion F1: {train_metrics['emotion_f1']:.4f} | "
              f"Sentiment Acc: {train_metrics['sentiment_acc']:.4f} | "
              f"Irony F1: {train_metrics['irony_f1']:.4f}")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Emotion F1: {val_metrics['emotion_f1']:.4f} | "
              f"Sentiment Acc: {val_metrics['sentiment_acc']:.4f} | "
              f"Irony F1: {val_metrics['irony_f1']:.4f}")
        
        # Métrique combinée pour early stopping (moyenne des 3 F1/Acc)
        val_combined_score = (val_metrics['emotion_f1'] + 
                             val_metrics['sentiment_acc'] + 
                             val_metrics['irony_f1']) / 3
        
        # Early stopping et sauvegarde
        if val_combined_score > best_val_f1 + training_config.min_delta:
            best_val_f1 = val_combined_score
            patience_counter = 0
            
            # Sauvegarder le meilleur modèle
            print(f"\n💾 Nouveau meilleur modèle ! Score: {best_val_f1:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'model_config': model_config,
                'training_config': training_config
            }, save_dir / "best_model.pt")
        else:
            patience_counter += 1
            print(f"\n⏳ Patience: {patience_counter}/{training_config.patience}")
            
            if patience_counter >= training_config.patience:
                print(f"\n🛑 Early stopping déclenché !")
                break
    
    print("\n" + "=" * 80)
    print("✅ Entraînement terminé !")
    print(f"📈 Meilleur score de validation: {best_val_f1:.4f}")
    
    # Sauvegarder l'historique
    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n💾 Modèle sauvegardé dans : {save_dir}")
    
    return model, history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Lancer l'entraînement
    model, history = train_model(
        data_path="../../data/processed"
    )

