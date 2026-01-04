"""
Modèle CamemBERT multi-tâches
Architecture : Un encodeur partagé + 3 têtes de classification
"""

import torch
import torch.nn as nn
from transformers import CamembertModel, CamembertConfig
from typing import Dict, Tuple, Optional
from .config import ModelConfig


class CamemBERTMultitask(nn.Module):
    """
    Modèle CamemBERT pour classification multi-tâches
    
    Architecture :
        Texte → CamemBERT (encodeur partagé) → [CLS] token
                                                    ↓
                                        ┌───────────┼───────────┐
                                        ↓           ↓           ↓
                                    Émotions    Sentiment    Ironie
                                    (7 classes) (3 classes)  (2 classes)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialise le modèle
        
        Args:
            config: Configuration du modèle
        """
        super(CamemBERTMultitask, self).__init__()
        
        if config is None:
            from .config import model_config
            config = model_config
        
        self.config = config
        
        # 1. Encodeur CamemBERT partagé (pré-entraîné)
        print(f"📥 Chargement de {config.model_name}...")
        self.camembert = CamembertModel.from_pretrained(config.model_name)
        
        # 2. Dropout pour régularisation
        self.dropout = nn.Dropout(config.dropout)
        
        # 3. Les 3 têtes de classification
        # Chaque tête est un simple linear layer : hidden_size → num_classes
        
        # Tête Émotions (7 classes)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_emotions)
        )
        
        # Tête Sentiment (3 classes)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_sentiments)
        )
        
        # Tête Ironie (2 classes)
        self.irony_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_irony)
        )
        
        # 4. Loss functions
        self.criterion_emotion = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_sentiment = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_irony = nn.CrossEntropyLoss(ignore_index=-1)
        
        print(f"✅ Modèle créé avec {self.count_parameters():,} paramètres")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emotion_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        irony_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass du modèle
        
        Args:
            input_ids: Tokens d'entrée (batch_size, seq_length)
            attention_mask: Masque d'attention (batch_size, seq_length)
            emotion_labels: Labels pour émotions (batch_size,)
            sentiment_labels: Labels pour sentiment (batch_size,)
            irony_labels: Labels pour ironie (batch_size,)
        
        Returns:
            Dictionnaire contenant les logits et les losses
        """
        # 1. Passer par CamemBERT pour obtenir les embeddings
        outputs = self.camembert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 2. Extraire la représentation du token [CLS] (premier token)
        # C'est cette représentation qui encode tout le sens de la phrase
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        
        # 3. Appliquer le dropout
        pooled_output = self.dropout(pooled_output)
        
        # 4. Passer dans les 3 têtes de classification
        emotion_logits = self.emotion_classifier(pooled_output)    # (batch_size, 7)
        sentiment_logits = self.sentiment_classifier(pooled_output) # (batch_size, 3)
        irony_logits = self.irony_classifier(pooled_output)        # (batch_size, 2)
        
        # 5. Calculer les losses si les labels sont fournis
        loss = None
        emotion_loss = None
        sentiment_loss = None
        irony_loss = None
        
        if emotion_labels is not None and sentiment_labels is not None and irony_labels is not None:
            # Calculer les 3 losses
            emotion_loss = self.criterion_emotion(emotion_logits, emotion_labels)
            sentiment_loss = self.criterion_sentiment(sentiment_logits, sentiment_labels)
            irony_loss = self.criterion_irony(irony_logits, irony_labels)
            
            # Loss totale pondérée
            loss = (
                self.config.loss_weight_emotion * emotion_loss +
                self.config.loss_weight_sentiment * sentiment_loss +
                self.config.loss_weight_irony * irony_loss
            )
        
        # 6. Retourner tout dans un dictionnaire
        return {
            'loss': loss,
            'emotion_loss': emotion_loss,
            'sentiment_loss': sentiment_loss,
            'irony_loss': irony_loss,
            'emotion_logits': emotion_logits,
            'sentiment_logits': sentiment_logits,
            'irony_logits': irony_logits
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fait des prédictions (sans calculer les losses)
        
        Args:
            input_ids: Tokens d'entrée
            attention_mask: Masque d'attention
        
        Returns:
            Tuple (emotion_preds, sentiment_preds, irony_preds)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            # Prendre les classes avec la plus haute probabilité
            emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1)
            irony_preds = torch.argmax(outputs['irony_logits'], dim=1)
        
        return emotion_preds, sentiment_preds, irony_preds
    
    def count_parameters(self) -> int:
        """Compte le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_encoder_params(self):
        """Retourne les paramètres de l'encodeur CamemBERT"""
        return self.camembert.parameters()
    
    def get_classifier_params(self):
        """Retourne les paramètres des têtes de classification"""
        params = []
        params.extend(self.emotion_classifier.parameters())
        params.extend(self.sentiment_classifier.parameters())
        params.extend(self.irony_classifier.parameters())
        return params
    
    def freeze_encoder(self):
        """Gèle l'encodeur CamemBERT (pour fine-tuning progressif)"""
        for param in self.camembert.parameters():
            param.requires_grad = False
        print("🔒 Encodeur CamemBERT gelé")
    
    def unfreeze_encoder(self):
        """Dégèle l'encodeur CamemBERT"""
        for param in self.camembert.parameters():
            param.requires_grad = True
        print("🔓 Encodeur CamemBERT dégelé")


if __name__ == "__main__":
    # Test rapide du modèle
    print("🧪 Test du modèle CamemBERT Multi-tâches\n")
    
    from .config import model_config
    
    # Créer le modèle
    model = CamemBERTMultitask(model_config)
    
    # Simuler un batch
    batch_size = 4
    seq_length = 32
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    emotion_labels = torch.randint(0, 7, (batch_size,))
    sentiment_labels = torch.randint(0, 3, (batch_size,))
    irony_labels = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        emotion_labels=emotion_labels,
        sentiment_labels=sentiment_labels,
        irony_labels=irony_labels
    )
    
    print(f"\n📊 Résultats du test :")
    print(f"   Loss totale: {outputs['loss'].item():.4f}")
    print(f"   Loss émotions: {outputs['emotion_loss'].item():.4f}")
    print(f"   Loss sentiment: {outputs['sentiment_loss'].item():.4f}")
    print(f"   Loss ironie: {outputs['irony_loss'].item():.4f}")
    print(f"\n   Shapes des logits:")
    print(f"   - Émotions: {outputs['emotion_logits'].shape}")
    print(f"   - Sentiment: {outputs['sentiment_logits'].shape}")
    print(f"   - Ironie: {outputs['irony_logits'].shape}")
    
    print("\n✅ Test réussi !")

