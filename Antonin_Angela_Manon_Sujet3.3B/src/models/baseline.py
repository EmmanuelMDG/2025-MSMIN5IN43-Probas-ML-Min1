"""
Modèle Baseline : TF-IDF + Logistic Regression

Ce module implémente un modèle baseline simple pour établir une performance
de référence avant d'utiliser des modèles deep learning comme CamemBERT.

Architecture : TF-IDF vectorization + Logistic Regression (3 modèles séparés)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class BaselineModel:
    """
    Modèle baseline avec TF-IDF + Logistic Regression
    
    Attributes:
        task_name (str): Nom de la tâche ('emotion', 'sentiment', 'irony')
        vectorizer (TfidfVectorizer): Vectoriseur TF-IDF
        classifier (LogisticRegression): Classificateur
        label_encoder (LabelEncoder): Encodeur pour les labels
    """
    
    def __init__(
        self,
        task_name: str,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        class_weight: Optional[str] = 'balanced',
        random_state: int = 42
    ):
        """
        Initialise le modèle baseline
        
        Args:
            task_name: Nom de la tâche ('emotion', 'sentiment', 'irony')
            max_features: Nombre maximum de features TF-IDF
            ngram_range: Plage de n-grams (unigrams + bigrams par défaut)
            class_weight: Pondération des classes ('balanced' pour gérer le déséquilibre)
            random_state: Seed pour la reproductibilité
        """
        self.task_name = task_name
        self.random_state = random_state
        
        # Vectoriseur TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignorer les mots qui apparaissent moins de 2 fois
            max_df=0.8,  # Ignorer les mots trop fréquents (>80%)
            strip_accents=None,  # Garder les accents (français)
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b|[!?]{1,}|[😀-🙏]'  # Mots + ponctuation + emojis
        )
        
        # Classificateur
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
            solver='lbfgs'
        )
        
        # Encodeur de labels
        self.label_encoder = LabelEncoder()
        
        self.is_fitted = False
    
    def fit(self, texts: List[str], labels: List[str]) -> 'BaselineModel':
        """
        Entraîne le modèle sur les données
        
        Args:
            texts: Liste de textes
            labels: Liste de labels correspondants
            
        Returns:
            self: Le modèle entraîné
        """
        # Encoder les labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Vectoriser les textes
        X = self.vectorizer.fit_transform(texts)
        
        # Entraîner le classificateur
        self.classifier.fit(X, y_encoded)
        
        self.is_fitted = True
        return self
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Prédit les labels pour de nouveaux textes
        
        Args:
            texts: Liste de textes
            
        Returns:
            predictions: Liste de labels prédits
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        # Vectoriser
        X = self.vectorizer.transform(texts)
        
        # Prédire
        y_pred_encoded = self.classifier.predict(X)
        
        # Décoder les labels
        predictions = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return predictions.tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Prédit les probabilités pour chaque classe
        
        Args:
            texts: Liste de textes
            
        Returns:
            probabilities: Matrice de probabilités (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retourne les features les plus importantes pour chaque classe
        
        Args:
            top_n: Nombre de features à retourner par classe
            
        Returns:
            feature_importance: Dictionnaire {classe: [(feature, importance), ...]}
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné")
        
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.classifier.coef_
        
        importance_dict = {}
        
        # Pour la classification binaire, coef_ a la shape (1, n_features)
        # Pour multi-classes, coef_ a la shape (n_classes, n_features)
        if len(self.label_encoder.classes_) == 2:
            # Binaire : une seule ligne de coefficients
            # Classe 0 : coefficients négatifs, Classe 1 : coefficients positifs
            for idx, class_name in enumerate(self.label_encoder.classes_):
                if idx == 0:
                    # Classe négative : on prend les features avec les coefficients les plus négatifs
                    top_indices = np.argsort(coefficients[0])[:top_n]
                else:
                    # Classe positive : on prend les features avec les coefficients les plus positifs
                    top_indices = np.argsort(coefficients[0])[-top_n:][::-1]
                
                top_features = [
                    (feature_names[i], float(coefficients[0][i]))
                    for i in top_indices
                ]
                
                importance_dict[class_name] = top_features
        else:
            # Multi-classes : une ligne par classe
            for idx, class_name in enumerate(self.label_encoder.classes_):
                # Obtenir les indices des top_n coefficients
                top_indices = np.argsort(coefficients[idx])[-top_n:][::-1]
                
                # Récupérer les features et leurs poids
                top_features = [
                    (feature_names[i], float(coefficients[idx][i]))
                    for i in top_indices
                ]
                
                importance_dict[class_name] = top_features
        
        return importance_dict
    
    def save(self, save_dir: Path) -> None:
        """
        Sauvegarde le modèle
        
        Args:
            save_dir: Répertoire de sauvegarde
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f'baseline_{self.task_name}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'task_name': self.task_name,
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'is_fitted': self.is_fitted
            }, f)
        
        print(f"✅ Modèle sauvegardé : {model_path}")
    
    @classmethod
    def load(cls, model_path: Path) -> 'BaselineModel':
        """
        Charge un modèle sauvegardé
        
        Args:
            model_path: Chemin vers le fichier .pkl
            
        Returns:
            model: Modèle chargé
        """
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(task_name=data['task_name'])
        model.vectorizer = data['vectorizer']
        model.classifier = data['classifier']
        model.label_encoder = data['label_encoder']
        model.is_fitted = data['is_fitted']
        
        return model
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"BaselineModel(task={self.task_name}, status={status})"
