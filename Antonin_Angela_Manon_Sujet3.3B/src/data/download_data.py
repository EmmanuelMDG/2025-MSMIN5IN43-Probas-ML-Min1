"""
Script de téléchargement et préparation des datasets français
Utilise Allociné (HuggingFace) pour le sentiment et des données synthétiques pour le reste.
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json

# Configuration
np.random.seed(42)
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 70)
print("TÉLÉCHARGEMENT DES DATASETS FRANÇAIS")
print("=" * 70)

def get_allocine_data():
    print("\n📥 1. Téléchargement du dataset Allociné (Sentiment)...")
    try:
        dataset = load_dataset("allocine")
        
        # Subsampling pour aller vite
        train_df = pd.DataFrame(dataset['train']).sample(n=2000, random_state=42)
        val_df = pd.DataFrame(dataset['validation']).sample(n=500, random_state=42)
        test_df = pd.DataFrame(dataset['test']).sample(n=500, random_state=42)
        
        # Mapping: 0=neg, 1=pos -> 0=neg, 2=pos (1=neutre)
        def map_sentiment(x):
            return 0 if x == 0 else 2
            
        for df in [train_df, val_df, test_df]:
            df['sentiment_label'] = df['label'].apply(map_sentiment)
            df['text'] = df['review']
            df['emotion_label'] = -1
            df['irony_label'] = -1
            
        return train_df, val_df, test_df
    except Exception as e:
        print(f"Erreur lors du chargement d'Allociné: {e}")
        print("Utilisation de données synthétiques pour Allociné.")
        return None, None, None

def get_synthetic_data():
    print("\n📥 2. Génération de données synthétiques (Émotions/Ironie)...")
    
    # --- EMOTIONS ---
    emotion_examples = {
        'joie': ["Je suis trop content !", "C'est génial !", "J'adore ça ❤️", "Quelle bonne nouvelle !"],
        'tristesse': ["Je suis triste 😢", "C'est déprimant", "Je me sens seul", "Quelle déception"],
        'colere': ["Ça m'énerve ! 😡", "C'est inadmissible", "Je suis furieux", "N'importe quoi !"],
        'peur': ["J'ai peur 😨", "C'est effrayant", "Je suis angoissé", "Au secours !"],
        'surprise': ["Oh ! Vraiment ? 😮", "Je ne m'y attendais pas", "Incroyable !", "Wow !"],
        'degout': ["C'est dégoûtant 🤢", "Beurk", "J'ai la nausée", "C'est immonde"],
        'neutre': ["Il fait beau.", "Je vais au travail.", "Le ciel est bleu.", "J'ai mangé une pomme."]
    }
    
    emotion_mapping = {'joie': 0, 'tristesse': 1, 'colere': 2, 'peur': 3, 'surprise': 4, 'degout': 5, 'neutre': 6}
    
    data = []
    for emo, texts in emotion_examples.items():
        for t in texts:
            for _ in range(20): # Augmenter la taille
                data.append({
                    'text': t,
                    'emotion_label': emotion_mapping[emo],
                    'sentiment_label': -1, # On pourrait inférer mais restons simple
                    'irony_label': -1
                })
                
    # --- IRONIE ---
    ironic = ["Super, il pleut encore ! 🙄", "Génial ce bouchon...", "Merci pour ce cadeau inutile.", "Bravo champion !"]
    not_ironic = ["Il pleut aujourd'hui.", "Il y a des bouchons.", "Merci pour le cadeau.", "Bravo pour ta victoire."]
    
    for t in ironic:
        for _ in range(25):
            data.append({'text': t, 'emotion_label': -1, 'sentiment_label': -1, 'irony_label': 1})
            
    for t in not_ironic:
        for _ in range(25):
            data.append({'text': t, 'emotion_label': -1, 'sentiment_label': -1, 'irony_label': 0})
            
    df = pd.DataFrame(data)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    
    return train, val, test

def main():
    # 1. Allociné
    train_allocine, val_allocine, test_allocine = get_allocine_data()
    
    # 2. Synthetic
    train_syn, val_syn, test_syn = get_synthetic_data()
    
    # 3. Merge
    cols = ['text', 'emotion_label', 'sentiment_label', 'irony_label']
    
    if train_allocine is not None:
        train_final = pd.concat([train_allocine[cols], train_syn[cols]])
        val_final = pd.concat([val_allocine[cols], val_syn[cols]])
        test_final = pd.concat([test_allocine[cols], test_syn[cols]])
    else:
        train_final = train_syn[cols]
        val_final = val_syn[cols]
        test_final = test_syn[cols]
        
    # Shuffle
    train_final = train_final.sample(frac=1, random_state=42).reset_index(drop=True)
    val_final = val_final.sample(frac=1, random_state=42).reset_index(drop=True)
    test_final = test_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Dataset final généré :")
    print(f"   Train: {len(train_final)}")
    print(f"   Val:   {len(val_final)}")
    print(f"   Test:  {len(test_final)}")
    
    train_final.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_final.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_final.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    
    # Stats
    stats = {
        'train_size': len(train_final),
        'val_size': len(val_final),
        'test_size': len(test_final)
    }
    with open(os.path.join(DATA_DIR, "dataset_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
