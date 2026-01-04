import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import classification_report
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from models.baseline import BaselineModel

def train_baseline():
    print("🚀 Entraînement de la Baseline (TF-IDF + LogReg)...")
    
    # Load data
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("❌ Dossier data/processed non trouvé. Lancez d'abord le téléchargement des données.")
        return

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # Fill NaN with -1 for tasks where label is missing
    train_df = train_df.fillna(-1)
    test_df = test_df.fillna(-1)
    
    tasks = [
        ('emotion_label', 'emotion'),
        ('sentiment_label', 'sentiment'),
        ('irony_label', 'irony')
    ]
    
    for label_col, task_name in tasks:
        print(f"\n📊 Tâche : {task_name}")
        
        # Filter data where label is present (not -1)
        train_task = train_df[train_df[label_col] != -1]
        test_task = test_df[test_df[label_col] != -1]
        
        if len(train_task) == 0:
            print(f"⚠️ Pas de données d'entraînement pour {task_name}")
            continue
            
        print(f"   Entraînement sur {len(train_task)} exemples...")
        model = BaselineModel(task_name=task_name)
        model.fit(train_task['text'].astype(str).tolist(), train_task[label_col].tolist())
        
        if len(test_task) > 0:
            preds = model.predict(test_task['text'].astype(str).tolist())
            print(classification_report(test_task[label_col].tolist(), preds))
        else:
            print("⚠️ Pas de données de test pour cette tâche.")

if __name__ == "__main__":
    train_baseline()
