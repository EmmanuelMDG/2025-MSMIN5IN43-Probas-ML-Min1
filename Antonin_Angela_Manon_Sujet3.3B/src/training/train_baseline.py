"""
Script d'entraînement des modèles baseline

Ce script entraîne 3 modèles TF-IDF + Logistic Regression (un par tâche)
et évalue leurs performances sur les ensembles train, validation et test.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent))

from models.baseline import BaselineModel
from evaluation.metrics import (
    evaluate_model,
    save_results,
    plot_confusion_matrix,
    print_results_summary
)


def load_data(data_dir: Path, preprocessed: bool = True):
    """
    Charge les données preprocessées
    
    Args:
        data_dir: Répertoire contenant les données
        preprocessed: Si True, charge les données preprocessées
        
    Returns:
        train_df, val_df, test_df
    """
    suffix = '_preprocessed' if preprocessed else ''
    
    train_df = pd.read_csv(data_dir / f'train{suffix}.csv')
    val_df = pd.read_csv(data_dir / f'val{suffix}.csv')
    test_df = pd.read_csv(data_dir / f'test{suffix}.csv')
    
    print(f"✅ Données chargées :")
    print(f"   - Train : {len(train_df)} exemples")
    print(f"   - Val   : {len(val_df)} exemples")
    print(f"   - Test  : {len(test_df)} exemples\n")
    
    return train_df, val_df, test_df


def train_and_evaluate_task(
    task_name: str,
    label_column: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: Path,
    results_dir: Path
):
    """
    Entraîne et évalue un modèle pour une tâche spécifique
    
    Args:
        task_name: Nom de la tâche ('emotion', 'sentiment', 'irony')
        label_column: Nom de la colonne contenant les labels
        train_df, val_df, test_df: DataFrames
        models_dir: Répertoire de sauvegarde des modèles
        results_dir: Répertoire de sauvegarde des résultats
    """
    print(f"\n{'='*70}")
    print(f"🎯 TÂCHE : {task_name.upper()}")
    print(f"{'='*70}\n")
    
    # Utiliser la colonne text_clean si disponible, sinon text
    text_column = 'text_clean' if 'text_clean' in train_df.columns else 'text'
    
    # Extraire les textes et labels
    X_train = train_df[text_column].fillna('').astype(str).tolist()
    y_train = train_df[label_column].fillna('').astype(str).tolist()
    
    X_val = val_df[text_column].fillna('').astype(str).tolist()
    y_val = val_df[label_column].fillna('').astype(str).tolist()
    
    X_test = test_df[text_column].fillna('').astype(str).tolist()
    y_test = test_df[label_column].fillna('').astype(str).tolist()
    
    print(f"📝 Textes et labels extraits")
    print(f"   Colonne texte : {text_column}")
    print(f"   Colonne label : {label_column}")
    print(f"   Classes : {sorted(set(y_train))}\n")
    
    # Créer et entraîner le modèle
    print(f"🔧 Entraînement du modèle baseline...")
    model = BaselineModel(
        task_name=task_name,
        max_features=5000,
        ngram_range=(1, 2),
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print(f"✅ Modèle entraîné\n")
    
    # Évaluer sur train, val et test
    all_results = {}
    
    for split_name, X, y in [('train', X_train, y_train), 
                              ('val', X_val, y_val), 
                              ('test', X_test, y_test)]:
        
        print(f"📊 Évaluation sur {split_name}...")
        results = evaluate_model(model, X, y, task_name, split_name)
        print_results_summary(results)
        
        # Sauvegarder les résultats
        results_path = results_dir / f'baseline_{task_name}_{split_name}.json'
        save_results(results, results_path)
        
        # Matrice de confusion
        confusion_path = results_dir / f'confusion_matrix_{task_name}_{split_name}.png'
        plot_confusion_matrix(y, results['predictions'], task_name, confusion_path)
        
        all_results[split_name] = results
    
    # Sauvegarder le modèle
    model.save(models_dir)
    
    # Analyse des features importantes
    print(f"\n🔍 Features les plus importantes par classe :\n")
    feature_importance = model.get_feature_importance(top_n=10)
    
    for class_name, features in feature_importance.items():
        print(f"  {class_name.upper()} :")
        for i, (feature, weight) in enumerate(features[:5], 1):
            print(f"    {i}. {feature:20s} (poids: {weight:.3f})")
        print()
    
    # Sauvegarder l'importance des features
    importance_path = results_dir / f'feature_importance_{task_name}.json'
    with open(importance_path, 'w', encoding='utf-8') as f:
        json.dump(feature_importance, f, indent=2, ensure_ascii=False)
    print(f"✅ Importance des features sauvegardée : {importance_path}\n")
    
    return all_results


def create_summary_table(all_task_results: dict, results_dir: Path):
    """
    Crée un tableau récapitulatif des résultats
    
    Args:
        all_task_results: {task_name: {split: results}}
        results_dir: Répertoire de sauvegarde
    """
    summary_data = []
    
    for task_name, task_results in all_task_results.items():
        for split_name, results in task_results.items():
            metrics = results['metrics_macro']
            summary_data.append({
                'Tâche': task_name,
                'Split': split_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1 (macro)': f"{metrics['f1_macro']:.4f}",
                'Precision (macro)': f"{metrics['precision_macro']:.4f}",
                'Recall (macro)': f"{metrics['recall_macro']:.4f}",
                'N': metrics['num_samples']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sauvegarder
    csv_path = results_dir / 'baseline_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"📋 RÉSUMÉ FINAL - BASELINE")
    print(f"{'='*70}\n")
    print(summary_df.to_string(index=False))
    print(f"\n✅ Résumé sauvegardé : {csv_path}\n")


def main():
    """
    Fonction principale
    """
    print("\n" + "="*70)
    print("🚀 ENTRAÎNEMENT DES MODÈLES BASELINE")
    print("="*70 + "\n")
    
    # Chemins
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / 'data' / 'processed'
    models_dir = project_dir / 'models'
    results_dir = project_dir / 'results' / 'baseline'
    
    # Créer les répertoires si nécessaire
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger les données
    train_df, val_df, test_df = load_data(data_dir, preprocessed=True)
    
    # Définir les tâches
    tasks = [
        ('emotion', 'emotion'),      # Classification d'émotions (7 classes)
        ('sentiment', 'sentiment'),  # Analyse de sentiment (3 classes)
        ('irony', 'is_ironic')       # Détection d'ironie (2 classes)
    ]
    
    # Entraîner et évaluer chaque tâche
    all_task_results = {}
    
    for task_name, label_column in tasks:
        task_results = train_and_evaluate_task(
            task_name=task_name,
            label_column=label_column,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            models_dir=models_dir,
            results_dir=results_dir
        )
        all_task_results[task_name] = task_results
    
    # Créer un tableau récapitulatif
    create_summary_table(all_task_results, results_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ ENTRAÎNEMENT TERMINÉ !")
    print(f"{'='*70}")
    print(f"📁 Modèles sauvegardés : {models_dir}")
    print(f"📊 Résultats sauvegardés : {results_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Fixer les seeds pour la reproductibilité
    np.random.seed(42)
    
    main()
