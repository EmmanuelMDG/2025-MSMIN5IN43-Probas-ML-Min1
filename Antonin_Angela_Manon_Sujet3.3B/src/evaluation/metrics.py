"""
Fonctions d'évaluation pour les modèles NLP

Ce module contient toutes les fonctions pour évaluer les performances
des modèles sur les 3 tâches : émotions, sentiment, ironie.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    task_name: str,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Calcule toutes les métriques pour une tâche
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        task_name: Nom de la tâche ('emotion', 'sentiment', 'irony')
        average: Type de moyenne ('macro', 'weighted', 'micro')
        
    Returns:
        metrics: Dictionnaire des métriques
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1-Score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    # F1-Score par classe (pour analyse détaillée)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'task': task_name,
        'accuracy': float(accuracy),
        f'precision_{average}': float(precision),
        f'recall_{average}': float(recall),
        f'f1_{average}': float(f1),
        'num_samples': len(y_true)
    }
    
    return metrics


def get_classification_report(
    y_true: List[str],
    y_pred: List[str],
    task_name: str,
    output_dict: bool = True
) -> Dict:
    """
    Génère un rapport de classification détaillé
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        task_name: Nom de la tâche
        output_dict: Si True, retourne un dictionnaire, sinon une string
        
    Returns:
        report: Rapport de classification
    """
    report = classification_report(
        y_true, y_pred,
        output_dict=output_dict,
        zero_division=0
    )
    
    if output_dict:
        report['task'] = task_name
    
    return report


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    task_name: str,
    save_path: Path,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Génère et sauvegarde une matrice de confusion
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        task_name: Nom de la tâche
        save_path: Chemin de sauvegarde
        figsize: Taille de la figure
    """
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Récupérer les labels uniques (triés)
    labels = sorted(set(y_true))
    
    # Créer la figure
    plt.figure(figsize=figsize)
    
    # Heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Nombre de prédictions'}
    )
    
    plt.title(f'Matrice de Confusion - {task_name.capitalize()}', fontsize=14, fontweight='bold')
    plt.xlabel('Prédictions', fontsize=12)
    plt.ylabel('Réalité', fontsize=12)
    plt.tight_layout()
    
    # Sauvegarder
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Matrice de confusion sauvegardée : {save_path}")


def evaluate_model(
    model,
    texts: List[str],
    labels: List[str],
    task_name: str,
    split_name: str = 'test'
) -> Dict:
    """
    Évalue un modèle sur un dataset
    
    Args:
        model: Modèle à évaluer (doit avoir une méthode predict())
        texts: Liste de textes
        labels: Labels réels
        task_name: Nom de la tâche
        split_name: Nom du split ('train', 'val', 'test')
        
    Returns:
        results: Dictionnaire des résultats
    """
    # Prédictions
    predictions = model.predict(texts)
    
    # Métriques macro (importante pour classes déséquilibrées)
    metrics_macro = compute_metrics(labels, predictions, task_name, average='macro')
    
    # Métriques weighted (tient compte de la taille des classes)
    metrics_weighted = compute_metrics(labels, predictions, task_name, average='weighted')
    
    # Rapport détaillé
    report = get_classification_report(labels, predictions, task_name)
    
    results = {
        'split': split_name,
        'task': task_name,
        'metrics_macro': metrics_macro,
        'metrics_weighted': metrics_weighted,
        'classification_report': report,
        'predictions': predictions
    }
    
    return results


def save_results(
    results: Dict,
    save_path: Path
) -> None:
    """
    Sauvegarde les résultats au format JSON
    
    Args:
        results: Dictionnaire des résultats
        save_path: Chemin de sauvegarde
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Résultats sauvegardés : {save_path}")


def compare_models(
    results_dict: Dict[str, Dict],
    save_path: Path
) -> pd.DataFrame:
    """
    Compare les performances de plusieurs modèles
    
    Args:
        results_dict: {model_name: results}
        save_path: Chemin de sauvegarde du tableau
        
    Returns:
        comparison_df: DataFrame de comparaison
    """
    comparison_data = []
    
    for model_name, results in results_dict.items():
        for task, task_results in results.items():
            metrics = task_results.get('metrics_macro', {})
            
            comparison_data.append({
                'Model': model_name,
                'Task': task,
                'Accuracy': metrics.get('accuracy', 0),
                'F1 (macro)': metrics.get('f1_macro', 0),
                'Precision (macro)': metrics.get('precision_macro', 0),
                'Recall (macro)': metrics.get('recall_macro', 0)
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sauvegarder
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(save_path, index=False)
    
    print(f"✅ Comparaison sauvegardée : {save_path}")
    
    return comparison_df


def print_results_summary(results: Dict) -> None:
    """
    Affiche un résumé des résultats
    
    Args:
        results: Dictionnaire des résultats
    """
    task = results['task']
    split = results['split']
    metrics = results['metrics_macro']
    
    print(f"\n{'='*60}")
    print(f"📊 RÉSULTATS - {task.upper()} ({split})")
    print(f"{'='*60}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  F1-Score (macro): {metrics['f1_macro']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro)  : {metrics['recall_macro']:.4f}")
    print(f"  Échantillons   : {metrics['num_samples']}")
    print(f"{'='*60}\n")
