"""
MLflow utility functions for experiment tracking.

This module provides functions to manage MLflow experiments,
including setup, tracking, and cleanup operations.
"""

import os
import shutil
import mlflow
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score, 
                              average_precision_score, confusion_matrix, 
                              precision_recall_curve, PrecisionRecallDisplay)

def setup_mlflow_tracking(tracking_dir="/home/jupyter/smart_ads/notebooks/mlruns", 
                          experiment_name="smart-ads-baseline", 
                          clean_previous=True):
    """
    Configure MLflow tracking directory and experiment.
    
    Args:
        tracking_dir: Directory for MLflow tracking
        experiment_name: Name of the experiment to create
        clean_previous: Whether to remove existing tracking directory
        
    Returns:
        Experiment ID of the created experiment
    """
    trash_dir = os.path.join(tracking_dir, ".trash")
    
    # Clean directories if requested
    if clean_previous and os.path.exists(tracking_dir):
        print(f"Removendo diretório MLflow existente: {tracking_dir}")
        shutil.rmtree(tracking_dir, ignore_errors=True)
    
    # Create necessary directories
    os.makedirs(tracking_dir, exist_ok=True)
    os.makedirs(trash_dir, exist_ok=True)  # Create .trash directory explicitly
    
    # Configure MLflow
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    print(f"MLflow configurado para usar: {mlflow.get_tracking_uri()}")
    
    # Check if experiment exists and remove it
    client = mlflow.tracking.MlflowClient()
    try:
        existing_exp = client.get_experiment_by_name(experiment_name)
        if existing_exp:
            print(f"Removendo experimento existente: {experiment_name}")
            client.delete_experiment(existing_exp.experiment_id)
    except Exception as e:
        print(f"Erro ao verificar experimento existente: {e}")
    
    # Create a new experiment
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Criado novo experimento: {experiment_name} (ID: {experiment_id})")
    
    return experiment_id

def get_data_hash(df):
    """
    Generate a hash for a dataframe to track data versions.
    
    Args:
        df: Pandas DataFrame to hash
        
    Returns:
        MD5 hash string of the dataframe
    """
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

def setup_artifact_directory(dir_path="/tmp/mlflow_artifacts"):
    """
    Create directory for temporary MLflow artifacts.
    
    Args:
        dir_path: Path to create for artifacts
        
    Returns:
        Path to the created directory
    """
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        filename: File path to save the plot
        
    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não Converteu', 'Converteu'],
                yticklabels=['Não Converteu', 'Converteu'])
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return cm

def plot_prob_histogram(y_true, y_pred_proba, threshold, title, filename):
    """
    Plot and save histogram of prediction probabilities.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Decision threshold
        title: Plot title
        filename: File path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5, color='blue', label='Classe 0 (Não converteu)')
    plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5, color='red', label='Classe 1 (Converteu)')
    plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.title(title)
    plt.xlabel('Probabilidade prevista')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, title, filename):
    """
    Plot and save precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        filename: File path to save the plot
    """
    display = PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba, name="PR curve")
    _, ax = plt.subplots(figsize=(10, 6))
    display.plot(ax=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_learning_curve(model, X, y, model_name, filename=None):
    """
    Plot and save learning curve.
    
    Args:
        model: Trained model
        X: Features
        y: Target
        model_name: Name of the model for the plot title
        filename: File path to save the plot
        
    Returns:
        File path where the plot was saved
    """
    from sklearn.model_selection import learning_curve
    train_sizes = np.linspace(0.3, 1.0, 3)
    cv = 3
    print(f"  Gerando curva de aprendizado leve para {model_name}...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring='f1',
        n_jobs=-1, shuffle=True, random_state=42
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.grid(True, alpha=0.3)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Treino")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validação")
    plt.title(f"Curva de Aprendizado - {model_name}")
    plt.xlabel("Tamanho do conjunto de treino")
    plt.ylabel("F1-Score")
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()
    return filename

def plot_threshold_analysis(thresholds, f1_scores, precisions, recalls, best_threshold, model_name, filename):
    """
    Plot and save threshold analysis.
    
    Args:
        thresholds: Array of thresholds
        f1_scores: F1 scores at each threshold
        precisions: Precision values at each threshold
        recalls: Recall values at each threshold
        best_threshold: Best threshold found
        model_name: Name of the model
        filename: File path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Melhor threshold: {best_threshold:.2f}')
    plt.title(f'Efeito do threshold nas métricas - {model_name}')
    plt.xlabel('Threshold')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_feature_importance(feature_cols, importances, model_name, filename, top_n=20):
    """
    Plot and save feature importance.
    
    Args:
        feature_cols: List of feature names
        importances: Feature importance values
        model_name: Name of the model
        filename: File path to save the plot
        top_n: Number of top features to display
    """
    # Create DataFrame of importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Save importance table
    importance_df.to_csv(f"{filename.replace('.png', '.csv')}", index=False)
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top {top_n} Features - {model_name}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return importance_df

def find_optimal_threshold(y_true, y_pred_proba, threshold_range=None):
    """
    Find the optimal threshold for classification based on F1 score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold_range: Range of thresholds to test (default: 0.01 to 0.5 in 0.01 increments)
        
    Returns:
        Dictionary with best threshold, F1 score, precision, recall, and arrays of all tested values
    """
    if threshold_range is None:
        threshold_range = np.arange(0.01, 0.5, 0.01)
    
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    f1_scores = []
    precisions = []
    recalls = []
    
    for threshold in threshold_range:
        y_pred_t = (y_pred_proba >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_t, average='binary')
        
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    return {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'thresholds': threshold_range,
        'f1_scores': np.array(f1_scores),
        'precisions': np.array(precisions),
        'recalls': np.array(recalls)
    }