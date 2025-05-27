"""
Baseline model training and evaluation module.

This module provides functions to train and evaluate baseline models
for binary classification, including RandomForest, LightGBM, and XGBoost.
"""

import pandas as pd
import shutil
import numpy as np
import os
import time
import re
import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM não está instalado. Alguns modelos não estarão disponíveis.")
try:
    import xgboost as xgb
except ImportError:
    print("XGBoost não está instalado. Alguns modelos não estarão disponíveis.")

def sanitize_column_names(df):
    """
    Sanitize column names to avoid issues with special characters.
    
    Args:
        df: Pandas DataFrame with columns to sanitize
        
    Returns:
        Dictionary mapping original column names to sanitized names
    """
    sanitized_columns = {}
    for col in df.columns:
        new_col = re.sub(r'[^\w\s]', '_', col)
        new_col = re.sub(r'\s+', '_', new_col)
        if new_col in sanitized_columns.values():
            new_col = f"{new_col}_{df.columns.get_loc(col)}"
        sanitized_columns[col] = new_col
    df.rename(columns=sanitized_columns, inplace=True)
    return sanitized_columns

def convert_integer_columns_to_float(train_df, val_df=None):
    """
    Convert integer columns to float for compatibility with some models.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional)
        
    Returns:
        List of column names that were converted
    """
    print("Convertendo colunas inteiras para float...")
    integer_columns = []
    
    for col in train_df.columns:
        if pd.api.types.is_integer_dtype(train_df[col].dtype):
            train_df[col] = train_df[col].astype(float)
            if val_df is not None and col in val_df.columns:
                val_df[col] = val_df[col].astype(float)
            integer_columns.append(col)
    
    return integer_columns

def prepare_data_for_training(train_path, val_path=None):
    """
    Load and prepare data for training a baseline model.
    
    Args:
        train_path: Path to training dataset
        val_path: Path to validation dataset (optional)
        
    Returns:
        Dictionary with prepared data components
    """
    print("Carregando datasets...")
    
    # Load training data
    train_df = pd.read_csv(train_path)
    
    # Load validation data if provided
    if val_path:
        val_df = pd.read_csv(val_path)
    else:
        # If no validation set provided, we'll split the training data later
        val_df = None
    
    # Sanitize column names
    column_mapping = sanitize_column_names(train_df)
    if val_df is not None:
        sanitize_column_names(val_df)
    
    # Identify target column
    target_col = 'target' if 'target' in train_df.columns else column_mapping.get('target', 'target')
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Convert integer columns to float
    integer_columns = convert_integer_columns_to_float(train_df, val_df)
    
    # Create X and y variables
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    
    if val_df is not None:
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col].copy()
    else:
        X_val = None
        y_val = None
    
    # Double-check for any remaining integer columns
    int_cols_remaining = [col for col in X_train.columns if pd.api.types.is_integer_dtype(X_train[col].dtype)]
    if int_cols_remaining:
        for col in int_cols_remaining:
            X_train[col] = X_train[col].astype(float)
            if X_val is not None:
                X_val[col] = X_val[col].astype(float)
    
    print(f"Dados carregados - treino: {X_train.shape}")
    if X_val is not None:
        print(f"validação: {X_val.shape}")
    print(f"Taxa de conversão - treino: {y_train.mean():.4f}")
    if y_val is not None:
        print(f"validação: {y_val.mean():.4f}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'integer_columns': integer_columns,
        'column_mapping': column_mapping,
        'train_df': train_df,
        'val_df': val_df
    }

def get_baseline_models(class_weight_rf='balanced', scale_pos_weight=50):
    """
    Create baseline model instances with appropriate weights for imbalanced data.
    
    Args:
        class_weight_rf: Class weight strategy for RandomForest
        scale_pos_weight: Positive class weight multiplier for boosting models
        
    Returns:
        Dictionary of initialized model instances
    """
    models = {
        'random_forest': RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight=class_weight_rf
        )
    }
    
    # Add LightGBM if available
    if 'lgb' in globals():
        models['lightgbm'] = lgb.LGBMClassifier(
            random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight
        )
    
    # Add XGBoost if available
    if 'xgb' in globals():
        models['xgboost'] = xgb.XGBClassifier(
            random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight
        )
    
    return models

def train_and_evaluate_model(model, name, X_train, y_train, X_val, y_val, 
                             experiment_id, artifact_dir, feature_cols,
                             train_data_hash, val_data_hash, integer_columns,
                             generate_learning_curves=False):
    """
    Train and evaluate a baseline model with MLflow tracking.
    
    Args:
        model: Model instance to train
        name: Model name
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        experiment_id: MLflow experiment ID
        artifact_dir: Directory for artifacts
        feature_cols: List of feature column names
        train_data_hash: Hash of training data
        val_data_hash: Hash of validation data
        integer_columns: List of integer columns converted to float
        generate_learning_curves: Whether to generate learning curves
        
    Returns:
        Dictionary of results
    """
    import os
    import time
    import shutil
    
    from src.utils.mlflow_utils import (
        plot_confusion_matrix, plot_prob_histogram, plot_precision_recall_curve,
        plot_learning_curve, plot_threshold_analysis, plot_feature_importance,
        find_optimal_threshold
    )
    
    print(f"Treinando {name}...")
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{name}_baseline") as run:
        run_id = run.info.run_id
        
        # Criar um diretório temporário específico para este run
        temp_artifact_dir = os.path.join(artifact_dir, run_id)
        os.makedirs(temp_artifact_dir, exist_ok=True)
        
        try:
            # Register tags
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("experiment_type", "baseline")
            mlflow.set_tag("class_balance", "weighted")
            mlflow.set_tag("train_data_hash", train_data_hash)
            mlflow.set_tag("val_data_hash", val_data_hash)
            mlflow.set_tag("feature_count", len(feature_cols))
            mlflow.set_tag("dataset_size", len(X_train))
            mlflow.set_tag("positive_ratio", float(y_train.mean()))
            mlflow.set_tag("converted_int_cols", ','.join(integer_columns))
            
            # Log model parameters
            for key, value in model.get_params().items():
                mlflow.log_param(key, value)
            
            # Train model with time measurement
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", train_time)
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Find optimal threshold
            threshold_results = find_optimal_threshold(y_val, y_pred_proba)
            best_threshold = threshold_results['best_threshold']
            best_f1 = threshold_results['best_f1']
            best_precision = threshold_results['best_precision']
            best_recall = threshold_results['best_recall']
            
            # Make final predictions with best threshold
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc_score = roc_auc_score(y_val, y_pred_proba)
            pr_auc = average_precision_score(y_val, y_pred_proba)
            
            # Count positive predictions
            positive_count = y_pred.sum()
            positive_pct = positive_count / len(y_pred) * 100
            
            # Log metrics
            mlflow.log_metric("precision", best_precision)
            mlflow.log_metric("recall", best_recall)
            mlflow.log_metric("f1", best_f1)
            mlflow.log_metric("threshold", best_threshold)
            mlflow.log_metric("auc", auc_score)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("positive_predictions", positive_count)
            mlflow.log_metric("positive_pct", positive_pct)
            
            # Create and log visualizations
            
            # Threshold analysis plot
            threshold_fig_path = os.path.join(temp_artifact_dir, f"threshold_plot_{name}.png")
            plot_threshold_analysis(
                threshold_results['thresholds'], 
                threshold_results['f1_scores'], 
                threshold_results['precisions'], 
                threshold_results['recalls'],
                best_threshold, name, threshold_fig_path
            )
            mlflow.log_artifact(threshold_fig_path)
            
            # Confusion matrix
            cm_fig_path = os.path.join(temp_artifact_dir, f"confusion_matrix_{name}.png")
            plot_confusion_matrix(
                y_val, y_pred, 
                f'Matriz de Confusão - {name} (threshold={best_threshold:.2f})', 
                cm_fig_path
            )
            mlflow.log_artifact(cm_fig_path)
            
            # Probability histogram
            hist_fig_path = os.path.join(temp_artifact_dir, f"prob_histogram_{name}.png")
            plot_prob_histogram(
                y_val, y_pred_proba, best_threshold, 
                f'Distribuição de Probabilidades - {name}', 
                hist_fig_path
            )
            mlflow.log_artifact(hist_fig_path)
            
            # Precision-recall curve
            pr_curve_path = os.path.join(temp_artifact_dir, f"pr_curve_{name}.png")
            plot_precision_recall_curve(
                y_val, y_pred_proba, 
                f'Curva Precision-Recall - {name}', 
                pr_curve_path
            )
            mlflow.log_artifact(pr_curve_path)
            
            # Generate learning curve if requested
            if generate_learning_curves:
                learning_curve_path = os.path.join(temp_artifact_dir, f"learning_curve_{name}.png")
                plot_learning_curve(
                    model, X_train, y_train, 
                    model_name=name.capitalize(), 
                    filename=learning_curve_path
                )
                mlflow.log_artifact(learning_curve_path)
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_fig_path = os.path.join(temp_artifact_dir, f"feature_importance_plot_{name}.png")
                importance_csv_path = os.path.join(temp_artifact_dir, f"feature_importance_{name}.csv")
                importance_df = plot_feature_importance(
                    feature_cols, model.feature_importances_,
                    name, importance_fig_path, top_n=20
                )
                importance_df.to_csv(importance_csv_path, index=False)
                mlflow.log_artifact(importance_fig_path)
                mlflow.log_artifact(importance_csv_path)
            
            # Log model
            input_example = X_train.iloc[:5].copy()
            
            if name == 'random_forest':
                mlflow.sklearn.log_model(model, name, input_example=input_example)
            elif name == 'lightgbm':
                mlflow.lightgbm.log_model(model, name, input_example=input_example)
            elif name == 'xgboost':
                mlflow.xgboost.log_model(model, name, input_example=input_example)
            
            model_uri = f"runs:/{run.info.run_id}/{name}"
            
            print(f"  Modelo {name} salvo em: {run.info.artifact_uri}/{name}")
            print(f"  URI do modelo para carregamento: {model_uri}")
            
            # Prepare results
            results = {
                "precision": float(best_precision),
                "recall": float(best_recall),
                "f1": float(best_f1),
                "threshold": float(best_threshold),
                "auc": float(auc_score),
                "pr_auc": float(pr_auc),
                "positive_count": int(positive_count),
                "positive_pct": float(positive_pct),
                "model_uri": model_uri,  # ÚNICA MODIFICAÇÃO: Incluir URI do modelo nos resultados
                "train_time": train_time
            }
            
            print(f"  {name} - F1: {best_f1:.4f}, PR-AUC: {pr_auc:.4f}, Threshold: {best_threshold:.4f}")
            print(f"  {name} - Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
            print(f"  {name} - Predições positivas: {positive_count} ({positive_pct:.2f}%)")
            
            return results
            
        except Exception as e:
            print(f"Erro durante treinamento de {name}: {e}")
            # Se houver erro, imprimir stack trace e re-levantar a exceção
            import traceback
            traceback.print_exc()
            raise e
            
        finally:
            # Limpar artefatos temporários após o upload para o MLflow
            try:
                if os.path.exists(temp_artifact_dir):
                    shutil.rmtree(temp_artifact_dir)
            except Exception as cleanup_error:
                print(f"Aviso: Erro ao limpar diretório temporário: {cleanup_error}")

def run_baseline_model_training(
    train_path,
    val_path,
    experiment_id,
    artifact_dir="/tmp/mlflow_artifacts",
    generate_learning_curves=False
):
    """
    Run the full baseline model training pipeline.
    """
    from src.utils.mlflow_utils import get_data_hash
    
    # NÃO configurar novamente o tracking URI aqui
    # NÃO chamar setup_mlflow_tracking aqui
    
    # Prepare data
    data = prepare_data_for_training(train_path, val_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    feature_cols = data['feature_cols']
    integer_columns = data['integer_columns']
    
    # Calculate data hashes for tracking
    train_hash = get_data_hash(data['train_df'])
    val_hash = get_data_hash(data['val_df'])
    
    # Create artifact directory if it doesn't exist
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Get baseline models
    models = get_baseline_models()
    
    # Train and evaluate each model
    print("\nTreinando modelos base com pesos balanceados...")
    all_results = {}
    
    for name, model in models.items():
        results = train_and_evaluate_model(
            model, name, 
            X_train, y_train, 
            X_val, y_val,
            experiment_id, artifact_dir,
            feature_cols, train_hash, val_hash,
            integer_columns, generate_learning_curves
        )
        
        # Store results
        if results:  # Verificar se results não é None
            for key, value in results.items():
                all_results[f"{name}_{key}"] = value
    
    # Print final results summary
    print("\nResultados dos modelos com threshold otimizado:")
    for model_name in models.keys():
        if f"{model_name}_f1" in all_results:
            print(f"\n{model_name.upper()}:")
            print(f"  F1: {all_results[f'{model_name}_f1']:.4f}")
            print(f"  Precisão: {all_results[f'{model_name}_precision']:.4f}")
            print(f"  Recall: {all_results[f'{model_name}_recall']:.4f}")
            print(f"  Threshold: {all_results[f'{model_name}_threshold']:.4f}")
            print(f"  AUC: {all_results[f'{model_name}_auc']:.4f}")
            print(f"  PR-AUC: {all_results[f'{model_name}_pr_auc']:.4f}")
            print(f"  Model URI: {all_results[f'{model_name}_model_uri']}")
    
    print("\nModelos treinados e registrados no MLflow com rastreabilidade.")
    
    return all_results

def get_latest_random_forest_run(mlflow_dir):
    """
    Obtém o run_id do modelo RandomForest mais recente.
    
    Args:
        mlflow_dir: Diretório do MLflow tracking
        
    Returns:
        Tuple com (run_id, threshold, model_uri) ou (None, None, None) se não encontrado
    """
    import os
    import mlflow
    
    # Configurar MLflow
    if os.path.exists(mlflow_dir):
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    else:
        print(f"AVISO: Diretório MLflow não encontrado: {mlflow_dir}")
        return None, None, None
    
    # Inicializar cliente MLflow
    client = mlflow.tracking.MlflowClient()
    
    # Procurar todos os experimentos
    experiments = client.search_experiments()
    
    for experiment in experiments:
        print(f"Verificando experimento: {experiment.name} (ID: {experiment.experiment_id})")
        
        # Buscar runs específicos do RandomForest ordenados pelo mais recente
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_type = 'random_forest'",
            order_by=["attribute.start_time DESC"]
        )
        
        if not runs:
            # Se não achou pela tag, procurar pelo nome do artefato
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"]
            )
        
        for run in runs:
            run_id = run.info.run_id
            print(f"  Encontrado run: {run_id}")
            
            # Verificar artefatos
            artifacts = client.list_artifacts(run_id)
            rf_artifact = None
            
            for artifact in artifacts:
                if artifact.is_dir and artifact.path == 'random_forest':
                    rf_artifact = artifact
                    break
            
            if rf_artifact:
                # Extrair o threshold das métricas
                threshold = run.data.metrics.get('threshold', 0.17)  # Fallback para 0.17 se não encontrar
                model_uri = f"runs:/{run_id}/random_forest"
                
                print(f"  Usando modelo RandomForest de {run.info.start_time}")
                print(f"  Run ID: {run_id}")
                print(f"  Model URI: {model_uri}")
                print(f"  Threshold: {threshold}")
                
                # Mostrar métricas registradas no MLflow
                precision = run.data.metrics.get('precision', None)
                recall = run.data.metrics.get('recall', None)
                f1 = run.data.metrics.get('f1', None)
                
                if precision and recall and f1:
                    print(f"  Métricas do MLflow: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                
                return run_id, threshold, model_uri
    
    print("Nenhum modelo RandomForest encontrado em MLflow.")
    return None, None, None