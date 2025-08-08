#!/usr/bin/env python
"""
Script para realizar o hyperparameter tuning do modelo RandomForest utilizando
o modelo mais recente disponível no MLflow.

Este script realiza:
1. Weight adjustments for imbalanced classes
2. GridSearchCV para otimização de hiperparâmetros
3. Calibração de probabilidades usando validação cruzada
4. Ajuste de threshold com diferentes métricas

O melhor modelo é salvo no MLflow.
"""

import os
import sys
import argparse
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            fbeta_score, precision_recall_curve, 
                            average_precision_score, roc_auc_score,
                            confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Importar funções necessárias dos módulos do projeto
from src.evaluation.mlflow_utils import (
    setup_mlflow_tracking,
    get_data_hash,
    setup_artifact_directory,
    plot_confusion_matrix, 
    plot_prob_histogram, 
    plot_precision_recall_curve,
    plot_threshold_analysis,
    plot_feature_importance,
    find_optimal_threshold
)

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

def get_best_model_run(mlflow_dir, metric='precision', min_value=0.5):
    """
    Obtém o run_id do melhor modelo RandomForest com base em uma métrica específica.
    
    Args:
        mlflow_dir: Diretório do MLflow tracking
        metric: Métrica para selecionar o melhor modelo ('precision', 'recall', 'f1')
        min_value: Valor mínimo da métrica para considerar um modelo
        
    Returns:
        Tuple com (run_id, threshold, model_uri) ou (None, None, None) se não encontrado
    """
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
    
    # Variáveis para armazenar o melhor modelo
    best_run_id = None
    best_metric_value = min_value
    best_threshold = None
    
    print(f"Procurando pelo melhor modelo com base na métrica: {metric} (valor mínimo: {min_value})")
    
    for experiment in experiments:
        print(f"Verificando experimento: {experiment.name} (ID: {experiment.experiment_id})")
        
        # Buscar runs do RandomForest
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.model_type = 'random_forest' and metrics.{metric} > {min_value}",
            order_by=[f"metrics.{metric} DESC"]  # Ordenar do melhor para o pior
        )
        
        if not runs:
            # Verificar todos os runs (pode haver modelos sem tags adequadas)
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"]
            )
        
        for run in runs:
            run_id = run.info.run_id
            
            # Verificar se tem métrica registrada
            metric_value = run.data.metrics.get(metric)
            
            if metric_value is None:
                print(f"  Run {run_id}: Sem métrica {metric} registrada")
                continue
                
            print(f"  Run {run_id}: {metric}={metric_value:.4f}")
            
            # Verificar artefatos
            artifacts = client.list_artifacts(run_id)
            has_rf_artifact = False
            
            for artifact in artifacts:
                if artifact.is_dir and (artifact.path == 'random_forest' or artifact.path == 'optimized_random_forest'):
                    has_rf_artifact = True
                    break
            
            if not has_rf_artifact:
                print(f"  Run {run_id}: Sem artefato de modelo RandomForest")
                continue
            
            # Verificar se é melhor que o atual
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_run_id = run_id
                best_threshold = run.data.metrics.get('threshold', 0.5)
                
                # Determinar o caminho do artefato
                if 'random_forest' in [a.path for a in artifacts if a.is_dir]:
                    model_path = 'random_forest'
                else:
                    model_path = 'optimized_random_forest'
                
                print(f"  Novo melhor modelo encontrado: {run_id}")
                print(f"  {metric}: {metric_value:.4f}, threshold: {best_threshold}")
    
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/{model_path}"
        print(f"\nMelhor modelo encontrado:")
        print(f"  Run ID: {best_run_id}")
        print(f"  Model URI: {model_uri}")
        print(f"  {metric}: {best_metric_value:.4f}")
        print(f"  Threshold: {best_threshold}")
        return best_run_id, best_threshold, model_uri
    
    print(f"Nenhum modelo com {metric} > {min_value} encontrado.")
    return None, None, None

def load_model_from_mlflow(model_uri):
    """
    Carrega um modelo a partir do MLflow usando seu URI.
    
    Args:
        model_uri: URI do modelo no formato 'runs:/<run_id>/<artifact_path>'
        
    Returns:
        Modelo carregado ou None se falhar
    """
    try:
        print(f"Carregando modelo de: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        return None

def load_datasets(train_path=None, val_path=None):
    """
    Carrega os datasets de treino e validação.
    
    Args:
        train_path: Caminho para o dataset de treino
        val_path: Caminho para o dataset de validação
        
    Returns:
        Tuple (train_df, val_df)
    """
    # Definir caminhos padrão se não fornecidos
    if not train_path:
        train_path = os.path.join(project_root, "data", "03_feature_selection_text_code6", "train.csv")
        if not os.path.exists(train_path):
            train_path = os.path.join(os.path.expanduser("~"), "desktop", "smart_ads", "data", "03_feature_selection_text_code6", "train.csv")
    
    if not val_path:
        val_path = os.path.join(project_root, "data", "03_feature_selection_text_code6", "validation.csv")
        if not os.path.exists(val_path):
            val_path = os.path.join(os.path.expanduser("~"), "desktop", "smart_ads", "data", "03_feature_selection_text_code6", "validation.csv")
    
    # Carregar os dados
    print(f"Carregando dados de treino: {train_path}")
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    
    print(f"Carregando dados de validação: {val_path}")
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    
    if train_df is None:
        train_path = input("Arquivo de treino não encontrado. Por favor, forneça o caminho completo: ")
        train_df = pd.read_csv(train_path)
    
    if val_df is None:
        val_path = input("Arquivo de validação não encontrado. Por favor, forneça o caminho completo: ")
        val_df = pd.read_csv(val_path)
    
    print(f"Dados carregados: treino {train_df.shape}, validação {val_df.shape}")
    return train_df, val_df

def prepare_data_for_model(model, df, target_col="target"):
    """
    Prepara os dados para serem compatíveis com o modelo.
    
    Args:
        model: Modelo treinado
        df: DataFrame a ser preparado
        target_col: Nome da coluna target
        
    Returns:
        X, y preparados para o modelo
    """
    # Extrair features e target
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    y = df[target_col] if target_col in df.columns else None
    
    # Aplicar a mesma sanitização de nomes de colunas usada no treinamento
    col_mapping = sanitize_column_names(X)
    
    # Converter inteiros para float
    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col].dtype):
            X.loc[:, col] = X[col].astype(float)
    
    # Verificar features do modelo
    if hasattr(model, 'feature_names_in_'):
        expected_features = set(model.feature_names_in_)
        actual_features = set(X.columns)
        
        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features
        
        if missing_features:
            print(f"AVISO: Faltam {len(missing_features)} features que o modelo espera")
            print(f"  Exemplos: {list(missing_features)[:5]}")
            
            # Criar DataFrame vazio com as colunas corretas para minimizar a fragmentação
            missing_cols = list(missing_features)
            missing_dict = {col: [0] * len(X) for col in missing_cols}
            missing_df = pd.DataFrame(missing_dict)
            
            # Concatenar em vez de adicionar uma por uma
            X = pd.concat([X, missing_df], axis=1)
        
        if extra_features:
            print(f"AVISO: Removendo {len(extra_features)} features extras")
            print(f"  Exemplos: {list(extra_features)[:5]}")
            X = X.drop(columns=list(extra_features))
        
        # Garantir a ordem correta das colunas
        X = X[model.feature_names_in_]
    
    return X, y

def plot_calibration_curve(y_val, probs_list, model_names, output_path):
    """
    Plota curva de calibração para diferentes modelos.
    
    Args:
        y_val: Array com os valores verdadeiros
        probs_list: Lista com probabilidades de cada modelo
        model_names: Lista com os nomes dos modelos
        output_path: Caminho para salvar o gráfico
    """
    plt.figure(figsize=(10, 10))
    
    from sklearn.calibration import calibration_curve
    
    # Gerar curvas de calibração
    for i, (probs, name) in enumerate(zip(probs_list, model_names)):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_val, probs, n_bins=10
        )
        
        plt.plot(mean_predicted_value, fraction_of_positives, 's-',
                label=name)
        
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    plt.ylabel('Fraction of positives')
    plt.xlabel('Mean predicted probability')
    plt.title('Calibration curves')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def optimize_threshold_for_metric(y_true, y_pred_proba, metric='f1', beta=1.0):
    """
    Encontra o threshold ótimo para uma métrica específica.
    
    Args:
        y_true: Valores verdadeiros
        y_pred_proba: Probabilidades preditas
        metric: Métrica a ser otimizada ('f1', 'f2', 'precision', 'recall')
        beta: Valor de beta para F-beta score
        
    Returns:
        Dicionário com o threshold ótimo e o valor da métrica correspondente
    """
    # Definir thresholds para testar
    thresholds = np.arange(0.01, 0.99, 0.01)
    
    # Inicializar variáveis para armazenar os melhores resultados
    best_metric_value = 0
    best_threshold = 0.5
    precisions = []
    recalls = []
    metric_values = []
    
    # Testar cada threshold
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calcular precision e recall
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # Armazenar para plotagem
        precisions.append(precision)
        recalls.append(recall)
        
        # Calcular a métrica desejada
        if metric == 'f1':
            metric_value = f1_score(y_true, y_pred)
        elif metric == 'f2':
            metric_value = fbeta_score(y_true, y_pred, beta=2.0)
        elif metric == 'fbeta':
            metric_value = fbeta_score(y_true, y_pred, beta=beta)
        elif metric == 'precision':
            metric_value = precision
        elif metric == 'recall':
            metric_value = recall
        elif metric == 'precision_at_90_recall':
            metric_value = precision if recall >= 0.9 else 0
        elif metric == 'recall_at_90_precision':
            metric_value = recall if precision >= 0.9 else 0
        else:
            metric_value = f1_score(y_true, y_pred)
            
        metric_values.append(metric_value)
        
        # Atualizar o melhor threshold se a métrica melhorou
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
    
    return {
        'best_threshold': best_threshold,
        'best_metric_value': best_metric_value,
        'thresholds': thresholds,
        'precisions': np.array(precisions),
        'recalls': np.array(recalls),
        'metric_values': np.array(metric_values)
    }

def run_hyperparameter_tuning_pipeline(
    train_df, 
    val_df, 
    base_model, 
    target_col="target", 
    output_dir=None, 
    experiment_id=None,
    artifact_dir="/tmp/mlflow_artifacts"
):
    """
    Executa o pipeline completo de hyperparameter tuning.
    
    Args:
        train_df: DataFrame de treino
        val_df: DataFrame de validação
        base_model: Modelo base a ser otimizado
        target_col: Nome da coluna target
        output_dir: Diretório para salvar resultados
        experiment_id: ID do experimento MLflow
        artifact_dir: Diretório para artefatos temporários
        
    Returns:
        Dicionário com o melhor modelo e resultados
    """
    # Criar diretório para resultados
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "reports", f"hyperparameter_tuning_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {output_dir}")
    
    # Preparar dados
    X_train, y_train = prepare_data_for_model(base_model, train_df, target_col)
    X_val, y_val = prepare_data_for_model(base_model, val_df, target_col)
    
    if y_train is None or y_val is None:
        raise ValueError(f"Coluna target '{target_col}' não encontrada no DataFrame")
    
    # Calcular a distribuição de classes para ajustar os pesos
    class_distribution = y_train.value_counts(normalize=True)
    
    print(f"Distribuição de classes (treino):")
    for cls, pct in class_distribution.items():
        cls_count = int(pct * len(y_train))
        print(f"  Classe {cls}: {pct:.4f} ({cls_count} amostras)")
    
    # Calcular weight para a classe minoritária
    if class_distribution.get(1, 0) < class_distribution.get(0, 1):
        minority_class = 1
        class_weight_ratio = class_distribution[0] / class_distribution[1]
    else:
        minority_class = 0
        class_weight_ratio = class_distribution[1] / class_distribution[0]
    
    print(f"Classe minoritária: {minority_class}, ratio de peso: {class_weight_ratio:.2f}")
    
    # Calcular hash dos dados para tracking
    train_data_hash = get_data_hash(train_df)
    val_data_hash = get_data_hash(val_df)
    
    # 1. Definir os hiperparâmetros a serem testados
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 10}, {0: 1, 1: 50}, {0: 1, 1: 100}]
    }
    
    # Start MLflow run for the hyperparameter tuning
    with mlflow.start_run(experiment_id=experiment_id, run_name="rf_hyperparameter_tuning") as run:
        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")
        
        # Register tags
        mlflow.set_tags({
            "model_type": "random_forest",
            "experiment_type": "hyperparameter_tuning",
            "train_data_hash": train_data_hash,
            "val_data_hash": val_data_hash,
            "feature_count": len(X_train.columns),
            "dataset_size": len(X_train),
            "positive_ratio_train": float(y_train.mean()),
            "positive_ratio_val": float(y_val.mean())
        })
        
        # 2. Realizar GridSearchCV
        print("\n1. Iniciando GridSearchCV para otimização de hiperparâmetros...")
        
        # Configurar GridSearchCV com estratificação e F2-score para lidar com o desbalanceamento
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Usar F2-score para dar mais peso ao recall
        scorer = make_scorer(fbeta_score, beta=2)
        
        # Configurar e executar GridSearch
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            scoring=scorer,  # Otimizar para F2-score
            cv=cv,
            n_jobs=-1,
            verbose=2,
            return_train_score=True,
            refit=True  # Refit com a melhor combinação
        )
        
        # Treinar o grid search
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        grid_search_time = time.time() - start_time
        
        print(f"\nGridSearchCV concluído em {grid_search_time:.2f} segundos")
        print(f"Melhor score: {grid_search.best_score_:.4f}")
        print(f"Melhores parâmetros: {grid_search.best_params_}")
        
        # Salvar os resultados do grid search
        grid_results = pd.DataFrame(grid_search.cv_results_)
        grid_results.to_csv(os.path.join(output_dir, "grid_search_results.csv"), index=False)
        
        # Log dos resultados no MLflow
        mlflow.log_param("best_params", str(grid_search.best_params_))
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        mlflow.log_metric("grid_search_time", grid_search_time)
        
        # 3. Calibrar as probabilidades do melhor modelo
        print("\n2. Calibrando probabilidades do melhor modelo...")
        
        best_model = grid_search.best_estimator_
        
        # Criar calibradores com diferentes métodos
        calibrators = {
            'uncalibrated': best_model,
            'isotonic': CalibratedClassifierCV(
                best_model, method='isotonic', cv=3, n_jobs=-1
            ),
            'sigmoid': CalibratedClassifierCV(
                best_model, method='sigmoid', cv=3, n_jobs=-1
            )
        }
        
        # Treinar calibradores
        calibrated_models = {}
        calibration_times = {}
        calibration_probs = {}
        
        for name, calibrator in calibrators.items():
            if name == 'uncalibrated':
                calibrated_models[name] = best_model
                calibration_probs[name] = best_model.predict_proba(X_val)[:, 1]
                calibration_times[name] = 0
            else:
                start_time = time.time()
                print(f"Treinando calibrador {name}...")
                calibrator.fit(X_train, y_train)
                calibration_time = time.time() - start_time
                calibration_times[name] = calibration_time
                
                calibrated_models[name] = calibrator
                calibration_probs[name] = calibrator.predict_proba(X_val)[:, 1]
                
                print(f"Calibrador {name} treinado em {calibration_time:.2f} segundos")
        
        # Plotar curvas de calibração
        calibration_fig_path = os.path.join(output_dir, "calibration_curves.png")
        plot_calibration_curve(
            y_val,
            [calibration_probs[name] for name in calibrators.keys()],
            list(calibrators.keys()),
            calibration_fig_path
        )
        mlflow.log_artifact(calibration_fig_path)
        
        # 4. Ajuste de threshold com diferentes métricas
        print("\n3. Ajustando thresholds para diferentes métricas...")
        
        # Definir métricas para otimização de threshold
        metrics = ['f1', 'f2', 'precision', 'recall']
        
        # Armazenar resultados para cada calibrador e métrica
        threshold_results = {}
        all_metrics = {}
        
        for name, probs in calibration_probs.items():
            threshold_results[name] = {}
            all_metrics[name] = {}
            
            for metric in metrics:
                print(f"Otimizando threshold para {name} com métrica {metric}...")
                
                results = optimize_threshold_for_metric(y_val, probs, metric=metric)
                threshold_results[name][metric] = results
                
                # Usar o threshold otimizado para calcular as métricas finais
                threshold = results['best_threshold']
                y_pred = (probs >= threshold).astype(int)
                
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                f2 = fbeta_score(y_val, y_pred, beta=2)
                
                all_metrics[name][metric] = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'f2': f2
                }
                
                print(f"  Threshold ótimo ({metric}): {threshold:.4f}")
                print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
                print(f"  F1: {f1:.4f}, F2: {f2:.4f}")
                
                # Plotar análise de threshold
                threshold_fig_path = os.path.join(output_dir, f"threshold_{name}_{metric}.png")
                plot_threshold_analysis(
                    results['thresholds'],
                    results['metric_values'],
                    results['precisions'],
                    results['recalls'],
                    results['best_threshold'],
                    f"{name}_{metric}",
                    threshold_fig_path
                )
                mlflow.log_artifact(threshold_fig_path)
        
        # Converter resultados das métricas para DataFrame para facilitar análise
        metrics_summary = []
        
        for calibrator, metric_dict in all_metrics.items():
            for metric, values in metric_dict.items():
                row = {
                    'calibrator': calibrator,
                    'optimization_metric': metric,
                    **values
                }
                metrics_summary.append(row)
        
        metrics_df = pd.DataFrame(metrics_summary)
        metrics_df.to_csv(os.path.join(output_dir, "threshold_metrics_summary.csv"), index=False)
        
        # Identificar a melhor combinação de calibrador e métrica
        # Vamos usar F2-score como critério final, dado o desbalanceamento das classes
        best_row = metrics_df.loc[metrics_df['f2'].idxmax()]
        
        best_calibrator = best_row['calibrator']
        best_metric = best_row['optimization_metric']
        best_threshold = best_row['threshold']
        
        print(f"\nMelhor configuração:")
        print(f"  Calibrador: {best_calibrator}")
        print(f"  Métrica de otimização: {best_metric}")
        print(f"  Threshold: {best_threshold:.4f}")
        print(f"  F2-score: {best_row['f2']:.4f}")
        print(f"  Precision: {best_row['precision']:.4f}")
        print(f"  Recall: {best_row['recall']:.4f}")
        
        # Salvar a melhor configuração
        best_model_final = calibrated_models[best_calibrator]
        
        # Gerar previsões finais com o melhor modelo e threshold
        final_probs = calibration_probs[best_calibrator]
        final_preds = (final_probs >= best_threshold).astype(int)
        
        # Criar matriz de confusão
        cm_fig_path = os.path.join(output_dir, "final_confusion_matrix.png")
        plot_confusion_matrix(
            y_val, final_preds, 
            f'Matriz de Confusão - Modelo Final (threshold={best_threshold:.4f})', 
            cm_fig_path
        )
        mlflow.log_artifact(cm_fig_path)
        
        # Histograma de probabilidades
        hist_fig_path = os.path.join(output_dir, "final_prob_histogram.png")
        plot_prob_histogram(
            y_val, final_probs, best_threshold, 
            'Distribuição de Probabilidades - Modelo Final', 
            hist_fig_path
        )
        mlflow.log_artifact(hist_fig_path)
        
        # Curva precision-recall
        pr_curve_path = os.path.join(output_dir, "final_pr_curve.png")
        plot_precision_recall_curve(
            y_val, final_probs, 
            'Curva Precision-Recall - Modelo Final', 
            pr_curve_path
        )
        mlflow.log_artifact(pr_curve_path)
        
        # Importância das features
        if hasattr(best_model_final, 'feature_importances_'):
            importance_fig_path = os.path.join(output_dir, "final_feature_importance.png")
            
            # Para modelos calibrados, precisamos acessar o estimador base
            if hasattr(best_model_final, 'estimator'):
                # É um modelo calibrado, mas o CalibratedClassifierCV pode ter múltiplos
                # estimadores base se treinado com cv > 0
                if hasattr(best_model_final, 'estimators_'):
                    # Calcular a média das importâncias dos estimadores
                    importances = np.mean([est.feature_importances_ for est in best_model_final.estimators_], axis=0)
                else:
                    # Somente um estimador
                    importances = best_model_final.estimator.feature_importances_
            else:
                # É o estimador diretamente
                importances = best_model_final.feature_importances_
                
            importance_df = plot_feature_importance(
                list(X_train.columns), 
                importances,
                "Final Model", 
                importance_fig_path, 
                top_n=30
            )
            mlflow.log_artifact(importance_fig_path)
            mlflow.log_artifact(importance_fig_path.replace('.png', '.csv'))
        
        # Registrar métricas finais no MLflow
        mlflow.log_metric("final_precision", float(best_row['precision']))
        mlflow.log_metric("final_recall", float(best_row['recall']))
        mlflow.log_metric("final_f1", float(best_row['f1']))
        mlflow.log_metric("final_f2", float(best_row['f2']))
        mlflow.log_metric("final_threshold", float(best_threshold))
        mlflow.log_param("best_calibrator", best_calibrator)
        mlflow.log_param("best_metric", best_metric)
        
        # Salvar o modelo final no MLflow
        # Para CalibratedClassifierCV, salvamos o modelo completo
        input_example = X_train.iloc[:5].copy()
        mlflow.sklearn.log_model(
            best_model_final, 
            "optimized_random_forest",
            input_example=input_example
        )
        
        model_uri = f"runs:/{run_id}/optimized_random_forest"
        print(f"\nModelo final salvo no MLflow: {model_uri}")
        
        # Salvar resultados completos
        return {
            'best_model': best_model_final,
            'best_calibrator': best_calibrator,
            'best_metric': best_metric,
            'best_threshold': best_threshold,
            'metrics': best_row.to_dict(),
            'model_uri': model_uri,
            'all_metrics': all_metrics,
            'calibrated_models': calibrated_models,
            'output_dir': output_dir
        }

def main():
    """
    Função principal para executar o pipeline de hyperparameter tuning.
    """
    parser = argparse.ArgumentParser(description='Executar hyperparameter tuning com o modelo RandomForest')
    parser.add_argument('--mlflow_dir', default=None, 
                      help='Diretório do MLflow tracking')
    parser.add_argument('--train_path', default=None,
                      help='Caminho para o dataset de treino')
    parser.add_argument('--val_path', default=None,
                      help='Caminho para o dataset de validação')
    parser.add_argument('--target_col', default='target',
                      help='Nome da coluna target')
    parser.add_argument('--run_id', default=None,
                      help='ID específico do run do RandomForest (opcional)')
    parser.add_argument('--model_uri', default=None,
                      help='URI específico do modelo RandomForest (opcional)')
    parser.add_argument('--metric', default='precision',
                      help='Métrica para selecionar o melhor modelo (precision, recall, f1)')
    parser.add_argument('--min_metric_value', type=float, default=0.5,
                      help='Valor mínimo da métrica para considerar um modelo')
    
    args = parser.parse_args()
    
    # Definir valor padrão para mlflow_dir se não fornecido
    if args.mlflow_dir is None:
        default_mlflow_dir = os.path.join(os.path.expanduser("~"), "desktop", "smart_ads", "models", "mlflow")
        if os.path.exists(default_mlflow_dir):
            args.mlflow_dir = default_mlflow_dir
        else:
            args.mlflow_dir = os.path.join(project_root, "models", "mlflow")
    
    print("=== Iniciando hyperparameter tuning com o modelo RandomForest ===")
    print(f"Diretório MLflow: {args.mlflow_dir}")
    
    # Configurar MLflow tracking
    experiment_id = setup_mlflow_tracking(
        tracking_dir=args.mlflow_dir,
        experiment_name="smart_ads_hyperparameter_tuning",
        clean_previous=False
    )
    
    # Carregar modelo do MLflow
    base_model = None
    
    # Se um model_uri específico foi fornecido, usar ele
    if args.model_uri:
        print(f"Usando model_uri fornecido: {args.model_uri}")
        base_model = load_model_from_mlflow(args.model_uri)
    # Se um run_id específico foi fornecido, construir o model_uri
    elif args.run_id:
        print(f"Usando run_id fornecido: {args.run_id}")
        model_uri = f"runs:/{args.run_id}/random_forest"
        base_model = load_model_from_mlflow(model_uri)
    # Caso contrário, procurar o melhor modelo com base na métrica
    else:
        print(f"Procurando o melhor modelo com base na métrica: {args.metric}")
        run_id, threshold_from_mlflow, model_uri = get_best_model_run(
            args.mlflow_dir, 
            metric=args.metric, 
            min_value=args.min_metric_value
        )
        
        if model_uri:
            base_model = load_model_from_mlflow(model_uri)
    
    # Verificar se o modelo foi carregado
    if base_model is None:
        print("ERRO: Não foi possível carregar o modelo RandomForest base.")
        sys.exit(1)
    
    # Definir path padrão para o dataset de validação se não fornecido
    if args.val_path is None:
        default_val_path = os.path.join(os.path.expanduser("~"), "desktop", "smart_ads", "data", 
                                     "03_3_feature_selection_text_code6", "validation.csv")
        if os.path.exists(default_val_path):
            args.val_path = default_val_path
    
    # Definir path padrão para o dataset de treino se não fornecido
    if args.train_path is None:
        default_train_path = os.path.join(os.path.expanduser("~"), "desktop", "smart_ads", "data", 
                                       "03_3_feature_selection_text_code6", "train.csv")
        if os.path.exists(default_train_path):
            args.train_path = default_train_path
    
    # Carregar datasets
    train_df, val_df = load_datasets(args.train_path, args.val_path)
    
    # Criar diretório para artefatos
    artifact_dir = os.path.join(os.path.expanduser("~"), "desktop", "smart_ads", "models", "artifacts", "hyperparameter_tuning")
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Executar o pipeline de hyperparameter tuning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "reports", f"hyperparameter_tuning_{timestamp}")
    
    results = run_hyperparameter_tuning_pipeline(
        train_df=train_df,
        val_df=val_df,
        base_model=base_model,
        target_col=args.target_col,
        output_dir=output_dir,
        experiment_id=experiment_id,
        artifact_dir=artifact_dir
    )
    
    print("\n=== Hyperparameter tuning concluído ===")
    print(f"Melhor modelo salvo no MLflow: {results['model_uri']}")
    print(f"Threshold ótimo: {results['best_threshold']:.4f}")
    print(f"Calibrador: {results['best_calibrator']}")
    print(f"Métrica de otimização: {results['best_metric']}")
    print(f"F2-score final: {results['metrics']['f2']:.4f}")
    print(f"Precision final: {results['metrics']['precision']:.4f}")
    print(f"Recall final: {results['metrics']['recall']:.4f}")
    print(f"\nTodos os resultados foram salvos em: {results['output_dir']}")

if __name__ == "__main__":
    # Importação para o scorer do GridSearchCV
    from sklearn.metrics import make_scorer, fbeta_score
    main()