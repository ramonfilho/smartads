#!/usr/bin/env python
"""
Script para análise de erros do modelo RandomForest utilizando
especificamente o último modelo treinado pelo script 04_baseline_model_training.py.
"""

import os
import sys
import argparse
import mlflow
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Importar funções necessárias para análise de erros
from src.evaluation.error_analysis import (
    analyze_high_value_false_negatives,
    cluster_errors,
    analyze_feature_distributions_by_error_type
)

# Importar a função de sanitização de nomes de colunas que foi usada no treinamento
from src.evaluation.baseline_model import sanitize_column_names

def get_latest_random_forest_run(mlflow_dir):
    """
    Obtém o run_id do modelo RandomForest mais recente.
    
    Args:
        mlflow_dir: Diretório do MLflow tracking
        
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

def prepare_data_for_model(model, val_df, target_col="target"):
    """
    Prepara os dados para serem compatíveis com o modelo.
    
    Args:
        model: Modelo treinado
        val_df: DataFrame de validação
        target_col: Nome da coluna target
        
    Returns:
        X_val, y_val preparados para o modelo
    """
    # Extrair features e target
    X_val = val_df.drop(columns=[target_col]) if target_col in val_df.columns else val_df.copy()
    y_val = val_df[target_col] if target_col in val_df.columns else None
    
    # Aplicar a mesma sanitização de nomes de colunas usada no treinamento
    col_mapping = sanitize_column_names(X_val)
    
    # Converter inteiros para float
    for col in X_val.columns:
        if pd.api.types.is_integer_dtype(X_val[col].dtype):
            X_val.loc[:, col] = X_val[col].astype(float)
    
    # Verificar features do modelo
    if hasattr(model, 'feature_names_in_'):
        expected_features = set(model.feature_names_in_)
        actual_features = set(X_val.columns)
        
        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features
        
        if missing_features:
            print(f"AVISO: Faltam {len(missing_features)} features que o modelo espera")
            print(f"  Exemplos: {list(missing_features)[:5]}")
            
            # Criar DataFrame vazio com as colunas corretas para minimizar a fragmentação
            missing_cols = list(missing_features)
            missing_dict = {col: [0] * len(X_val) for col in missing_cols}
            missing_df = pd.DataFrame(missing_dict)
            
            # Concatenar em vez de adicionar uma por uma
            X_val = pd.concat([X_val, missing_df], axis=1)
        
        if extra_features:
            print(f"AVISO: Removendo {len(extra_features)} features extras")
            print(f"  Exemplos: {list(extra_features)[:5]}")
            X_val = X_val.drop(columns=list(extra_features))
        
        # Garantir a ordem correta das colunas
        X_val = X_val[model.feature_names_in_]
    
    return X_val, y_val

def run_error_analysis(model, val_df, target_col="target", threshold=0.17, output_dir=None):
    """
    Executa análise de erros no modelo e dados.
    
    Args:
        model: Modelo treinado
        val_df: DataFrame de validação
        target_col: Nome da coluna target
        threshold: Threshold para classificação
        output_dir: Diretório para salvar resultados
        
    Returns:
        Dicionário com resultados da análise
    """
    # Criar diretório para resultados
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "reports", f"error_analysis_rf_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {output_dir}")
    
    # Preparar dados
    X_val, y_val = prepare_data_for_model(model, val_df, target_col)
    
    if y_val is None:
        raise ValueError(f"Coluna target '{target_col}' não encontrada no DataFrame")
    
    # Gerar previsões
    print(f"Gerando previsões com threshold={threshold}...")
    try:
        y_pred_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)
        
        # Métricas básicas
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"Métricas com threshold={threshold}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        
        # Análise de falsos negativos
        print("\nRealizando análise de falsos negativos...")
        fn_data = analyze_high_value_false_negatives(
            val_df=val_df,
            y_val=y_val,
            y_pred_val=y_pred,
            y_pred_prob_val=y_pred_prob,
            target_col=target_col,
            results_dir=output_dir
        )
        
        # Análise de distribuições
        print("\nAnalisando distribuições por tipo de erro...")
        error_distributions = analyze_feature_distributions_by_error_type(
            val_df=val_df,
            y_val=y_val,
            y_pred_val=y_pred,
            y_pred_prob_val=y_pred_prob,
            target_col=target_col,
            results_dir=output_dir
        )
        
        # Clustering de erros
        print("\nRealizando clustering de erros...")
        analysis_df = val_df.copy()
        analysis_df['error_type'] = 'correct'
        analysis_df.loc[(y_val == 1) & (y_pred == 0), 'error_type'] = 'false_negative'
        analysis_df.loc[(y_val == 0) & (y_pred == 1), 'error_type'] = 'false_positive'
        analysis_df['probability'] = y_pred_prob
        
        # Selecionar features para clustering
        numeric_cols = [col for col in val_df.columns if 
                      pd.api.types.is_numeric_dtype(val_df[col]) and 
                      col != target_col]
        
        # Limitar número de features para clustering
        if len(numeric_cols) > 30:
            numeric_cols = numeric_cols[:30]
        
        clustered_df = cluster_errors(
            analysis_df=analysis_df,
            features=[f for f in numeric_cols if f in analysis_df.columns],
            n_clusters=3
        )
        
        # Salvar dados com clusters
        clustered_df.to_csv(os.path.join(output_dir, "leads_clusters.csv"), index=False)
        
        print(f"\nAnálise de erros concluída. Resultados salvos em: {output_dir}")
        return {
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'fn_data': fn_data,
            'error_distributions': error_distributions,
            'output_dir': output_dir
        }
        
    except Exception as e:
        print(f"Erro durante a análise: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='Executar análise de erros com o último modelo RandomForest treinado')
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
    parser.add_argument('--threshold', type=float, default=None,
                      help='Threshold personalizado para classificação (opcional)')
    
    args = parser.parse_args()
    
    # Definir valor padrão para mlflow_dir se não fornecido
    if args.mlflow_dir is None:
        default_mlflow_dir = os.path.join(os.path.expanduser("~"), "desktop", "smart_ads", "models", "mlflow")
        if os.path.exists(default_mlflow_dir):
            args.mlflow_dir = default_mlflow_dir
        else:
            args.mlflow_dir = os.path.join(project_root, "models", "mlflow")
    
    print("=== Iniciando análise de erros com o último modelo RandomForest ===")
    print(f"Diretório MLflow: {args.mlflow_dir}")
    
    # Carregar modelo do MLflow
    model = None
    threshold = args.threshold
    
    # Se um model_uri específico foi fornecido, usar ele
    if args.model_uri:
        model = load_model_from_mlflow(args.model_uri)
    # Se um run_id específico foi fornecido, construir o model_uri
    elif args.run_id:
        model_uri = f"runs:/{args.run_id}/random_forest"
        model = load_model_from_mlflow(model_uri)
    # Caso contrário, procurar o mais recente
    else:
        run_id, threshold_from_mlflow, model_uri = get_latest_random_forest_run(args.mlflow_dir)
        
        if model_uri:
            model = load_model_from_mlflow(model_uri)
            # Usar o threshold do MLflow se não especificado manualmente
            if threshold is None:
                threshold = threshold_from_mlflow
                print(f"Usando threshold do MLflow: {threshold}")
    
    # Se ainda não tem threshold, usar o valor padrão
    if threshold is None:
        threshold = 0.17
        print(f"Usando threshold padrão: {threshold}")
    
    # Verificar se o modelo foi carregado
    if model is None:
        print("ERRO: Não foi possível carregar o modelo RandomForest.")
        sys.exit(1)
    
    # Confirmar que é RandomForest
    model_type = str(type(model))
    if 'RandomForest' not in model_type:
        print(f"AVISO: O modelo carregado não é RandomForest! É {model_type}")
        proceed = input("Deseja continuar mesmo assim? (s/n): ")
        if proceed.lower() != 's':
            print("Análise cancelada.")
            return
    
    # Carregar dados
    train_df, val_df = load_datasets(args.train_path, args.val_path)
    
    # Executar análise
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "reports", f"error_analysis_rf_{timestamp}")
    
    results = run_error_analysis(
        model=model,
        val_df=val_df,
        target_col=args.target_col,
        threshold=threshold,
        output_dir=output_dir
    )
    
    print("\n=== Análise de erros concluída ===")
    print(f"Todos os resultados foram salvos em: {results['output_dir']}")

if __name__ == "__main__":
    main()