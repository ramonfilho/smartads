#!/usr/bin/env python
"""
Script para análise de erros do modelo RandomForest utilizando
um modelo específico salvo no MLflow.
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

# Caminho fixo para o modelo que queremos analisar
MODEL_PATH = "/Users/ramonmoreira/desktop/smart_ads/models/mlflow/783044341530730386/2c54d70c822c4e42ad92313f4f2bfe8e/artifacts/random_forest"

# Caminhos fixos para os datasets
TRAIN_PATH = "/Users/ramonmoreira/desktop/smart_ads/data/03_4_feature_selection_final/train.csv"
VAL_PATH = "/Users/ramonmoreira/desktop/smart_ads/data/03_4_feature_selection_final/validation.csv"

# Threshold padrão (fallback)
DEFAULT_THRESHOLD = 0.12

def load_datasets(train_path=TRAIN_PATH, val_path=VAL_PATH):
    """
    Carrega os datasets de treino e validação.
    
    Args:
        train_path: Caminho para o dataset de treino
        val_path: Caminho para o dataset de validação
        
    Returns:
        Tuple (train_df, val_df)
    """
    # Verificar se os arquivos existem
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_path}")
    
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Arquivo de validação não encontrado: {val_path}")
    
    # Carregar os dados
    print(f"Carregando dados de treino: {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"Carregando dados de validação: {val_path}")
    val_df = pd.read_csv(val_path)
    
    print(f"Dados carregados: treino {train_df.shape}, validação {val_df.shape}")
    return train_df, val_df

def get_model_info(model):
    """
    Obtém informações detalhadas sobre o modelo RandomForest.
    
    Args:
        model: Modelo RandomForest carregado
        
    Returns:
        Dicionário com informações do modelo
    """
    model_info = {}
    
    # Informações básicas
    model_info['tipo'] = str(type(model).__name__)
    
    # Hiperparâmetros
    if hasattr(model, 'get_params'):
        params = model.get_params()
        model_info['hiperparametros'] = params
    
    # Número de estimadores (árvores)
    if hasattr(model, 'n_estimators'):
        model_info['n_estimadores'] = model.n_estimators
    
    # Profundidade máxima
    if hasattr(model, 'max_depth'):
        model_info['profundidade_maxima'] = model.max_depth
    
    # Número de features
    if hasattr(model, 'n_features_in_'):
        model_info['n_features'] = model.n_features_in_
    
    # Critério de divisão
    if hasattr(model, 'criterion'):
        model_info['criterio'] = model.criterion
    
    # Importância de features
    if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
        # Obter as 10 features mais importantes
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:10]
        top_features = [(model.feature_names_in_[i], importances[i]) for i in top_indices]
        model_info['top_10_features'] = top_features
    
    # Classes
    if hasattr(model, 'classes_'):
        model_info['classes'] = model.classes_
    
    # Balanceamento de classe
    if hasattr(model, 'class_weight') and model.class_weight is not None:
        model_info['class_weight'] = model.class_weight
    
    return model_info

def recover_model_threshold(model_path):
    """
    Tenta recuperar o threshold salvo junto com o modelo, diretamente dos arquivos.
    
    Args:
        model_path: Caminho para o diretório do modelo
        
    Returns:
        threshold (float) ou None se não encontrado
    """
    # Caminho base para a execução (subindo um nível a partir do diretório random_forest)
    run_dir = os.path.dirname(os.path.dirname(model_path))
    
    # Verificar se existe um arquivo de métrica específico para threshold
    metrics_dir = os.path.join(run_dir, "metrics")
    if os.path.exists(metrics_dir):
        threshold_file = os.path.join(metrics_dir, "threshold")
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, 'r') as f:
                    # O arquivo de métrica do MLflow normalmente contém linhas com timestamp valor
                    lines = f.readlines()
                    if lines:
                        # Pegar o valor mais recente (última linha)
                        last_line = lines[-1].strip()
                        # O formato é geralmente: timestamp valor
                        parts = last_line.split()
                        if len(parts) >= 2:
                            threshold_value = float(parts[1])
                            print(f"Threshold encontrado no arquivo de métrica: {threshold_value}")
                            return threshold_value
            except Exception as e:
                print(f"Erro ao ler arquivo de threshold: {e}")
    
    # Se não encontrou no arquivo de métrica específico, buscar no diretório de métricas geral
    # O MLflow pode armazenar as métricas em arquivos separados
    print("Buscando threshold nos arquivos de métricas...")
    
    # Carregar outras métricas importantes para mostrar junto
    metrics = {}
    metrics_found = False
    
    if os.path.exists(metrics_dir):
        for metric_file in os.listdir(metrics_dir):
            metric_path = os.path.join(metrics_dir, metric_file)
            if os.path.isfile(metric_path):
                try:
                    with open(metric_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Pegar o valor mais recente
                            last_line = lines[-1].strip()
                            parts = last_line.split()
                            if len(parts) >= 2:
                                metrics[metric_file] = float(parts[1])
                                metrics_found = True
                except Exception as e:
                    print(f"Erro ao ler métrica {metric_file}: {e}")
    
    if metrics_found:
        print("Métricas encontradas:")
        for metric_name, metric_value in metrics.items():
            print(f"  - {metric_name}: {metric_value:.4f}")
        
        if "threshold" in metrics:
            print(f"Threshold encontrado nas métricas: {metrics['threshold']}")
            return metrics["threshold"]
    
    # Se ainda não encontrou, buscar nos parâmetros
    params_dir = os.path.join(run_dir, "params")
    if os.path.exists(params_dir):
        threshold_param = os.path.join(params_dir, "threshold")
        if os.path.exists(threshold_param):
            try:
                with open(threshold_param, 'r') as f:
                    threshold_value = float(f.read().strip())
                    print(f"Threshold encontrado nos parâmetros: {threshold_value}")
                    return threshold_value
            except Exception as e:
                print(f"Erro ao ler parâmetro threshold: {e}")
    
    # Como último recurso, tentar ler o arquivo meta.yaml
    meta_path = os.path.join(model_path, "meta.yaml")
    if os.path.exists(meta_path):
        try:
            import yaml
            with open(meta_path, 'r') as f:
                meta_data = yaml.safe_load(f)
            
            # Verificar se há informações sobre threshold no metadata
            if 'threshold' in meta_data:
                threshold_value = float(meta_data['threshold'])
                print(f"Threshold encontrado no meta.yaml: {threshold_value}")
                return threshold_value
        except Exception as e:
            print(f"Erro ao ler meta.yaml: {e}")
    
    print("Não foi possível encontrar o threshold nos arquivos do modelo.")
    return None

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

def run_error_analysis(model, val_df, target_col="target", threshold=DEFAULT_THRESHOLD, output_dir=None):
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
        
        # Salvar métricas em um arquivo
        metrics_df = pd.DataFrame({
            'threshold': [threshold],
            'precision': [precision],
            'recall': [recall],
            'f1': [f1],
            'positives': [int(y_pred.sum())],
            'positives_pct': [float(y_pred.mean() * 100)],
            'false_positives': [int(((y_val == 0) & (y_pred == 1)).sum())],
            'false_negatives': [int(((y_val == 1) & (y_pred == 0)).sum())]
        })
        metrics_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
        
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
    parser = argparse.ArgumentParser(description='Executar análise de erros com um modelo RandomForest específico')
    parser.add_argument('--train_path', default=TRAIN_PATH,
                      help=f'Caminho para o dataset de treino (padrão: {TRAIN_PATH})')
    parser.add_argument('--val_path', default=VAL_PATH,
                      help=f'Caminho para o dataset de validação (padrão: {VAL_PATH})')
    parser.add_argument('--target_col', default='target',
                      help='Nome da coluna target')
    parser.add_argument('--threshold', type=float, default=None,
                      help=f'Threshold personalizado para classificação (padrão: recuperado do modelo ou {DEFAULT_THRESHOLD})')
    
    args = parser.parse_args()
    
    print("=== Iniciando análise de erros com modelo RandomForest específico ===")
    print(f"Caminho do modelo: {MODEL_PATH}")
    print(f"Caminho do dataset de treino: {args.train_path}")
    print(f"Caminho do dataset de validação: {args.val_path}")
    
    # Carregar o modelo do caminho específico
    try:
        print("\nCarregando modelo diretamente do caminho especificado...")
        model = mlflow.sklearn.load_model(MODEL_PATH)
        print("Modelo carregado com sucesso!")
        
        # Recuperar o threshold salvo com o modelo
        saved_threshold = recover_model_threshold(MODEL_PATH)
        
        # Determinar qual threshold usar
        if args.threshold is not None:
            threshold = args.threshold
            print(f"Usando threshold fornecido via argumento: {threshold}")
        elif saved_threshold is not None:
            threshold = saved_threshold
            print(f"Usando threshold recuperado do modelo: {threshold}")
        else:
            threshold = DEFAULT_THRESHOLD
            print(f"Usando threshold padrão: {threshold}")
        
        # Obter e imprimir informações do modelo
        print("\n=== Informações do Modelo ===")
        model_info = get_model_info(model)
        
        print(f"Tipo de modelo: {model_info['tipo']}")
        if 'n_estimadores' in model_info:
            print(f"Número de árvores: {model_info['n_estimadores']}")
        if 'profundidade_maxima' in model_info:
            print(f"Profundidade máxima: {model_info['profundidade_maxima']}")
        if 'criterio' in model_info:
            print(f"Critério de divisão: {model_info['criterio']}")
        if 'n_features' in model_info:
            print(f"Número de features: {model_info['n_features']}")
        if 'class_weight' in model_info:
            print(f"Balanceamento de classes: {model_info['class_weight']}")
        
        if 'top_10_features' in model_info:
            print("\nTop 10 features mais importantes:")
            for i, (feature, importance) in enumerate(model_info['top_10_features']):
                print(f"{i+1}. {feature}: {importance:.4f}")
        
        print("\nPrincipais hiperparâmetros:")
        if 'hiperparametros' in model_info:
            important_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
            for param in important_params:
                if param in model_info['hiperparametros']:
                    print(f"- {param}: {model_info['hiperparametros'][param]}")
        
    except Exception as e:
        print(f"ERRO ao carregar o modelo: {e}")
        sys.exit(1)
    
    # Verificar se o modelo é RandomForest
    model_type = str(type(model))
    if 'RandomForest' not in model_type:
        print(f"AVISO: O modelo carregado não é RandomForest! É {model_type}")
        proceed = input("Deseja continuar mesmo assim? (s/n): ")
        if proceed.lower() != 's':
            print("Análise cancelada.")
            return
    
    # Carregar dados
    try:
        train_df, val_df = load_datasets(args.train_path, args.val_path)
    except FileNotFoundError as e:
        print(f"ERRO: {str(e)}")
        sys.exit(1)
    
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