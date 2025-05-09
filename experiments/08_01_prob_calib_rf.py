#ESSE SCRIPT CALIBRA O MODELO TREINADO EM CIMA DAS FEATURES SELECIONADAS APÓS O TRATAMENTO DE VALORES AUSENTES EM TODO O DATASET, COM O CÓDIGO EXPERIMENTAL 04_01

#!/usr/bin/env python
"""
Script para calibrar as probabilidades do modelo RandomForest.
Versão melhorada com rastreamento e consistência de features.
"""

import os
import re
import sys
import argparse
import mlflow
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Importar a função de sanitização de nomes de colunas que foi usada no treinamento
from src.evaluation.baseline_model import sanitize_column_names
from src.evaluation.mlflow_utils import find_optimal_threshold

# Caminho para o modelo Random Forest
MODEL_PATH = "/Users/ramonmoreira/desktop/smart_ads/mlflow/460572322746954574/3a1a0381e74e4a4db31c6236c062e975/artifacts/random_forest"

# Caminhos para os datasets
TRAIN_PATH = "/Users/ramonmoreira/desktop/smart_ads/data/03_5_feature_selection_final_treated/train.csv"
VAL_PATH = "/Users/ramonmoreira/desktop/smart_ads/data/03_5_feature_selection_final_treated/validation.csv"
TEST_PATH = "/Users/ramonmoreira/desktop/smart_ads/data/03_5_feature_selection_final_treated/test.csv"

# Threshold padrão (fallback)
DEFAULT_THRESHOLD = 0.12

def load_datasets(train_path=TRAIN_PATH, val_path=VAL_PATH, test_path=TEST_PATH):
    """
    Carrega os datasets de treino, validação e teste.
    
    Args:
        train_path: Caminho para o dataset de treino
        val_path: Caminho para o dataset de validação
        test_path: Caminho para o dataset de teste
        
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    # Verificar se os arquivos existem
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_path}")
    
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Arquivo de validação não encontrado: {val_path}")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {test_path}")
    
    # Carregar os dados
    print(f"Carregando dados de treino: {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"Carregando dados de validação: {val_path}")
    val_df = pd.read_csv(val_path)
    
    print(f"Carregando dados de teste: {test_path}")
    test_df = pd.read_csv(test_path)
    
    print(f"Dados carregados: treino {train_df.shape}, validação {val_df.shape}, teste {test_df.shape}")
    return train_df, val_df, test_df

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
    
    print("Não foi possível encontrar o threshold nos arquivos do modelo.")
    return DEFAULT_THRESHOLD

def analyze_dataset_stats(df, dataset_name="Dataset"):
    """
    Analisa estatísticas básicas do dataset para verificar a consistência.
    
    Args:
        df: DataFrame a ser analisado
        dataset_name: Nome do dataset para exibição
        
    Returns:
        Dicionário com estatísticas
    """
    stats = {}
    
    # Informações básicas
    stats['num_rows'] = len(df)
    stats['num_columns'] = df.shape[1]
    
    # Tipos de dados
    type_counts = df.dtypes.value_counts().to_dict()
    stats['data_types'] = {str(k): v for k, v in type_counts.items()}
    
    # Valores ausentes
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if not cols_with_missing.empty:
        stats['missing_values'] = cols_with_missing.to_dict()
        print(f"\n{dataset_name}: Encontradas {len(cols_with_missing)} colunas com valores ausentes")
        for col, count in cols_with_missing.items():
            missing_pct = (count / len(df)) * 100
            print(f"  {col}: {count} valores ausentes ({missing_pct:.2f}%)")
    else:
        stats['missing_values'] = {}
        print(f"\n{dataset_name}: Nenhum valor ausente encontrado")
    
    # Valores extremos em colunas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns
    stats['numeric_stats'] = {}
    for col in numeric_cols:
        if col in df.columns:  # Garantir que a coluna existe
            stats['numeric_stats'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std()
            }
    
    return stats

def prepare_data_for_model_improved(model, df, train_df=None, target_col="target", allow_missing=False, output_dir=None):
    """
    Prepara os dados para serem compatíveis com o modelo, com melhor tratamento
    de features ausentes e registro de estatísticas.
    
    Args:
        model: Modelo treinado
        df: DataFrame com os dados a serem preparados
        train_df: DataFrame de treino (opcional, para estatísticas de referência)
        target_col: Nome da coluna target
        allow_missing: Se True, preenche features ausentes; se False, falha se faltarem features
        output_dir: Diretório para salvar estatísticas (opcional)
        
    Returns:
        Tuple (X, y, feature_stats)
    """
    # Salvar estatísticas originais do dataframe
    original_stats = analyze_dataset_stats(df, "Dataset original")
    
    # Extrair features e target
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    y = df[target_col] if target_col in df.columns else None
    
    print(f"\nPreparando dados (shape inicial: {X.shape})...")
    
    # Salvar nomes originais das colunas
    original_columns = X.columns.tolist()
    
    # Aplicar a mesma sanitização de nomes de colunas usada no treinamento
    # Modificado para preservar o mapeamento explicitamente
    col_mapping = {}
    for col in X.columns:
        # Sanitizar nome
        new_col = re.sub(r'[^\w\s]', '_', col)
        new_col = re.sub(r'\s+', '_', new_col)
        # Garantir que não comece com número
        if new_col[0].isdigit():
            new_col = 'f_' + new_col
        # Verificar se já existe esse novo nome
        i = 1
        temp_col = new_col
        while temp_col in col_mapping.values():
            temp_col = f"{new_col}_{i}"
            i += 1
        new_col = temp_col
        # Adicionar ao mapeamento apenas se o nome mudou
        if col != new_col:
            col_mapping[col] = new_col
    
    # Renomear colunas usando o mapeamento
    if col_mapping:
        print(f"Sanitizando {len(col_mapping)} nomes de colunas...")
        X = X.rename(columns=col_mapping)
    
    # Converter inteiros para float (preservar compatibilidade com modelos)
    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col].dtype):
            X.loc[:, col] = X[col].astype(float)
    
    # Verificar features do modelo
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        actual_features = list(X.columns)
        
        # Verificar se as features estão na ordem correta
        reorder_needed = (actual_features != expected_features[:len(actual_features)] or 
                         len(actual_features) != len(expected_features))
        
        if reorder_needed:
            print(f"Verificando compatibilidade de features...")
            
            # Colunas presentes no modelo mas ausentes no dataset
            missing_features = [f for f in expected_features if f not in actual_features]
            
            # Colunas presentes no dataset mas não esperadas pelo modelo
            extra_features = [f for f in actual_features if f not in expected_features]
            
            if missing_features and not allow_missing:
                raise ValueError(
                    f"Faltam {len(missing_features)} features que o modelo espera. "
                    f"Use allow_missing=True para preencher automaticamente: {missing_features[:5]}..."
                )
            
            if missing_features:
                print(f"AVISO: Adicionando {len(missing_features)} features ausentes...")
                
                # Obter estatísticas do treino para preencher valores sensatos
                if train_df is not None:
                    print("Usando dataset de treino para estatísticas de preenchimento...")
                    
                    # Preparar dataset de treino (sem adicionar features ausentes)
                    X_train_temp = train_df.drop(columns=[target_col]) if target_col in train_df.columns else train_df.copy()
                    
                    # Aplicar o mesmo mapeamento de nomes
                    if col_mapping:
                        X_train_temp = X_train_temp.rename(columns=col_mapping)
                    
                    # Obter médias/medianas para colunas faltantes
                    fill_values = {}
                    for col in missing_features:
                        if col in X_train_temp.columns:
                            # Usar mediana para valores numéricos
                            if pd.api.types.is_numeric_dtype(X_train_temp[col]):
                                fill_values[col] = X_train_temp[col].median()
                            else:
                                # Usar moda para categóricas
                                fill_values[col] = X_train_temp[col].mode().iloc[0] if not X_train_temp[col].mode().empty else None
                    
                    # Adicionar colunas ausentes com valores baseados na distribuição de treino
                    for col in missing_features:
                        if col in fill_values:
                            X[col] = fill_values[col]
                            print(f"  Preenchendo '{col}' com valor {fill_values[col]} (do treino)")
                        else:
                            # Fallback: usar zero
                            X[col] = 0
                            print(f"  Preenchendo '{col}' com zero (valor padrão)")
                else:
                    # Sem dataset de treino, usar valores padrão
                    print("AVISO: Preenchendo features ausentes com valores padrão (zero)...")
                    for col in missing_features:
                        X[col] = 0
            
            if extra_features:
                print(f"AVISO: Removendo {len(extra_features)} features extras...")
                X = X.drop(columns=extra_features)
            
            # Reordenar colunas para corresponder exatamente à ordem do modelo
            X = X[expected_features]
            
            print(f"Dados preparados com shape final: {X.shape}")
    
    # Analisar estatísticas após o processamento
    processed_stats = analyze_dataset_stats(X, "Dataset processado")
    
    # Registrar estatísticas em um arquivo se especificado
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stats_file = os.path.join(output_dir, "feature_stats.txt")
        
        with open(stats_file, 'w') as f:
            f.write("=== Estatísticas de Features ===\n\n")
            
            f.write("--- Colunas Originais ---\n")
            for col in original_columns:
                f.write(f"- {col}\n")
            
            f.write("\n--- Mapeamento de Colunas ---\n")
            for old_col, new_col in col_mapping.items():
                f.write(f"- {old_col} -> {new_col}\n")
            
            if hasattr(model, 'feature_names_in_'):
                f.write("\n--- Features Esperadas pelo Modelo ---\n")
                for col in model.feature_names_in_:
                    f.write(f"- {col}\n")
            
            f.write("\n--- Features Finais ---\n")
            for col in X.columns:
                f.write(f"- {col}\n")
        
        print(f"Estatísticas de features salvas em: {stats_file}")
    
    feature_stats = {
        'original': original_stats,
        'processed': processed_stats,
        'column_mapping': col_mapping,
        'original_columns': original_columns,
        'final_columns': X.columns.tolist()
    }
    
    # Verificar valores NaN no DataFrame processado
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"\nAVISO: Ainda existem {len(nan_cols)} colunas com valores NaN após processamento:")
        for col in nan_cols:
            nan_count = X[col].isna().sum()
            nan_pct = (nan_count / len(X)) * 100
            print(f"  {col}: {nan_count} NaNs ({nan_pct:.2f}%)")
        
        print("\nPreenchendo valores NaN restantes com 0...")
        X = X.fillna(0)
        print("Valores NaN foram preenchidos.")
    
    return X, y, feature_stats

def plot_calibration_curve(y_true, y_prob, model_name, output_dir):
    """
    Plota a curva de calibração de probabilidade.
    
    Args:
        y_true: Valores reais
        y_prob: Probabilidades preditas
        model_name: Nome do modelo para o título
        output_dir: Diretório para salvar o gráfico
        
    Returns:
        Caminho para o gráfico salvo
    """
    plt.figure(figsize=(10, 6))
    
    # Calcular a curva de calibração
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    # Plotar a curva de calibração
    plt.plot(prob_pred, prob_true, "s-", label=model_name)
    
    # Plotar a linha de referência (calibração perfeita)
    plt.plot([0, 1], [0, 1], "k--", label="Perfeitamente Calibrado")
    
    plt.xlabel('Probabilidade Média Predita (Confidence)')
    plt.ylabel('Fração de Positivos (Accuracy)')
    plt.title(f'Curva de Calibração - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar o gráfico
    fig_path = os.path.join(output_dir, f"calibration_curve_{model_name}.png")
    plt.savefig(fig_path)
    plt.close()
    
    return fig_path

def plot_precision_recall_curve_with_threshold(y_true, y_prob, threshold, model_name, output_dir):
    """
    Plota a curva precision-recall com marcação do threshold ótimo.
    
    Args:
        y_true: Valores reais
        y_prob: Probabilidades preditas
        threshold: Threshold ótimo
        model_name: Nome do modelo para o título
        output_dir: Diretório para salvar o gráfico
        
    Returns:
        Caminho para o gráfico salvo
    """
    plt.figure(figsize=(10, 6))
    
    # Calcular precision e recall para diferentes thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Encontrar o índice do threshold mais próximo do ótimo
    idx = np.argmin(np.abs(thresholds - threshold)) if len(thresholds) > 0 else 0
    
    # Plotar a curva precision-recall
    plt.plot(recall, precision, label='Precision-Recall curve')
    
    # Marcar o ponto do threshold ótimo
    if idx < len(precision):
        plt.plot(recall[idx], precision[idx], 'ro', 
                 label=f'Threshold: {threshold:.4f}, Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall - {model_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Salvar o gráfico
    fig_path = os.path.join(output_dir, f"precision_recall_curve_{model_name}.png")
    plt.savefig(fig_path)
    plt.close()
    
    return fig_path

def calibrate_model(model, X_train, y_train, X_val, y_val, method='isotonic', output_dir=None):
    """
    Calibra as probabilidades do modelo usando validação cruzada.
    
    Args:
        model: Modelo não calibrado
        X_train: Features de treino
        y_train: Target de treino
        X_val: Features de validação
        y_val: Target de validação
        method: Método de calibração ('isotonic' ou 'sigmoid')
        output_dir: Diretório para salvar resultados
        
    Returns:
        Modelo calibrado e threshold
    """
    print(f"\nCalibrando o modelo com o método '{method}'...")
    
    # Criar diretório para resultados se não especificado
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "smart_ads", "models", "calibrated", f"rf_calibrated_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {output_dir}")
    
    # Calibrar o modelo - versão atualizada para compatibilidade
    calibrated_model = CalibratedClassifierCV(
        estimator=model,  # Usando 'estimator' ao invés de 'base_estimator'
        method=method,
        cv='prefit'  # O modelo já está ajustado, não faça validação cruzada
    )
    
    # Ajustar o calibrador às probabilidades existentes
    print(f"Treinando calibrador com {len(X_val)} amostras de validação...")
    calibrated_model.fit(X_val, y_val)
    
    # Obter probabilidades antes e depois da calibração
    proba_before = model.predict_proba(X_val)[:, 1]
    proba_after = calibrated_model.predict_proba(X_val)[:, 1]
    
    # Verificar e lidar com NaNs nas probabilidades após calibração
    nan_count = np.isnan(proba_after).sum()
    if nan_count > 0:
        print(f"AVISO: Detectados {nan_count} valores NaN nas probabilidades após calibração!")
        print("Substituindo valores NaN pela média das probabilidades válidas...")
        
        # Calcular média das probabilidades válidas
        valid_probs = proba_after[~np.isnan(proba_after)]
        mean_prob = valid_probs.mean() if len(valid_probs) > 0 else 0.5
        
        # Substituir NaNs
        proba_after = np.nan_to_num(proba_after, nan=mean_prob)
        print(f"Valores NaN substituídos pela média: {mean_prob:.4f}")
    
    # Plotar curvas de calibração
    plot_calibration_curve(y_val, proba_before, "RandomForest (Não Calibrado)", output_dir)
    plot_calibration_curve(y_val, proba_after, "RandomForest (Calibrado)", output_dir)
    
    # Calcular threshold ótimo para o modelo calibrado
    threshold_results = find_optimal_threshold(y_val, proba_after)
    best_threshold = threshold_results['best_threshold']
    best_f1 = threshold_results['best_f1']
    best_precision = threshold_results['best_precision']
    best_recall = threshold_results['best_recall']
    
    print(f"Threshold ótimo após calibração: {best_threshold:.4f}")
    print(f"Métricas com threshold calibrado: F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}")
    
    # Plotar curva precision-recall com threshold ótimo
    plot_precision_recall_curve_with_threshold(
        y_val, proba_after, best_threshold, "RandomForest (Calibrado)", output_dir
    )
    
    # Salvar modelo calibrado com informações de feature
    calibration_metadata = {
        'model': calibrated_model,
        'feature_names': list(X_val.columns),
        'threshold': best_threshold,
        'method': method,
        'calibration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    model_path = os.path.join(output_dir, "rf_calibrated.joblib")
    joblib.dump(calibration_metadata, model_path)
    print(f"Modelo calibrado salvo com metadata em: {model_path}")
    
    # Salvar threshold ótimo separadamente
    threshold_path = os.path.join(output_dir, "threshold.txt")
    with open(threshold_path, 'w') as f:
        f.write(str(best_threshold))
    print(f"Threshold salvo em: {threshold_path}")
    
    # Salvar valores das features para validação
    features_path = os.path.join(output_dir, "features.txt")
    with open(features_path, 'w') as f:
        for feature in X_val.columns:
            f.write(f"{feature}\n")
    print(f"Lista de features salva em: {features_path}")
    
    # Calcular e salvar as métricas
    y_pred_before = (proba_before >= best_threshold).astype(int)
    y_pred_after = (proba_after >= best_threshold).astype(int)
    
    metrics_before = {
        'precision': precision_score(y_val, y_pred_before),
        'recall': recall_score(y_val, y_pred_before),
        'f1': f1_score(y_val, y_pred_before),
        'threshold': best_threshold
    }
    
    metrics_after = {
        'precision': precision_score(y_val, y_pred_after),
        'recall': recall_score(y_val, y_pred_after),
        'f1': f1_score(y_val, y_pred_after),
        'threshold': best_threshold
    }
    
    metrics_df = pd.DataFrame({
        'Metric': list(metrics_before.keys()),
        'Before Calibration': list(metrics_before.values()),
        'After Calibration': list(metrics_after.values())
    })
    
    metrics_path = os.path.join(output_dir, "calibration_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Métricas de comparação salvas em: {metrics_path}")
    
    return calibrated_model, best_threshold, calibration_metadata

def evaluate_on_test(model, X_test, y_test, threshold, model_name="RandomForest", output_dir=None):
    """
    Avalia o modelo no conjunto de teste.
    
    Args:
        model: Modelo calibrado
        X_test: Features de teste
        y_test: Target de teste
        threshold: Threshold ótimo
        model_name: Nome do modelo
        output_dir: Diretório para salvar resultados
        
    Returns:
        Dicionário com métricas de avaliação
    """
    print(f"\nAvaliando modelo {model_name} no conjunto de teste...")
    
    # Criar diretório para resultados se não especificado
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "reports", f"test_evaluation_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Fazer previsões
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Verificar NaNs nas probabilidades
    nan_count = np.isnan(y_pred_proba).sum()
    if nan_count > 0:
        print(f"AVISO: Detectados {nan_count} valores NaN nas probabilidades!")
        
        # Substituir NaNs pela mediana das probabilidades válidas
        valid_probs = y_pred_proba[~np.isnan(y_pred_proba)]
        median_prob = np.median(valid_probs) if len(valid_probs) > 0 else 0.5
        
        y_pred_proba = np.nan_to_num(y_pred_proba, nan=median_prob)
        print(f"Valores NaN substituídos pela mediana: {median_prob:.4f}")
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calcular métricas
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Métricas para {model_name} no conjunto de teste:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    
    # Calcular distribuição de probabilidades
    prob_bins = np.linspace(0, 1, 11)  # 10 bins
    bin_counts, _ = np.histogram(y_pred_proba, bins=prob_bins)
    bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
    
    print("\nDistribuição de probabilidades:")
    for center, count in zip(bin_centers, bin_counts):
        print(f"  {center:.1f}: {count} amostras ({count/len(y_pred_proba)*100:.1f}%)")
    
    # Salvar métricas
    metrics = {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'positive_count': int(y_pred.sum()),
        'positive_ratio': float(y_pred.mean())
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, f"test_metrics_{model_name}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Criar DataFrame de previsões para análise
    results_df = pd.DataFrame({
        'true': y_test,
        'prediction': y_pred,
        'probability': y_pred_proba
    })
    
    results_path = os.path.join(output_dir, f"test_predictions_{model_name}.csv")
    results_df.to_csv(results_path, index=False)
    
    print(f"Resultados da avaliação salvos em: {output_dir}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Calibrar e avaliar o modelo RandomForest')
    parser.add_argument('--model_path', default=MODEL_PATH,
                      help=f'Caminho para o modelo RandomForest (padrão: {MODEL_PATH})')
    parser.add_argument('--train_path', default=TRAIN_PATH,
                      help=f'Caminho para o dataset de treino (padrão: {TRAIN_PATH})')
    parser.add_argument('--val_path', default=VAL_PATH,
                      help=f'Caminho para o dataset de validação (padrão: {VAL_PATH})')
    parser.add_argument('--test_path', default=TEST_PATH,
                      help=f'Caminho para o dataset de teste (padrão: {TEST_PATH})')
    parser.add_argument('--method', default='isotonic', choices=['isotonic', 'sigmoid'],
                      help='Método de calibração (padrão: isotonic)')
    parser.add_argument('--evaluate_test', action='store_true',
                      help='Avaliar no conjunto de teste após calibração')
    
    args = parser.parse_args()
    
    print("=== Iniciando calibração do modelo RandomForest ===")
    print(f"Caminho do modelo: {args.model_path}")
    
    # Criar diretório de saída com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/Users/ramonmoreira/desktop/smart_ads/models/calibrated/rf_calibrated_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar o modelo
    try:
        print("\nCarregando modelo original...")
        model = mlflow.sklearn.load_model(args.model_path)
        print("Modelo carregado com sucesso!")
        
        # Recuperar o threshold original
        threshold = recover_model_threshold(args.model_path)
        
        # Obter informações do modelo
        model_info = get_model_info(model)
        print(f"\nTipo de modelo: {model_info['tipo']}")
        if 'n_estimadores' in model_info:
            print(f"Número de árvores: {model_info['n_estimadores']}")
        if 'profundidade_maxima' in model_info:
            print(f"Profundidade máxima: {model_info['profundidade_maxima']}")
        
    except Exception as e:
        print(f"ERRO ao carregar o modelo: {e}")
        sys.exit(1)
    
    # Carregar datasets
    try:
        train_df, val_df, test_df = load_datasets(args.train_path, args.val_path, args.test_path)
    except FileNotFoundError as e:
        print(f"ERRO: {str(e)}")
        sys.exit(1)
    
    # Preparar dados para treino, validação e teste com a função melhorada
    print("\nPreparando dados para calibração e avaliação...")
    
    # Primeiro preparar dados de treino
    X_train, y_train, train_stats = prepare_data_for_model_improved(
        model=model,
        df=train_df,
        target_col="target",
        allow_missing=True,
        output_dir=output_dir
    )
    
    # Preparar dados de validação usando estatísticas do treino
    X_val, y_val, val_stats = prepare_data_for_model_improved(
        model=model,
        df=val_df,
        train_df=train_df,  # Usar treino para estatísticas
        target_col="target",
        allow_missing=True,
        output_dir=output_dir
    )
    
    # Preparar dados de teste usando estatísticas do treino
    X_test, y_test, test_stats = prepare_data_for_model_improved(
        model=model,
        df=test_df,
        train_df=train_df,  # Usar treino para estatísticas
        target_col="target",
        allow_missing=True,
        output_dir=output_dir
    )
    
    # Calibrar modelo com a função melhorada
    calibrated_model, calibrated_threshold, calibration_metadata = calibrate_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        method=args.method,
        output_dir=output_dir
    )
    
    # Avaliar no conjunto de teste, se solicitado
    if args.evaluate_test or True:  # Sempre avaliar no teste
        # Avaliar modelo original no teste
        evaluate_on_test(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=threshold,
            model_name="RandomForest_Original",
            output_dir=output_dir
        )
        
        # Avaliar modelo calibrado no teste
        evaluate_on_test(
            model=calibrated_model,
            X_test=X_test,
            y_test=y_test,
            threshold=calibrated_threshold,
            model_name="RandomForest_Calibrated",
            output_dir=output_dir
        )
    
    # Salvar uma função auxiliar para carregar o modelo calibrado corretamente
    helper_code = """import joblib
import pandas as pd
import numpy as np

def load_calibrated_model(model_path):
    \"\"\"
    Carrega o modelo calibrado com metadata de features e threshold.
    
    Args:
        model_path: Caminho para o arquivo joblib com o modelo calibrado
        
    Returns:
        Tuple (model, feature_names, threshold)
    \"\"\"
    calibration_data = joblib.load(model_path)
    
    if isinstance(calibration_data, dict):
        model = calibration_data['model']
        feature_names = calibration_data['feature_names']
        threshold = calibration_data['threshold']
        print(f"Modelo carregado com {len(feature_names)} features e threshold {threshold:.4f}")
        return model, feature_names, threshold
    else:
        # Formato antigo (apenas o modelo)
        print("AVISO: Formato antigo de modelo detectado, sem metadata.")
        return calibration_data, None, None

def prepare_inference_data(df, feature_names, fill_missing=True):
    \"\"\"
    Prepara dados para inferência com o modelo calibrado.
    
    Args:
        df: DataFrame com os dados brutos
        feature_names: Lista de nomes de features esperadas pelo modelo
        fill_missing: Se True, preenche features ausentes com zeros
        
    Returns:
        DataFrame preparado para inferência
    \"\"\"
    # Converter colunas inteiras para float
    for col in df.columns:
        if col in feature_names and pd.api.types.is_integer_dtype(df[col].dtype):
            df[col] = df[col].astype(float)
    
    # Verificar features ausentes
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        if fill_missing:
            print(f"Preenchendo {len(missing_features)} features ausentes com zeros...")
            for feat in missing_features:
                df[feat] = 0
        else:
            raise ValueError(f"Faltam {len(missing_features)} features: {missing_features[:5]}...")
    
    # Remover features extras
    extra_features = [f for f in df.columns if f not in feature_names]
    if extra_features:
        print(f"Removendo {len(extra_features)} features extras...")
        df = df.drop(columns=extra_features)
    
    # Garantir ordenação exata das colunas
    df = df[feature_names]
    
    # Verificar e substituir NaNs
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"Encontrados valores NaN em {len(nan_cols)} colunas, preenchendo com zeros...")
        df = df.fillna(0)
    
    return df

def predict_with_calibrated_model(model, df, feature_names, threshold):
    \"\"\"
    Faz previsões usando o modelo calibrado.
    
    Args:
        model: Modelo calibrado
        df: DataFrame com os dados
        feature_names: Lista de nomes de features esperadas
        threshold: Threshold para classificação binária
        
    Returns:
        Tuple (predicted_class, probabilities)
    \"\"\"
    # Preparar dados
    X_pred = prepare_inference_data(df, feature_names)
    
    # Fazer previsões
    try:
        probas = model.predict_proba(X_pred)[:, 1]
    except Exception as e:
        print(f"Erro ao fazer previsões: {e}")
        # Tente descobrir qual é o problema
        if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
            # Verificar o primeiro classificador calibrado
            clf = model.calibrated_classifiers_[0]
            if hasattr(clf, 'estimator') and hasattr(clf.estimator, 'feature_names_in_'):
                expected = clf.estimator.feature_names_in_
                print(f"O modelo espera estas features: {expected[:5]}...")
                print(f"Features fornecidas: {X_pred.columns[:5]}...")
                # Compare as duas listas
                missing = [f for f in expected if f not in X_pred.columns]
                extra = [f for f in X_pred.columns if f not in expected]
                if missing:
                    print(f"Features ausentes: {missing[:5]}...")
                if extra:
                    print(f"Features extras: {extra[:5]}...")
        raise
    
    # Substituir NaNs se houver
    if np.isnan(probas).any():
        print(f"AVISO: {np.isnan(probas).sum()} valores NaN nas probabilidades")
        probas = np.nan_to_num(probas, nan=0.5)
    
    # Aplicar threshold
    predicted_class = (probas >= threshold).astype(int)
    
    return predicted_class, probas

# Exemplo de uso:
# model, feature_names, threshold = load_calibrated_model('caminho/para/modelo_calibrado.joblib')
# predictions, probabilities = predict_with_calibrated_model(model, novo_df, feature_names, threshold)
"""
    
    helper_path = os.path.join(output_dir, "calibrated_model_helper.py")
    with open(helper_path, 'w') as f:
        f.write(helper_code)
    
    print(f"\nFunção auxiliar para uso do modelo calibrado salva em: {helper_path}")
    
    print("\n=== Calibração e avaliação concluídas ===")
    print(f"Todos os resultados foram salvos em: {output_dir}")

if __name__ == "__main__":
    main()