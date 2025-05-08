#!/usr/bin/env python
"""
Script para calibrar as probabilidades do modelo RandomForest.
Baseado no script 06_error_analysis.py.
"""

import os
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
MODEL_PATH = "/Users/ramonmoreira/desktop/smart_ads/models/mlflow/783044341530730386/2c54d70c822c4e42ad92313f4f2bfe8e/artifacts/random_forest"

# Caminhos para os datasets
TRAIN_PATH = "/Users/ramonmoreira/desktop/smart_ads/data/03_4_feature_selection_final/train.csv"
VAL_PATH = "/Users/ramonmoreira/desktop/smart_ads/data/03_4_feature_selection_final/validation.csv"
TEST_PATH = "/Users/ramonmoreira/desktop/smart_ads/data/03_4_feature_selection_final/test.csv"

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

def prepare_data_for_model(model, df, target_col="target"):
    """
    Prepara os dados para serem compatíveis com o modelo.
    
    Args:
        model: Modelo treinado
        df: DataFrame com os dados
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

def calibrate_model(model, X_val, y_val, method='isotonic', output_dir=None):
    """
    Calibra as probabilidades do modelo usando validação cruzada.
    
    Args:
        model: Modelo não calibrado
        X_val: Features de validação
        y_val: Target de validação
        method: Método de calibração ('isotonic' ou 'sigmoid')
        output_dir: Diretório para salvar resultados
        
    Returns:
        Modelo calibrado
    """
    print(f"\nCalibrando o modelo com o método '{method}'...")
    
    # Criar diretório para resultados se não especificado
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "models", "calibrated", f"rf_calibrated_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {output_dir}")
    
    # Calibrar o modelo - versão atualizada para compatibilidade
    calibrated_model = CalibratedClassifierCV(
        estimator=model,  # Usando 'estimator' ao invés de 'base_estimator'
        method=method,
        cv='prefit'  # O modelo já está ajustado, não faça validação cruzada
    )
    
    # Ajustar o calibrador às probabilidades existentes
    calibrated_model.fit(X_val, y_val)
    
    # Obter probabilidades antes e depois da calibração
    proba_before = model.predict_proba(X_val)[:, 1]
    proba_after = calibrated_model.predict_proba(X_val)[:, 1]
    
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
    
    # Salvar modelo calibrado
    model_path = os.path.join(output_dir, "rf_calibrated.joblib")
    joblib.dump(calibrated_model, model_path)
    print(f"Modelo calibrado salvo em: {model_path}")
    
    # Salvar threshold ótimo
    threshold_path = os.path.join(output_dir, "threshold.txt")
    with open(threshold_path, 'w') as f:
        f.write(str(best_threshold))
    print(f"Threshold salvo em: {threshold_path}")
    
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
    
    return calibrated_model, best_threshold

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
    
    # Salvar métricas
    metrics = {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
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
    
    # Preparar dados
    X_val, y_val = prepare_data_for_model(model, val_df)
    
    # Calibrar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "models", "calibrated", f"rf_calibrated_{timestamp}")
    
    calibrated_model, calibrated_threshold = calibrate_model(
        model=model,
        X_val=X_val,
        y_val=y_val,
        method=args.method,
        output_dir=output_dir
    )
    
    # Avaliar no conjunto de teste, se solicitado
    if args.evaluate_test or True:  # Sempre avaliar no teste
        X_test, y_test = prepare_data_for_model(model, test_df)
        
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
    
    print("\n=== Calibração e avaliação concluídas ===")
    print(f"Todos os resultados foram salvos em: {output_dir}")

if __name__ == "__main__":
    main()