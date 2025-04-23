#!/usr/bin/env python
"""
Script para implementação de calibração de probabilidades e ajuste de threshold otimizado.
Baseado nos resultados da análise de falsos negativos, este script:
1. Implementa calibração isotônica para corrigir a subestimação de probabilidades
2. Otimiza o threshold para balancear precisão e recall
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
import mlflow
import warnings
warnings.filterwarnings('ignore')

# Configuração de caminhos principais
PROJECT_ROOT = "/Users/ramonmoreira/Desktop/smart_ads"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "03_feature_selection_textv2")
MLFLOW_DIR = os.path.join(PROJECT_ROOT, "models", "mlflow")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "validation.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", f"calibrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

def setup_mlflow(mlflow_dir, experiment_name="smart_ads_calibrated"):
    """Configura o MLflow para tracking de experimentos."""
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    print(f"MLflow configurado para usar: {mlflow.get_tracking_uri()}")
    
    # Verificar se o experimento já existe
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment:
        print(f"Experimento '{experiment_name}' já existe (ID: {experiment.experiment_id})")
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experimento '{experiment_name}' criado (ID: {experiment_id})")
    
    return experiment_id

def load_model_from_mlflow(model_uri):
    """Carrega um modelo a partir do MLflow usando seu URI."""
    try:
        print(f"Carregando modelo de: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        return None

def find_latest_model_uri():
    """Localiza o URI do modelo RandomForest mais recente no MLflow."""
    client = mlflow.tracking.MlflowClient()
    
    # Buscar todos os experimentos
    experiments = client.search_experiments()
    
    for experiment in experiments:
        print(f"Verificando experimento: {experiment.name} (ID: {experiment.experiment_id})")
        
        # Buscar runs de RandomForest, mais recentes primeiro
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_type = 'random_forest'",
            order_by=["attribute.start_time DESC"]
        )
        
        if runs:
            run = runs[0]  # Pegar o run mais recente
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/random_forest"
            
            print(f"Encontrado modelo RandomForest mais recente (Run ID: {run_id})")
            print(f"Model URI: {model_uri}")
            
            # Tentar obter threshold do MLflow
            threshold = run.data.metrics.get('threshold', 0.12)  # Fallback para 0.12
            
            return model_uri, threshold, run_id
    
    print("Nenhum modelo RandomForest encontrado.")
    return None, None, None

def load_data(data_path, target_col='target'):
    """Carrega e prepara os dados para análise."""
    print(f"Carregando dados de: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"ERRO: Arquivo não encontrado: {data_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
    
    # Carregar DataFrame
    df = pd.read_csv(data_path)
    print(f"Dataset carregado: {df.shape[0]} exemplos, {df.shape[1]} colunas")
    
    # Identificar a coluna target se não especificada
    if target_col not in df.columns:
        target_candidates = ['target', 'label', 'class', 'y']
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col not in df.columns:
            print("Aviso: Coluna target não encontrada. Assumindo última coluna.")
            target_col = df.columns[-1]
    
    # Separar features e target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Sanitizar nomes de colunas - método simplificado
    X.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in X.columns]
    
    # Garantir que todas as colunas numéricas sejam float
    for col in X.select_dtypes(include=['int']).columns:
        X[col] = X[col].astype(float)
    
    print(f"Distribuição do target: {y.mean()*100:.2f}% positivos")
    
    return df, X, y, target_col

def calibrate_model(base_model, X_train, y_train, method='isotonic'):
    """Calibra as probabilidades do modelo usando o método especificado."""
    print(f"\n=== Calibrando modelo usando método '{method}' ===")
    
    if method == 'direct_isotonic':
        # Implementação direta de calibração isotônica (sem CalibratedClassifierCV)
        # Gera probabilidades brutas com o modelo base
        y_prob = base_model.predict_proba(X_train)[:, 1]
        
        # Treina um modelo de regressão isotônica para calibração
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_prob, y_train)
        
        # Cria um modelo calibrado através de closure
        def calibrated_predict_proba(X):
            # Obter probabilidades do modelo base
            base_probs = base_model.predict_proba(X)
            # Calibrar a coluna de probabilidade positiva
            calibrated_pos_probs = calibrator.predict(base_probs[:, 1])
            # Reconstruir o array de probabilidades 2D
            calibrated_probs = np.zeros_like(base_probs)
            calibrated_probs[:, 1] = calibrated_pos_probs
            calibrated_probs[:, 0] = 1 - calibrated_pos_probs
            return calibrated_probs
        
        # Criar modelo "calibrado" com mesmos métodos que o original
        calibrated_model = base_model
        calibrated_model.predict_proba_original = calibrated_model.predict_proba
        calibrated_model.predict_proba = calibrated_predict_proba
        
        # Armazenar o calibrador para uso posterior
        calibrated_model.calibrator = calibrator
        
    else:
        # Usar CalibratedClassifierCV do scikit-learn
        # Dica: cv='prefit' significa que não será feito novo treino do modelo base
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv='prefit'
        )
        
        # Treinar apenas o calibrador
        calibrated_model.fit(X_train, y_train)
    
    print("Calibração concluída com sucesso")
    return calibrated_model

def analyze_calibration(model, X, y, output_dir, title="Curva de Calibração"):
    """Analisa a qualidade da calibração do modelo."""
    # Criar diretório de saída se necessário
    os.makedirs(output_dir, exist_ok=True)
    
    # Gerar probabilidades
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calcular curva de calibração
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    
    # Calcular MSE da calibração
    calibration_mse = np.mean((prob_true - prob_pred)**2)
    
    # Verificar se há subestimação ou superestimação
    underestimation = np.mean(prob_true > prob_pred)
    
    # Determinar se o modelo precisa de calibração
    needs_calibration = calibration_mse > 0.01
    
    # Plotar curva de calibração
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(prob_pred, prob_true, 's-', label=f'Calibração (MSE={calibration_mse:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfeitamente Calibrado')
    plt.title(title)
    plt.xlabel('Probabilidade Média Predita')
    plt.ylabel('Fração de Positivos Observados')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Histograma de probabilidades
    plt.subplot(2, 1, 2)
    
    # Histograma para cada classe
    plt.hist(y_prob[y == 0], bins=20, alpha=0.5, density=True, label='Classe Negativa')
    plt.hist(y_prob[y == 1], bins=20, alpha=0.5, density=True, label='Classe Positiva')
    
    plt.xlabel('Probabilidade Predita')
    plt.ylabel('Densidade')
    plt.title('Histograma de Probabilidades por Classe')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar figura
    cal_path = os.path.join(output_dir, f'calibration_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(cal_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAnálise de calibração:")
    if needs_calibration:
        if underestimation > 0.5:
            status = "subestimadas"
        else:
            status = "superestimadas"
        print(f"- O modelo apresenta problemas de calibração (MSE: {calibration_mse:.4f})")
        print(f"- As probabilidades estão {status}")
    else:
        print(f"- O modelo está bem calibrado (MSE: {calibration_mse:.4f})")
    
    return {
        'calibration_mse': calibration_mse,
        'underestimation': underestimation,
        'needs_calibration': needs_calibration,
        'plot_path': cal_path
    }

def optimize_threshold(y_true, y_prob, min_precision=0.75, output_dir=None):
    """
    Encontra o threshold ótimo para maximizar recall mantendo precisão mínima.
    
    Args:
        y_true: Labels verdadeiros
        y_prob: Probabilidades preditas
        min_precision: Precisão mínima aceitável (default: 0.75)
        output_dir: Diretório para salvar gráficos
    
    Returns:
        Dict com thresholds e métricas
    """
    print(f"\n=== Otimizando threshold (precisão mínima: {min_precision:.2f}) ===")
    
    # Calcular curva precision-recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Adicionar threshold=1.0 para compatibilidade de tamanho
    thresholds = np.append(thresholds, [1.0])
    
    # Criar DataFrame de métricas
    metrics_df = pd.DataFrame({
        'threshold': thresholds,
        'precision': precision,
        'recall': recall
    })
    
    # Calcular F1 para cada ponto
    metrics_df['f1'] = 2 * (metrics_df['precision'] * metrics_df['recall']) / (metrics_df['precision'] + metrics_df['recall'] + 1e-10)
    
    # Encontrar threshold para máximo F1
    idx_max_f1 = metrics_df['f1'].idxmax()
    best_f1_threshold = metrics_df.loc[idx_max_f1, 'threshold']
    best_f1 = metrics_df.loc[idx_max_f1, 'f1']
    best_f1_precision = metrics_df.loc[idx_max_f1, 'precision']
    best_f1_recall = metrics_df.loc[idx_max_f1, 'recall']
    
    # Encontrar o threshold que maximiza recall mantendo precisão mínima
    high_precision_df = metrics_df[metrics_df['precision'] >= min_precision]
    
    if len(high_precision_df) > 0:
        idx_max_recall = high_precision_df['recall'].idxmax()
        optimal_threshold = high_precision_df.loc[idx_max_recall, 'threshold']
        optimal_precision = high_precision_df.loc[idx_max_recall, 'precision']
        optimal_recall = high_precision_df.loc[idx_max_recall, 'recall']
        optimal_f1 = high_precision_df.loc[idx_max_recall, 'f1']
        
        status = "encontrado"
    else:
        print(f"Aviso: Não foi possível encontrar threshold com precisão >= {min_precision:.2f}")
        print(f"Usando threshold de máximo F1 como fallback")
        
        optimal_threshold = best_f1_threshold
        optimal_precision = best_f1_precision
        optimal_recall = best_f1_recall
        optimal_f1 = best_f1
        
        status = "não encontrado, usando máximo F1"
    
    print(f"Threshold para máximo F1: {best_f1_threshold:.4f} (P={best_f1_precision:.4f}, R={best_f1_recall:.4f}, F1={best_f1:.4f})")
    print(f"Threshold ótimo para precisão >= {min_precision:.2f}: {optimal_threshold:.4f} ({status})")
    print(f"  - Precisão: {optimal_precision:.4f}")
    print(f"  - Recall: {optimal_recall:.4f}")
    print(f"  - F1-Score: {optimal_f1:.4f}")
    
    # Salvar gráficos se o diretório for fornecido
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plotar curva PR
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.scatter([best_f1_recall], [best_f1_precision], color='red', marker='o',
                   label=f'Max F1: {best_f1:.3f} (t={best_f1_threshold:.3f})')
        plt.scatter([optimal_recall], [optimal_precision], color='green', marker='*', s=200,
                   label=f'Optimal (P>={min_precision:.2f}): (P={optimal_precision:.3f}, R={optimal_recall:.3f})')
        
        plt.axhline(y=min_precision, color='r', linestyle='--', alpha=0.3, label=f'Min Precision: {min_precision:.2f}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        pr_path = os.path.join(output_dir, f'pr_curve_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(pr_path, dpi=300)
        plt.close()
        
        # Plotar threshold vs. métricas
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision')
        plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall')
        plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1')
        
        plt.axvline(x=best_f1_threshold, color='red', linestyle='--', 
                   label=f'Max F1 (t={best_f1_threshold:.3f})')
        plt.axvline(x=optimal_threshold, color='green', linestyle='--', 
                   label=f'Optimal (t={optimal_threshold:.3f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold vs. Metrics')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        thresh_path = os.path.join(output_dir, f'threshold_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(thresh_path, dpi=300)
        plt.close()
    
    return {
        'best_f1': {
            'threshold': best_f1_threshold,
            'precision': best_f1_precision,
            'recall': best_f1_recall,
            'f1': best_f1
        },
        'optimal': {
            'threshold': optimal_threshold,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1': optimal_f1,
            'status': status
        },
        'metrics_df': metrics_df
    }

def evaluate_model(model, X, y, threshold=0.5, name="Model"):
    """Avalia o modelo usando as métricas padrão."""
    # Gerar probabilidades
    y_prob = model.predict_proba(X)[:, 1]
    
    # Aplicar threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calcular métricas
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Contar predições
    n_positives = y_pred.sum()
    pos_rate = n_positives / len(y_pred)
    
    # Contagem de erro
    true_pos = ((y == 1) & (y_pred == 1)).sum()
    false_pos = ((y == 0) & (y_pred == 1)).sum()
    false_neg = ((y == 1) & (y_pred == 0)).sum()
    true_neg = ((y == 0) & (y_pred == 0)).sum()
    
    print(f"\nAvaliação de {name} (threshold={threshold:.4f}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Positivos: {n_positives} ({pos_rate:.2%})")
    print(f"  Matriz de confusão:")
    print(f"    TP: {true_pos}, FP: {false_pos}")
    print(f"    FN: {false_neg}, TN: {true_neg}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'positive_count': n_positives,
        'true_pos': true_pos,
        'false_pos': false_pos,
        'false_neg': false_neg,
        'true_neg': true_neg
    }

def save_model(model, output_dir, model_name="calibrated_model"):
    """Salva o modelo e metadados relacionados."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar o modelo
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    print(f"\nModelo salvo em: {model_path}")
    return model_path

def predict_with_calibration_and_threshold(model, X, threshold=0.5):
    """Gera predições usando o modelo calibrado e threshold otimizado."""
    # Obter probabilidades calibradas
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        # Para casos em que estamos usando um modelo envelopado manualmente
        y_prob = model.predict_proba_calibrated(X)[:, 1]
    
    # Aplicar threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_pred, y_prob

def validate_model_is_not_calibrated(model):
    """Verifica se o modelo não é um modelo já calibrado."""
    has_calibration = False
    
    # Verificar atributos que indicam calibração
    if hasattr(model, 'calibrated_classifiers_'):
        has_calibration = True
    
    # Verificar se existe um método predict_proba_original
    if hasattr(model, 'predict_proba_original'):
        has_calibration = True
    
    # Verificar se o nome da classe contém "Calibrated"
    if 'Calibrated' in model.__class__.__name__:
        has_calibration = True
    
    return not has_calibration

def main():
    # Configurar argumentos
    parser = argparse.ArgumentParser(description='Calibração de probabilidades e ajuste de threshold')
    parser.add_argument('--mlflow_dir', default=MLFLOW_DIR, help='Diretório MLflow')
    parser.add_argument('--model_uri', default=None, help='URI do modelo a ser calibrado')
    parser.add_argument('--data_path', default=DEFAULT_DATA_PATH, help='Caminho para dados de validação')
    parser.add_argument('--train_path', default=TRAIN_DATA_PATH, help='Caminho para dados de treino')
    parser.add_argument('--min_precision', type=float, default=0.80, help='Precisão mínima desejada')
    parser.add_argument('--calibration', choices=['isotonic', 'sigmoid', 'direct_isotonic'], default='isotonic',
                        help='Método de calibração')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    
    # Verificar se o modelo não está já calibrado
    if not validate_model_is_not_calibrated(base_model):
        ("AVISO: O modelo carregado parece já estar calibrado!")
    proceed = input("Deseja continuar mesmo assim? (s/n): ")
    if proceed.lower() != 's':
        print("Operação cancelada pelo usuário.")
        return

    # Configurar MLflow
    experiment_id = setup_mlflow(args.mlflow_dir)
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="calibration_threshold_optimization") as run:
        # Registrar parâmetros
        mlflow.log_params({
            'calibration_method': args.calibration,
            'min_precision': args.min_precision,
            'data_path': args.data_path,
            'train_path': args.train_path
        })
        
        # Carregar modelo base
        if args.model_uri is None:
            model_uri, baseline_threshold, _ = find_latest_model_uri()
            if model_uri is None:
                print("ERRO: Nenhum modelo encontrado. Especifique o URI via --model_uri")
                return
        else:
            model_uri = args.model_uri
            baseline_threshold = 0.12  # Valor padrão para caso não seja encontrado
        
        # Carregar o modelo
        base_model = load_model_from_mlflow(model_uri)
        if base_model is None:
            print("ERRO: Falha ao carregar o modelo.")
            return
        
        # Registrar URI do modelo base
        mlflow.log_param('base_model_uri', model_uri)
        mlflow.log_param('baseline_threshold', baseline_threshold)
        
        # Carregar dados de validação
        _, X_val, y_val, _ = load_data(args.data_path)
        
        # Avaliar modelo base com threshold original
        print("\n=== Avaliando modelo base ===")
        baseline_results = evaluate_model(
            base_model, X_val, y_val, 
            threshold=baseline_threshold, 
            name="Modelo Base"
        )
        
        # Analisar calibração do modelo base
        print("\n=== Analisando calibração do modelo base ===")
        base_calibration = analyze_calibration(
            base_model, X_val, y_val, 
            output_dir=args.output_dir,
            title="Calibração do Modelo Base"
        )
        
        # Registrar métricas do modelo base
        for key, value in baseline_results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f'baseline_{key}', value)
        
        # Carregar dados de treino para calibração
        print("\n=== Carregando dados de treino para calibração ===")
        _, X_train, y_train, _ = load_data(args.train_path)
        
        # Calibrar modelo se necessário
        if base_calibration['needs_calibration']:
            print("\n=== Calibrando modelo ===")
            calibrated_model = calibrate_model(
                base_model, X_train, y_train, 
                method=args.calibration
            )
            
            # Analisar calibração do modelo calibrado
            print("\n=== Verificando calibração do modelo calibrado ===")
            cal_results = analyze_calibration(
                calibrated_model, X_val, y_val, 
                output_dir=args.output_dir,
                title=f"Calibração Após {args.calibration.title()}"
            )
            
            # Registrar artefato de calibração
            mlflow.log_artifact(cal_results['plot_path'])
            
            # Avaliar modelo calibrado com threshold original
            print("\n=== Avaliando modelo calibrado com threshold original ===")
            calibrated_base_results = evaluate_model(
                calibrated_model, X_val, y_val, 
                threshold=baseline_threshold, 
                name=f"Modelo Calibrado (threshold original)"
            )
            
            # Otimizar threshold para modelo calibrado
            print("\n=== Otimizando threshold para modelo calibrado ===")
            y_prob_cal = calibrated_model.predict_proba(X_val)[:, 1]
            
            threshold_results = optimize_threshold(
                y_val, y_prob_cal,
                min_precision=args.min_precision,
                output_dir=args.output_dir
            )
            
            # Avaliar modelo calibrado com threshold otimizado
            optimal_threshold = threshold_results['optimal']['threshold']
            
            calibrated_opt_results = evaluate_model(
                calibrated_model, X_val, y_val, 
                threshold=optimal_threshold, 
                name=f"Modelo Calibrado (threshold otimizado)"
            )
            
            # Registrar métricas
            for key, value in calibrated_opt_results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f'calibrated_{key}', value)
            
            # Salvar modelo calibrado
            model_path = save_model(
                calibrated_model, 
                output_dir=args.output_dir,
                model_name="calibrated_model"
            )
            
            # Salvar também o threshold otimizado
            threshold_path = os.path.join(args.output_dir, "optimal_threshold.txt")
            with open(threshold_path, 'w') as f:
                f.write(f"{optimal_threshold}")
            
            print(f"Threshold otimizado salvo em: {threshold_path}")
            
            # Registrar modelo no MLflow
            mlflow.sklearn.log_model(calibrated_model, "calibrated_model")
            
            # Gravar resumo de resultados
            summary = {
                'baseline': baseline_results,
                'calibrated_base_threshold': calibrated_base_results,
                'calibrated_optimal_threshold': calibrated_opt_results,
                'calibration': {
                    'method': args.calibration,
                    'base_mse': base_calibration['calibration_mse'],
                    'calibrated_mse': cal_results['calibration_mse'],
                },
                'threshold': {
                    'baseline': baseline_threshold,
                    'optimal': optimal_threshold,
                    'min_precision': args.min_precision
                }
            }
            
            summary_path = os.path.join(args.output_dir, "results_summary.json")
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nResumão de resultados salvo em: {summary_path}")
            
            print("\n=== Comparação de Resultados ===")
            print(f"Modelo Base (t={baseline_threshold:.4f}):")
            print(f"  - Precision: {baseline_results['precision']:.4f}")
            print(f"  - Recall: {baseline_results['recall']:.4f}")
            print(f"  - F1: {baseline_results['f1']:.4f}")
            
            print(f"\nModelo Calibrado (t={baseline_threshold:.4f}):")
            print(f"  - Precision: {calibrated_base_results['precision']:.4f}")
            print(f"  - Recall: {calibrated_base_results['recall']:.4f}")
            print(f"  - F1: {calibrated_base_results['f1']:.4f}")
            
            print(f"\nModelo Calibrado com threshold otimizado (t={optimal_threshold:.4f}):")
            print(f"  - Precision: {calibrated_opt_results['precision']:.4f}")
            print(f"  - Recall: {calibrated_opt_results['recall']:.4f}")
            print(f"  - F1: {calibrated_opt_results['f1']:.4f}")
            
            # Calcular melhorias
            recall_improvement = (calibrated_opt_results['recall'] / baseline_results['recall'] - 1) * 100
            precision_change = (calibrated_opt_results['precision'] / baseline_results['precision'] - 1) * 100
            f1_improvement = (calibrated_opt_results['f1'] / baseline_results['f1'] - 1) * 100
            
            print(f"\nMelhorias em relação ao modelo base:")
            print(f"  - Recall: {recall_improvement:+.1f}%")
            print(f"  - Precisão: {precision_change:+.1f}%")
            print(f"  - F1: {f1_improvement:+.1f}%")
            
        else:
            print("\nModelo base já está bem calibrado. Prosseguindo apenas com otimização de threshold.")
            
            # Otimizar threshold para modelo base
            y_prob_base = base_model.predict_proba(X_val)[:, 1]
            
            threshold_results = optimize_threshold(
                y_val, y_prob_base,
                min_precision=args.min_precision,
                output_dir=args.output_dir
            )
            
            # Avaliar modelo base com threshold otimizado
            optimal_threshold = threshold_results['optimal']['threshold']
            
            base_opt_results = evaluate_model(
                base_model, X_val, y_val, 
                threshold=optimal_threshold, 
                name=f"Modelo Base (threshold otimizado)"
            )
            
            # Registrar métricas
            for key, value in base_opt_results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f'base_optimized_{key}', value)
            
            # Salvar modelo base e threshold
            model_path = save_model(
                base_model, 
                output_dir=args.output_dir,
                model_name="base_model_with_optimal_threshold"
            )
            
            # Salvar threshold otimizado
            threshold_path = os.path.join(args.output_dir, "optimal_threshold.txt")
            with open(threshold_path, 'w') as f:
                f.write(f"{optimal_threshold}")
            
            print(f"Threshold otimizado salvo em: {threshold_path}")
            
            # Gravar resumo de resultados
            summary = {
                'baseline': baseline_results,
                'base_optimal_threshold': base_opt_results,
                'calibration': {
                    'base_mse': base_calibration['calibration_mse'],
                    'needs_calibration': False
                },
                'threshold': {
                    'baseline': baseline_threshold,
                    'optimal': optimal_threshold,
                    'min_precision': args.min_precision
                }
            }
            
            summary_path = os.path.join(args.output_dir, "results_summary.json")
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nResumão de resultados salvo em: {summary_path}")
            
            print("\n=== Comparação de Resultados ===")
            print(f"Modelo Base (t={baseline_threshold:.4f}):")
            print(f"  - Precision: {baseline_results['precision']:.4f}")
            print(f"  - Recall: {baseline_results['recall']:.4f}")
            print(f"  - F1: {baseline_results['f1']:.4f}")
            
            print(f"\nModelo Base com threshold otimizado (t={optimal_threshold:.4f}):")
            print(f"  - Precision: {base_opt_results['precision']:.4f}")
            print(f"  - Recall: {base_opt_results['recall']:.4f}")
            print(f"  - F1: {base_opt_results['f1']:.4f}")
            
            # Calcular melhorias
            recall_improvement = (base_opt_results['recall'] / baseline_results['recall'] - 1) * 100
            precision_change = (base_opt_results['precision'] / baseline_results['precision'] - 1) * 100
            f1_improvement = (base_opt_results['f1'] / baseline_results['f1'] - 1) * 100
            
            print(f"\nMelhorias em relação ao modelo base:")
            print(f"  - Recall: {recall_improvement:+.1f}%")
            print(f"  - Precisão: {precision_change:+.1f}%")
            print(f"  - F1: {f1_improvement:+.1f}%")

if __name__ == "__main__":
    main()