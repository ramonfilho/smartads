#!/usr/bin/env python
"""
Script para calibrar as probabilidades do modelo GMM.
Baseado no script de calibração do RandomForest.
"""

import os
import sys
import argparse
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Caminhos absolutos para o GMM
BASE_DIR = "/Users/ramonmoreira/desktop/smart_ads"
DATA_DIR = "/Users/ramonmoreira/desktop/smart_ads/data/04_feature_engineering_2"
MODELS_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/artifacts/gmm_optimized"
VALIDATION_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# Threshold padrão (fallback)
DEFAULT_THRESHOLD = 0.15

class GMM_Wrapper:
    """
    Classe wrapper para o GMM que implementa a API sklearn para calibração.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pca_model = pipeline['pca_model']
        self.gmm_model = pipeline['gmm_model']
        self.scaler_model = pipeline['scaler_model']
        self.cluster_models = pipeline['cluster_models']
        self.n_clusters = pipeline['n_clusters']
        self.threshold = pipeline.get('threshold', DEFAULT_THRESHOLD)
        
        # Adicionar atributos necessários para a API sklearn
        self.classes_ = np.array([0, 1])  # Classes binárias
        self._fitted = True  # Marcar como já ajustado
        self._estimator_type = "classifier"  # Indicar explicitamente que é um classificador
        
    def fit(self, X, y):
        # Como o modelo já está treinado, apenas verificamos as classes
        self.classes_ = np.unique(y)
        self._fitted = True
        return self
        
    def predict_proba(self, X):
        # Preparar os dados para o modelo GMM
        X_numeric = X.select_dtypes(include=['number'])
        
        # Substituir valores NaN por 0
        X_numeric = X_numeric.fillna(0)
        
        # Aplicar o scaler
        if hasattr(self.scaler_model, 'feature_names_in_'):
            # Garantir que temos exatamente as features esperadas pelo scaler
            scaler_features = self.scaler_model.feature_names_in_
            
            # Remover features extras e adicionar as que faltam
            features_to_remove = [col for col in X_numeric.columns if col not in scaler_features]
            X_numeric = X_numeric.drop(columns=features_to_remove, errors='ignore')
            
            for col in scaler_features:
                if col not in X_numeric.columns:
                    X_numeric[col] = 0.0
            
            # Garantir a ordem correta das colunas
            X_numeric = X_numeric[scaler_features]
        
        # Verificar novamente por NaNs após o ajuste de colunas
        X_numeric = X_numeric.fillna(0)
        
        X_scaled = self.scaler_model.transform(X_numeric)
        
        # Verificar por NaNs no array após scaling
        if np.isnan(X_scaled).any():
            # Se ainda houver NaNs, substitua-os por zeros
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # Aplicar PCA
        X_pca = self.pca_model.transform(X_scaled)
        
        # Aplicar GMM para obter cluster labels e probabilidades
        cluster_labels = self.gmm_model.predict(X_pca)
        cluster_probs = self.gmm_model.predict_proba(X_pca)
        
        # Inicializar array de probabilidades
        n_samples = len(X)
        y_pred_proba = np.zeros((n_samples, 2), dtype=float)
        
        # Para cada cluster, fazer previsões
        for cluster_id, model_info in self.cluster_models.items():
            # Converter cluster_id para inteiro se for string
            cluster_id_int = int(cluster_id) if isinstance(cluster_id, str) else cluster_id
            
            # Selecionar amostras deste cluster
            cluster_mask = (cluster_labels == cluster_id_int)
            
            if not any(cluster_mask):
                continue
            
            # Obter modelo específico do cluster
            model = model_info['model']
            
            # Detectar features necessárias
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                
                # Criar um DataFrame temporário com as features corretas
                X_temp = X.copy()
                
                # Lidar com features ausentes ou extras
                missing_features = [col for col in expected_features if col not in X.columns]
                for col in missing_features:
                    X_temp[col] = 0.0
                
                # Garantir a ordem correta das colunas
                features_to_use = [col for col in expected_features if col in X_temp.columns]
                X_cluster = X_temp.loc[cluster_mask, features_to_use].astype(float)
                
                # Substituir NaNs por zeros
                X_cluster = X_cluster.fillna(0)
            else:
                # Usar todas as features numéricas disponíveis
                X_cluster = X.loc[cluster_mask].select_dtypes(include=['number']).fillna(0)
            
            if len(X_cluster) > 0:
                # Fazer previsões
                try:
                    proba = model.predict_proba(X_cluster)
                    
                    # Armazenar resultados
                    y_pred_proba[cluster_mask] = proba
                except Exception as e:
                    print(f"ERRO ao fazer previsões para o cluster {cluster_id_int}: {e}")
                    # Em caso de erro, usar probabilidades default
                    y_pred_proba[cluster_mask, 0] = 0.9  # classe negativa (majoritária)
                    y_pred_proba[cluster_mask, 1] = 0.1  # classe positiva (minoritária)
        
        return y_pred_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

def load_gmm_pipeline():
    """
    Carrega todos os componentes do pipeline de GMM.
    """
    print("Carregando componentes do pipeline GMM...")
    
    # Carregar modelos principais
    pca_model = joblib.load(os.path.join(MODELS_DIR, "pca_model.joblib"))
    gmm_model = joblib.load(os.path.join(MODELS_DIR, "gmm_model.joblib"))
    scaler_model = joblib.load(os.path.join(MODELS_DIR, "scaler_model.joblib"))
    
    # Carregar informações de avaliação e configuração
    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), 'r') as f:
        eval_results = json.load(f)
    
    with open(os.path.join(MODELS_DIR, "experiment_config.json"), 'r') as f:
        config = json.load(f)
    
    # Extrair informações relevantes
    n_clusters = config.get('gmm_params', {}).get('n_components', 3)
    threshold = eval_results.get('threshold', DEFAULT_THRESHOLD)
    
    print(f"Modelo GMM com {n_clusters} componentes")
    print(f"Threshold global: {threshold:.4f}")
    
    # Carregar estatísticas de clusters, se existirem
    train_cluster_stats = None
    val_cluster_stats = None
    try:
        train_cluster_stats = pd.read_json(os.path.join(MODELS_DIR, "train_cluster_stats.json"))
        val_cluster_stats = pd.read_json(os.path.join(MODELS_DIR, "val_cluster_stats.json"))
    except:
        print("Informações estatísticas dos clusters não encontradas")
    
    # Carregar modelos específicos de cada cluster
    cluster_models = {}
    
    for cluster_id in range(n_clusters):
        model_path = os.path.join(MODELS_DIR, f"cluster_{cluster_id}_model.joblib")
        
        try:
            model = joblib.load(model_path)
            
            # Definir threshold para este modelo
            cluster_models[cluster_id] = {
                'model': model,
                'threshold': threshold,
                'features': [],  # Será preenchido depois
                'info': {'threshold': threshold}
            }
            
            print(f"  Carregado modelo para cluster {cluster_id}, threshold: {threshold:.4f}")
        except FileNotFoundError as e:
            print(f"  AVISO: {e}")
            print(f"  Modelo para cluster {cluster_id} não encontrado. Este cluster será ignorado.")
    
    return {
        'pca_model': pca_model,
        'gmm_model': gmm_model,
        'scaler_model': scaler_model,
        'threshold': threshold,
        'config': config,
        'cluster_models': cluster_models,
        'n_clusters': n_clusters,
        'train_stats': train_cluster_stats,
        'val_stats': val_cluster_stats
    }

def load_datasets(validation_path=VALIDATION_PATH, test_path=TEST_PATH):
    """
    Carrega os datasets de validação e teste.
    
    Args:
        validation_path: Caminho para o dataset de validação
        test_path: Caminho para o dataset de teste
        
    Returns:
        Tuple (val_df, test_df)
    """
    # Verificar se os arquivos existem
    if not os.path.exists(validation_path):
        raise FileNotFoundError(f"Arquivo de validação não encontrado: {validation_path}")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {test_path}")
    
    # Carregar os dados
    print(f"Carregando dados de validação: {validation_path}")
    val_df = pd.read_csv(validation_path)
    
    print(f"Carregando dados de teste: {test_path}")
    test_df = pd.read_csv(test_path)
    
    print(f"Dados carregados: validação {val_df.shape}, teste {test_df.shape}")
    return val_df, test_df

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

def find_optimal_threshold(y_true, y_pred_proba, threshold_range=None):
    """
    Encontra o threshold ótimo baseado no F1-score.
    
    Args:
        y_true: Valores reais
        y_pred_proba: Probabilidades preditas
        threshold_range: Range de thresholds para testar
        
    Returns:
        Dicionário com threshold ótimo e métricas
    """
    # Adicionar a importação necessária
    from sklearn.metrics import precision_recall_fscore_support
    
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
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_t, average='binary', zero_division=0
        )
        
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

def calibrate_model(gmm_wrapper, X_val, y_val, method='isotonic', output_dir=None):
    """
    Calibra as probabilidades do modelo usando validação cruzada.
    """
    print(f"\nCalibrando o modelo GMM com o método '{method}'...")
    
    # Criar diretório para resultados
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(BASE_DIR, "models", "calibrated", f"gmm_calibrated_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {output_dir}")
    
    # Certificar-se de que o wrapper está "ajustado"
    gmm_wrapper.fit(X_val, y_val)
    
    # Obter probabilidades antes da calibração
    proba_before = gmm_wrapper.predict_proba(X_val)[:, 1]
    
    # Calibrar o modelo
    calibrated_model = CalibratedClassifierCV(
        estimator=gmm_wrapper,
        method=method,
        cv='prefit'  # O modelo já está ajustado
    )
    
    # Ajustar o calibrador
    print("Ajustando o calibrador ao conjunto de validação...")
    calibrated_model.fit(X_val, y_val)
    
    # Obter probabilidades depois da calibração
    proba_after = calibrated_model.predict_proba(X_val)[:, 1]
    
    # Plotar curvas de calibração
    plot_calibration_curve(y_val, proba_before, "GMM (Não Calibrado)", output_dir)
    plot_calibration_curve(y_val, proba_after, "GMM (Calibrado)", output_dir)
    
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
        y_val, proba_after, best_threshold, "GMM (Calibrado)", output_dir
    )
    
    # Salvar modelo calibrado
    model_path = os.path.join(output_dir, "gmm_calibrated.joblib")
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
        'precision': precision_score(y_val, y_pred_before, zero_division=0),
        'recall': recall_score(y_val, y_pred_before, zero_division=0),
        'f1': f1_score(y_val, y_pred_before, zero_division=0),
        'threshold': best_threshold
    }
    
    metrics_after = {
        'precision': precision_score(y_val, y_pred_after, zero_division=0),
        'recall': recall_score(y_val, y_pred_after, zero_division=0),
        'f1': f1_score(y_val, y_pred_after, zero_division=0),
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

def evaluate_on_test(model, X_test, y_test, threshold, model_name="GMM", output_dir=None):
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
    
    # Fazer previsões
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calcular métricas
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
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
    parser = argparse.ArgumentParser(description='Calibrar e avaliar o modelo GMM')
    parser.add_argument('--method', default='isotonic', choices=['isotonic', 'sigmoid'],
                      help='Método de calibração (padrão: isotonic)')
    parser.add_argument('--evaluate_test', action='store_true',
                      help='Avaliar no conjunto de teste após calibração')
    
    args = parser.parse_args()
    
    print("=== Iniciando calibração do modelo GMM ===")
    
    # Carregar o pipeline GMM
    try:
        pipeline = load_gmm_pipeline()
        print("Pipeline GMM carregado com sucesso!")
        
        # Verificar componentes principais
        if not all(k in pipeline for k in ['pca_model', 'gmm_model', 'scaler_model', 'cluster_models']):
            raise ValueError("Pipeline GMM incompleto. Verifique se todos os componentes foram carregados.")
        
        # Criar wrapper para o modelo GMM
        gmm_wrapper = GMM_Wrapper(pipeline)
        
    except Exception as e:
        print(f"ERRO ao carregar o pipeline GMM: {e}")
        sys.exit(1)
    
    # Carregar datasets
    try:
        val_df, test_df = load_datasets()
    except FileNotFoundError as e:
        print(f"ERRO: {str(e)}")
        sys.exit(1)
    
    # Preparar dados
    X_val = val_df.drop(columns=['target'])
    y_val = val_df['target']
    
    # Calibrar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, "models", "calibrated", f"gmm_calibrated_{timestamp}")
    
    calibrated_model, calibrated_threshold = calibrate_model(
        gmm_wrapper=gmm_wrapper,
        X_val=X_val,
        y_val=y_val,
        method=args.method,
        output_dir=output_dir
    )
    
    # Avaliar no conjunto de teste, se solicitado
    if args.evaluate_test or True:  # Sempre avaliar no teste
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        # Avaliar modelo original no teste
        evaluate_on_test(
            model=gmm_wrapper,
            X_test=X_test,
            y_test=y_test,
            threshold=pipeline['threshold'],
            model_name="GMM_Original",
            output_dir=output_dir
        )
        
        # Avaliar modelo calibrado no teste
        evaluate_on_test(
            model=calibrated_model,
            X_test=X_test,
            y_test=y_test,
            threshold=calibrated_threshold,
            model_name="GMM_Calibrated",
            output_dir=output_dir
        )
    
    print("\n=== Calibração e avaliação do GMM concluídas ===")
    print(f"Todos os resultados foram salvos em: {output_dir}")

if __name__ == "__main__":
    main()