#!/usr/bin/env python
"""
Script simplificado para avaliar a calibração e performance dos modelos RF e GMM calibrados no conjunto de teste.
"""

import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

# Caminhos absolutos
BASE_DIR = "/Users/ramonmoreira/desktop/smart_ads"
DATA_DIR_GMM = "/Users/ramonmoreira/desktop/smart_ads/data/02_3_processed_text_code6"
DATA_DIR_RF = "/Users/ramonmoreira/desktop/smart_ads/data/03_4_feature_selection_final"
TEST_PATH_GMM = os.path.join(DATA_DIR_GMM, "test.csv")
TEST_PATH_RF = os.path.join(DATA_DIR_RF, "test.csv")

# Caminhos para os modelos calibrados
RF_CALIB_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/rf_calibrated_20250508_132208"
GMM_CALIB_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250508_130725"

# Diretório para salvar resultados
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "calibration_validation")

# Classe GMM_Wrapper necessária para carregar o modelo GMM
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
        self.threshold = pipeline.get('threshold', 0.15)
        
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
            
            # Identificar features em X_numeric que não estão no scaler
            unseen_features = [col for col in X_numeric.columns if col not in scaler_features]
            if unseen_features:
                print(f"Removendo {len(unseen_features)} features não vistas durante treinamento")
                X_numeric = X_numeric.drop(columns=unseen_features)
            
            # Identificar features que faltam em X_numeric mas estão no scaler
            missing_features = [col for col in scaler_features if col not in X_numeric.columns]
            if missing_features:
                print(f"Adicionando {len(missing_features)} features ausentes vistas durante treinamento")
                for col in missing_features:
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
            
            # Detectar quais features o modelo espera
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

def prepare_data_for_rf_model(model, df, target_col="target"):
    """
    Prepara os dados para serem compatíveis com o modelo RandomForest.
    Adaptado do arquivo 08_prob_calib_rf.py.
    """
    # Extrair features e target
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    y = df[target_col] if target_col in df.columns else None
    
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

def load_models():
    """
    Carrega os modelos calibrados.
    """
    print("Carregando modelos calibrados...")
    models = {}
    
    # Random Forest Calibrado
    rf_model_path = os.path.join(RF_CALIB_DIR, "rf_calibrated.joblib")
    rf_threshold_path = os.path.join(RF_CALIB_DIR, "threshold.txt")
    
    try:
        rf_model = joblib.load(rf_model_path)
        with open(rf_threshold_path, 'r') as f:
            rf_threshold = float(f.read().strip())
        print(f"Random Forest calibrado carregado com threshold: {rf_threshold:.4f}")
        models['RF'] = {'model': rf_model, 'threshold': rf_threshold}
    except Exception as e:
        print(f"Erro ao carregar Random Forest: {e}")
    
    # GMM Calibrado
    gmm_model_path = os.path.join(GMM_CALIB_DIR, "gmm_calibrated.joblib")
    gmm_threshold_path = os.path.join(GMM_CALIB_DIR, "threshold.txt")
    
    try:
        gmm_model = joblib.load(gmm_model_path)
        with open(gmm_threshold_path, 'r') as f:
            gmm_threshold = float(f.read().strip())
        print(f"GMM calibrado carregado com threshold: {gmm_threshold:.4f}")
        models['GMM'] = {'model': gmm_model, 'threshold': gmm_threshold}
    except Exception as e:
        print(f"Erro ao carregar GMM: {e}")
    
    return models

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calcula o Expected Calibration Error (ECE).
    """
    # Criar bins de probabilidade
    bin_indices = np.digitize(y_prob, np.linspace(0, 1, n_bins + 1)) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Inicializar arrays para estatísticas
    bin_accs = np.zeros(n_bins)  # Accuracy média por bin
    bin_confs = np.zeros(n_bins)  # Confiança média por bin
    bin_sizes = np.zeros(n_bins)  # Número de amostras por bin
    
    # Calcular estatísticas por bin
    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        if np.any(mask):
            bin_sizes[bin_idx] = mask.sum()
            bin_accs[bin_idx] = y_true[mask].mean()
            bin_confs[bin_idx] = y_prob[mask].mean()
    
    # Calcular ECE
    ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_sizes / len(y_true)))
    
    return ece, {
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_sizes': bin_sizes
    }

def plot_calibration_curve(y_true, y_prob, model_name, output_dir):
    """
    Plota a curva de calibração.
    """
    plt.figure(figsize=(10, 6))
    
    # Calcular curva de calibração
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    # Calcular ECE
    ece, _ = calculate_ece(y_true, y_prob)
    
    # Plotar curva
    plt.plot(prob_pred, prob_true, "s-", label=f'{model_name} (ECE={ece:.4f})')
    plt.plot([0, 1], [0, 1], "k--", label="Perfeitamente Calibrado")
    
    plt.xlabel('Probabilidade Média Predita')
    plt.ylabel('Fração de Positivos')
    plt.title(f'Curva de Calibração - {model_name}')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    # Salvar figura
    plt.savefig(os.path.join(output_dir, f"calibration_curve_{model_name}.png"))
    plt.close()
    
    return ece

def main():
    # Criar diretório para resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Resultados serão salvos em: {RESULTS_DIR}")
    
    # Carregar modelos
    models = load_models()
    
    # Avaliar Random Forest
    if 'RF' in models:
        print("\nAvaliando Random Forest calibrado...")
        
        # Carregar dados de teste para RF
        print(f"Carregando dados de teste para Random Forest: {TEST_PATH_RF}")
        rf_test_df = pd.read_csv(TEST_PATH_RF)
        
        rf_model = models['RF']['model']
        rf_threshold = models['RF']['threshold']
        
        # Preparar dados usando a função do arquivo 08_prob_calib_rf.py
        X_rf_test, y_rf_test = prepare_data_for_rf_model(rf_model, rf_test_df)
        
        print(f"Gerando probabilidades para {len(X_rf_test)} instâncias...")
        rf_probs = rf_model.predict_proba(X_rf_test)[:, 1]
        rf_preds = (rf_probs >= rf_threshold).astype(int)
        
        # Calcular métricas
        rf_precision = precision_score(y_rf_test, rf_preds)
        rf_recall = recall_score(y_rf_test, rf_preds)
        rf_f1 = f1_score(y_rf_test, rf_preds)
        rf_auc = roc_auc_score(y_rf_test, rf_probs)
        
        print("Métricas do Random Forest:")
        print(f"  Precision: {rf_precision:.4f}")
        print(f"  Recall: {rf_recall:.4f}")
        print(f"  F1 Score: {rf_f1:.4f}")
        print(f"  AUC-ROC: {rf_auc:.4f}")
        
        # Avaliar calibração
        print("Avaliando calibração do Random Forest...")
        rf_ece = plot_calibration_curve(y_rf_test, rf_probs, "Random Forest", RESULTS_DIR)
        print(f"Expected Calibration Error (ECE): {rf_ece:.4f}")
        
        # Salvar resultados
        rf_results = pd.DataFrame({
            'true': y_rf_test,
            'prediction': rf_preds,
            'probability': rf_probs
        })
        rf_results.to_csv(os.path.join(RESULTS_DIR, "rf_test_results.csv"), index=False)
    
    # Avaliar GMM
    if 'GMM' in models:
        print("\nAvaliando GMM calibrado...")
        
        # Carregar dados de teste para GMM
        print(f"Carregando dados de teste para GMM: {TEST_PATH_GMM}")
        gmm_test_df = pd.read_csv(TEST_PATH_GMM)
        
        gmm_model = models['GMM']['model']
        gmm_threshold = models['GMM']['threshold']
        
        # Separar features e target
        X_gmm_test = gmm_test_df.drop(columns=['target'])
        y_gmm_test = gmm_test_df['target']
        
        print(f"Gerando probabilidades para {len(X_gmm_test)} instâncias...")
        gmm_probs = gmm_model.predict_proba(X_gmm_test)[:, 1]
        gmm_preds = (gmm_probs >= gmm_threshold).astype(int)
        
        # Calcular métricas
        gmm_precision = precision_score(y_gmm_test, gmm_preds)
        gmm_recall = recall_score(y_gmm_test, gmm_preds)
        gmm_f1 = f1_score(y_gmm_test, gmm_preds)
        gmm_auc = roc_auc_score(y_gmm_test, gmm_probs)
        
        print("Métricas do GMM:")
        print(f"  Precision: {gmm_precision:.4f}")
        print(f"  Recall: {gmm_recall:.4f}")
        print(f"  F1 Score: {gmm_f1:.4f}")
        print(f"  AUC-ROC: {gmm_auc:.4f}")
        
        # Avaliar calibração
        print("Avaliando calibração do GMM...")
        gmm_ece = plot_calibration_curve(y_gmm_test, gmm_probs, "GMM", RESULTS_DIR)
        print(f"Expected Calibration Error (ECE): {gmm_ece:.4f}")
        
        # Salvar resultados
        gmm_results = pd.DataFrame({
            'true': y_gmm_test,
            'prediction': gmm_preds,
            'probability': gmm_probs
        })
        gmm_results.to_csv(os.path.join(RESULTS_DIR, "gmm_test_results.csv"), index=False)
    
    # Criar tabela comparativa
    if 'RF' in models and 'GMM' in models:
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1', 'AUC-ROC', 'ECE', 'Threshold'],
            'Random Forest': [rf_precision, rf_recall, rf_f1, rf_auc, rf_ece, rf_threshold],
            'GMM': [gmm_precision, gmm_recall, gmm_f1, gmm_auc, gmm_ece, gmm_threshold]
        })
    elif 'RF' in models:
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1', 'AUC-ROC', 'ECE', 'Threshold'],
            'Random Forest': [rf_precision, rf_recall, rf_f1, rf_auc, rf_ece, rf_threshold]
        })
    elif 'GMM' in models:
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1', 'AUC-ROC', 'ECE', 'Threshold'],
            'GMM': [gmm_precision, gmm_recall, gmm_f1, gmm_auc, gmm_ece, gmm_threshold]
        })
    else:
        metrics_df = pd.DataFrame({'Metric': ['Nenhum modelo avaliado com sucesso']})
    
    # Salvar comparação
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print(f"\nComparação de modelos salva em: {os.path.join(RESULTS_DIR, 'model_comparison.csv')}")
    
    print("\nAvaliação de modelos concluída!")
    print(f"Resultados e visualizações salvas em: {RESULTS_DIR}")

if __name__ == "__main__":
    main()