#!/usr/bin/env python
"""
Módulo para aplicar GMM e fazer predições no pipeline de inferência, usando o modelo calibrado.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Adicionar caminho do projeto ao sys.path se necessário
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Diretório para parâmetros
PARAMS_DIR = os.path.join(project_root, "inference/params")

# Definir a classe GMM_Wrapper exatamente como no script 10_prob_calib_gmm.py
# Esta classe é necessária para carregar o modelo calibrado
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
        self.n_clusters = pipeline.get('n_clusters', 3)
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

def load_calibrated_model(params_dir=PARAMS_DIR):
    """
    Carrega o modelo GMM calibrado.
    
    Args:
        params_dir: Diretório com os parâmetros
        
    Returns:
        Tuple (modelo calibrado, threshold)
    """
    print("\nCarregando modelo GMM calibrado...")
    
    # Caminho absoluto para o modelo calibrado
    model_path = "/Users/ramonmoreira/desktop/smart_ads/inference/params/10_gmm_calibrated.joblib"
    
    # Verificar se o arquivo existe
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo calibrado não encontrado em: {model_path}")
        
        # Tentar caminhos alternativos
        alt_paths = [
            os.path.join(params_dir, "10_gmm_calibrated.joblib"),
            os.path.join(params_dir, "models", "10_gmm_calibrated.joblib")
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"Modelo encontrado em: {alt_path}")
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Modelo calibrado não encontrado em nenhum caminho conhecido")
    
    # Caminho para o threshold (assumindo que está no mesmo diretório que o modelo)
    threshold_path = os.path.join(os.path.dirname(model_path), "10_threshold.txt")
    
    # Verificar se o arquivo de threshold existe
    if not os.path.exists(threshold_path):
        print(f"AVISO: Arquivo de threshold não encontrado: {threshold_path}")
        print("Usando threshold padrão: 0.5")
        threshold = 0.5
    else:
        # Carregar threshold
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
    
    # Carregar modelo calibrado
    print(f"  Carregando modelo de: {model_path}")
    try:
        calibrated_model = joblib.load(model_path)
        print(f"  Modelo carregado com sucesso!")
        print(f"  Threshold: {threshold:.4f}")
        return calibrated_model, threshold
    except Exception as e:
        print(f"  ERRO ao carregar modelo calibrado: {e}")
        import traceback
        print(traceback.format_exc())
        raise

def apply_gmm_and_predict(df, params_dir=PARAMS_DIR):
    """
    Aplica o modelo GMM calibrado para fazer predições.
    
    Args:
        df: DataFrame com features processadas
        params_dir: Diretório com parâmetros do modelo
        
    Returns:
        DataFrame com predições adicionadas
    """
    print("\n=== Aplicando GMM calibrado e fazendo predição ===")
    print(f"Processando {len(df)} amostras...")
    
    # Copiar o DataFrame para não modificar o original
    result_df = df.copy()
    
    try:
        # Carregar modelo calibrado
        calibrated_model, threshold = load_calibrated_model(params_dir)
        
        # Fazer predições
        print("Fazendo predições...")
        start_time = datetime.now()
        
        # Calcular probabilidades
        probabilities = calibrated_model.predict_proba(df)[:, 1]
        
        # Aplicar threshold para classes
        predictions = (probabilities >= threshold).astype(int)
        
        # Adicionar ao DataFrame
        result_df['probability'] = probabilities
        result_df['prediction'] = predictions
        
        # Calcular estatísticas
        prediction_counts = dict(zip(*np.unique(predictions, return_counts=True)))
        print(f"  Distribuição de predições: {prediction_counts}")
        
        if 1 in prediction_counts:
            positive_rate = prediction_counts[1] / len(df)
            print(f"  Taxa de positivos: {positive_rate:.4f} ({prediction_counts.get(1, 0)} de {len(df)})")
        
        # Tempo de processamento
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"Predições concluídas em {elapsed_time:.2f} segundos.")
        
    except Exception as e:
        print(f"  ERRO durante predição: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Em caso de erro, adicionar colunas com valores default
        result_df['prediction'] = 0
        result_df['probability'] = 0.0
    
    return result_df

# Função para testar o carregamento do modelo
def test_model_loading(params_dir=PARAMS_DIR):
    """
    Testa se conseguimos carregar o modelo calibrado.
    
    Args:
        params_dir: Diretório com os parâmetros
    """
    print("Testando carregamento do modelo calibrado...")
    model_path = os.path.join(params_dir, "10_gmm_calibrated.joblib")
    
    try:
        # Carregar o modelo
        model = joblib.load(model_path)
        
        # Verificar o tipo do modelo
        print(f"Tipo do modelo: {type(model).__name__}")
        
        # Verificar atributos
        if hasattr(model, 'base_estimator'):
            print(f"Tipo do estimador base: {type(model.base_estimator).__name__}")
            
            # Se o base_estimator for o GMM_Wrapper
            if hasattr(model.base_estimator, 'pipeline'):
                # Verifica os componentes do pipeline
                pipeline = model.base_estimator.pipeline
                for key, value in pipeline.items():
                    print(f"Pipeline contém: {key} ({type(value).__name__})")
                
                # Verifica os modelos por cluster
                if 'cluster_models' in pipeline:
                    print(f"Número de modelos por cluster: {len(pipeline['cluster_models'])}")
        
        print("Modelo carregado com sucesso!")
        return True
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Primeiro testar se conseguimos carregar o modelo
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = test_model_loading()
        sys.exit(0 if success else 1)
    
    # Pipeline completo
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Default para teste
        input_file = os.path.join(project_root, "inference/output/predictions_latest.csv")
        output_file = os.path.join(project_root, "inference/output/final_predictions.csv")
    
    # Verificar se o arquivo existe
    if not os.path.exists(input_file):
        print(f"ERRO: Arquivo de entrada não encontrado: {input_file}")
        sys.exit(1)
    
    # Carregar dados
    print(f"Carregando dados de: {input_file}")
    df = pd.read_csv(input_file)
    
    # Aplicar GMM e fazer predições
    result_df = apply_gmm_and_predict(df)
    
    # Salvar resultados se output_file especificado
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result_df.to_csv(output_file, index=False)
        print(f"Resultados salvos em: {output_file}")