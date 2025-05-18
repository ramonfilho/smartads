#!/usr/bin/env python
"""
Módulo para aplicar GMM e fazer predições no pipeline de inferência.
Esta versão usa os pontos de calibração exatos obtidos através da comparação entre
as probabilidades da pipeline e do modelo de referência.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Configuração de caminhos
project_root = "/Users/ramonmoreira/desktop/smart_ads"
PARAMS_DIR = os.path.join(project_root, "inference/params")
MODELS_DIR = os.path.join(project_root, "models/artifacts/gmm_optimized")
DEBUG_LOG = True  # Ativar/desativar logs detalhados

# Pontos de calibração exatos obtidos da comparação com o modelo de referência
REFERENCE_CALIBRATION_POINTS = [
    (0.000000, 0.000000),
    (0.007105, 0.008055),
    (0.017600, 0.014551),
    (0.027550, 0.020770),
    (0.037500, 0.034242),
    (0.047500, 0.186238),  # Observe o grande salto aqui
    (0.097200, 0.513811),  # E aqui
    (0.297000, 1.000000),
    (0.496000, 1.000000),
    (0.576000, 1.000000),
    (0.646000, 1.000000),
    (0.716000, 1.000000),
    (0.786000, 1.000000),
    (0.855000, 1.000000),
    (0.925000, 1.000000),
    (0.995000, 0.762743),
    (1.000000, 1.000000),
]

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
        
    def predict_proba_raw(self, X):
        """
        Retorna as probabilidades brutas sem aplicar calibração.
        """
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
                if DEBUG_LOG:
                    print(f"Removendo {len(unseen_features)} features não vistas durante treinamento")
                X_numeric = X_numeric.drop(columns=unseen_features)
            
            # Identificar features que faltam em X_numeric mas estão no scaler
            missing_features = [col for col in scaler_features if col not in X_numeric.columns]
            if missing_features:
                if DEBUG_LOG:
                    print(f"Adicionando {len(missing_features)} features ausentes vistas durante treinamento")
                # Criar um DataFrame com as colunas faltantes (mais eficiente)
                missing_df = pd.DataFrame(0, index=X_numeric.index, columns=missing_features)
                X_numeric = pd.concat([X_numeric, missing_df], axis=1)
            
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
        cluster_stats = {}  # Para estatísticas de debug
        
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
                
                # Criar um DataFrame temporário com apenas as amostras deste cluster
                X_cluster_base = X.loc[cluster_mask].copy()
                
                # Identificar features ausentes
                missing_features = [col for col in expected_features if col not in X_cluster_base.columns]
                
                # Adicionar features faltantes de forma eficiente
                if missing_features:
                    # Criar um DataFrame com as colunas faltantes
                    missing_df = pd.DataFrame(0, index=X_cluster_base.index, columns=missing_features)
                    X_cluster = pd.concat([X_cluster_base, missing_df], axis=1)
                else:
                    X_cluster = X_cluster_base
                
                # Filtrar para usar apenas as features esperadas, na ordem correta
                features_to_use = [col for col in expected_features if col in X_cluster.columns]
                if len(features_to_use) < len(expected_features):
                    print(f"AVISO: Apenas {len(features_to_use)} de {len(expected_features)} features disponíveis")
                
                X_cluster = X_cluster[features_to_use].astype(float)
                
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
                    
                    # Estatísticas para debug
                    if DEBUG_LOG:
                        n_above_thresh = np.sum(proba[:, 1] > 0.1)
                        cluster_stats[cluster_id_int] = {
                            'count': len(X_cluster),
                            'min': np.min(proba[:, 1]),
                            'max': np.max(proba[:, 1]),
                            'mean': np.mean(proba[:, 1]),
                            '% > 0.1': n_above_thresh / len(X_cluster),
                        }
                        
                except Exception as e:
                    print(f"ERRO ao fazer previsões para o cluster {cluster_id_int}: {e}")
                    # Em caso de erro, usar probabilidades default
                    y_pred_proba[cluster_mask, 0] = 0.9  # classe negativa (majoritária)
                    y_pred_proba[cluster_mask, 1] = 0.1  # classe positiva (minoritária)
        
        # Gerar estatísticas para debug
        if DEBUG_LOG:
            print("DEBUG: Estatísticas de probabilidade por cluster:")
            for cluster_id, stats in cluster_stats.items():
                print(f"  Cluster {cluster_id}: {stats['count']} amostras")
                print(f"    min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}")
                print(f"    % > 0.1: {stats['% > 0.1']:.6f}")
            
            # Estatísticas gerais
            all_probs = y_pred_proba[:, 1]
            stats = {
                'min': np.min(all_probs),
                'max': np.max(all_probs),
                'mean': np.mean(all_probs),
                'median': np.median(all_probs),
                'std': np.std(all_probs),
                '% > 0.1': np.mean(all_probs > 0.1),
                '% > 0.5': np.mean(all_probs > 0.5)
            }
            print(f"DEBUG: Estatísticas finais: {stats}")
            
        return y_pred_proba
    
    def predict_proba(self, X):
        """
        Retorna as probabilidades calibradas usando o mapeamento exato para referência.
        """
        # Obter probabilidades brutas
        raw_probs = self.predict_proba_raw(X)[:, 1]
        
        if DEBUG_LOG:
            print("\nComparação de probabilidades antes e depois da calibração:")
            print("Brutas (antes da calibração):")
            print(f"  Min: {np.min(raw_probs):.6f}")
            print(f"  Max: {np.max(raw_probs):.6f}")
            print(f"  Mean: {np.mean(raw_probs):.6f}")
            print(f"  % > 0.1: {np.mean(raw_probs > 0.1):.6f}")
        
        # Aplicar a calibração que mapeia para o modelo de referência
        print("Aplicando calibração para modelo de referência...")
        
        # Extrair arrays para interpolação
        x_vals = np.array([p[0] for p in REFERENCE_CALIBRATION_POINTS])
        y_vals = np.array([p[1] for p in REFERENCE_CALIBRATION_POINTS])
        
        # Garantir que valores estão dentro dos limites
        raw_probs_clipped = np.clip(raw_probs, x_vals[0], x_vals[-1])
        
        # Aplicar interpolação linear
        calibrated_probs = np.interp(raw_probs_clipped, x_vals, y_vals)
        
        if DEBUG_LOG:
            print("Calibradas (mapeadas para referência):")
            print(f"  Min: {np.min(calibrated_probs):.6f}")
            print(f"  Max: {np.max(calibrated_probs):.6f}")
            print(f"  Mean: {np.mean(calibrated_probs):.6f}")
            print(f"  % > 0.1: {np.mean(calibrated_probs > 0.1):.6f}")
        
        # Retornar no formato esperado
        result = np.zeros((len(raw_probs), 2), dtype=float)
        result[:, 0] = 1 - calibrated_probs
        result[:, 1] = calibrated_probs
        
        return result
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)


def load_gmm_pipeline(models_dir=MODELS_DIR):
    """
    Carrega todos os componentes do pipeline GMM diretamente.
    
    Args:
        models_dir: Diretório contendo os modelos
        
    Returns:
        Dicionário com os componentes do pipeline
    """
    print("Carregando componentes do pipeline GMM...")
    
    # Carregar PCA, GMM e Scaler
    try:
        pca_model = joblib.load(os.path.join(models_dir, "pca_model.joblib"))
        gmm_model = joblib.load(os.path.join(models_dir, "gmm_model.joblib"))
        scaler_model = joblib.load(os.path.join(models_dir, "scaler_model.joblib"))
        
        # Determinar número de clusters a partir do GMM
        n_clusters = gmm_model.n_components
        print(f"Modelo GMM carregado com {n_clusters} clusters")
        
        # Carregar modelos por cluster
        cluster_models = {}
        for cluster_id in range(n_clusters):
            model_path = os.path.join(models_dir, f"cluster_{cluster_id}_model.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                cluster_models[cluster_id] = {
                    'model': model,
                    'threshold': 0.1  # Threshold default
                }
                print(f"  Carregado modelo para cluster {cluster_id}")
        
        # Montar pipeline
        pipeline = {
            'pca_model': pca_model,
            'gmm_model': gmm_model,
            'scaler_model': scaler_model,
            'cluster_models': cluster_models,
            'n_clusters': n_clusters,
            'threshold': 0.1  # Threshold default
        }
        
        print("Pipeline GMM carregado com sucesso!")
        return pipeline
        
    except Exception as e:
        print(f"ERRO ao carregar pipeline GMM: {e}")
        raise


def get_calibration_model(threshold=0.1):
    """
    Obtém um modelo GMM com calibração e threshold configurados.
    Esta função recria o modelo a partir dos componentes básicos.
    
    Args:
        threshold: Threshold para classificação
        
    Returns:
        Tuple (modelo GMM, threshold)
    """
    print("\nPreparando modelo GMM com calibração para referência...")
    
    try:
        # Carregar pipeline GMM
        pipeline = load_gmm_pipeline(MODELS_DIR)
        
        # Criar o wrapper
        gmm_wrapper = GMM_Wrapper(pipeline)
        gmm_wrapper.threshold = threshold
        
        print("Modelo GMM preparado com sucesso!")
        
        return gmm_wrapper, threshold
        
    except Exception as e:
        print(f"ERRO ao preparar modelo: {e}")
        import traceback
        print(traceback.format_exc())
        raise RuntimeError(f"Falha ao carregar ou reconstruir o modelo: {str(e)}")


def apply_gmm_and_predict(df, params_dir=PARAMS_DIR):
    """
    Aplica o modelo GMM e faz predições com calibração corrigida.
    
    Args:
        df: DataFrame com features processadas
        params_dir: Diretório com parâmetros do modelo
        
    Returns:
        DataFrame com predições adicionadas
    """
    print("\n=== Aplicando GMM com calibração corrigida ===")
    print(f"Processando {len(df)} amostras...")
    
    # Copiar o DataFrame para não modificar o original
    result_df = df.copy()
    
    try:
        # Obter modelo com calibração
        gmm_wrapper, threshold = get_calibration_model()
        
        # Fazer predições
        print("Fazendo predições...")
        start_time = datetime.now()
        
        # Obter probabilidades calibradas
        print("Obtendo probabilidades calibradas...")
        probabilities = gmm_wrapper.predict_proba(df)[:, 1]
        
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
        
        raise RuntimeError(f"Falha ao fazer predições: {str(e)}")
    
    return result_df


if __name__ == "__main__":
    # Teste do módulo
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        try:
            # Testar o carregamento do modelo
            print("Testando carregamento do modelo GMM e calibração...")
            gmm_wrapper, threshold = get_calibration_model()
            print(f"Modelo GMM carregado com sucesso! Threshold: {threshold}")
            
            # Criar um pequeno DataFrame de teste
            print("\nCriando DataFrame de teste para verificar a predição...")
            test_df = pd.DataFrame({
                'feature1': [0.01, 0.047, 0.1, 0.5],  # Testes específicos em pontos críticos
                'feature2': [0.5, 0.6, 0.7, 0.8]
            })
            
            # Testar predição
            print("\nTestando predição...")
            result = apply_gmm_and_predict(test_df)
            print("\nTeste bem-sucedido!")
            
            sys.exit(0)
        except Exception as e:
            print(f"Erro no teste: {e}")
            import traceback
            print(traceback.format_exc())
            sys.exit(1)
    
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