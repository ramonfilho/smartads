#GMM_InferenceWrapper
#!/usr/bin/env python
"""
Wrapper para o modelo GMM calibrado, replicando exatamente a lógica 
implementada no script 09_prob_calib_gmm.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin

class GMM_InferenceWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper para GMM que implementa a mesma API do script 09_prob_calib_gmm.py
    """
    def __init__(self, models_dir=None, threshold=0.15):
        """
        Inicializa o wrapper.
        
        Args:
            models_dir: Diretório contendo componentes do modelo
            threshold: Threshold para classificação binária
        """
        self.models_dir = models_dir
        self.threshold = threshold
        self.pipeline = None
        self.pca_model = None
        self.gmm_model = None
        self.scaler_model = None
        self.cluster_models = {}
        self.n_clusters = None
        self.feature_names_used = []
        
        # Atributos necessários para a API sklearn
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"
    
    def fit(self, X=None, y=None):
        """
        Carrega os componentes do modelo.
        
        Args:
            X: Não utilizado, mantido para compatibilidade
            y: Não utilizado, mantido para compatibilidade
            
        Returns:
            self
        """
        if self.models_dir is None:
            raise ValueError("models_dir deve ser especificado")
            
        print(f"Carregando componentes do modelo de {self.models_dir}...")
        
        # Carregar componentes principais
        try:
            self.pca_model = joblib.load(os.path.join(self.models_dir, "pca_model.joblib"))
            self.gmm_model = joblib.load(os.path.join(self.models_dir, "gmm_model.joblib"))
            self.scaler_model = joblib.load(os.path.join(self.models_dir, "scaler_model.joblib"))
            
            # Carregar informações de configuração
            import json
            with open(os.path.join(self.models_dir, "experiment_config.json"), 'r') as f:
                config = json.load(f)
                self.n_clusters = config.get('gmm_params', {}).get('n_components', 3)
            
            # Carregar threshold específico, se existir
            threshold_path = os.path.join(self.models_dir, "threshold.txt")
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    self.threshold = float(f.read().strip())
                print(f"Usando threshold: {self.threshold}")
            
            # Carregar modelos de cluster
            for cluster_id in range(self.n_clusters):
                model_path = os.path.join(self.models_dir, f"cluster_{cluster_id}_model.joblib")
                
                try:
                    model = joblib.load(model_path)
                    self.cluster_models[cluster_id] = {
                        'model': model,
                        'threshold': self.threshold,
                        'features': []
                    }
                    
                    # Extrair nomes de features se disponíveis
                    if hasattr(model, 'feature_names_in_'):
                        self.cluster_models[cluster_id]['features'] = model.feature_names_in_.tolist()
                        # Atualizar lista central de features
                        self.feature_names_used.extend(model.feature_names_in_)
                    
                    print(f"  Carregado modelo para cluster {cluster_id}")
                except Exception as e:
                    print(f"  Aviso: Não foi possível carregar modelo para cluster {cluster_id}: {e}")
            
            # Remover duplicatas da lista de features
            self.feature_names_used = list(set(self.feature_names_used))
            print(f"Total de {len(self.feature_names_used)} features usadas pelos modelos")
            
            return self
        
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar componentes do modelo: {e}")
    
    def predict_proba(self, X):
        """
        Gera probabilidades de previsão para novos dados.
        
        Args:
            X: DataFrame de entrada
            
        Returns:
            Array com probabilidades para cada classe [p(0), p(1)]
        """
        if self.gmm_model is None or self.pca_model is None:
            raise ValueError("Componentes do modelo não carregados. Execute fit() antes de predict_proba()")
        
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
            
            # Preparar features para este modelo
            cluster_features = []
            if hasattr(model, 'feature_names_in_'):
                cluster_features = model.feature_names_in_
            
            # Selecionar amostras e features para este cluster
            if len(cluster_features) > 0:
                # Criar cópia temporária com as features necessárias
                X_temp = X.copy()
                missing_features = [col for col in cluster_features if col not in X.columns]
                
                # Adicionar features ausentes
                for col in missing_features:
                    X_temp[col] = 0.0
                
                # Selecionar apenas as amostras deste cluster e as features necessárias
                X_cluster = X_temp.loc[cluster_mask, cluster_features].astype(float)
                X_cluster = X_cluster.fillna(0)  # Substituir NaNs por zeros
            else:
                # Se não temos lista de features, usar todas as numéricas
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
        """
        Gera previsões binárias para novos dados.
        
        Args:
            X: DataFrame de entrada
            
        Returns:
            Array com previsões binárias (0 ou 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)