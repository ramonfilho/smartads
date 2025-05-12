"""
Pipeline de inferência para o modelo GMM.
Este módulo implementa uma classe que encapsula todas as etapas do pipeline de inferência
para fazer predições com o modelo GMM treinado e calibrado.
"""

import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Garantir que o caminho do projeto esteja no sys.path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

# Importar funções de normalização de email do script 1
from src.preprocessing.email_processing import normalize_emails_in_dataframe

# Importar funções de pré-processamento do script 2
from src.preprocessing.data_cleaning import (
    handle_missing_values, 
    handle_outliers,
    normalize_values,
    convert_data_types,
    consolidate_quality_columns
)

# Importar funções de feature engineering do script 3
from src.preprocessing.feature_engineering import feature_engineering
from src.preprocessing.text_processing import text_feature_engineering
from src.preprocessing.advanced_feature_engineering import advanced_feature_engineering

# Define o caminho padrão para os modelos e parâmetros
DEFAULT_MODEL_DIR = os.path.join(project_root, "models/calibrated/gmm_calibrated_20250508_130725")
DEFAULT_PARAMS_DIR = os.path.join(project_root, "src/preprocessing/preprocessing_params")


class GMMInferencePipeline:
    """
    Pipeline completo para fazer inferência com o modelo GMM.
    
    Esta classe implementa toda a lógica necessária para processar novos dados
    e fazer predições usando o modelo GMM treinado e calibrado.
    """
    
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, params_dir=DEFAULT_PARAMS_DIR):
        """
        Inicializa o pipeline de inferência GMM.
        
        Args:
            model_dir: Diretório onde o modelo GMM calibrado está salvo
            params_dir: Diretório onde os parâmetros de preprocessamento estão salvos
        """
        self.model_dir = model_dir
        self.params_dir = params_dir
        
        # Inicializar atributos que serão carregados
        self.calibrated_model = None
        self.threshold = None
        self.preprocessing_params = None
        self.decile_boundaries = None
        self.pca_model = None
        self.scaler_model = None
        self.gmm_model = None
        self.cluster_models = {}
        
        # Carregar todos os componentes necessários
        self._load_models()
        self._load_preprocessing_params()
        self._calculate_decile_boundaries()
        
        print(f"Pipeline de inferência GMM inicializado com sucesso.")
        print(f"Usando modelo de: {model_dir}")
        print(f"Usando parâmetros de: {params_dir}")
        print(f"Threshold calibrado: {self.threshold:.4f}")
    
    def _load_models(self):
        """Carrega o modelo GMM calibrado e seus componentes."""
        try:
            # Carregar o modelo GMM calibrado
            model_path = os.path.join(self.model_dir, "gmm_calibrated.joblib")
            self.calibrated_model = joblib.load(model_path)
            
            # Carregar o threshold calibrado
            threshold_path = os.path.join(self.model_dir, "threshold.txt")
            with open(threshold_path, 'r') as f:
                self.threshold = float(f.read().strip())
            
            # Carregar componentes individuais (PCA, scaler, GMM)
            # Para isso, vamos usar a base do modelo (diretório pai)
            base_model_dir = os.path.join(project_root, "models/artifacts/gmm_optimized")
            
            self.pca_model = joblib.load(os.path.join(base_model_dir, "pca_model.joblib"))
            self.scaler_model = joblib.load(os.path.join(base_model_dir, "scaler_model.joblib"))
            self.gmm_model = joblib.load(os.path.join(base_model_dir, "gmm_model.joblib"))
            
            # Carregar modelos específicos de cada cluster
            n_clusters = self.gmm_model.n_components
            for cluster_id in range(n_clusters):
                cluster_model_path = os.path.join(base_model_dir, f"cluster_{cluster_id}_model.joblib")
                if os.path.exists(cluster_model_path):
                    self.cluster_models[cluster_id] = joblib.load(cluster_model_path)
            
            print(f"Modelos carregados: GMM calibrado com {len(self.cluster_models)} modelos de cluster")
            
        except Exception as e:
            print(f"Erro ao carregar modelos: {e}")
            raise RuntimeError(f"Falha ao carregar modelos: {str(e)}")
    
    def _load_preprocessing_params(self):
        """Carrega os parâmetros de pré-processamento."""
        try:
            params_path = os.path.join(self.params_dir, "all_preprocessing_params.joblib")
            if os.path.exists(params_path):
                self.preprocessing_params = joblib.load(params_path)
                print(f"Parâmetros de pré-processamento carregados de: {params_path}")
            else:
                print(f"AVISO: Arquivo de parâmetros não encontrado: {params_path}")
                print(f"Usando parâmetros padrão para pré-processamento")
                self.preprocessing_params = {}
        except Exception as e:
            print(f"Erro ao carregar parâmetros de pré-processamento: {e}")
            self.preprocessing_params = {}
    
    def _calculate_decile_boundaries(self):
        """Calcula os limites dos decis para categorização das probabilidades."""
        try:
            # Primeiro tentamos carregar os limites dos decis se já tiverem sido calculados
            deciles_path = os.path.join(self.model_dir, "decile_boundaries.json")
            
            if os.path.exists(deciles_path):
                with open(deciles_path, 'r') as f:
                    self.decile_boundaries = json.load(f)
                print(f"Limites de decis carregados de: {deciles_path}")
            else:
                # Se não existirem, calculamos com base nos resultados de validação
                results_path = os.path.join(self.model_dir, "test_predictions_GMM_Calibrated.csv")
                if os.path.exists(results_path):
                    results_df = pd.read_csv(results_path)
                    if 'probability' in results_df.columns:
                        # Calcular os decis
                        self.decile_boundaries = np.percentile(
                            results_df['probability'], 
                            np.arange(0, 101, 10)
                        ).tolist()
                        
                        # Salvar para uso futuro
                        with open(deciles_path, 'w') as f:
                            json.dump(self.decile_boundaries, f)
                        
                        print(f"Limites de decis calculados e salvos em: {deciles_path}")
                    else:
                        print("AVISO: Coluna 'probability' não encontrada nos resultados")
                        self._set_default_decile_boundaries()
                else:
                    print(f"AVISO: Arquivo de resultados não encontrado: {results_path}")
                    self._set_default_decile_boundaries()
        except Exception as e:
            print(f"Erro ao calcular limites de decis: {e}")
            self._set_default_decile_boundaries()
    
    def _set_default_decile_boundaries(self):
        """Define limites de decis padrão quando não é possível calculá-los."""
        self.decile_boundaries = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print("Usando limites de decis padrão uniformes")
    
    def _assign_decile(self, probability):
        """
        Atribui um decil (1-10) para uma probabilidade com base nos limites calculados.
        
        Args:
            probability: Valor de probabilidade para categorizar
            
        Returns:
            Número do decil (1-10)
        """
        if self.decile_boundaries is None:
            self._calculate_decile_boundaries()
        
        for i in range(len(self.decile_boundaries) - 1):
            if probability >= self.decile_boundaries[i] and probability < self.decile_boundaries[i+1]:
                return i + 1
        
        return 10  # Para o caso de probabilidade == 1.0
    
    def _normalize_emails(self, data):
        """
        Normaliza endereços de email no DataFrame.
        
        Args:
            data: DataFrame com dados a serem processados
            
        Returns:
            DataFrame com emails normalizados
        """
        if 'email' in data.columns:
            return normalize_emails_in_dataframe(data, email_col='email')
        return data
    
    def _preprocess_data(self, data):
        """
        Aplica todas as etapas de pré-processamento aos dados.
        
        Args:
            data: DataFrame com dados a serem pré-processados
            
        Returns:
            DataFrame pré-processado
        """
        # 1. Começar com uma cópia do DataFrame para não modificar o original
        processed_df = data.copy()
        
        # 2. Normalizar emails (Script 1)
        processed_df = self._normalize_emails(processed_df)
        
        # 3. Aplicar etapas de pré-processamento do Script 2
        # Consolidar colunas de qualidade
        processed_df, _ = consolidate_quality_columns(
            processed_df, 
            fit=False, 
            params=self.preprocessing_params.get('quality_columns', {})
        )
        
        # Tratar valores ausentes
        processed_df, _ = handle_missing_values(
            processed_df, 
            fit=False, 
            params=self.preprocessing_params.get('missing_values', {})
        )
        
        # Tratar outliers
        processed_df, _ = handle_outliers(
            processed_df, 
            fit=False, 
            params=self.preprocessing_params.get('outliers', {})
        )
        
        # Normalizar valores numéricos
        processed_df, _ = normalize_values(
            processed_df, 
            fit=False, 
            params=self.preprocessing_params.get('normalization', {})
        )
        
        # Converter tipos de dados
        processed_df, _ = convert_data_types(
            processed_df, 
            fit=False
        )
        
        # 4. Aplicar feature engineering do Script 3
        # Features não-textuais
        processed_df, _ = feature_engineering(
            processed_df, 
            fit=False, 
            params=self.preprocessing_params.get('feature_engineering', {})
        )
        
        # Features textuais
        processed_df, _ = text_feature_engineering(
            processed_df, 
            fit=False, 
            params=self.preprocessing_params.get('text_processing', {})
        )
        
        # Features avançadas
        processed_df, _ = advanced_feature_engineering(
            processed_df, 
            fit=False, 
            params=self.preprocessing_params.get('advanced_features', {})
        )
        
        return processed_df
    
    def _prepare_for_gmm(self, data):
        """
        Prepara os dados para o modelo GMM aplicando scaling e PCA.
        
        Args:
            data: DataFrame pré-processado
            
        Returns:
            Dados transformados e labels de cluster
        """
        # 1. Selecionar features numéricas
        numeric_cols = []
        for col in data.columns:
            try:
                data[col].astype(float)
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        X_numeric = data[numeric_cols].copy().astype(float)
        
        # 2. Substituir valores ausentes
        X_numeric.fillna(0, inplace=True)
        
        # 3. Aplicar o scaler
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
        
        X_scaled = self.scaler_model.transform(X_numeric)
        
        # 4. Aplicar PCA
        X_pca = self.pca_model.transform(X_scaled)
        
        # 5. Aplicar GMM para obter labels de cluster
        cluster_labels = self.gmm_model.predict(X_pca)
        
        return X_pca, cluster_labels, X_numeric
    
    def _get_probabilities(self, data, X_numeric, cluster_labels):
        """
        Gera probabilidades de predição usando os modelos específicos por cluster.
        
        Args:
            data: DataFrame completo
            X_numeric: Features numéricas selecionadas
            cluster_labels: Labels de cluster para cada amostra
            
        Returns:
            Array de probabilidades
        """
        n_samples = len(data)
        probabilities = np.zeros(n_samples, dtype=float)
        
        # Para cada cluster, usar o modelo específico
        for cluster_id, model in self.cluster_models.items():
            # Selecionar amostras deste cluster
            cluster_mask = (cluster_labels == cluster_id)
            
            if not any(cluster_mask):
                continue
            
            # Detectar features necessárias para este modelo
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                
                # Criar DataFrame temporário com features corretas
                X_temp = data.copy()
                
                # Adicionar features ausentes
                for col in expected_features:
                    if col not in X_temp.columns:
                        X_temp[col] = 0.0
                
                # Selecionar amostras e features para este cluster
                X_cluster = X_temp.loc[cluster_mask, expected_features].astype(float)
                X_cluster.fillna(0, inplace=True)
            else:
                # Usar features numéricas selecionadas anteriormente
                X_cluster = X_numeric.loc[cluster_mask].copy()
            
            # Fazer predições se temos amostras
            if len(X_cluster) > 0:
                try:
                    proba = model.predict_proba(X_cluster)[:, 1]
                    probabilities[cluster_mask] = proba
                except Exception as e:
                    print(f"Erro ao prever para cluster {cluster_id}: {e}")
                    # Usar valor default conservador
                    probabilities[cluster_mask] = 0.1
        
        return probabilities
    
    def predict(self, data):
        """
        Faz predições para novos dados.
        
        Args:
            data: DataFrame com dados para predição
            
        Returns:
            Dictionary com predições, probabilidades e decis
        """
        # 1. Pré-processar os dados
        print("Pré-processando dados...")
        processed_data = self._preprocess_data(data)
        
        # 2. Preparar dados para o GMM
        print("Preparando dados para o GMM...")
        _, cluster_labels, X_numeric = self._prepare_for_gmm(processed_data)
        
        # 3. Gerar probabilidades
        print("Gerando probabilidades...")
        probabilities = self._get_probabilities(processed_data, X_numeric, cluster_labels)
        
        # 4. Aplicar threshold para obter predições
        predictions = (probabilities >= self.threshold).astype(int)
        
        # 5. Atribuir decil para cada probabilidade
        deciles = np.array([self._assign_decile(p) for p in probabilities])
        
        # 6. Criar DataFrame de resultados
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'decile': deciles
        })
        
        # 7. Adicionar metadados
        metadata = {
            'model_path': self.model_dir,
            'threshold': float(self.threshold),
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(data),
            'positive_rate': float(predictions.mean()),
            'probability_mean': float(probabilities.mean()),
            'probability_median': float(np.median(probabilities))
        }
        
        return {
            'results': results,
            'metadata': metadata
        }