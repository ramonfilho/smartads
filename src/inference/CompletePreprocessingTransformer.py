#CompletePreprocessingTransformer
#!/usr/bin/env python
"""
Transformador de pré-processamento completo para pipeline de inferência,
implementando todas as etapas do script 02_preprocessing.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Import dos módulos de pré-processamento
from src.preprocessing.data_cleaning import (
    consolidate_quality_columns,
    handle_missing_values,
    handle_outliers,
    normalize_values,
    convert_data_types
)
from src.preprocessing.feature_engineering import feature_engineering

class CompletePreprocessingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador que aplica todas as etapas de pré-processamento 
    implementadas no script 02_preprocessing.py
    """
    
    def __init__(self, params_path=None, preserve_text=True):
        """
        Inicializa o transformador de pré-processamento.
        
        Args:
            params_path: Caminho para o arquivo joblib com parâmetros salvos
            preserve_text: Se True, preserva as colunas de texto originais
        """
        self.params_path = params_path
        self.preserve_text = preserve_text
        self.params = None
        self.feature_names = None
        
    def fit(self, X, y=None):
        """
        Carrega os parâmetros salvos durante o treino.
        
        Args:
            X: DataFrame de entrada (não utilizado)
            y: Target (não utilizado)
            
        Returns:
            self
        """
        if self.params_path and os.path.exists(self.params_path):
            self.params = joblib.load(self.params_path)
            print(f"Parâmetros carregados de {self.params_path}")
        else:
            raise ValueError(f"Arquivo de parâmetros não encontrado em {self.params_path}")
            
        return self
    
    def transform(self, X):
        """
        Aplica a pipeline completa de pré-processamento em novos dados.
        
        Args:
            X: DataFrame de entrada
            
        Returns:
            DataFrame processado
        """
        if self.params is None:
            raise ValueError("O transformador precisa ser ajustado com fit() antes de usar transform()")
        
        print(f"Aplicando pré-processamento para DataFrame: {X.shape}")
        df_result = X.copy()
        
        # 1. Consolidar colunas de qualidade
        print("1. Consolidando colunas de qualidade...")
        quality_params = self.params.get('quality_columns', {})
        df_result, _ = consolidate_quality_columns(df_result, fit=False, params=quality_params)
        
        # 2. Tratamento de valores ausentes
        print("2. Tratando valores ausentes...")
        missing_params = self.params.get('missing_values', {})
        df_result, _ = handle_missing_values(df_result, fit=False, params=missing_params)
        
        # 3. Tratamento de outliers
        print("3. Tratando outliers...")
        outlier_params = self.params.get('outliers', {})
        df_result, _ = handle_outliers(df_result, fit=False, params=outlier_params)
        
        # 4. Normalização de valores
        print("4. Normalizando valores numéricos...")
        norm_params = self.params.get('normalization', {})
        df_result, _ = normalize_values(df_result, fit=False, params=norm_params)
        
        # 5. Converter tipos de dados
        print("5. Convertendo tipos de dados...")
        df_result, _ = convert_data_types(df_result, fit=False)
        
        # Identificar colunas de texto antes do processamento
        text_cols = [
            col for col in df_result.columns 
            if df_result[col].dtype == 'object' and any(term in col for term in [
                'mensaje', 'inglés', 'vida', 'oportunidades', 'esperas', 'aprender', 
                'Semana', 'Inmersión', 'Déjame', 'fluidez'
            ])
        ]
        
        # Criar cópia das colunas de texto originais
        if text_cols and self.preserve_text:
            print("Preservando colunas de texto originais...")
            for col in text_cols:
                df_result[f"{col}_original"] = df_result[col].copy()
        
        # 6. Feature engineering não-textual
        print("6. Aplicando feature engineering não-textual...")
        feature_params = self.params.get('feature_engineering', {})
        df_result, _ = feature_engineering(df_result, fit=False, params=feature_params)
        
        # Guardar nomes das features para referência
        self.feature_names = df_result.columns.tolist()
        
        print(f"Pré-processamento básico concluído! Dimensões: {df_result.shape}")
        return df_result
    
    def get_feature_names_out(self, input_features=None):
        """
        Retorna os nomes das features após a transformação.
        
        Args:
            input_features: Lista das features de entrada (não utilizado)
            
        Returns:
            Lista com nomes das features de saída
        """
        if self.feature_names is None:
            raise ValueError("O método transform deve ser chamado antes de get_feature_names_out")
        return self.feature_names