#TextFeatureEngineeringTransformer
#!/usr/bin/env python
"""
Transformador para processamento de texto conforme implementado
nos scripts 02_preprocessing.py e 03_feature_engineering.py.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Importar módulos de processamento de texto
from src.preprocessing.text_processing import text_feature_engineering
from src.preprocessing.advanced_feature_engineering import advanced_feature_engineering
from src.preprocessing.professional_motivation_features import enhance_professional_features

class TextFeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador que aplica todas as etapas de processamento de texto
    implementadas nos scripts de treinamento.
    """
    
    def __init__(self, params_path=None):
        """
        Inicializa o transformador.
        
        Args:
            params_path: Caminho para o arquivo joblib com parâmetros salvos
        """
        self.params_path = params_path
        self.params = None
        self.feature_names = None
        
        # Colunas de texto relevantes
        self.text_cols = [
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
            '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?', 
            'Déjame un mensaje',
            '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
        ]
        
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
            print(f"Parâmetros de texto carregados de {self.params_path}")
        else:
            raise ValueError(f"Arquivo de parâmetros não encontrado em {self.params_path}")
            
        return self
    
    def transform(self, X):
        """
        Aplica processamento de texto aos dados.
        
        Args:
            X: DataFrame de entrada
            
        Returns:
            DataFrame com features de texto adicionadas
        """
        if self.params is None:
            raise ValueError("O transformador precisa ser ajustado com fit() antes de usar transform()")
        
        print(f"Aplicando processamento de texto ao DataFrame: {X.shape}")
        df_result = X.copy()
        
        # 1. Features textuais básicas
        print("1. Processando features textuais básicas...")
        text_params = self.params.get('text_processing', {})
        df_result, _ = text_feature_engineering(df_result, fit=False, params=text_params)
        
        # 2. Features avançadas
        print("2. Aplicando feature engineering avançada para texto...")
        advanced_params = self.params.get('advanced_features', {})
        df_result, _ = advanced_feature_engineering(df_result, fit=False, params=advanced_params)
        
        # 3. Features de motivação profissional
        print("3. Criando features de motivação profissional...")
        if 'professional_features' in self.params:
            prof_params = self.params.get('professional_features', {})
            # Filtrar colunas de texto existentes
            text_cols = [col for col in self.text_cols if col in df_result.columns]
            if text_cols:
                df_result, _ = enhance_professional_features(df_result, text_cols, fit=False, params=prof_params)
        
        # Guardar nomes das features para referência
        self.feature_names = df_result.columns.tolist()
        
        print(f"Processamento de texto concluído! Dimensões finais: {df_result.shape}")
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