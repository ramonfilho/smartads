#EmailNormalizationTransformer
#!/usr/bin/env python
"""
Transformador para normalização de emails, seguindo exatamente
o mesmo processo usado no treinamento.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.preprocessing.email_processing import normalize_emails_in_dataframe

class EmailNormalizationTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador que aplica normalização de emails como feito no script 01.
    """
    
    def __init__(self, email_col='email'):
        """
        Inicializa o transformador.
        
        Args:
            email_col: Nome da coluna contendo emails
        """
        self.email_col = email_col
        
    def fit(self, X, y=None):
        """
        Não há necessidade de ajuste para este transformador.
        
        Args:
            X: DataFrame de entrada
            y: Target (não utilizado)
            
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """
        Aplica normalização de emails ao DataFrame.
        
        Args:
            X: DataFrame de entrada
            
        Returns:
            DataFrame com emails normalizados
        """
        # Verificar se a coluna de email existe
        if self.email_col not in X.columns:
            print(f"Aviso: Coluna '{self.email_col}' não encontrada no DataFrame.")
            return X
        
        # Normalizar emails
        print(f"Normalizando emails na coluna '{self.email_col}'...")
        df_result = normalize_emails_in_dataframe(X.copy(), email_col=self.email_col)
        
        return df_result
    
    def get_feature_names_out(self, input_features=None):
        """
        Retorna os nomes das features de saída.
        
        Args:
            input_features: Lista das features de entrada (não utilizado)
            
        Returns:
            Lista com nomes das features de saída
        """
        if input_features is None:
            return []
        
        # Se incluímos 'email_norm', adicionamos à lista
        result = list(input_features)
        if self.email_col in input_features and 'email_norm' not in input_features:
            result.append('email_norm')
        
        return result