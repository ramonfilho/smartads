# Inferência em produção
"""
Módulo para inferência usando modelos de stacking.

Este módulo fornece classes e funções para realizar inferência (previsões)
usando modelos de stacking previamente treinados, incluindo funções para
preparar os dados no formato esperado pelo modelo.
"""

import pandas as pd
import numpy as np
import os
import joblib
import time
from collections import defaultdict
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class StackingPredictor:
    """
    Classe para fazer previsões com modelos de stacking.
    """
    
    def __init__(self, model_path, feature_groups_path, threshold=None):
        """
        Inicializa o preditor com um modelo de stacking treinado.
        
        Args:
            model_path: Caminho para o modelo de stacking salvo
            feature_groups_path: Caminho para o arquivo com grupos de features
            threshold: Threshold para classificação (opcional)
        """
        self.model_path = model_path
        self.feature_groups_path = feature_groups_path
        self.threshold = threshold
        
        # Variáveis que serão inicializadas pelo método load
        self.model = None
        self.feature_groups = None
        self.specialists = None
        self.feature_types = None
        
        # Carregar modelo e configuração
        self.load()
    
    def load(self):
        """
        Carrega o modelo e configuração.
        """
        logger.info(f"Carregando modelo de {self.model_path}")
        
        # Verificar se o arquivo existe
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {self.model_path}")
        
        # Carregar modelo
        self.model = joblib.load(self.model_path)
        
        # Extrair especialistas treinados do modelo
        if hasattr(self.model, 'trained_specialists'):
            self.specialists = self.model.trained_specialists
        else:
            logger.warning("Modelo não possui atributo 'trained_specialists'")
            self.specialists = []
        
        # Extrair tipos de features
        self.feature_types = [s.feature_type for s in self.specialists] if self.specialists else []
        
        # Carregar grupos de features
        logger.info(f"Carregando grupos de features de {self.feature_groups_path}")
        
        # Verificar se o arquivo existe
        if not os.path.exists(self.feature_groups_path):
            raise FileNotFoundError(f"Arquivo de grupos de features não encontrado: {self.feature_groups_path}")
        
        # Carregar grupos
        self.feature_groups = self.load_feature_groups(self.feature_groups_path)
        
        # Determinar threshold
        if self.threshold is None and hasattr(self.model, 'threshold'):
            self.threshold = self.model.threshold
        elif self.threshold is None:
            self.threshold = 0.5
            logger.warning(f"Threshold não fornecido e não encontrado no modelo. Usando padrão: {self.threshold}")
        
        logger.info("Modelo e configuração carregados com sucesso")
        logger.info(f"Tipos de features: {self.feature_types}")
        logger.info(f"Threshold: {self.threshold}")
    
    def load_feature_groups(self, filepath):
        """
        Carrega grupos de features de um arquivo CSV.
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            Dicionário com grupos de features
        """
        import csv
        from collections import defaultdict
        
        feature_groups = defaultdict(list)
        
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Pular cabeçalho
            
            for row in reader:
                group, feature = row
                feature_groups[group].append(feature)
        
        return dict(feature_groups)
    
    def sanitize_column_names(self, df):
        """
        Sanitiza nomes das colunas para corresponder aos nomes esperados pelo modelo.
        
        Args:
            df: DataFrame para sanitizar
            
        Returns:
            Dicionário mapeando nomes originais para sanitizados
        """
        import re
        
        sanitized_columns = {}
        for col in df.columns:
            new_col = re.sub(r'[^\w\s]', '_', col)
            new_col = re.sub(r'\s+', '_', new_col)
            if new_col in sanitized_columns.values():
                new_col = f"{new_col}_{df.columns.get_loc(col)}"
            sanitized_columns[col] = new_col
        
        df.rename(columns=sanitized_columns, inplace=True)
        return sanitized_columns
    
    def prepare_features(self, df, sanitize=True):
        """
        Prepara features para o modelo de stacking.
        
        Args:
            df: DataFrame com dados para predição
            sanitize: Se True, sanitiza nomes das colunas
            
        Returns:
            Dicionário com features por grupo
        """
        # Sanitizar nomes das colunas, se solicitado
        if sanitize:
            self.sanitize_column_names(df)
        
        # Criar dicionário de features por grupo
        X = {}
        
        for group, features in self.feature_groups.items():
            # Verificar se este grupo está sendo usado pelo modelo
            if group in self.feature_types:
                # Identificar features presentes no DataFrame
                valid_features = [f for f in features if f in df.columns]
                
                if not valid_features:
                    logger.warning(f"Nenhuma feature do grupo '{group}' encontrada no DataFrame")
                    continue
                
                X[group] = df[valid_features].copy()
                
                # Verificar se há tipos inteiros e converter para float
                for col in X[group].columns:
                    if pd.api.types.is_integer_dtype(X[group][col].dtype):
                        X[group][col] = X[group][col].astype(float)
        
        # Verificar se todos os grupos necessários estão presentes
        missing_groups = set(self.feature_types) - set(X.keys())
        if missing_groups:
            logger.warning(f"Grupos de features ausentes: {missing_groups}")
        
        return X
    
    def predict(self, df, return_proba=False):
        """
        Faz previsões usando o modelo de stacking.
        
        Args:
            df: DataFrame com dados para predição
            return_proba: Se True, retorna probabilidades em vez de classes
            
        Returns:
            Array com previsões
        """
        # Medir tempo
        start_time = time.time()
        
        # Preparar features
        X = self.prepare_features(df)
        
        # Fazer predição
        try:
            if return_proba:
                y_pred = self.model.predict_proba(X)[:, 1]
            else:
                y_pred = self.model.predict(X, threshold=self.threshold)
        except Exception as e:
            logger.error(f"Erro ao fazer predição: {e}")
            raise
        
        # Calcular tempo
        elapsed_time = time.time() - start_time
        logger