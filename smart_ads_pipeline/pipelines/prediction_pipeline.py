# smart_ads_pipeline/pipelines/prediction_pipeline.py

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, Union
import sys

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smart_ads_pipeline.core import PipelineState, ExtendedParameterManager
from smart_ads_pipeline.components import (
    DataPreprocessor, FeatureEngineer, TextProcessor, 
    ProfessionalFeatures, FeatureSelector
)

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Pipeline para fazer predições em novos dados.
    
    Carrega parâmetros salvos e aplica todas as transformações
    exatamente como foram aprendidas durante o treino.
    """
    
    def __init__(self, params_path: str, model_path: Optional[str] = None):
        """
        Inicializa o pipeline de predição.
        
        Args:
            params_path: Caminho para os parâmetros salvos
            model_path: Caminho para o modelo treinado (opcional)
        """
        logger.info("Inicializando PredictionPipeline")
        
        self.params_path = Path(params_path)
        self.model_path = Path(model_path) if model_path else None
        
        # Verificar se parâmetros existem
        if not self.params_path.exists():
            raise FileNotFoundError(f"Parâmetros não encontrados: {params_path}")
        
        # Carregar parâmetros
        logger.info(f"Carregando parâmetros de: {params_path}")
        self.param_manager = ExtendedParameterManager()
        self.param_manager = joblib.load(params_path)
        
        # Carregar modelo se fornecido
        self.model = None
        if self.model_path and self.model_path.exists():
            logger.info(f"Carregando modelo de: {model_path}")
            self.model = joblib.load(model_path)
        
        # Inicializar componentes com parâmetros salvos
        self._initialize_components()
        
        # Obter features selecionadas
        self.selected_features = self.param_manager.get_selected_features()
        if not self.selected_features:
            raise ValueError("Nenhuma feature selecionada encontrada nos parâmetros")
        
        logger.info(f"Pipeline inicializado com {len(self.selected_features)} features")
    
    def _initialize_components(self):
        """Inicializa todos os componentes com parâmetros salvos."""
        logger.info("Inicializando componentes...")
        
        # DataPreprocessor
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_params(self.param_manager)
        
        # FeatureEngineer
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.load_params(self.param_manager)
        
        # TextProcessor
        self.text_processor = TextProcessor()
        self.text_processor.load_params(self.param_manager)
        
        # ProfessionalFeatures
        self.professional_features = ProfessionalFeatures()
        self.professional_features.load_params(self.param_manager)
        
        logger.info("Componentes inicializados com sucesso")
    
    def predict(self, df: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Faz predições em novos dados.
        
        Args:
            df: DataFrame com novos dados (formato de surveys)
            return_proba: Se True, retorna probabilidades de ambas as classes
            
        Returns:
            Array com predições ou probabilidades
        """
        logger.info(f"Iniciando predição para {len(df)} registros")
        
        # Validar entrada
        if df.empty:
            raise ValueError("DataFrame vazio fornecido")
        
        # Criar cópia para não modificar original
        df_work = df.copy()
        
        # Aplicar transformações
        df_work = self._apply_transformations(df_work)
        
        # Filtrar apenas features selecionadas
        logger.info(f"Filtrando para {len(self.selected_features)} features selecionadas")
        
        # Garantir que temos todas as features necessárias
        missing_features = set(self.selected_features) - set(df_work.columns)
        if missing_features:
            logger.warning(f"Features ausentes: {len(missing_features)}")
            # Adicionar features ausentes com zeros
            for feat in missing_features:
                df_work[feat] = 0
        
        # Manter apenas features selecionadas na ordem correta
        df_final = df_work[self.selected_features]
        
        # Fazer predições se modelo disponível
        if self.model:
            logger.info("Realizando predições com modelo")
            if return_proba:
                # Retorna probabilidades de ambas as classes [prob_classe_0, prob_classe_1]
                predictions = self.model.predict_proba(df_final)
                logger.info(f"Predições concluídas: {predictions.shape}")
            else:
                predictions = self.model.predict(df_final)
                logger.info(f"Predições concluídas: {len(predictions)} resultados")
            
            return predictions
        else:
            logger.warning("Nenhum modelo carregado - retornando features processadas")
            return df_final
    
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as transformações na ordem correta.
        
        Args:
            df: DataFrame para transformar
            
        Returns:
            DataFrame transformado
        """
        logger.info("Aplicando transformações...")
        
        # 1. Preprocessamento
        logger.info("Aplicando DataPreprocessor...")
        df = self.preprocessor.transform(df)
        logger.info(f"Após preprocessamento: {df.shape}")
        
        # 2. Feature Engineering
        logger.info("Aplicando FeatureEngineer...")
        df = self.feature_engineer.transform(df)
        logger.info(f"Após feature engineering: {df.shape}")
        
        # 3. Text Processing
        logger.info("Aplicando TextProcessor...")
        df = self.text_processor.transform(df)
        logger.info(f"Após text processing: {df.shape}")
        
        # 4. Professional Features
        logger.info("Aplicando ProfessionalFeatures...")
        df = self.professional_features.transform(df)
        logger.info(f"Após professional features: {df.shape}")
        
        return df
    
    def validate_input(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida se o DataFrame de entrada tem as colunas esperadas.
        
        Args:
            df: DataFrame para validar
            
        Returns:
            Dicionário com relatório de validação
        """
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Colunas esperadas (de INFERENCE_COLUMNS)
        expected_columns = [
            'marca_temporal', 'cual_es_tu_e_mail', 'cual_es_tu_genero',
            'cual_es_tu_edad', 'cual_es_tu_pais', 'utm_source', 'utm_medium',
            'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida_que_oportunidades_se_abriran_para_ti',
            'que_esperas_aprender_en_el_evento_cero_a_ingles_fluido',
            'dejame_un_mensaje'
        ]
        
        # Verificar colunas críticas
        missing_critical = []
        for col in expected_columns[:5]:  # Primeiras 5 são mais críticas
            if col not in df.columns:
                missing_critical.append(col)
        
        if missing_critical:
            report['is_valid'] = False
            report['errors'].append(f"Colunas críticas ausentes: {missing_critical}")
        
        # Verificar colunas recomendadas
        all_missing = set(expected_columns) - set(df.columns)
        if all_missing:
            report['warnings'].append(f"Colunas recomendadas ausentes: {list(all_missing)}")
        
        # Informações sobre o dataset
        report['info'].append(f"Shape: {df.shape}")
        report['info'].append(f"Colunas presentes: {len(df.columns)}")
        
        return report
    
    def get_feature_names(self) -> list:
        """Retorna lista de features usadas pelo modelo."""
        return self.selected_features
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo do pipeline."""
        return {
            'n_features': len(self.selected_features),
            'has_model': self.model is not None,
            'params_path': str(self.params_path),
            'model_path': str(self.model_path) if self.model_path else None,
            'components': [
                'DataPreprocessor',
                'FeatureEngineer',
                'TextProcessor',
                'ProfessionalFeatures'
            ]
        }