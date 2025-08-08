# smart_ads_pipeline/core/pipeline_state.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """
    Mantém o estado completo do pipeline.
    
    Esta classe armazena todos os dados e metadados necessários
    durante a execução do pipeline, tanto para treino quanto predição.
    """
    
    # Dados principais
    train_df: Optional[pd.DataFrame] = None
    val_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    
    # Para predição
    input_df: Optional[pd.DataFrame] = None
    predictions: Optional[pd.DataFrame] = None
    
    # Parâmetros e configurações
    param_manager: Optional[Any] = None  # ParameterManager instance
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadados do processo
    selected_features: List[str] = field(default_factory=list)
    feature_importance: Optional[pd.DataFrame] = None
    excluded_columns: List[str] = field(default_factory=list)
    
    # Rastreamento de execução
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    # Métricas e resultados
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def log_step(self, step_name: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Registra uma etapa executada no pipeline.
        
        Args:
            step_name: Nome da etapa
            details: Detalhes adicionais sobre a execução
        """
        entry = {
            'step': step_name,
            'timestamp': datetime.now(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'details': details or {}
        }
        self.execution_history.append(entry)
        logger.info(f"Pipeline step completed: {step_name}")
    
    def get_active_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Retorna o DataFrame ativo baseado no contexto.
        
        Returns:
            DataFrame ativo (train_df para treino, input_df para predição)
        """
        if self.input_df is not None:
            return self.input_df
        return self.train_df
    
    def update_dataframes(self, train: Optional[pd.DataFrame] = None,
                         val: Optional[pd.DataFrame] = None,
                         test: Optional[pd.DataFrame] = None,
                         input_df: Optional[pd.DataFrame] = None) -> None:
        """
        Atualiza os DataFrames no estado.
        
        Args:
            train: DataFrame de treino
            val: DataFrame de validação
            test: DataFrame de teste
            input_df: DataFrame de entrada (para predição)
        """
        if train is not None:
            self.train_df = train
            logger.debug(f"Train DataFrame atualizado: shape {train.shape}")
        
        if val is not None:
            self.val_df = val
            logger.debug(f"Validation DataFrame atualizado: shape {val.shape}")
        
        if test is not None:
            self.test_df = test
            logger.debug(f"Test DataFrame atualizado: shape {test.shape}")
        
        if input_df is not None:
            self.input_df = input_df
            logger.debug(f"Input DataFrame atualizado: shape {input_df.shape}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo do estado atual.
        
        Returns:
            Dicionário com informações resumidas
        """
        summary = {
            'data_shapes': {
                'train': self.train_df.shape if self.train_df is not None else None,
                'val': self.val_df.shape if self.val_df is not None else None,
                'test': self.test_df.shape if self.test_df is not None else None,
                'input': self.input_df.shape if self.input_df is not None else None,
            },
            'n_selected_features': len(self.selected_features),
            'n_excluded_columns': len(self.excluded_columns),
            'execution_time_seconds': (datetime.now() - self.start_time).total_seconds(),
            'n_steps_executed': len(self.execution_history),
            'has_predictions': self.predictions is not None
        }
        
        if self.metrics:
            summary['metrics'] = self.metrics
        
        return summary
    
    def validate_for_training(self) -> None:
        """
        Valida se o estado tem os dados necessários para treino.
        
        Raises:
            ValueError: Se dados essenciais estiverem faltando
        """
        if self.train_df is None:
            raise ValueError("train_df é necessário para treino")
        
        if 'target' not in self.train_df.columns:
            raise ValueError("Coluna 'target' não encontrada no train_df")
        
        logger.info("Estado validado para treino")
    
    def validate_for_prediction(self) -> None:
        """
        Valida se o estado tem os dados necessários para predição.
        
        Raises:
            ValueError: Se dados essenciais estiverem faltando
        """
        if self.input_df is None:
            raise ValueError("input_df é necessário para predição")
        
        if self.param_manager is None:
            raise ValueError("param_manager é necessário para predição")
        
        if not self.selected_features:
            raise ValueError("selected_features é necessário para predição")
        
        logger.info("Estado validado para predição")