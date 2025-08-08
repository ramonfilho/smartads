# smart_ads_pipeline/core/base_component.py

from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd
import os
import sys
import logging

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.parameter_manager import ParameterManager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseComponent(ABC):
    """
    Classe base para todos os componentes do pipeline.
    Implementa interface similar ao scikit-learn com fit/transform.
    """
    
    def __init__(self, name: str):
        """
        Inicializa o componente.
        
        Args:
            name: Nome do componente (usado para logging e salvamento de parâmetros)
        """
        self.name = name
        self.is_fitted = False
        logger.info(f"Inicializando componente: {self.name}")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseComponent':
        """
        Aprende parâmetros a partir dos dados de treino.
        
        Args:
            X: DataFrame com features
            y: Series com target (opcional)
            
        Returns:
            self: Retorna a própria instância (para permitir chaining)
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma os dados usando parâmetros aprendidos.
        
        Args:
            X: DataFrame para transformar
            
        Returns:
            DataFrame transformado
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit seguido de transform (conveniência).
        
        Args:
            X: DataFrame com features
            y: Series com target (opcional)
            
        Returns:
            DataFrame transformado
        """
        logger.info(f"{self.name}: Executando fit_transform")
        return self.fit(X, y).transform(X)
    
    def save_params(self, param_manager: ParameterManager) -> None:
        """
        Salva parâmetros do componente no ParameterManager.
        
        Args:
            param_manager: Instância do ParameterManager
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name}: Componente deve ser fitted antes de salvar parâmetros")
        
        logger.info(f"{self.name}: Salvando parâmetros")
        self._save_component_params(param_manager)
    
    def load_params(self, param_manager: ParameterManager) -> None:
        """
        Carrega parâmetros do componente do ParameterManager.
        
        Args:
            param_manager: Instância do ParameterManager
        """
        logger.info(f"{self.name}: Carregando parâmetros")
        self._load_component_params(param_manager)
        self.is_fitted = True
    
    @abstractmethod
    def _save_component_params(self, param_manager: ParameterManager) -> None:
        """
        Implementação específica para salvar parâmetros.
        Deve ser implementado por cada componente.
        """
        pass
    
    @abstractmethod
    def _load_component_params(self, param_manager: ParameterManager) -> None:
        """
        Implementação específica para carregar parâmetros.
        Deve ser implementado por cada componente.
        """
        pass
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validação básica de entrada.
        
        Args:
            X: DataFrame para validar
            
        Raises:
            ValueError: Se a entrada não for válida
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"{self.name}: Entrada deve ser um pandas DataFrame")
        
        if X.empty:
            raise ValueError(f"{self.name}: DataFrame está vazio")
        
        logger.debug(f"{self.name}: Entrada validada - shape: {X.shape}")
    
    def _check_is_fitted(self) -> None:
        """
        Verifica se o componente foi fitted.
        
        Raises:
            ValueError: Se o componente não foi fitted
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name}: Componente deve ser fitted antes de transform")
    
    @classmethod
    def from_params(cls, param_manager: ParameterManager, **kwargs) -> 'BaseComponent':
        """
        Cria uma instância do componente a partir de parâmetros salvos.
        
        Args:
            param_manager: ParameterManager com parâmetros salvos
            **kwargs: Argumentos adicionais para o construtor
            
        Returns:
            Instância do componente com parâmetros carregados
        """
        instance = cls(**kwargs)
        instance.load_params(param_manager)
        return instance