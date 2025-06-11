# smart_ads_pipeline/components/text_processor.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import sys

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smart_ads_pipeline.core import BaseComponent, ExtendedParameterManager

logger = logging.getLogger(__name__)


class TextProcessor(BaseComponent):
    """
    Componente responsável pelo processamento de texto usando TF-IDF e features derivadas.
    
    IMPORTANTE: Este componente é um wrapper OOP das funções existentes em
    src.preprocessing.text_processing para garantir 100% de compatibilidade.
    
    Responsabilidades:
    - Limpar texto
    - Extrair features básicas (comprimento, palavras, etc)
    - Análise de sentimento
    - TF-IDF
    - Features de motivação
    - Features discriminativas
    """
    
    def __init__(self):
        super().__init__(name="text_processor")
        
        # Este componente não mantém estado próprio
        # Tudo é gerenciado pelo ParameterManager através das funções originais
        self._param_manager = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TextProcessor':
        """
        Aprende parâmetros de processamento de texto.
        
        IMPORTANTE: Chama exatamente a mesma função text_feature_engineering
        do pipeline original com fit=True.
        
        Args:
            X: DataFrame de treino
            y: Target (não usado diretamente, mas passado para discriminative features)
            
        Returns:
            self
        """
        self._validate_input(X)
        logger.info(f"{self.name}: Iniciando fit com shape {X.shape}")
        
        # Importar a função original
        from src.preprocessing.text_processing import text_feature_engineering
        
        # Criar ParameterManager se não existir
        if self._param_manager is None:
            from src.utils.parameter_manager import ParameterManager
            self._param_manager = ParameterManager()
        
        # Chamar EXATAMENTE a mesma função do pipeline original
        X_processed, updated_param_manager = text_feature_engineering(
            df=X.copy(),  # Copiar para não modificar original
            fit=True,
            param_manager=self._param_manager
        )
        
        # Atualizar param_manager
        self._param_manager = updated_param_manager
        
        # Marcar como fitted
        self.is_fitted = True
        logger.info(f"{self.name}: Fit concluído. Shape resultante: {X_processed.shape}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma dados usando parâmetros aprendidos.
        
        IMPORTANTE: Chama exatamente a mesma função text_feature_engineering
        do pipeline original com fit=False.
        
        Args:
            X: DataFrame para transformar
            
        Returns:
            DataFrame transformado com features de texto
        """
        self._check_is_fitted()
        self._validate_input(X)
        logger.info(f"{self.name}: Iniciando transform com shape {X.shape}")
        
        # Verificar que temos param_manager
        if self._param_manager is None:
            raise ValueError(f"{self.name}: ParameterManager não encontrado. Execute fit primeiro.")
        
        # Importar a função original
        from src.preprocessing.text_processing import text_feature_engineering
        
        # Chamar EXATAMENTE a mesma função do pipeline original
        X_processed, _ = text_feature_engineering(
            df=X.copy(),  # Copiar para não modificar original
            fit=False,
            param_manager=self._param_manager
        )
        
        logger.info(f"{self.name}: Transform concluído. Shape resultante: {X_processed.shape}")
        
        return X_processed
    
    def _save_component_params(self, param_manager: ExtendedParameterManager) -> None:
        """
        Salva parâmetros do componente.
        
        NOTA: Os parâmetros reais já foram salvos pela função text_feature_engineering
        no ParameterManager. Aqui apenas salvamos uma referência.
        """
        # Transferir o ParameterManager interno para o ExtendedParameterManager
        if self._param_manager is not None:
            # Copiar parâmetros relevantes
            param_manager.params['text_processing'] = self._param_manager.params.get('text_processing', {})
            
            # Salvar flag indicando que foi fitted
            param_manager.save_component_params(self.name, {
                'is_fitted': True,
                'has_text_params': True
            })
    
    def _load_component_params(self, param_manager: ExtendedParameterManager) -> None:
        """
        Carrega parâmetros do componente.
        
        NOTA: Carrega os parâmetros salvos pela função text_feature_engineering.
        """
        # Criar ParameterManager interno e copiar parâmetros
        from src.utils.parameter_manager import ParameterManager
        self._param_manager = ParameterManager()
        
        # Copiar parâmetros de text_processing
        if 'text_processing' in param_manager.params:
            self._param_manager.params['text_processing'] = param_manager.params['text_processing']
        
        # Verificar se foi fitted
        component_params = param_manager.get_component_params(self.name)
        if not component_params.get('has_text_params', False):
            raise ValueError(f"{self.name}: Parâmetros de texto não encontrados")
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre as features criadas.
        
        Returns:
            Dicionário com informações sobre features criadas
        """
        if self._param_manager is None:
            return {}
        
        text_params = self._param_manager.params.get('text_processing', {})
        
        info = {
            'n_tfidf_vectorizers': len(text_params.get('tfidf_vectorizers', {})),
            'has_discriminative_terms': 'discriminative_terms' in text_params,
            'text_columns_processed': list(text_params.get('tfidf_vectorizers', {}).keys())
        }
        
        # Contar features TF-IDF criadas
        tfidf_feature_count = 0
        for col, vectorizer_data in text_params.get('tfidf_vectorizers', {}).items():
            if 'feature_names' in vectorizer_data:
                tfidf_feature_count += len(vectorizer_data['feature_names'])
        
        info['total_tfidf_features'] = tfidf_feature_count
        
        return info