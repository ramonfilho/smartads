# smart_ads_pipeline/core/extended_parameter_manager.py

import os
import sys
import joblib
import logging
from typing import Dict, Any, List

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.parameter_manager import ParameterManager

logger = logging.getLogger(__name__)


class ExtendedParameterManager:
    """
    Extensão do ParameterManager original para suportar componentes OOP.
    
    Esta classe encapsula o ParameterManager original e adiciona funcionalidades
    para salvar/carregar parâmetros específicos de componentes.
    """
    
    def __init__(self):
        """Inicializa o manager com um ParameterManager interno."""
        self._internal_param_manager = ParameterManager()
        self.component_params = {}
        
        # Garantir estrutura de componentes
        if 'components' not in self._internal_param_manager.params:
            self._internal_param_manager.params['components'] = {}
            
    @property
    def params(self):
        """Acesso direto aos parâmetros internos para compatibilidade."""
        return self._internal_param_manager.params
    
    def save_component_params(self, component_name: str, params: Dict[str, Any]) -> None:
        """
        Salva parâmetros de um componente específico.
        
        Args:
            component_name: Nome do componente
            params: Dicionário com parâmetros do componente
        """
        self.params['components'][component_name] = params
        self.component_params[component_name] = params
        self._update_timestamp()
        logger.debug(f"Parâmetros salvos para componente: {component_name}")
    
    def get_component_params(self, component_name: str) -> Dict[str, Any]:
        """
        Recupera parâmetros de um componente específico.
        
        Args:
            component_name: Nome do componente
            
        Returns:
            Dicionário com parâmetros do componente
        """
        # Tentar primeiro dos params internos, depois do component_params
        params = self.params['components'].get(component_name, {})
        if not params:
            params = self.component_params.get(component_name, {})
        return params
    
    def has_component_params(self, component_name: str) -> bool:
        """
        Verifica se existem parâmetros salvos para um componente.
        
        Args:
            component_name: Nome do componente
            
        Returns:
            True se existem parâmetros salvos
        """
        return component_name in self.params['components'] or component_name in self.component_params
    
    def save_selected_features(self, features: List[str]) -> None:
        """
        Salva lista de features selecionadas.
        
        Args:
            features: Lista com nomes das features selecionadas
        """
        # Garantir estrutura
        if 'feature_selection' not in self.params:
            self.params['feature_selection'] = {}
            
        self.params['feature_selection']['selected_features'] = features
        self.params['feature_selection']['n_features'] = len(features)
        self._update_timestamp()
        
        # Salvar também na estrutura que o pipeline original espera
        if hasattr(self._internal_param_manager, 'params'):
            if 'feature_selection' not in self._internal_param_manager.params:
                self._internal_param_manager.params['feature_selection'] = {}
            self._internal_param_manager.params['feature_selection']['selected_features'] = features
    
    def get_selected_features(self) -> List[str]:
        """
        Recupera lista de features selecionadas.
        
        Returns:
            Lista com nomes das features selecionadas
        """
        # Tentar primeiro da estrutura local
        features = self.params.get('feature_selection', {}).get('selected_features', [])
        if features:
            return features
            
        # Tentar do param_manager interno
        if hasattr(self._internal_param_manager, 'params'):
            features = self._internal_param_manager.params.get('feature_selection', {}).get('selected_features', [])
            
        return features
    
    def save_preprocessing_params(self, key: str, value: Any) -> None:
        """Salva parâmetros de pré-processamento."""
        self._internal_param_manager.save_preprocessing_params(key, value)
    
    def get_preprocessing_params(self, key: str) -> Any:
        """Recupera parâmetros de pré-processamento."""
        return self._internal_param_manager.get_preprocessing_params(key)
    
    def save_text_params(self, key: str, value: Any) -> None:
        """Salva parâmetros de processamento de texto."""
        self._internal_param_manager.save_text_params(key, value)
    
    def get_text_params(self, key: str) -> Any:
        """Recupera parâmetros de processamento de texto."""
        return self._internal_param_manager.get_text_params(key)
    
    def get_component_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo dos componentes salvos.
        
        Returns:
            Dicionário com informações sobre componentes
        """
        summary = {
            'n_components': len(self.params.get('components', {})),
            'components': list(self.params.get('components', {}).keys()),
            'has_selected_features': len(self.get_selected_features()) > 0,
            'n_selected_features': len(self.get_selected_features())
        }
        
        # Adicionar detalhes de cada componente
        for comp_name, comp_params in self.params.get('components', {}).items():
            summary[f'{comp_name}_params'] = len(comp_params) if isinstance(comp_params, dict) else 0
        
        return summary
    
    def validate_for_prediction(self) -> List[str]:
        """
        Valida se os parâmetros necessários para predição estão presentes.
        
        Returns:
            Lista de problemas encontrados (vazia se tudo OK)
        """
        issues = []
        
        # Verificar componentes essenciais
        essential_components = [
            'data_preprocessor',
            'feature_engineer', 
            'text_processor',
            'feature_selector'
        ]
        
        for component in essential_components:
            if not self.has_component_params(component):
                issues.append(f"Parâmetros não encontrados para: {component}")
        
        # Verificar features selecionadas
        if not self.get_selected_features():
            issues.append("Lista de features selecionadas está vazia")
        
        # Verificar vetorizadores
        text_params = self.params.get('text_processing', {})
        if not text_params.get('tfidf_vectorizers'):
            issues.append("Vetorizadores TF-IDF não encontrados")
        
        return issues
    
    def save(self, filepath: str) -> None:
        """
        Salva todos os parâmetros em arquivo.
        
        Args:
            filepath: Caminho do arquivo
        """
        # Salvar parâmetros do manager original
        self._internal_param_manager.save(filepath)
        
        # Salvar parâmetros de componentes em arquivo separado se houver
        if self.component_params:
            component_filepath = filepath.replace('.joblib', '_components.joblib')
            joblib.dump(self.component_params, component_filepath)
            logger.info(f"Parâmetros de componentes salvos em: {component_filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carrega parâmetros de arquivo.
        
        Args:
            filepath: Caminho do arquivo
        """
        # Carregar parâmetros do manager original
        self._internal_param_manager.load(filepath)
        
        # Carregar parâmetros de componentes se existir
        component_filepath = filepath.replace('.joblib', '_components.joblib')
        if os.path.exists(component_filepath):
            self.component_params = joblib.load(component_filepath)
            logger.info(f"Parâmetros de componentes carregados de: {component_filepath}")
    
    def _update_timestamp(self):
        """Atualiza timestamp interno quando houver mudanças."""
        if hasattr(self._internal_param_manager, '_update_timestamp'):
            self._internal_param_manager._update_timestamp()
    
    def __getattr__(self, name):
        """
        Delega atributos não encontrados para o ParameterManager interno.
        Isso garante compatibilidade total com o ParameterManager original.
        """
        return getattr(self._internal_param_manager, name)