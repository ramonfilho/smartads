# smart_ads_pipeline/core/extended_parameter_manager.py

import os
import sys
from typing import Dict, Any, List

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.parameter_manager import ParameterManager


class ExtendedParameterManager(ParameterManager):
    """
    Extensão do ParameterManager para a nova arquitetura OOP.
    Adiciona métodos específicos para componentes.
    """
    
    def __init__(self):
        super().__init__()
        # Adicionar nova seção para componentes
        if 'components' not in self.params:
            self.params['components'] = {}
    
    def save_component_params(self, component_name: str, params: Dict[str, Any]) -> None:
        """
        Salva parâmetros de um componente específico.
        
        Args:
            component_name: Nome do componente
            params: Dicionário com parâmetros do componente
        """
        self.params['components'][component_name] = params
        self._update_timestamp()
    
    def get_component_params(self, component_name: str) -> Dict[str, Any]:
        """
        Recupera parâmetros de um componente específico.
        
        Args:
            component_name: Nome do componente
            
        Returns:
            Dicionário com parâmetros do componente
        """
        return self.params['components'].get(component_name, {})
    
    def has_component_params(self, component_name: str) -> bool:
        """
        Verifica se existem parâmetros salvos para um componente.
        
        Args:
            component_name: Nome do componente
            
        Returns:
            True se existem parâmetros salvos
        """
        return component_name in self.params['components']
    
    def save_selected_features(self, features: List[str]) -> None:
        """
        Salva lista de features selecionadas.
        
        Args:
            features: Lista com nomes das features selecionadas
        """
        self.params['feature_selection']['selected_features'] = features
        self.params['feature_selection']['n_features'] = len(features)
        self._update_timestamp()
    
    def get_selected_features(self) -> List[str]:
        """
        Recupera lista de features selecionadas.
        
        Returns:
            Lista com nomes das features selecionadas
        """
        return self.params['feature_selection'].get('selected_features', [])
    
    def get_component_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo dos componentes salvos.
        
        Returns:
            Dicionário com informações sobre componentes
        """
        summary = {
            'n_components': len(self.params['components']),
            'components': list(self.params['components'].keys()),
            'has_selected_features': len(self.get_selected_features()) > 0,
            'n_selected_features': len(self.get_selected_features())
        }
        
        # Adicionar detalhes de cada componente
        for comp_name, comp_params in self.params['components'].items():
            summary[f'{comp_name}_params'] = len(comp_params)
        
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
        if not self.params['text_processing']['tfidf_vectorizers']:
            issues.append("Vetorizadores TF-IDF não encontrados")
        
        return issues