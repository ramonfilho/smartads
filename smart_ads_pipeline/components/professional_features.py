# smart_ads_pipeline/components/professional_features.py

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


class ProfessionalFeatures(BaseComponent):
    """
    Componente responsável pela criação de features profissionais avançadas.
    
    IMPORTANTE: Este componente é um wrapper OOP das funções existentes em
    src.preprocessing.professional_motivation_features para garantir 100% de compatibilidade.
    
    Responsabilidades:
    - Score de motivação profissional
    - Análise de sentimento de aspiração
    - Detecção de expressões de compromisso
    - Detector de termos de carreira
    - TF-IDF aprimorado para termos de carreira
    - LDA para extração de tópicos
    """
    
    def __init__(self):
        super().__init__(name="professional_features")
        
        # Este componente não mantém estado próprio
        # Tudo é gerenciado pelo ParameterManager através das funções originais
        self._param_manager = None
        self._text_columns = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ProfessionalFeatures':
        """
        Aprende parâmetros para features profissionais.
        
        IMPORTANTE: Chama exatamente as mesmas funções do pipeline original
        com fit=True.
        
        Args:
            X: DataFrame de treino
            y: Target (não usado)
            
        Returns:
            self
        """
        self._validate_input(X)
        logger.info(f"{self.name}: Iniciando fit com shape {X.shape}")
        
        # Importar funções necessárias
        from src.preprocessing.professional_motivation_features import (
            create_professional_motivation_score,
            analyze_aspiration_sentiment,
            detect_commitment_expressions,
            create_career_term_detector,
            enhance_tfidf_for_career_terms
        )
        
        # Criar ParameterManager se não existir
        if self._param_manager is None:
            from src.utils.parameter_manager import ParameterManager
            self._param_manager = ParameterManager()
        
        # Detectar colunas de texto usando a mesma lógica do pipeline original
        self._text_columns = self._identify_text_columns(X)
        
        if not self._text_columns:
            logger.warning(f"{self.name}: Nenhuma coluna de texto encontrada para processamento")
            self.is_fitted = True
            return self
        
        logger.info(f"{self.name}: Processando {len(self._text_columns)} colunas de texto")
        
        # Fazer cópia para não modificar original
        X_work = X.copy()
        
        # 1. Score de motivação profissional
        logger.info(f"{self.name}: Criando score de motivação profissional...")
        X_work, self._param_manager = create_professional_motivation_score(
            X_work, self._text_columns, fit=True, param_manager=self._param_manager
        )
        
        # 2. Análise de sentimento de aspiração
        logger.info(f"{self.name}: Analisando sentimento de aspiração...")
        aspiration_df, self._param_manager = analyze_aspiration_sentiment(
            X, self._text_columns, fit=True, param_manager=self._param_manager
        )
        X_work = pd.concat([X_work, aspiration_df], axis=1)
        
        # 3. Detecção de expressões de compromisso
        logger.info(f"{self.name}: Detectando expressões de compromisso...")
        commitment_df, self._param_manager = detect_commitment_expressions(
            X, self._text_columns, fit=True, param_manager=self._param_manager
        )
        X_work = pd.concat([X_work, commitment_df], axis=1)
        
        # 4. Detector de termos de carreira
        logger.info(f"{self.name}: Criando detector de termos de carreira...")
        career_df, self._param_manager = create_career_term_detector(
            X, self._text_columns, fit=True, param_manager=self._param_manager
        )
        X_work = pd.concat([X_work, career_df], axis=1)
        
        # 5. TF-IDF aprimorado para termos de carreira
        logger.info(f"{self.name}: Aplicando TF-IDF aprimorado para termos de carreira...")
        tfidf_df, self._param_manager = enhance_tfidf_for_career_terms(
            X, self._text_columns, fit=True, param_manager=self._param_manager
        )
        X_work = pd.concat([X_work, tfidf_df], axis=1)
        
        # Marcar como fitted
        self.is_fitted = True
        
        # Contar features criadas
        features_created = len(X_work.columns) - len(X.columns)
        logger.info(f"{self.name}: Fit concluído. {features_created} features profissionais criadas")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma dados usando parâmetros aprendidos.
        
        IMPORTANTE: Chama exatamente as mesmas funções do pipeline original
        com fit=False.
        
        Args:
            X: DataFrame para transformar
            
        Returns:
            DataFrame transformado com features profissionais
        """
        self._check_is_fitted()
        self._validate_input(X)
        logger.info(f"{self.name}: Iniciando transform com shape {X.shape}")
        
        # Verificar que temos param_manager
        if self._param_manager is None:
            raise ValueError(f"{self.name}: ParameterManager não encontrado. Execute fit primeiro.")
        
        # Importar funções necessárias
        from src.preprocessing.professional_motivation_features import (
            create_professional_motivation_score,
            analyze_aspiration_sentiment,
            detect_commitment_expressions,
            create_career_term_detector,
            enhance_tfidf_for_career_terms
        )
        
        # Detectar colunas de texto
        text_columns = self._identify_text_columns(X)
        
        if not text_columns:
            logger.warning(f"{self.name}: Nenhuma coluna de texto encontrada para transformar")
            return X
        
        # Fazer cópia
        X_result = X.copy()
        
        # 1. Score de motivação profissional
        motiv_df, _ = create_professional_motivation_score(
            X, text_columns, fit=False, param_manager=self._param_manager
        )
        X_result = pd.concat([X_result, motiv_df], axis=1)
        
        # 2. Análise de sentimento de aspiração
        aspiration_df, _ = analyze_aspiration_sentiment(
            X, text_columns, fit=False, param_manager=self._param_manager
        )
        X_result = pd.concat([X_result, aspiration_df], axis=1)
        
        # 3. Detecção de expressões de compromisso
        commitment_df, _ = detect_commitment_expressions(
            X, text_columns, fit=False, param_manager=self._param_manager
        )
        X_result = pd.concat([X_result, commitment_df], axis=1)
        
        # 4. Detector de termos de carreira
        career_df, _ = create_career_term_detector(
            X, text_columns, fit=False, param_manager=self._param_manager
        )
        X_result = pd.concat([X_result, career_df], axis=1)
        
        # 5. TF-IDF aprimorado para termos de carreira
        tfidf_df, _ = enhance_tfidf_for_career_terms(
            X, text_columns, fit=False, param_manager=self._param_manager
        )
        X_result = pd.concat([X_result, tfidf_df], axis=1)
        
        # Remover duplicatas de colunas se houver
        X_result = X_result.loc[:, ~X_result.columns.duplicated()]
        
        features_created = len(X_result.columns) - len(X.columns)
        logger.info(f"{self.name}: Transform concluído. {features_created} features profissionais aplicadas")
        
        return X_result
    
    def _identify_text_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identifica colunas de texto para processamento.
        Usa a mesma lógica do pipeline original.
        """
        # Padrões de colunas de texto conhecidas
        text_patterns = [
            'cuando_hables_ingles',
            'que_esperas_aprender',
            'dejame_un_mensaje',
            'que_cambiara_en_tu_vida',
            'oportunidades_se_abriran'
        ]
        
        text_columns = []
        for col in df.columns:
            # Verificar se é uma coluna de texto conhecida
            if any(pattern in col.lower() for pattern in text_patterns):
                text_columns.append(col)
        
        return text_columns
    
    def _save_component_params(self, param_manager: ExtendedParameterManager) -> None:
        """
        Salva parâmetros do componente.
        
        NOTA: Os parâmetros reais já foram salvos pelas funções originais
        no ParameterManager. Aqui apenas salvamos uma referência.
        """
        # Transferir parâmetros profissionais
        if self._param_manager is not None:
            # Copiar parâmetros relevantes
            if 'professional_features' in self._param_manager.params:
                param_manager.params['professional_features'] = self._param_manager.params['professional_features']
            
            # Salvar flag indicando que foi fitted
            param_manager.save_component_params(self.name, {
                'is_fitted': True,
                'has_professional_params': True,
                'text_columns': self._text_columns
            })
    
    def _load_component_params(self, param_manager: ExtendedParameterManager) -> None:
        """
        Carrega parâmetros do componente.
        
        NOTA: Carrega os parâmetros salvos pelas funções originais.
        """
        # Criar ParameterManager interno e copiar parâmetros
        from src.utils.parameter_manager import ParameterManager
        self._param_manager = ParameterManager()
        
        # Copiar parâmetros profissionais
        if 'professional_features' in param_manager.params:
            self._param_manager.params['professional_features'] = param_manager.params['professional_features']
        
        # Verificar componente específico
        component_params = param_manager.get_component_params(self.name)
        if not component_params.get('has_professional_params', False):
            raise ValueError(f"{self.name}: Parâmetros profissionais não encontrados")
        
        self._text_columns = component_params.get('text_columns', [])
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre as features criadas.
        
        Returns:
            Dicionário com informações sobre features criadas
        """
        if self._param_manager is None:
            return {}
        
        prof_params = self._param_manager.params.get('professional_features', {})
        
        info = {
            'has_motivation_keywords': 'motivation_keywords' in prof_params,
            'has_aspiration_phrases': 'aspiration_phrases' in prof_params,
            'has_commitment_phrases': 'commitment_phrases' in prof_params,
            'has_career_terms': 'career_terms' in prof_params,
            'n_career_tfidf_vectorizers': len(prof_params.get('career_tfidf_vectorizers', {})),
            'text_columns_processed': self._text_columns
        }
        
        return info