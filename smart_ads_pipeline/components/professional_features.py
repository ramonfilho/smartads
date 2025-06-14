# smart_ads_pipeline/components/professional_features.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import sys
import re

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
    
    def __init__(self, n_topics: int = 5):
        """
        Inicializa o componente.
        
        Args:
            n_topics: Número de tópicos para LDA
        """
        super().__init__(name="professional_features")
        
        self.n_topics = n_topics
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
        
        # IMPORTANTE: Trabalhar com cópia para preservar X original
        X_result = X.copy()
        
        # 1. Score de motivação profissional
        logger.info(f"{self.name}: Criando score de motivação profissional...")
        motiv_df, self._param_manager = create_professional_motivation_score(
            X, self._text_columns, fit=True, param_manager=self._param_manager
        )
        X_result = pd.concat([X_result, motiv_df], axis=1)
        
        # 2. Análise de sentimento de aspiração
        logger.info(f"{self.name}: Analisando sentimento de aspiração...")
        aspiration_df, self._param_manager = analyze_aspiration_sentiment(
            X, self._text_columns, fit=True, param_manager=self._param_manager
        )
        X_result = pd.concat([X_result, aspiration_df], axis=1)
        
        # 3. Detecção de expressões de compromisso
        logger.info(f"{self.name}: Detectando expressões de compromisso...")
        commitment_df, self._param_manager = detect_commitment_expressions(
            X, self._text_columns, fit=True, param_manager=self._param_manager
        )
        X_result = pd.concat([X_result, commitment_df], axis=1)
        
        # 4. Detector de termos de carreira
        logger.info(f"{self.name}: Criando detector de termos de carreira...")
        career_df, self._param_manager = create_career_term_detector(
            X, self._text_columns, fit=True, param_manager=self._param_manager
        )
        X_result = pd.concat([X_result, career_df], axis=1)
        
        # 5. TF-IDF aprimorado para termos de carreira
        logger.info(f"{self.name}: Aplicando TF-IDF aprimorado para termos de carreira...")
        tfidf_df, self._param_manager = enhance_tfidf_for_career_terms(
            X, self._text_columns, fit=True, param_manager=self._param_manager
        )
        X_result = pd.concat([X_result, tfidf_df], axis=1)
        
        # 6. LDA para extração de tópicos
        logger.info(f"{self.name}: Aplicando LDA para extração de tópicos...")
        # IMPORTANTE: Passar X original com as colunas de texto originais
        X_with_lda, self._param_manager = self._perform_topic_modeling(
            X, self._text_columns, n_topics=self.n_topics, fit=True
        )
        
        # Adicionar apenas as features LDA (não todo o DataFrame)
        lda_features = [col for col in X_with_lda.columns if col not in X.columns]
        if lda_features:
            X_result = pd.concat([X_result, X_with_lda[lda_features]], axis=1)
            logger.info(f"LDA adicionou {len(lda_features)} features")
        
        # Marcar como fitted
        self.is_fitted = True
        
        # Contar features criadas corretamente
        features_created = len(X_result.columns) - len(X.columns)
        logger.info(f"{self.name}: Fit concluído. {features_created} features profissionais criadas")
        
        # Guardar informação sobre features criadas para uso posterior
        self._features_created = features_created
        
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
        
        # 6. LDA para extração de tópicos (NOVO!)
        X_result, _ = self._perform_topic_modeling(
            X_result, text_columns, n_topics=self.n_topics, fit=False
        )
        
        # Remover duplicatas de colunas se houver
        X_result = X_result.loc[:, ~X_result.columns.duplicated()]
        
        features_created = len(X_result.columns) - len(X.columns)
        logger.info(f"{self.name}: Transform concluído. {features_created} features profissionais aplicadas")
        
        return X_result
    
    def _perform_topic_modeling(self, df: pd.DataFrame, text_cols: List[str], 
                               n_topics: int = 5, fit: bool = True) -> tuple:
        """
        Aplica LDA para extração de tópicos.
        
        Esta função é adaptada de perform_topic_modeling_fixed() do unified_pipeline.py
        """
        df_result = df.copy()
        
        # Importar função standardize_feature_name
        from src.utils.feature_naming import standardize_feature_name
        
        # Validar e filtrar colunas de texto
        valid_text_cols = []
        for col in text_cols:
            if col in df.columns:
                # Verificar se é realmente uma coluna de texto
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Tentar converter para string e verificar se tem conteúdo
                    try:
                        str_values = sample_values.astype(str)
                        if any(len(str(v)) > 5 for v in str_values):
                            valid_text_cols.append(col)
                            logger.debug(f"Coluna {col} validada para LDA")
                        else:
                            logger.debug(f"Coluna {col} não tem conteúdo suficiente para LDA")
                    except:
                        logger.debug(f"Coluna {col} não pode ser convertida para texto")
            else:
                logger.debug(f"Coluna {col} não encontrada no DataFrame")
        
        if not valid_text_cols:
            logger.warning("Nenhuma coluna de texto válida para LDA após validação")
            return df_result, self._param_manager
        
        logger.info(f"Aplicando LDA para {len(valid_text_cols)} colunas de texto com {n_topics} tópicos")
        
        for col_idx, col in enumerate(valid_text_cols):
            logger.debug(f"Processando LDA para coluna {col_idx+1}/{len(valid_text_cols)}: {col[:50]}...")
            
            col_clean = re.sub(r'[^\w]', '', col.replace(' ', '_'))[:30]
            
            # Verificar se temos texto válido
            texts = df[col].fillna('').astype(str)
            valid_texts = texts[texts.str.len() > 5]
            
            min_texts_for_lda = 20
            
            if len(valid_texts) < min_texts_for_lda:
                logger.debug(f"Poucos textos válidos para LDA em {col} ({len(valid_texts)} < {min_texts_for_lda})")
                continue
            
            if fit:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.decomposition import LatentDirichletAllocation
                    
                    # Vetorizar textos com parâmetros mais conservadores
                    vectorizer = TfidfVectorizer(
                        max_features=100,
                        min_df=2,  # Reduzido de 5 para 2
                        max_df=0.95,
                        stop_words=None,
                        token_pattern=r'(?u)\b\w+\b'  # Aceitar palavras de 1+ caracteres
                    )
                    
                    try:
                        doc_term_matrix = vectorizer.fit_transform(valid_texts)
                        
                        # Verificar se temos features suficientes
                        if doc_term_matrix.shape[1] < n_topics:
                            logger.warning(f"Vocabulário insuficiente para {n_topics} tópicos em {col}. "
                                         f"Apenas {doc_term_matrix.shape[1]} termos únicos.")
                            continue
                        
                        # Aplicar LDA
                        lda = LatentDirichletAllocation(
                            n_components=n_topics,
                            max_iter=20,
                            learning_method='online',
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        topic_dist_valid = lda.fit_transform(doc_term_matrix)
                        
                        # Criar distribuição completa
                        topic_distribution = np.zeros((len(df), n_topics))
                        valid_mask = texts.str.len() > 5
                        valid_positions = np.where(valid_mask)[0]
                        
                        for i, pos in enumerate(valid_positions):
                            topic_distribution[pos] = topic_dist_valid[i]
                        
                        # Salvar modelo LDA
                        self._param_manager.save_lda_model(
                            {
                                'model': lda,
                                'vectorizer': vectorizer,
                                'n_topics': n_topics,
                                'feature_names': vectorizer.get_feature_names_out().tolist()
                            },
                            name=col_clean
                        )
                        
                        # Adicionar features
                        for topic_idx in range(n_topics):
                            feature_name = standardize_feature_name(f'{col_clean}_topic_{topic_idx+1}')
                            df_result[feature_name] = topic_distribution[:, topic_idx]
                        
                        # Tópico dominante
                        dominant_topic_name = standardize_feature_name(f'{col_clean}_dominant_topic')
                        df_result[dominant_topic_name] = np.argmax(topic_distribution, axis=1) + 1
                        
                        logger.info(f"✓ LDA aplicado com sucesso em {col}: {n_topics + 1} features criadas")
                        
                    except Exception as e:
                        logger.warning(f"Erro ao vetorizar {col} para LDA: {e}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Erro ao aplicar LDA em {col}: {e}")
            
            else:  # transform mode
                model_data = self._param_manager.get_lda_model(col_clean)
                
                if not model_data:
                    logger.debug(f"Modelo LDA não encontrado para '{col_clean}'")
                    continue
                
                try:
                    lda = model_data['model']
                    vectorizer = model_data['vectorizer']
                    n_topics = model_data['n_topics']
                    
                    # Transformar
                    doc_term_matrix = vectorizer.transform(valid_texts)
                    topic_dist_valid = lda.transform(doc_term_matrix)
                    
                    # Criar distribuição completa
                    topic_distribution = np.zeros((len(df), n_topics))
                    valid_mask = texts.str.len() > 5
                    valid_positions = np.where(valid_mask)[0]
                    
                    for i, pos in enumerate(valid_positions):
                        topic_distribution[pos] = topic_dist_valid[i]
                    
                    # Adicionar features
                    for topic_idx in range(n_topics):
                        feature_name = standardize_feature_name(f'{col_clean}_topic_{topic_idx+1}')
                        df_result[feature_name] = topic_distribution[:, topic_idx]
                    
                    # Tópico dominante
                    dominant_topic_name = standardize_feature_name(f'{col_clean}_dominant_topic')
                    df_result[dominant_topic_name] = np.argmax(topic_distribution, axis=1) + 1
                    
                    logger.info(f"✓ LDA transform aplicado em {col}")
                    
                except Exception as e:
                    logger.error(f"Erro ao transformar com LDA em {col}: {e}")
        
        return df_result, self._param_manager
    
    def _identify_text_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identifica colunas de texto para processamento profissional.
        IMPORTANTE: Usa EXATAMENTE a mesma lógica de identify_text_columns_for_professional()
        do pipeline original para garantir 100% de compatibilidade.
        """
        # Opção 1: Tentar recuperar colunas preservadas do param_manager
        if self._param_manager:
            preserved_columns = self._param_manager.params.get('feature_engineering', {}).get('preserved_columns', {})
            if preserved_columns:
                # As colunas podem estar com sufixo _TEMP_PROF ou nome original
                text_columns = []
                for col_original in preserved_columns.keys():
                    # Verificar variações do nome da coluna
                    if f"{col_original}_TEMP_PROF" in df.columns:
                        text_columns.append(f"{col_original}_TEMP_PROF")
                    elif col_original in df.columns:
                        text_columns.append(col_original)
                
                if text_columns:
                    logger.info(f"Usando {len(text_columns)} colunas preservadas do param_manager")
                    return text_columns
        
        # Opção 2: Tentar recuperar classificações existentes
        if self._param_manager:
            classifications = self._param_manager.get_preprocessing_params('column_classifications')
            if classifications:
                logger.info("Usando classificações existentes do param_manager")
                # Usar exatamente a mesma lógica do text_processing.py
                text_columns = [
                    col for col, info in classifications.items()
                    if col in df.columns 
                    and info['type'] == 'text'
                    and info['confidence'] >= 0.7
                ]
                
                if text_columns:
                    return text_columns
        
        # Se chegou aqui, é um erro - não devemos classificar novamente
        logger.error("ERRO: Nenhuma classificação de colunas encontrada no param_manager!")
        logger.error("O ProfessionalFeatures requer que as colunas já tenham sido classificadas anteriormente.")
        raise ValueError(
            "ProfessionalFeatures requer classificações prévias de colunas. "
            "Execute o pipeline completo ou garanta que column_classifications "
            "esteja disponível no param_manager."
        )
    
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
                'text_columns': self._text_columns,
                'n_topics': self.n_topics
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
        self.n_topics = component_params.get('n_topics', 5)
    
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
            'n_lda_models': len(prof_params.get('lda_models', {})),
            'text_columns_processed': self._text_columns,
            'n_topics': self.n_topics
        }
        
        return info