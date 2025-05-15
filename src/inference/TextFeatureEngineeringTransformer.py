# src/inference/TextFeatureEngineeringTransformer.py
#!/usr/bin/env python
"""
Transformador para processamento de texto conforme implementado
nos scripts 03_feature_engineering_1.py e 04_feature_engineering_2.py.
"""

import os
import pandas as pd
import numpy as np
import re
import joblib
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ignorar warnings
warnings.filterwarnings('ignore')

# Importar módulos de processamento de texto
from src.preprocessing.text_processing import text_feature_engineering, clean_text
from src.preprocessing.advanced_feature_engineering import advanced_feature_engineering
from src.preprocessing.professional_motivation_features import (
    enhance_professional_features,
    analyze_aspiration_sentiment,
    detect_commitment_expressions,
    create_career_term_detector,
    create_professional_motivation_score
)

# Garantir que recursos NLTK necessários estejam disponíveis
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Baixando recursos NLTK necessários...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

class TextFeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador que aplica todas as etapas de processamento de texto
    implementadas nos scripts de treinamento.
    """
    
    def __init__(self, params_path=None, script03_params_path=None):
        """
        Inicializa o transformador.
        
        Args:
            params_path: Caminho para o arquivo joblib com parâmetros gerais salvos
            script03_params_path: Caminho para o arquivo joblib com parâmetros específicos do script 3
        """
        self.params_path = params_path
        self.script03_params_path = script03_params_path
        self.params = None
        self.script03_params = None
        self.feature_names = None
        self.tfidf_vectorizers = {}
        self.lda_models = {}
        
        # Colunas de texto relevantes
        self.text_cols = [
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
            '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?', 
            'Déjame un mensaje',
            '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
        ]
        
        # Mapeamento de colunas originais para os nomes usados nos vetorizadores
        self.column_mapping = {
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?': 'cuando_hables_inglés_con_fluid',
            '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?': 'qué_esperas_aprender_en_la',
            'Déjame un mensaje': 'déjame_un_mensaje',
            '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?': 'qué_esperas_aprender_en_la'  # Usando o mesmo vetorizador para coluna similar
        }
        
    def fit(self, X, y=None):
        """
        Carrega os parâmetros salvos durante o treino.
        
        Args:
            X: DataFrame de entrada (não utilizado)
            y: Target (não utilizado)
            
        Returns:
            self
        """
        # Carregar parâmetros principais
        if self.params_path and os.path.exists(self.params_path):
            self.params = joblib.load(self.params_path)
            print(f"Parâmetros gerais carregados de {self.params_path}")
        else:
            raise ValueError(f"Arquivo de parâmetros gerais não encontrado em {self.params_path}")
        
        # Carregar parâmetros do script 3 se especificados
        if self.script03_params_path and os.path.exists(self.script03_params_path):
            self.script03_params = joblib.load(self.script03_params_path)
            print(f"Parâmetros do script 3 carregados de {self.script03_params_path}")
            
            # Adicionar parâmetros do script 3 ao dicionário de parâmetros principal
            if self.script03_params:
                # Verificar se já existe uma chave para os parâmetros do script 3
                if 'script03_features' not in self.params:
                    self.params['script03_features'] = self.script03_params
        
        # Verificar e carregar vetorizadores TF-IDF e modelos LDA (script 03)
        # Usar caminho absoluto para garantir que encontrará os arquivos
        models_dir = "/Users/ramonmoreira/desktop/smart_ads/data/fixed_models"
        if os.path.exists(models_dir):
            tfidf_path = os.path.join(models_dir, 'tfidf_vectorizers.joblib')
            lda_path = os.path.join(models_dir, 'lda_models.joblib')
            
            if os.path.exists(tfidf_path):
                self.tfidf_vectorizers = joblib.load(tfidf_path)
                print(f"Vetorizadores TF-IDF do script 03 carregados de {tfidf_path}")
            else:
                print(f"Arquivo de vetorizadores TF-IDF não encontrado: {tfidf_path}")
            
            if os.path.exists(lda_path):
                self.lda_models = joblib.load(lda_path)
                print(f"Modelos LDA do script 03 carregados de {lda_path}")
            else:
                print(f"Arquivo de modelos LDA não encontrado: {lda_path}")
        else:
            print(f"Diretório de modelos do script 03 não encontrado: {models_dir}")
            print("Apenas o processamento do script 04 será aplicado.")
        
        return self
    
    def _preprocess_text(self, text):
        """
        Realiza pré-processamento de texto conforme o script 03.
        
        Args:
            text: Texto para pré-processar
            
        Returns:
            Texto pré-processado
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Converter para minúsculas
        text = text.lower()

        # Remover URLs e emails
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)

        # Remover caracteres especiais e números (preservando letras acentuadas)
        text = re.sub(r'[^\w\s\áéíóúñü]', ' ', text)
        text = re.sub(r'\d+', ' ', text)

        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenização simplificada
        tokens = text.split()

        # Remover stopwords (espanhol)
        stop_words = set(stopwords.words('spanish'))
        tokens = [token for token in tokens if token not in stop_words]

        # Lematização 
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Remover tokens muito curtos
        tokens = [token for token in tokens if len(token) > 2]

        # Juntar tokens novamente
        processed_text = ' '.join(tokens)

        return processed_text
    
    def _extract_basic_features(self, text):
        """
        Extrai features básicas de texto conforme o script 03.
        
        Args:
            text: Texto para extrair features
            
        Returns:
            Dicionário com features básicas
        """
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'word_count': 0,
                'char_count': 0,
                'avg_word_length': 0,
                'has_question': 0,
                'has_exclamation': 0
            }

        # Limpar e tokenizar o texto
        text = text.strip()
        words = text.split()

        # Extrair features
        word_count = len(words)
        char_count = len(text)
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        has_question = 1 if '?' in text else 0
        has_exclamation = 1 if '!' in text else 0

        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'has_question': has_question,
            'has_exclamation': has_exclamation
        }
    
    def _apply_tfidf_vectorization(self, df, col, processed_texts):
        """
        Aplica o vetorizador TF-IDF para a coluna especificada.
        
        Args:
            df: DataFrame a ser processado
            col: Nome da coluna de texto
            processed_texts: Série com textos pré-processados
            
        Returns:
            DataFrame com features TF-IDF adicionadas
        """
        col_key = self._get_column_key(col)
        
        # Verificar se temos vetorizador para esta coluna
        if col_key in self.tfidf_vectorizers:
            vectorizer = self.tfidf_vectorizers[col_key]
            
            try:
                # Transformar textos
                tfidf_matrix = vectorizer.transform(processed_texts.fillna(''))
                feature_names = vectorizer.get_feature_names_out()
                
                # Converter para DataFrame
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(), 
                    index=df.index,
                    columns=[f'{col_key}_tfidf_{term}' for term in feature_names]
                )
                
                # Aplicar boost para termos importantes do trabalho
                important_terms = ['trabajo', 'empleo', 'profesional', 'oportunidades', 'laboral',
                                  'carrera', 'mejor trabajo', 'oportunidades laborales', 'profesión']
                
                for term in important_terms:
                    boost_cols = [c for c in tfidf_df.columns if term in c]
                    for boost_col in boost_cols:
                        tfidf_df[boost_col] = tfidf_df[boost_col] * 1.5
                
                return tfidf_df
                
            except Exception as e:
                print(f"Erro ao aplicar TF-IDF para {col_key}: {e}")
                return pd.DataFrame(index=df.index)
        else:
            print(f"Vetorizador não encontrado para {col_key}, pulando...")
            return pd.DataFrame(index=df.index)
    
    def _apply_lda(self, df, col, processed_texts):
        """
        Aplica o modelo LDA para a coluna especificada.
        
        Args:
            df: DataFrame a ser processado
            col: Nome da coluna de texto
            processed_texts: Série com textos pré-processados
            
        Returns:
            DataFrame com features de tópicos adicionadas
        """
        col_key = self._get_column_key(col)
        
        # Verificar se temos modelo LDA para esta coluna
        if col_key in self.lda_models:
            lda_model_info = self.lda_models[col_key]
            
            try:
                # Extrair componentes
                lda_model = lda_model_info['lda_model']
                vectorizer = lda_model_info['vectorizer']
                
                # Transformar textos
                dtm = vectorizer.transform(processed_texts.fillna(''))
                topic_distributions = lda_model.transform(dtm)
                
                # Converter para DataFrame
                n_topics = topic_distributions.shape[1]
                topic_df = pd.DataFrame(
                    topic_distributions,
                    index=df.index,
                    columns=[f'{col_key}_topic_{i+1}' for i in range(n_topics)]
                )
                
                # Adicionar tópico dominante
                topic_df[f'{col_key}_dominant_topic'] = topic_distributions.argmax(axis=1) + 1
                
                return topic_df
                
            except Exception as e:
                print(f"Erro ao aplicar LDA para {col_key}: {e}")
                return pd.DataFrame(index=df.index)
        else:
            print(f"Modelo LDA não encontrado para {col_key}, pulando...")
            return pd.DataFrame(index=df.index)
    
    def _get_column_key(self, col):
        """
        Gera uma chave limpa para a coluna de texto.
        
        Args:
            col: Nome da coluna de texto
            
        Returns:
            Chave limpa para a coluna
        """
        # Primeiro tenta usar o mapeamento definido
        if col in self.column_mapping:
            return self.column_mapping[col]
        
        # Caso contrário, gera um nome limpo
        col_key = col.replace(' ', '_').replace('?', '').replace('¿', '')
        col_key = re.sub(r'[^\w]', '', col_key)[:30].lower()
        return col_key
        
    def _apply_script03_transformations(self, df):
        """
        Aplica as transformações do script 03_feature_engineering_1.py
        
        Args:
            df: DataFrame de entrada
            
        Returns:
            DataFrame com transformações do script 03 aplicadas
        """
        print("Aplicando transformações do script 03 (NLP básico e LDA)...")
        
        # Resultado final combinará todos os DataFrames de features
        df_result = df.copy()
        
        # Filtrar colunas de texto existentes
        text_cols = [col for col in self.text_cols if col in df.columns]
        
        if not text_cols:
            print("Nenhuma coluna de texto encontrada para processamento script 03")
            return df_result
        
        # Para cada coluna de texto
        for col in text_cols:
            col_key = self._get_column_key(col)
            print(f"Processando coluna: {col}")
            
            # 1. Pré-processar texto
            processed_texts = df[col].apply(self._preprocess_text)
            
            # 2. Extrair features básicas
            print(f"  Extraindo features básicas para {col_key}...")
            basic_features = processed_texts.apply(self._extract_basic_features)
            
            # Converter dicionário para DataFrame
            for feature in ['word_count', 'char_count', 'avg_word_length', 'has_question', 'has_exclamation']:
                df_result[f'{col_key}_{feature}'] = basic_features.apply(lambda x: x.get(feature, 0))
            
            # 3. Aplicar TF-IDF
            print(f"  Aplicando TF-IDF para {col_key}...")
            tfidf_df = self._apply_tfidf_vectorization(df, col, processed_texts)
            
            # Adicionar features TF-IDF ao DataFrame final
            for tfidf_col in tfidf_df.columns:
                df_result[tfidf_col] = tfidf_df[tfidf_col]
            
            # 4. Aplicar LDA
            print(f"  Aplicando LDA para {col_key}...")
            topic_df = self._apply_lda(df, col, processed_texts)
            
            # Adicionar features de tópicos ao DataFrame final
            for topic_col in topic_df.columns:
                df_result[topic_col] = topic_df[topic_col]
            
            # 5. Guardar texto processado para uso posterior
            df_result[f'{col}_clean'] = processed_texts
        
        # Contar número de features adicionadas
        num_added_features = df_result.shape[1] - df.shape[1]
        print(f"Script 03: Adicionadas {num_added_features} novas features")
        
        return df_result
    
    def transform(self, X):
        """
        Aplica processamento de texto aos dados.
        
        Args:
            X: DataFrame de entrada
            
        Returns:
            DataFrame com features de texto adicionadas
        """
        if self.params is None:
            raise ValueError("O transformador precisa ser ajustado com fit() antes de usar transform()")
        
        print(f"Aplicando processamento de texto ao DataFrame: {X.shape}")
        df_result = X.copy()
        
        # 1. Aplicar transformações do script 03_feature_engineering_1.py
        df_result = self._apply_script03_transformations(df_result)
        
        # 2. Features textuais básicas (script 02 e script 04)
        print("2. Processando features textuais básicas...")
        text_params = self.params.get('text_processing', {})
        df_result, _ = text_feature_engineering(df_result, fit=False, params=text_params)
        
        # 3. Features avançadas
        print("3. Aplicando feature engineering avançada para texto...")
        advanced_params = self.params.get('advanced_features', {})
        df_result, _ = advanced_feature_engineering(df_result, fit=False, params=advanced_params)
        
        # 4. Features de motivação profissional - usar parâmetros do script 3 se disponíveis
        print("4. Criando features de motivação profissional...")
        
        # Primeiro verifica se temos parâmetros específicos do script 3
        if self.script03_params:
            # Verifica cada conjunto de parâmetros no script03_params
            text_cols = [col for col in self.text_cols if col in df_result.columns]
            
            if text_cols:
                # Usar os parâmetros do script 3 diretamente
                print("   Usando parâmetros específicos do script 3 para processamento de texto...")
                
                # Aplicar TF-IDF vetorizadores diretamente
                if 'vectorizers' in self.script03_params:
                    vectorizers = self.script03_params.get('vectorizers', {})
                    
                    # Processar cada coluna de texto com seu vetorizador correspondente
                    for col in text_cols:
                        # Usar o mapeamento para obter o nome correto da coluna para o vetorizador
                        mapped_col = self.column_mapping.get(col)
                        
                        if mapped_col and mapped_col in vectorizers:
                            print(f"   Aplicando vetorizador para '{mapped_col}'")
                            vectorizer = vectorizers[mapped_col]
                            
                            # Processar e aplicar o vetorizador
                            texts = df_result[col].fillna('')
                            
                            try:
                                # Transformar textos usando o vetorizador
                                tfidf_matrix = vectorizer.transform(texts)
                                feature_names = vectorizer.get_feature_names_out()
                                
                                # Converter para array para manipulação
                                tfidf_array = tfidf_matrix.toarray()
                                
                                # Adicionar colunas para cada feature do vetorizador
                                for i, term in enumerate(feature_names):
                                    feature_name = f"{mapped_col}_tfidf_{term}"
                                    df_result[feature_name] = tfidf_array[:, i]
                                
                                print(f"      Adicionadas {len(feature_names)} features TF-IDF para '{mapped_col}'")
                            except Exception as e:
                                print(f"      ERRO ao aplicar vetorizador para '{mapped_col}': {e}")
                        else:
                            print(f"   Vetorizador não encontrado para '{mapped_col if mapped_col else col}', pulando...")
                
                # Aplicar o score de motivação profissional diretamente
                if 'professional_motivation' in self.script03_params:
                    try:
                        prof_mot_params = self.script03_params.get('professional_motivation', {})
                        print("   Aplicando score de motivação profissional diretamente...")
                        df_result_prof, _ = create_professional_motivation_score(
                            df_result, text_cols, fit=False, params=prof_mot_params
                        )
                        # Adicionar colunas sem duplicação
                        for col in df_result_prof.columns:
                            if col not in df_result.columns:
                                df_result[col] = df_result_prof[col]
                        print(f"      Adicionadas {len(df_result_prof.columns)} features de motivação profissional")
                    except Exception as e:
                        print(f"      ERRO ao aplicar score de motivação profissional: {e}")
                
                # Aplicar aspiration_sentiment
                if 'aspiration_sentiment' in self.script03_params:
                    try:
                        asp_sent_params = self.script03_params.get('aspiration_sentiment', {})
                        print("   Aplicando análise de sentimento de aspiração diretamente...")
                        df_result_asp, _ = analyze_aspiration_sentiment(
                            df_result, text_cols, fit=False, params=asp_sent_params
                        )
                        # Adicionar colunas sem duplicação
                        for col in df_result_asp.columns:
                            if col not in df_result.columns:
                                df_result[col] = df_result_asp[col]
                        print(f"      Adicionadas {len(df_result_asp.columns)} features de sentimento de aspiração")
                    except Exception as e:
                        print(f"      ERRO ao aplicar análise de sentimento de aspiração: {e}")
                
                # Aplicar commitment
                if 'commitment' in self.script03_params:
                    try:
                        commitment_params = self.script03_params.get('commitment', {})
                        print("   Aplicando detecção de expressões de compromisso diretamente...")
                        df_result_comm, _ = detect_commitment_expressions(
                            df_result, text_cols, fit=False, params=commitment_params
                        )
                        # Adicionar colunas sem duplicação
                        for col in df_result_comm.columns:
                            if col not in df_result.columns:
                                df_result[col] = df_result_comm[col]
                        print(f"      Adicionadas {len(df_result_comm.columns)} features de expressões de compromisso")
                    except Exception as e:
                        print(f"      ERRO ao aplicar detecção de expressões de compromisso: {e}")
                
                # Aplicar career
                if 'career' in self.script03_params:
                    try:
                        career_params = self.script03_params.get('career', {})
                        print("   Aplicando detector de termos de carreira diretamente...")
                        df_result_career, _ = create_career_term_detector(
                            df_result, text_cols, fit=False, params=career_params
                        )
                        # Adicionar colunas sem duplicação
                        for col in df_result_career.columns:
                            if col not in df_result.columns:
                                df_result[col] = df_result_career[col]
                        print(f"      Adicionadas {len(df_result_career.columns)} features de termos de carreira")
                    except Exception as e:
                        print(f"      ERRO ao aplicar detector de termos de carreira: {e}")
            
            # IMPORTANTE: Não use enhance_professional_features aqui, pois já aplicamos
            # todas as funções diretamente e evitamos a duplicação
        
        # Use o enhance_professional_features como fallback apenas se não tivermos os parâmetros do script 3
        elif 'professional_features' in self.params:
            print("   Usando parâmetros gerais de professional_features como fallback...")
            prof_params = self.params.get('professional_features', {})
            # Filtrar colunas de texto existentes
            text_cols = [col for col in self.text_cols if col in df_result.columns]
            if text_cols:
                df_result, _ = enhance_professional_features(df_result, text_cols, fit=False, params=prof_params)
        
        # Verificar o número total de features geradas por coluna de texto
        if self.script03_params:
            for col in self.text_cols:
                if col in df_result.columns:
                    mapped_col = self.column_mapping.get(col, "unknown")
                    feature_count = sum(1 for f in df_result.columns if mapped_col in f)
                    print(f"   Total de features para '{mapped_col}': {feature_count}")
        
        # Guardar nomes das features para referência
        self.feature_names = df_result.columns.tolist()
        
        print(f"Processamento de texto concluído! Dimensões finais: {df_result.shape}")
        return df_result
    
    def get_feature_names_out(self, input_features=None):
        """
        Retorna os nomes das features após a transformação.
        
        Args:
            input_features: Lista das features de entrada (não utilizado)
            
        Returns:
            Lista com nomes das features de saída
        """
        if self.feature_names is None:
            raise ValueError("O método transform deve ser chamado antes de get_feature_names_out")
        return self.feature_names