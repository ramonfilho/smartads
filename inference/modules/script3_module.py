"""
Módulo para aplicar transformações equivalentes ao script 03_feature_engineering_1.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Biblioteca transformers não encontrada. Funcionalidades de embedding podem não estar disponíveis.")

# Adicionar o diretório raiz do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# Suprimir avisos
warnings.filterwarnings('ignore')

# Baixar recursos NLTK essenciais
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Baixando recursos NLTK necessários...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Verificar se os recursos foram baixados corretamente
try:
    from nltk.tokenize import word_tokenize
    word_tokenize("Teste de tokenização")
    print("✓ Tokenizer NLTK configurado com sucesso")
except LookupError:
    print("Erro ao carregar tokenizer. Baixando recursos adicionais...")
    nltk.download('punkt', quiet=False) # Download explícito com feedback

def preprocess_text(text, language='spanish'):
    """
    Realiza pré-processamento básico de texto:
    - Converte para minúsculas
    - Remove caracteres especiais e números
    - Remove stopwords
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

    # Usar divisão simples em vez de word_tokenize para evitar problemas com NLTK
    try:
        # Remover stopwords (espanhol)
        stop_words = set(stopwords.words(language))
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]

        # Remover tokens muito curtos
        tokens = [token for token in tokens if len(token) > 2]

        # Juntar tokens novamente
        processed_text = ' '.join(tokens)
    except:
        # Fallback simples se word_tokenize falhar
        processed_text = text
        
    return processed_text

# NOVA FUNÇÃO: Extração de features básicas de texto
def extract_basic_features(text):
    """
    Extrai features básicas de uma string de texto.
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

def extract_tfidf_features(texts, max_features=50, fit=True, vectorizer=None):
    """
    Extrai features TF-IDF preservando a indexação original.
    """
    # Criar array para rastrear índices de documentos não vazios
    valid_indices = []
    filtered_texts = []

    # Filtrar textos vazios mantendo controle dos índices
    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text.strip()) > 0:
            filtered_texts.append(text)
            valid_indices.append(i)

    # Inicializar DataFrame de resultado para todos os documentos
    total_docs = len(texts)
    tfidf_df = pd.DataFrame(index=range(total_docs))

    if fit:
        # Lista de stop words em espanhol
        try:
            spanish_stopwords = stopwords.words('spanish')
        except:
            spanish_stopwords = None

        # Configurar TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=5,
            stop_words=spanish_stopwords,
            ngram_range=(1, 2)
        )

        # Ajustar e transformar textos filtrados
        if len(filtered_texts) > 0:
            tfidf_matrix = vectorizer.fit_transform(filtered_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Refinar pesos para termos importantes
            filtered_matrix = tfidf_matrix.toarray()
            mean_tfidf = np.array(filtered_matrix.mean(axis=0)).flatten()
            top_indices = np.argsort(mean_tfidf)[-int(len(mean_tfidf)*0.2):]
            important_terms = [feature_names[i] for i in top_indices]

            # Aumentar pesos de termos importantes (refinamento)
            for term in important_terms:
                term_idx = np.where(feature_names == term)[0][0]
                filtered_matrix[:, term_idx] *= 1.5

            # Normalizar evitando divisão por zero
            row_sums = filtered_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Evitar divisão por zero
            filtered_matrix = filtered_matrix / row_sums[:, np.newaxis]

            # Preencher DataFrame para todos os documentos com zeros
            for term_idx, term in enumerate(feature_names):
                tfidf_df[f'tfidf_{term}'] = 0.0

            # Preencher apenas para documentos válidos
            for i, orig_idx in enumerate(valid_indices):
                for j, term in enumerate(feature_names):
                    tfidf_df.iloc[orig_idx, j] = filtered_matrix[i, j]

    else:
        # Transformar usando o vetorizador existente
        if vectorizer is not None and len(filtered_texts) > 0:
            tfidf_matrix = vectorizer.transform(filtered_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Inicializar colunas no DataFrame
            for term in feature_names:
                tfidf_df[f'tfidf_{term}'] = 0.0

            # Preencher apenas para documentos válidos
            matrix_array = tfidf_matrix.toarray()
            for i, orig_idx in enumerate(valid_indices):
                for j, term in enumerate(feature_names):
                    tfidf_df.iloc[orig_idx, j] = matrix_array[i, j]

    return tfidf_df, vectorizer

def extract_topics_lda(texts, n_topics=5, max_features=1000, vectorizer=None, lda_model=None):
    """
    Extrai tópicos latentes usando LDA e preserva a indexação original.
    """
    print(f"Extraindo {n_topics} tópicos latentes via LDA...")

    # Criar array para rastrear índices de documentos não vazios
    valid_indices = []
    filtered_texts = []

    # Filtrar textos vazios mantendo controle dos índices
    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text.strip()) > 0:
            filtered_texts.append(text)
            valid_indices.append(i)

    if len(filtered_texts) < 10:
        print("Poucos textos válidos para LDA. Pulando.")
        return None, None, None

    # Inicializar DataFrame para resultados
    total_docs = len(texts)
    topic_df = pd.DataFrame(index=range(total_docs))
    
    # Modo fit - treinamento do modelo
    if vectorizer is None or lda_model is None:
        # Preparar vetorizador
        vectorizer = TfidfVectorizer(max_features=max_features)
        dtm = vectorizer.fit_transform(filtered_texts)

        # Treinar modelo LDA
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=42,
            batch_size=128,
            n_jobs=-1
        )

        # Obter distribuição de tópicos
        doc_topic_dist = lda_model.fit_transform(dtm)

        # Inicializar colunas para todos os documentos
        for topic in range(n_topics):
            topic_df[f'topic_{topic}_prob'] = 0.0
        topic_df['dominant_topic'] = 0

        # Preencher apenas para documentos válidos
        for i, orig_idx in enumerate(valid_indices):
            for topic in range(n_topics):
                topic_df.iloc[orig_idx, topic] = doc_topic_dist[i, topic]
            topic_df.iloc[orig_idx, n_topics] = doc_topic_dist[i].argmax()

    # Modo transform - aplicação do modelo existente
    else:
        try:
            # Transformar usando o vetorizador existente
            dtm = vectorizer.transform(filtered_texts)
            
            # Aplicar modelo LDA existente
            doc_topic_dist = lda_model.transform(dtm)
            
            # Inicializar colunas para todos os documentos
            for topic in range(n_topics):
                topic_df[f'topic_{topic}_prob'] = 0.0
            topic_df['dominant_topic'] = 0
            
            # Preencher apenas para documentos válidos
            for i, orig_idx in enumerate(valid_indices):
                for topic in range(n_topics):
                    topic_df.iloc[orig_idx, topic] = doc_topic_dist[i, topic]
                topic_df.iloc[orig_idx, n_topics] = doc_topic_dist[i].argmax()
                
        except Exception as e:
            print(f"Erro ao aplicar LDA: {str(e)}")
            return None, None, None

    return topic_df, vectorizer, lda_model

def process_column(col, texts, params, col_key):
    """
    Processa uma coluna de texto aplicando TF-IDF, LDA e features básicas.
    
    Args:
        col: Nome da coluna
        texts: Textos a serem processados
        params: Parâmetros carregados
        col_key: Chave limpa para a coluna
        
    Returns:
        DataFrame com features extraídas
    """
    results_df = pd.DataFrame(index=range(len(texts)))
    
    # Pré-processar textos
    processed_texts = [preprocess_text(text) for text in texts]
    
    # NOVA PARTE: Extrair features básicas de texto
    print(f"  Extraindo features básicas para {col_key}...")
    basic_features_list = [extract_basic_features(text) for text in texts]
    
    # Adicionar features básicas ao DataFrame
    for feature in ['word_count', 'char_count', 'avg_word_length', 'has_question', 'has_exclamation']:
        feature_values = [feat.get(feature, 0) for feat in basic_features_list]
        results_df[f"{col_key}_{feature}"] = feature_values
    
    # Verificar se temos parâmetros para esta coluna
    tfidf_vectorizer = None
    lda_vectorizer = None
    lda_model = None
    
    if 'tfidf_vectorizers' in params and col_key in params['tfidf_vectorizers']:
        tfidf_vectorizer = params['tfidf_vectorizers'][col_key]
        print(f"  Vetorizador TF-IDF carregado para {col_key}")
    
    if 'lda_models' in params and col_key in params['lda_models']:
        lda_vectorizer = params['lda_models'][col_key]['vectorizer']
        lda_model = params['lda_models'][col_key]['lda_model']
        print(f"  Modelo LDA carregado para {col_key}")
    
    # Extrair features TF-IDF com refinamento de pesos
    print(f"  Extraindo features TF-IDF para {col_key}...")
    tfidf_df, _ = extract_tfidf_features(
        processed_texts, 
        max_features=50, 
        fit=False,
        vectorizer=tfidf_vectorizer
    )
    
    if not tfidf_df.empty:
        # Renomear colunas para evitar colisões
        tfidf_df = tfidf_df.add_prefix(f"{col_key}_")
        
        # Mesclar com resultados
        for col_name in tfidf_df.columns:
            results_df[col_name] = tfidf_df[col_name].values
        print(f"  Adicionadas {tfidf_df.shape[1]} features TF-IDF")
    
    # Extrair tópicos com LDA se temos modelos
    if lda_vectorizer is not None and lda_model is not None:
        print(f"  Extraindo tópicos LDA para {col_key}...")
        topic_df, _, _ = extract_topics_lda(
            processed_texts,
            vectorizer=lda_vectorizer,
            lda_model=lda_model
        )
        
        if topic_df is not None:
            # Renomear colunas para evitar colisões
            topic_df = topic_df.add_prefix(f"{col_key}_")
            
            # Mesclar com resultados
            for col_name in topic_df.columns:
                results_df[col_name] = topic_df[col_name].values
            print(f"  Adicionadas {topic_df.shape[1]} features de tópicos")
    
    return results_df

def process_text_features(df, tfidf_vectorizers, lda_models):
    """
    Processa features de texto usando TF-IDF e modelos LDA
    """
    # Identificar colunas de texto existentes
    text_columns = [
        col for col in df.columns 
        if '_original' in col and any(term in col for term in [
            'mensaje', 'inglés', 'vida', 'oportunidades', 'esperas', 'aprender', 
            'Semana', 'Inmersión', 'Déjame', 'fluidez'
        ])
    ]
    
    if not text_columns:
        # Tentar encontrar colunas de texto originais
        text_columns = [
            col for col in df.columns 
            if df[col].dtype == 'object' and any(term in col for term in [
                'mensaje', 'inglés', 'vida', 'oportunidades', 'esperas', 'aprender', 
                'Semana', 'Inmersión', 'Déjame', 'fluidez'
            ])
        ]
    
    print(f"Encontradas {len(text_columns)} colunas de texto para processamento")
    
    if not text_columns:
        print("AVISO: Nenhuma coluna de texto encontrada. Pulando processamento de texto.")
        return df
    
    # DataFrame para armazenar todas as features extraídas
    text_features_df = pd.DataFrame(index=df.index)
    
    # Processar cada coluna de texto
    for col in text_columns:
        print(f"\nProcessando coluna: {col}")
        
        # Criar chave limpa para nomes de colunas
        col_key = col.replace(' ', '_').replace('?', '').replace('¿', '').replace('_original', '')
        col_key = re.sub(r'[^\w]', '', col_key)[:30]
        
        # Processar coluna e obter features
        params = {'tfidf_vectorizers': tfidf_vectorizers, 'lda_models': lda_models}
        column_features = process_column(col, df[col].values, params, col_key)
        
        # Combinar com outras features
        if not column_features.empty:
            # Remover colunas vazias
            column_features = column_features.loc[:, column_features.columns[column_features.notna().any()]]
            
            # Adicionar ao DataFrame de features de texto
            for feat_col in column_features.columns:
                text_features_df[feat_col] = column_features[feat_col].values
    
    # Combinar o DataFrame original com as features de texto
    result_df = pd.concat([df, text_features_df], axis=1)
    
    # Remover colunas duplicadas se houver
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    
    print(f"\nProcessamento de texto avançado concluído.")
    print(f"Dimensões iniciais: {df.shape} → Dimensões finais: {result_df.shape}")
    print(f"Adicionadas {result_df.shape[1] - df.shape[1]} novas features de texto")
    
    return result_df

def apply_script3_transformations(df, params_path):
    """
    Função principal para aplicar transformações do script 3.
    
    Args:
        df: DataFrame de entrada (output do script 2)
        params_path: Caminho base para os parâmetros (vamos ignorar e usar os arquivos específicos)
        
    Returns:
        DataFrame processado
    """
    print(f"\n=== Aplicando transformações do script 3 (processamento avançado de texto) ===")
    
    # Definir os caminhos corretos para os arquivos de modelos
    params_dir = os.path.dirname(params_path)
    tfidf_path = os.path.join(params_dir, "03_tfidf_vectorizers.joblib")
    lda_path = os.path.join(params_dir, "03_lda_models.joblib")
    
    # Carregar modelos TF-IDF
    print(f"Carregando vetorizadores TF-IDF de: {tfidf_path}")
    try:
        tfidf_vectorizers = joblib.load(tfidf_path)
        print(f"Carregados {len(tfidf_vectorizers)} vetorizadores TF-IDF")
    except Exception as e:
        print(f"AVISO: Não foi possível carregar vetorizadores TF-IDF: {e}")
        tfidf_vectorizers = {}
    
    # Carregar modelos LDA
    print(f"Carregando modelos LDA de: {lda_path}")
    try:
        lda_models = joblib.load(lda_path)
        print(f"Carregados {len(lda_models)} modelos LDA")
    except Exception as e:
        print(f"AVISO: Não foi possível carregar modelos LDA: {e}")
        lda_models = {}
    
    # Aplicar processamento de texto
    result_df = process_text_features(df, tfidf_vectorizers, lda_models)
    
    print(f"Transformações do script 3 concluídas. Dimensões: {result_df.shape}")
    
    return result_df