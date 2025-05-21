#!/usr/bin/env python
"""
Script adaptado para inferência, baseado no 03_feature_engineering_1.py original.
Função principal apply() recebe um DataFrame e retorna o DataFrame processado.
"""

# Importar e baixar recursos NLTK necessários
import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings

warnings.filterwarnings('ignore')

# Verificar se os recursos NLTK necessários estão disponíveis
def ensure_nltk_resources():
    """Garante que os recursos NLTK necessários estão disponíveis."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            print(f"Baixando recurso NLTK: {resource}")
            nltk.download(resource, quiet=True)

# 1. Função para pré-processamento de texto
def preprocess_text(text, language='spanish'):
    """
    Realiza pré-processamento básico de texto:
    - Converte para minúsculas
    - Remove caracteres especiais e números
    - Remove stopwords
    - Lematização básica
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

    # Tokenização simplificada (sem dependência de parâmetro de idioma)
    tokens = text.split()

    # Remover stopwords (espanhol)
    stop_words = set(stopwords.words(language))
    tokens = [token for token in tokens if token not in stop_words]

    # Lematização
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remover tokens muito curtos
    tokens = [token for token in tokens if len(token) > 2]

    # Juntar tokens novamente
    processed_text = ' '.join(tokens)

    return processed_text

# 2. Função para extrair features básicas de texto
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

# 3. Função para extrair features TF-IDF com refinamento de pesos
def extract_tfidf_features(texts, max_features=50, fit=False, vectorizer=None):
    """
    Extrai features TF-IDF preservando a indexação original.
    Para mode de inferência, fit=False e vectorizer deve ser fornecido.
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

    # No modo de inferência, utilizamos o vectorizer já treinado
    if not fit and vectorizer:
        # Transformar usando o vetorizador existente
        if len(filtered_texts) > 0:
            try:
                tfidf_matrix = vectorizer.transform(filtered_texts)
                feature_names = vectorizer.get_feature_names_out()

                # Inicializar DataFrame para todos os documentos com zeros
                tfidf_df = pd.DataFrame(
                    np.zeros((total_docs, len(feature_names))),
                    columns=[f'tfidf_{term}' for term in feature_names]
                )

                # Preencher apenas para documentos válidos
                matrix_array = tfidf_matrix.toarray()
                for i, orig_idx in enumerate(valid_indices):
                    for j, term in enumerate(feature_names):
                        tfidf_df.iloc[orig_idx, j] = matrix_array[i, j]

                return tfidf_df, vectorizer
            except Exception as e:
                print(f"Erro ao aplicar TF-IDF: {e}")
                # Retornar DataFrame vazio em caso de erro
                empty_df = pd.DataFrame(index=range(total_docs))
                return empty_df, vectorizer
        else:
            # Retornar DataFrame vazio se não houver textos válidos
            empty_df = pd.DataFrame(index=range(total_docs))
            return empty_df, vectorizer
    # Este código do modo "fit" é mantido, mas não será usado na inferência
    else:
        # Lista de stop words em espanhol
        spanish_stopwords = stopwords.words('spanish')

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

            # Inicializar DataFrame para todos os documentos com zeros
            tfidf_df = pd.DataFrame(
                np.zeros((total_docs, len(feature_names))),
                columns=[f'tfidf_{term}' for term in feature_names]
            )

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

            # Preencher apenas para documentos válidos
            for i, orig_idx in enumerate(valid_indices):
                for j, term in enumerate(feature_names):
                    tfidf_df.iloc[orig_idx, j] = filtered_matrix[i, j]

            return tfidf_df, vectorizer
        else:
            # Retornar DataFrame vazio se não houver textos válidos
            empty_df = pd.DataFrame(index=range(total_docs))
            return empty_df, vectorizer

# 4. Função para extrair tópicos usando LDA (usando modelos pré-treinados em inferência)
def apply_lda_transform(texts, lda_model, vectorizer):
    """
    Aplica um modelo LDA pré-treinado em um conjunto de textos.
    """
    # Criar array para rastrear índices de documentos não vazios
    valid_indices = []
    filtered_texts = []

    # Filtrar textos vazios mantendo controle dos índices
    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text.strip()) > 0:
            filtered_texts.append(text)
            valid_indices.append(i)

    if len(filtered_texts) < 5:
        print("Poucos textos válidos para LDA. Pulando.")
        return None
        
    try:
        # Transformar textos usando o vetorizador do LDA
        dtm = vectorizer.transform(filtered_texts)
        
        # Obter distribuição de tópicos por documento
        doc_topic_dist = lda_model.transform(dtm)
        n_topics = doc_topic_dist.shape[1]
        dominant_topics = doc_topic_dist.argmax(axis=1)
        
        # Criar DataFrame para armazenar resultados de tópicos
        # Inicializar com zeros para todos os documentos originais
        total_docs = len(texts)
        topic_features = pd.DataFrame(
            np.zeros((total_docs, n_topics + 1)),  # +1 para o tópico dominante
            columns=[f'topic_{i}_prob' for i in range(n_topics)] + ['dominant_topic']
        )
        
        # Preencher apenas para documentos válidos
        for i, orig_idx in enumerate(valid_indices):
            for topic in range(n_topics):
                topic_features.iloc[orig_idx, topic] = doc_topic_dist[i, topic]
            topic_features.iloc[orig_idx, n_topics] = dominant_topics[i]
            
        return topic_features
    except Exception as e:
        print(f"Erro ao aplicar LDA: {e}")
        return None

# 6. Função para processar um conjunto de dados em modo de inferência
def process_dataset_inference(df, text_columns, tfidf_vectorizers=None, lda_models=None):
    """
    Processa o conjunto de dados para extrair features de NLP em modo de inferência.
    """
    print(f"\n=== Processando DataFrame para inferência ===")
    print(f"Dimensões iniciais: {df.shape}")

    # Criar um DF para armazenar features de texto
    text_features_df = pd.DataFrame(index=df.index)

    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df.columns:
            continue

        print(f"Processando coluna: {col}")

        # Criar chave limpa para nomes de colunas
        col_key = col.replace(' ', '_').replace('?', '')
        col_key = re.sub(r'[^\w]', '', col_key)[:30]

        # Pré-processar textos
        processed_texts = df[col].apply(lambda x: preprocess_text(x))

        # 1. Extrair features básicas
        print("  Extraindo features básicas...")
        basic_features = processed_texts.apply(extract_basic_features)

        # Adicionar features básicas ao DataFrame
        for feature in ['word_count', 'char_count', 'avg_word_length', 'has_question', 'has_exclamation']:
            text_features_df[f"{col_key}_{feature}"] = basic_features.apply(lambda x: x.get(feature, 0))

        # Verificar se temos textos suficientes
        non_empty_texts = [text for text in processed_texts if isinstance(text, str) and len(text.strip()) > 0]

        if len(non_empty_texts) > 0:
            # 2. Aplicar TF-IDF usando vetorizador pré-treinado
            if tfidf_vectorizers and col_key in tfidf_vectorizers:
                print("  Aplicando features TF-IDF pré-treinadas...")
                tfidf_df, _ = extract_tfidf_features(
                    processed_texts, 
                    fit=False, 
                    vectorizer=tfidf_vectorizers[col_key]
                )
                
                # Adicionar features TF-IDF ao DataFrame
                if tfidf_df is not None and not tfidf_df.empty:
                    for tfidf_col in tfidf_df.columns:
                        text_features_df[f"{col_key}_{tfidf_col}"] = tfidf_df[tfidf_col].values

            # 3. Aplicar LDA usando modelo pré-treinado
            if lda_models and col_key in lda_models:
                print("  Aplicando LDA pré-treinado...")
                try:
                    lda_model = lda_models[col_key]['lda_model']
                    vectorizer = lda_models[col_key]['vectorizer']
                    
                    # Aplicar LDA
                    topic_features = apply_lda_transform(
                        processed_texts.tolist(), 
                        lda_model, 
                        vectorizer
                    )
                    
                    # Adicionar features de tópicos
                    if topic_features is not None:
                        for topic_col in topic_features.columns:
                            text_features_df[f"{col_key}_{topic_col}"] = topic_features[topic_col].values
                except Exception as e:
                    print(f"Erro ao aplicar LDA para {col_key}: {e}")

    # Concatenar features de texto com o DataFrame original
    final_df = pd.concat([df, text_features_df], axis=1)
    
    # Remover colunas duplicadas
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    
    print(f"Processamento concluído. Dimensões finais: {final_df.shape}")
    return final_df

def apply(df, params=None):
    """
    Função principal para aplicar feature engineering de texto em modo de inferência.
    
    Args:
        df: DataFrame com dados da etapa anterior
        params: Dicionário contendo vetorizadores e modelos pré-treinados
        
    Returns:
        DataFrame com features de texto adicionadas
    """
    # Garantir que os recursos NLTK estão disponíveis
    ensure_nltk_resources()
    
    # Carregar parâmetros se não fornecidos
    if params is None:
        print("Parâmetros não fornecidos. Carregando modelos do disco...")
        try:
            # Definir caminhos apenas quando necessário
            models_dir = "/Users/ramonmoreira/desktop/smart_ads/data/03_feature_engineering_1/models"
            tfidf_path = os.path.join(models_dir, "03_tfidf_vectorizers.joblib")
            lda_path = os.path.join(models_dir, "03_lda_models.joblib")
            
            print(f"Carregando vetorizadores TF-IDF de: {tfidf_path}")
            tfidf_vectorizers = joblib.load(tfidf_path)
            
            print(f"Carregando modelos LDA de: {lda_path}")
            lda_models = joblib.load(lda_path)
            
            params = {
                "tfidf_vectorizers": tfidf_vectorizers,
                "lda_models": lda_models
            }
        except Exception as e:
            print(f"ERRO ao carregar modelos: {e}")
            print("Prosseguindo sem modelos pré-treinados")
            params = {
                "tfidf_vectorizers": {},
                "lda_models": {}
            }
    
    # Definir colunas de texto para processamento (originais e com sufixo _original)
    text_columns = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
        'Déjame un mensaje',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
    ]
    
    # Adicionar versões originais se existirem
    original_cols = [col for col in df.columns if '_original' in col]
    text_columns.extend(original_cols)
    
    # Remover duplicatas e manter apenas colunas existentes
    text_columns = list(set(text_columns))
    text_columns = [col for col in text_columns if col in df.columns]
    
    print(f"Aplicando processamento de texto em {len(text_columns)} colunas")
    
    # Processar o DataFrame
    df_processed = process_dataset_inference(
        df,
        text_columns,
        tfidf_vectorizers=params.get("tfidf_vectorizers", {}),
        lda_models=params.get("lda_models", {})
    )
    
    return df_processed

# Manter a funcionalidade original para uso direto do script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aplicar feature engineering de texto em modo de inferência.")
    parser.add_argument("--input", type=str, required=True, 
                       help="Caminho para o arquivo CSV de entrada")
    parser.add_argument("--output", type=str, required=True,
                       help="Caminho para salvar o arquivo CSV processado")
    
    args = parser.parse_args()
    
    # Carregar dados
    print(f"Carregando dados de: {args.input}")
    input_data = pd.read_csv(args.input)
    
    # Processar
    processed_data = apply(input_data)
    
    # Salvar resultado
    print(f"Salvando resultado em: {args.output}")
    processed_data.to_csv(args.output, index=False)
    
    print("Processamento concluído!")