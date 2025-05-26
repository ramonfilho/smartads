#!/usr/bin/env python
"""
Script para processamento avançado de texto para o projeto Smart Ads.
Este script aplica processamento de NLP em colunas de texto para gerar features avançadas.
"""

# Importar e baixar recursos NLTK necessários
import pandas as pd
import numpy as np
import os
import re
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings('ignore')

# Configurar projeto - ajustado para seu ambiente local
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
INPUT_DIR = os.path.join(PROJECT_ROOT, "data/01_split")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/02_processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/03_feature_engineering_1")
PARAMS_DIR = os.path.join(PROJECT_ROOT, "src/preprocessing/03_params")  # Nova pasta para parâmetros

# Garantir que ambos os diretórios existam
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)  # Criar pasta de parâmetros

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    nltk.download('punkt', quiet=False)  # Download explícito com feedback

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
def extract_tfidf_features(texts, max_features=200, fit=True, vectorizer=None):
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

    if fit:
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
    else:
        # Transformar usando o vetorizador existente
        if vectorizer and len(filtered_texts) > 0:
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
        else:
            # Retornar DataFrame vazio se não houver textos válidos ou vetorizador
            empty_df = pd.DataFrame(index=range(total_docs))
            return empty_df, vectorizer

# 4. Função para extrair tópicos usando LDA
def extract_topics_lda(texts, n_topics=5, max_features=1000):
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
        return None

    # Preparar vetorizador
    vectorizer = CountVectorizer(max_features=max_features)
    dtm = vectorizer.fit_transform(filtered_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Treinar modelo LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        random_state=42,
        batch_size=128,
        n_jobs=-1
    )

    # Obter distribuição de tópicos por documento
    doc_topic_dist = lda.fit_transform(dtm)
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

    # Obter palavras por tópico
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Top 10 palavras
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append((topic_idx, top_words))

    return {
        'topics': topics,
        'doc_topic_dist': doc_topic_dist,
        'dominant_topics': dominant_topics,
        'lda_model': lda,
        'vectorizer': vectorizer,
        'topic_features': topic_features,
        'valid_indices': valid_indices
    }

# 5. Função para extrair embeddings (desativada por padrão para manter compatibilidade)
def extract_embeddings(texts, model_name='distilbert-base-multilingual-cased', max_length=128, batch_size=32):
    """
    Extrai embeddings de uma coleção de textos usando um modelo pré-treinado.
    """
    # Filtrar textos vazios
    filtered_texts = [text if isinstance(text, str) else "" for text in texts]
    filtered_texts = [text if text.strip() else "" for text in filtered_texts]

    # Ajustar tamanho do batch para textos longos
    if max(len(text.split()) for text in filtered_texts) > 100:
        batch_size = min(16, batch_size)

    print(f"Extraindo embeddings para {len(filtered_texts)} textos (batch_size={batch_size})...")

    # Carregar tokenizer e modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Mover modelo para GPU se disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Lista para armazenar embeddings
    all_embeddings = []

    # Processar em batches
    for i in range(0, len(filtered_texts), batch_size):
        batch_texts = filtered_texts[i:i+batch_size]

        print(f"Processando batch {i//batch_size + 1}/{(len(filtered_texts) + batch_size - 1)//batch_size}...", end="\r")

        # Tokenizar textos
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True,
                               max_length=max_length, return_tensors='pt')

        # Mover inputs para GPU
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        # Extrair embeddings (sem cálculo de gradientes para economizar memória)
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Usar a média dos embeddings da última camada
        embeddings = model_output.last_hidden_state.mean(dim=1)

        # Mover para CPU e converter para numpy
        embeddings = embeddings.cpu().numpy()
        all_embeddings.append(embeddings)

    print("\nProcessamento de embeddings concluído!")

    # Concatenar todos os batches
    all_embeddings = np.vstack(all_embeddings)

    return all_embeddings

# 6. Função para processar um conjunto de dados
def process_dataset(original_df, processed_df, text_columns, dataset_name,
                   tfidf_vectorizers=None, lda_models=None, save_dir=None,
                   use_embeddings=False):  # Desativado por padrão para compatibilidade com o dataset antigo
    """
    Processa o conjunto de dados para extrair features de NLP e combinar com features processadas.

    Implementa:
    1. Features básicas de texto
    2. TF-IDF com pesos refinados
    3. LDA para extração de tópicos
    4. Embeddings via modelo multilíngue (opcional)
    """
    print(f"\n=== Processando conjunto {dataset_name} ===")

    # Verificar se ambos os DataFrames têm o mesmo número de linhas
    if len(original_df) != len(processed_df):
        print(f"ERRO: Os DataFrames original e processado têm tamanhos diferentes: {len(original_df)} vs {len(processed_df)}")
        return None, None, None

    # Criar um DF para armazenar features de texto
    text_features_df = pd.DataFrame(index=original_df.index)

    # Verificar qual tipo de conjunto estamos processando
    is_training = dataset_name.lower() == 'train'

    # Inicializar dicionários para armazenar modelos
    if is_training:
        if tfidf_vectorizers is None:
            tfidf_vectorizers = {}
        if lda_models is None:
            lda_models = {}

    # Processar cada coluna de texto
    for col in text_columns:
        if col not in original_df.columns:
            continue

        print(f"Processando coluna: {col}")

        # Criar chave limpa para nomes de colunas
        col_key = col.replace(' ', '_').replace('?', '')
        col_key = re.sub(r'[^\w]', '', col_key)[:30]

        # Pré-processar textos
        processed_texts = original_df[col].apply(lambda x: preprocess_text(x))

        # 1. Extrair features básicas
        print("  Extraindo features básicas...")
        basic_features = processed_texts.apply(extract_basic_features)

        # Adicionar features básicas ao DataFrame
        for feature in ['word_count', 'char_count', 'avg_word_length', 'has_question', 'has_exclamation']:
            text_features_df[f"{col_key}_{feature}"] = basic_features.apply(lambda x: x.get(feature, 0))

        # Verificar se temos textos suficientes
        non_empty_texts = [text for text in processed_texts if isinstance(text, str) and len(text.strip()) > 0]

        if len(non_empty_texts) > 10:
            # 2. Extrair features TF-IDF com refinamento de pesos
            print("  Extraindo features TF-IDF refinadas...")
            if is_training:
                # Treinar vetorizador no conjunto de treino
                tfidf_df, vectorizer = extract_tfidf_features(processed_texts, max_features=200, fit=True)
                tfidf_vectorizers[col_key] = vectorizer
            else:
                # Usar vetorizador treinado no conjunto de treino
                if col_key in tfidf_vectorizers:
                    tfidf_df, _ = extract_tfidf_features(processed_texts, fit=False,
                                                       vectorizer=tfidf_vectorizers[col_key])
                else:
                    print(f"AVISO: Vetorizador para {col_key} não encontrado. Pulando TF-IDF.")
                    tfidf_df = None

            # Adicionar features TF-IDF ao DataFrame
            if tfidf_df is not None:
                for tfidf_col in tfidf_df.columns:
                    text_features_df[f"{col_key}_{tfidf_col}"] = tfidf_df[tfidf_col].values

            # 3. Extrair tópicos com LDA
            print("  Aplicando LDA para extração de tópicos...")
            if is_training:
                # Treinar modelo LDA no conjunto de treino
                lda_results = extract_topics_lda(processed_texts.tolist(), n_topics=5)
                if lda_results is not None:
                    lda_models[col_key] = {
                        'lda_model': lda_results['lda_model'],
                        'vectorizer': lda_results['vectorizer']
                    }

                    # Adicionar features de tópicos
                    for topic_col in lda_results['topic_features'].columns:
                        text_features_df[f"{col_key}_{topic_col}"] = lda_results['topic_features'][topic_col].values
            else:
                # Aplicar modelo LDA do treino
                if col_key in lda_models:
                    try:
                        lda_model = lda_models[col_key]['lda_model']
                        vectorizer = lda_models[col_key]['vectorizer']

                        # Transformar textos
                        dtm = vectorizer.transform(processed_texts)
                        doc_topic_dist = lda_model.transform(dtm)

                        # Adicionar features
                        for i in range(doc_topic_dist.shape[1]):
                            text_features_df[f"{col_key}_topic_{i}_prob"] = doc_topic_dist[:, i]

                        text_features_df[f"{col_key}_dominant_topic"] = doc_topic_dist.argmax(axis=1)
                    except Exception as e:
                        print(f"Erro ao aplicar LDA: {str(e)}")

            # 4. Extrair embeddings (opcional e desativado por padrão)
            if use_embeddings:
                print("  Extraindo embeddings (limitado às primeiras 1000 linhas)...")
                # Limitar a 1000 linhas para evitar uso excessivo de memória
                embeddings = extract_embeddings(processed_texts.head(1000).tolist())

                # Salvar embeddings
                if save_dir is not None:
                    emb_path = os.path.join(save_dir, f"{dataset_name}_{col_key}_embeddings.npy")
                    np.save(emb_path, embeddings)
                    print(f"  Embeddings salvos em: {emb_path}")

    # Concatenar features processadas com features de texto
    final_df = pd.concat([processed_df, text_features_df], axis=1)

    # Salvar arquivo processado
    if save_dir is not None:
        output_path = os.path.join(save_dir, f"{dataset_name}.csv")
        final_df.to_csv(output_path, index=False)
        print(f"Dataset com features de NLP salvo em: {output_path}")

    return final_df, tfidf_vectorizers, lda_models

def main():
    """Função principal para executar o processamento."""
    
    # Carregar datasets
    print("\nCarregando datasets processados...")
    try:
        processed_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
        processed_valid = pd.read_csv(os.path.join(PROCESSED_DIR, 'validation.csv'))
        processed_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))
        print(f"Datasets processados carregados: {processed_train.shape}, {processed_valid.shape}, {processed_test.shape}")
    except Exception as e:
        print(f"Erro ao carregar datasets processados: {e}")
        return

    print("\nCarregando datasets com texto original...")
    try:
        original_train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'), low_memory=False)
        original_valid = pd.read_csv(os.path.join(INPUT_DIR, 'validation.csv'), low_memory=False)
        original_test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'), low_memory=False)
        print(f"Datasets originais carregados: {original_train.shape}, {original_valid.shape}, {original_test.shape}")
    except Exception as e:
        print(f"Erro ao carregar datasets originais: {e}")
        return

    # Definir colunas de texto para processamento
    text_columns = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
        'Déjame un mensaje',
    ]
    
    print(f"\nColunas de texto a processar: {text_columns}")

    # Processar primeiro o conjunto de treino
    train_with_nlp, tfidf_vectorizers, lda_models = process_dataset(
        original_train,
        processed_train,
        text_columns,
        'train',
        save_dir=OUTPUT_DIR,
        use_embeddings=False  # Manter desativado para reproduzir 02_2_processed
    )

    # Processar validação e teste usando os modelos treinados
    valid_with_nlp, _, _ = process_dataset(
        original_valid,
        processed_valid,
        text_columns,
        'validation',
        tfidf_vectorizers=tfidf_vectorizers,
        lda_models=lda_models,
        save_dir=OUTPUT_DIR,
        use_embeddings=False
    )

    test_with_nlp, _, _ = process_dataset(
        original_test,
        processed_test,
        text_columns,
        'test',
        tfidf_vectorizers=tfidf_vectorizers,
        lda_models=lda_models,
        save_dir=OUTPUT_DIR,
        use_embeddings=False
    )

    # Salvar modelos para uso na pipeline de inferência
    os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
    import joblib
    joblib.dump(tfidf_vectorizers, os.path.join(PARAMS_DIR, "03_tfidf_vectorizers.joblib"))
    joblib.dump(lda_models, os.path.join(PARAMS_DIR, "03_lda_models.joblib"))
    
    print("\n=== Processamento de NLP concluído! ===")
    print(f"Modelos salvos em: {PARAMS_DIR}")
    print(f"Dados processados salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()