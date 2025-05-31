import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from textblob import TextBlob
from src.utils.feature_naming import standardize_feature_name
from src.utils.text_detection import detect_text_columns

def clean_text(text):
    """Limpa e normaliza texto.
    
    Args:
        text: Texto a ser limpo
        
    Returns:
        Texto limpo e normalizado
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Converter para minúsculas
    text = text.lower()
    # Remover URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remover emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remover pontuação e caracteres especiais, mas manter espaços e letras com acentos
    text = re.sub(r'[^\w\s\á\é\í\ó\ú\ñ\ü]', ' ', text)
    # Remover números
    text = re.sub(r'\d+', '', text)
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_basic_text_features(df, text_cols, fit=True, params=None):
    """Extrai features básicas de texto como comprimento, contagem de palavras, etc.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit (não usado nesta função)
        params: Dicionário com parâmetros (não usado nesta função)
        
    Returns:
        DataFrame com features de texto básicas adicionadas
        Dicionário com parâmetros inalterados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df_result.columns]
    
    # Limpar e normalizar texto
    for col in text_cols:
        df_result[f'{col}_clean'] = df_result[col].apply(clean_text)
    
    # Extrair features básicas
    for col in text_cols:
        # Comprimento do texto
        df_result[standardize_feature_name(f'{col}_length')] = df_result[col].str.len()
        
        # Contagem de palavras
        df_result[standardize_feature_name(f'{col}_word_count')] = df_result[col].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Presença de caracteres específicos
        df_result[standardize_feature_name(f'{col}_has_question')] = df_result[col].str.contains('\?', regex=True, na=False).astype(int)
        df_result[standardize_feature_name(f'{col}_has_exclamation')] = df_result[col].str.contains('!', regex=True, na=False).astype(int)
        
        # Média do tamanho das palavras
        df_result[standardize_feature_name(f'{col}_avg_word_length')] = df_result[col].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) and len(str(x).split()) > 0 else 0
        )
    
    return df_result, params

def extract_sentiment_features(df, text_cols, fit=True, params=None):
    """Extrai features de sentimento dos textos.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit (não usado nesta função)
        params: Dicionário com parâmetros (não usado nesta função)
        
    Returns:
        DataFrame com features de sentimento adicionadas
        Dicionário com parâmetros inalterados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df_result.columns]
    
    # Função para análise de sentimento com TextBlob
    def get_sentiment(text):
        try:
            if not isinstance(text, str) or pd.isna(text) or text == '':
                return 0
            return TextBlob(text).sentiment.polarity
        except:
            return 0
    
    # Aplicar análise de sentimento
    for col in text_cols:
        df_result[standardize_feature_name(f'{col}_sentiment')] = df_result[col].apply(get_sentiment)
    
    return df_result, params

def extract_tfidf_features(df, text_cols, fit=True, params=None):
    """Extrai features TF-IDF dos textos com parâmetros MELHORADOS.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto para processamento
        fit: Se True, ajusta os vetorizadores, caso contrário usa modelos existentes
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame com features TF-IDF adicionadas
        Dicionário com parâmetros atualizados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    if 'tfidf' not in params:
        params['tfidf'] = {}
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df_result.columns]
    
    for col in text_cols:
        clean_col = f'{col}_clean'
        
        if clean_col not in df_result.columns or df_result[clean_col].str.len().sum() == 0:
            continue
        
        if fit:
            tfidf = TfidfVectorizer(
                max_features=200,
                min_df=5,
                max_df=0.95,
                ngram_range=(1, 3),
                stop_words=None,
                use_idf=True,
                norm='l2',
                sublinear_tf=True,
                token_pattern=r'\b\w+\b|@\w+|\d+'
            )
            
            try:
                print(f"  Processando TF-IDF para '{col}' com novos parâmetros...")
                tfidf_matrix = tfidf.fit_transform(df_result[clean_col].fillna(''))
                feature_names = tfidf.get_feature_names_out()
                
                print(f"    - Features extraídas: {len(feature_names)}")
                print(f"    - Exemplos de n-gramas: {list(feature_names[:10])}")
                
                params['tfidf'][col] = {
                    'vectorizer': tfidf,
                    'feature_names': feature_names.tolist()
                }
                
                # CORREÇÃO: Usar o índice do df_result ao criar o DataFrame TF-IDF
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(), 
                    index=df_result.index,  # IMPORTANTE: Manter o mesmo índice
                    columns=[standardize_feature_name(f'{col}_tfidf_{term}') for term in feature_names]
                )
                
                # CORREÇÃO: Concatenar as novas colunas ao invés de atribuir uma por uma
                df_result = pd.concat([df_result, tfidf_df], axis=1)
                
            except Exception as e:
                print(f"Erro ao processar TF-IDF para '{col}': {e}")
                
        else:  # transform mode
            if col in params['tfidf'] and 'vectorizer' in params['tfidf'][col]:
                tfidf = params['tfidf'][col]['vectorizer']
                feature_names = params['tfidf'][col]['feature_names']
                
                try:
                    tfidf_matrix = tfidf.transform(df_result[clean_col].fillna(''))
                    
                    # CORREÇÃO: Usar o índice do df_result
                    tfidf_df = pd.DataFrame(
                        tfidf_matrix.toarray(), 
                        index=df_result.index,  # IMPORTANTE: Manter o mesmo índice
                        columns=[standardize_feature_name(f'{col}_tfidf_{term}') for term in feature_names]
                    )
                    
                    # CORREÇÃO: Concatenar as novas colunas
                    df_result = pd.concat([df_result, tfidf_df], axis=1)
                    
                except Exception as e:
                    print(f"Erro ao transformar TF-IDF para '{col}': {e}")
    
    return df_result, params

def extract_motivation_features(df, text_cols, fit=True, params=None):
    """Extrai features de motivação baseadas em palavras-chave.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit (não usado nesta função)
        params: Dicionário com parâmetros (não usado nesta função)
        
    Returns:
        DataFrame com features de motivação adicionadas
        Dicionário com parâmetros inalterados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Agrupar por categoria de motivação
    motivation_keywords = {
        'trabajo': 'work', 'empleo': 'work', 'carrera': 'work', 'profesional': 'work', 
        'puesto': 'work', 'laboral': 'work', 'sueldo': 'work', 'trabajar': 'work',
        'viaje': 'travel', 'viajar': 'travel', 'turismo': 'travel', 'países': 'travel', 
        'extranjero': 'travel', 'mundo': 'travel', 'internacional': 'travel',
        'comunicar': 'communication', 'comunicación': 'communication', 'hablar': 'communication', 
        'entender': 'communication', 'expresar': 'communication',
        'estudio': 'education', 'estudiar': 'education', 'universidad': 'education', 
        'curso': 'education', 'aprender': 'education', 'educación': 'education',
        'mejor': 'improvement', 'mejorar': 'improvement', 'crecer': 'improvement', 
        'avanzar': 'improvement', 'progresar': 'improvement', 'desarrollar': 'improvement',
        'oportunidad': 'opportunity', 'futuro': 'opportunity', 'posibilidad': 'opportunity', 
        'chance': 'opportunity', 'opción': 'opportunity'
    }
    
    for col in text_cols:
        # Verificar se temos a coluna limpa
        clean_col = f'{col}_clean'
        if clean_col not in df_result.columns:
            continue
        
        # Inicializar colunas de categorias de motivação
        for category in set(motivation_keywords.values()):
            df_result[standardize_feature_name(f'{col}_motiv_{category}')] = 0
        
        # Processar cada linha
        for i, (idx, text) in enumerate(df_result[clean_col].items()):
            if not isinstance(text, str) or text == '':
                continue
                
            # Verificar presença de cada palavra-chave
            for keyword, category in motivation_keywords.items():
                if keyword in text:
                    # Use .at para acesso seguro por índice
                    df_result.at[idx, standardize_feature_name(f'{col}_motiv_{category}')] += 1
        
        # Normalizar por comprimento do texto (para textos não vazios)
        word_count_col = f'{col}_word_count'
        if word_count_col in df_result.columns:
            mask = df_result[word_count_col] > 0
            for category in set(motivation_keywords.values()):
                col_name = standardize_feature_name(f'{col}_motiv_{category}')
                df_result.loc[mask, standardize_feature_name(f'{col_name}_norm')] = df_result.loc[mask, col_name] / df_result.loc[mask, word_count_col]
    
    return df_result, params

def extract_discriminative_features(df, text_cols, fit=True, params=None):
    """Identifica e cria features para termos com maior poder discriminativo para conversão.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto para processamento
        fit: Se True, calcula termos discriminativos, caso contrário usa os já identificados
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame com features discriminativas adicionadas
        Dicionário com parâmetros atualizados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    if 'discriminative_terms' not in params:
        params['discriminative_terms'] = {}
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df_result.columns]
    
    # Verificar se a coluna target existe
    if 'target' not in df_result.columns:
        return df_result, params
    
    if fit:
        # Calcular taxa geral de conversão
        overall_conv_rate = df_result['target'].mean()
        min_term_freq = 50  # Mínimo de ocorrências para considerar um termo relevante
        
        for col in text_cols:
            # Só analisar termos do TF-IDF se existirem no dataframe
            tfidf_cols = [c for c in df_result.columns if c.startswith(f'{col}_tfidf_')]
            if not tfidf_cols:
                continue
            
            term_stats = []
            for tfidf_col in tfidf_cols:
                # Extrair o termo do nome da coluna
                term = tfidf_col.split(f'{col}_tfidf_')[1]
                
                # Verificar presença do termo nos dados
                has_term = df_result[tfidf_col] > 0
                term_freq = has_term.sum()
                
                if term_freq >= min_term_freq:
                    # Calcular taxa de conversão quando o termo está presente
                    conv_with_term = df_result.loc[has_term, 'target'].mean()
                    
                    # Calcular lift (razão entre taxa com termo e taxa geral)
                    if overall_conv_rate > 0:
                        lift = conv_with_term / overall_conv_rate
                        
                        # Significância estatística básica - margem de 20% ou mais
                        if abs(lift - 1.0) >= 0.2:
                            term_stats.append((term, lift, term_freq, conv_with_term))
            
            # Ordenar por lift (poder discriminativo)
            term_stats.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)
            
            # Guardar os top 5 termos com maior lift positivo e negativo
            pos_terms = [(t, l) for t, l, _, _ in term_stats if l > 1.0][:5]
            neg_terms = [(t, l) for t, l, _, _ in term_stats if l < 1.0][:5]
            
            params['discriminative_terms'][col] = {
                'positive': pos_terms,
                'negative': neg_terms
            }
    
    # Criar features para termos discriminativos
    for col, terms_dict in params['discriminative_terms'].items():
        # Termos positivos (associados à conversão)
        for term, lift in terms_dict.get('positive', []):
            term_col = standardize_feature_name(f'{col}_tfidf_{term}')
            if term_col in df_result.columns:
                # Criar feature binária para presença do termo
                df_result[standardize_feature_name(f'{col}_high_conv_term_{term}')] = (df_result[term_col] > 0).astype(int)
        
        # Termos negativos (associados à não-conversão)
        for term, lift in terms_dict.get('negative', []):
            term_col = standardize_feature_name(f'{col}_tfidf_{term}')
            if term_col in df_result.columns:
                # Criar feature binária para presença do termo
                df_result[standardize_feature_name(f'{col}_low_conv_term_{term}')] = (df_result[term_col] > 0).astype(int)
        
        # Criar feature agregada para presença de qualquer termo positivo ou negativo
        if terms_dict.get('positive', []):
            pos_features = [standardize_feature_name(f'{col}_high_conv_term_{term}') for term, _ in terms_dict['positive'] 
                           if standardize_feature_name(f'{col}_high_conv_term_{term}') in df_result.columns]
            if pos_features:
                df_result[standardize_feature_name(f'{col}_has_any_high_conv_term')] = df_result[pos_features].max(axis=1)
                df_result[standardize_feature_name(f'{col}_num_high_conv_terms')] = df_result[pos_features].sum(axis=1)
        
        if terms_dict.get('negative', []):
            neg_features = [standardize_feature_name(f'{col}_low_conv_term_{term}') for term, _ in terms_dict['negative'] 
                           if standardize_feature_name(f'{col}_low_conv_term_{term}') in df_result.columns]
            if neg_features:
                df_result[standardize_feature_name(f'{col}_has_any_low_conv_term')] = df_result[neg_features].max(axis=1)
                df_result[standardize_feature_name(f'{col}_num_low_conv_terms')] = df_result[neg_features].sum(axis=1)
    
    return df_result, params

def text_feature_engineering(df, fit=True, params=None):
    """Executa todo o pipeline de processamento de texto."""
    if params is None:
        params = {}
    
    # NOVA ABORDAGEM: Usar detector unificado
    print("\n🔍 Detectando colunas de texto para processamento...")
    text_cols = detect_text_columns(
        df,
        confidence_threshold=0.6,
        exclude_patterns=['_encoded', '_norm', '_clean', '_tfidf', '_original']
    )
    
    if not text_cols:
        print("  ⚠️ Nenhuma coluna de texto detectada")
        return df, params
    
    print(f"\n✓ Processamento de texto iniciado para {len(text_cols)} colunas detectadas")
    
    # Filtrar apenas colunas existentes no dataframe
    text_cols = [col for col in text_cols if col in df.columns]
    
    if not text_cols:
        return df, params
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    print(f"\nProcessamento de texto iniciado para {len(text_cols)} colunas...")
    
    # Aplicar pipeline de processamento de texto
    df_result, params = extract_basic_text_features(df_result, text_cols, fit, params)
    df_result, params = extract_sentiment_features(df_result, text_cols, fit, params)
    df_result, params = extract_tfidf_features(df_result, text_cols, fit, params)
    df_result, params = extract_motivation_features(df_result, text_cols, fit, params)
    df_result, params = extract_discriminative_features(df_result, text_cols, fit, params)
    
    # Remover colunas temporárias de texto limpo
    columns_to_drop = [f'{col}_clean' for col in text_cols if f'{col}_clean' in df_result.columns]
    df_result = df_result.drop(columns=columns_to_drop, errors='ignore')
    
    # Contar features TF-IDF criadas
    tfidf_count = len([col for col in df_result.columns if '_tfidf_' in col])
    print(f"Total de features TF-IDF criadas: {tfidf_count}")
    
    return df_result, params