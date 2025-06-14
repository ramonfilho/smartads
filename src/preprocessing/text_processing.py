import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from textblob import TextBlob
from src.utils.feature_naming import standardize_feature_name
from src.utils.column_type_classifier import ColumnTypeClassifier
from src.utils.parameter_manager import ParameterManager

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

def extract_basic_text_features(df, text_cols, fit=True, param_manager=None):
    """Extrai features básicas de texto como comprimento, contagem de palavras, etc.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit (não usado nesta função)
        param_manager: Instância do ParameterManager
        
    Returns:
        DataFrame com features de texto básicas adicionadas
        ParameterManager inalterado
    """
    # Inicializar parâmetros
    if param_manager is None:
        from src.utils.parameter_manager import ParameterManager
        param_manager = ParameterManager()
    
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
    
    return df_result, param_manager

def extract_sentiment_features(df, text_cols, fit=True, param_manager=None):
    """Extrai features de sentimento dos textos.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit (não usado nesta função)
        param_manager: Instância do ParameterManager
        
    Returns:
        DataFrame com features de sentimento adicionadas
        ParameterManager inalterado
    """
    # Inicializar parâmetros
    if param_manager is None:
        from src.utils.parameter_manager import ParameterManager
        param_manager = ParameterManager()
    
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
    
    return df_result, param_manager

def extract_tfidf_features(df, text_cols, fit=True, param_manager=None):
    """Extrai features TF-IDF dos textos - VERSÃO COM PARAMETER MANAGER"""
    
    # Compatibilidade: criar param_manager se não fornecido
    if param_manager is None:
        param_manager = ParameterManager()
    
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df_result.columns]
    
    for col in text_cols:
        clean_col = f'{col}_clean'
        
        if clean_col not in df_result.columns:
            continue
            
        # Verificar se temos textos válidos
        texts = df_result[clean_col].fillna('')
        if texts.str.len().sum() == 0:
            print(f"  ⚠️ Coluna {clean_col} sem texto válido, pulando...")
            continue
        
        if fit:
            # MODO FIT: Criar e treinar vetorizador
            from sklearn.feature_extraction.text import TfidfVectorizer
            
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
                print(f"  🔄 Processando TF-IDF para '{col}'...")
                
                # Fit e transform
                tfidf_matrix = tfidf.fit_transform(texts)
                feature_names = tfidf.get_feature_names_out()
                
                print(f"    ✓ Features extraídas: {len(feature_names)}")
                
                # MUDANÇA: Salvar usando param_manager
                param_manager.save_vectorizer(
                    {
                        'vectorizer': tfidf,
                        'feature_names': feature_names.tolist()
                    },
                    name=col,
                    category='tfidf'
                )
                
                # Criar DataFrame com as features
                from src.utils.feature_naming import create_tfidf_column_name
                column_names = [create_tfidf_column_name(col, term) for term in feature_names]
                
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    index=df_result.index,
                    columns=column_names
                )
                
                df_result = pd.concat([df_result, tfidf_df], axis=1)
                
                # Rastrear features criadas
                param_manager.track_created_features(column_names)
                
                print(f"    ✓ {len(column_names)} features TF-IDF adicionadas")
                
            except Exception as e:
                print(f"  ❌ Erro ao processar TF-IDF para '{col}': {e}")
                
        else:
            # MODO TRANSFORM: Usar vetorizador existente
            # MUDANÇA: Recuperar usando param_manager
            vectorizer_data = param_manager.get_vectorizer(col, 'tfidf')
            
            if not vectorizer_data:
                print(f"  ⚠️ Vetorizador não encontrado para '{col}', pulando...")
                continue
                
            try:
                tfidf = vectorizer_data['vectorizer']
                feature_names = vectorizer_data['feature_names']
                
                print(f"  🔄 Aplicando TF-IDF pré-treinado para '{col}'...")
                
                # Transform
                tfidf_matrix = tfidf.transform(texts)
                
                # Criar DataFrame
                from src.utils.feature_naming import create_tfidf_column_name
                column_names = [create_tfidf_column_name(col, term) for term in feature_names]
                
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    index=df_result.index,
                    columns=column_names
                )
                
                df_result = pd.concat([df_result, tfidf_df], axis=1)
                
                print(f"    ✓ {len(column_names)} features TF-IDF aplicadas")
                
            except Exception as e:
                print(f"  ❌ Erro ao transformar TF-IDF para '{col}': {e}")
    
    # Retornar apenas DataFrame (params agora está no param_manager)
    return df_result, param_manager

def extract_motivation_features(df, text_cols, fit=True, param_manager=None):
    """Extrai features de motivação baseadas em palavras-chave.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit (não usado nesta função)
        param_manager: Instância do ParameterManager
        
    Returns:
        DataFrame com features de motivação adicionadas
        ParameterManager inalterado
    """
    # Inicializar parâmetros
    if param_manager is None:
        from src.utils.parameter_manager import ParameterManager
        param_manager = ParameterManager()
    
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
    
    return df_result, param_manager

def extract_discriminative_features(df, text_cols, fit=True, param_manager=None):
    """Identifica e cria features para termos com maior poder discriminativo para conversão."""
    if param_manager is None:
        param_manager = ParameterManager()
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df_result.columns]
    
    # Verificar se a coluna target existe
    if 'target' not in df_result.columns:
        return df_result, param_manager
    
    if fit:
        # Calcular taxa geral de conversão
        overall_conv_rate = df_result['target'].mean()
        min_term_freq = 50
        
        # Dicionário para armazenar termos discriminativos
        discriminative_terms = {}
        
        for col in text_cols:
            # Filtrar colunas TF-IDF
            tfidf_cols = [c for c in df_result.columns 
                         if c.startswith(f'{col}_tfidf_') and 
                         c.count('_tfidf_') == 1]
            
            if not tfidf_cols:
                continue
            
            print(f"\nProcessando {col}:")
            print(f"  Número de colunas TF-IDF: {len(tfidf_cols)}")
            print(f"  DataFrame tem {len(df_result)} linhas")
            
            term_stats = []
            
            for tfidf_col in tfidf_cols:
                try:
                    # Extrair o termo do nome da coluna
                    term = tfidf_col.split(f'{col}_tfidf_')[1]
                    
                    # Verificar se a coluna existe e tem dados válidos
                    if tfidf_col not in df_result.columns:
                        continue
                        
                    # Verificar presença do termo (garantir que é Series numérica)
                    tfidf_values = pd.to_numeric(df_result[tfidf_col], errors='coerce').fillna(0)
                    has_term = tfidf_values > 0
                    term_freq = has_term.sum()
                    
                    if term_freq >= min_term_freq:
                        # Calcular taxa de conversão para quem tem o termo
                        conv_with_term = df_result.loc[has_term, 'target'].mean()
                        
                        # Calcular lift
                        if overall_conv_rate > 0:
                            lift = conv_with_term / overall_conv_rate
                            
                            if abs(lift - 1.0) >= 0.2:
                                term_stats.append((term, lift, term_freq, conv_with_term))
                
                except Exception as e:
                    # Remover o print do aviso ou torná-lo mais silencioso
                    # print(f"  ⚠️ Aviso: Erro ao processar {tfidf_col}: {str(e)}")
                    continue
            
            # Ordenar e selecionar top termos
            term_stats.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)
            
            pos_terms = [(t, l) for t, l, _, _ in term_stats if l > 1.0][:5]
            neg_terms = [(t, l) for t, l, _, _ in term_stats if l < 1.0][:5]
            
            discriminative_terms[col] = {
                'positive': pos_terms,
                'negative': neg_terms
            }
            
            # Debug output...
        
        # Salvar no param_manager
        param_manager.params['text_processing']['discriminative_terms'] = discriminative_terms
    else:
        # Recuperar do param_manager
        discriminative_terms = param_manager.params['text_processing'].get('discriminative_terms', {})
    
    # Criar features para termos discriminativos
    for col, terms_dict in discriminative_terms.items():
        # Termos positivos
        for term, lift in terms_dict.get('positive', []):
            term_col = standardize_feature_name(f'{col}_tfidf_{term}')
            if term_col in df_result.columns:
                feature_name = standardize_feature_name(f'{col}_high_conv_term_{term}')
                df_result[feature_name] = (df_result[term_col] > 0).astype(int)
        
        # Termos negativos
        for term, lift in terms_dict.get('negative', []):
            term_col = standardize_feature_name(f'{col}_tfidf_{term}')
            if term_col in df_result.columns:
                feature_name = standardize_feature_name(f'{col}_low_conv_term_{term}')
                df_result[feature_name] = (df_result[term_col] > 0).astype(int)
        
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
    
    print(f"  DEBUG: DataFrame final: {len(df_result)} linhas")
    
    return df_result, param_manager

def text_feature_engineering(df, fit=True, param_manager=None):
    """Executa todo o pipeline de processamento de texto."""

    # Criar param_manager se não fornecido
    if param_manager is None:
        from src.utils.parameter_manager import ParameterManager
        param_manager = ParameterManager()
    
    print("\n=== DEBUG: RASTREAMENTO DE COLUNAS EM text_feature_engineering ===")
    print(f"Colunas na entrada: {len(df.columns)}")
    initial_cols = set(df.columns)

    # Verificar se temos classificações
    classifications = param_manager.get_preprocessing_params('column_classifications')
    
    if classifications:
        print("\n✓ Usando classificações existentes para processamento de texto")
        
        # Filtrar apenas colunas de texto
        exclude_patterns = ['_encoded', '_norm', '_clean', '_tfidf', '_original']
        text_cols = [
            col for col, info in classifications.items()
            if col in df.columns  # Importante: verificar se a coluna ainda existe
            and info['type'] == 'text'
            and info['confidence'] >= 0.6
            and not any(pattern in col for pattern in exclude_patterns)
        ]
    else:
        # Fallback: reclassificar se necessário
        print("\n🔍 Detectando colunas de texto para processamento...")
        
        from src.utils.column_type_classifier import ColumnTypeClassifier
        classifier = ColumnTypeClassifier(
            use_llm=False,
            use_classification_cache=True,
            confidence_threshold=0.6
        )
        
        classifications = classifier.classify_dataframe(df)
        param_manager.save_preprocessing_params('column_classifications', classifications)
        
        exclude_patterns = ['_encoded', '_norm', '_clean', '_tfidf', '_original']
        text_cols = [
            col for col, info in classifications.items()
            if info['type'] == classifier.TEXT 
            and info['confidence'] >= 0.6
            and not any(pattern in col for pattern in exclude_patterns)
        ]
    
    # APLICAR EXCLUSÕES
    excluded_cols = param_manager.params['feature_engineering'].get('excluded_columns', [])
    original_text_cols = text_cols.copy()
    text_cols = [col for col in text_cols if col not in excluded_cols]
    
    if len(original_text_cols) != len(text_cols):
        print(f"  ℹ️ {len(original_text_cols) - len(text_cols)} colunas de texto excluídas do processamento")
        for col in original_text_cols:
            if col not in text_cols:
                print(f"     - {col}")
    
    if not text_cols:
        print("  ⚠️ Nenhuma coluna de texto para processar após exclusões")
        return df, param_manager
    
    print(f"\n✓ Processamento de texto iniciado para {len(text_cols)} colunas")
    
    # Filtrar apenas colunas existentes no dataframe
    text_cols = [col for col in text_cols if col in df.columns]
    
    if not text_cols:
        return df, param_manager
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Aplicar pipeline de processamento de texto
    df_result, param_manager = extract_basic_text_features(df_result, text_cols, fit, param_manager)
    df_result, param_manager = extract_sentiment_features(df_result, text_cols, fit, param_manager)
    df_result, param_manager = extract_tfidf_features(df_result, text_cols, fit, param_manager=param_manager)
    df_result, param_manager = extract_motivation_features(df_result, text_cols, fit, param_manager)
    df_result, param_manager = extract_discriminative_features(df_result, text_cols, fit, param_manager=param_manager)
    
    # Remover colunas temporárias de texto limpo
    columns_to_drop = [f'{col}_clean' for col in text_cols if f'{col}_clean' in df_result.columns]
    df_result = df_result.drop(columns=columns_to_drop, errors='ignore')
    
    # Contar features TF-IDF criadas
    tfidf_count = len([col for col in df_result.columns if '_tfidf_' in col])
    print(f"Total de features TF-IDF criadas: {tfidf_count}")
    
    final_cols = set(df_result.columns)
    removed_cols = initial_cols - final_cols
    added_cols = final_cols - initial_cols
    
    print(f"\nColunas removidas em text_feature_engineering: {len(removed_cols)}")
    for col in sorted(removed_cols):
        print(f"  - {col}")
    
    print(f"\nColunas adicionadas: {len(added_cols)}")
    
    return df_result, param_manager