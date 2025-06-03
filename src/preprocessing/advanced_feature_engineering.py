"""
Módulo de engenharia de features avançada.

Este módulo implementa técnicas avançadas de processamento de texto
e engenharia de features baseadas na análise de erros do modelo.
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
from src.preprocessing.text_processing import clean_text
from src.utils.feature_naming import standardize_feature_name

warnings.filterwarnings('ignore')

def identify_text_columns(df):
    """
    Identifica colunas de texto no DataFrame usando o sistema unificado.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        Lista de nomes de colunas de texto identificadas
    """
    from src.utils.column_type_classifier import ColumnTypeClassifier
    
    # Usar o classificador unificado
    classifier = ColumnTypeClassifier(
        use_llm=False,
        use_classification_cache=True,
        confidence_threshold=0.6
    )
    
    classifications = classifier.classify_dataframe(df)
    
    # Filtrar apenas colunas de texto
    exclude_patterns = ['_encoded', '_norm', '_clean', '_tfidf', '_original']
    text_cols = [
        col for col, info in classifications.items()
        if info['type'] == classifier.TEXT 
        and info['confidence'] >= 0.6
        and not any(pattern in col for pattern in exclude_patterns)
    ]
    
    print(f"Colunas de texto identificadas: {len(text_cols)}")
    for col in text_cols:
        print(f"  - {col}")
    
    return text_cols

def refine_tfidf_weights_fixed(df, text_cols, fit=True, params=None):
    """
    Versão corrigida da função refine_tfidf_weights que lida adequadamente com
    tipos de colunas e garante tipos consistentes.
    """
    if params is None:
        params = {}
    
    if 'refined_tfidf' not in params:
        params['refined_tfidf'] = {}
    
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df.columns]
    
    # Identificar apenas colunas de texto natural, não derivadas
    original_text_cols = []
    for col in text_cols:
        # Verificar se é uma coluna de texto original e não derivada
        if not any(suffix in col for suffix in ['_length', '_word_count', '_sentiment', '_tfidf_', '_has_', '_motiv_']):
            original_text_cols.append(col)
    
    print(f"Refinando pesos TF-IDF para {len(original_text_cols)} colunas de texto original")
    
    # Definir termos importantes
    identified_terms = {
        # Termos padrão baseados na análise prévia
        'Déjame_un_mensaje': ['aprender', 'inglés', 'fluido', 'fluidez', 'comunicarme', 'trabajo', 'mejora'],
        'Cuando_hables_inglés_con_fluidez': ['oportunidades', 'trabajo', 'mejor', 'comunicación', 'viajar', 'mejorar'],
        '_Qué_esperas_aprender_en_la_Semana': ['hablar', 'entender', 'comunicarme', 'fluidez', 'método']
    }
    
    # Lista de stop words em espanhol
    spanish_stopwords = [
        'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra',
        'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e', 'el', 'ella'
    ]
    
    for col in original_text_cols:
        # Encontrar o nome base da coluna para termos importantes
        col_base = None
        for possible_base in identified_terms.keys():
            if possible_base in col:
                col_base = possible_base
                break
        
        if col_base is None:
            col_base = list(identified_terms.keys())[0]  # Usar o primeiro como fallback
            
        # Verificar se temos texto limpo, senão usar a coluna original
        clean_col = f'{col}_original' if f'{col}_original' in df_result.columns else col
        
        # Filtrar apenas textos não vazios
        non_empty = df_result[clean_col].fillna('').str.len() > 5
        if non_empty.sum() < 10:
            print(f"  Ignorando coluna {col}: poucos dados não vazios")
            continue
        
        if fit:
            important_terms = identified_terms.get(col_base, [])
            
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Usar apenas textos não vazios para o fit
                valid_texts = df_result.loc[non_empty, clean_col].fillna('')
                
                tfidf = TfidfVectorizer(
                    max_features=20,  # Limitar para evitar problemas
                    min_df=5,         # Requer pelo menos 5 documentos
                    max_df=0.9,       # Ignorar termos muito comuns
                    use_idf=True,
                    norm='l2',
                    ngram_range=(1, 2),
                    stop_words=spanish_stopwords
                )
                
                # Ajustar e transformar
                tfidf_matrix = tfidf.fit_transform(valid_texts)
                feature_names = tfidf.get_feature_names_out()
                
                if len(feature_names) == 0:
                    print(f"  Ignorando coluna {col}: nenhum termo extraído no TF-IDF")
                    continue
                
                # Armazenar o vetorizador e os nomes das features
                params['refined_tfidf'][col] = {
                    'vectorizer': tfidf,
                    'feature_names': feature_names.tolist(),
                    'important_terms': important_terms
                }
                
                # Criar colunas TF-IDF com pesos ajustados
                tfidf_array = tfidf_matrix.toarray()
                
                # Preencher DataFrame original incluindo linhas que foram filtradas
                for i, term in enumerate(feature_names):
                    # Inicializar coluna com zeros
                    df_result[standardize_feature_name(f'{col}_refined_tfidf_{term}')] = 0.0
                    
                    # Preencher valores para textos não vazios
                    df_result.loc[non_empty, standardize_feature_name(f'{col}_refined_tfidf_{term}')] = tfidf_array[:, i]
                    
                    # Aumentar o peso se for um termo importante
                    if any(imp_term in term for imp_term in important_terms):
                        df_result[standardize_feature_name(f'{col}_refined_tfidf_{term}')] *= 2.0  # Dobrar o peso
                
                print(f"  Criadas {len(feature_names)} features TF-IDF para {col}")
                
            except Exception as e:
                print(f"  Erro ao processar TF-IDF para '{col}': {e}")
                # Não criar colunas parciais ou inconsistentes
                
        else:
            # Usar vetorizador treinado anteriormente
            if col in params['refined_tfidf'] and 'vectorizer' in params['refined_tfidf'][col]:
                try:
                    tfidf = params['refined_tfidf'][col]['vectorizer']
                    feature_names = params['refined_tfidf'][col]['feature_names']
                    important_terms = params['refined_tfidf'][col].get('important_terms', [])
                    
                    # Transformar apenas textos não vazios
                    valid_texts = df_result.loc[non_empty, clean_col].fillna('')
                    tfidf_matrix = tfidf.transform(valid_texts)
                    tfidf_array = tfidf_matrix.toarray()
                    
                    # Preencher DataFrame completo
                    for i, term in enumerate(feature_names):
                        # Inicializar coluna com zeros
                        df_result[f'{col}_refined_tfidf_{term}'] = 0.0
                        
                        # Preencher valores para textos não vazios
                        df_result.loc[non_empty, f'{col}_refined_tfidf_{term}'] = tfidf_array[:, i]
                        
                        # Aumentar o peso se for um termo importante
                        if any(imp_term in term for imp_term in important_terms):
                            df_result[f'{col}_refined_tfidf_{term}'] *= 2.0
                    
                    print(f"  Aplicadas {len(feature_names)} features TF-IDF para {col}")
                    
                except Exception as e:
                    print(f"  Erro ao transformar TF-IDF para '{col}': {e}")
    
    return df_result, params

def create_text_embeddings_simple_fixed(df, text_cols, fit=True, params=None):
    """
    Versão corrigida da função create_text_embeddings_simple que verifica
    tipos de dados e evita operações mistas.
    """
    if params is None:
        params = {}
    
    if 'embeddings' not in params:
        params['embeddings'] = {}
    
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df.columns]
    
    print(f"Criando embeddings para {len(text_cols)} colunas de texto")
    
    # Usar a média de TF-IDF como embedding simplificado
    for col in text_cols:
        # Verificar colunas TF-IDF
        tfidf_cols = [c for c in df_result.columns if 
                      (c.startswith(f'{col}_tfidf_') or c.startswith(f'{col}_refined_tfidf_'))]
        
        if not tfidf_cols:
            print(f"  Ignorando {col}: nenhuma coluna TF-IDF encontrada")
            continue
        
        # Garantir que apenas colunas numéricas sejam usadas
        numeric_tfidf_cols = []
        for tfidf_col in tfidf_cols:
            # Verificar se a coluna é numérica
            if pd.api.types.is_numeric_dtype(df_result[tfidf_col]):
                numeric_tfidf_cols.append(tfidf_col)
            else:
                print(f"  Ignorando coluna não numérica: {tfidf_col}")
        
        if not numeric_tfidf_cols:
            print(f"  Ignorando {col}: nenhuma coluna TF-IDF numérica")
            continue
        
        print(f"  Criando embeddings para {col} usando {len(numeric_tfidf_cols)} features TF-IDF")
        
        try:
            # Dimensão 1: Média dos valores TF-IDF
            df_result[standardize_feature_name(f'{col}_embedding_mean')] = df_result[numeric_tfidf_cols].mean(axis=1)
            
            # Dimensão 2: Máximo dos valores TF-IDF
            df_result[standardize_feature_name(f'{col}_embedding_max')] = df_result[numeric_tfidf_cols].max(axis=1)
            
            # Dimensão 3: Desvio padrão
            df_result[standardize_feature_name(f'{col}_embedding_std')] = df_result[numeric_tfidf_cols].std(axis=1)
            
            # Dimensão 4: Número de termos não-zero
            df_result[standardize_feature_name(f'{col}_embedding_nonzero')] = (df_result[numeric_tfidf_cols] > 0).sum(axis=1) / len(numeric_tfidf_cols)
            
            if fit:
                params['embeddings'][col] = {
                    'method': 'tfidf_stats',
                    'dimensions': ['mean', 'max', 'std', 'nonzero'],
                    'tfidf_cols': numeric_tfidf_cols
                }
                
        except Exception as e:
            print(f"  Erro ao criar embeddings para {col}: {e}")
            # Limpar qualquer coluna parcial que possa ter sido criada
            for dim in ['mean', 'max', 'std', 'nonzero']:
                if f'{col}_embedding_{dim}' in df_result.columns:
                    df_result = df_result.drop(columns=[f'{col}_embedding_{dim}'])
    
    return df_result, params

def refine_tfidf_weights(df, text_cols, identified_terms=None, fit=True, params=None):
    """
    Refina os pesos TF-IDF para dar mais importância aos termos identificados
    na análise de erros.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto
        identified_terms: Dicionário de termos identificados por coluna
        fit: Se True, ajusta vetorizadores, caso contrário usa existentes
        params: Parâmetros para transform
        
    Returns:
        DataFrame com features TF-IDF refinadas
        Dicionário com parâmetros atualizados
    """
    if params is None:
        params = {}
    
    if 'refined_tfidf' not in params:
        params['refined_tfidf'] = {}
    
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df.columns]
    
    # Definir termos de alta importância para cada coluna
    if not identified_terms:
        # Termos padrão baseados na análise de erros
        identified_terms = {
            'Déjame_un_mensaje': ['aprender', 'inglés', 'fluido', 'fluidez', 'comunicarme', 'trabajo', 'mejora'],
            'Cuando_hables_inglés_con_fluidez': ['oportunidades', 'trabajo', 'mejor', 'comunicación', 'viajar', 'mejorar'],
            '_Qué_esperas_aprender_en_la_Semana': ['hablar', 'entender', 'comunicarme', 'fluidez', 'método']
        }
    
    # Lista de stop words em espanhol
    spanish_stopwords = [
        'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra',
        'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e', 'el', 'ella',
        'ellas', 'ellos', 'en', 'entre', 'era', 'erais', 'eran', 'eras', 'eres', 'es',
        'esa', 'esas', 'ese', 'eso', 'esos', 'esta', 'estaba', 'estabais', 'estaban',
        'estabas', 'estad', 'estada', 'estadas', 'estado', 'estados', 'estamos', 'estando',
        'estar', 'estaremos', 'estará', 'estarán', 'estarás', 'estaré', 'estaréis',
        'estaría', 'estaríais', 'estaríamos', 'estarían', 'estarías'
    ]
    
    for col in text_cols:
        # Encontrar o nome base da coluna para associar aos termos identificados
        col_base = None
        for possible_base in identified_terms.keys():
            if possible_base in col:
                col_base = possible_base
                break
        
        if col_base is None:
            # Se não encontrou uma base exata, usar a coluna mais similar
            for possible_base in identified_terms.keys():
                # Verificar similaridade de strings simplificada
                if any(term in col.lower() for term in possible_base.lower().split('_')):
                    col_base = possible_base
                    break
        
        if col_base is None:
            # Se ainda não encontrou, usar a primeira base como fallback
            col_base = list(identified_terms.keys())[0]
            
        # Verificar se temos texto limpo, senão limpar
        clean_col = f'{col}_clean'
        if clean_col not in df_result.columns:
            # Limpar texto
            df_result[clean_col] = df_result[col].apply(lambda x: 
                clean_text(str(x)) if pd.notna(x) else "")
        
        if fit:
            # Configurar TF-IDF com termos importantes
            important_terms = identified_terms.get(col_base, [])
            
            # Aumentar pesos para termos importantes
            # Usamos n-gramas (1,2) para capturar frases como "hablar inglés"
            tfidf = TfidfVectorizer(
                max_features=100,  # Aumentar para capturar mais termos
                min_df=3,
                use_idf=True,
                norm='l2',
                ngram_range=(1, 2),
                stop_words=spanish_stopwords
            )
            
            # Ajustar e transformar
            try:
                tfidf_matrix = tfidf.fit_transform(df_result[clean_col].fillna(''))
                feature_names = tfidf.get_feature_names_out()
                
                # Armazenar o vetorizador e os nomes das features
                params['refined_tfidf'][col] = {
                    'vectorizer': tfidf,
                    'feature_names': feature_names.tolist(),
                    'important_terms': important_terms
                }
                
                # Criar colunas TF-IDF com pesos ajustados
                tfidf_array = tfidf_matrix.toarray()
                
                # Aplicar boosting para termos importantes
                boost_factor = 2.0  # Dobrar o peso de termos importantes
                for i, term in enumerate(feature_names):
                    df_result[f'{col}_refined_tfidf_{term}'] = tfidf_array[:, i]
                    
                    # Aumentar o peso se for um termo importante ou contiver um termo importante
                    if any(imp_term in term for imp_term in important_terms):
                        df_result[f'{col}_refined_tfidf_{term}'] *= boost_factor
                
            except Exception as e:
                print(f"Erro ao processar TF-IDF refinado para '{col}': {e}")
                
        else:
            # Usar vetorizador treinado anteriormente
            if col in params['refined_tfidf'] and 'vectorizer' in params['refined_tfidf'][col]:
                tfidf = params['refined_tfidf'][col]['vectorizer']
                feature_names = params['refined_tfidf'][col]['feature_names']
                important_terms = params['refined_tfidf'][col].get('important_terms', [])
                
                try:
                    # Transformar usando o vetorizador existente
                    tfidf_matrix = tfidf.transform(df_result[clean_col].fillna(''))
                    tfidf_array = tfidf_matrix.toarray()
                    
                    # Adicionar colunas com boost nos termos importantes
                    boost_factor = 2.0
                    for i, term in enumerate(feature_names):
                        df_result[f'{col}_refined_tfidf_{term}'] = tfidf_array[:, i]
                        
                        # Aplicar boost para termos importantes
                        if any(imp_term in term for imp_term in important_terms):
                            df_result[f'{col}_refined_tfidf_{term}'] *= boost_factor
                    
                except Exception as e:
                    print(f"Erro ao transformar TF-IDF refinado para '{col}': {e}")
    
    return df_result, params

def create_text_embeddings_simple(df, text_cols, fit=True, params=None):
    """
    Cria embeddings simples para capturar o contexto semântico dos textos
    usando estatísticas do TF-IDF (sem depender de modelos externos).
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto
        fit: Se True, armazena parâmetros, caso contrário usa existentes
        params: Parâmetros para transform
        
    Returns:
        DataFrame com embeddings de texto adicionados
        Dicionário com parâmetros atualizados
    """
    if params is None:
        params = {}
    
    if 'embeddings' not in params:
        params['embeddings'] = {}
    
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df.columns]
    
    # Usar a média de TF-IDF como embedding simplificado
    for col in text_cols:
        # Verificar se temos colunas TF-IDF para esta coluna de texto
        tfidf_cols = [c for c in df_result.columns if 
                     (c.startswith(f'{col}_tfidf_') or c.startswith(f'{col}_refined_tfidf_'))]
        
        if not tfidf_cols:
            continue
            
        # Criar diferentes dimensões de embedding usando estatísticas
        embeddings = {}
        
        # Dimensão 1: Média dos valores TF-IDF (captura tema geral)
        embeddings['mean'] = df_result[tfidf_cols].mean(axis=1)
        
        # Dimensão 2: Máximo dos valores TF-IDF (captura termo mais importante)
        embeddings['max'] = df_result[tfidf_cols].max(axis=1)
        
        # Dimensão 3: Desvio padrão (captura variabilidade/complexidade do texto)
        embeddings['std'] = df_result[tfidf_cols].std(axis=1)
        
        # Dimensão 4: Número de termos não-zero (captura riqueza léxica)
        embeddings['nonzero'] = (df_result[tfidf_cols] > 0).sum(axis=1) / len(tfidf_cols)
        
        # Adicionar ao dataframe
        for dim_name, values in embeddings.items():
            df_result[f'{col}_embedding_{dim_name}'] = values
        
        if fit:
            params['embeddings'][col] = {
                'method': 'tfidf_stats',
                'dimensions': list(embeddings.keys()),
                'tfidf_cols': tfidf_cols
            }
    
    return df_result, params

def perform_topic_modeling(df, text_cols, n_topics=3, fit=True, params=None):
    """
    Extrai tópicos latentes dos textos usando LDA.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto
        n_topics: Número de tópicos por coluna
        fit: Se True, ajusta modelo LDA, caso contrário usa existente
        params: Parâmetros para transform
        
    Returns:
        DataFrame com probabilidades de tópicos adicionadas
        Dicionário com parâmetros atualizados
    """
    if params is None:
        params = {}
    
    if 'lda' not in params:
        params['lda'] = {}
    
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df.columns]
    print(f"Iniciando processamento LDA para {len(text_cols)} colunas de texto")
    
    for i, col in enumerate(text_cols):
        print(f"Processando LDA para coluna {i+1}/{len(text_cols)}: {col}")
        
        # Verificar se temos colunas TF-IDF para esta coluna de texto
        tfidf_cols = [c for c in df_result.columns if c.startswith(f'{col}_tfidf_') or 
                       c.startswith(f'{col}_refined_tfidf_')]
        
        if not tfidf_cols:
            print(f"  Ignorando: Nenhuma coluna TF-IDF encontrada para {col}")
            continue
            
        print(f"  Encontradas {len(tfidf_cols)} colunas TF-IDF")
        
        # Extrair matriz TF-IDF
        tfidf_matrix = df_result[tfidf_cols].values
        
        if fit:
            # Ajustar LDA
            print(f"  Ajustando modelo LDA com {n_topics} tópicos...")
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42,
                batch_size=128,
                n_jobs=-1,
                verbose=1  # Adiciona verbosidade
            )
            
            try:
                print(f"  Aplicando LDA em matriz de {tfidf_matrix.shape[0]} linhas x {tfidf_matrix.shape[1]} colunas...")
                
                # Mostrar progresso durante o ajuste
                from tqdm import tqdm
                import time
                
                topic_distribution = None
                with tqdm(total=10) as pbar:  # Assumindo max_iter=10
                    lda.max_iter = 1  # Ajustar uma iteração por vez para mostrar progresso
                    for iter_idx in range(10):
                        if iter_idx == 0:
                            topic_distribution = lda.fit_transform(tfidf_matrix)
                        else:
                            lda.n_iter_ = iter_idx
                            topic_distribution = lda.transform(tfidf_matrix)
                        pbar.update(1)
                        time.sleep(0.1)  # Apenas para visualizar a barra
                
                print(f"  LDA concluído! Perplexidade: {lda.perplexity(tfidf_matrix):.2f}")
                
                # Armazenar modelo LDA e colunas usadas
                params['lda'][col] = {
                    'model': lda,
                    'n_topics': n_topics,
                    'tfidf_cols': tfidf_cols,
                    'components': lda.components_.tolist() if hasattr(lda, 'components_') else None
                }
                
                # Visualizar top termos por tópico (para diagnóstico)
                feature_names = [c.split('_tfidf_')[-1] for c in tfidf_cols]
                print("  Top termos por tópico:")
                for topic_idx, topic in enumerate(lda.components_):
                    top_terms = [feature_names[j] for j in topic.argsort()[:-6:-1]]
                    print(f"    Tópico {topic_idx+1}: {', '.join(top_terms)}")
                
                # Adicionar distribuição de tópicos ao dataframe com nomes padronizados
                for topic_idx in range(n_topics):
                    feature_name = standardize_feature_name(f'{col}_topic_{topic_idx+1}')
                    df_result[feature_name] = topic_distribution[:, topic_idx]
                
                print(f"  Adicionadas {n_topics} colunas de tópicos ao DataFrame")
                    
            except Exception as e:
                print(f"Erro ao ajustar LDA para '{col}': {e}")
                
        else:
            # Usar modelo LDA treinado anteriormente
            if col in params['lda'] and 'components' in params['lda'][col] and params['lda'][col]['components']:
                # Reconstruir o modelo LDA
                n_topics = params['lda'][col]['n_topics']
                lda = LatentDirichletAllocation(n_components=n_topics)
                
                # Tentar restaurar os componentes
                try:
                    lda.components_ = np.array(params['lda'][col]['components'])
                    
                    # Garantir que temos as mesmas colunas TF-IDF
                    stored_cols = params['lda'][col]['tfidf_cols']
                    
                    # Alinhar colunas
                    aligned_matrix = np.zeros((len(df_result), len(stored_cols)))
                    for j, col_name in enumerate(stored_cols):
                        if col_name in df_result.columns:
                            aligned_matrix[:, j] = df_result[col_name].values
                    
                    # Transformar usando componentes restaurados
                    topic_distribution = lda.transform(aligned_matrix)
                    
                    # Adicionar distribuição de tópicos ao dataframe com nomes padronizados
                    for topic_idx in range(n_topics):
                        feature_name = standardize_feature_name(f'{col}_topic_{topic_idx+1}')
                        df_result[feature_name] = topic_distribution[:, topic_idx]
                        
                except Exception as e:
                    print(f"Erro ao reconstruir LDA para '{col}': {e}")
                    print("Usando transformação simplificada para tópicos...")
                    
                    # Fallback: computar tópicos usando função linear nos TF-IDF
                    stored_cols = params['lda'][col]['tfidf_cols']
                    components = np.array(params['lda'][col]['components'])
                    
                    # Para cada tópico, calcular a soma ponderada de TF-IDF disponíveis
                    for topic_idx in range(n_topics):
                        topic_weights = components[topic_idx]
                        feature_name = standardize_feature_name(f'{col}_topic_{topic_idx+1}')
                        df_result[feature_name] = 0  # Inicializar
                        
                        # Aplicar pesos para colunas disponíveis
                        for j, col_name in enumerate(stored_cols):
                            if col_name in df_result.columns and j < len(topic_weights):
                                df_result[feature_name] += df_result[col_name] * topic_weights[j]
            else:
                print(f"Aviso: Não há modelo LDA salvo para '{col}' ou dados de componentes")
    
    return df_result, params

def create_salary_features(df, fit=True, params=None):
    """
    Cria features baseadas na relação entre salário atual e desejado.
    
    Args:
        df: DataFrame pandas
        fit: Se True, armazena estatísticas, caso contrário usa existentes
        params: Parâmetros para transform
        
    Returns:
        DataFrame com features salariais adicionadas
        Dicionário com parâmetros atualizados
    """
    if params is None:
        params = {}
    
    if 'salary_features' not in params:
        params['salary_features'] = {}
    
    df_result = df.copy()
    
    # Verificar se temos as colunas de salário
    if 'current_salary_encoded' not in df_result.columns or 'desired_salary_encoded' not in df_result.columns:
        return df_result, params
    
    # Garantir que são numéricas
    df_result['current_salary_encoded'] = pd.to_numeric(df_result['current_salary_encoded'], errors='coerce')
    df_result['desired_salary_encoded'] = pd.to_numeric(df_result['desired_salary_encoded'], errors='coerce')
    
    # Tratar missing values
    df_result['current_salary_encoded'] = df_result['current_salary_encoded'].fillna(0)
    df_result['desired_salary_encoded'] = df_result['desired_salary_encoded'].fillna(0)
    
    # 1. Diferença simples
    salary_diff_col = standardize_feature_name('salary_diff')
    df_result[salary_diff_col] = df_result['desired_salary_encoded'] - df_result['current_salary_encoded']
    
    # 2. Razão (evitando divisão por zero)
    df_result[standardize_feature_name('salary_ratio')] = np.where(
        df_result['current_salary_encoded'] > 0,
        df_result['desired_salary_encoded'] / df_result['current_salary_encoded'],
        df_result['desired_salary_encoded']
    )
    
    # 3. Diferença logarítmica
    # Adicionar 1 antes do log para evitar log(0)
    df_result[standardize_feature_name('salary_log_diff')] = np.log1p(df_result['desired_salary_encoded']) - np.log1p(df_result['current_salary_encoded'])
    
    # 4. Features com transformação sigmoide para limitar outliers
    df_result[standardize_feature_name('salary_growth_potential')] = 2 / (1 + np.exp(-df_result[salary_diff_col])) - 1
    
    if fit:
        # Armazenar estatísticas para transformação
        params['salary_features'] = {
            'mean_diff': df_result[salary_diff_col].mean(),
            'std_diff': df_result[salary_diff_col].std(),
            'mean_ratio': df_result[standardize_feature_name('salary_ratio')].mean(),
            'std_ratio': df_result[standardize_feature_name('salary_ratio')].std()
        }
    
    # 5. Z-score da diferença (normalizado)
    df_result[standardize_feature_name('salary_diff_zscore')] = (df_result[salary_diff_col] - params['salary_features']['mean_diff']) / \
                                     (params['salary_features']['std_diff'] if params['salary_features']['std_diff'] > 0 else 1)
    
    return df_result, params

def create_country_interaction_features(df, fit=True, params=None):
    """
    Cria features de interação entre país e outras variáveis.
    
    Args:
        df: DataFrame pandas
        fit: Se True, armazena estatísticas, caso contrário usa existentes
        params: Parâmetros para transform
        
    Returns:
        DataFrame com features de interação de país adicionadas
        Dicionário com parâmetros atualizados
    """
    if params is None:
        params = {}
    
    if 'country_interactions' not in params:
        params['country_interactions'] = {}
    
    df_result = df.copy()
    
    # Verificar se temos a coluna de país
    if 'country_encoded' not in df_result.columns:
        return df_result, params
    
    # Lista de features para criar interações
    interacting_features = [
        'current_salary_encoded', 'desired_salary_encoded', 
        'age_encoded', 'hour', 'day_of_week'
    ]
    
    # Filtrar apenas as features disponíveis
    interacting_features = [f for f in interacting_features if f in df_result.columns]
    
    for feature in interacting_features:
        # Criar interação multiplicativa com país
        df_result[standardize_feature_name(f'country_x_{feature}')] = df_result['country_encoded'] * df_result[feature]
        
        if fit:
            # Calcular estatísticas de cada país para cada feature
            country_stats = df_result.groupby('country_encoded')[feature].agg(['mean', 'std']).reset_index()
            params['country_interactions'][feature] = country_stats.set_index('country_encoded').to_dict()
        
        # Adicionar desvio em relação à média do país
        # Inicializar coluna com zeros
        df_result[standardize_feature_name(f'{feature}_country_deviation')] = 0
        
        # Para cada país, calcular o desvio
        for country_code in params['country_interactions'][feature]['mean'].keys():
            country_mean = params['country_interactions'][feature]['mean'][country_code]
            mask = df_result['country_encoded'] == country_code
            df_result.loc[mask, standardize_feature_name(f'{feature}_country_deviation')] = df_result.loc[mask, feature] - country_mean
    
    return df_result, params

def create_age_interaction_features(df, fit=True, params=None):
    """
    Cria features de interação entre idade e outras variáveis.
    
    Args:
        df: DataFrame pandas
        fit: Se True, armazena estatísticas, caso contrário usa existentes
        params: Parâmetros para transform
        
    Returns:
        DataFrame com features de interação de idade adicionadas
        Dicionário com parâmetros atualizados
    """
    if params is None:
        params = {}
    
    if 'age_interactions' not in params:
        params['age_interactions'] = {}
    
    df_result = df.copy()
    
    # Verificar se temos a coluna de idade
    if 'age_encoded' not in df_result.columns:
        return df_result, params
    
    # Lista de features para criar interações
    interacting_features = [
        'current_salary_encoded', 'desired_salary_encoded', 
        'hour', 'day_of_week', 'profession_encoded'
    ]
    
    # Filtrar apenas as features disponíveis
    interacting_features = [f for f in interacting_features if f in df_result.columns]
    
    # Criar faixas etárias (mais interpretáveis que valores contínuos)
    df_result['age_group'] = pd.cut(df_result['age_encoded'], 
                                  bins=[0, 1, 2, 3, 4, 5], 
                                  labels=['Unknown', 'Young', 'Early_Career', 'Mid_Career', 'Senior'])
    
    for feature in interacting_features:
        # Criar interação multiplicativa com idade
        df_result[standardize_feature_name(f'age_x_{feature}')] = df_result['age_encoded'] * df_result[feature]
        
        if fit:
            # Calcular estatísticas de cada faixa etária para cada feature
            age_stats = df_result.groupby('age_group')[feature].agg(['mean', 'std']).reset_index()
            params['age_interactions'][feature] = age_stats.set_index('age_group').to_dict()
        
        # Adicionar índice de progressão (feature relativa à idade)
        if feature in ['current_salary_encoded', 'desired_salary_encoded']:
            # Calcular salário por idade (índice de progressão)
            per_age_col = standardize_feature_name(f'{feature}_per_age')
            df_result[per_age_col] = np.where(
                df_result['age_encoded'] > 0,
                df_result[feature] / df_result['age_encoded'],
                df_result[feature]
            )
            
            # Normalizar (tanto no fit quanto transform)
            feature_mean = df_result[per_age_col].mean()
            feature_std = df_result[per_age_col].std()
            
            if fit:
                params['age_interactions'][f'{feature}_per_age'] = {
                    'mean': feature_mean,
                    'std': feature_std
                }
            
            # Calcular z-score
            df_result[standardize_feature_name(f'{feature}_per_age_zscore')] = (df_result[per_age_col] - 
                                                    params['age_interactions'][f'{feature}_per_age']['mean']) / \
                                                   (params['age_interactions'][f'{feature}_per_age']['std'] 
                                                    if params['age_interactions'][f'{feature}_per_age']['std'] > 0 else 1)
    
    return df_result, params

def create_temporal_interaction_features(df, fit=True, params=None):
    """
    Cria features de interação entre variáveis temporais (hora, dia) e outras.
    
    Args:
        df: DataFrame pandas
        fit: Se True, armazena estatísticas, caso contrário usa existentes
        params: Parâmetros para transform
        
    Returns:
        DataFrame com features de interação temporais adicionadas
        Dicionário com parâmetros atualizados
    """
    if params is None:
        params = {}
    
    if 'temporal_interactions' not in params:
        params['temporal_interactions'] = {}
    
    df_result = df.copy()
    
    # Verificar se temos as colunas temporais
    temporal_cols = ['hour', 'day_of_week', 'month']
    if not any(col in df_result.columns for col in temporal_cols):
        return df_result, params
    
    # 1. Criar features categóricas de período do dia
    if 'hour' in df_result.columns:
        # Transformar hora em período do dia
        df_result['day_period'] = pd.cut(
            df_result['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['early_morning', 'morning', 'afternoon', 'evening']
        )
        
        # Transformar em dummies (mais interpretável para o modelo)
        for period in ['early_morning', 'morning', 'afternoon', 'evening']:
            df_result[standardize_feature_name(f'period_{period}')] = (df_result['day_period'] == period).astype(int)
    
    # 2. Criar feature dia de semana vs. fim de semana
    if 'day_of_week' in df_result.columns:
        df_result[standardize_feature_name('is_weekend')] = (df_result['day_of_week'] >= 5).astype(int)
    
    # 3. Criar features combinadas de dia/período
    if 'hour' in df_result.columns and 'day_of_week' in df_result.columns:
        # Combinar dia da semana e período em um único código
        day_names = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        periods = ['morning', 'afternoon', 'evening', 'night']
        
        # Identificar período simplificado
        df_result['period_simple'] = pd.cut(
            df_result['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        # Criar coluna combinada para cada combinação possível
        for day_idx, day in enumerate(day_names):
            for period in periods:
                col_name = standardize_feature_name(f'{day}_{period}')
                day_match = (df_result['day_of_week'] == day_idx)
                period_match = (df_result['period_simple'] == period)
                df_result[col_name] = (day_match & period_match).astype(int)
    
    # 4. Interações com outras features
    interacting_features = [
        'current_salary_encoded', 'desired_salary_encoded', 
        'country_encoded', 'profession_encoded'
    ]
    
    # Filtrar apenas as features disponíveis
    interacting_features = [f for f in interacting_features if f in df_result.columns]
    
    for temp_col in [col for col in temporal_cols if col in df_result.columns]:
        for feature in interacting_features:
            # Criar interação multiplicativa
            df_result[standardize_feature_name(f'{temp_col}_x_{feature}')] = df_result[temp_col] * df_result[feature]
            
            if fit:
                # Calcular estatísticas por período temporal
                if temp_col == 'hour':
                    # Agrupar por período do dia para estatísticas mais robustas
                    temp_stats = df_result.groupby('period_simple')[feature].agg(['mean', 'median']).reset_index()
                    params['temporal_interactions'][f'{temp_col}_{feature}'] = temp_stats.set_index('period_simple').to_dict()
                else:
                    # Usar valor direto para outros temporais
                    temp_stats = df_result.groupby(temp_col)[feature].agg(['mean', 'median']).reset_index()
                    params['temporal_interactions'][f'{temp_col}_{feature}'] = temp_stats.set_index(temp_col).to_dict()
    
    return df_result, params

def advanced_feature_engineering(df, fit=True, params=None):
    """
    Executa engenharia de features avançada, mantendo apenas interações numéricas.
    
    Args:
        df: DataFrame pandas
        fit: Se True, aprende parâmetros, caso contrário usa existentes
        params: Dicionário com parâmetros existentes
        
    Returns:
        DataFrame com features avançadas adicionadas
        Dicionário com parâmetros atualizados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    # Cópia para não modificar o original
    df_result = df.copy()
    
    print("Aplicando engenharia de features avançada (apenas interações numéricas)...")
    
    # 4. Criando features de relação salarial
    print("1. Criando features de relação salarial...")
    df_result, params = create_salary_features(df_result, fit, params)
    
    # 5. Criando interações com país
    print("2. Criando interações com país...")
    df_result, params = create_country_interaction_features(df_result, fit, params)
    
    # 6. Criando interações com idade
    print("3. Criando interações com idade...")
    df_result, params = create_age_interaction_features(df_result, fit, params)
    
    # 7. Criando interações temporais
    print("4. Criando interações temporais...")
    df_result, params = create_temporal_interaction_features(df_result, fit, params)
    
    # Contar número de features adicionadas
    num_added_features = df_result.shape[1] - df.shape[1]
    print(f"Engenharia de features avançada concluída. Adicionadas {num_added_features} novas features.")
    
    return df_result, params