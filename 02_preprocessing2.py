#!/usr/bin/env python
"""
Script para aplicar a pipeline de pré-processamento nos conjuntos de treino, validação e teste,
com versões corrigidas das funções de engenharia de features avançadas que estavam causando erros.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
import argparse

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

# Importar módulos de pré-processamento
from src.preprocessing.email_processing import normalize_emails_in_dataframe
from src.preprocessing.data_cleaning import (
    consolidate_quality_columns,
    handle_missing_values,
    handle_outliers,
    normalize_values,
    convert_data_types
)
from src.preprocessing.feature_engineering import feature_engineering
from src.preprocessing.text_processing import text_feature_engineering

# Importar apenas a função principal de advanced_feature_engineering
# (vamos substituir as versões problemáticas)
from src.preprocessing.advanced_feature_engineering import (
    create_salary_features,
    create_country_interaction_features,
    create_age_interaction_features,
    create_temporal_interaction_features,
    perform_topic_modeling
)

# Definindo as versões corrigidas das funções problemáticas

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
    
    print(f"    Refinando pesos TF-IDF para {len(original_text_cols)} colunas de texto original")
    
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
            print(f"      Ignorando coluna {col}: poucos dados não vazios")
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
                    print(f"      Ignorando coluna {col}: nenhum termo extraído no TF-IDF")
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
                    df_result[f'{col}_refined_tfidf_{term}'] = 0.0
                    
                    # Preencher valores para textos não vazios
                    df_result.loc[non_empty, f'{col}_refined_tfidf_{term}'] = tfidf_array[:, i]
                    
                    # Aumentar o peso se for um termo importante
                    if any(imp_term in term for imp_term in important_terms):
                        df_result[f'{col}_refined_tfidf_{term}'] *= 2.0  # Dobrar o peso
                
                print(f"      Criadas {len(feature_names)} features TF-IDF para {col}")
                
            except Exception as e:
                print(f"      Erro ao processar TF-IDF para '{col}': {e}")
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
                    
                    print(f"      Aplicadas {len(feature_names)} features TF-IDF para {col}")
                    
                except Exception as e:
                    print(f"      Erro ao transformar TF-IDF para '{col}': {e}")
    
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
    
    print(f"    Criando embeddings para {len(text_cols)} colunas de texto")
    
    # Usar a média de TF-IDF como embedding simplificado
    for col in text_cols:
        # Verificar colunas TF-IDF
        tfidf_cols = [c for c in df_result.columns if 
                      (c.startswith(f'{col}_tfidf_') or c.startswith(f'{col}_refined_tfidf_'))]
        
        if not tfidf_cols:
            print(f"      Ignorando {col}: nenhuma coluna TF-IDF encontrada")
            continue
        
        # Garantir que apenas colunas numéricas sejam usadas
        numeric_tfidf_cols = []
        for tfidf_col in tfidf_cols:
            # Verificar se a coluna é numérica
            if pd.api.types.is_numeric_dtype(df_result[tfidf_col]):
                numeric_tfidf_cols.append(tfidf_col)
            else:
                print(f"      Ignorando coluna não numérica: {tfidf_col}")
        
        if not numeric_tfidf_cols:
            print(f"      Ignorando {col}: nenhuma coluna TF-IDF numérica")
            continue
        
        print(f"      Criando embeddings para {col} usando {len(numeric_tfidf_cols)} features TF-IDF")
        
        try:
            # Dimensão 1: Média dos valores TF-IDF
            df_result[f'{col}_embedding_mean'] = df_result[numeric_tfidf_cols].mean(axis=1)
            
            # Dimensão 2: Máximo dos valores TF-IDF
            df_result[f'{col}_embedding_max'] = df_result[numeric_tfidf_cols].max(axis=1)
            
            # Dimensão 3: Desvio padrão
            df_result[f'{col}_embedding_std'] = df_result[numeric_tfidf_cols].std(axis=1)
            
            # Dimensão 4: Número de termos não-zero
            df_result[f'{col}_embedding_nonzero'] = (df_result[numeric_tfidf_cols] > 0).sum(axis=1) / len(numeric_tfidf_cols)
            
            if fit:
                params['embeddings'][col] = {
                    'method': 'tfidf_stats',
                    'dimensions': ['mean', 'max', 'std', 'nonzero'],
                    'tfidf_cols': numeric_tfidf_cols
                }
            
            print(f"      Sucesso: 4 dimensões de embedding criadas para {col}")
                
        except Exception as e:
            print(f"      Erro ao criar embeddings para {col}: {e}")
            # Limpar qualquer coluna parcial que possa ter sido criada
            for dim in ['mean', 'max', 'std', 'nonzero']:
                if f'{col}_embedding_{dim}' in df_result.columns:
                    df_result = df_result.drop(columns=[f'{col}_embedding_{dim}'])
    
    return df_result, params

def perform_topic_modeling_fixed(df, text_cols, n_topics=3, fit=True, params=None):
    """
    Versão melhorada de perform_topic_modeling que lida melhor com
    diferentes tipos de colunas e contém tratamento de erros mais robusto.
    """
    if params is None:
        params = {}
    
    if 'lda' not in params:
        params['lda'] = {}
    
    df_result = df.copy()
    
    # Filtrar apenas colunas de texto originais
    original_text_cols = []
    for col in text_cols:
        if col in df.columns and not any(suffix in col for suffix in 
                                        ['_length', '_word_count', '_sentiment', '_tfidf_', '_has_', '_motiv_']):
            original_text_cols.append(col)
    
    print(f"    Realizando modelagem de tópicos para {len(original_text_cols)} colunas de texto")
    
    for i, col in enumerate(original_text_cols):
        print(f"      Processando LDA para coluna {i+1}/{len(original_text_cols)}: {col}")
        
        # Verificar se a coluna existe e tem dados
        if col not in df_result.columns:
            print(f"        Coluna {col} não encontrada, pulando")
            continue
            
        # Verificar se temos a coluna limpa ou original
        if f'{col}_original' in df_result.columns:
            text_col = f'{col}_original'
        else:
            text_col = col
            
        # Filtrar textos não vazios
        non_empty = df_result[text_col].fillna('').str.len() > 20
        if non_empty.sum() < 50:  # Requer pelo menos 50 textos não vazios
            print(f"        Ignorando coluna {col}: poucos textos substantivos")
            continue
            
        # Verificar se temos colunas TF-IDF para esta coluna
        tfidf_cols = [c for c in df_result.columns if c.startswith(f'{col}_tfidf_') or 
                       c.startswith(f'{col}_refined_tfidf_')]
        
        if not tfidf_cols:
            print(f"        Ignorando: Nenhuma coluna TF-IDF encontrada para {col}")
            continue
            
        # Garantir que todas as colunas são numéricas
        numeric_tfidf_cols = []
        for tfidf_col in tfidf_cols:
            if pd.api.types.is_numeric_dtype(df_result[tfidf_col]):
                numeric_tfidf_cols.append(tfidf_col)
            
        if len(numeric_tfidf_cols) < 5:  # Requer pelo menos 5 features TF-IDF
            print(f"        Ignorando: Poucas colunas TF-IDF numéricas para {col}")
            continue
            
        print(f"        Usando {len(numeric_tfidf_cols)} colunas TF-IDF numéricas")
        
        # Extrair matriz TF-IDF
        tfidf_matrix = df_result[numeric_tfidf_cols].values
        
        if fit:
            # Ajustar LDA
            print(f"        Ajustando modelo LDA com {n_topics} tópicos...")
            
            try:
                from sklearn.decomposition import LatentDirichletAllocation
                
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    max_iter=10,
                    learning_method='online',
                    random_state=42,
                    batch_size=128,
                    n_jobs=-1
                )
                
                topic_distribution = lda.fit_transform(tfidf_matrix)
                
                # Armazenar modelo LDA e colunas usadas
                params['lda'][col] = {
                    'model': lda,
                    'n_topics': n_topics,
                    'tfidf_cols': numeric_tfidf_cols,
                    'components': lda.components_.tolist() if hasattr(lda, 'components_') else None
                }
                
                # Visualizar top termos por tópico
                feature_names = [c.split('_tfidf_')[-1] if '_tfidf_' in c else c.split('_refined_tfidf_')[-1] 
                                for c in numeric_tfidf_cols]
                print("        Top termos por tópico:")
                for topic_idx, topic in enumerate(lda.components_):
                    top_terms = [feature_names[i] for i in topic.argsort()[:-6:-1]]
                    print(f"          Tópico {topic_idx+1}: {', '.join(top_terms)}")
                
                # Adicionar distribuição de tópicos ao dataframe
                for i in range(n_topics):
                    df_result[f'{col}_topic_{i+1}'] = topic_distribution[:, i]
                
                print(f"        Adicionadas {n_topics} colunas de tópicos ao DataFrame")
                    
            except Exception as e:
                print(f"        Erro ao ajustar LDA para '{col}': {e}")
                
        else:
            # Usar modelo LDA treinado anteriormente
            if col in params['lda'] and 'components' in params['lda'][col] and params['lda'][col]['components']:
                try:
                    # Reconstruir o modelo LDA
                    from sklearn.decomposition import LatentDirichletAllocation
                    
                    n_topics = params['lda'][col]['n_topics']
                    lda = LatentDirichletAllocation(n_components=n_topics)
                    
                    # Restaurar os componentes
                    lda.components_ = np.array(params['lda'][col]['components'])
                    
                    # Garantir que temos as mesmas colunas TF-IDF
                    stored_cols = params['lda'][col]['tfidf_cols']
                    
                    # Alinhar colunas
                    aligned_matrix = np.zeros((len(df_result), len(stored_cols)))
                    for i, col_name in enumerate(stored_cols):
                        if col_name in df_result.columns:
                            aligned_matrix[:, i] = df_result[col_name].values
                    
                    # Transformar usando componentes restaurados
                    topic_distribution = lda.transform(aligned_matrix)
                    
                    # Adicionar distribuição de tópicos ao dataframe
                    for i in range(n_topics):
                        df_result[f'{col}_topic_{i+1}'] = topic_distribution[:, i]
                        
                    print(f"        Sucesso: {n_topics} tópicos aplicados")
                        
                except Exception as e:
                    print(f"        Erro ao reconstruir LDA para '{col}': {e}")
                    print("        Usando transformação simplificada para tópicos...")
                    
                    # Fallback: computar tópicos manualmente
                    stored_cols = params['lda'][col]['tfidf_cols']
                    components = np.array(params['lda'][col]['components'])
                    
                    try:
                        # Para cada tópico, calcular a soma ponderada de TF-IDF disponíveis
                        for i in range(n_topics):
                            topic_weights = components[i]
                            df_result[f'{col}_topic_{i+1}'] = 0  # Inicializar
                            
                            # Aplicar pesos para colunas disponíveis
                            weight_count = 0
                            for j, col_name in enumerate(stored_cols):
                                if col_name in df_result.columns and j < len(topic_weights):
                                    df_result[f'{col}_topic_{i+1}'] += df_result[col_name] * topic_weights[j]
                                    weight_count += 1
                            
                            # Normalizar pela quantidade de pesos disponíveis
                            if weight_count > 0:
                                df_result[f'{col}_topic_{i+1}'] /= weight_count
                    except Exception as e:
                        print(f"        Erro no fallback para '{col}': {e}")
            else:
                print(f"        Aviso: Não há modelo LDA salvo para '{col}' ou dados de componentes")
    
    return df_result, params

def advanced_feature_engineering_fixed(df, fit=True, params=None):
    """
    Versão corrigida da função principal de engenharia de features avançada,
    que chama as funções corrigidas e lida adequadamente com erros.
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    # Cópia para não modificar o original
    df_result = df.copy()
    
    print("Aplicando engenharia de features avançada com tratamento de erros aprimorado...")
    
    # 1. Extrair colunas de texto
    text_cols = [
        col for col in df_result.columns 
        if df_result[col].dtype == 'object' and any(term in col for term in [
            'mensaje', 'inglés', 'vida', 'oportunidades', 'esperas', 'aprender', 
            'Semana', 'Inmersión', 'Déjame', 'fluidez'
        ])
    ]
    
    # 2. Refinar pesos TF-IDF
    try:
        print("1. Refinando pesos TF-IDF...")
        df_result, params = refine_tfidf_weights_fixed(df_result, text_cols, fit, params)
    except Exception as e:
        print(f"Erro ao refinar pesos TF-IDF: {e}")
    
    # 3. Criar embeddings simples
    try:
        print("2. Criando embeddings de texto simples...")
        df_result, params = create_text_embeddings_simple_fixed(df_result, text_cols, fit, params)
    except Exception as e:
        print(f"Erro ao criar embeddings de texto: {e}")
    
    # 4. Modelagem de tópicos
    try:
        print("3. Realizando modelagem de tópicos...")
        df_result, params = perform_topic_modeling_fixed(df_result, text_cols, n_topics=5, fit=fit, params=params)
    except Exception as e:
        print(f"Erro ao realizar modelagem de tópicos: {e}")
    
    # 5. Criando features de relação salarial
    try:
        print("4. Criando features de relação salarial...")
        df_result, params = create_salary_features(df_result, fit, params)
    except Exception as e:
        print(f"Erro ao criar features de relação salarial: {e}")
    
    # 6. Criando interações com país
    try:
        print("5. Criando interações com país...")
        df_result, params = create_country_interaction_features(df_result, fit, params)
    except Exception as e:
        print(f"Erro ao criar interações com país: {e}")
    
    # 7. Criando interações com idade
    try:
        print("6. Criando interações com idade...")
        df_result, params = create_age_interaction_features(df_result, fit, params)
    except Exception as e:
        print(f"Erro ao criar interações com idade: {e}")
    
    # 8. Criando interações temporais
    try:
        print("7. Criando interações temporais...")
        df_result, params = create_temporal_interaction_features(df_result, fit, params)
    except Exception as e:
        print(f"Erro ao criar interações temporais: {e}")
    
    # Contar número de features adicionadas
    num_added_features = df_result.shape[1] - df.shape[1]
    print(f"Engenharia de features avançada concluída. Adicionadas {num_added_features} novas features.")
    
    return df_result, params

def apply_preprocessing_pipeline(df, params=None, fit=False, preserve_text=True):
    """
    Aplica a pipeline completa de pré-processamento.
    
    Args:
        df: DataFrame a ser processado
        params: Parâmetros para transformações (None para começar do zero)
        fit: Se True, ajusta as transformações, se False, apenas aplica
        preserve_text: Se True, preserva as colunas de texto originais
        
    Returns:
        DataFrame processado e parâmetros atualizados
    """
    # Inicializar parâmetros se não fornecidos
    if params is None:
        params = {}
    
    print(f"Iniciando pipeline de pré-processamento para DataFrame: {df.shape}")
    
    # 1. Normalizar emails
    print("1. Normalizando emails...")
    df = normalize_emails_in_dataframe(df, email_col='email')
    
    # 2. Consolidar colunas de qualidade
    print("2. Consolidando colunas de qualidade...")
    quality_params = params.get('quality_columns', {})
    df, quality_params = consolidate_quality_columns(df, fit=fit, params=quality_params)
    
    # 3. Tratamento de valores ausentes
    print("3. Tratando valores ausentes...")
    missing_params = params.get('missing_values', {})
    df, missing_params = handle_missing_values(df, fit=fit, params=missing_params)
    
    # 4. Tratamento de outliers
    print("4. Tratando outliers...")
    outlier_params = params.get('outliers', {})
    df, outlier_params = handle_outliers(df, fit=fit, params=outlier_params)
    
    # 5. Normalização de valores
    print("5. Normalizando valores numéricos...")
    norm_params = params.get('normalization', {})
    df, norm_params = normalize_values(df, fit=fit, params=norm_params)
    
    # 6. Converter tipos de dados
    print("6. Convertendo tipos de dados...")
    df, _ = convert_data_types(df, fit=fit)
    
    # Identificar colunas de texto antes do processamento
    text_cols = [
        col for col in df.columns 
        if df[col].dtype == 'object' and any(term in col for term in [
            'mensaje', 'inglés', 'vida', 'oportunidades', 'esperas', 'aprender', 
            'Semana', 'Inmersión', 'Déjame', 'fluidez'
        ])
    ]
    print(f"Colunas de texto identificadas ({len(text_cols)}): {text_cols[:3]}...")
    
    # Criar cópia das colunas de texto originais (com sufixo _original)
    if text_cols and preserve_text:
        print("Preservando colunas de texto originais...")
        for col in text_cols:
            df[f"{col}_original"] = df[col].copy()
    
    # 7. Feature engineering não-textual
    print("7. Aplicando feature engineering não-textual...")
    feature_params = params.get('feature_engineering', {})
    df, feature_params = feature_engineering(df, fit=fit, params=feature_params)
    
    # 8. Processamento de texto
    print("8. Processando features textuais...")
    text_params = params.get('text_processing', {})
    df, text_params = text_feature_engineering(df, fit=fit, params=text_params)
    
    # 9. Feature engineering avançada (usando a versão corrigida)
    print("9. Aplicando feature engineering avançada (versão corrigida)...")
    advanced_params = params.get('advanced_features', {})
    df, advanced_params = advanced_feature_engineering_fixed(df, fit=fit, params=advanced_params)
    
    # 10. Compilar parâmetros atualizados
    updated_params = {
        'quality_columns': quality_params,
        'missing_values': missing_params,
        'outliers': outlier_params,
        'normalization': norm_params,
        'feature_engineering': feature_params,
        'text_processing': text_params,
        'advanced_features': advanced_params
    }
    
    print(f"Pipeline concluída! Dimensões finais: {df.shape}")
    return df, updated_params

def ensure_column_consistency(train_df, test_df):
    """
    Garante que o DataFrame de teste tenha as mesmas colunas que o de treinamento.
    
    Args:
        train_df: DataFrame de treinamento
        test_df: DataFrame de teste para alinhar
        
    Returns:
        DataFrame de teste com colunas alinhadas
    """
    print("Alinhando colunas entre conjuntos de dados...")
    
    # Colunas presentes no treino, mas ausentes no teste
    missing_cols = set(train_df.columns) - set(test_df.columns)
    
    # Adicionar colunas faltantes com valores padrão
    for col in missing_cols:
        if col in train_df.select_dtypes(include=['number']).columns:
            test_df[col] = 0
        else:
            test_df[col] = None
        print(f"  Adicionada coluna ausente: {col}")
    
    # Remover colunas extras no teste não presentes no treino
    extra_cols = set(test_df.columns) - set(train_df.columns)
    if extra_cols:
        test_df = test_df.drop(columns=list(extra_cols))
        print(f"  Removidas colunas extras: {', '.join(list(extra_cols)[:5])}" + 
              (f" e mais {len(extra_cols)-5} outras" if len(extra_cols) > 5 else ""))
    
    # Garantir a mesma ordem de colunas
    test_df = test_df[train_df.columns]
    
    print(f"Alinhamento concluído: {len(missing_cols)} colunas adicionadas, {len(extra_cols)} removidas")
    return test_df

def process_datasets(input_dir, output_dir, params_dir=None, preserve_text=True):
    """
    Função principal que processa todos os conjuntos na ordem correta.
    
    Args:
        input_dir: Diretório contendo os arquivos de entrada
        output_dir: Diretório para salvar os arquivos processados
        params_dir: Diretório para salvar os parâmetros (opcional)
        preserve_text: Se True, preserva as colunas de texto originais
        
    Returns:
        Dicionário com os DataFrames processados e parâmetros
    """
    # 1. Definir caminhos dos datasets
    train_path = os.path.join(input_dir, "train.csv")
    cv_path = os.path.join(input_dir, "validation.csv")
    test_path = os.path.join(input_dir, "test.csv")
    
    # Garantir que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar se os arquivos existem
    print(f"Verificando existência dos arquivos de entrada:")
    print(f"  Train path: {train_path} - Existe: {os.path.exists(train_path)}")
    print(f"  CV path: {cv_path} - Existe: {os.path.exists(cv_path)}")
    print(f"  Test path: {test_path} - Existe: {os.path.exists(test_path)}")
    
    if not all([os.path.exists(train_path), os.path.exists(cv_path), os.path.exists(test_path)]):
        print("ERRO: Um ou mais arquivos de entrada não foram encontrados!")
        print("Por favor, verifique o caminho dos arquivos.")
        return None
    
    # 2. Carregar os datasets
    print(f"Carregando datasets de {input_dir}...")
    train_df = pd.read_csv(train_path)
    cv_df = pd.read_csv(cv_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Datasets carregados: treino {train_df.shape}, validação {cv_df.shape}, teste {test_df.shape}")
    
    # 3. Processar o conjunto de treinamento com fit=True para aprender parâmetros
    print("\n--- Processando conjunto de treinamento ---")
    train_processed, params = apply_preprocessing_pipeline(train_df, fit=True, preserve_text=preserve_text)
    
    # 4. Salvar parâmetros aprendidos
    if params_dir:
        os.makedirs(params_dir, exist_ok=True)
        params_path = os.path.join(params_dir, "all_preprocessing_params_fixed.joblib")
        joblib.dump(params, params_path)
        print(f"Parâmetros de pré-processamento salvos em {params_path}")
    
    # 5. Salvar conjunto de treino processado
    train_processed.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    print(f"Dataset de treino processado e salvo em {os.path.join(output_dir, 'train.csv')}")
    
    # 6. Processar o conjunto de validação com fit=False para aplicar parâmetros aprendidos
    print("\n--- Processando conjunto de validação ---")
    cv_processed, _ = apply_preprocessing_pipeline(cv_df, params=params, fit=False, preserve_text=preserve_text)
    
    # 7. Garantir consistência de colunas com o treino
    cv_processed = ensure_column_consistency(train_processed, cv_processed)
    
    # 8. Salvar conjunto de validação processado
    cv_processed.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    print(f"Dataset de validação processado e salvo em {os.path.join(output_dir, 'validation.csv')}")
    
    # 9. Processar o conjunto de teste com fit=False para aplicar parâmetros aprendidos
    print("\n--- Processando conjunto de teste ---")
    test_processed, _ = apply_preprocessing_pipeline(test_df, params=params, fit=False, preserve_text=preserve_text)
    
    # 10. Garantir consistência de colunas com o treino
    test_processed = ensure_column_consistency(train_processed, test_processed)
    
    # 11. Salvar conjunto de teste processado
    test_processed.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Dataset de teste processado e salvo em {os.path.join(output_dir, 'test.csv')}")
    
    print("\nPré-processamento dos conjuntos concluído com sucesso!")
    print(f"Os datasets processados foram salvos em {output_dir}/")
    
    return {
        'train': train_processed,
        'cv': cv_processed,
        'test': test_processed,
        'params': params
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplicar pipeline de pré-processamento nos conjuntos de dados com correções.")
    parser.add_argument("--input-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/01_split"), 
                        help="Diretório contendo os arquivos de entrada (train.csv, validation.csv, test.csv)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/02_fixed_processed"), 
                        help="Diretório para salvar os arquivos processados")
    parser.add_argument("--params-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/src/preprocessing/preprocessing_params"), 
                        help="Diretório para salvar os parâmetros aprendidos")
    parser.add_argument("--preserve-text", action="store_true", default=True,
                        help="Preservar as colunas de texto originais (default: True)")
    
    args = parser.parse_args()
    
    # Chamada da função principal
    try:
        results = process_datasets(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            params_dir=args.params_dir,
            preserve_text=args.preserve_text
        )
        
        if results is None:
            sys.exit(1)  # Sair com código de erro
    except Exception as e:
        import traceback
        print(f"ERRO FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)