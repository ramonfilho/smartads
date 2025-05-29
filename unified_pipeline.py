#!/usr/bin/env python
"""
Script unificado para coleta, integração, pré-processamento e feature engineering.
Combina as funcionalidades dos scripts:
- 01_data_collection_and_integration.py
- preprocessing_02.py  
- feature_engineering_03.py

Este script:
1. Coleta e integra dados das fontes
2. Divide em train/val/test (para evitar data leakage)
3. Aplica pré-processamento (fit no train, transform nos outros)
4. Aplica feature engineering profissional de NLP
5. Salva parâmetros de todas as transformações
6. Retorna DataFrames processados em memória
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
import json
import time
import gc
import re
import nltk
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

# Adicionar o diretório raiz do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# IMPORTS DA PARTE 1 - COLETA E INTEGRAÇÃO
# ============================================================================

from src.utils.local_storage import (
    connect_to_gcs, list_files_by_extension, categorize_files,
    load_csv_or_excel, load_csv_with_auto_delimiter, extract_launch_id
)
from src.preprocessing.email_processing import normalize_emails_in_dataframe, normalize_email
from src.preprocessing.column_normalization import normalize_survey_columns, validate_normalized_columns

# ============================================================================
# IMPORTS DA PARTE 2 - PRÉ-PROCESSAMENTO
# ============================================================================

from src.preprocessing.data_cleaning import (
    consolidate_quality_columns,
    handle_missing_values,
    handle_outliers,
    normalize_values,
    convert_data_types
)
from src.preprocessing.feature_engineering import feature_engineering
from src.preprocessing.text_processing import text_feature_engineering
from src.preprocessing.advanced_feature_engineering import (
    refine_tfidf_weights,
    create_text_embeddings_simple,
    perform_topic_modeling,
    create_salary_features,
    create_country_interaction_features,
    create_age_interaction_features,
    create_temporal_interaction_features,
    advanced_feature_engineering
)

# ============================================================================
# IMPORTS DA PARTE 3 - FEATURE ENGINEERING PROFISSIONAL
# ============================================================================

from src.preprocessing.professional_motivation_features import (
    create_professional_motivation_score,
    analyze_aspiration_sentiment,
    detect_commitment_expressions,
    create_career_term_detector,
    enhance_tfidf_for_career_terms
)

# ============================================================================
# CONFIGURAÇÃO: COLUNAS PERMITIDAS NA INFERÊNCIA
# ============================================================================

INFERENCE_COLUMNS = [
    # Dados de UTM
    'DATA',
    'E-MAIL',
    'UTM_CAMPAING',
    'UTM_SOURCE',
    'UTM_MEDIUM',
    'UTM_CONTENT',
    'UTM_TERM',
    'GCLID',
    
    # Dados da pesquisa
    'Marca temporal',
    '¿Cómo te llamas?',
    '¿Cuál es tu género?',
    '¿Cuál es tu edad?',
    '¿Cual es tu país?',
    '¿Cuál es tu e-mail?',
    '¿Cual es tu telefono?',
    '¿Cuál es tu instagram?',
    '¿Hace quánto tiempo me conoces?',
    '¿Cuál es tu disponibilidad de tiempo para estudiar inglés?',
    'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
    '¿Cuál es tu profesión?',
    '¿Cuál es tu sueldo anual? (en dólares)',
    '¿Cuánto te gustaría ganar al año?',
    '¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?',
    '¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?',
    '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
    'Déjame un mensaje',
    
    # Features novas do L22
    '¿Cuáles son tus principales razones para aprender inglés?',
    '¿Has comprado algún curso para aprender inglés antes?',
    
    # Qualidade
    'Qualidade (Nome)',
    'Qualidade (Número)',
    
    # Variável alvo (adicionada durante o treino)
    'target'
]

# ============================================================================
# CONFIGURAÇÃO NLTK
# ============================================================================

def setup_nltk_resources():
    """Configura recursos NLTK necessários sem mensagens repetidas."""
    nltk_logger = logging.getLogger('nltk')
    nltk_logger.setLevel(logging.CRITICAL)
    
    resources = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass

# Configurar NLTK uma única vez
setup_nltk_resources()

# ============================================================================
# FUNÇÕES DA PARTE 1 - COLETA E INTEGRAÇÃO
# ============================================================================

def find_email_column(df):
    """Encontra a coluna que contém emails em um DataFrame."""
    email_patterns = ['email', 'e-mail', 'correo', '@', 'mail']
    for col in df.columns:
        if any(pattern in col.lower() for pattern in email_patterns):
            return col
    return None

def normalize_emails_preserving_originals(df, survey_email_col='¿Cuál es tu e-mail?', utm_email_col='E-MAIL'):
    """Normaliza emails criando email_norm para matching, mas preserva as colunas originais."""
    df_result = df.copy()
    
    # Criar lista de possíveis fontes de email
    email_sources = []
    
    # Verificar coluna de email das pesquisas
    if survey_email_col in df_result.columns:
        email_sources.append(survey_email_col)
    
    # Verificar coluna de email das UTMs
    if utm_email_col in df_result.columns:
        email_sources.append(utm_email_col)
    
    # Verificar coluna genérica 'email'
    if 'email' in df_result.columns:
        email_sources.append('email')
    
    # Inicializar email_norm com tipo string para evitar FutureWarning
    df_result['email_norm'] = pd.Series(dtype='object')
    
    for source_col in email_sources:
        # Preencher email_norm com valores não-nulos da fonte
        mask = df_result['email_norm'].isna() & df_result[source_col].notna()
        if mask.any():
            df_result.loc[mask, 'email_norm'] = df_result.loc[mask, source_col].apply(normalize_email)
    
    return df_result

def preserve_email_data(df):
    """Preserva dados de email movendo-os para as colunas corretas antes da filtragem."""
    df_result = df.copy()
    
    # Se existe coluna 'email' genérica, distribuir para as colunas corretas
    if 'email' in df_result.columns:
        # Para registros de pesquisa (não têm E-MAIL preenchido)
        if '¿Cuál es tu e-mail?' not in df_result.columns:
            df_result['¿Cuál es tu e-mail?'] = np.nan
        
        # Copiar emails para a coluna de pesquisa onde apropriado
        mask_survey = df_result['email'].notna() & df_result['¿Cuál es tu e-mail?'].isna()
        if 'E-MAIL' in df_result.columns:
            # Se tem E-MAIL, só copiar onde E-MAIL está vazio (indica que é da pesquisa)
            mask_survey = mask_survey & df_result['E-MAIL'].isna()
        
        df_result.loc[mask_survey, '¿Cuál es tu e-mail?'] = df_result.loc[mask_survey, 'email']
        
        # Para registros de UTM (têm E-MAIL)
        if 'E-MAIL' not in df_result.columns:
            df_result['E-MAIL'] = np.nan
            
        mask_utm = df_result['email'].notna() & df_result['E-MAIL'].isna()
        if '¿Cuál es tu e-mail?' in df_result.columns:
            # Se tem coluna de pesquisa, só copiar onde está vazia
            mask_utm = mask_utm & df_result['¿Cuál es tu e-mail?'].isna()
            
        df_result.loc[mask_utm, 'E-MAIL'] = df_result.loc[mask_utm, 'email']
    
    return df_result

def filter_to_inference_columns_preserving_data(df, add_missing=True):
    """Filtra DataFrame para conter apenas colunas disponíveis na inferência."""
    # Primeiro, preservar dados de email se existirem em colunas genéricas
    df_with_preserved_emails = preserve_email_data(df)
    
    # Colunas que existem no DataFrame e estão na lista de inferência
    existing_inference_cols = [col for col in INFERENCE_COLUMNS if col in df_with_preserved_emails.columns]
    
    # Filtrar DataFrame
    filtered_df = df_with_preserved_emails[existing_inference_cols].copy()
    
    if add_missing:
        # Adicionar colunas faltantes com NaN (exceto target)
        missing_cols = [col for col in INFERENCE_COLUMNS 
                       if col not in filtered_df.columns and col != 'target']
        
        for col in missing_cols:
            filtered_df[col] = np.nan
        
        # Reordenar colunas conforme INFERENCE_COLUMNS (exceto target se não existir)
        final_cols = [col for col in INFERENCE_COLUMNS 
                     if col in filtered_df.columns]
        filtered_df = filtered_df[final_cols]
    
    return filtered_df

def process_survey_file(bucket, file_path):
    """Processa um arquivo de pesquisa preservando a coluna de email original."""
    df = load_csv_or_excel(bucket, file_path)
    if df is None:
        return None
        
    # Identificar o lançamento deste arquivo
    launch_id = extract_launch_id(file_path)
    
    # Normalizar colunas
    df = normalize_survey_columns(df, launch_id)
    validate_normalized_columns(df, launch_id)
    
    # Adicionar identificador de lançamento se disponível
    if launch_id:
        df['lançamento'] = launch_id
    
    return df

def process_buyer_file(bucket, file_path):
    """Processa um arquivo de compradores."""
    df = load_csv_or_excel(bucket, file_path)
    if df is None:
        return None
        
    # Identificar o lançamento deste arquivo
    launch_id = extract_launch_id(file_path)
    
    # Para buyers, ainda precisamos da coluna 'email' para normalização
    email_col = find_email_column(df)
    if email_col and email_col != 'email':
        df = df.rename(columns={email_col: 'email'})
    elif not email_col:
        print(f"  - Warning: No email column found in {file_path}. Available columns: {', '.join(df.columns[:5])}...")
    
    # Adicionar identificador de lançamento se disponível
    if launch_id:
        df['lançamento'] = launch_id
    
    return df

def process_utm_file(bucket, file_path):
    """Processa um arquivo de UTM preservando a coluna E-MAIL."""
    df = load_csv_with_auto_delimiter(bucket, file_path)
    if df is None:
        return None
        
    # Identificar o lançamento deste arquivo
    launch_id = extract_launch_id(file_path)
    
    # Adicionar identificador de lançamento se disponível
    if launch_id:
        df['lançamento'] = launch_id
    
    return df

def load_survey_files(bucket, survey_files):
    """Carrega todos os arquivos de pesquisa preservando emails originais."""
    survey_dfs = []
    launch_data = {}
    
    print("\nLoading survey files...")
    for file_path in survey_files:
        try:
            df = process_survey_file(bucket, file_path)
            if df is not None:
                survey_dfs.append(df)
                
                launch_id = extract_launch_id(file_path)
                if launch_id:
                    if launch_id not in launch_data:
                        launch_data[launch_id] = {}
                    launch_data[launch_id]['survey'] = df
                    
                print(f"  - Loaded: {file_path} ({launch_id if launch_id else ''}), {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
    
    return survey_dfs, launch_data

def load_buyer_files(bucket, buyer_files):
    """Carrega todos os arquivos de compradores."""
    buyer_dfs = []
    launch_data = {}
    
    print("\nLoading buyer files...")
    for file_path in buyer_files:
        try:
            df = process_buyer_file(bucket, file_path)
            if df is not None:
                buyer_dfs.append(df)
                
                launch_id = extract_launch_id(file_path)
                if launch_id:
                    if launch_id not in launch_data:
                        launch_data[launch_id] = {}
                    launch_data[launch_id]['buyer'] = df
                    
                print(f"  - Loaded: {file_path} ({launch_id if launch_id else ''}), {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
    
    return buyer_dfs, launch_data

def load_utm_files(bucket, utm_files):
    """Carrega todos os arquivos de UTM preservando E-MAIL original."""
    utm_dfs = []
    launch_data = {}
    
    print("\nLoading UTM files...")
    for file_path in utm_files:
        try:
            df = process_utm_file(bucket, file_path)
            if df is not None:
                utm_dfs.append(df)
                
                launch_id = extract_launch_id(file_path)
                if launch_id:
                    if launch_id not in launch_data:
                        launch_data[launch_id] = {}
                    launch_data[launch_id]['utm'] = df
                    
                print(f"  - Loaded: {file_path} ({launch_id if launch_id else ''}), {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
    
    return utm_dfs, launch_data

def match_surveys_with_buyers_improved(surveys, buyers, utms=None):
    """Realiza correspondência entre pesquisas e compradores usando email_norm."""
    print("\nMatching surveys with buyers...")
    start_time = time.time()
    
    # Verificar se podemos prosseguir com a correspondência
    if surveys.empty or buyers.empty or 'email_norm' not in buyers.columns or 'email_norm' not in surveys.columns:
        print("Warning: Cannot perform matching. Missing email_norm column.")
        return pd.DataFrame(columns=['buyer_id', 'survey_id', 'match_type', 'score'])
    
    # Preparar estruturas de dados para matching eficiente
    survey_emails_dict = dict(zip(surveys['email_norm'], surveys.index))
    survey_emails_set = set(surveys['email_norm'].dropna())
    
    matches = []
    match_count = 0
    
    # Processar cada comprador
    for idx, buyer in buyers.iterrows():
        buyer_email_norm = buyer.get('email_norm')
        if pd.isna(buyer_email_norm):
            continue
        
        # Matching direto com email_norm
        if buyer_email_norm in survey_emails_set:
            survey_idx = survey_emails_dict[buyer_email_norm]
            
            match_data = {
                'buyer_id': idx,
                'survey_id': survey_idx,
                'match_type': 'exact',
                'score': 1.0
            }
            
            # Adicionar informação de lançamento se disponível
            if 'lançamento' in buyer and not pd.isna(buyer['lançamento']):
                match_data['lançamento'] = buyer['lançamento']
            
            matches.append(match_data)
            match_count += 1
    
    print(f"   Found {match_count} matches out of {len(buyers)} buyers")
    print(f"   Match rate: {(match_count/len(buyers)*100):.1f}%")
    
    matches_df = pd.DataFrame(matches)
    
    # Calcular tempo gasto
    end_time = time.time()
    print(f"   Matching completed in {end_time - start_time:.2f} seconds.")
    
    return matches_df

def create_target_variable(surveys_df, matches_df):
    """Cria a variável alvo com base nas correspondências de compras."""
    if surveys_df.empty:
        print("No surveys data - creating empty DataFrame with target variable")
        return pd.DataFrame(columns=['target'])
    
    # Copiar o DataFrame para não modificar o original
    result_df = surveys_df.copy()
    
    # Adicionar coluna de target inicializada com 0
    result_df['target'] = 0
    
    # Se não houver correspondências, retornar com todos zeros
    if matches_df.empty:
        print("No matches found - target variable will be all zeros")
        return result_df
    
    # Marcar os registros correspondentes como positivos
    matched_count = 0
    for _, match in matches_df.iterrows():
        survey_id = match['survey_id']
        if survey_id in result_df.index:
            result_df.loc[survey_id, 'target'] = 1
            matched_count += 1
    
    # Verificar integridade do target
    positive_count = result_df['target'].sum()
    expected_positives = len(matches_df)
    
    print(f"   Created target variable: {positive_count} positive examples out of {len(result_df)}")
    
    if positive_count != expected_positives:
        print(f"   WARNING: Target integrity check failed!")
        print(f"      Expected {expected_positives} positives (from matches)")
        print(f"      Got {positive_count} positives in target")
    
    return result_df

def merge_datasets(surveys_df, utm_df, buyers_df):
    """Mescla as diferentes fontes de dados em um único dataset."""
    print("Merging datasets...")
    
    if surveys_df.empty:
        print("WARNING: Empty survey data")
        return pd.DataFrame()
    
    # Mesclar pesquisas com UTM usando email_norm
    if not utm_df.empty and 'email_norm' in utm_df.columns and 'email_norm' in surveys_df.columns:
        # Remover duplicatas de email_norm nas UTMs antes do merge
        print(f"   UTM records before deduplication: {len(utm_df):,}")
        utm_df_dedup = utm_df.drop_duplicates(subset=['email_norm'])
        print(f"   UTM records after deduplication: {len(utm_df_dedup):,}")
        
        merged_df = pd.merge(
            surveys_df,
            utm_df_dedup,
            on='email_norm',
            how='left',
            suffixes=('', '_utm')
        )
        print(f"   Merged surveys with UTM data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        # Consolidar colunas de email das UTMs
        if 'E-MAIL_utm' in merged_df.columns and 'E-MAIL' not in merged_df.columns:
            merged_df['E-MAIL'] = merged_df['E-MAIL_utm']
        elif 'E-MAIL_utm' in merged_df.columns and 'E-MAIL' in merged_df.columns:
            # Preencher valores vazios de E-MAIL com E-MAIL_utm
            mask = merged_df['E-MAIL'].isna() & merged_df['E-MAIL_utm'].notna()
            merged_df.loc[mask, 'E-MAIL'] = merged_df.loc[mask, 'E-MAIL_utm']
    else:
        merged_df = surveys_df.copy()
        print("   No UTM data available for merging")
    
    print(f"   Final merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    return merged_df

def prepare_final_dataset(df):
    """Prepara o dataset final removendo apenas email_norm e preservando emails originais."""
    print("\nPreparing final dataset...")
    
    print(f"   Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Listar todas as colunas originais
    original_columns = set(df.columns)
    
    # Primeiro, preservar dados de email nas colunas corretas
    df_preserved = preserve_email_data(df)
    
    # Remover email_norm (usado apenas para matching)
    if 'email_norm' in df_preserved.columns:
        df_preserved = df_preserved.drop(columns=['email_norm'])
    
    # Remover coluna 'email' genérica se ainda existir
    if 'email' in df_preserved.columns:
        df_preserved = df_preserved.drop(columns=['email'])
    
    # Filtrar para colunas de inferência (preservando dados)
    df_filtered = filter_to_inference_columns_preserving_data(df_preserved, add_missing=True)
    
    # Análise detalhada de colunas removidas
    final_columns = set(df_filtered.columns)
    removed_columns = original_columns - final_columns
    
    print(f"\n   COLUMNS REMOVED FOR PRODUCTION COMPATIBILITY:")
    print(f"   Total columns removed: {len(removed_columns)}")
    if removed_columns:
        print(f"   Removed columns list:")
        for col in sorted(removed_columns):
            print(f"      - {col}")
    else:
        print(f"   No columns were removed (all original columns are in INFERENCE_COLUMNS)")
    
    print(f"\n   Final dataset: {df_filtered.shape[0]} rows, {df_filtered.shape[1]} columns")
    
    return df_filtered

def validate_production_compatibility(df, show_warnings=True):
    """Valida se o DataFrame é compatível com produção."""
    validation_report = {
        'is_compatible': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Colunas esperadas em produção (sem target)
    production_columns = [col for col in INFERENCE_COLUMNS if col != 'target']
    
    # 1. Verificar colunas obrigatórias
    df_columns = set(df.columns)
    missing_columns = set(production_columns) - df_columns
    extra_columns = df_columns - set(production_columns) - {'target'}  # target é permitido em treino
    
    if missing_columns:
        validation_report['is_compatible'] = False
        validation_report['errors'].append(f"Missing required columns: {missing_columns}")
    
    if extra_columns:
        validation_report['warnings'].append(f"Extra columns found: {extra_columns}")
    
    # 2. Verificar ordem das colunas
    expected_order = [col for col in production_columns if col in df.columns]
    actual_order = [col for col in df.columns if col in production_columns]
    
    if expected_order != actual_order:
        validation_report['warnings'].append("Column order differs from expected")
    
    # 3. Verificar dados nas colunas críticas
    critical_data_columns = ['¿Cuál es tu e-mail?', 'E-MAIL', 'DATA']
    for col in critical_data_columns:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            null_percentage = (df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 0
            
            validation_report['info'].append(f"{col}: {non_null_count} non-null values ({100-null_percentage:.1f}% coverage)")
            
            if null_percentage > 90:
                validation_report['warnings'].append(f"{col} has {null_percentage:.1f}% null values")
    
    # 4. Verificar target (se presente)
    if 'target' in df.columns:
        target_values = df['target'].unique()
        if not set(target_values).issubset({0, 1, np.nan}):
            validation_report['errors'].append(f"Invalid target values: {target_values}")
            validation_report['is_compatible'] = False
        else:
            positive_rate = (df['target'] == 1).sum() / len(df) * 100 if len(df) > 0 else 0
            validation_report['info'].append(f"Target positive rate: {positive_rate:.2f}%")
    
    return validation_report['is_compatible'], validation_report

# ============================================================================
# FUNÇÕES DA PARTE 2 - PRÉ-PROCESSAMENTO
# ============================================================================

def apply_preprocessing_pipeline(df, params=None, fit=False, preserve_text=True):
    """Aplica a pipeline completa de pré-processamento."""
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
    
    # 9. Feature engineering avançada (nova etapa)
    print("9. Aplicando feature engineering avançada...")
    advanced_params = params.get('advanced_features', {})
    df, advanced_params = advanced_feature_engineering(df, fit=fit, params=advanced_params)
    
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
    """Garante que o DataFrame de teste tenha as mesmas colunas que o de treinamento."""
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

# ============================================================================
# FUNÇÕES DA PARTE 3 - FEATURE ENGINEERING PROFISSIONAL
# ============================================================================

def identify_text_columns_for_professional(df):
    """
    Identifica colunas de texto no DataFrame para processamento profissional.
    """
    text_columns_normalized = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
        'Déjame un mensaje'
    ]
    
    text_columns = []
    for col in text_columns_normalized:
        if col in df.columns:
            text_columns.append(col)
            print(f"  ✓ Encontrada: {col[:60]}...")
    
    # Se não encontrou a versão normalizada, buscar variações
    if '¿Qué esperas aprender en el evento Cero a Inglés Fluido?' not in text_columns:
        variations = [
            '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
            '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
        ]
        for var in variations:
            if var in df.columns:
                text_columns.append(var)
                print(f"  ✓ Encontrada variação: {var[:60]}...")
    
    return text_columns

def perform_topic_modeling_fixed(df, text_cols, n_topics=5, fit=True, params=None):
    """
    Extrai tópicos latentes dos textos usando LDA - VERSÃO CORRIGIDA.
    """
    if params is None:
        params = {}
    
    if 'lda' not in params:
        params['lda'] = {}
    
    df_result = df.copy()
    
    # Filtrar colunas de texto existentes
    text_cols = [col for col in text_cols if col in df.columns]
    print(f"\n🔍 Iniciando processamento LDA para {len(text_cols)} colunas de texto")
    
    # Contador de features LDA criadas
    lda_features_created = 0
    
    for i, col in enumerate(text_cols):
        print(f"\n[{i+1}/{len(text_cols)}] Processando LDA para: {col[:60]}...")
        
        col_clean = re.sub(r'[^\w]', '', col.replace(' ', '_'))[:30]
        
        # Verificar se temos texto limpo
        texts = df[col].fillna('').astype(str)
        valid_texts = texts[texts.str.len() > 10]
        
        print(f"  📊 Textos válidos: {len(valid_texts)} de {len(texts)} total")
        
        if len(valid_texts) < 50:  # Reduzido de 10 para 50 para garantir qualidade
            print(f"  ⚠️ Poucos textos válidos para LDA. Pulando esta coluna.")
            continue
        
        if fit:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import LatentDirichletAllocation
                
                # Vetorizar textos
                print(f"  🔄 Vetorizando textos...")
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    min_df=5,
                    max_df=0.95,
                    stop_words=None
                )
                
                doc_term_matrix = vectorizer.fit_transform(valid_texts)
                print(f"  ✓ Matriz documento-termo: {doc_term_matrix.shape}")
                
                # Aplicar LDA
                print(f"  🔄 Aplicando LDA com {n_topics} tópicos...")
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    max_iter=20,
                    learning_method='online',
                    random_state=42,
                    n_jobs=-1
                )
                
                # Transformar apenas textos válidos
                topic_dist_valid = lda.fit_transform(doc_term_matrix)
                
                # Criar distribuição completa (zeros para textos inválidos)
                topic_distribution = np.zeros((len(df), n_topics))
                valid_indices = texts[texts.str.len() > 10].index
                for idx, valid_idx in enumerate(valid_indices):
                    topic_distribution[valid_idx] = topic_dist_valid[idx]
                
                # Armazenar modelo
                params['lda'][col_clean] = {
                    'model': lda,
                    'vectorizer': vectorizer,
                    'n_topics': n_topics,
                    'feature_names': vectorizer.get_feature_names_out().tolist()
                }
                
                # Adicionar features ao DataFrame
                for topic_idx in range(n_topics):
                    feature_name = f'{col_clean}_topic_{topic_idx+1}'
                    df_result[feature_name] = topic_distribution[:, topic_idx]
                    lda_features_created += 1
                    print(f"  ✓ Criada feature: {feature_name}")
                
                # Adicionar tópico dominante
                dominant_topic_name = f'{col_clean}_dominant_topic'
                df_result[dominant_topic_name] = np.argmax(topic_distribution, axis=1) + 1
                lda_features_created += 1
                print(f"  ✓ Criada feature: {dominant_topic_name}")
                
                print(f"  ✅ LDA concluído! {n_topics + 1} features criadas para esta coluna")
                
            except Exception as e:
                print(f"  ❌ Erro ao aplicar LDA: {e}")
                import traceback
                traceback.print_exc()
        
        else:  # transform mode
            if col_clean in params['lda']:
                try:
                    print(f"  🔄 Aplicando LDA pré-treinado...")
                    
                    # Recuperar modelo e vetorizador
                    lda = params['lda'][col_clean]['model']
                    vectorizer = params['lda'][col_clean]['vectorizer']
                    n_topics = params['lda'][col_clean]['n_topics']
                    
                    # Vetorizar e transformar textos válidos
                    doc_term_matrix = vectorizer.transform(valid_texts)
                    topic_dist_valid = lda.transform(doc_term_matrix)
                    
                    # Criar distribuição completa
                    topic_distribution = np.zeros((len(df), n_topics))
                    valid_indices = texts[texts.str.len() > 10].index
                    for idx, valid_idx in enumerate(valid_indices):
                        topic_distribution[valid_idx] = topic_dist_valid[idx]
                    
                    # Adicionar features
                    for topic_idx in range(n_topics):
                        feature_name = f'{col_clean}_topic_{topic_idx+1}'
                        df_result[feature_name] = topic_distribution[:, topic_idx]
                        lda_features_created += 1
                    
                    # Tópico dominante
                    dominant_topic_name = f'{col_clean}_dominant_topic'
                    df_result[dominant_topic_name] = np.argmax(topic_distribution, axis=1) + 1
                    lda_features_created += 1
                    
                    print(f"  ✅ LDA aplicado! {n_topics + 1} features criadas")
                    
                except Exception as e:
                    print(f"  ❌ Erro ao transformar com LDA: {e}")
            else:
                print(f"  ⚠️ Modelo LDA não encontrado para '{col_clean}'")
    
    print(f"\n📊 RESUMO LDA: Total de {lda_features_created} features LDA criadas")
    
    return df_result, params

def apply_professional_features_pipeline(df, params=None, fit=False, batch_size=5000):
    """
    Aplica pipeline de features profissionais de NLP.
    """
    if params is None:
        params = {}
    
    # Inicializar subparâmetros se não existirem
    if 'professional_features' not in params:
        params['professional_features'] = {
            'professional_motivation': {},
            'aspiration_sentiment': {},
            'commitment': {},
            'career_terms': {},
            'career_tfidf': {},
            'lda': {}
        }
    
    print(f"\nIniciando pipeline de features profissionais para DataFrame: {df.shape}")
    start_time = time.time()
    
    # Identificar colunas de texto
    text_columns = identify_text_columns_for_professional(df)
    
    if not text_columns:
        print("AVISO: Nenhuma coluna de texto encontrada para processamento profissional!")
        return df, params
    
    print(f"\n✓ {len(text_columns)} colunas de texto identificadas para processamento profissional")
    
    # Processar em batches para economia de memória
    n_samples = len(df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"Processando {n_samples} amostras em {n_batches} batches (batch_size: {batch_size})")
    
    # Processar cada coluna de texto
    for col_idx, col in enumerate(text_columns):
        if col not in df.columns:
            continue
        
        print(f"\n[{col_idx+1}/{len(text_columns)}] Processando: {col[:60]}...")
        
        col_key = re.sub(r'[^\w]', '', col.replace(' ', '_'))[:30]
        
        # Processar em batches
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, n_samples)
            
            # Mostrar progresso
            if batch_idx % 5 == 0:
                elapsed = time.time() - start_time
                if batch_idx > 0:
                    avg_time_per_batch = elapsed / batch_idx
                    remaining_batches = n_batches - batch_idx
                    eta = avg_time_per_batch * remaining_batches
                    print(f"\r  Batch {batch_idx+1}/{n_batches} "
                          f"({(batch_idx+1)/n_batches*100:.1f}%) "
                          f"ETA: {eta/60:.1f} min", end='', flush=True)
            
            # Criar DataFrame do batch
            batch_df = df.iloc[batch_start:batch_end][[col]].copy()
            
            # 1. Score de motivação profissional
            if batch_idx == 0:
                print("\n  1. Calculando score de motivação profissional...")
            
            motiv_df, motiv_params = create_professional_motivation_score(
                batch_df, [col], 
                fit=fit and batch_idx == 0,
                params=params['professional_features']['professional_motivation'] if not (fit and batch_idx == 0) else None
            )
            if fit and batch_idx == 0:
                params['professional_features']['professional_motivation'] = motiv_params
            
            for motiv_col in motiv_df.columns:
                if motiv_col not in df.columns:
                    df[motiv_col] = np.nan
                df.loc[batch_start:batch_end-1, motiv_col] = motiv_df[motiv_col].values
            
            # 2. Análise de sentimento de aspiração
            if batch_idx == 0:
                print("\n  2. Analisando sentimento de aspiração...")
            
            asp_df, asp_params = analyze_aspiration_sentiment(
                batch_df, [col],
                fit=fit and batch_idx == 0,
                params=params['professional_features']['aspiration_sentiment'] if not (fit and batch_idx == 0) else None
            )
            if fit and batch_idx == 0:
                params['professional_features']['aspiration_sentiment'] = asp_params
            
            for asp_col in asp_df.columns:
                if asp_col not in df.columns:
                    df[asp_col] = np.nan
                df.loc[batch_start:batch_end-1, asp_col] = asp_df[asp_col].values
            
            # 3. Detecção de expressões de compromisso
            if batch_idx == 0:
                print("\n  3. Detectando expressões de compromisso...")
            
            comm_df, comm_params = detect_commitment_expressions(
                batch_df, [col],
                fit=fit and batch_idx == 0,
                params=params['professional_features']['commitment'] if not (fit and batch_idx == 0) else None
            )
            if fit and batch_idx == 0:
                params['professional_features']['commitment'] = comm_params
            
            for comm_col in comm_df.columns:
                if comm_col not in df.columns:
                    df[comm_col] = np.nan
                df.loc[batch_start:batch_end-1, comm_col] = comm_df[comm_col].values
            
            # 4. Detector de termos de carreira
            if batch_idx == 0:
                print("\n  4. Detectando termos de carreira...")
            
            career_df, career_params = create_career_term_detector(
                batch_df, [col],
                fit=fit and batch_idx == 0,
                params=params['professional_features']['career_terms'] if not (fit and batch_idx == 0) else None
            )
            if fit and batch_idx == 0:
                params['professional_features']['career_terms'] = career_params
            
            for career_col in career_df.columns:
                if career_col not in df.columns:
                    df[career_col] = np.nan
                df.loc[batch_start:batch_end-1, career_col] = career_df[career_col].values
            
            # Limpar memória após cada batch
            del batch_df
            gc.collect()
        
        print()  # Nova linha após progresso
        
        # 5. TF-IDF aprimorado (processa coluna inteira)
        print("  5. Aplicando TF-IDF aprimorado para termos de carreira...")

        temp_df = df[[col]].copy()

        tfidf_df, tfidf_params = enhance_tfidf_for_career_terms(
            temp_df, [col],
            fit=fit,
            params=params['professional_features']['career_tfidf'].get(col_key) if not fit else None
        )

        if fit:
            if 'career_tfidf' not in params['professional_features']:
                params['professional_features']['career_tfidf'] = {}
            params['professional_features']['career_tfidf'][col_key] = tfidf_params
        
        added_count = 0
        for tfidf_col in tfidf_df.columns:
            if tfidf_col not in df.columns and tfidf_col != col:
                df[tfidf_col] = tfidf_df[tfidf_col].values
                added_count += 1
        print(f"     ✓ Adicionadas {added_count} features TF-IDF de carreira")
    
    # 6. Aplicar LDA após processar todas as features profissionais
    print("\n6. Aplicando LDA para extração de tópicos...")
    df, params['professional_features'] = perform_topic_modeling_fixed(
        df, text_columns, n_topics=5, fit=fit, params=params['professional_features']
    )
    
    # Relatório final
    elapsed_time = time.time() - start_time
    print(f"\n✓ Processamento de features profissionais concluído em {elapsed_time/60:.1f} minutos")
    
    return df, params

def summarize_features(df, dataset_name, original_shape=None):
    """
    Sumariza as features criadas no DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"SUMÁRIO DE FEATURES - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    if original_shape:
        print(f"\n📊 DIMENSÕES:")
        print(f"   Original: {original_shape[0]} linhas × {original_shape[1]} colunas")
        print(f"   Atual:    {df.shape[0]} linhas × {df.shape[1]} colunas")
        print(f"   Features adicionadas: {df.shape[1] - original_shape[1]}")
    
    # Categorizar features por tipo
    feature_categories = {
        'professional_motivation': [],
        'aspiration': [],
        'commitment': [],
        'career_terms': [],
        'career_tfidf': [],
        'topic': [],
        'tfidf': [],
        'sentiment': [],
        'motivation': [],
        'outras': []
    }
    
    for col in df.columns:
        if 'professional_motivation' in col or 'career_keyword' in col:
            feature_categories['professional_motivation'].append(col)
        elif 'aspiration' in col:
            feature_categories['aspiration'].append(col)
        elif 'commitment' in col:
            feature_categories['commitment'].append(col)
        elif 'career_tfidf' in col:
            feature_categories['career_tfidf'].append(col)
        elif 'career_term' in col:
            feature_categories['career_terms'].append(col)
        elif 'topic_' in col or 'dominant_topic' in col:
            feature_categories['topic'].append(col)
        elif '_tfidf_' in col and 'career' not in col:
            feature_categories['tfidf'].append(col)
        elif '_sentiment' in col and 'aspiration' not in col:
            feature_categories['sentiment'].append(col)
        elif '_motiv_' in col and 'professional' not in col:
            feature_categories['motivation'].append(col)
    
    print(f"\n📈 FEATURES POR CATEGORIA:")
    total_features = 0
    for category, features in feature_categories.items():
        if features and category != 'outras':
            print(f"\n   {category.upper()} ({len(features)} features):")
            # Mostrar até 3 exemplos
            for i, feat in enumerate(features[:3]):
                print(f"      • {feat}")
            if len(features) > 3:
                print(f"      ... e mais {len(features) - 3} features")
            total_features += len(features)
    
    print(f"\n   TOTAL DE FEATURES CATEGORIZADAS: {total_features}")
    
    print(f"\n{'='*60}\n")

# ============================================================================
# PIPELINE UNIFICADO PRINCIPAL
# ============================================================================

def unified_data_pipeline(raw_data_path="/Users/ramonmoreira/desktop/smart_ads/data/raw_data",
                         params_output_dir="/Users/ramonmoreira/desktop/smart_ads/src/preprocessing/params/unified_v2",
                         test_size=0.3,
                         val_size=0.5,
                         random_state=42,
                         preserve_text=True,
                         batch_size=5000):
    """
    Pipeline unificado que executa coleta, integração, pré-processamento e feature engineering.
    
    Args:
        raw_data_path: Caminho para os dados brutos
        params_output_dir: Diretório para salvar os parâmetros de pré-processamento
        test_size: Proporção do conjunto de teste
        val_size: Proporção do conjunto de validação dentro do conjunto de teste
        random_state: Semente aleatória para reprodutibilidade
        preserve_text: Se True, preserva as colunas de texto originais
        batch_size: Tamanho do batch para processamento de features profissionais
        
    Returns:
        Dicionário com os DataFrames processados:
        {
            'train': DataFrame de treino processado,
            'validation': DataFrame de validação processado,
            'test': DataFrame de teste processado,
            'params': Parâmetros de pré-processamento aprendidos
        }
    """
    print("========================================================================")
    print("INICIANDO PIPELINE UNIFICADO DE COLETA, INTEGRAÇÃO, PRÉ-PROCESSAMENTO")
    print("E FEATURE ENGINEERING PROFISSIONAL")
    print("========================================================================")
    print(f"Diretório de dados brutos: {raw_data_path}")
    print(f"Diretório de parâmetros: {params_output_dir}")
    print(f"Test size: {test_size}, Val size: {val_size}")
    print(f"Random state: {random_state}")
    print(f"Preserve text: {preserve_text}")
    print(f"Batch size: {batch_size}")
    print()
    
    # ========================================================================
    # PARTE 1: COLETA E INTEGRAÇÃO DE DADOS
    # ========================================================================
    
    print("\n=== PARTE 1: COLETA E INTEGRAÇÃO DE DADOS ===")
    
    # 1. Conectar ao armazenamento local
    bucket = connect_to_gcs("local_bucket", data_path=raw_data_path)
    
    # 2. Listar e categorizar arquivos
    file_paths = list_files_by_extension(bucket, prefix="")
    print(f"Found {len(file_paths)} files in: {raw_data_path}")
    
    # 3. Categorizar arquivos
    survey_files, buyer_files, utm_files, all_files_by_launch = categorize_files(file_paths)
    
    print(f"Survey files: {len(survey_files)}")
    print(f"Buyer files: {len(buyer_files)}")
    print(f"UTM files: {len(utm_files)}")
    
    # 4. Carregar dados (preservando colunas originais)
    survey_dfs, _ = load_survey_files(bucket, survey_files)
    buyer_dfs, _ = load_buyer_files(bucket, buyer_files)
    utm_dfs, _ = load_utm_files(bucket, utm_files)
    
    # 5. Combinar datasets
    surveys = pd.concat(survey_dfs, ignore_index=True) if survey_dfs else pd.DataFrame()
    buyers = pd.concat(buyer_dfs, ignore_index=True) if buyer_dfs else pd.DataFrame()
    utms = pd.concat(utm_dfs, ignore_index=True) if utm_dfs else pd.DataFrame()
    
    print(f"Survey data: {surveys.shape[0]:,} rows, {surveys.shape[1]} columns")
    print(f"Buyer data: {buyers.shape[0]:,} rows, {buyers.shape[1]} columns")
    print(f"UTM data: {utms.shape[0]:,} rows, {utms.shape[1]} columns")
    
    # 6. Normalizar emails (criando email_norm para matching)
    if not surveys.empty:
        surveys = normalize_emails_preserving_originals(surveys)
    
    if not buyers.empty and 'email' in buyers.columns:
        buyers = normalize_emails_in_dataframe(buyers)
    
    if not utms.empty:
        utms = normalize_emails_preserving_originals(utms)
    
    # 7. Matching
    matches_df = match_surveys_with_buyers_improved(surveys, buyers, utms)
    
    # 8. Criar variável alvo
    surveys_with_target = create_target_variable(surveys, matches_df)
    
    # 9. Mesclar datasets
    merged_data = merge_datasets(surveys_with_target, utms, pd.DataFrame())
    
    # 10. Preparar dataset final (preservando emails)
    final_data = prepare_final_dataset(merged_data)
    
    # Validar compatibilidade com produção
    is_compatible, validation_report = validate_production_compatibility(final_data)
    
    # 11. Estatísticas finais da integração
    if 'target' in final_data.columns:
        target_counts = final_data['target'].value_counts()
        total_records = len(final_data)
        positive_rate = (target_counts.get(1, 0) / total_records * 100) if total_records > 0 else 0
        print(f"\nTarget variable distribution:")
        print(f"   Negative (0): {target_counts.get(0, 0):,} ({100-positive_rate:.2f}%)")
        print(f"   Positive (1): {target_counts.get(1, 0):,} ({positive_rate:.2f}%)")
    
    # ========================================================================
    # PARTE 2: DIVISÃO DOS DADOS (ANTES DO PRÉ-PROCESSAMENTO)
    # ========================================================================
    
    print("\n=== PARTE 2: DIVISÃO DOS DADOS ===")
    print("Splitting data into train/val/test sets...")
    
    if final_data.shape[0] == 0:
        print("WARNING: Empty dataset - cannot proceed.")
        return None
    
    # Verificar se temos target para estratificar
    if 'target' in final_data.columns and final_data['target'].nunique() > 1:
        print("   Using stratified split based on target variable")
        strat_col = final_data['target']
    else:
        print("   Using random split (no stratification)")
        strat_col = None
    
    # Primeira divisão: treino vs. (validação + teste)
    train_df, temp_df = train_test_split(
        final_data,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_col
    )
    
    # Segunda divisão: validação vs. teste
    if 'target' in temp_df.columns and temp_df['target'].nunique() > 1:
        strat_col_temp = temp_df['target']
    else:
        strat_col_temp = None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        random_state=random_state,
        stratify=strat_col_temp
    )
    
    print(f"   Train set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"   Validation set: {val_df.shape[0]} rows, {val_df.shape[1]} columns")
    print(f"   Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    
    # Guardar shapes originais para comparação
    train_original_shape = train_df.shape
    val_original_shape = val_df.shape
    test_original_shape = test_df.shape
    
    # ========================================================================
    # PARTE 3: PRÉ-PROCESSAMENTO
    # ========================================================================
    
    print("\n=== PARTE 3: PRÉ-PROCESSAMENTO ===")
    
    # 1. Processar o conjunto de treinamento com fit=True para aprender parâmetros
    print("\n--- Processando conjunto de treinamento ---")
    train_processed, params = apply_preprocessing_pipeline(train_df, fit=True, preserve_text=preserve_text)
    
    # 2. Processar o conjunto de validação com fit=False para aplicar parâmetros aprendidos
    print("\n--- Processando conjunto de validação ---")
    val_processed, _ = apply_preprocessing_pipeline(val_df, params=params, fit=False, preserve_text=preserve_text)
    
    # 3. Garantir consistência de colunas com o treino
    val_processed = ensure_column_consistency(train_processed, val_processed)
    
    # 4. Processar o conjunto de teste com fit=False para aplicar parâmetros aprendidos
    print("\n--- Processando conjunto de teste ---")
    test_processed, _ = apply_preprocessing_pipeline(test_df, params=params, fit=False, preserve_text=preserve_text)
    
    # 5. Garantir consistência de colunas com o treino
    test_processed = ensure_column_consistency(train_processed, test_processed)
    
    # Guardar shapes após pré-processamento
    train_after_preproc_shape = train_processed.shape
    val_after_preproc_shape = val_processed.shape
    test_after_preproc_shape = test_processed.shape
    
    # ========================================================================
    # PARTE 4: FEATURE ENGINEERING PROFISSIONAL
    # ========================================================================
    
    print("\n=== PARTE 4: FEATURE ENGINEERING PROFISSIONAL ===")
    
    # 1. Aplicar features profissionais no conjunto de treinamento
    print("\n--- Aplicando features profissionais no conjunto de treinamento ---")
    train_final, params = apply_professional_features_pipeline(
        train_processed, params=params, fit=True, batch_size=batch_size
    )
    
    # 2. Aplicar features profissionais no conjunto de validação
    print("\n--- Aplicando features profissionais no conjunto de validação ---")
    val_final, _ = apply_professional_features_pipeline(
        val_processed, params=params, fit=False, batch_size=batch_size
    )
    
    # 3. Garantir consistência de colunas
    val_final = ensure_column_consistency(train_final, val_final)
    
    # 4. Aplicar features profissionais no conjunto de teste
    print("\n--- Aplicando features profissionais no conjunto de teste ---")
    test_final, _ = apply_professional_features_pipeline(
        test_processed, params=params, fit=False, batch_size=batch_size
    )
    
    # 5. Garantir consistência de colunas
    test_final = ensure_column_consistency(train_final, test_final)
    
    # ========================================================================
    # SALVAR PARÂMETROS E RELATÓRIO
    # ========================================================================
    
    print("\n=== SALVANDO PARÂMETROS E RELATÓRIOS ===")
    
    # Criar diretório de parâmetros
    os.makedirs(params_output_dir, exist_ok=True)
    
    # Salvar parâmetros completos
    params_path = os.path.join(params_output_dir, "all_preprocessing_params.joblib")
    joblib.dump(params, params_path)
    print(f"Parâmetros de pré-processamento salvos em {params_path}")
    
    # Salvar relatório de validação
    validation_report_path = os.path.join(params_output_dir, "validation_report.json")
    with open(validation_report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    print(f"Relatório de validação salvo em {validation_report_path}")
    
    # ========================================================================
    # VERIFICAÇÃO DE CONSISTÊNCIA FINAL
    # ========================================================================
    
    print("\n=== VERIFICAÇÃO DE CONSISTÊNCIA FINAL ===")
    
    train_cols = set(train_final.columns)
    val_cols = set(val_final.columns)
    test_cols = set(test_final.columns)
    
    if train_cols == val_cols == test_cols:
        print("✓ Todos os datasets têm exatamente as mesmas colunas")
        print(f"  Total de colunas: {len(train_cols)}")
    else:
        print("✗ AVISO: Inconsistência detectada nas colunas!")
        if train_cols - val_cols:
            print(f"  Colunas em train mas não em valid: {len(train_cols - val_cols)}")
        if train_cols - test_cols:
            print(f"  Colunas em train mas não em test: {len(train_cols - test_cols)}")
    
    # ========================================================================
    # RESUMOS E ESTATÍSTICAS
    # ========================================================================
    
    # Sumarizar features para cada conjunto
    summarize_features(train_final, 'train', train_original_shape)
    summarize_features(val_final, 'validation', val_original_shape)
    summarize_features(test_final, 'test', test_original_shape)
    
    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    
    print("\n========================================================================")
    print("PIPELINE UNIFICADO CONCLUÍDO COM SUCESSO!")
    print("========================================================================")
    
    print(f"\n📊 EVOLUÇÃO DAS DIMENSÕES:")
    print(f"   Dataset   | Original | Após Preproc | Final   | Total Adicionadas")
    print(f"   ----------|----------|--------------|---------|------------------")
    print(f"   Train     | {train_original_shape[1]:>8} | {train_after_preproc_shape[1]:>12} | {train_final.shape[1]:>7} | {train_final.shape[1] - train_original_shape[1]:>17}")
    print(f"   Valid     | {val_original_shape[1]:>8} | {val_after_preproc_shape[1]:>12} | {val_final.shape[1]:>7} | {val_final.shape[1] - val_original_shape[1]:>17}")
    print(f"   Test      | {test_original_shape[1]:>8} | {test_after_preproc_shape[1]:>12} | {test_final.shape[1]:>7} | {test_final.shape[1] - test_original_shape[1]:>17}")
    
    print(f"\n📁 ARQUIVOS SALVOS:")
    print(f"   Parâmetros: {params_output_dir}")
    
    print(f"\n✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"   Tempo total: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("========================================================================\n")
    
    # Retornar DataFrames processados e parâmetros
    return {
        'train': train_final,
        'validation': val_final,
        'test': test_final,
        'params': params
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Executar o pipeline unificado
    results = unified_data_pipeline(
        raw_data_path="/Users/ramonmoreira/desktop/smart_ads/data/raw_data",
        params_output_dir="/Users/ramonmoreira/desktop/smart_ads/src/preprocessing/params/unified_v2",
        test_size=0.3,
        val_size=0.5,
        random_state=42,
        preserve_text=True,
        batch_size=5000
    )
    
    if results:
        print("\nDataFrames processados disponíveis em memória:")
        print("- results['train']")
        print("- results['validation']")
        print("- results['test']")
        print("- results['params']")
        
        # Mostrar exemplo de como acessar os dados
        print("\nExemplo de uso:")
        print("train_df = results['train']")
        print("print(f'Shape do treino: {train_df.shape}')")
        print("print(f'Colunas: {train_df.columns.tolist()[:5]} ...')")