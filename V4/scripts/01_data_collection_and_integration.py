#!/usr/bin/env python
"""
Script completo para coleta e integração de dados - VERSÃO FINAL CORRIGIDA
Inclui todas as correções:
1. Preservação de emails originais
2. Deduplicação de UTMs no merge
3. Correção do warning de dtype
4. Validação de integridade do target
5. Validação de compatibilidade com produção
"""

import os
import sys
import pandas as pd
import time
import numpy as np
import json

# Adicionar o diretório raiz do projeto ao sys.path
project_root = "/Users/ramonmoreira/Desktop/smart_ads"
sys.path.insert(0, project_root)

# Importar módulos existentes
from src.utils.local_storage import (
    connect_to_gcs, list_files_by_extension, categorize_files,
    load_csv_or_excel, load_csv_with_auto_delimiter, extract_launch_id
)
from src.preprocessing.email_processing import normalize_emails_in_dataframe, normalize_email
from src.preprocessing.column_normalization import normalize_survey_columns, validate_normalized_columns
from sklearn.model_selection import train_test_split

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
# FUNÇÕES AUXILIARES
# ============================================================================

def find_email_column(df):
    """Encontra a coluna que contém emails em um DataFrame."""
    email_patterns = ['email', 'e-mail', 'correo', '@', 'mail']
    for col in df.columns:
        if any(pattern in col.lower() for pattern in email_patterns):
            return col
    return None

def normalize_emails_preserving_originals(df, survey_email_col='¿Cuál es tu e-mail?', utm_email_col='E-MAIL'):
    """
    Normaliza emails criando email_norm para matching, mas preserva as colunas originais.
    
    Args:
        df: DataFrame para normalizar
        survey_email_col: Nome da coluna de email das pesquisas
        utm_email_col: Nome da coluna de email das UTMs
        
    Returns:
        DataFrame com email_norm adicionado e colunas originais preservadas
    """
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
    
    # CORREÇÃO: Inicializar email_norm com tipo string para evitar FutureWarning
    df_result['email_norm'] = pd.Series(dtype='object')
    
    for source_col in email_sources:
        # Preencher email_norm com valores não-nulos da fonte
        mask = df_result['email_norm'].isna() & df_result[source_col].notna()
        if mask.any():
            df_result.loc[mask, 'email_norm'] = df_result.loc[mask, source_col].apply(normalize_email)
    
    return df_result

def preserve_email_data(df):
    """
    Preserva dados de email movendo-os para as colunas corretas antes da filtragem.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame com emails preservados nas colunas corretas
    """
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
    """
    Filtra DataFrame para conter apenas colunas disponíveis na inferência,
    mas preserva os dados existentes nas colunas de email corretas.
    
    Args:
        df: DataFrame a ser filtrado
        add_missing: Se deve adicionar colunas faltantes com NaN
        
    Returns:
        DataFrame filtrado com dados preservados
    """
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

def validate_production_compatibility(df, show_warnings=True):
    """
    Valida se o DataFrame é compatível com produção.
    
    Args:
        df: DataFrame a ser validado
        show_warnings: Se deve mostrar avisos detalhados
        
    Returns:
        Tuple (is_compatible, validation_report)
    """
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
# FUNÇÕES DE PROCESSAMENTO
# ============================================================================

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
    
    # NÃO renomear coluna de email para 'email' genérico
    # Manter a coluna original (¿Cuál es tu e-mail?)
    
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
    
    # NÃO renomear E-MAIL para 'email' genérico
    # Manter a coluna original (E-MAIL)
    
    # Adicionar identificador de lançamento se disponível
    if launch_id:
        df['lançamento'] = launch_id
    
    return df

# ============================================================================
# FUNÇÕES DE CARREGAMENTO
# ============================================================================

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

# ============================================================================
# MATCHING E INTEGRAÇÃO
# ============================================================================

def match_surveys_with_buyers_improved(surveys, buyers, utms=None):
    """
    Realiza correspondência entre pesquisas e compradores usando email_norm.
    Adaptado para trabalhar com as colunas de email originais preservadas.
    """
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
    
    # VALIDAÇÃO: Verificar integridade do target
    positive_count = result_df['target'].sum()
    expected_positives = len(matches_df)
    
    print(f"   Created target variable: {positive_count} positive examples out of {len(result_df)}")
    
    if positive_count != expected_positives:
        print(f"   WARNING: Target integrity check failed!")
        print(f"      Expected {expected_positives} positives (from matches)")
        print(f"      Got {positive_count} positives in target")
    
    return result_df

def merge_datasets(surveys_df, utm_df, buyers_df):
    """
    Mescla as diferentes fontes de dados em um único dataset.
    CORRIGIDO: Remove duplicatas de UTM antes do merge para evitar explosão de registros.
    """
    print("Merging datasets...")
    
    if surveys_df.empty:
        print("WARNING: Empty survey data")
        return pd.DataFrame()
    
    # Mesclar pesquisas com UTM usando email_norm
    if not utm_df.empty and 'email_norm' in utm_df.columns and 'email_norm' in surveys_df.columns:
        # CORREÇÃO: Remover duplicatas de email_norm nas UTMs antes do merge
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
    """
    Prepara o dataset final removendo apenas email_norm e preservando emails originais.
    """
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

def split_data(df, output_dir, test_size=0.3, val_size=0.5, stratify=True, random_state=42):
    """Divide os dados em conjuntos de treino, validação e teste."""
    print("\nSplitting data into train/val/test sets...")
    
    if df.shape[0] == 0:
        print("WARNING: Empty dataset - cannot split.")
        empty_df = pd.DataFrame(columns=df.columns)
        return empty_df, empty_df, empty_df
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar se temos target para estratificar
    if stratify and 'target' in df.columns and df['target'].nunique() > 1:
        print("   Using stratified split based on target variable")
        strat_col = df['target']
    else:
        print("   Using random split (no stratification)")
        strat_col = None
    
    # Primeira divisão: treino vs. (validação + teste)
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_col
    )
    
    # Segunda divisão: validação vs. teste
    if stratify and 'target' in temp_df.columns and temp_df['target'].nunique() > 1:
        strat_col_temp = temp_df['target']
    else:
        strat_col_temp = None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        random_state=random_state,
        stratify=strat_col_temp
    )
    
    # Salvar os conjuntos
    print(f"   Train set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    
    print(f"   Validation set: {val_df.shape[0]} rows, {val_df.shape[1]} columns")
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    
    print(f"   Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    return train_df, val_df, test_df

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """Pipeline principal corrigido para preservar dados de email."""
    print("Starting data collection and integration pipeline...")
    
    # NOVO: Definir caminho dos dados brutos aqui
    raw_data_path = "/Users/ramonmoreira/Desktop/smart_ads/data/raw_data"
    
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
    
    # NOVA VALIDAÇÃO: Verificar compatibilidade com produção
    is_compatible, validation_report = validate_production_compatibility(final_data)
    
    # 11. Estatísticas finais
    if 'target' in final_data.columns:
        target_counts = final_data['target'].value_counts()
        total_records = len(final_data)
        positive_rate = (target_counts.get(1, 0) / total_records * 100) if total_records > 0 else 0
        print(f"\nTarget variable distribution:")
        print(f"   Negative (0): {target_counts.get(0, 0):,} ({100-positive_rate:.2f}%)")
        print(f"   Positive (1): {target_counts.get(1, 0):,} ({positive_rate:.2f}%)")
    
    # 12. Split dos dados
    output_dir = os.path.join(project_root, "data", "new", "01_split")
    
    if final_data.shape[0] > 0:
        train_df, val_df, test_df = split_data(final_data, output_dir, stratify=True)
        
        print(f"\nPipeline completed successfully!")
        print(f"Data saved to: {output_dir}")
        
        # Salvar relatório de validação
        validation_report_path = os.path.join(output_dir, "validation_report.json")
        with open(validation_report_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
    else:
        print("\nNo data was processed successfully!")

if __name__ == "__main__":
    main()