#!/usr/bin/env python
"""
Script para coleta e integração de dados usando armazenamento local.
Esta versão é robusta para lidar com arquivos ausentes ou problemas de carregamento.
"""

import os
import sys
import pandas as pd

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar módulo de armazenamento local
from src.utils.local_storage import (
    connect_to_gcs, list_files_by_extension, categorize_files,
    load_csv_or_excel, load_csv_with_auto_delimiter, extract_launch_id
)
from src.preprocessing.email_processing import normalize_emails_in_dataframe
from src.preprocessing.data_matching import match_surveys_with_buyers
from src.preprocessing.data_integration import create_target_variable, merge_datasets, split_data
from src.preprocessing.column_normalization import normalize_survey_columns, validate_normalized_columns

# Funções adaptadas do data_loader.py para usar o armazenamento local
def find_email_column(df):
    """Encontra a coluna que contém emails em um DataFrame."""
    email_patterns = ['email', 'e-mail', 'correo', '@', 'mail']
    for col in df.columns:
        if any(pattern in col.lower() for pattern in email_patterns):
            return col
    return None

def process_survey_file(bucket, file_path):
    """Processa um arquivo de pesquisa COM NORMALIZAÇÃO DE COLUNAS."""
    df = load_csv_or_excel(bucket, file_path)
    if df is None:
        return None
        
    # Identificar o lançamento deste arquivo
    launch_id = extract_launch_id(file_path)
    
    # ====== NOVA FUNCIONALIDADE: NORMALIZAR COLUNAS ======
    df = normalize_survey_columns(df, launch_id)
    validate_normalized_columns(df, launch_id)
    # ====================================================
    
    # Encontrar e renomear coluna de email
    email_col = find_email_column(df)
    if email_col and email_col != 'email':
        df = df.rename(columns={email_col: 'email'})
    
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
    
    # Encontrar e renomear coluna de email
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
    """Processa um arquivo de UTM."""
    df = load_csv_with_auto_delimiter(bucket, file_path)
    if df is None:
        return None
        
    # Identificar o lançamento deste arquivo
    launch_id = extract_launch_id(file_path)
    
    # Encontrar e renomear coluna de email
    email_col = find_email_column(df)
    if email_col and email_col != 'email':
        df = df.rename(columns={email_col: 'email'})
    elif not email_col:
        print(f"  - Warning: No email column found in {file_path}. Available columns: {', '.join(df.columns[:5])}...")
    
    # Adicionar identificador de lançamento se disponível
    if launch_id:
        df['lançamento'] = launch_id
    
    return df

def load_survey_files(bucket, survey_files):
    """Carrega todos os arquivos de pesquisa COM NORMALIZAÇÃO e tratamento de erro."""
    survey_dfs = []
    launch_data = {}
    
    print("\nLoading survey files...")
    for file_path in survey_files:
        try:
            df = process_survey_file(bucket, file_path)  # Já inclui normalização que você implementou
            if df is not None:
                survey_dfs.append(df)
                
                # Armazenar por lançamento se disponível
                launch_id = extract_launch_id(file_path)
                if launch_id:
                    if launch_id not in launch_data:
                        launch_data[launch_id] = {}
                    launch_data[launch_id]['survey'] = df
                    
                print(f"  - Loaded: {file_path} ({launch_id if launch_id else ''}), {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
            # Continuar processamento mesmo com erro
    
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
                
                # Armazenar por lançamento se disponível
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
    """Carrega todos os arquivos de UTM."""
    utm_dfs = []
    launch_data = {}
    
    print("\nLoading UTM files...")
    for file_path in utm_files:
        try:
            df = process_utm_file(bucket, file_path)
            if df is not None:
                utm_dfs.append(df)
                
                # Armazenar por lançamento se disponível
                launch_id = extract_launch_id(file_path)
                if launch_id:
                    if launch_id not in launch_data:
                        launch_data[launch_id] = {}
                    launch_data[launch_id]['utm'] = df
                    
                print(f"  - Loaded: {file_path} ({launch_id if launch_id else ''}), {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
    
    return utm_dfs, launch_data

def main():
    """Pipeline principal para coleta e integração de dados."""
    # 1. Conectar ao armazenamento local
    print("Setting up connection to local storage...")
    bucket = connect_to_gcs("local_bucket")  # O nome do bucket é ignorado
    
    # 2. Listar e categorizar arquivos
    file_paths = list_files_by_extension(bucket, prefix="")
    print(f"Found {len(file_paths)} files")
    
    # 3. Categorizar arquivos por tipo e lançamento
    print("\nCategorizing files by type and launch...")
    survey_files, buyer_files, utm_files, all_files_by_launch = categorize_files(file_paths)
    
    # Mostrar informações de categorização
    print(f"Survey files: {len(survey_files)}")
    print(f"Buyer files: {len(buyer_files)}")
    print(f"UTM files: {len(utm_files)}")
    
    for launch_id, files in all_files_by_launch.items():
        if files:
            print(f"{launch_id} files: {len(files)}")
    
    # 4. Carregar dados
    survey_dfs, survey_launch_data = load_survey_files(bucket, survey_files)
    buyer_dfs, buyer_launch_data = load_buyer_files(bucket, buyer_files)
    utm_dfs, utm_launch_data = load_utm_files(bucket, utm_files)
    
    # 5. Combinar datasets
    print("\nCombining datasets...")
    surveys = pd.concat(survey_dfs, ignore_index=True) if survey_dfs else pd.DataFrame()
    buyers = pd.concat(buyer_dfs, ignore_index=True) if buyer_dfs else pd.DataFrame()
    utms = pd.concat(utm_dfs, ignore_index=True) if utm_dfs else pd.DataFrame()
    
    print(f"Survey data: {surveys.shape[0]} rows, {surveys.shape[1]} columns")
    print(f"Buyer data: {buyers.shape[0]} rows, {buyers.shape[1]} columns")
    print(f"UTM data: {utms.shape[0]} rows, {utms.shape[1]} columns")
    
    # 6. Documentar estrutura de dados
    print("\nDocumenting data structure...")
    for name, df in {"Surveys": surveys, "Buyers": buyers, "UTMs": utms}.items():
        if not df.empty:
            print(f"\n{name} structure:")
            print(f"  - Columns: {df.shape[1]}")
            print(f"  - Data types: {df.dtypes.value_counts().to_dict()}")
            print(f"  - Sample columns: {', '.join(df.columns[:5])}")
    
    # 7. Normalizar emails
    print("\nNormalizing email addresses...")
    if not surveys.empty and 'email' in surveys.columns:
        surveys = normalize_emails_in_dataframe(surveys)
    else:
        print("Warning: Cannot normalize survey emails - empty or missing email column")
        if surveys.empty:
            # Criar DataFrame para respostas de pesquisa com colunas mínimas
            surveys = pd.DataFrame(columns=['email', 'email_norm'])
    
    if not buyers.empty and 'email' in buyers.columns:
        buyers = normalize_emails_in_dataframe(buyers)
    else:
        print("Warning: Cannot normalize emails in buyers dataframe - empty or missing email column")
        if buyers.empty:
            buyers = pd.DataFrame(columns=['email', 'email_norm'])
    
    if not utms.empty and 'email' in utms.columns:
        utms = normalize_emails_in_dataframe(utms)
    else:
        print("Warning: Cannot normalize emails in UTM dataframe - empty or missing email column")
    
    # 8. Correspondência de pesquisas com dados de compradores
    matches_df = match_surveys_with_buyers(surveys, buyers)
    
    # 9. Criar variável alvo
    surveys_with_target = create_target_variable(surveys, matches_df)
    
    # 10. Mesclar datasets (sem incluir dados de compradores para evitar vazamento)
    merged_data = merge_datasets(surveys_with_target, utms, pd.DataFrame())
    
    # 11. Estatísticas de lançamento
    if 'lançamento' in merged_data.columns:
        launch_counts = merged_data['lançamento'].value_counts(dropna=False)
        print("\nRegistros por lançamento:")
        for launch, count in launch_counts.items():
            launch_str = "Sem lançamento identificado" if pd.isna(launch) else launch
            print(f"  - {launch_str}: {count} registros")
    
    # 12. Split dos dados para evitar vazamento
    # Ajustando para usar caminho relativo ao diretório do projeto
    output_dir = os.path.join(project_root, "data/split")
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar se há dados antes de fazer o split
    if merged_data.shape[0] > 0:
        # Verificar número de classes para stratify
        if 'target' in merged_data.columns and merged_data['target'].nunique() > 1:
            print("Performing stratified split")
            train_df, val_df, test_df = split_data(merged_data, output_dir, stratify=True)
        else:
            print("Performing random split (no stratification possible)")
            train_df, val_df, test_df = split_data(merged_data, output_dir, stratify=False)
        
        print("\nData collection and integration completed!")
        print(f"Data saved to: {output_dir}")
    else:
        print("\nWARNING: No data was merged. Check if:")
        print("1. You have installed openpyxl for Excel files: pip install openpyxl")
        print("2. The files can be properly loaded")
        print("3. The data files have expected structure with email columns")
    
if __name__ == "__main__":
    main()