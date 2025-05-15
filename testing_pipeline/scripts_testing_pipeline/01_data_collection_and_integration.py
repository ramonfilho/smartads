#!/usr/bin/env python
"""
Script para coleta e integração de dados usando armazenamento local.
Esta versão usa apenas um arquivo de cada tipo por lançamento, na pasta principal.
"""

import os
import sys
import pandas as pd

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# ===== CONFIGURAÇÃO MANUAL DOS DIRETÓRIOS =====
# Defina aqui o caminho completo para o diretório contendo os dados de entrada
DATA_DIR = "/Users/ramonmoreira/Desktop/smart_ads/data/00_raw_data"

# Defina aqui o caminho completo para o diretório de saída dos dados processados
OUTPUT_DIR = "/Users/ramonmoreira/Desktop/smart_ads/testing_pipeline/data_testing_pipeline/01_split_testing_pipeline"
# =================================================

# Importar funções necessárias
from src.preprocessing.email_processing import normalize_emails_in_dataframe
from src.preprocessing.data_matching import match_surveys_with_buyers
from src.preprocessing.data_integration import create_target_variable, merge_datasets, split_data

# Classes para simular o comportamento de objetos de armazenamento
class LocalBlob:
    """Simula um blob para acesso a arquivos locais."""
    
    def __init__(self, full_path, name=None):
        self.full_path = full_path
        self.name = name if name else os.path.basename(full_path)
        
    def download_as_bytes(self):
        """Retorna o conteúdo do arquivo como bytes."""
        try:
            with open(self.full_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Erro ao ler o arquivo {self.full_path}: {str(e)}")
            return b""

class LocalBucket:
    """Simula um bucket usando o sistema de arquivos local."""
    
    def __init__(self, base_path):
        self.base_path = base_path
        
    def blob(self, path):
        """Retorna um blob para o caminho especificado."""
        full_path = os.path.join(self.base_path, path)
        return LocalBlob(full_path, path)
    
    def list_blobs(self, prefix=""):
        """Lista todos os arquivos em um diretório."""
        full_prefix_path = os.path.join(self.base_path, prefix)
        blobs = []
        
        if not os.path.exists(full_prefix_path):
            return blobs
        
        # Listar apenas arquivos no diretório principal (sem percorrer subdiretórios)
        if os.path.isdir(full_prefix_path):
            # Listar apenas arquivos no diretório principal, não em subdiretórios
            for filename in os.listdir(full_prefix_path):
                full_path = os.path.join(full_prefix_path, filename)
                # Pular diretórios
                if os.path.isdir(full_path):
                    continue
                # Incluir apenas arquivos
                rel_path = os.path.relpath(full_path, self.base_path)
                blobs.append(LocalBlob(full_path, rel_path))
        
        return blobs

# Funções de utilidade
def find_email_column(df):
    """Encontra a coluna que contém emails em um DataFrame."""
    email_patterns = ['email', 'e-mail', 'correo', '@', 'mail']
    for col in df.columns:
        if any(pattern in col.lower() for pattern in email_patterns):
            return col
    return None

def extract_launch_id(filename):
    """Extrai o ID de lançamento de um nome de arquivo."""
    import re
    patterns = [
        r'L(\d+)[_\s\-]',  # L16_, L16-, L16 
        r'[_\s\-]L(\d+)',  # _L16, -L16, L16
        r'L(\d+)\.csv',    # L16.csv
        r'L(\d+)\.xls',    # L16.xls
        r'L(\d+)$'         # termina com L16
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            launch_num = match.group(1)
            return f"L{launch_num}"
    
    return None

def list_files_by_extension(bucket, prefix="", extensions=(".xlsx", ".xls", ".csv")):
    """Lista arquivos com extensões específicas."""
    blobs = list(bucket.list_blobs(prefix=prefix))
    file_paths = [blob.name for blob in blobs if any(blob.name.lower().endswith(ext.lower()) for ext in extensions)]
    return file_paths

def categorize_files(file_paths):
    """Categoriza arquivos por tipo e lançamento."""
    survey_files_by_launch = {f"L{i}": [] for i in range(16, 22)}
    buyer_files_by_launch = {f"L{i}": [] for i in range(16, 22)}
    utm_files_by_launch = {f"L{i}": [] for i in range(16, 22)}
    
    for file_path in file_paths:
        # Determinar o tipo de arquivo
        file_path_lower = file_path.lower()
        file_type = None
        
        if any(keyword in file_path_lower for keyword in ['pesquisa', 'survey', 'respuestas', 'ayudame']):
            file_type = "survey"
        elif any(keyword in file_path_lower for keyword in ['comprador', 'mario']):
            file_type = "buyer"
        elif any(keyword in file_path_lower for keyword in ['utm']):
            file_type = "utm"
        
        # Identificar o lançamento
        launch_id = extract_launch_id(file_path)
        if launch_id:
            if launch_id in survey_files_by_launch and file_type == "survey":
                survey_files_by_launch[launch_id].append(file_path)
            elif launch_id in buyer_files_by_launch and file_type == "buyer":
                buyer_files_by_launch[launch_id].append(file_path)
            elif launch_id in utm_files_by_launch and file_type == "utm":
                utm_files_by_launch[launch_id].append(file_path)
    
    # Selecionar apenas um arquivo por tipo e lançamento
    survey_files = []
    buyer_files = []
    utm_files = []
    
    for launch in survey_files_by_launch:
        if survey_files_by_launch[launch]:
            survey_files.append(survey_files_by_launch[launch][0])
    
    for launch in buyer_files_by_launch:
        if buyer_files_by_launch[launch]:
            buyer_files.append(buyer_files_by_launch[launch][0])
    
    for launch in utm_files_by_launch:
        if utm_files_by_launch[launch]:
            utm_files.append(utm_files_by_launch[launch][0])
    
    return survey_files, buyer_files, utm_files

def load_csv_or_excel(bucket, file_path):
    """Carrega um arquivo CSV ou Excel."""
    import io
    try:
        blob = bucket.blob(file_path)
        content = blob.download_as_bytes()
        
        if file_path.lower().endswith('.csv'):
            return pd.read_csv(io.BytesIO(content))
        else:  # Excel
            return pd.read_excel(io.BytesIO(content), engine='openpyxl')
    except Exception as e:
        print(f"  - Error loading {file_path}: {str(e)}")
        return None

def load_csv_with_auto_delimiter(bucket, file_path):
    """Carrega um arquivo CSV com detecção automática de delimitador."""
    import io
    try:
        blob = bucket.blob(file_path)
        content = blob.download_as_bytes()
        
        # Detectar o delimitador do arquivo
        try:
            content_str = content.decode('utf-8')
            test_lines = content_str.split('\n')[:10]  # Primeiras 10 linhas
            
            # Contar ocorrências de delimitadores potenciais na primeira linha
            comma_count = test_lines[0].count(',')
            semicolon_count = test_lines[0].count(';')
            
            # Usar o delimitador que aparece mais vezes
            delimiter = ';' if semicolon_count > comma_count else ','
            print(f"  - Detected delimiter '{delimiter}' for {file_path}")
                
            # Processar o arquivo com o delimitador correto
            df = pd.read_csv(
                io.BytesIO(content),
                sep=delimiter,
                encoding='utf-8',
                on_bad_lines='skip',
                low_memory=False
            )
            
            # Verificação pós-carregamento - Se só temos 1-2 colunas, algo deu errado
            if df.shape[1] <= 2:
                print(f"  - Warning: Only {df.shape[1]} columns detected. Trying alternative delimiter...")
                
                # Tentar o delimitador oposto
                alt_delimiter = ';' if delimiter == ',' else ','
                df = pd.read_csv(
                    io.BytesIO(content),
                    sep=alt_delimiter,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    low_memory=False
                )
                
                print(f"  - After using delimiter '{alt_delimiter}': {df.shape[1]} columns detected")
            
            return df
            
        except UnicodeDecodeError:
            # Tentar outra codificação se utf-8 falhar
            print(f"  - Unicode error. Trying latin-1 encoding for {file_path}")
            content_str = content.decode('latin-1')
            test_lines = content_str.split('\n')[:10]
            
            # Detectar delimitador novamente
            comma_count = test_lines[0].count(',')
            semicolon_count = test_lines[0].count(';')
            delimiter = ';' if semicolon_count > comma_count else ','
            
            df = pd.read_csv(
                io.BytesIO(content),
                sep=delimiter,
                encoding='latin-1',
                on_bad_lines='skip',
                low_memory=False
            )
            
            return df
            
    except Exception as e:
        print(f"  - Error loading {file_path}: {str(e)}")
        return None

def process_survey_file(bucket, file_path):
    """Processa um arquivo de pesquisa."""
    df = load_csv_or_excel(bucket, file_path)
    if df is None:
        return None
        
    # Identificar o lançamento deste arquivo
    launch_id = extract_launch_id(file_path)
    
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
    """Carrega todos os arquivos de pesquisa."""
    survey_dfs = []
    launch_data = {}
    
    print("\nLoading survey files...")
    for file_path in survey_files:
        try:
            df = process_survey_file(bucket, file_path)
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

# Modificamos esta função para aceitar o caminho do output_dir como parâmetro
def custom_split_data(df, output_dir, test_size=0.3, val_size=0.5, stratify=True, random_state=42):
    """
    Divide os dados em conjuntos de treino, validação e teste e salva em output_dir.
    
    Args:
        df: DataFrame a ser dividido
        output_dir: Diretório para salvar os conjuntos
        test_size: Proporção do conjunto de teste
        val_size: Proporção do conjunto de validação dentro do conjunto de teste
        stratify: Se deve estratificar por classe
        random_state: Semente aleatória para reprodutibilidade
        
    Returns:
        Tuple com (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    print(f"Splitting data and saving to: {output_dir}")
    
    # Verificar se temos dados suficientes para dividir
    if df.shape[0] == 0:
        print("WARNING: Empty dataset - cannot split. Creating empty dataframes.")
        empty_df = pd.DataFrame(columns=df.columns)
        return empty_df, empty_df, empty_df
    
    # Criar os diretórios se não existirem
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar se temos target para estratificar
    if stratify and 'target' in df.columns and df['target'].nunique() > 1:
        print("Using stratified split based on target variable")
        strat_col = df['target']
    else:
        print("Using random split (no stratification)")
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
    print(f"Train set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    
    print(f"Validation set: {val_df.shape[0]} rows, {val_df.shape[1]} columns")
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    
    print(f"Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    return train_df, val_df, test_df

def main():
    """Pipeline principal para coleta e integração de dados."""
    # Verificar se o diretório de dados existe
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: O diretório de dados não existe: {DATA_DIR}")
        print("Por favor, verifique o caminho e tente novamente.")
        sys.exit(1)
    
    # 1. Conectar ao armazenamento local usando o caminho definido manualmente
    print(f"Setting up connection to local storage...")
    print(f"Using data directory: {DATA_DIR}")
    bucket = LocalBucket(DATA_DIR)
    
    # 2. Listar e categorizar arquivos
    file_paths = list_files_by_extension(bucket, prefix="")
    print(f"Found {len(file_paths)} files")
    
    # 3. Categorizar arquivos por tipo e lançamento e selecionar apenas um por tipo
    print("\nCategorizing files by type and launch...")
    survey_files, buyer_files, utm_files = categorize_files(file_paths)
    
    # Mostrar informações de categorização
    print(f"Survey files (1 per launch): {len(survey_files)}")
    for file in survey_files:
        launch_id = extract_launch_id(file)
        print(f"  - {file} ({launch_id if launch_id else ''})")
        
    print(f"Buyer files (1 per launch): {len(buyer_files)}")
    for file in buyer_files:
        launch_id = extract_launch_id(file)
        print(f"  - {file} ({launch_id if launch_id else ''})")
        
    print(f"UTM files (1 per launch): {len(utm_files)}")
    for file in utm_files:
        launch_id = extract_launch_id(file)
        print(f"  - {file} ({launch_id if launch_id else ''})")
    
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
    
    # 10. Mesclar datasets
    merged_data = merge_datasets(surveys_with_target, utms, buyers)
    
    # 11. Estatísticas de lançamento
    if 'lançamento' in merged_data.columns:
        launch_counts = merged_data['lançamento'].value_counts(dropna=False)
        print("\nRegistros por lançamento:")
        for launch, count in launch_counts.items():
            launch_str = "Sem lançamento identificado" if pd.isna(launch) else launch
            print(f"  - {launch_str}: {count} registros")
    
    # 12. Split dos dados para evitar vazamento
    # Criar os diretórios de saída se não existirem
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Verificar se há dados antes de fazer o split
    if merged_data.shape[0] > 0:
        # Verificar número de classes para stratify
        if 'target' in merged_data.columns and merged_data['target'].nunique() > 1:
            print("Performing stratified split")
            train_df, val_df, test_df = custom_split_data(merged_data, OUTPUT_DIR, stratify=True)
        else:
            print("Performing random split (no stratification possible)")
            train_df, val_df, test_df = custom_split_data(merged_data, OUTPUT_DIR, stratify=False)
        
        print("\nData collection and integration completed!")
        print(f"Data saved to: {OUTPUT_DIR}")
    else:
        print("\nWARNING: No data was merged. Check if:")
        print("1. You have installed openpyxl for Excel files: pip install openpyxl")
        print("2. The files can be properly loaded")
        print("3. The data files have expected structure with email columns")
    
if __name__ == "__main__":
    main()