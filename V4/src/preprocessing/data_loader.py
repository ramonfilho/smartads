import pandas as pd
from ..utils.cloud_storage import load_csv_or_excel, load_csv_with_auto_delimiter, extract_launch_id

def find_email_column(df):
    """Encontra a coluna que contém emails em um DataFrame.
    
    Args:
        df: DataFrame a ser analisado
        
    Returns:
        Nome da coluna de email encontrada ou None
    """
    email_patterns = ['email', 'e-mail', 'correo', '@', 'mail']
    for col in df.columns:
        if any(pattern in col.lower() for pattern in email_patterns):
            return col
    return None

def process_survey_file(bucket, file_path):
    """Processa um arquivo de pesquisa.
    
    Args:
        bucket: Objeto bucket do GCS
        file_path: Caminho do arquivo
        
    Returns:
        DataFrame processado ou None em caso de erro
    """
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
    """Processa um arquivo de compradores.
    
    Args:
        bucket: Objeto bucket do GCS
        file_path: Caminho do arquivo
        
    Returns:
        DataFrame processado ou None em caso de erro
    """
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
    """Processa um arquivo de UTM.
    
    Args:
        bucket: Objeto bucket do GCS
        file_path: Caminho do arquivo
        
    Returns:
        DataFrame processado ou None em caso de erro
    """
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
    """Carrega todos os arquivos de pesquisa.
    
    Args:
        bucket: Objeto bucket do GCS
        survey_files: Lista de caminhos de arquivos de pesquisa
        
    Returns:
        Lista de DataFrames carregados
    """
    survey_dfs = []
    launch_data = {}
    
    print("\nLoading survey files...")
    for file_path in survey_files:
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
    
    return survey_dfs, launch_data

def load_buyer_files(bucket, buyer_files):
    """Carrega todos os arquivos de compradores.
    
    Args:
        bucket: Objeto bucket do GCS
        buyer_files: Lista de caminhos de arquivos de compradores
        
    Returns:
        Lista de DataFrames carregados
    """
    buyer_dfs = []
    launch_data = {}
    
    print("\nLoading buyer files...")
    for file_path in buyer_files:
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
    
    return buyer_dfs, launch_data

def load_utm_files(bucket, utm_files):
    """Carrega todos os arquivos de UTM.
    
    Args:
        bucket: Objeto bucket do GCS
        utm_files: Lista de caminhos de arquivos de UTM
        
    Returns:
        Lista de DataFrames carregados
    """
    utm_dfs = []
    launch_data = {}
    
    print("\nLoading UTM files...")
    for file_path in utm_files:
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
    
    return utm_dfs, launch_data