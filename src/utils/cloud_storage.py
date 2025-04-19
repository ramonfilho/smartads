from google.cloud import storage
import pandas as pd
import io
import re

def connect_to_gcs(bucket_name):
    """Estabelece conexão com o Google Cloud Storage.
    
    Args:
        bucket_name: Nome do bucket a ser conectado
        
    Returns:
        Objeto bucket do GCS
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    return bucket

def list_files_by_extension(bucket, prefix="", extensions=(".xlsx", ".xls", ".csv")):
    """Lista arquivos no bucket com extensões específicas.
    
    Args:
        bucket: Objeto bucket do GCS
        prefix: Prefixo para filtrar a busca
        extensions: Tuple de extensões a serem filtradas
        
    Returns:
        Lista de caminhos dos arquivos
    """
    blobs = list(bucket.list_blobs(prefix=prefix))
    file_paths = [blob.name for blob in blobs if blob.name.endswith(extensions)]
    return file_paths

def extract_launch_id(filename):
    """Extrai o ID de lançamento de um nome de arquivo.
    
    Args:
        filename: Nome do arquivo para extrair o ID
        
    Returns:
        ID de lançamento no formato L{número} ou None se não encontrado
    """
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

def categorize_files(file_paths):
    """Categoriza arquivos por tipo e lançamento.
    
    Args:
        file_paths: Lista de caminhos de arquivos para categorizar
        
    Returns:
        Tuple com listas de categorias e dicionário por lançamento
    """
    survey_files = []
    buyer_files = []
    utm_files = []
    # Criar dicionário para lançamentos de L16 a L21
    all_files_by_launch = {f"L{i}": [] for i in range(16, 22)}
    
    for file_path in file_paths:
        # Determinar o tipo de arquivo
        if any(keyword in file_path.lower() for keyword in ['pesquisa', 'survey', 'respuestas', 'ayudame']):
            survey_files.append(file_path)
        elif any(keyword in file_path.lower() for keyword in ['comprador', 'mario']):
            buyer_files.append(file_path)
        elif any(keyword in file_path.lower() for keyword in ['utm']):
            utm_files.append(file_path)
        
        # Identificar o lançamento
        launch_id = extract_launch_id(file_path)
        if launch_id and launch_id in all_files_by_launch:
            all_files_by_launch[launch_id].append(file_path)
    
    return survey_files, buyer_files, utm_files, all_files_by_launch

def load_csv_or_excel(bucket, file_path):
    """Carrega um arquivo CSV ou Excel do Google Cloud Storage.
    
    Args:
        bucket: Objeto bucket do GCS
        file_path: Caminho do arquivo no bucket
        
    Returns:
        DataFrame pandas com o conteúdo do arquivo ou None em caso de erro
    """
    try:
        blob = bucket.blob(file_path)
        content = blob.download_as_bytes()
        
        if file_path.endswith('.csv'):
            return pd.read_csv(io.BytesIO(content))
        else:  # Excel
            return pd.read_excel(io.BytesIO(content), engine='openpyxl')
    except Exception as e:
        print(f"  - Error loading {file_path}: {str(e)}")
        return None

def load_csv_with_auto_delimiter(bucket, file_path):
    """Carrega um arquivo CSV com detecção automática de delimitador.
    
    Args:
        bucket: Objeto bucket do GCS
        file_path: Caminho do arquivo no bucket
        
    Returns:
        DataFrame pandas com o conteúdo do arquivo ou None em caso de erro
    """
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