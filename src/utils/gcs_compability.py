"""
Módulo de compatibilidade para emular funções do Google Cloud Storage com armazenamento local.
Este módulo permite que o código escrito para GCS funcione com arquivos locais.
"""

import os
import pandas as pd
import re
from src.utils.local_storage import list_files_by_extension as list_local_files

class LocalBucket:
    """Classe que emula um bucket do GCS usando armazenamento local."""
    
    def __init__(self, base_path):
        """
        Inicializa um bucket local.
        
        Args:
            base_path: Caminho base para simular um bucket
        """
        self.base_path = base_path
        
    def blob(self, path):
        """
        Retorna um objeto que emula um blob do GCS.
        
        Args:
            path: Caminho relativo ao bucket
            
        Returns:
            Um objeto LocalBlob
        """
        return LocalBlob(os.path.join(self.base_path, path))
    
    def list_blobs(self, prefix=""):
        """
        Lista todos os arquivos em um diretório.
        
        Args:
            prefix: Prefixo para filtrar arquivos
            
        Returns:
            Lista de objetos LocalBlob
        """
        full_prefix_path = os.path.join(self.base_path, prefix)
        blobs = []
        
        if not os.path.exists(full_prefix_path):
            return blobs
            
        if os.path.isdir(full_prefix_path):
            for root, _, files in os.walk(full_prefix_path):
                for file in files:
                    # Caminho completo do arquivo
                    full_path = os.path.join(root, file)
                    # Caminho relativo ao base_path (simulando nome no bucket)
                    rel_path = os.path.relpath(full_path, self.base_path)
                    blobs.append(LocalBlob(full_path, rel_path))
        
        return blobs

class LocalBlob:
    """Classe que emula um blob do GCS."""
    
    def __init__(self, full_path, name=None):
        """
        Inicializa um blob local.
        
        Args:
            full_path: Caminho completo do arquivo
            name: Nome do blob (caminho relativo ao bucket)
        """
        self.full_path = full_path
        self.name = name if name else os.path.basename(full_path)
        
    def download_as_bytes(self):
        """
        Baixa o conteúdo do arquivo como bytes.
        
        Returns:
            Bytes do arquivo
        """
        with open(self.full_path, 'rb') as f:
            return f.read()

def connect_to_gcs(bucket_name):
    """
    Emula a conexão ao GCS retornando um bucket local.
    
    Args:
        bucket_name: Nome do bucket (usado como diretório base)
        
    Returns:
        Um objeto LocalBucket
    """
    # Usar "../data/raw_data" como diretório base
    base_path = "../data/raw_data"
    print(f"Conectando ao armazenamento local em: {os.path.abspath(base_path)}")
    return LocalBucket(base_path)

def list_files_by_extension(bucket, prefix="", extensions=(".xlsx", ".xls", ".csv")):
    """
    Lista arquivos com extensões específicas.
    
    Args:
        bucket: Objeto bucket local
        prefix: Prefixo para filtrar a busca
        extensions: Tuple de extensões a serem filtradas
        
    Returns:
        Lista de caminhos dos arquivos
    """
    blobs = bucket.list_blobs(prefix=prefix)
    file_paths = [blob.name for blob in blobs if any(blob.name.endswith(ext) for ext in extensions)]
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
    """Carrega um arquivo CSV ou Excel.
    
    Args:
        bucket: Objeto bucket local
        file_path: Caminho do arquivo no bucket
        
    Returns:
        DataFrame pandas com o conteúdo do arquivo ou None em caso de erro
    """
    try:
        blob = bucket.blob(file_path)
        content = blob.download_as_bytes()
        
        if file_path.endswith('.csv'):
            return pd.read_csv(pd.io.common.BytesIO(content))
        else:  # Excel
            return pd.read_excel(pd.io.common.BytesIO(content), engine='openpyxl')
    except Exception as e:
        print(f"  - Error loading {file_path}: {str(e)}")
        return None

def load_csv_with_auto_delimiter(bucket, file_path):
    """Carrega um arquivo CSV com detecção automática de delimitador.
    
    Args:
        bucket: Objeto bucket local
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
                pd.io.common.BytesIO(content),
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
                    pd.io.common.BytesIO(content),
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
                pd.io.common.BytesIO(content),
                sep=delimiter,
                encoding='latin-1',
                on_bad_lines='skip',
                low_memory=False
            )
            
            return df
            
    except Exception as e:
        print(f"  - Error loading {file_path}: {str(e)}")
        return None