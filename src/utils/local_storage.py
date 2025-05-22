"""
Módulo para manipulação de arquivos no armazenamento local.
Este módulo fornece funções equivalentes às funções GCS, mas para sistemas de arquivos locais.
"""

import os
import pandas as pd
import json
import pickle
import re
import io
from typing import List, Dict, Any, Optional, Union


def list_files_by_extension(directory: str, extension: str) -> List[str]:
    """
    Lista todos os arquivos com uma determinada extensão em um diretório.
    
    Args:
        directory: Caminho do diretório a ser pesquisado
        extension: Extensão de arquivo a ser filtrada (ex: '.csv', '.json')
        
    Returns:
        Lista de caminhos completos dos arquivos encontrados
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


def categorize_local_files(data_dir: str) -> Dict[str, List[str]]:
    """
    Categoriza os arquivos locais por tipo de extensão.
    
    Args:
        data_dir: Diretório raiz contendo os dados
        
    Returns:
        Dicionário com extensões como chaves e listas de arquivos como valores
    """
    file_categories = {}
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()
            if extension:
                if extension not in file_categories:
                    file_categories[extension] = []
                file_categories[extension].append(os.path.join(root, filename))
    return file_categories


def load_local_file(file_path: str) -> Any:
    """
    Carrega um arquivo local baseado em sua extensão.
    
    Args:
        file_path: Caminho completo do arquivo a ser carregado
        
    Returns:
        Conteúdo do arquivo carregado no formato apropriado
    
    Raises:
        ValueError: Se a extensão do arquivo não for suportada
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    if extension == '.csv':
        return pd.read_csv(file_path)
    elif extension == '.json':
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    elif extension == '.pickle' or extension == '.pkl':
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    elif extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        raise ValueError(f"Formato de arquivo não suportado: {extension}")


def save_local_file(data: Any, file_path: str) -> None:
    """
    Salva dados em um arquivo local baseado em sua extensão.
    
    Args:
        data: Dados a serem salvos
        file_path: Caminho completo onde o arquivo será salvo
        
    Raises:
        ValueError: Se a extensão do arquivo não for suportada
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    # Cria diretórios se não existirem
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if extension == '.csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        else:
            raise ValueError("Dados devem ser um DataFrame para salvar como CSV")
    elif extension == '.json':
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    elif extension == '.pickle' or extension == '.pkl':
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    elif extension == '.txt':
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(str(data))
    else:
        raise ValueError(f"Formato de arquivo não suportado: {extension}")


def get_filename_without_extension(file_path: str) -> str:
    """
    Extrai o nome do arquivo sem a extensão.
    
    Args:
        file_path: Caminho completo do arquivo
        
    Returns:
        Nome do arquivo sem a extensão
    """
    base_name = os.path.basename(file_path)
    return os.path.splitext(base_name)[0]


# ----- Funções de compatibilidade com GCS -----

class LocalBucket:
    """Simula um bucket do GCS usando o sistema de arquivos local."""
    
    def __init__(self, base_path):
        """
        Inicializa o bucket local.
        
        Args:
            base_path: Caminho base do sistema de arquivos
        """
        self.base_path = base_path
        
    def blob(self, path):
        """
        Retorna um objeto LocalBlob para o caminho fornecido.
        
        Args:
            path: Caminho relativo ao bucket
            
        Returns:
            Um objeto LocalBlob
        """
        return LocalBlob(os.path.join(self.base_path, path))
    
    def list_blobs(self, prefix=""):
        """
        Lista arquivos em um diretório.
        
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
                    # Caminho relativo ao base_path
                    rel_path = os.path.relpath(full_path, self.base_path)
                    blobs.append(LocalBlob(full_path, rel_path))
        
        return blobs


class LocalBlob:
    """Simula um blob do GCS."""
    
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
        Retorna o conteúdo do arquivo como bytes.
        
        Returns:
            Conteúdo do arquivo como bytes
        """
        with open(self.full_path, 'rb') as f:
            return f.read()


def connect_to_gcs(bucket_name):
    """
    Conecta a um "bucket" local.
    
    Args:
        bucket_name: Nome do bucket (ignorado, apenas para compatibilidade)
        
    Returns:
        Objeto LocalBucket
    """
    # CORREÇÃO: Usar o caminho correto para os dados
    base_path = "/Users/ramonmoreira/Desktop/smart_ads/data/00_raw_data"
    print(f"Conectando ao armazenamento local em: {os.path.abspath(base_path)}")
    return LocalBucket(base_path)


def list_files_by_extension(bucket, prefix="", extensions=(".xlsx", ".xls", ".csv")):
    """
    Lista arquivos com extensões específicas.
    
    Args:
        bucket: Objeto bucket
        prefix: Prefixo para filtrar a busca
        extensions: Tupla de extensões a serem filtradas
        
    Returns:
        Lista de caminhos dos arquivos
    """
    blobs = list(bucket.list_blobs(prefix=prefix))
    file_paths = [blob.name for blob in blobs if any(blob.name.endswith(ext) for ext in extensions)]
    return file_paths


def extract_launch_id(filename):
    """
    Extrai o ID de lançamento de um nome de arquivo.
    
    Args:
        filename: Nome do arquivo para extrair o ID
        
    Returns:
        ID de lançamento no formato L{número} ou None se não encontrado
    """
    patterns = [
        r'L(\d+)[_\s\-]',  # L16_, L16-, L16 
        r'[_\s\-]L(\d+)',  # _L16, -L16, L16
        r'L(\d+)\.csv',    # L16.csv
        r'L(\d+)\.xlsx',   # L16.xlsx (ADICIONADO)
        r'L(\d+)\.xls',    # L16.xls
        r'L(\d+)$'         # termina com L16
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            launch_num = match.group(1)
            return f"L{launch_num}"
    
    return None


def prioritize_file_format(file_paths):
    """
    Quando há múltiplos arquivos do mesmo tipo/lançamento, prioriza por formato:
    - Para UTMs: .csv > .xlsx > .xls (CSV é mais confiável para UTMs)
    - Para outros: .xlsx > .xls > .csv (Excel é mais confiável)
    
    Args:
        file_paths: Lista de caminhos de arquivos
        
    Returns:
        Lista de arquivos filtrada (um por tipo/lançamento)
    """
    # Agrupar por tipo e lançamento
    grouped = {}
    
    for file_path in file_paths:
        launch_id = extract_launch_id(file_path)
        file_type = None
        
        # Determinar tipo
        if any(keyword in file_path.lower() for keyword in ['pesquisa', 'survey', 'respuestas']):
            file_type = 'survey'
        elif any(keyword in file_path.lower() for keyword in ['comprador', 'mario']):
            file_type = 'buyer'
        elif any(keyword in file_path.lower() for keyword in ['utm']):
            file_type = 'utm'
            
        if file_type and launch_id:
            key = f"{file_type}_{launch_id}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(file_path)
    
    # Selecionar melhor arquivo de cada grupo
    selected_files = []
    for key, files in grouped.items():
        if len(files) == 1:
            selected_files.append(files[0])
        else:
            # Separar por extensão
            xlsx_files = [f for f in files if f.endswith('.xlsx')]
            xls_files = [f for f in files if f.endswith('.xls')]
            csv_files = [f for f in files if f.endswith('.csv')]
            
            # Determinar tipo do arquivo para aplicar priorização correta
            file_type = key.split('_')[0]
            
            if file_type == 'utm':
                # Para UTMs: priorizar CSV > XLSX > XLS
                if csv_files:
                    selected_files.append(csv_files[0])
                    print(f"  Priorizando .csv para {key}: {os.path.basename(csv_files[0])}")
                elif xlsx_files:
                    selected_files.append(xlsx_files[0])
                    print(f"  Priorizando .xlsx para {key}: {os.path.basename(xlsx_files[0])}")
                elif xls_files:
                    selected_files.append(xls_files[0])
                    print(f"  Priorizando .xls para {key}: {os.path.basename(xls_files[0])}")
            else:
                # Para outros tipos: priorizar XLSX > XLS > CSV
                if xlsx_files:
                    selected_files.append(xlsx_files[0])
                    print(f"  Priorizando .xlsx para {key}: {os.path.basename(xlsx_files[0])}")
                elif xls_files:
                    selected_files.append(xls_files[0])
                    print(f"  Priorizando .xls para {key}: {os.path.basename(xls_files[0])}")
                elif csv_files:
                    selected_files.append(csv_files[0])
                    print(f"  Priorizando .csv para {key}: {os.path.basename(csv_files[0])}")
    
    return selected_files


def categorize_files(file_paths):
    """
    Categoriza arquivos por tipo e lançamento.
    
    Args:
        file_paths: Lista de caminhos de arquivos para categorizar
        
    Returns:
        Tuple com listas de categorias e dicionário por lançamento
    """
    # NOVA FUNCIONALIDADE: Priorizar formatos para evitar duplicatas
    file_paths = prioritize_file_format(file_paths)
    print(f"Após priorização: {len(file_paths)} arquivos únicos")
    
    survey_files = []
    buyer_files = []
    utm_files = []
    # CORREÇÃO: Incluir L22
    all_files_by_launch = {f"L{i}": [] for i in range(16, 23)}  # Agora vai até L22
    
    for file_path in file_paths:
        # Determinar o tipo de arquivo
        if any(keyword in file_path.lower() for keyword in ['pesquisa', 'survey', 'respuestas']):
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
    """
    Carrega um arquivo CSV ou Excel.
    
    Args:
        bucket: Objeto bucket
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
    """
    Carrega um arquivo CSV com detecção automática de delimitador.
    
    Args:
        bucket: Objeto bucket
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