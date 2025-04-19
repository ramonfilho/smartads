"""
Módulo para manipulação de arquivos no armazenamento local.
Este módulo fornece funções equivalentes às funções GCS, mas para sistemas de arquivos locais.
"""

import os
import pandas as pd
import json
import pickle
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