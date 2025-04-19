"""
Módulo de configuração para o projeto.
Fornece funções para carregar e acessar configurações do projeto.
"""

import os
import json
import yaml
from datetime import datetime
from typing import Dict, Any, Optional

# Caminho padrão para o arquivo de configuração
DEFAULT_CONFIG_PATH = 'config/config.yaml'

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carrega as configurações do projeto a partir de um arquivo YAML.
    
    Args:
        config_path: Caminho para o arquivo de configuração. Se None, usa o caminho padrão.
        
    Returns:
        Dicionário com as configurações carregadas
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    config = {}  # Inicializa como dicionário vazio em vez de None
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(file) or {}  # Garante que não seja None
                elif config_path.endswith('.json'):
                    config = json.load(file) or {}  # Garante que não seja None
                else:
                    print(f"Formato de arquivo de configuração não suportado: {config_path}")
        else:
            print(f"Arquivo de configuração não encontrado: {config_path}, usando configurações padrão")
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}, usando configurações padrão")
    
    # Garantir que as configurações de diretórios existam
    if 'data_directories' not in config:
        config['data_directories'] = {}
    
    # Configurar diretórios padrão se não existirem
    default_dirs = {
        'raw_data': 'data/raw',
        'processed_data': 'data/processed',
        'model_data': 'data/model',
        'output_data': 'data/output'
    }
    
    for key, default_value in default_dirs.items():
        if key not in config['data_directories']:
            config['data_directories'][key] = default_value
    
    return config

def get_project_root() -> str:
    """
    Retorna o diretório raiz do projeto.
    
    Returns:
        Caminho absoluto para o diretório raiz do projeto
    """
    # Assumindo que este arquivo está em src/utils/config.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Voltar dois níveis (src/utils -> src -> raiz)
    return os.path.abspath(os.path.join(current_dir, '..', '..'))