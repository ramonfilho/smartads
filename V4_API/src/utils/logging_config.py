"""
Módulo para configuração de logging do projeto.
"""

import logging
import os
from typing import Optional

def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> None:
    """
    Configura o sistema de logging para o projeto.
    
    Args:
        log_file: Caminho para o arquivo de log. Se None, logs serão enviados somente para console.
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Converter string de nível para constante do logging
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configuração básica
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Se um arquivo de log foi especificado, adicionar handler para arquivo
    if log_file:
        # Criar diretório para o arquivo de log, se não existir
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Adicionar handler para arquivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            '%Y-%m-%d %H:%M:%S'
        ))
        
        # Adicionar handler ao logger raiz
        logging.getLogger().addHandler(file_handler)