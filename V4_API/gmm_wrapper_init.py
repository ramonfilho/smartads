# V4_API/gmm_wrapper_init.py
"""
Módulo de inicialização para garantir que a classe GMM_Wrapper esteja disponível
para desserialização em todos os contextos da aplicação.
"""

import sys
import os
import builtins

# Adicionar diretório raiz ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Importar a classe GMM_Wrapper
from src.modeling.gmm_wrapper import GMM_Wrapper

# Registrar globalmente para desserialização
builtins.GMM_Wrapper = GMM_Wrapper

print("GMM_Wrapper inicializado e registrado globalmente para desserialização.")