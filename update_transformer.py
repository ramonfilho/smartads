#!/usr/bin/env python
"""
Script para atualizar o TextFeatureEngineeringTransformer para usar os modelos corrigidos.
"""

import os
import sys

# Adicionar diretório do projeto ao path
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.append(PROJECT_ROOT)

# Atualizar o método fit() do transformador
from src.inference.TextFeatureEngineeringTransformer import TextFeatureEngineeringTransformer

def update_transformer():
    """
    Modifica o código do transformador para usar os modelos corrigidos.
    """
    # Caminho para o arquivo do transformador
    transformer_path = os.path.join(PROJECT_ROOT, "src/inference/TextFeatureEngineeringTransformer.py")
    
    # Ler o conteúdo atual
    with open(transformer_path, 'r') as f:
        content = f.read()
    
    # Modificar o caminho dos modelos
    new_content = content.replace(
        'models_dir = "/Users/ramonmoreira/desktop/smart_ads/data/02_2_reprocessed/models"',
        'models_dir = "/Users/ramonmoreira/desktop/smart_ads/data/fixed_models"'
    )
    
    # Salvar o arquivo modificado
    with open(transformer_path, 'w') as f:
        f.write(new_content)
    
    print(f"Transformador atualizado para usar os modelos corrigidos!")

if __name__ == "__main__":
    update_transformer()
