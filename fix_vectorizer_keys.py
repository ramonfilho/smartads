#!/usr/bin/env python
"""
Script para corrigir o problema de compatibilidade entre as chaves dos vetorizadores TF-IDF
e modelos LDA.

Este script salva uma versão corrigida dos arquivos que mapeia as novas chaves para as antigas.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import copy

# Adicionar diretório do projeto ao path
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.append(PROJECT_ROOT)

# Configuração de caminhos
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(DATA_DIR, "02_2_reprocessed/models")
FIXED_MODELS_DIR = os.path.join(DATA_DIR, "fixed_models")

def create_key_mapping():
    """
    Cria um mapeamento entre as chaves nos diferentes formatos
    
    Returns:
        Dict com mapeamento entre formatos de chaves
    """
    # Colunas originais de texto
    text_columns = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?', 
        'Déjame un mensaje',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
    ]
    
    # Mapeamentos usados pelo transformador
    transformer_mapping = {
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?': 'cuando_hables_inglés_con_fluid',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?': 'qué_esperas_aprender_en_la',
        'Déjame un mensaje': 'déjame_un_mensaje',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?': 'qué_esperas_aprender_en_la'
    }
    
    # Tentar encontrar os vetorizadores
    vectorizers_path = os.path.join(MODELS_DIR, "tfidf_vectorizers.joblib")
    if os.path.exists(vectorizers_path):
        vectorizers = joblib.load(vectorizers_path)
        print(f"Vetorizadores carregados: {list(vectorizers.keys())}")
    else:
        vectorizers = {}
        print(f"Arquivo de vetorizadores não encontrado: {vectorizers_path}")
    
    # Criar mapeamento
    mapping = {}
    reverse_mapping = {}
    
    # Para cada coluna
    for col in text_columns:
        # Formato do transformador
        transformer_key = transformer_mapping.get(col)
        
        # Formatos possíveis do modelo
        possible_keys = [
            col.replace(' ', '_').replace('?', '').replace('¿', '')[:30],
            col.replace(' ', '_').replace('?', '').replace('¿', '').capitalize()[:30],
            col.replace(' ', '_').replace('?', '').replace('¿', '').title()[:30]
        ]
        
        # Para colunas de Inmersión, tentar variante com Sem
        if "Inmersión" in col:
            column_base = col.split("Inmersión")[0]
            sem_key = f"{column_base.strip()}Semana".replace(' ', '_').replace('?', '').replace('¿', '')[:30].capitalize()
            possible_keys.append(sem_key)
        
        # Encontrar quais destas chaves existem nos vetorizadores
        for key in possible_keys:
            if key in vectorizers:
                # Encontrou uma chave que existe!
                mapping[transformer_key] = key
                reverse_mapping[key] = transformer_key
                print(f"Mapeamento: '{transformer_key}' -> '{key}'")
                break
    
    return mapping, reverse_mapping

def fix_tfidf_vectorizers(mapping):
    """
    Corrige os vetorizadores TF-IDF para usar as chaves do transformador
    
    Args:
        mapping: Dicionário com mapeamento de chaves
    """
    print("\n===== Corrigindo vetorizadores TF-IDF =====")
    vectorizers_path = os.path.join(MODELS_DIR, "tfidf_vectorizers.joblib")
    
    if not os.path.exists(vectorizers_path):
        print(f"Arquivo de vetorizadores não encontrado: {vectorizers_path}")
        return
    
    # Carregar vetorizadores
    vectorizers = joblib.load(vectorizers_path)
    print(f"Vetorizadores originais: {list(vectorizers.keys())}")
    
    # Criar cópia com as chaves mapeadas
    fixed_vectorizers = {}
    
    # Para cada vetorizador
    for old_key, vectorizer in vectorizers.items():
        # Tentar encontrar a chave no mapeamento reverso
        if old_key in mapping.values():
            for new_key, mapped_key in mapping.items():
                if mapped_key == old_key:
                    # Adicionar com a nova chave
                    fixed_vectorizers[new_key] = vectorizer
                    print(f"  Mapeando '{old_key}' para '{new_key}'")
        else:
            # Manter a chave original
            fixed_vectorizers[old_key] = vectorizer
            print(f"  Mantendo chave original: '{old_key}'")
    
    # Salvar vetorizadores corrigidos
    os.makedirs(FIXED_MODELS_DIR, exist_ok=True)
    fixed_path = os.path.join(FIXED_MODELS_DIR, "tfidf_vectorizers.joblib")
    joblib.dump(fixed_vectorizers, fixed_path)
    print(f"Vetorizadores corrigidos salvos em: {fixed_path}")
    print(f"Novas chaves: {list(fixed_vectorizers.keys())}")

def fix_lda_models(mapping):
    """
    Corrige os modelos LDA para usar as chaves do transformador
    
    Args:
        mapping: Dicionário com mapeamento de chaves
    """
    print("\n===== Corrigindo modelos LDA =====")
    lda_models_path = os.path.join(MODELS_DIR, "lda_models.joblib")
    
    if not os.path.exists(lda_models_path):
        print(f"Arquivo de modelos LDA não encontrado: {lda_models_path}")
        return
    
    # Carregar modelos LDA
    lda_models = joblib.load(lda_models_path)
    print(f"Modelos LDA originais: {list(lda_models.keys())}")
    
    # Criar cópia com as chaves mapeadas
    fixed_lda_models = {}
    
    # Para cada modelo LDA
    for old_key, model_info in lda_models.items():
        # Tentar encontrar a chave no mapeamento reverso
        if old_key in mapping.values():
            for new_key, mapped_key in mapping.items():
                if mapped_key == old_key:
                    # Adicionar com a nova chave
                    fixed_lda_models[new_key] = model_info
                    print(f"  Mapeando '{old_key}' para '{new_key}'")
        else:
            # Manter a chave original
            fixed_lda_models[old_key] = model_info
            print(f"  Mantendo chave original: '{old_key}'")
    
    # Salvar modelos LDA corrigidos
    os.makedirs(FIXED_MODELS_DIR, exist_ok=True)
    fixed_path = os.path.join(FIXED_MODELS_DIR, "lda_models.joblib")
    joblib.dump(fixed_lda_models, fixed_path)
    print(f"Modelos LDA corrigidos salvos em: {fixed_path}")
    print(f"Novas chaves: {list(fixed_lda_models.keys())}")

def update_transformer_paths():
    """
    Cria um script de atualização para apontar para os modelos corrigidos
    """
    print("\n===== Criando script de atualização para o transformador =====")
    
    update_script = """#!/usr/bin/env python
\"\"\"
Script para atualizar o TextFeatureEngineeringTransformer para usar os modelos corrigidos.
\"\"\"

import os
import sys

# Adicionar diretório do projeto ao path
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.append(PROJECT_ROOT)

# Atualizar o método fit() do transformador
from src.inference.TextFeatureEngineeringTransformer import TextFeatureEngineeringTransformer

def update_transformer():
    \"\"\"
    Modifica o código do transformador para usar os modelos corrigidos.
    \"\"\"
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
"""
    
    # Salvar o script
    script_path = os.path.join(PROJECT_ROOT, "update_transformer.py")
    with open(script_path, 'w') as f:
        f.write(update_script)
    
    print(f"Script de atualização criado em: {script_path}")
    print("Execute este script para atualizar o transformador para usar os modelos corrigidos.")

def main():
    """
    Função principal
    """
    print("=" * 60)
    print("CORRIGINDO COMPATIBILIDADE DE CHAVES PARA MODELOS")
    print("=" * 60)
    
    try:
        # Criar mapeamento de chaves
        mapping, reverse_mapping = create_key_mapping()
        
        # Corrigir vetorizadores TF-IDF
        fix_tfidf_vectorizers(mapping)
        
        # Corrigir modelos LDA
        fix_lda_models(mapping)
        
        # Criar script de atualização para o transformador
        update_transformer_paths()
        
        print("\n===== PROCESSO CONCLUÍDO COM SUCESSO =====")
        print(f"Modelos corrigidos salvos em: {FIXED_MODELS_DIR}")
        print("Execute o script update_transformer.py para atualizar o transformador.")
        
    except Exception as e:
        print(f"\nERRO durante o processo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()