#!/usr/bin/env python
"""
Script para diagnosticar e identificar colunas de texto nos datasets.
Analisa os datasets processados e identifica quais colunas são de texto livre.
"""

import pandas as pd
import os
import sys

# Configurar projeto
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, PROJECT_ROOT)

# Diretórios - USANDO OS NOVOS CAMINHOS
INPUT_DIR = os.path.join(PROJECT_ROOT, "data/new/01_split")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/new/02_processed")

def analyze_text_columns(df, dataset_name):
    """Analisa e identifica colunas de texto em um DataFrame."""
    print(f"\n=== Analisando {dataset_name} ===")
    print(f"Dimensões: {df.shape}")
    
    # Colunas de texto conhecidas (originais)
    known_text_cols = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
        'Déjame un mensaje',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?',
        '¿Qué esperas aprender en el evento Cero a Inglés Fluido?'  # Possível coluna normalizada
    ]
    
    # Padrões para identificar colunas de texto
    text_patterns = [
        'mensaje', 'esperas', 'qué', '¿Qué', 'Cuando', 'vida', 'oportunidad',
        'aprender', 'inglés', 'fluido', 'fluidez', 'Déjame', 'cambiará',
        'Semana', 'Inmersión', 'evento'
    ]
    
    print("\n1. Colunas conhecidas encontradas:")
    found_known = []
    for col in known_text_cols:
        if col in df.columns:
            print(f"  ✓ {col}")
            found_known.append(col)
    
    print(f"\nTotal de colunas conhecidas encontradas: {len(found_known)}")
    
    # Buscar colunas com padrões de texto
    print("\n2. Colunas com padrões de texto (possíveis colunas de texto):")
    pattern_matches = []
    for col in df.columns:
        if any(pattern in col for pattern in text_patterns):
            if col not in found_known:  # Evitar duplicatas
                print(f"  - {col}")
                pattern_matches.append(col)
    
    # Verificar colunas do tipo object
    print("\n3. Colunas do tipo 'object' (possíveis textos):")
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Filtrar colunas que parecem ser texto livre
    text_candidates = []
    for col in object_cols:
        if col in df.columns and col not in ['email', 'email_norm', 'target']:
            # Verificar se tem características de texto livre
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                avg_length = sample.astype(str).str.len().mean()
                if avg_length > 20:  # Textos com mais de 20 caracteres em média
                    text_candidates.append(col)
                    print(f"  - {col} (comprimento médio: {avg_length:.0f})")
    
    # Verificar colunas com sufixo _original
    print("\n4. Colunas com sufixo '_original' (textos preservados):")
    original_cols = [col for col in df.columns if col.endswith('_original')]
    for col in original_cols:
        print(f"  - {col}")
    
    # Resumo final
    all_text_cols = list(set(found_known + pattern_matches + text_candidates))
    
    print(f"\n=== RESUMO para {dataset_name} ===")
    print(f"Total de possíveis colunas de texto: {len(all_text_cols)}")
    print("\nColunas de texto identificadas:")
    for i, col in enumerate(all_text_cols, 1):
        print(f"{i}. {col}")
    
    return all_text_cols

def main():
    """Função principal para diagnóstico."""
    print("=== DIAGNÓSTICO DE COLUNAS DE TEXTO ===")
    
    # Verificar datasets originais primeiro
    print("\n>>> ANALISANDO DATASETS ORIGINAIS (01_split) <<<")
    try:
        for dataset in ['train', 'validation', 'test']:
            df = pd.read_csv(os.path.join(INPUT_DIR, f"{dataset}.csv"), nrows=100)
            text_cols = analyze_text_columns(df, f"Original {dataset}")
    except Exception as e:
        print(f"Erro ao carregar datasets originais: {e}")
    
    # Verificar datasets processados
    print("\n\n>>> ANALISANDO DATASETS PROCESSADOS (02_processed) <<<")
    try:
        for dataset in ['train', 'validation', 'test']:
            df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{dataset}.csv"), nrows=100)
            text_cols = analyze_text_columns(df, f"Processado {dataset}")
    except Exception as e:
        print(f"Erro ao carregar datasets processados: {e}")
    
    # Análise do módulo text_processing
    print("\n\n>>> ANALISANDO MÓDULO text_processing.py <<<")
    try:
        from src.preprocessing.text_processing import text_feature_engineering
        import inspect
        
        # Obter o código fonte da função
        source = inspect.getsource(text_feature_engineering)
        
        # Buscar definição de text_cols
        import re
        text_cols_pattern = r'text_cols\s*=\s*\[(.*?)\]'
        match = re.search(text_cols_pattern, source, re.DOTALL)
        
        if match:
            print("Colunas de texto definidas no módulo:")
            cols_str = match.group(1)
            # Extrair strings entre aspas
            cols = re.findall(r"'([^']+)'|\"([^\"]+)\"", cols_str)
            for col in cols:
                col_name = col[0] if col[0] else col[1]
                print(f"  - {col_name}")
    except Exception as e:
        print(f"Erro ao analisar módulo: {e}")
    
    # Verificar column_normalization se existir
    print("\n\n>>> VERIFICANDO NORMALIZAÇÕES DE COLUNAS <<<")
    try:
        from src.preprocessing.column_normalization import COLUMN_MAPPINGS
        
        print("Mapeamentos de normalização encontrados:")
        for old_name, new_name in COLUMN_MAPPINGS.items():
            if 'esperas' in old_name or 'esperas' in new_name:
                print(f"  {old_name} → {new_name}")
    except Exception as e:
        print(f"Módulo column_normalization não encontrado ou erro: {e}")

if __name__ == "__main__":
    main()