#!/usr/bin/env python
"""
Script para diagnosticar os valores reais da coluna '¿Hace quánto tiempo me conoces?'
e identificar por que time_known_encoded tem 85% de valores ausentes.
"""

import pandas as pd
import os
import sys

# Adicionar o diretório raiz ao path
project_root = "/Users/ramonmoreira/Desktop/smart_ads"
sys.path.insert(0, project_root)

def diagnose_time_known_column(file_path):
    """
    Analisa os valores únicos da coluna de tempo e sua distribuição.
    """
    print(f"\nAnalisando arquivo: {file_path}")
    print("=" * 80)
    
    # Carregar dados
    df = pd.read_csv(file_path)
    
    # Nome da coluna original
    col_name = '¿Hace quánto tiempo me conoces?'
    
    if col_name not in df.columns:
        print(f"ERRO: Coluna '{col_name}' não encontrada!")
        print(f"Colunas disponíveis: {list(df.columns)}")
        return
    
    # Análise dos valores
    print(f"\n1. ANÁLISE DA COLUNA '{col_name}':")
    print(f"   - Total de registros: {len(df):,}")
    print(f"   - Valores não-nulos: {df[col_name].notna().sum():,}")
    print(f"   - Valores nulos: {df[col_name].isna().sum():,}")
    print(f"   - % de nulos: {df[col_name].isna().sum() / len(df) * 100:.2f}%")
    
    # Valores únicos
    print(f"\n2. VALORES ÚNICOS NA COLUNA (com contagem):")
    value_counts = df[col_name].value_counts(dropna=False)
    for value, count in value_counts.items():
        if pd.isna(value):
            print(f"   - NaN: {count:,} ({count/len(df)*100:.1f}%)")
        else:
            # Mostrar valor completo entre aspas para ver espaços/caracteres especiais
            print(f"   - '{value}': {count:,} ({count/len(df)*100:.1f}%)")
    
    # Verificar se existe time_known_encoded
    if 'time_known_encoded' in df.columns:
        print(f"\n3. ANÁLISE DE 'time_known_encoded':")
        print(f"   - Valores não-nulos: {df['time_known_encoded'].notna().sum():,}")
        print(f"   - Valores nulos: {df['time_known_encoded'].isna().sum():,}")
        print(f"   - % de nulos: {df['time_known_encoded'].isna().sum() / len(df) * 100:.2f}%")
        
        # Comparar valores originais com encoded
        print(f"\n4. MAPEAMENTO ATUAL vs VALORES REAIS:")
        
        # Mapeamento atual no código
        current_map = {
            'Te acabo de conocer a través d...': 0,
            'Te sigo desde hace 1 mes': 1,
            'Te sigo desde hace 3 meses': 2,
            'Te sigo desde hace más de 5 me...': 3,
            'Te sigo desde hace 1 año': 4,
            'Te sigo hace más de 1 año': 5,
            'desconhecido': -1
        }
        
        print("\n   Valores no mapeamento atual:")
        for key, val in current_map.items():
            print(f"   - '{key}' -> {val}")
        
        print("\n   Valores reais não mapeados:")
        real_values = df[col_name].dropna().unique()
        for value in real_values:
            # Verificar se alguma chave do mapeamento corresponde ao início do valor
            matched = False
            for map_key in current_map.keys():
                if map_key in str(value) or str(value).startswith(map_key.replace('...', '')):
                    matched = True
                    break
            
            if not matched and value != 'desconhecido':
                print(f"   ❌ '{value}'")
    
    # Sugerir novo mapeamento
    print("\n5. SUGESTÃO DE NOVO MAPEAMENTO:")
    print("```python")
    print("time_map = {")
    for value in df[col_name].dropna().unique():
        if 'acabo de conocer' in str(value).lower():
            print(f"    '{value}': 0,")
        elif '1 mes' in str(value):
            print(f"    '{value}': 1,")
        elif '3 meses' in str(value):
            print(f"    '{value}': 2,")
        elif 'más de 5 me' in str(value) or 'mas de 5 me' in str(value):
            print(f"    '{value}': 3,")
        elif '1 año' in str(value) and 'más' not in str(value) and 'mas' not in str(value):
            print(f"    '{value}': 4,")
        elif ('más de 1 año' in str(value) or 'mas de 1 año' in str(value) or 
              'hace más de 1 año' in str(value) or 'hace mas de 1 año' in str(value)):
            print(f"    '{value}': 5,")
    print("    'desconhecido': -1")
    print("}")
    print("```")

def main():
    """Executa diagnóstico nos datasets."""
    
    # Caminhos dos arquivos
    paths = {
        'raw_train': os.path.join(project_root, "data/new/01_split/train.csv"),
        'processed_train': os.path.join(project_root, "data/new/02_processed/train.csv")
    }
    
    print("DIAGNÓSTICO DO PROBLEMA COM 'time_known_encoded'")
    print("=" * 80)
    
    # Analisar dados brutos primeiro
    if os.path.exists(paths['raw_train']):
        diagnose_time_known_column(paths['raw_train'])
    else:
        print(f"Arquivo não encontrado: {paths['raw_train']}")
    
    # Depois analisar dados processados
    if os.path.exists(paths['processed_train']):
        print("\n\n")
        diagnose_time_known_column(paths['processed_train'])
    else:
        print(f"Arquivo não encontrado: {paths['processed_train']}")

if __name__ == "__main__":
    main()