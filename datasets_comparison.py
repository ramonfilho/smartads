#!/usr/bin/env python
"""
Script para comparar datasets entre diferentes versões de processamento.

Compara os conjuntos de dados nas pastas:
- data/02_1_1_processed
- data/02_1_processed
- data/02_2_processed

Foco nas características básicas: número de features, tipos, valores ausentes, etc.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Caminho base do projeto
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"

# Diretórios a serem comparados
DIRS = [
    os.path.join(PROJECT_ROOT, "data/02_2_processed"),
    os.path.join(PROJECT_ROOT, "data/02_fixed_processed")
]

# Nome amigável para cada diretório
DIR_NAMES = {
    "data/02_fixed_processed": "Versão nova",
    "data/02_2_processed": "Versão anterior"
}

def load_datasets(directory):
    """
    Carrega os datasets de um diretório.
    
    Args:
        directory: Caminho do diretório
        
    Returns:
        Dict com datasets carregados ou None se não encontrados
    """
    datasets = {}
    
    for file_name in ["train.csv", "validation.csv", "test.csv"]:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            try:
                datasets[file_name.split('.')[0]] = pd.read_csv(file_path)
                print(f"  Carregado {file_name} de {directory}")
            except Exception as e:
                print(f"  ERRO ao carregar {file_path}: {e}")
        else:
            print(f"  AVISO: Arquivo {file_path} não encontrado")
    
    return datasets if datasets else None

def analyze_dataset(df):
    """
    Analisa características básicas de um DataFrame.
    
    Args:
        df: DataFrame para análise
        
    Returns:
        Dict com estatísticas do DataFrame
    """
    # Tipos de dados
    dtypes = df.dtypes.value_counts().to_dict()
    
    # Converte tipos numpy para strings para melhor legibilidade
    dtypes = {str(k).split('.')[-1].replace("'>", ""): v for k, v in dtypes.items()}
    
    # Informações sobre valores ausentes
    missing_values = df.isna().sum().sum()
    missing_pct = (missing_values / (df.shape[0] * df.shape[1])) * 100
    
    # Colunas com mais valores ausentes
    top_missing = df.isna().sum()
    top_missing = top_missing[top_missing > 0].sort_values(ascending=False)
    top_missing_dict = {}
    
    # Corrigido o problema com [:5]
    for col, count in list(top_missing.items())[:5]:  # Top 5 colunas com mais valores ausentes
        top_missing_dict[col] = {
            'count': count,
            'pct': (count / df.shape[0]) * 100
        }
    
    # Verificar colunas específicas de texto
    text_cols = [col for col in df.columns if 'mensaje' in col or 'inglés' in col or 'esperas' in col]
    
    # Categorizar colunas por prefixo/sufixo para entender tipos de features
    feature_categories = {
        'tfidf_': 0,
        '_tfidf': 0,
        'text_': 0,
        '_embedding': 0,
        'topic_': 0,
        'sentiment_': 0,
        'count_': 0,
        'interaction_': 0,
        'ratio_': 0,
        'original': 0
    }
    
    for col in df.columns:
        for category in feature_categories:
            if category in col:
                feature_categories[category] += 1
    
    return {
        'shape': df.shape,
        'dtypes': dtypes,
        'missing_values': {
            'total': missing_values,
            'pct': missing_pct,
            'top_cols': top_missing_dict
        },
        'text_cols': len(text_cols),
        'feature_categories': {k: v for k, v in feature_categories.items() if v > 0}
    }

def compare_datasets(dirs=DIRS):
    """
    Compara datasets entre diferentes diretórios.
    
    Args:
        dirs: Lista de diretórios para comparação
        
    Returns:
        Dict com resultados da comparação
    """
    print("\n====== COMPARAÇÃO DE DATASETS ======")
    
    # Verificar existência dos diretórios
    for dir_path in dirs:
        print(f"Verificando diretório: {dir_path}")
        if not os.path.exists(dir_path):
            print(f"  AVISO: Diretório {dir_path} não encontrado!")
    
    # Carregar datasets de cada diretório
    all_datasets = {}
    for dir_path in dirs:
        dir_name = os.path.basename(os.path.dirname(dir_path)) + "/" + os.path.basename(dir_path)
        print(f"\nCarregando datasets de {dir_name}...")
        datasets = load_datasets(dir_path)
        if datasets:
            all_datasets[dir_path] = datasets
    
    if not all_datasets:
        print("ERRO: Nenhum dataset encontrado para comparação!")
        return None
    
    # Iniciar comparação
    comparison = {}
    
    # Para cada tipo de dataset (train, validation, test)
    for dataset_type in ["train", "validation", "test"]:
        print(f"\nAnalisando conjuntos de {dataset_type}...")
        comparison[dataset_type] = {}
        
        # Verificar quais diretórios têm este tipo de dataset
        dirs_with_dataset = [d for d in all_datasets if dataset_type in all_datasets[d]]
        
        if not dirs_with_dataset:
            print(f"  AVISO: Nenhum conjunto de {dataset_type} encontrado!")
            continue
        
        # Analisar cada dataset individualmente
        for dir_path in dirs_with_dataset:
            dir_name = os.path.basename(os.path.dirname(dir_path)) + "/" + os.path.basename(dir_path)
            friendly_name = DIR_NAMES.get(os.path.relpath(dir_path, PROJECT_ROOT), dir_name)
            
            df = all_datasets[dir_path][dataset_type]
            print(f"  Analisando {friendly_name} ({df.shape[0]} linhas, {df.shape[1]} colunas)")
            
            comparison[dataset_type][dir_path] = analyze_dataset(df)
        
        # Comparar colunas entre datasets do mesmo tipo
        if len(dirs_with_dataset) > 1:
            column_comparisons = []
            for i, dir1 in enumerate(dirs_with_dataset[:-1]):
                for dir2 in dirs_with_dataset[i+1:]:
                    dir1_name = os.path.basename(os.path.dirname(dir1)) + "/" + os.path.basename(dir1)
                    dir2_name = os.path.basename(os.path.dirname(dir2)) + "/" + os.path.basename(dir2)
                    
                    friendly_name1 = DIR_NAMES.get(os.path.relpath(dir1, PROJECT_ROOT), dir1_name)
                    friendly_name2 = DIR_NAMES.get(os.path.relpath(dir2, PROJECT_ROOT), dir2_name)
                    
                    cols1 = set(all_datasets[dir1][dataset_type].columns)
                    cols2 = set(all_datasets[dir2][dataset_type].columns)
                    
                    print(f"\n  Comparando colunas entre {friendly_name1} e {friendly_name2}:")
                    print(f"    Colunas exclusivas em {friendly_name1}: {len(cols1 - cols2)}")
                    print(f"    Colunas exclusivas em {friendly_name2}: {len(cols2 - cols1)}")
                    print(f"    Colunas em comum: {len(cols1 & cols2)}")
                    
                    # Adicionar esta comparação nos resultados
                    column_comparisons.append({
                        "dir1": friendly_name1,
                        "dir2": friendly_name2,
                        f"{friendly_name1}_only": list(cols1 - cols2),
                        f"{friendly_name2}_only": list(cols2 - cols1),
                        "common": len(cols1 & cols2)
                    })
            
            if column_comparisons:
                comparison[f"{dataset_type}_column_comparison"] = column_comparisons
    
    return comparison

def print_comparison_report(comparison):
    """
    Imprime um relatório detalhado da comparação.
    
    Args:
        comparison: Resultados da comparação
    """
    print("\n\n======================================================")
    print("               RELATÓRIO DE COMPARAÇÃO                ")
    print("======================================================\n")
    
    for dataset_type in ["train", "validation", "test"]:
        if dataset_type not in comparison:
            continue
            
        print(f"\n----- CONJUNTO DE {dataset_type.upper()} -----\n")
        
        for dir_path, stats in comparison[dataset_type].items():
            dir_name = os.path.basename(os.path.dirname(dir_path)) + "/" + os.path.basename(dir_path)
            friendly_name = DIR_NAMES.get(os.path.relpath(dir_path, PROJECT_ROOT), dir_name)
            
            print(f"Dataset: {friendly_name}")
            print(f"  Dimensões: {stats['shape'][0]} linhas, {stats['shape'][1]} colunas")
            
            # Tipos de dados
            print("  Tipos de dados:")
            for dtype, count in stats['dtypes'].items():
                print(f"    {dtype}: {count} colunas")
            
            # Valores ausentes
            print(f"  Valores ausentes: {stats['missing_values']['total']} ({stats['missing_values']['pct']:.2f}%)")
            
            if stats['missing_values']['top_cols']:
                print("  Top colunas com valores ausentes:")
                for col, info in stats['missing_values']['top_cols'].items():
                    print(f"    {col}: {info['count']} ({info['pct']:.2f}%)")
            
            # Colunas de texto
            print(f"  Colunas de texto identificadas: {stats['text_cols']}")
            
            # Categorias de features
            if stats['feature_categories']:
                print("  Categorias de features:")
                for category, count in stats['feature_categories'].items():
                    print(f"    {category}: {count} colunas")
            
            print("\n" + "-" * 50 + "\n")
    
    # Imprimir comparações de colunas
    for key in comparison:
        if "_column_comparison" in key:
            dataset_type = key.split("_column_comparison")[0]
            print(f"\n----- COMPARAÇÃO DE COLUNAS ({dataset_type.upper()}) -----\n")
            
            for comp_info in comparison[key]:
                dir1 = comp_info["dir1"]
                dir2 = comp_info["dir2"]
                print(f"Comparação entre {dir1} e {dir2}:")
                print(f"  Colunas em comum: {comp_info['common']}")
                
                for dir_name in [dir1, dir2]:
                    col_list = comp_info.get(f"{dir_name}_only", [])
                    print(f"  Colunas exclusivas em {dir_name}: {len(col_list)}")
                    # Exibir algumas colunas de exemplo
                    if col_list:
                        examples = col_list[:5]
                        print(f"    Exemplos: {', '.join(examples)}")
                        if len(col_list) > 5:
                            print(f"    ... e mais {len(col_list) - 5} outras colunas")
                
                print()
            
            print("-" * 50)

def write_report_to_file(comparison, output_file=None):
    """
    Escreve o relatório de comparação em um arquivo.
    
    Args:
        comparison: Resultados da comparação
        output_file: Caminho para salvar o relatório (opcional)
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(PROJECT_ROOT, "reports", f"dataset_comparison_{timestamp}.txt")
    
    # Garantir que o diretório de saída existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nSalvando relatório em {output_file}...")
    
    # Redirecionar a saída para o arquivo
    import sys
    original_stdout = sys.stdout
    
    with open(output_file, 'w') as f:
        sys.stdout = f
        print_comparison_report(comparison)
        sys.stdout = original_stdout
    
    print(f"Relatório salvo com sucesso!")
    return output_file

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Comparar datasets entre diferentes versões de processamento.")
    
    parser.add_argument("--output", type=str, 
                        help="Caminho para salvar o relatório (opcional)")
    
    args = parser.parse_args()
    
    # Executar comparação
    comparison = compare_datasets()
    
    if comparison:
        # Imprimir relatório no console
        print_comparison_report(comparison)
        
        # Salvar relatório em arquivo
        if args.output:
            write_report_to_file(comparison, args.output)
        else:
            write_report_to_file(comparison)

if __name__ == "__main__":
    main()