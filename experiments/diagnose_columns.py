#!/usr/bin/env python
"""
Script de diagnóstico para detectar mudanças de nomes de colunas entre lançamentos.
Analisa quais features estão isoladas (não se repetem) e sugere possíveis mapeamentos.
"""

import os
import sys
import pandas as pd
from difflib import SequenceMatcher
from collections import defaultdict, Counter
import warnings

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

# Importar módulos do projeto
from src.utils.local_storage import (
    connect_to_gcs, list_files_by_extension, categorize_files,
    load_csv_or_excel, extract_launch_id
)

def similarity_score(str1, str2):
    """Calcula score de similaridade entre duas strings."""
    if not str1 or not str2:
        return 0
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def clean_column_name(col_name):
    """Limpa nome da coluna para comparação mais efetiva."""
    if pd.isna(col_name):
        return ""
    
    # Converter para string e limpar
    clean_name = str(col_name).lower().strip()
    
    # Remover caracteres especiais comuns
    replacements = {
        '¿': '', '?': '', '¡': '', '!': '', 
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n',
        'ü': 'u', '\n': ' ', '\t': ' '
    }
    
    for old, new in replacements.items():
        clean_name = clean_name.replace(old, new)
    
    # Normalizar espaços
    clean_name = ' '.join(clean_name.split())
    
    return clean_name

def load_survey_data_by_launch():
    """Carrega dados de pesquisa separados por lançamento."""
    print("Carregando dados de pesquisa por lançamento...")
    
    # Conectar ao armazenamento
    bucket = connect_to_gcs("local_bucket")
    file_paths = list_files_by_extension(bucket, prefix="")
    
    # Categorizar arquivos
    survey_files, _, _, _ = categorize_files(file_paths)
    
    # Carregar dados por lançamento
    launch_data = {}
    
    for file_path in survey_files:
        launch_id = extract_launch_id(file_path)
        if not launch_id:
            continue
            
        print(f"  Carregando {file_path}...")
        df = load_csv_or_excel(bucket, file_path)
        
        if df is not None:
            launch_data[launch_id] = {
                'dataframe': df,
                'file_path': file_path,
                'columns': list(df.columns),
                'shape': df.shape
            }
            print(f"    {launch_id}: {df.shape[0]} linhas, {df.shape[1]} colunas")
        else:
            print(f"    ERRO ao carregar {file_path}")
    
    return launch_data

def analyze_column_distribution(launch_data):
    """Analisa a distribuição de colunas entre lançamentos."""
    print("\n=== ANÁLISE DE DISTRIBUIÇÃO DE COLUNAS ===")
    
    # Coletar todas as colunas com seus lançamentos
    column_launches = defaultdict(list)
    all_columns = set()
    
    for launch_id, data in launch_data.items():
        for col in data['columns']:
            clean_col = clean_column_name(col)
            column_launches[clean_col].append({
                'launch': launch_id,
                'original_name': col
            })
            all_columns.add(clean_col)
    
    # Contar frequências
    column_freq = Counter()
    for col, launches in column_launches.items():
        column_freq[col] = len(launches)
    
    total_launches = len(launch_data)
    
    # Categorizar colunas
    universal_cols = []  # Presentes em todos os lançamentos
    frequent_cols = []   # Presentes na maioria (>=75%)
    occasional_cols = [] # Presentes em alguns (25-75%)
    rare_cols = []       # Presentes em poucos (<25%)
    
    for col, freq in column_freq.items():
        percentage = freq / total_launches
        
        if percentage == 1.0:
            universal_cols.append(col)
        elif percentage >= 0.75:
            frequent_cols.append(col)
        elif percentage >= 0.25:
            occasional_cols.append(col)
        else:
            rare_cols.append(col)
    
    print(f"\nTotal de lançamentos analisados: {total_launches}")
    print(f"Total de colunas únicas: {len(all_columns)}")
    print(f"\nDistribuição:")
    print(f"  📊 Universais (100%): {len(universal_cols)} colunas")
    print(f"  🔄 Frequentes (>=75%): {len(frequent_cols)} colunas")
    print(f"  ⚠️  Ocasionais (25-75%): {len(occasional_cols)} colunas")
    print(f"  🚨 Raras (<25%): {len(rare_cols)} colunas")
    
    return {
        'column_launches': column_launches,
        'universal': universal_cols,
        'frequent': frequent_cols,
        'occasional': occasional_cols,
        'rare': rare_cols,
        'total_launches': total_launches
    }

def detect_isolated_features(analysis_result, launch_data):
    """Detecta features que aparecem em poucos lançamentos."""
    print("\n=== FEATURES ISOLADAS (SUSPEITAS) ===")
    
    isolated_features = []
    
    # Analisar colunas ocasionais e raras
    suspicious_cols = analysis_result['occasional'] + analysis_result['rare']
    
    for col in suspicious_cols:
        launches_info = analysis_result['column_launches'][col]
        launches_list = [info['launch'] for info in launches_info]
        
        isolated_features.append({
            'clean_name': col,
            'original_names': [info['original_name'] for info in launches_info],
            'launches': launches_list,
            'frequency': len(launches_list),
            'percentage': len(launches_list) / analysis_result['total_launches'] * 100
        })
    
    # Ordenar por frequência (menos frequentes primeiro)
    isolated_features.sort(key=lambda x: x['frequency'])
    
    print(f"\nEncontradas {len(isolated_features)} features suspeitas:")
    print("-" * 80)
    
    for i, feature in enumerate(isolated_features[:20]):  # Mostrar top 20
        print(f"{i+1:2d}. {feature['clean_name']}")
        print(f"    📍 Presente em: {', '.join(feature['launches'])} ({feature['frequency']}/{analysis_result['total_launches']} = {feature['percentage']:.1f}%)")
        
        # Mostrar nomes originais se diferentes
        unique_names = list(set(feature['original_names']))
        if len(unique_names) > 1:
            print(f"    🔄 Variações: {unique_names}")
        else:
            print(f"    📝 Nome: {unique_names[0]}")
        print()
    
    if len(isolated_features) > 20:
        print(f"    ... e mais {len(isolated_features) - 20} features")
    
    return isolated_features

def suggest_column_mappings(isolated_features, analysis_result, similarity_threshold=0.7):
    """Sugere possíveis mapeamentos entre colunas similares."""
    print("\n=== SUGESTÕES DE MAPEAMENTO ===")
    
    suggestions = []
    
    # Para cada feature isolada, procurar similares
    for feature in isolated_features:
        if feature['frequency'] == 1:  # Features que aparecem em apenas um lançamento
            clean_name = feature['clean_name']
            best_matches = []
            
            # Comparar com todas as outras features
            for other_feature in isolated_features:
                if other_feature['clean_name'] != clean_name:
                    similarity = similarity_score(clean_name, other_feature['clean_name'])
                    
                    if similarity >= similarity_threshold:
                        best_matches.append({
                            'target': other_feature,
                            'similarity': similarity
                        })
            
            # Ordenar por similaridade
            best_matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            if best_matches:
                suggestions.append({
                    'source': feature,
                    'matches': best_matches[:3]  # Top 3 matches
                })
    
    print(f"Sugestões de mapeamento (similaridade >= {similarity_threshold}):")
    print("-" * 80)
    
    for i, suggestion in enumerate(suggestions[:15]):  # Top 15 sugestões
        source = suggestion['source']
        print(f"{i+1:2d}. ORIGEM: {source['clean_name']}")
        print(f"    📍 {', '.join(source['launches'])}")
        print(f"    📝 {source['original_names'][0]}")
        
        print(f"    🔗 POSSÍVEIS MAPEAMENTOS:")
        for match in suggestion['matches']:
            target = match['target']
            sim_score = match['similarity']
            print(f"       → {target['clean_name']} (similaridade: {sim_score:.2f})")
            print(f"         📍 {', '.join(target['launches'])}")
            print(f"         📝 {target['original_names'][0]}")
        print()
    
    return suggestions

def generate_mapping_template(suggestions):
    """Gera um template de mapeamento para implementação."""
    print("\n=== TEMPLATE DE MAPEAMENTO ===")
    
    print("# Adicione este dicionário ao seu código para mapear colunas:")
    print("COLUMN_MAPPINGS = {")
    
    for suggestion in suggestions[:10]:  # Top 10 para o template
        source = suggestion['source']
        if suggestion['matches']:
            best_match = suggestion['matches'][0]['target']
            
            source_name = source['original_names'][0]
            target_name = best_match['original_names'][0]
            
            print(f"    '{source_name}': '{target_name}',")
    
    print("}")
    print("\n# Como usar:")
    print("# 1. Revise cada mapeamento sugerido")
    print("# 2. Remova mapeamentos incorretos")
    print("# 3. Adicione mapeamentos adicionais se necessário")
    print("# 4. Implemente no script de coleta de dados")

def main():
    """Função principal do diagnóstico."""
    print("=== DIAGNÓSTICO DE COLUNAS ENTRE LANÇAMENTOS ===")
    print("Analisando mudanças de nomes de features...")
    
    try:
        # 1. Carregar dados por lançamento
        launch_data = load_survey_data_by_launch()
        
        if not launch_data:
            print("ERRO: Nenhum dado de pesquisa encontrado!")
            return
        
        # 2. Analisar distribuição de colunas
        analysis_result = analyze_column_distribution(launch_data)
        
        # 3. Detectar features isoladas
        isolated_features = detect_isolated_features(analysis_result, launch_data)
        
        # 4. Sugerir mapeamentos
        suggestions = suggest_column_mappings(isolated_features, analysis_result)
        
        # 5. Gerar template
        generate_mapping_template(suggestions)
        
        print("\n=== DIAGNÓSTICO CONCLUÍDO ===")
        print("Review as sugestões acima e implemente os mapeamentos necessários.")
        
    except Exception as e:
        print(f"ERRO durante o diagnóstico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()