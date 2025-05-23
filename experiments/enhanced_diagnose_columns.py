#!/usr/bin/env python
"""
Script de diagnóstico expandido para detectar mudanças de nomes de colunas.
Inclui análise do arquivo de produção e busca mais detalhada por variações.
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

def load_production_reference():
    """Carrega arquivo de produção como referência."""
    production_file = "/Users/ramonmoreira/Desktop/smart_ads/data/l24_data.csv"
    
    try:
        if os.path.exists(production_file):
            print(f"📋 Carregando arquivo de produção: {production_file}")
            df = pd.read_csv(production_file)
            
            # Filtrar colunas de predição
            prediction_cols = ['prediction', 'decil', 'Prediction', 'Decil']
            original_cols = [col for col in df.columns if col not in prediction_cols]
            
            print(f"   Shape: {df.shape}")
            print(f"   Colunas (excluindo predições): {len(original_cols)}")
            print(f"   Primeiras 5 colunas: {original_cols[:5]}")
            
            return {
                'dataframe': df,
                'columns': original_cols,
                'all_columns': list(df.columns)
            }
        else:
            print(f"⚠️ Arquivo de produção não encontrado: {production_file}")
            return None
    except Exception as e:
        print(f"❌ Erro ao carregar arquivo de produção: {e}")
        return None

def load_survey_data_by_launch():
    """Carrega dados de pesquisa separados por lançamento."""
    print("\n📊 Carregando dados de pesquisa por lançamento...")
    
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
            
        print(f"  📁 Carregando {file_path}...")
        df = load_csv_or_excel(bucket, file_path)
        
        if df is not None:
            launch_data[launch_id] = {
                'dataframe': df,
                'file_path': file_path,
                'columns': list(df.columns),
                'shape': df.shape
            }
            print(f"     {launch_id}: {df.shape[0]} linhas, {df.shape[1]} colunas")
        else:
            print(f"     ❌ ERRO ao carregar {file_path}")
    
    return launch_data

def analyze_quality_columns(launch_data):
    """Analisa especificamente as colunas de qualidade."""
    print("\n=== 🔍 ANÁLISE DETALHADA DAS COLUNAS DE QUALIDADE ===")
    
    quality_patterns = ['qualidade', 'qualidad', 'quality']
    
    for launch_id, data in launch_data.items():
        quality_cols = []
        for col in data['columns']:
            if any(pattern in col.lower() for pattern in quality_patterns):
                quality_cols.append(col)
        
        print(f"\n📋 {launch_id}:")
        if quality_cols:
            for col in quality_cols:
                # Analisar conteúdo da coluna
                df = data['dataframe']
                sample_values = df[col].dropna().head(10).tolist()
                unique_count = df[col].nunique()
                
                print(f"   🔸 {col}")
                print(f"      Valores únicos: {unique_count}")
                print(f"      Amostra: {sample_values[:3]}")
        else:
            print("   ❌ Nenhuma coluna de qualidade encontrada")

def search_inmersion_variations(launch_data, target_question):
    """Busca variações da pergunta sobre Inmersión com threshold mais baixo."""
    print(f"\n=== 🔎 BUSCANDO VARIAÇÕES DA PERGUNTA INMERSIÓN ===")
    print(f"Pergunta alvo: {target_question}")
    
    target_clean = clean_column_name(target_question)
    keywords = ['inmersion', 'aprender', 'ingles', 'horas', '72']
    
    print(f"Palavras-chave: {keywords}")
    print("-" * 80)
    
    variations_found = []
    
    for launch_id, data in launch_data.items():
        print(f"\n📋 {launch_id}:")
        best_matches = []
        
        for col in data['columns']:
            col_clean = clean_column_name(col)
            
            # Verificar se contém palavras-chave relevantes
            keyword_matches = sum(1 for keyword in keywords if keyword in col_clean)
            
            # Calcular similaridade
            similarity = similarity_score(target_clean, col_clean)
            
            # Considerar match se tem palavras-chave OU alta similaridade
            if keyword_matches >= 2 or similarity >= 0.3:
                best_matches.append({
                    'column': col,
                    'similarity': similarity,
                    'keyword_matches': keyword_matches,
                    'clean_name': col_clean
                })
        
        # Ordenar por relevância (palavras-chave + similaridade)
        best_matches.sort(key=lambda x: (x['keyword_matches'], x['similarity']), reverse=True)
        
        if best_matches:
            for i, match in enumerate(best_matches[:3]):  # Top 3
                print(f"   {i+1}. {match['column']}")
                print(f"      Similaridade: {match['similarity']:.3f}")
                print(f"      Palavras-chave: {match['keyword_matches']}/{len(keywords)}")
                
                variations_found.append({
                    'launch': launch_id,
                    'column': match['column'],
                    'similarity': match['similarity'],
                    'keyword_matches': match['keyword_matches']
                })
        else:
            print("   ❌ Nenhuma variação encontrada")
    
    return variations_found

def compare_with_production(launch_data, production_data):
    """Compara colunas dos lançamentos com arquivo de produção."""
    if not production_data:
        return
        
    print(f"\n=== 📊 COMPARAÇÃO COM DADOS DE PRODUÇÃO ===")
    
    prod_cols_clean = {clean_column_name(col): col for col in production_data['columns']}
    
    print(f"Colunas no arquivo de produção (sem predições): {len(prod_cols_clean)}")
    
    # Para cada lançamento, ver quais colunas estão faltando vs produção
    for launch_id, data in launch_data.items():
        launch_cols_clean = {clean_column_name(col): col for col in data['columns']}
        
        missing_in_launch = set(prod_cols_clean.keys()) - set(launch_cols_clean.keys())
        extra_in_launch = set(launch_cols_clean.keys()) - set(prod_cols_clean.keys())
        
        print(f"\n📋 {launch_id}:")
        print(f"   ✅ Colunas em comum: {len(set(prod_cols_clean.keys()) & set(launch_cols_clean.keys()))}")
        
        if missing_in_launch:
            print(f"   ❌ Faltando no {launch_id}: {len(missing_in_launch)}")
            for col_clean in list(missing_in_launch)[:5]:  # Mostrar apenas 5
                original_name = prod_cols_clean[col_clean]
                print(f"      - {original_name}")
            if len(missing_in_launch) > 5:
                print(f"      ... e mais {len(missing_in_launch) - 5}")
        
        if extra_in_launch:
            print(f"   ➕ Extras no {launch_id}: {len(extra_in_launch)}")
            for col_clean in list(extra_in_launch)[:5]:  # Mostrar apenas 5
                original_name = launch_cols_clean[col_clean]
                print(f"      - {original_name}")
            if len(extra_in_launch) > 5:
                print(f"      ... e mais {len(extra_in_launch) - 5}")

def generate_enhanced_mapping(launch_data, production_data, inmersion_variations):
    """Gera mapeamento mais abrangente baseado em todas as análises."""
    print(f"\n=== 📝 MAPEAMENTO RECOMENDADO ===")
    
    print("# Dicionário de mapeamento de colunas")
    print("COLUMN_MAPPINGS = {")
    
    # 1. Colunas de qualidade
    print("    # === COLUNAS DE QUALIDADE ===")
    quality_mappings = [
        ("'Qualidade (Nome) '", "'Qualidade (Nome)'"),  # Remover espaço
        ("'Qualidade (nome)'", "'Qualidade (Nome)'"),   # Padronizar
        ("'Qualidade (Número) '", "'Qualidade (Número)'"),  # Remover espaço
        ("'Qualidade (número)'", "'Qualidade (Número)'"),   # Padronizar
        ("'Qualidade (Numero)'", "'Qualidade (Número)'"),   # Acentuação
    ]
    
    for old, new in quality_mappings:
        print(f"    {old}: {new},")
    
    # 2. Variações da pergunta Inmersión
    if inmersion_variations:
        print("\n    # === PERGUNTA INMERSIÓN ===")
        target_question = "¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?"
        
        for var in inmersion_variations:
            if var['similarity'] > 0.5 or var['keyword_matches'] >= 3:
                print(f"    '{var['column']}': '{target_question}',")
    
    print("}")
    
    # 3. Colunas para remover
    print("\n# Colunas para remover")
    print("COLUMNS_TO_REMOVE = [")
    print("    'teste',")
    print("    'prediction',") 
    print("    'class',")
    print("    'Prediction',")
    print("    'Decil',")
    print("]")
    
    # 4. Tratamento especial para Qualidade simples
    print("\n# TRATAMENTO ESPECIAL NECESSÁRIO:")
    print("# Para L16 e L17 que têm apenas 'Qualidade':")
    print("# - Verificar se essa coluna contém nomes ou números")
    print("# - Criar lógica para dividir em 'Qualidade (Nome)' e 'Qualidade (Número)'")
    print("# - Ou mapear para uma das duas baseado no conteúdo")

def main():
    """Função principal do diagnóstico expandido."""
    print("=== 🔍 DIAGNÓSTICO EXPANDIDO DE COLUNAS ===")
    
    try:
        # 1. Carregar arquivo de produção como referência
        production_data = load_production_reference()
        
        # 2. Carregar dados por lançamento
        launch_data = load_survey_data_by_launch()
        
        if not launch_data:
            print("❌ ERRO: Nenhum dado de pesquisa encontrado!")
            return
        
        # 3. Análise detalhada das colunas de qualidade
        analyze_quality_columns(launch_data)
        
        # 4. Buscar variações da pergunta Inmersión
        target_inmersion = "¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?"
        inmersion_variations = search_inmersion_variations(launch_data, target_inmersion)
        
        # 5. Comparar com produção (se disponível)
        if production_data:
            compare_with_production(launch_data, production_data)
        
        # 6. Gerar mapeamento abrangente
        generate_enhanced_mapping(launch_data, production_data, inmersion_variations)
        
        print(f"\n=== ✅ DIAGNÓSTICO EXPANDIDO CONCLUÍDO ===")
        
    except Exception as e:
        print(f"❌ ERRO durante o diagnóstico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()