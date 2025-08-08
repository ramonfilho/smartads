#!/usr/bin/env python
"""
Script para entender a diferença entre taxa de matching e taxa real.
"""

import os
import sys
import pandas as pd
import numpy as np

# Adicionar o diretório raiz do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

from src.utils.local_storage import (
    connect_to_gcs, list_files_by_extension, categorize_files,
    load_csv_or_excel, load_csv_with_auto_delimiter, extract_launch_id
)
from src.preprocessing.email_processing import normalize_email

def analyze_matching_vs_real_rates():
    """Compara as diferentes formas de calcular taxa de conversão."""
    print("=== ANÁLISE: TAXA DE MATCHING vs TAXA REAL ===\n")
    
    # Carregar dados processados do script 01
    split_dir = os.path.join(project_root, "data", "new", "01_split")
    
    if os.path.exists(split_dir):
        print("1. DADOS DO MODELO (após matching):")
        train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(split_dir, "validation.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))
        
        total_after_matching = len(train_df) + len(val_df) + len(test_df)
        total_positives = train_df['target'].sum() + val_df['target'].sum() + test_df['target'].sum()
        
        print(f"   - Total de registros (pesquisas com matching): {total_after_matching:,}")
        print(f"   - Total de conversões (target=1): {total_positives:,}")
        print(f"   - Taxa de conversão no modelo: {(total_positives/total_after_matching*100):.2f}%")
        print()
    
    # Agora vamos calcular as taxas reais
    raw_data_path = "/Users/ramonmoreira/desktop/smart_ads/data/raw_data"
    bucket = connect_to_gcs("local_bucket", data_path=raw_data_path)
    
    file_paths = list_files_by_extension(bucket, prefix="")
    survey_files, buyer_files, utm_files, _ = categorize_files(file_paths)
    
    # Estatísticas globais
    total_utm_leads = 0
    total_survey_responses = 0
    total_unique_buyers = 0
    total_matched_buyers = 0
    
    print("2. ANÁLISE POR LANÇAMENTO:")
    print("-" * 100)
    print(f"{'Launch':<8} {'UTM Leads':<12} {'Pesquisas':<12} {'Compradores':<12} {'Taxa Real':<10} {'Taxa Base Pesq':<15}")
    print("-" * 100)
    
    # Processar cada lançamento
    all_launches = set()
    for f in buyer_files + utm_files + survey_files:
        launch_id = extract_launch_id(f)
        if launch_id:
            all_launches.add(launch_id)
    
    for launch_id in sorted(all_launches):
        # Carregar dados do lançamento
        launch_buyer = [f for f in buyer_files if extract_launch_id(f) == launch_id]
        launch_utm = [f for f in utm_files if extract_launch_id(f) == launch_id]
        launch_survey = [f for f in survey_files if extract_launch_id(f) == launch_id]
        
        if launch_buyer and launch_utm and launch_survey:
            # Carregar DataFrames
            buyer_df = load_csv_or_excel(bucket, launch_buyer[0])
            utm_df = load_csv_with_auto_delimiter(bucket, launch_utm[0])
            survey_df = load_csv_or_excel(bucket, launch_survey[0])
            
            # Contar UTMs
            utm_count = len(utm_df)
            total_utm_leads += utm_count
            
            # Contar pesquisas
            survey_count = len(survey_df)
            total_survey_responses += survey_count
            
            # Contar compradores únicos
            buyer_email_col = None
            for col in buyer_df.columns:
                if 'email' in col.lower():
                    buyer_email_col = col
                    break
            
            if buyer_email_col:
                unique_buyers = buyer_df[buyer_email_col].dropna().apply(normalize_email).dropna().nunique()
                total_unique_buyers += unique_buyers
            else:
                unique_buyers = 0
            
            # Taxas
            taxa_real = (unique_buyers / utm_count * 100) if utm_count > 0 else 0
            taxa_base_pesquisa = (unique_buyers / survey_count * 100) if survey_count > 0 else 0
            
            print(f"{launch_id:<8} {utm_count:<12,} {survey_count:<12,} {unique_buyers:<12,} "
                  f"{taxa_real:<10.2f}% {taxa_base_pesquisa:<15.2f}%")
    
    print("-" * 100)
    
    # Resumo final
    print(f"\n3. RESUMO GERAL:")
    print(f"   - Total de leads (UTM): {total_utm_leads:,}")
    print(f"   - Total de respostas de pesquisa: {total_survey_responses:,}")
    print(f"   - Total de compradores únicos: {total_unique_buyers:,}")
    print()
    
    # Diferentes formas de calcular a taxa
    taxa_real_global = (total_unique_buyers / total_utm_leads * 100) if total_utm_leads > 0 else 0
    taxa_base_pesquisa_global = (total_unique_buyers / total_survey_responses * 100) if total_survey_responses > 0 else 0
    proporcao_pesquisa = (total_survey_responses / total_utm_leads * 100) if total_utm_leads > 0 else 0
    
    print(f"4. DIFERENTES MÉTRICAS DE CONVERSÃO:")
    print(f"   a) Taxa Real (Compradores/UTMs): {taxa_real_global:.2f}%")
    print(f"   b) Taxa Base Pesquisa (Compradores/Pesquisas): {taxa_base_pesquisa_global:.2f}%")
    print(f"   c) Taxa de Resposta da Pesquisa: {proporcao_pesquisa:.2f}%")
    print()
    
    print(f"5. INTERPRETAÇÃO:")
    print(f"   - A cada 100 leads (UTMs), {taxa_real_global:.1f} compram")
    print(f"   - A cada 100 que respondem a pesquisa, {taxa_base_pesquisa_global:.1f} compram")
    print(f"   - Apenas {proporcao_pesquisa:.1f}% dos leads respondem a pesquisa")
    print()
    
    print(f"6. EXPLICAÇÃO DA DIFERENÇA:")
    print(f"   - Taxa de matching (1.15%) = compradores que fizeram match com pesquisa / total de pesquisas")
    print(f"   - Taxa real ({taxa_real_global:.2f}%) = todos os compradores / todos os leads")
    print(f"   - A taxa de matching é MENOR porque:")
    print(f"     1. Nem todos os compradores respondem a pesquisa")
    print(f"     2. Muitos usam emails diferentes na pesquisa vs compra")
    print(f"     3. O matching captura apenas uma fração dos compradores reais")

if __name__ == "__main__":
    analyze_matching_vs_real_rates()