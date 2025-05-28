#!/usr/bin/env python
"""
Script para calcular a taxa de conversão real por lançamento.
Taxa real = compradores únicos / total de leads (arquivo UTM)
Usa a mesma lógica do script 01_data_collection_and_integration.py
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

# ============================================================================
# FUNÇÕES AUXILIARES (mesmas do script 01)
# ============================================================================

def find_email_column(df):
    """Encontra a coluna que contém emails em um DataFrame."""
    email_patterns = ['email', 'e-mail', 'correo', '@', 'mail']
    for col in df.columns:
        if any(pattern in col.lower() for pattern in email_patterns):
            return col
    return None

def process_buyer_file(bucket, file_path):
    """Processa um arquivo de compradores."""
    df = load_csv_or_excel(bucket, file_path)
    if df is None:
        return None
        
    # Identificar o lançamento deste arquivo
    launch_id = extract_launch_id(file_path)
    
    # Para buyers, ainda precisamos da coluna 'email' para normalização
    email_col = find_email_column(df)
    if email_col and email_col != 'email':
        df = df.rename(columns={email_col: 'email'})
    elif not email_col:
        print(f"  - Warning: No email column found in {file_path}. Available columns: {', '.join(df.columns[:5])}...")
    
    # Adicionar identificador de lançamento se disponível
    if launch_id:
        df['lançamento'] = launch_id
    
    return df

def process_utm_file(bucket, file_path):
    """Processa um arquivo de UTM preservando a coluna E-MAIL."""
    df = load_csv_with_auto_delimiter(bucket, file_path)
    if df is None:
        return None
        
    # Identificar o lançamento deste arquivo
    launch_id = extract_launch_id(file_path)
    
    # Adicionar identificador de lançamento se disponível
    if launch_id:
        df['lançamento'] = launch_id
    
    return df

def load_buyer_files(bucket, buyer_files):
    """Carrega todos os arquivos de compradores."""
    buyer_dfs = []
    launch_data = {}
    
    print("\nLoading buyer files...")
    for file_path in buyer_files:
        try:
            df = process_buyer_file(bucket, file_path)
            if df is not None:
                buyer_dfs.append(df)
                
                launch_id = extract_launch_id(file_path)
                if launch_id:
                    if launch_id not in launch_data:
                        launch_data[launch_id] = {}
                    launch_data[launch_id]['buyer'] = df
                    
                print(f"  - Loaded: {file_path} ({launch_id if launch_id else ''}), {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
    
    return buyer_dfs, launch_data

def load_utm_files(bucket, utm_files):
    """Carrega todos os arquivos de UTM preservando E-MAIL original."""
    utm_dfs = []
    launch_data = {}
    
    print("\nLoading UTM files...")
    for file_path in utm_files:
        try:
            df = process_utm_file(bucket, file_path)
            if df is not None:
                utm_dfs.append(df)
                
                launch_id = extract_launch_id(file_path)
                if launch_id:
                    if launch_id not in launch_data:
                        launch_data[launch_id] = {}
                    launch_data[launch_id]['utm'] = df
                    
                print(f"  - Loaded: {file_path} ({launch_id if launch_id else ''}), {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
    
    return utm_dfs, launch_data

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def calculate_real_conversion_rates():
    """Calcula a taxa de conversão real por lançamento usando a lógica do script 01."""
    print("=== CALCULANDO TAXA DE CONVERSÃO REAL POR LANÇAMENTO ===\n")
    
    # NOVO: Definir caminho dos dados brutos aqui
    raw_data_path = "/Users/ramonmoreira/desktop/smart_ads/data/raw_data"
    
    # 1. Conectar ao armazenamento local
    bucket = connect_to_gcs("local_bucket", data_path=raw_data_path)
    
    # 2. Listar e categorizar arquivos
    file_paths = list_files_by_extension(bucket, prefix="")
    print(f"Found {len(file_paths)} files in: {raw_data_path}")
    
    # 3. Categorizar arquivos
    survey_files, buyer_files, utm_files, all_files_by_launch = categorize_files(file_paths)
    
    print(f"Survey files: {len(survey_files)}")
    print(f"Buyer files: {len(buyer_files)}")
    print(f"UTM files: {len(utm_files)}")
    
    # 4. Carregar dados
    buyer_dfs, buyer_launch_data = load_buyer_files(bucket, buyer_files)
    utm_dfs, utm_launch_data = load_utm_files(bucket, utm_files)
    
    # 5. Combinar launch_data
    all_launch_data = {}
    for launch_id in set(list(buyer_launch_data.keys()) + list(utm_launch_data.keys())):
        all_launch_data[launch_id] = {
            'buyer': buyer_launch_data.get(launch_id, {}).get('buyer'),
            'utm': utm_launch_data.get(launch_id, {}).get('utm')
        }
    
    # 6. Calcular taxa de conversão por lançamento
    conversion_rates = []
    
    print("\n" + "="*70)
    print("ANÁLISE POR LANÇAMENTO")
    print("="*70)
    
    for launch_id in sorted(all_launch_data.keys()):
        print(f"\n{'='*60}")
        print(f"LANÇAMENTO: {launch_id}")
        print(f"{'='*60}")
        
        buyer_df = all_launch_data[launch_id].get('buyer')
        utm_df = all_launch_data[launch_id].get('utm')
        
        # Processar compradores
        buyer_emails = set()
        if buyer_df is not None and not buyer_df.empty:
            if 'email' in buyer_df.columns:
                # Normalizar emails antes de adicionar ao conjunto
                unique_emails = buyer_df['email'].dropna().apply(normalize_email).dropna().unique()
                buyer_emails.update(unique_emails)
                print(f"  - Compradores: {len(buyer_df):,} registros")
                print(f"  - Emails únicos de compradores: {len(buyer_emails):,}")
            else:
                print(f"  - AVISO: Arquivo de compradores sem coluna 'email'")
        else:
            print(f"  - Sem dados de compradores")
        
        # Processar UTMs
        utm_count = 0
        if utm_df is not None and not utm_df.empty:
            utm_count = len(utm_df)
            print(f"  - UTMs (leads): {utm_count:,} registros")
        else:
            print(f"  - Sem dados de UTM")
        
        # Calcular taxa de conversão
        unique_buyers = len(buyer_emails)
        conversion_rate = (unique_buyers / utm_count * 100) if utm_count > 0 else 0
        
        print(f"\n📊 MÉTRICAS DO {launch_id}:")
        print(f"  - Total de leads (UTM): {utm_count:,}")
        print(f"  - Compradores únicos: {unique_buyers:,}")
        print(f"  - Taxa de conversão real: {conversion_rate:.2f}%")
        
        conversion_rates.append({
            'launch_id': launch_id,
            'total_leads_utm': utm_count,
            'unique_buyers': unique_buyers,
            'conversion_rate': conversion_rate
        })
    
    # 7. Criar DataFrame com resultados
    results_df = pd.DataFrame(conversion_rates)
    
    # 8. Estatísticas gerais
    print(f"\n{'='*70}")
    print("RESUMO GERAL")
    print(f"{'='*70}")
    
    if not results_df.empty:
        # Filtrar apenas lançamentos válidos (com UTMs)
        valid_results = results_df[results_df['total_leads_utm'] > 0]
        
        if not valid_results.empty:
            total_leads = valid_results['total_leads_utm'].sum()
            total_buyers = valid_results['unique_buyers'].sum()
            overall_rate = (total_buyers / total_leads * 100) if total_leads > 0 else 0
            
            print(f"Total de leads (todos lançamentos): {total_leads:,}")
            print(f"Total de compradores únicos: {total_buyers:,}")
            print(f"Taxa de conversão geral: {overall_rate:.2f}%")
            print(f"\nTaxa de conversão por lançamento (apenas com dados válidos):")
            print(f"  - Mínima: {valid_results['conversion_rate'].min():.2f}%")
            print(f"  - Máxima: {valid_results['conversion_rate'].max():.2f}%")
            print(f"  - Média: {valid_results['conversion_rate'].mean():.2f}%")
            print(f"  - Mediana: {valid_results['conversion_rate'].median():.2f}%")
            
            # Comparação com a taxa do modelo
            model_rate = 1.15
            print(f"\n📈 COMPARAÇÃO COM O MODELO:")
            print(f"  - Taxa usada no modelo (matching): {model_rate:.2f}%")
            print(f"  - Taxa de conversão real média: {valid_results['conversion_rate'].mean():.2f}%")
            print(f"  - Diferença: {valid_results['conversion_rate'].mean() - model_rate:.2f} pontos percentuais")
            
            # Comparação com a taxa via matching
            print(f"\n📊 ANÁLISE DA DIFERENÇA:")
            if valid_results['conversion_rate'].mean() > model_rate:
                print(f"  - A taxa real é {(valid_results['conversion_rate'].mean() / model_rate - 1) * 100:.1f}% MAIOR que a taxa de matching")
                print(f"  - Isso sugere que muitos compradores usam emails diferentes")
            else:
                print(f"  - A taxa real é {(1 - valid_results['conversion_rate'].mean() / model_rate) * 100:.1f}% MENOR que a taxa de matching")
        
        # Salvar resultados
        output_dir = os.path.join(project_root, "analysis")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "real_conversion_rates.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Resultados salvos em: {output_path}")
        
        # Mostrar tabela completa
        print(f"\nTABELA DETALHADA:")
        print("-" * 70)
        print(f"{'Lançamento':<12} {'Leads (UTM)':<15} {'Compradores':<15} {'Taxa Conv.':<12}")
        print("-" * 70)
        for _, row in results_df.iterrows():
            print(f"{row['launch_id']:<12} {row['total_leads_utm']:<15,} {row['unique_buyers']:<15,} {row['conversion_rate']:<12.2f}%")
        print("-" * 70)
    else:
        print("Nenhum dado foi processado.")
    
    return results_df

if __name__ == "__main__":
    results = calculate_real_conversion_rates()