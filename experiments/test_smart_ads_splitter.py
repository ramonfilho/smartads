#!/usr/bin/env python
"""
Script para testar o data splitter específico para o Smart Ads.

Este script carrega os dados, categoriza as features e salva os resultados.
"""

import os
import sys
import pandas as pd

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Importar funções do data splitter específico para Smart Ads
from src.modeling.stacking.data_splitter import (
    categorize_smart_ads_features,
    print_feature_group_stats,
    validate_feature_groups,
    save_feature_groups
)

def main():
    # Configurar caminhos
    data_path = "/Users/ramonmoreira/desktop/smart_ads/data/02_3_processed_text_code6/validation.csv"
    output_dir = "/Users/ramonmoreira/desktop/smart_ads/feature_groups"
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar dados
    print(f"Carregando dados de {data_path}...")
    df = pd.read_csv(data_path)
    print(f"DataFrame carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Categorizar features com o novo método específico para Smart Ads
    print("\nCategorizando features do Smart Ads...")
    feature_groups = categorize_smart_ads_features(df, target_col="target")
    
    # Validar grupos
    feature_groups = validate_feature_groups(feature_groups, df)
    
    # Mostrar estatísticas
    print_feature_group_stats(feature_groups, df)
    
    # Salvar grupos
    output_path = os.path.join(output_dir, "smart_ads_feature_groups.csv")
    save_feature_groups(feature_groups, output_path)
    print(f"Grupos de features salvos em: {output_path}")
    
    # Características específicas por grupo
    print("\nCaracterísticas por grupo:")
    for group, features in sorted(feature_groups.items()):
        value_counts = []
        missing_counts = []
        
        if features and len(features) > 0:
            # Obter tipo de dados predominante
            dtypes = df[features].dtypes.value_counts()
            dominant_type = dtypes.index[0] if len(dtypes) > 0 else "N/A"
            
            # Contagem de valores únicos (média para o grupo)
            try:
                avg_unique = df[features].nunique().mean()
                value_counts.append(f"média de valores únicos: {avg_unique:.1f}")
            except:
                pass
            
            # Contagem de valores ausentes
            try:
                missing_pct = df[features].isna().mean().mean() * 100
                missing_counts.append(f"média de valores ausentes: {missing_pct:.1f}%")
            except:
                pass
            
            # Imprimir resumo
            print(f"  {group}: {len(features)} features, tipo predominante: {dominant_type}")
            if value_counts:
                print(f"    {', '.join(value_counts)}")
            if missing_counts:
                print(f"    {', '.join(missing_counts)}")
    
    return feature_groups

if __name__ == "__main__":
    main()