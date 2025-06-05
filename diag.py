# Crie um arquivo temporário: analyze_column_dependencies.py

import pandas as pd
import numpy as np
from unified_pipeline import unified_data_pipeline

# Execute o pipeline em modo de teste pequeno
print("=== ANÁLISE DE DEPENDÊNCIAS DE COLUNAS ===\n")

# Configurar para análise rápida
results = unified_data_pipeline(
    test_mode=True,
    max_samples=100,  # Apenas 100 amostras para ser rápido
    apply_feature_selection=False,  # Pular feature selection
    use_checkpoints=False,
    clear_cache=True
)

# Se conseguir rodar (mesmo que parcialmente), vamos analisar
if results and 'train' in results:
    train_df = results['train']
    
    print("\n=== COLUNAS NO DATASET FINAL ===")
    print(f"Total de colunas: {len(train_df.columns)}")
    
    # Agrupar por tipo/padrão
    print("\n=== ANÁLISE POR PADRÃO ===")
    
    # Colunas originais (sem sufixo)
    original_cols = [col for col in train_df.columns 
                    if not any(suffix in col for suffix in 
                    ['_encoded', '_clean', '_tfidf', '_sentiment', '_motiv', 
                     '_embedding', '_topic', '_x_', '_per_', '_zscore'])]
    print(f"\nColunas originais preservadas: {len(original_cols)}")
    for col in sorted(original_cols)[:10]:
        print(f"  - {col}")
    if len(original_cols) > 10:
        print(f"  ... e mais {len(original_cols)-10}")
    
    # Colunas encoded
    encoded_cols = [col for col in train_df.columns if '_encoded' in col]
    print(f"\nColunas encoded: {len(encoded_cols)}")
    for col in sorted(encoded_cols)[:10]:
        print(f"  - {col}")
    
    # Colunas TF-IDF
    tfidf_cols = [col for col in train_df.columns if '_tfidf_' in col]
    print(f"\nColunas TF-IDF: {len(tfidf_cols)}")
    print(f"  (mostrando apenas contagem devido ao volume)")
    
    # Colunas de interação
    interaction_cols = [col for col in train_df.columns if '_x_' in col]
    print(f"\nColunas de interação: {len(interaction_cols)}")
    for col in sorted(interaction_cols)[:10]:
        print(f"  - {col}")