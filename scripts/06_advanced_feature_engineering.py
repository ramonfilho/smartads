#!/usr/bin/env python
"""
Script para testar a engenharia de features avançada nos
conjuntos de treino e validação, com foco na geração de features de texto.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

warnings.filterwarnings('ignore')

# Importe o módulo avançado
from src.preprocessing.advanced_feature_engineering import advanced_feature_engineering

def print_dataset_info(df, sample_size=5):
    """Imprime informações básicas sobre o dataset para diagnóstico."""
    print(f"Dimensões do dataset: {df.shape}")
    
    # Procurar colunas de texto potenciais
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Colunas do tipo object (potencialmente texto): {len(object_cols)}")
    
    # Mostrar algumas das colunas
    if object_cols:
        print("Primeiras colunas do tipo object:")
        for i, col in enumerate(object_cols[:10]):
            # Estatísticas básicas sobre a coluna
            non_null = df[col].dropna()
            if len(non_null) > 0:
                avg_len = non_null.str.len().mean()
                max_len = non_null.str.len().max()
                print(f"  {i+1}. '{col}': {df[col].nunique()} valores únicos, comprimento médio: {avg_len:.1f}, máximo: {max_len}")
                
                # Mostrar uma amostra
                sample = df[col].dropna().sample(min(sample_size, len(non_null))).values
                print(f"     Amostra: {[s[:30] + '...' if len(s) > 30 else s for s in sample]}")
    
    # Encontrar colunas com sufixos comuns que poderiam indicar processamento de texto anterior
    tfidf_cols = [col for col in df.columns if '_tfidf_' in col]
    if tfidf_cols:
        print(f"\nEncontradas {len(tfidf_cols)} colunas TF-IDF já existentes.")
        print(f"Exemplos: {tfidf_cols[:3]}")
        
        # Tentar identificar as colunas de texto originais
        base_cols = set()
        for col in tfidf_cols:
            parts = col.split('_tfidf_')
            if len(parts) > 0:
                base_cols.add(parts[0])
        
        print(f"Possíveis colunas de texto originais: {base_cols}")

def main():
    """Testa a implementação do módulo de features avançadas."""
    # 1. Definir caminhos dos datasets
    base_dir = os.path.expanduser("~")
    train_path = os.path.join(base_dir, "desktop/smart_ads/data/feature_selection/train.csv")
    val_path = os.path.join(base_dir, "desktop/smart_ads/data/feature_selection/validation.csv")
    output_dir = os.path.join(base_dir, "desktop/smart_ads/data/advanced_features")
    params_dir = os.path.join(base_dir, "desktop/smart_ads/src/preprocessing/preprocessing_params")
    
    # Criar diretórios de saída
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    
    # 2. Carregar datasets
    print(f"Carregando conjunto de treino: {train_path}")
    train_df = pd.read_csv(train_path)
    
    # 2.1 Analisar o dataset para identificar colunas de texto
    print("\n=== Informações do conjunto de treino ===")
    print_dataset_info(train_df)
    
    print(f"\nCarregando conjunto de validação: {val_path}")
    val_df = pd.read_csv(val_path)
    
    # 3. Aplicar engenharia de features avançada no conjunto de treino (fit)
    print("\n=== Aplicando engenharia de features avançada no conjunto de treino (fit=True) ===")
    train_df_advanced, params = advanced_feature_engineering(train_df, fit=True)
    
    # 4. Salvar parâmetros aprendidos
    params_path = os.path.join(params_dir, "advanced_features_params.joblib")
    joblib.dump(params, params_path)
    print(f"Parâmetros salvos em: {params_path}")
    
    # 5. Aplicar no conjunto de validação (transform)
    print("\n=== Aplicando engenharia de features avançada no conjunto de validação (fit=False) ===")
    val_df_advanced, _ = advanced_feature_engineering(val_df, fit=False, params=params)
    
    # 6. Verificar consistência
    print("\n=== Verificando consistência entre treino e validação ===")
    
    train_cols = set(train_df_advanced.columns)
    val_cols = set(val_df_advanced.columns)
    
    print(f"Colunas no treino: {len(train_cols)}")
    print(f"Colunas na validação: {len(val_cols)}")
    print(f"Colunas em comum: {len(train_cols & val_cols)}")
    
    if train_cols != val_cols:
        print("ATENÇÃO: Diferença entre colunas!")
        diff_train = train_cols - val_cols
        diff_val = val_cols - train_cols
        
        if diff_train:
            print(f"Colunas apenas no treino ({len(diff_train)}): {list(diff_train)[:5]}...")
            # Adicionar colunas ausentes no validation com valores neutros
            for col in diff_train:
                val_df_advanced[col] = 0
        
        if diff_val:
            print(f"Colunas apenas na validação ({len(diff_val)}): {list(diff_val)[:5]}...")
            # Adicionar colunas ausentes no treino com valores neutros
            for col in diff_val:
                train_df_advanced[col] = 0
                
        print("Correções aplicadas para alinhar conjuntos.")
        
        # Verificar novamente após correções
        train_cols = set(train_df_advanced.columns)
        val_cols = set(val_df_advanced.columns)
        print(f"Após correção - Colunas em comum: {len(train_cols & val_cols)}")
    else:
        print("Conjuntos alinhados corretamente!")
    
    # 7. Salvar datasets processados
    train_output_path = os.path.join(output_dir, "train.csv")
    val_output_path = os.path.join(output_dir, "validation.csv")
    
    train_df_advanced.to_csv(train_output_path, index=False)
    val_df_advanced.to_csv(val_output_path, index=False)
    
    print(f"Dataset de treino salvo em: {train_output_path}")
    print(f"Dataset de validação salvo em: {val_output_path}")
    
    # 8. Exibir estatísticas das novas features
    print("\n=== Novas features criadas ===")
    new_features = list(set(train_df_advanced.columns) - set(train_df.columns))
    print(f"Total de novas features: {len(new_features)}")
    
    # Agrupar por tipo
    feature_groups = {
        'tfidf': [f for f in new_features if 'refined_tfidf' in f],
        'embedding': [f for f in new_features if 'embedding' in f],
        'topic': [f for f in new_features if 'topic' in f],
        'salary': [f for f in new_features if 'salary' in f],
        'country': [f for f in new_features if 'country' in f or 'country_x' in f],
        'age': [f for f in new_features if 'age_' in f or 'age_x' in f],
        'temporal': [f for f in new_features if any(x in f for x in ['hour_x', 'day_', 'period_'])]
    }
    
    for group, features in feature_groups.items():
        print(f"- {group}: {len(features)} features")
        if features and group in ['tfidf', 'embedding', 'topic']:
            print(f"  Exemplos: {features[:3]}")
    
    print("\nTeste de engenharia de features avançada concluído!")

if __name__ == "__main__":
    main()