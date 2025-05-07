import os
import pandas as pd
import numpy as np

# Configuração de caminhos
dir1 = "data/03_3_feature_selection_text_code6"
dir2 = "data/03_4_feature_selection_final"
dataset_types = ["train", "validation", "test"]

print(f"Comparando features entre diretórios:")
print(f"- {dir1}")
print(f"- {dir2}")

# Função para carregar arquivos
def load_csv_if_exists(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"Arquivo não encontrado: {file_path}")
        return None

# Loop pelos tipos de dataset
for dataset_type in dataset_types:
    print(f"\n=== Comparando {dataset_type}.csv ===")
    
    # Carregar arquivos
    path1 = os.path.join(dir1, f"{dataset_type}.csv")
    path2 = os.path.join(dir2, f"{dataset_type}.csv")
    
    df1 = load_csv_if_exists(path1)
    df2 = load_csv_if_exists(path2)
    
    if df1 is None or df2 is None:
        print(f"Não foi possível comparar {dataset_type}.csv")
        continue
    
    # Obter colunas de cada arquivo
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    # Computar estatísticas
    total_cols1 = len(cols1)
    total_cols2 = len(cols2)
    common_cols = cols1.intersection(cols2)
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    # Exibir resumo
    print(f"Total de features em {dir1}: {total_cols1}")
    print(f"Total de features em {dir2}: {total_cols2}")
    print(f"Features comuns: {len(common_cols)}")
    
    # Exibir discrepâncias
    if len(only_in_1) > 0:
        print(f"\nFeatures exclusivas de {dir1} ({len(only_in_1)}):")
        for i, col in enumerate(sorted(only_in_1)):
            print(f"  {i+1}. {col}")
    
    if len(only_in_2) > 0:
        print(f"\nFeatures exclusivas de {dir2} ({len(only_in_2)}):")
        for i, col in enumerate(sorted(only_in_2)):
            print(f"  {i+1}. {col}")
    
    # Verificar se existem diferenças nos dados para as colunas comuns
    if len(common_cols) > 0:
        print("\nVerificando diferenças nos valores das colunas comuns...")
        
        mismatch_found = False
        for col in common_cols:
            # Verificar se os tipos de dados são compatíveis
            if df1[col].dtype != df2[col].dtype:
                print(f"  - Coluna '{col}' tem tipos diferentes: {df1[col].dtype} vs {df2[col].dtype}")
                mismatch_found = True
                continue
            
            # Verificar se os valores são iguais
            # Lidando com NaN apropriadamente
            if df1[col].dtype == 'float64' or df2[col].dtype == 'float64':
                # Para colunas numéricas, comparar com NaN considerado igual a NaN
                is_equal = np.array_equal(df1[col].fillna(np.nan), df2[col].fillna(np.nan), equal_nan=True)
            else:
                # Para outros tipos, comparar com valores missing considerados iguais
                is_equal = df1[col].equals(df2[col])
            
            if not is_equal:
                print(f"  - Valores diferentes na coluna '{col}'")
                mismatch_found = True
        
        if not mismatch_found:
            print("  Todas as colunas comuns têm valores idênticos.")

print("\n=== Resumo da Comparação ===")
print(f"Foram analisados {len(dataset_types)} tipos de datasets entre os diretórios.")
print(f"Verifique as discrepâncias reportadas acima.")