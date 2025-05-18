# Script para gerar um arquivo válido de colunas de treino
import pandas as pd
import os

# Caminhos
project_root = "/Users/ramonmoreira/desktop/smart_ads"
# Ajuste o caminho abaixo para o local dos dados de treino que contêm todas as colunas finais
train_data_path = os.path.join(project_root, "data/02_3_processed/train.csv")
output_path = os.path.join(project_root, "inference/params/04_train_columns.csv")

# Carregar dados de treino (ou apenas os nomes das colunas)
try:
    # Tentar carregar apenas os cabeçalhos para economizar memória
    train_cols = pd.read_csv(train_data_path, nrows=0).columns.tolist()
    print(f"Carregadas {len(train_cols)} colunas do dataset de treino.")
    
    # Criar DataFrame com colunas como uma linha (formato correto)
    cols_df = pd.DataFrame([train_cols], columns=train_cols)
    
    # Ou criar DataFrame com colunas como coluna única
    # cols_df = pd.DataFrame({"column_name": train_cols})
    
    # Salvar
    cols_df.to_csv(output_path, index=False)
    print(f"Arquivo de colunas salvo em: {output_path}")
    
except Exception as e:
    print(f"Erro ao gerar arquivo de colunas: {e}")