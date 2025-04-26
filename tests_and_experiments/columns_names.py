import pandas as pd
import numpy as np

# Ler o CSV
df = pd.read_csv("/Users/ramonmoreira/desktop/smart_ads/data/02_3_processed_text_code6/validation.csv")

# Criar listas para nomes e tipos
column_names = []
column_types = []

# Preencher as listas
for col in df.columns:
    column_names.append(col)
    column_types.append(str(df[col].dtype))

# Criar DataFrame com os resultados
columns_df = pd.DataFrame({
    'column_name': column_names,
    'column_type': column_types
})

# Salvar como CSV
columns_df.to_csv("/Users/ramonmoreira/desktop/smart_ads/columns.csv", index=False)