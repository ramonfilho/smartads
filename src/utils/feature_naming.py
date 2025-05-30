def standardize_feature_name(name):
    """
    Padroniza nomes de features para garantir consistência em toda pipeline.
    
    Regras:
    1. Remove acentos
    2. Converte para minúsculas
    3. Substitui caracteres especiais por underscore
    4. Remove underscores múltiplos
    5. Trunca em 100 caracteres
    """
    if not isinstance(name, str):
        return str(name)
    
    import unicodedata
    import re
    
    # 1. Remover acentos
    name = ''.join(c for c in unicodedata.normalize('NFD', name) 
                   if unicodedata.category(c) != 'Mn')
    
    # 2. Converter para minúsculas
    name = name.lower()
    
    # 3. Substituir caracteres especiais por underscore
    name = re.sub(r'[^\w\s]', '_', name)
    
    # 4. Substituir espaços por underscore
    name = re.sub(r'\s+', '_', name)
    
    # 5. Remover underscores múltiplos
    name = re.sub(r'_+', '_', name)
    
    # 6. Remover underscores no início/fim
    name = name.strip('_')
    
    # 7. Truncar se muito longo
    if len(name) > 100:
        name = name[:100]
    
    return name

def standardize_dataframe_columns(df):
    """Padroniza todos os nomes de colunas de um DataFrame."""
    rename_dict = {}
    for col in df.columns:
        new_name = standardize_feature_name(col)
        if new_name != col:
            rename_dict[col] = new_name
    
    if rename_dict:
        print(f"  Padronizando {len(rename_dict)} nomes de colunas...")
        return df.rename(columns=rename_dict)
    return df