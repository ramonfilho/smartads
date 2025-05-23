"""
Módulo para integração de diferentes fontes de dados.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_target_variable(surveys_df, matches_df):
    """
    Cria a variável alvo com base nas correspondências de compras.
    
    Args:
        surveys_df: DataFrame com respostas da pesquisa
        matches_df: DataFrame com correspondências entre pesquisas e compradores
        
    Returns:
        DataFrame com a variável alvo adicionada
    """
    # Se o surveys_df estiver vazio, retornar um DataFrame vazio com as colunas necessárias
    if surveys_df.empty:
        print("No surveys data - creating empty DataFrame with target variable")
        return pd.DataFrame(columns=['email', 'email_norm', 'target'])
    
    # Se não houver correspondências, todos os registros são negativos
    if matches_df.empty:
        print("No matches found - target variable will be all zeros")
        surveys_df['target'] = 0
        return surveys_df
    
    # Copiar o DataFrame para não modificar o original
    result_df = surveys_df.copy()
    
    # Adicionar coluna de target inicializada com 0 (não converteu)
    result_df['target'] = 0
    
    # Marcar os registros correspondentes como positivos (converteu)
    for _, match in matches_df.iterrows():
        survey_id = match['survey_id']
        result_df.loc[survey_id, 'target'] = 1
    
    print(f"Created target variable: {result_df['target'].sum()} positive examples out of {len(result_df)}")
    return result_df

def merge_datasets(surveys_df, utm_df, buyers_df):
    """
    Mescla as diferentes fontes de dados em um único dataset.
    
    Args:
        surveys_df: DataFrame com respostas da pesquisa e target
        utm_df: DataFrame com dados de UTM
        buyers_df: DataFrame com dados de compradores
        
    Returns:
        DataFrame combinado
    """
    print("Merging datasets...")
    
    # Verificar se temos dados suficientes para mesclar
    if surveys_df.empty or 'email_norm' not in surveys_df.columns:
        print("WARNING: Survey data is empty or missing email_norm column")
        print("Creating simple dataset with available data...")
        
        # Se temos dados de UTM com email, usar esses
        if not utm_df.empty and 'email' in utm_df.columns:
            print("Using UTM data as base")
            result_df = utm_df.copy()
            # Adicionar coluna target (todos 0 por padrão)
            result_df['target'] = 0
            
            # Tentar normalizar emails do UTM se não estiver já normalizado
            if 'email_norm' not in result_df.columns and 'email' in result_df.columns:
                from ..preprocessing.email_processing import normalize_email
                result_df['email_norm'] = result_df['email'].apply(normalize_email)
                
            # Marcar compradores se possível
            if not buyers_df.empty and 'email_norm' in buyers_df.columns:
                buyer_emails = set(buyers_df['email_norm'].dropna())
                if 'email_norm' in result_df.columns:
                    result_df['target'] = result_df['email_norm'].apply(
                        lambda email: 1 if email in buyer_emails else 0
                    )
        
        # Se temos apenas dados de compradores, usar esses
        elif not buyers_df.empty:
            print("Using buyer data as base")
            result_df = buyers_df.copy()
            result_df['target'] = 1  # Todos os registros são compradores
        
        # Se não temos nada, criar DataFrame vazio
        else:
            print("No data available - creating empty dataset")
            result_df = pd.DataFrame(columns=['email', 'email_norm', 'target'])
        
        print(f"Final merged dataset: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
        return result_df
    
    # Mesclar pesquisas com UTM usando email_norm
    if not utm_df.empty and 'email' in utm_df.columns:
        # Normalizar emails do UTM se ainda não estiverem
        if 'email_norm' not in utm_df.columns:
            from ..preprocessing.email_processing import normalize_email
            utm_df['email_norm'] = utm_df['email'].apply(normalize_email)
        
        merged_df = pd.merge(
            surveys_df,
            utm_df,
            on='email_norm',
            how='left',
            suffixes=('', '_utm')
        )
        print(f"Merged surveys with UTM data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    else:
        merged_df = surveys_df.copy()
        print("No UTM data available or missing email column")
    
    # Mesclar com dados de compradores, se disponíveis
    if not buyers_df.empty and 'email_norm' in buyers_df.columns:
        # Selecionar colunas úteis de compradores para não duplicar colunas desnecessárias
        buyer_cols = ['email_norm']
        for col in buyers_df.columns:
            if col not in merged_df.columns and col != 'email_norm':
                buyer_cols.append(col)
        
        buyers_subset = buyers_df[buyer_cols].copy()
        
        # Realizar a mesclagem
        final_df = pd.merge(
            merged_df,
            buyers_subset,
            on='email_norm',
            how='left',
            suffixes=('', '_buyer')
        )
        print(f"Merged with buyer data: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
    else:
        final_df = merged_df.copy()
        print("No buyer data available or missing email_norm column")
    
    print(f"Final merged dataset: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
    return final_df

def split_data(df, output_dir, test_size=0.3, val_size=0.5, stratify=True, random_state=42):
    """
    Divide os dados em conjuntos de treino, validação e teste.
    
    Args:
        df: DataFrame a ser dividido
        output_dir: Diretório para salvar os conjuntos
        test_size: Proporção do conjunto de teste
        val_size: Proporção do conjunto de validação dentro do conjunto de teste
        stratify: Se deve estratificar por classe
        random_state: Semente aleatória para reprodutibilidade
        
    Returns:
        Tuple com (train_df, val_df, test_df)
    """
    print("Splitting data to avoid data leakage...")
    
    # Verificar se temos dados suficientes para dividir
    if df.shape[0] == 0:
        print("WARNING: Empty dataset - cannot split. Creating empty dataframes.")
        empty_df = pd.DataFrame(columns=df.columns)
        return empty_df, empty_df, empty_df
    
    # Criar os diretórios se não existirem
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar se temos target para estratificar
    if stratify and 'target' in df.columns and df['target'].nunique() > 1:
        print("Using stratified split based on target variable")
        strat_col = df['target']
    else:
        print("Using random split (no stratification)")
        strat_col = None
    
    # Primeira divisão: treino vs. (validação + teste)
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_col
    )
    
    # Segunda divisão: validação vs. teste
    if stratify and 'target' in temp_df.columns and temp_df['target'].nunique() > 1:
        strat_col_temp = temp_df['target']
    else:
        strat_col_temp = None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        random_state=random_state,
        stratify=strat_col_temp
    )
    
    # Salvar os conjuntos
    print(f"Train set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    
    print(f"Validation set: {val_df.shape[0]} rows, {val_df.shape[1]} columns")
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    
    print(f"Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    return train_df, val_df, test_df