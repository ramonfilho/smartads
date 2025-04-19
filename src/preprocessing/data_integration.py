import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_target_variable(surveys, matches_df):
    """Cria a variável alvo (target) no dataset de pesquisas.
    
    Args:
        surveys: DataFrame com dados de pesquisas
        matches_df: DataFrame com correspondências
        
    Returns:
        DataFrame com a variável target adicionada
    """
    print("\nCreating target variable...")
    # Cria uma cópia para não modificar o original
    surveys_with_target = surveys.copy()
    
    # Inicializa target com 0
    surveys_with_target['target'] = 0
    
    if not matches_df.empty and 'survey_id' in matches_df.columns:
        match_survey_ids = set(matches_df['survey_id'])
        surveys_with_target.loc[surveys_with_target.index.isin(match_survey_ids), 'target'] = 1
        
        # Adicionar informação de lançamento às surveys com base nas correspondências
        if 'lançamento' in matches_df.columns:
            if 'lançamento' not in surveys_with_target.columns:
                surveys_with_target['lançamento'] = None
            
            # Iterar pelas correspondências para atribuir o lançamento correto
            for _, match in matches_df.iterrows():
                if 'lançamento' in match and not pd.isna(match['lançamento']):
                    surveys_with_target.loc[match['survey_id'], 'lançamento'] = match['lançamento']
        
        conversion_rate = surveys_with_target['target'].mean() * 100
        print(f"Conversion rate: {conversion_rate:.2f}%")
    else:
        print("No matches found - target variable will be all zeros")
    
    return surveys_with_target

def merge_datasets(surveys, utms, buyers=None):
    """Mescla os datasets de pesquisas, UTMs e compradores.
    
    Args:
        surveys: DataFrame com dados de pesquisas
        utms: DataFrame com dados de UTM
        buyers: DataFrame com dados de compradores (opcional)
        
    Returns:
        DataFrame mesclado
    """
    print("\nMerging datasets...")
    merged_data = surveys.copy()
    
    # Mesclar com dados de UTM
    if not utms.empty and 'email_norm' in utms.columns:
        merged_data = pd.merge(
            surveys,
            utms,
            on='email_norm',
            how='left',
            suffixes=('', '_utm')
        )
        print(f"Merged survey data with UTM data")
    
    # Adicionar informação de lançamento se ainda não foi adicionada
    if buyers is not None and 'lançamento' not in merged_data.columns and 'lançamento' in buyers.columns:
        # Criar um dicionário mapeando email_norm para lançamento
        email_to_launch = dict(zip(buyers['email_norm'], buyers['lançamento']))
        
        # Adicionar coluna de lançamento ao DataFrame mesclado
        merged_data['lançamento'] = merged_data['email_norm'].map(email_to_launch)
        print("Added launch information from buyer data")
    
    print(f"Final merged dataset: {merged_data.shape[0]} rows, {merged_data.shape[1]} columns")
    return merged_data

def split_data(merged_data, output_dir="datasets/split", test_size=0.3, random_state=42):
    """Divide os dados em conjuntos de treino, validação e teste.
    
    Args:
        merged_data: DataFrame a ser dividido
        output_dir: Diretório para salvar os datasets
        test_size: Proporção dos dados para teste e validação
        random_state: Semente para reprodutibilidade
        
    Returns:
        Tuple de DataFrames (train, validation, test)
    """
    print("\nSplitting data to avoid data leakage...")
    
    # Garantir que temos a coluna target
    if 'target' not in merged_data.columns:
        print("WARNING: Target column not found. Creating with value 0.")
        merged_data['target'] = 0
    
    # Verificar se há valores nulos na coluna target
    if merged_data['target'].isnull().any():
        print(f"WARNING: Found {merged_data['target'].isnull().sum()} null values in target column. Replacing with 0.")
        merged_data['target'].fillna(0, inplace=True)
    
    # Fazer o split estratificado dos dados (70% treino, 15% validação, 15% teste)
    train_df, temp_df = train_test_split(
        merged_data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=merged_data['target']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=random_state,
        stratify=temp_df['target']
    )
    
    # Verificar as proporções
    print(f"\nSplit: train={len(train_df)} ({len(train_df)/len(merged_data):.1%}), " + 
          f"validation={len(val_df)} ({len(val_df)/len(merged_data):.1%}), " + 
          f"test={len(test_df)} ({len(test_df)/len(merged_data):.1%})")
    
    print("\nTarget proportions:")
    print(f"Original: {merged_data['target'].mean():.4f}")
    print(f"Train: {train_df['target'].mean():.4f}")
    print(f"Validation: {val_df['target'].mean():.4f}")
    print(f"Test: {test_df['target'].mean():.4f}")
    
    # Criar diretório para os datasets separados
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar os datasets separados
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/validation.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"Datasets saved in '{output_dir}' directory")
    
    return train_df, val_df, test_df