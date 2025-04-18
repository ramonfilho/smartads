import pandas as pd
import os
from src.utils.cloud_storage import connect_to_gcs, list_files_by_extension, categorize_files
from src.preprocessing.email_processing import normalize_emails_in_dataframe
from src.preprocessing.data_loader import load_survey_files, load_buyer_files, load_utm_files
from src.preprocessing.data_matching import match_surveys_with_buyers
from src.preprocessing.data_integration import create_target_variable, merge_datasets, split_data

def main():
    """Pipeline principal para coleta e integração de dados."""
    # 1. Conectar ao Google Cloud Storage
    print("Setting up connection to Google Cloud Storage...")
    bucket = connect_to_gcs("new_times")
    
    # 2. Listar e categorizar arquivos
    file_paths = list_files_by_extension(bucket, prefix="raw_data/")
    print(f"Found {len(file_paths)} files")
    
    # 3. Categorizar arquivos por tipo e lançamento
    print("\nCategorizing files by type and launch...")
    survey_files, buyer_files, utm_files, all_files_by_launch = categorize_files(file_paths)
    
    # Mostrar informações de categorização
    print(f"Survey files: {len(survey_files)}")
    print(f"Buyer files: {len(buyer_files)}")
    print(f"UTM files: {len(utm_files)}")
    
    for launch_id, files in all_files_by_launch.items():
        if files:
            print(f"{launch_id} files: {len(files)}")
    
    # 4. Carregar dados
    survey_dfs, survey_launch_data = load_survey_files(bucket, survey_files)
    buyer_dfs, buyer_launch_data = load_buyer_files(bucket, buyer_files)
    utm_dfs, utm_launch_data = load_utm_files(bucket, utm_files)
    
    # 5. Combinar datasets
    print("\nCombining datasets...")
    surveys = pd.concat(survey_dfs, ignore_index=True) if survey_dfs else pd.DataFrame()
    buyers = pd.concat(buyer_dfs, ignore_index=True) if buyer_dfs else pd.DataFrame()
    utms = pd.concat(utm_dfs, ignore_index=True) if utm_dfs else pd.DataFrame()
    
    print(f"Survey data: {surveys.shape[0]} rows, {surveys.shape[1]} columns")
    print(f"Buyer data: {buyers.shape[0]} rows, {buyers.shape[1]} columns")
    print(f"UTM data: {utms.shape[0]} rows, {utms.shape[1]} columns")
    
    # 6. Documentar estrutura de dados
    print("\nDocumenting data structure...")
    for name, df in {"Surveys": surveys, "Buyers": buyers, "UTMs": utms}.items():
        if not df.empty:
            print(f"\n{name} structure:")
            print(f"  - Columns: {df.shape[1]}")
            print(f"  - Data types: {df.dtypes.value_counts().to_dict()}")
            print(f"  - Sample columns: {', '.join(df.columns[:5])}")
    
    # 7. Normalizar emails
    print("\nNormalizing email addresses...")
    surveys = normalize_emails_in_dataframe(surveys)
    
    if not buyers.empty and 'email' in buyers.columns:
        buyers = normalize_emails_in_dataframe(buyers)
    else:
        print("Warning: Cannot normalize emails in buyers dataframe - empty or missing email column")
        if buyers.empty:
            buyers = pd.DataFrame(columns=['email', 'email_norm'])
    
    if not utms.empty and 'email' in utms.columns:
        utms = normalize_emails_in_dataframe(utms)
    else:
        print("Warning: Cannot normalize emails in UTM dataframe - empty or missing email column")
    
    # 8. Correspondência de pesquisas com dados de compradores
    matches_df = match_surveys_with_buyers(surveys, buyers)
    
    # 9. Criar variável alvo
    surveys_with_target = create_target_variable(surveys, matches_df)
    
    # 10. Mesclar datasets
    merged_data = merge_datasets(surveys_with_target, utms, buyers)
    
    # 11. Estatísticas de lançamento
    if 'lançamento' in merged_data.columns:
        launch_counts = merged_data['lançamento'].value_counts(dropna=False)
        print("\nRegistros por lançamento:")
        for launch, count in launch_counts.items():
            launch_str = "Sem lançamento identificado" if pd.isna(launch) else launch
            print(f"  - {launch_str}: {count} registros")
    
    # 12. Split dos dados para evitar vazamento
    train_df, val_df, test_df = split_data(merged_data, "../data/split")
    
    print("\nData collection and integration completed!")
    
if __name__ == "__main__":
    main()