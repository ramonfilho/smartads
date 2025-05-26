#!/usr/bin/env python
"""
Script para diagnosticar onde os 134 matches estão sendo perdidos no pipeline.
"""

import pandas as pd
import os
import sys

# Adicionar o diretório raiz ao path
project_root = "/Users/ramonmoreira/Desktop/smart_ads"
sys.path.insert(0, project_root)

from src.utils.local_storage import connect_to_gcs, list_files_by_extension, categorize_files
from src.preprocessing.email_processing import normalize_emails_in_dataframe, normalize_email
from src.preprocessing.data_matching import match_surveys_with_buyers
from src.preprocessing.data_integration import create_target_variable

def diagnose_match_loss():
    """Diagnostica onde os matches estão sendo perdidos."""
    
    print("DIAGNÓSTICO DE MATCHES PERDIDOS")
    print("=" * 80)
    
    # Conectar ao armazenamento
    raw_data_path = "/Users/ramonmoreira/Desktop/smart_ads/data/raw_data"
    bucket = connect_to_gcs("local_bucket", data_path=raw_data_path)
    
    # Listar arquivos
    file_paths = list_files_by_extension(bucket, prefix="")
    survey_files, buyer_files, utm_files, _ = categorize_files(file_paths)
    
    # 1. CARREGAR DADOS BRUTOS
    print("\n1. CARREGANDO DADOS BRUTOS")
    print("-" * 40)
    
    # Carregar apenas um exemplo de cada tipo para análise
    from src.utils.local_storage import load_csv_or_excel
    
    # Combinar todos os surveys
    all_surveys = []
    for file in survey_files:
        df = load_csv_or_excel(bucket, file)
        if df is not None:
            all_surveys.append(df)
    
    surveys_raw = pd.concat(all_surveys, ignore_index=True)
    print(f"Surveys carregados: {len(surveys_raw):,} registros")
    
    # Combinar todos os buyers
    all_buyers = []
    for file in buyer_files:
        df = load_csv_or_excel(bucket, file)
        if df is not None:
            all_buyers.append(df)
    
    buyers_raw = pd.concat(all_buyers, ignore_index=True)
    print(f"Buyers carregados: {len(buyers_raw):,} registros")
    
    # 2. NORMALIZAR EMAILS
    print("\n2. NORMALIZANDO EMAILS")
    print("-" * 40)
    
    # Encontrar e normalizar emails nos surveys
    email_cols_survey = [col for col in surveys_raw.columns if 'mail' in col.lower()]
    print(f"Colunas de email em surveys: {email_cols_survey}")
    
    if email_cols_survey:
        surveys_raw['email'] = surveys_raw[email_cols_survey[0]]
        surveys_norm = normalize_emails_in_dataframe(surveys_raw.copy())
    else:
        print("ERRO: Nenhuma coluna de email encontrada em surveys!")
        return
    
    # Encontrar e normalizar emails nos buyers
    email_cols_buyer = [col for col in buyers_raw.columns if 'mail' in col.lower() or 'email' in col.lower()]
    print(f"Colunas de email em buyers: {email_cols_buyer}")
    
    if email_cols_buyer:
        buyers_raw['email'] = buyers_raw[email_cols_buyer[0]]
        buyers_norm = normalize_emails_in_dataframe(buyers_raw.copy())
    else:
        print("ERRO: Nenhuma coluna de email encontrada em buyers!")
        return
    
    # 3. EXECUTAR MATCHING
    print("\n3. EXECUTANDO MATCHING")
    print("-" * 40)
    
    # Guardar índices originais
    surveys_norm['original_survey_index'] = surveys_norm.index
    
    # Fazer matching
    matches_df = match_surveys_with_buyers(surveys_norm, buyers_norm)
    print(f"Total de matches encontrados: {len(matches_df):,}")
    
    # 4. ANALISAR MATCHES
    print("\n4. ANÁLISE DETALHADA DOS MATCHES")
    print("-" * 40)
    
    if not matches_df.empty:
        # Verificar quais survey_ids existem no DataFrame
        valid_survey_ids = matches_df['survey_id'].isin(surveys_norm.index)
        print(f"Survey IDs válidos: {valid_survey_ids.sum():,}")
        print(f"Survey IDs inválidos: {(~valid_survey_ids).sum():,}")
        
        # Mostrar exemplos de IDs inválidos
        if (~valid_survey_ids).any():
            invalid_ids = matches_df[~valid_survey_ids]['survey_id'].head(10).tolist()
            print(f"Exemplos de IDs inválidos: {invalid_ids}")
            print(f"Índice máximo em surveys: {surveys_norm.index.max()}")
        
        # Analisar distribuição dos matches por lançamento
        if 'lançamento' in matches_df.columns:
            print("\nMatches por lançamento:")
            print(matches_df['lançamento'].value_counts())
    
    # 5. CRIAR TARGET E VERIFICAR
    print("\n5. CRIANDO VARIÁVEL TARGET")
    print("-" * 40)
    
    surveys_with_target = create_target_variable(surveys_norm, matches_df)
    
    # Verificar quantos targets foram criados
    if 'target' in surveys_with_target.columns:
        targets_created = surveys_with_target['target'].sum()
        print(f"Targets criados: {targets_created:,}")
        print(f"Diferença: {len(matches_df) - targets_created:,}")
        
        # Análise mais detalhada
        print("\n6. DIAGNÓSTICO DA DIFERENÇA")
        print("-" * 40)
        
        # Verificar se há duplicatas de email_norm que podem causar problemas
        email_duplicates = surveys_norm['email_norm'].value_counts()
        duplicated_emails = email_duplicates[email_duplicates > 1]
        
        if not duplicated_emails.empty:
            print(f"Emails duplicados em surveys: {len(duplicated_emails):,}")
            print("Top 5 emails mais duplicados:")
            print(duplicated_emails.head())
            
            # Ver quantos matches são com emails duplicados
            matches_with_dup_emails = matches_df[
                matches_df['survey_id'].isin(
                    surveys_norm[surveys_norm['email_norm'].isin(duplicated_emails.index)].index
                )
            ]
            print(f"\nMatches com emails duplicados: {len(matches_with_dup_emails):,}")
    
    # 7. RASTREAR MATCHES ESPECÍFICOS
    print("\n7. RASTREAMENTO DE MATCHES PERDIDOS")
    print("-" * 40)
    
    # Pegar uma amostra de matches e rastrear o que acontece
    sample_matches = matches_df.head(10)
    
    for idx, match in sample_matches.iterrows():
        survey_id = match['survey_id']
        buyer_id = match['buyer_id']
        
        print(f"\nMatch {idx}:")
        print(f"  Survey ID: {survey_id}")
        print(f"  Buyer ID: {buyer_id}")
        
        # Verificar se o survey_id existe
        if survey_id in surveys_norm.index:
            survey_email = surveys_norm.loc[survey_id, 'email_norm']
            print(f"  Survey email_norm: {survey_email}")
            
            # Verificar se foi marcado como target
            if 'target' in surveys_with_target.columns and survey_id in surveys_with_target.index:
                target_value = surveys_with_target.loc[survey_id, 'target']
                print(f"  Target marcado: {target_value}")
            else:
                print(f"  ❌ Survey ID não encontrado no DataFrame final!")
        else:
            print(f"  ❌ Survey ID não existe no DataFrame de surveys!")
    
    # 8. SALVAR RELATÓRIO
    print("\n8. SALVANDO RELATÓRIO DETALHADO")
    print("-" * 40)
    
    report_path = os.path.join(project_root, "reports", "missing_matches_diagnosis.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("RELATÓRIO DE DIAGNÓSTICO DE MATCHES PERDIDOS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total de matches encontrados: {len(matches_df):,}\n")
        if 'target' in surveys_with_target.columns:
            f.write(f"Total de targets criados: {surveys_with_target['target'].sum():,}\n")
            f.write(f"Diferença (matches perdidos): {len(matches_df) - surveys_with_target['target'].sum():,}\n")
        
        f.write("\n\nDETALHES DOS MATCHES:\n")
        f.write("-" * 40 + "\n")
        
        # Salvar todos os matches com seus status
        for idx, match in matches_df.iterrows():
            survey_id = match['survey_id']
            status = "✓" if survey_id in surveys_norm.index else "✗"
            f.write(f"{status} Match {idx}: Survey ID {survey_id}\n")
    
    print(f"Relatório salvo em: {report_path}")

if __name__ == "__main__":
    diagnose_match_loss()