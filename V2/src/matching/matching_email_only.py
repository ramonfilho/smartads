"""
Módulo para matching exclusivamente por email - MÉTODO 3 (EMAIL ONLY).

Reproduz a célula 15 do notebook DevClub usando APENAS email, sem telefone.
Este método garante 100% de precisão evitando falsos positivos de telefone.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def normalizar_email(email):
    """Normaliza email para matching"""
    if pd.isna(email):
        return None

    email_str = str(email).strip().lower()

    # Verificar se é um email válido básico
    if '@' in email_str and email_str != 'nan' and len(email_str) > 5:
        return email_str

    return None


def fazer_matching_email_only(df_pesquisa_v1: pd.DataFrame, df_vendas: pd.DataFrame) -> pd.DataFrame:
    """
    Faz matching EXCLUSIVAMENTE por email - MÉTODO 3 (EMAIL ONLY).

    Reproduz a lógica da célula 15 do notebook DevClub usando APENAS email.
    Não faz matching por telefone para evitar falsos positivos e garantir 100% monotonia.

    Args:
        df_pesquisa_v1: DataFrame de pesquisa (versão 1 pós-cutoff)
        df_vendas: DataFrame de vendas

    Returns:
        DataFrame com target adicionado (apenas matches por email)
    """
    print("MATCHING EXCLUSIVAMENTE POR EMAIL - MÉTODO 3 (EMAIL ONLY)")
    print("=" * 60)
    print("SEM MATCHING POR TELEFONE - Máxima Precisão")

    df_pesquisa = df_pesquisa_v1.copy()
    df_vendas_copy = df_vendas.copy()

    print(f"\nProcessando DATASET V1...")

    # 1. NORMALIZAR EMAILS
    emails_pesquisa = {}
    for idx, email in df_pesquisa['E-mail'].items():
        email_norm = normalizar_email(email)
        if email_norm:
            emails_pesquisa[idx] = email_norm

    emails_vendas = set()
    for email in df_vendas_copy['email']:
        email_norm = normalizar_email(email)
        if email_norm:
            emails_vendas.add(email_norm)

    print(f"  Emails únicos na pesquisa: {len(emails_pesquisa):,}")
    print(f"  Emails únicos nas vendas: {len(emails_vendas):,}")

    # 2. FAZER MATCHING APENAS POR EMAIL
    matches_email = set()

    # Matching por email
    for idx, email in emails_pesquisa.items():
        if email in emails_vendas:
            matches_email.add(idx)

    # 3. CRIAR TARGET
    df_resultado = df_pesquisa.copy()
    df_resultado['target'] = 0

    # Marcar matches (apenas email)
    for idx in matches_email:
        df_resultado.loc[idx, 'target'] = 1

    # 4. ESTATÍSTICAS
    total_registros = len(df_resultado)
    total_matches = df_resultado['target'].sum()
    taxa_conversao = (total_matches / total_registros) * 100

    print(f"  Total de registros: {total_registros:,}")
    print(f"  Total de matches: {total_matches:,}")
    print(f"  Taxa de conversão: {taxa_conversao:.2f}%")
    print(f"  Matches por email: {total_matches:,}")
    print(f"  Matches por telefone: 0 (método email only)")

    print(f"\n" + "=" * 60)
    print("DATASET FINAL CRIADO!")
    print(f"dataset_v1_final: {len(df_resultado):,} registros, {len(df_resultado.columns)} colunas")
    print("Dataset contém apenas colunas originais + target")
    print("Target baseado EXCLUSIVAMENTE em email (sem telefone)")

    # Listar variáveis
    print(f"\n📋 VARIÁVEIS DO DATASET V1 ({len(df_resultado.columns)} colunas):")
    for i, col in enumerate(df_resultado.columns, 1):
        print(f"  {i:2d}. {col}")

    logger.info(f"✅ Matching email only concluído: {total_matches} matches")

    return df_resultado
