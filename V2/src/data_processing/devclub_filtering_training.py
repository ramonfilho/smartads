"""
Módulo para filtragem de vendas DevClub - PIPELINE DE TREINO.

Reproduz a célula 17 do notebook DevClub.
Cria dataset com target baseado apenas em produtos DevClub.
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


def criar_dataset_devclub(df_v1_final: pd.DataFrame, df_vendas_unificado: pd.DataFrame) -> pd.DataFrame:
    """
    Cria dataset V1 com target baseado apenas em vendas DevClub.

    Reproduz a lógica da célula 17 do notebook DevClub.

    Args:
        df_v1_final: DataFrame V1 com target original (todos produtos)
        df_vendas_unificado: DataFrame de vendas unificado

    Returns:
        DataFrame V1 com target apenas DevClub
    """
    print("CRIAÇÃO DO DATASET DEVCLUB")
    print("=" * 40)

    # 1. PRODUTOS DEVCLUB A MANTER
    produtos_devclub_manter = [
        'DevClub - Full Stack 2025',
        'DevClub FullStack Pro - OFICIAL',
        'Formação DevClub FullStack Pro - OFICI',
        'DevClub - Full Stack 2025 - EV',
        'DevClub - FS - Vitalício',
        '[Vitalício] Formação DevClub FullStack',
        'Formação DevClub FullStack Pro - COMER',
        'DevClub Vitalício',
        'DevClub 3.0 - 2024',
        '(Desativado) DevClub 3.0 - 2024',
        '(Desativado) DevClub 3.0 - 2024 - Novo'
    ]

    # 2. IDENTIFICAR COMPRADORES DEVCLUB
    df_vendas_devclub = df_vendas_unificado[df_vendas_unificado['produto'].isin(produtos_devclub_manter)].copy()

    df_vendas_devclub['email_clean'] = df_vendas_devclub['email'].apply(normalizar_email)
    emails_compradores_devclub = set(df_vendas_devclub['email_clean'].dropna())

    print(f"Produtos DevClub identificados: {len(produtos_devclub_manter)}")
    print(f"Vendas DevClub: {len(df_vendas_devclub):,}")
    print(f"Emails únicos compradores DevClub: {len(emails_compradores_devclub):,}")

    # 3. CRIAR DATASET DEVCLUB
    df_devclub = df_v1_final.copy()

    # Normalizar emails do dataset de pesquisa
    df_devclub['email_temp'] = df_devclub['E-mail'].apply(normalizar_email)

    # Criar novo target baseado apenas em DevClub
    df_devclub['target_devclub'] = df_devclub['email_temp'].isin(emails_compradores_devclub).astype(int)

    # Remover coluna temporária e target antigo
    df_devclub = df_devclub.drop(columns=['email_temp', 'target'])

    # Renomear para target final
    df_devclub = df_devclub.rename(columns={'target_devclub': 'target'})

    # 4. ESTATÍSTICAS
    total_registros = len(df_devclub)
    leads_qualificados = df_devclub['target'].sum()
    taxa_conversao = (leads_qualificados / total_registros * 100) if total_registros > 0 else 0

    print(f"\nDATASET V1 DEVCLUB:")
    print(f"  Total de registros: {total_registros:,}")
    print(f"  Leads qualificados DevClub: {leads_qualificados:,}")
    print(f"  Taxa de conversão DevClub: {taxa_conversao:.2f}%")
    print(f"  Colunas: {len(df_devclub.columns)}")

    logger.info(f"✅ Dataset DevClub criado com sucesso")

    return df_devclub
