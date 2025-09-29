"""
Módulo de engenharia de features para o pipeline de lead scoring.
Mantém a lógica EXATA do notebook original para garantir reprodutibilidade.
"""

import pandas as pd
import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def normalizar_telefone_robusto(telefone):
    """
    Normaliza telefone considerando notação científica e padrões brasileiros.

    Função EXATA copiada da Seção 18 do notebook original.
    CORREÇÃO: Trata float64 do Excel que adiciona .0 extra
    """
    if pd.isna(telefone):
        return None

    # Converter para string e lidar com notação científica
    # CORREÇÃO: Se é float, converter diretamente para int para remover .0
    if isinstance(telefone, float):
        tel_str = str(int(telefone))
    else:
        tel_str = str(telefone)

    # Se está em notação científica, converter para número inteiro
    if 'e+' in tel_str.lower() or 'E+' in tel_str:
        try:
            tel_str = str(int(float(tel_str)))
        except:
            pass

    # Extrair apenas dígitos
    digitos = re.sub(r'\D', '', tel_str)

    if len(digitos) < 8:
        return None

    # Remover código do país (55) se presente
    if digitos.startswith('55') and len(digitos) > 10:
        digitos = digitos[2:]

    # Verificar se é um telefone válido brasileiro
    if len(digitos) in [10, 11]:  # DDD + 8 ou 9 dígitos
        return digitos
    elif len(digitos) in [8, 9]:  # Sem DDD
        return digitos

    return None


def validar_email_robusto(email):
    """
    Valida email com regex rigoroso.

    Função EXATA copiada da Seção 18 do notebook original.
    """
    if pd.isna(email):
        return False

    email_str = str(email).strip().lower()

    # Regex básico para email
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    return bool(re.match(pattern, email_str))


def validar_nome_robusto(nome):
    """
    Valida se nome não é apenas números ou caracteres especiais.

    Função EXATA copiada da Seção 18 do notebook original.
    """
    if pd.isna(nome):
        return False

    nome_str = str(nome).strip()

    # Verificar se tem pelo menos algumas letras
    tem_letras = bool(re.search(r'[a-zA-ZÀ-ÿ]', nome_str))

    # Verificar se não é só números
    nao_so_numeros = not nome_str.replace(' ', '').replace('.', '').replace('-', '').isdigit()

    return tem_letras and nao_so_numeros and len(nome_str) >= 2


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas e remove colunas desnecessárias.

    Segue lógica EXATA da Seção 18 do notebook original:
    1. Features temporais (dia_semana)
    2. Features de qualidade (nome, email, telefone)
    3. Remove colunas originais após criar features

    Args:
        df: DataFrame com colunas para feature engineering

    Returns:
        DataFrame com features derivadas
    """
    # Print do cabeçalho para comparação com notebook
    logger.info("FEATURE ENGINEERING COMPLETO - 4 DATASETS")
    logger.info("=" * 45)

    df_fe = df.copy()

    logger.info(f"\nProcessando DATASET V1 DEVCLUB...")
    logger.info(f"Registros: {len(df_fe):,}")
    logger.info(f"Colunas antes: {len(df_fe.columns)}")
    logger.info(f"Nomes das colunas antes:")
    for i, col in enumerate(df_fe.columns, 1):
        logger.info(f"  {i:2d}. {col}")

    # 1. FEATURES TEMPORAIS
    if 'Data' in df_fe.columns:
        df_fe['Data'] = pd.to_datetime(df_fe['Data'], errors='coerce')
        df_fe['dia_semana'] = df_fe['Data'].dt.dayofweek

    # 2. FEATURES DE QUALIDADE DOS IDENTIFICADORES

    # Nome
    if 'Nome Completo' in df_fe.columns:
        df_fe['nome_comprimento'] = df_fe['Nome Completo'].astype(str).str.len()
        df_fe['nome_tem_sobrenome'] = df_fe['Nome Completo'].astype(str).str.split().str.len() >= 2
        df_fe['nome_valido'] = df_fe['Nome Completo'].apply(validar_nome_robusto)

    # Email
    if 'E-mail' in df_fe.columns:
        df_fe['email_valido'] = df_fe['E-mail'].apply(validar_email_robusto)

    # Telefone
    if 'Telefone' in df_fe.columns:
        df_fe['telefone_normalizado'] = df_fe['Telefone'].apply(normalizar_telefone_robusto)
        df_fe['telefone_valido'] = df_fe['telefone_normalizado'].notna()
        df_fe['telefone_comprimento'] = df_fe['telefone_normalizado'].astype(str).str.len()

        # ANÁLISE DE TELEFONES VÁLIDOS POR ARQUIVO DE ORIGEM (simulação para ambiente de produção)
        logger.info(f"\n% de telefones válidos por arquivo de origem:")
        total_registros = len(df_fe)
        telefones_validos = df_fe['telefone_valido'].sum()
        pct_valido = (telefones_validos / total_registros * 100) if total_registros > 0 else 0
        logger.info(f"  Lead score LF 24.xlsx: {telefones_validos:,}/{total_registros:,} ({pct_valido:.1f}%)")

    # 3. REMOVER COLUNAS DESNECESSÁRIAS
    colunas_remover = [
        'aba_origem', 'arquivo_origem', 'Data',
        'Nome Completo', 'E-mail', 'Telefone', 'telefone_normalizado'
    ]

    # Verificar quais colunas existem antes de remover
    colunas_existentes = [col for col in colunas_remover if col in df_fe.columns]

    if colunas_existentes:
        df_fe = df_fe.drop(columns=colunas_existentes)
        logger.info(f"Colunas removidas: {len(colunas_existentes)}")
        for col in colunas_existentes:
            logger.info(f"  - {col}")

    logger.info(f"Colunas depois: {len(df_fe.columns)}")
    logger.info(f"Nomes das colunas depois:")
    for i, col in enumerate(df_fe.columns, 1):
        logger.info(f"  {i:2d}. {col}")

    # Mostrar features criadas
    features_criadas = []
    for col in ['dia_semana', 'nome_comprimento', 'nome_tem_sobrenome', 'nome_valido',
                'email_valido', 'telefone_valido', 'telefone_comprimento']:
        if col in df_fe.columns:
            features_criadas.append(col)

    # Estatísticas das features criadas
    logger.info(f"\nEstatísticas das features criadas:")
    if 'nome_valido' in df_fe.columns:
        nome_valido_count = df_fe['nome_valido'].sum()
        nome_valido_pct = df_fe['nome_valido'].mean() * 100
        logger.info(f"Nome válido: {nome_valido_count:,} ({nome_valido_pct:.1f}%)")

    if 'nome_tem_sobrenome' in df_fe.columns:
        sobrenome_count = df_fe['nome_tem_sobrenome'].sum()
        sobrenome_pct = df_fe['nome_tem_sobrenome'].mean() * 100
        logger.info(f"Nome com sobrenome: {sobrenome_count:,} ({sobrenome_pct:.1f}%)")

    if 'email_valido' in df_fe.columns:
        email_count = df_fe['email_valido'].sum()
        email_pct = df_fe['email_valido'].mean() * 100
        logger.info(f"Email válido: {email_count:,} ({email_pct:.1f}%)")

    if 'telefone_valido' in df_fe.columns:
        telefone_count = df_fe['telefone_valido'].sum()
        telefone_pct = df_fe['telefone_valido'].mean() * 100
        logger.info(f"Telefone válido: {telefone_count:,} ({telefone_pct:.1f}%)")

    # Distribuição da feature temporal
    if 'dia_semana' in df_fe.columns:
        logger.info(f"\nDistribuição da feature temporal:")
        dia_semana_counts = df_fe['dia_semana'].value_counts().sort_index()
        nomes_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        for dia, count in dia_semana_counts.items():
            pct = (count / len(df_fe)) * 100
            logger.info(f"  {dia} ({nomes_dias[dia]}): {count:,} ({pct:.1f}%)")

    return df_fe


def get_feature_engineering_summary(df_original: pd.DataFrame, df_fe: pd.DataFrame) -> Dict:
    """
    Gera resumo das features criadas no feature engineering.

    Args:
        df_original: DataFrame original antes do FE
        df_fe: DataFrame após feature engineering

    Returns:
        Dicionário com estatísticas das features criadas
    """
    summary = {
        'original_columns': len(df_original.columns),
        'fe_columns': len(df_fe.columns),
        'columns_added': len(df_fe.columns) - len(df_original.columns),
        'rows': len(df_fe)
    }

    # Estatísticas das features criadas (se existirem)
    if 'nome_valido' in df_fe.columns:
        summary['nome_valido_pct'] = df_fe['nome_valido'].mean() * 100
        summary['nome_tem_sobrenome_pct'] = df_fe['nome_tem_sobrenome'].mean() * 100

    if 'email_valido' in df_fe.columns:
        summary['email_valido_pct'] = df_fe['email_valido'].mean() * 100

    if 'telefone_valido' in df_fe.columns:
        summary['telefone_valido_pct'] = df_fe['telefone_valido'].mean() * 100

    # Distribuição da feature temporal
    if 'dia_semana' in df_fe.columns:
        summary['dia_semana_dist'] = df_fe['dia_semana'].value_counts().sort_index().to_dict()

    return summary


def get_removed_columns_list() -> List[str]:
    """
    Lista de colunas que são removidas no feature engineering.

    Returns:
        Lista com nomes das colunas removidas
    """
    return [
        'aba_origem', 'arquivo_origem', 'Data',
        'Nome Completo', 'E-mail', 'Telefone', 'telefone_normalizado'
    ]


def get_created_features_list() -> List[str]:
    """
    Lista de features criadas no feature engineering.

    Returns:
        Lista com nomes das features criadas
    """
    return [
        'dia_semana',
        'nome_comprimento', 'nome_tem_sobrenome', 'nome_valido',
        'email_valido',
        'telefone_valido', 'telefone_comprimento'
    ]