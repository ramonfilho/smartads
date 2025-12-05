"""
Módulo para classificação de campanhas em COM_ML vs SEM_ML.

Identifica campanhas de captação que usaram ML vs outras abordagens.

Lógica de classificação:
1. Filtro base: Deve conter "DEVLF | CAP | FRIO" (campanhas de captação)
2. COM_ML: Contém "MACHINE LEARNING"
3. SEM_ML: Outros padrões (ESCALA SCORE, FAIXA A, etc.)
4. EXCLUIR: Não contém filtro base (não é campanha de captação)
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def is_captacao_campaign(campaign_name: str) -> bool:
    """
    Verifica se é campanha de captação para lançamento.

    Args:
        campaign_name: Nome da campanha

    Returns:
        True se contém "DEVLF | CAP | FRIO"

    Examples:
        >>> is_captacao_campaign("DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING")
        True
        >>> is_captacao_campaign("DEVLF | AQUECIMENTO | FASE 01")
        False
    """
    if not campaign_name or pd.isna(campaign_name):
        return False

    campaign_lower = str(campaign_name).lower()
    return 'devlf | cap | frio' in campaign_lower


def classify_campaign(campaign_name: str) -> str:
    """
    Classifica campanha de captação em COM_ML, SEM_ML ou EXCLUIR.

    Lógica:
    1. Se não contém "DEVLF | CAP | FRIO" → 'EXCLUIR' (não é de captação)
    2. Se contém "MACHINE LEARNING" → 'COM_ML'
    3. Senão (ex: "ESCALA SCORE", "FAIXA A") → 'SEM_ML'

    Args:
        campaign_name: Nome da campanha

    Returns:
        'COM_ML', 'SEM_ML' ou 'EXCLUIR'

    Examples:
        >>> classify_campaign("DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2")
        'COM_ML'
        >>> classify_campaign("DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2")
        'SEM_ML'
        >>> classify_campaign("DEVLF | AQUECIMENTO | FASE 01 | ...")
        'EXCLUIR'
        >>> classify_campaign(None)
        'EXCLUIR'
    """
    if not campaign_name or pd.isna(campaign_name):
        return 'EXCLUIR'

    campaign_lower = str(campaign_name).lower()

    # 1. Verificar se é campanha de captação
    if 'devlf | cap | frio' not in campaign_lower:
        return 'EXCLUIR'

    # 2. Classificar COM_ML vs SEM_ML
    if 'machine learning' in campaign_lower or '| ml |' in campaign_lower:
        return 'COM_ML'
    else:
        return 'SEM_ML'


def add_ml_classification(df: pd.DataFrame, campaign_col: str = 'campaign') -> pd.DataFrame:
    """
    Adiciona coluna 'ml_type' ao DataFrame e filtra campanhas excluídas.

    Args:
        df: DataFrame com dados de leads
        campaign_col: Nome da coluna que contém o nome da campanha

    Returns:
        DataFrame filtrado com nova coluna 'ml_type'
        (apenas campanhas COM_ML e SEM_ML, EXCLUIR removido)

    Examples:
        >>> df = pd.DataFrame({
        ...     'campaign': [
        ...         'DEVLF | CAP | FRIO | MACHINE LEARNING',
        ...         'DEVLF | CAP | FRIO | ESCALA SCORE',
        ...         'DEVLF | AQUECIMENTO | ...'
        ...     ]
        ... })
        >>> result = add_ml_classification(df)
        >>> len(result)  # Apenas 2 (EXCLUIR removido)
        2
        >>> list(result['ml_type'])
        ['COM_ML', 'SEM_ML']
    """
    if campaign_col not in df.columns:
        raise ValueError(f"Coluna '{campaign_col}' não encontrada no DataFrame")

    logger.info("🏷️ Classificando campanhas...")

    # Adicionar coluna ml_type
    df['ml_type'] = df[campaign_col].apply(classify_campaign)

    # Contar por tipo antes de filtrar
    type_counts = df['ml_type'].value_counts()
    logger.info(f"   Total por tipo:")
    for ml_type, count in type_counts.items():
        logger.info(f"     {ml_type}: {count} leads")

    # Filtrar apenas campanhas de captação (COM_ML ou SEM_ML)
    before_count = len(df)
    df_filtered = df[df['ml_type'] != 'EXCLUIR'].copy()
    after_count = len(df_filtered)

    excluded_count = before_count - after_count
    if excluded_count > 0:
        logger.info(f"   ⚠️ {excluded_count} leads de campanhas não-captação foram excluídos")

    # Calcular percentuais
    if after_count > 0:
        com_ml = len(df_filtered[df_filtered['ml_type'] == 'COM_ML'])
        sem_ml = len(df_filtered[df_filtered['ml_type'] == 'SEM_ML'])

        com_ml_pct = (com_ml / after_count) * 100
        sem_ml_pct = (sem_ml / after_count) * 100

        logger.info(f"   ✅ COM ML: {com_ml} leads ({com_ml_pct:.1f}%)")
        logger.info(f"   ✅ SEM ML: {sem_ml} leads ({sem_ml_pct:.1f}%)")
    else:
        logger.warning("   ⚠️ Nenhuma campanha de captação encontrada!")

    return df_filtered


def get_classification_stats(df: pd.DataFrame) -> dict:
    """
    Retorna estatísticas sobre a classificação de campanhas.

    Args:
        df: DataFrame com coluna 'ml_type'

    Returns:
        Dicionário com estatísticas

    Examples:
        >>> df = pd.DataFrame({'ml_type': ['COM_ML', 'COM_ML', 'SEM_ML']})
        >>> stats = get_classification_stats(df)
        >>> stats['total']
        3
        >>> stats['com_ml_count']
        2
        >>> stats['com_ml_percentage']
        66.67
    """
    if 'ml_type' not in df.columns:
        raise ValueError("DataFrame deve conter coluna 'ml_type'")

    total = len(df)
    com_ml = len(df[df['ml_type'] == 'COM_ML'])
    sem_ml = len(df[df['ml_type'] == 'SEM_ML'])

    return {
        'total': total,
        'com_ml_count': com_ml,
        'sem_ml_count': sem_ml,
        'com_ml_percentage': round((com_ml / total * 100) if total > 0 else 0, 2),
        'sem_ml_percentage': round((sem_ml / total * 100) if total > 0 else 0, 2),
    }


def list_unique_campaigns(df: pd.DataFrame, ml_type: Optional[str] = None) -> list:
    """
    Lista campanhas únicas, opcionalmente filtradas por tipo.

    Args:
        df: DataFrame com colunas 'campaign' e 'ml_type'
        ml_type: Filtrar por tipo ('COM_ML', 'SEM_ML'), ou None para todos

    Returns:
        Lista de nomes de campanhas únicas

    Examples:
        >>> df = pd.DataFrame({
        ...     'campaign': ['CAMP A', 'CAMP A', 'CAMP B'],
        ...     'ml_type': ['COM_ML', 'COM_ML', 'SEM_ML']
        ... })
        >>> list_unique_campaigns(df, 'COM_ML')
        ['CAMP A']
    """
    if 'campaign' not in df.columns:
        raise ValueError("DataFrame deve conter coluna 'campaign'")

    if ml_type:
        if 'ml_type' not in df.columns:
            raise ValueError("DataFrame deve conter coluna 'ml_type' quando ml_type é especificado")
        df_filtered = df[df['ml_type'] == ml_type]
    else:
        df_filtered = df

    campaigns = df_filtered['campaign'].dropna().unique().tolist()
    return sorted(campaigns)


# =============================================================================
# CLASSIFICAÇÃO CAPI (COM vs SEM EVENTOS CUSTOMIZADOS)
# =============================================================================

def classify_campaign_capi(optimization_goal: str) -> str:
    """
    Classifica campanha baseado no optimization_goal (eventos CAPI).

    Lógica:
    - COM_CAPI: optimization_goal é 'LeadQualified' ou 'LeadQualifiedHighQuality'
    - SEM_CAPI: qualquer outro optimization_goal (LEAD, OFFSITE_CONVERSIONS, etc.)

    Args:
        optimization_goal: String com optimization_goal do adset

    Returns:
        'COM_CAPI' ou 'SEM_CAPI'

    Examples:
        >>> classify_campaign_capi('LeadQualified')
        'COM_CAPI'
        >>> classify_campaign_capi('LeadQualifiedHighQuality')
        'COM_CAPI'
        >>> classify_campaign_capi('LEAD')
        'SEM_CAPI'
        >>> classify_campaign_capi(None)
        'SEM_CAPI'
    """
    if not optimization_goal or pd.isna(optimization_goal):
        return 'SEM_CAPI'

    optimization_goal_str = str(optimization_goal)

    # Eventos CAPI customizados
    if optimization_goal_str in ['LeadQualified', 'LeadQualifiedHighQuality']:
        return 'COM_CAPI'
    else:
        return 'SEM_CAPI'


def add_capi_classification(df: pd.DataFrame, optimization_goal_col: str = 'optimization_goal') -> pd.DataFrame:
    """
    Adiciona coluna 'capi_type' ao DataFrame baseado no optimization_goal.

    IMPORTANTE: Esta classificação é INDEPENDENTE da classificação ML (ml_type).
    Uma campanha pode ser:
    - COM_ML + COM_CAPI: Usa dashboard E eventos CAPI
    - COM_ML + SEM_CAPI: Usa dashboard mas sem eventos CAPI
    - SEM_ML + COM_CAPI: Não usa dashboard mas usa eventos CAPI
    - SEM_ML + SEM_CAPI: Não usa dashboard nem eventos CAPI

    Args:
        df: DataFrame com dados de leads
        optimization_goal_col: Nome da coluna que contém optimization_goal

    Returns:
        DataFrame com nova coluna 'capi_type'

    Examples:
        >>> df = pd.DataFrame({
        ...     'campaign': ['CAMP A', 'CAMP B'],
        ...     'optimization_goal': ['LeadQualified', 'LEAD']
        ... })
        >>> result = add_capi_classification(df)
        >>> list(result['capi_type'])
        ['COM_CAPI', 'SEM_CAPI']
    """
    if optimization_goal_col not in df.columns:
        logger.warning(f"⚠️ Coluna '{optimization_goal_col}' não encontrada no DataFrame")
        logger.warning(f"   Todas as campanhas serão classificadas como SEM_CAPI")
        df['capi_type'] = 'SEM_CAPI'
        return df

    logger.info("🎯 Classificando campanhas por CAPI...")

    # Adicionar coluna capi_type
    df['capi_type'] = df[optimization_goal_col].apply(classify_campaign_capi)

    # Contar por tipo
    type_counts = df['capi_type'].value_counts()
    logger.info(f"   Total por tipo CAPI:")
    for capi_type, count in type_counts.items():
        logger.info(f"     {capi_type}: {count} leads")

    # Calcular percentuais
    total_count = len(df)
    if total_count > 0:
        com_capi = len(df[df['capi_type'] == 'COM_CAPI'])
        sem_capi = len(df[df['capi_type'] == 'SEM_CAPI'])

        com_capi_pct = (com_capi / total_count) * 100
        sem_capi_pct = (sem_capi / total_count) * 100

        logger.info(f"   ✅ COM CAPI: {com_capi} leads ({com_capi_pct:.1f}%)")
        logger.info(f"   ✅ SEM CAPI: {sem_capi} leads ({sem_capi_pct:.1f}%)")
    else:
        logger.warning("   ⚠️ Nenhuma campanha encontrada!")

    return df


def get_capi_classification_stats(df: pd.DataFrame) -> dict:
    """
    Retorna estatísticas sobre a classificação CAPI de campanhas.

    Args:
        df: DataFrame com coluna 'capi_type'

    Returns:
        Dicionário com estatísticas

    Examples:
        >>> df = pd.DataFrame({'capi_type': ['COM_CAPI', 'COM_CAPI', 'SEM_CAPI']})
        >>> stats = get_capi_classification_stats(df)
        >>> stats['total']
        3
        >>> stats['com_capi_count']
        2
        >>> stats['com_capi_percentage']
        66.67
    """
    if 'capi_type' not in df.columns:
        raise ValueError("DataFrame deve conter coluna 'capi_type'")

    total = len(df)
    com_capi = len(df[df['capi_type'] == 'COM_CAPI'])
    sem_capi = len(df[df['capi_type'] == 'SEM_CAPI'])

    return {
        'total': total,
        'com_capi_count': com_capi,
        'sem_capi_count': sem_capi,
        'com_capi_percentage': round((com_capi / total * 100) if total > 0 else 0, 2),
        'sem_capi_percentage': round((sem_capi / total * 100) if total > 0 else 0, 2),
    }
