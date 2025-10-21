"""
Módulo de Cálculo de Métricas Econômicas
Calcula CPL, ROAS projetado, margem de manobra e ações recomendadas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_cpl(spend: float, leads: int) -> float:
    """
    Calcula Custo por Lead (CPL)

    Args:
        spend: Gasto total em R$
        leads: Número de leads

    Returns:
        CPL em R$ (0.0 se leads = 0)
    """
    if leads == 0:
        return 0.0
    return spend / leads


def calculate_projected_conversion_rate(
    decile_distribution: Dict[str, float],
    conversion_rates: Dict[str, float]
) -> float:
    """
    Calcula taxa de conversão projetada baseada na distribuição de decis

    Args:
        decile_distribution: {'D1': 0.10, 'D2': 0.15, ...} (percentuais que somam 1.0)
        conversion_rates: {'D1': 0.0026, 'D2': 0.0026, ...} (taxas históricas)

    Returns:
        Taxa de conversão projetada (0.0 a 1.0)
    """
    projected_rate = 0.0

    for decile, percentage in decile_distribution.items():
        rate = conversion_rates.get(decile, 0.0)
        projected_rate += percentage * rate

    return projected_rate


def calculate_projected_roas(
    product_value: float,
    projected_conversion_rate: float,
    cpl: float
) -> float:
    """
    Calcula ROAS projetado

    Formula: (Valor Produto × Taxa Projetada) / CPL

    Args:
        product_value: Valor do produto em R$
        projected_conversion_rate: Taxa de conversão projetada (0.0 a 1.0)
        cpl: Custo por Lead em R$

    Returns:
        ROAS projetado (0.0 se CPL = 0)
    """
    if cpl == 0:
        return 0.0

    expected_revenue = product_value * projected_conversion_rate
    return expected_revenue / cpl


def calculate_max_acceptable_cpl(
    product_value: float,
    projected_conversion_rate: float,
    min_roas: float
) -> float:
    """
    Calcula CPL máximo aceitável para manter ROAS mínimo

    Formula: (Valor Produto × Taxa Projetada) / ROAS Mínimo

    Args:
        product_value: Valor do produto em R$
        projected_conversion_rate: Taxa de conversão projetada
        min_roas: ROAS mínimo desejado (ex: 2.0x)

    Returns:
        CPL máximo em R$
    """
    if min_roas == 0:
        return 0.0

    expected_revenue = product_value * projected_conversion_rate
    return expected_revenue / min_roas


def calculate_margin(cpl_actual: float, cpl_max: float) -> float:
    """
    Calcula margem de manobra percentual

    Formula: ((CPL Max - CPL Atual) / CPL Max) × 100%

    Args:
        cpl_actual: CPL atual em R$
        cpl_max: CPL máximo aceitável em R$

    Returns:
        Margem em % (-100 a +100)
    """
    if cpl_max == 0:
        return -100.0

    margin = ((cpl_max - cpl_actual) / cpl_max) * 100
    return margin


def calculate_confidence_level(leads: int, period_days: int = 1) -> str:
    """
    Calcula nível de confiança estatística baseado no volume de leads e período

    Baseado em requisitos da Learning Phase do Meta Ads (2024):
    - Meta: 10 conversões em 3 dias (~3.3/dia)
    - Google Ads: 30 conversões em 30 dias (1/dia)

    Thresholds escalados por período:
    - Base: 3 leads/dia (insuficiente), 10/dia (baixa), 20/dia (média)

    Args:
        leads: Número de leads no período
        period_days: Número de dias do período (1, 3, 7, 30, etc.)

    Returns:
        Nível de confiança: 'insuficiente', 'baixa', 'media', 'alta'
    """
    # Thresholds ajustados por período (leads/dia × período)
    threshold_insuficiente = 3 * period_days     # 3/dia
    threshold_baixa = 10 * period_days           # 10/dia
    threshold_media = 20 * period_days           # 20/dia

    if leads < threshold_insuficiente:
        return 'insuficiente'
    elif leads < threshold_baixa:
        return 'baixa'
    elif leads < threshold_media:
        return 'media'
    else:
        return 'alta'


def calculate_budget_variation(
    margin: float,
    confidence: str,
    cpl_actual: float,
    cpl_max: float
) -> float:
    """
    Calcula percentual de variação de budget recomendado

    Lógica simples e consistente:
    - Margem > 0: variation = margem × fator_confiança
    - Margem < 0: variation = redução necessária × ajuste por confiança

    Args:
        margin: Margem de manobra em %
        confidence: Nível de confiança ('baixa', 'media', 'alta')
        cpl_actual: CPL atual em R$
        cpl_max: CPL máximo aceitável em R$

    Returns:
        Percentual de variação (-100 a +100)
        Positivo = aumentar, Negativo = reduzir
    """
    # Fatores de confiança (quanto usar da margem disponível)
    confidence_factors = {
        'baixa': 0.3,   # 30% da margem
        'media': 0.5,   # 50% da margem
        'alta': 0.8     # 80% da margem
    }

    factor = confidence_factors.get(confidence, 0.5)

    if margin > 0:
        # AUMENTAR: usar margem disponível ajustado por confiança
        # Sempre consistente: margem × fator
        variation = (margin / 100.0) * factor
        return round(variation * 100, 0)  # Retornar em %

    else:
        # REDUZIR (margin <= 0): calcular redução necessária para atingir CPL Max
        if cpl_actual == 0:
            return -100.0
        variation = (cpl_max / cpl_actual) - 1.0
        # Aplicar fator de confiança para redução (ser mais agressivo com alta confiança)
        variation_adjusted = variation * (1.0 + (factor - 0.5))  # Mais agressivo com alta confiança
        return round(variation_adjusted * 100, 0)  # Será negativo


def determine_action(
    margin: float,
    dimension: str,
    has_budget_control: bool = True,
    leads: int = 0,
    cpl_actual: float = 0.0,
    cpl_max: float = 0.0,
    period_days: int = 1,
    spend: float = 0.0
) -> str:
    """
    Determina ação recomendada baseada na margem, tipo de dimensão e volume de leads

    Args:
        margin: Margem de manobra em %
        dimension: Tipo de dimensão (campaign, adset, ad, medium, term, content)
        has_budget_control: Se False, indica que não pode controlar orçamento neste nível
                           - Para campaign: retorna "ABO" (budget no adset)
                           - Para medium (adset): retorna "CBO" (budget na campanha)
        leads: Número de leads (para validação estatística)
        cpl_actual: CPL atual (para cálculo de variação)
        cpl_max: CPL máximo aceitável (para cálculo de variação)
        period_days: Número de dias do período analisado
        spend: Gasto total em R$ (para detectar alto gasto sem leads)

    Returns:
        Ação recomendada como string (ex: "Aumentar 25%", "Reduzir 30%", "Manter")
    """
    # REGRA ESPECIAL: Gasto alto sem NENHUM lead = Remover/Pausar imediatamente
    # Threshold fixo: R$ 50
    SPEND_THRESHOLD = 50.0

    if leads == 0 and spend > SPEND_THRESHOLD:
        # Dimensões sem controle de orçamento (apenas on/off)
        onoff_dimensions = ['ad', 'content']
        if dimension in onoff_dimensions:
            return "Remover"
        else:
            # Dimensões com budget: pausar completamente
            return "Reduzir 100%"

    # Verificar confiança estatística
    confidence = calculate_confidence_level(leads, period_days)

    # Se dados insuficientes, mostrar quantos faltam
    if confidence == 'insuficiente':
        threshold_min = 3 * period_days
        faltam = threshold_min - leads
        return f"Aguardar dados (falta {'1 lead' if faltam == 1 else f'{faltam} leads'})"

    # Se não tem controle de orçamento nessa dimensão
    if not has_budget_control:
        # Campaign sem budget → orçamento está no AdSet (ABO)
        if dimension == 'campaign':
            return "ABO"
        # AdSet sem budget → orçamento está na Campaign (CBO)
        elif dimension == 'medium':
            return "CBO"
        else:
            return "N/A"

    # Dimensões com controle de orçamento
    budget_dimensions = ['campaign', 'adset', 'medium', 'term']

    # Dimensões sem controle de orçamento (apenas on/off)
    onoff_dimensions = ['ad', 'content']

    if dimension in budget_dimensions:
        # Calcular variação percentual
        variation_pct = calculate_budget_variation(margin, confidence, cpl_actual, cpl_max)

        if variation_pct > 0:
            return f"Aumentar {variation_pct:.1f}%"
        elif variation_pct < 0:
            return f"Reduzir {abs(variation_pct):.1f}%"
        else:
            return "Manter"

    elif dimension in onoff_dimensions:
        if margin >= 0:
            return "Manter"
        else:
            return "Remover"

    else:
        # Dimensão desconhecida, usar lógica padrão de orçamento
        variation_pct = calculate_budget_variation(margin, confidence, cpl_actual, cpl_max)

        if variation_pct > 0:
            return f"Aumentar {variation_pct:.1f}%"
        elif variation_pct < 0:
            return f"Reduzir {abs(variation_pct):.1f}%"
        else:
            return "Manter"


def calculate_decile_distribution(df: pd.DataFrame, decile_col: str = 'decil') -> Dict[str, float]:
    """
    Calcula distribuição percentual de leads por decil

    Args:
        df: DataFrame com predições
        decile_col: Nome da coluna de decil

    Returns:
        Dict com percentual por decil {'D1': 0.10, 'D2': 0.15, ...}
    """
    if len(df) == 0:
        return {}

    decile_counts = df[decile_col].value_counts()
    decile_distribution = (decile_counts / len(df)).to_dict()

    return decile_distribution


def enrich_utm_with_economic_metrics(
    utm_df: pd.DataFrame,
    product_value: float,
    min_roas: float,
    conversion_rates: Dict[str, float],
    dimension: str,
    budget_control_col: str = None,
    period_days: int = 1
) -> pd.DataFrame:
    """
    Enriquece DataFrame de análise UTM com métricas econômicas

    Args:
        utm_df: DataFrame com colunas: value, leads, spend, decile distribution
        product_value: Valor do produto em R$
        min_roas: ROAS mínimo desejado
        conversion_rates: Dict com taxas de conversão por decil
        dimension: Tipo de dimensão (campaign, adset, ad, etc.)
        budget_control_col: Nome da coluna que indica se tem controle de orçamento (opcional)
                           Para campaigns: 'has_campaign_budget'
        period_days: Número de dias do período analisado (para ajustar thresholds)

    Returns:
        DataFrame enriquecido com colunas adicionais:
        - cpl: Custo por Lead
        - taxa_proj: Taxa de conversão projetada
        - roas_proj: ROAS projetado
        - cpl_max: CPL máximo aceitável
        - margem: Margem de manobra em %
        - acao: Ação recomendada (ou "N/A" se não tem controle de orçamento)
    """
    df = utm_df.copy()

    # Inicializar colunas
    df['cpl'] = 0.0
    df['taxa_proj'] = 0.0
    df['roas_proj'] = 0.0
    df['cpl_max'] = 0.0
    df['margem'] = 0.0
    df['acao'] = ""
    df['budget_current'] = 0.0
    df['budget_target'] = 0.0

    for idx, row in df.iterrows():
        spend = row.get('spend', 0.0)
        leads = row.get('leads', 0)

        # CPL
        cpl = calculate_cpl(spend, leads)
        df.at[idx, 'cpl'] = cpl

        # Distribuição de decis (assumindo colunas %D1, %D2, ..., %D10)
        decile_dist = {}
        for i in range(1, 11):
            decile_key = f'D{i}'
            pct_col = f'%{decile_key}'
            if pct_col in row:
                # Converter de % para decimal (ex: 15.5% -> 0.155)
                decile_dist[decile_key] = row[pct_col] / 100.0

        # Taxa de conversão projetada
        if decile_dist:
            taxa_proj = calculate_projected_conversion_rate(decile_dist, conversion_rates)
        else:
            taxa_proj = 0.0
        df.at[idx, 'taxa_proj'] = taxa_proj

        # ROAS projetado
        roas_proj = calculate_projected_roas(product_value, taxa_proj, cpl)
        df.at[idx, 'roas_proj'] = roas_proj

        # CPL máximo
        cpl_max = calculate_max_acceptable_cpl(product_value, taxa_proj, min_roas)
        df.at[idx, 'cpl_max'] = cpl_max

        # Margem
        margin = calculate_margin(cpl, cpl_max)
        df.at[idx, 'margem'] = margin

        # Verificar se tem controle de orçamento
        has_budget_control = True
        if budget_control_col and budget_control_col in row:
            has_budget_control = bool(row[budget_control_col])

        # Ação (inclui validação estatística, cálculo de variação e detecção de gasto alto sem leads)
        action = determine_action(margin, dimension, has_budget_control, leads, cpl, cpl_max, period_days, spend)
        df.at[idx, 'acao'] = action

        # Orçamento atual (gasto do período)
        budget_current = spend
        df.at[idx, 'budget_current'] = budget_current

        # Orçamento alvo (baseado na ação)
        budget_target = spend  # Default: manter

        # Extrair variação percentual da ação
        import re
        if 'Aumentar' in action or 'Reduzir' in action:
            match = re.search(r'(\d+\.?\d*)%', action)
            if match:
                variation_pct = float(match.group(1))
                if 'Reduzir' in action:
                    variation_pct = -variation_pct
                budget_target = spend * (1 + variation_pct / 100.0)
        elif action == 'Remover' or 'Reduzir 100' in action:
            budget_target = 0.0

        df.at[idx, 'budget_target'] = budget_target

    logger.info(f"✅ Métricas econômicas calculadas para dimensão '{dimension}': {len(df)} registros")

    return df


def format_economic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formata colunas de métricas econômicas para exibição

    Args:
        df: DataFrame com métricas calculadas

    Returns:
        DataFrame com valores formatados
    """
    df = df.copy()

    # Formatar valores monetários (R$)
    money_cols = ['spend', 'cpl', 'cpl_max']
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "R$ 0,00")

    # Formatar percentuais
    pct_cols = ['taxa_proj', 'margem']
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "0.00%")

    # Formatar ROAS
    if 'roas_proj' in df.columns:
        df['roas_proj'] = df['roas_proj'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "0.00x")

    return df


def calculate_tier(roas_proj: float, min_roas: float) -> str:
    """
    Determina tier baseado no ROAS projetado

    Args:
        roas_proj: ROAS projetado
        min_roas: ROAS mínimo

    Returns:
        Tier (A+, A, B, C, D)
    """
    if roas_proj >= min_roas * 2:
        return "A+"
    elif roas_proj >= min_roas * 1.5:
        return "A"
    elif roas_proj >= min_roas:
        return "B"
    elif roas_proj >= min_roas * 0.5:
        return "C"
    else:
        return "D"
