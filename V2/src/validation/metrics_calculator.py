"""
Módulo para cálculo de métricas de performance de campanhas e decis.

Calcula todas as métricas necessárias para validação:
- Por campanha: leads, conversões, CPL, ROAS, margem
- Por decil: performance real vs esperada (Guru vs Guru+TMB)
- Integração com Meta API para buscar custos
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

# Importar integrações existentes
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from api.meta_integration import MetaAdsIntegration
from api.economic_metrics import calculate_cpl, calculate_contribution_margin
from api.business_config import CONVERSION_RATES, PRODUCT_VALUE

logger = logging.getLogger(__name__)


class CampaignMetricsCalculator:
    """
    Calcula métricas de performance por campanha.

    Busca custos via Meta API e calcula:
    - CPL (Custo por Lead)
    - Taxa de conversão
    - ROAS (Return on Ad Spend)
    - Margem de Contribuição
    """

    def __init__(self, meta_api_integration: MetaAdsIntegration, product_value: float):
        """
        Args:
            meta_api_integration: Cliente da Meta Ads API
            product_value: Valor do produto em R$
        """
        self.meta_api = meta_api_integration
        self.product_value = product_value

    def calculate_campaign_metrics(
        self,
        matched_df: pd.DataFrame,
        account_id: str,
        period_start: str,
        period_end: str
    ) -> pd.DataFrame:
        """
        Calcula métricas completas por campanha.

        Args:
            matched_df: DataFrame com matching realizado (leads + vendas)
            account_id: ID da conta Meta (act_XXXXXXXXX)
            period_start: Data início (YYYY-MM-DD)
            period_end: Data fim (YYYY-MM-DD)

        Returns:
            DataFrame com métricas por campanha:
            - ml_type: COM_ML ou SEM_ML
            - campaign: Nome da campanha
            - leads: Total de leads
            - conversions: Total de conversões
            - conversion_rate: Taxa de conversão (%)
            - total_revenue: Receita total
            - spend: Gasto total (Meta API)
            - cpl: Custo por lead
            - roas: Return on Ad Spend
            - contribution_margin: Margem de contribuição (R$)
            - margin_percent: Margem (%)
        """
        logger.info("📊 Calculando métricas por campanha...")

        # 1. Agregar dados de conversão por campanha
        logger.info("   Agregando dados de conversão...")
        campaign_stats = matched_df.groupby(['ml_type', 'campaign']).agg({
            'email': 'count',  # total leads
            'converted': 'sum',  # conversions
            'sale_value': 'sum'  # revenue total
        }).reset_index()

        campaign_stats.columns = ['ml_type', 'campaign', 'leads', 'conversions', 'total_revenue']

        # Calcular taxa de conversão
        campaign_stats['conversion_rate'] = (
            campaign_stats['conversions'] / campaign_stats['leads'] * 100
        ).round(2)

        # Se sale_value não estava disponível, calcular receita baseado em product_value
        if campaign_stats['total_revenue'].sum() == 0:
            campaign_stats['total_revenue'] = campaign_stats['conversions'] * self.product_value

        logger.info(f"   {len(campaign_stats)} campanhas agregadas")

        # 2. Buscar custos via Meta API
        logger.info("   Buscando custos via Meta API...")
        try:
            costs_hierarchy = self.meta_api.get_costs_hierarchy(
                account_id=account_id,
                since_date=period_start,
                until_date=period_end
            )

            # Mapear custos para campanhas
            campaign_stats['spend'] = campaign_stats['campaign'].apply(
                lambda camp: self._get_campaign_spend(camp, costs_hierarchy)
            )

            total_spend = campaign_stats['spend'].sum()
            logger.info(f"   ✅ Custos obtidos: R$ {total_spend:,.2f}")

        except Exception as e:
            logger.error(f"   ❌ Erro ao buscar custos Meta API: {e}")
            logger.warning("   Usando spend = 0 para todas as campanhas")
            campaign_stats['spend'] = 0.0

        # 3. Calcular métricas finais
        logger.info("   Calculando CPL, ROAS e Margem...")

        # CPL
        campaign_stats['cpl'] = campaign_stats.apply(
            lambda row: calculate_cpl(row['spend'], row['leads']) if row['leads'] > 0 else 0,
            axis=1
        ).round(2)

        # ROAS
        campaign_stats['roas'] = campaign_stats.apply(
            lambda row: (row['total_revenue'] / row['spend']) if row['spend'] > 0 else 0,
            axis=1
        ).round(2)

        # Margem de Contribuição
        campaign_stats['contribution_margin'] = (
            campaign_stats['total_revenue'] - campaign_stats['spend']
        ).round(2)

        # Margem %
        campaign_stats['margin_percent'] = campaign_stats.apply(
            lambda row: (row['contribution_margin'] / row['spend'] * 100) if row['spend'] > 0 else 0,
            axis=1
        ).round(2)

        # Ordenar por margem de contribuição (maior para menor)
        campaign_stats = campaign_stats.sort_values('contribution_margin', ascending=False)

        logger.info(f"   ✅ Métricas calculadas para {len(campaign_stats)} campanhas")

        return campaign_stats

    def _get_campaign_spend(self, campaign_name: str, costs_hierarchy: Dict) -> float:
        """
        Busca o gasto de uma campanha específica na hierarquia de custos.

        Args:
            campaign_name: Nome da campanha
            costs_hierarchy: Dicionário retornado por get_costs_hierarchy()

        Returns:
            Valor gasto (float)
        """
        if not costs_hierarchy:
            return 0.0

        # Procurar por nome exato
        for camp_id, camp_data in costs_hierarchy.items():
            if camp_data.get('name', '').strip() == campaign_name.strip():
                return float(camp_data.get('spend', 0))

        # Se não encontrou exato, tentar match parcial (normalizado)
        campaign_lower = campaign_name.lower().strip()
        for camp_id, camp_data in costs_hierarchy.items():
            camp_name_lower = camp_data.get('name', '').lower().strip()
            if campaign_lower in camp_name_lower or camp_name_lower in campaign_lower:
                return float(camp_data.get('spend', 0))

        # Não encontrou
        logger.debug(f"   Campanha '{campaign_name}' não encontrada nos custos Meta")
        return 0.0


class DecileMetricsCalculator:
    """
    Calcula métricas de performance por decil (D1-D10).

    IMPORTANTE: Modelo foi treinado APENAS com vendas Guru.
    Por isso calculamos métricas separadas:
    - Guru: Performance nos dados de treinamento
    - Guru+TMB: Performance em todos os dados (generalização)
    """

    def __init__(self):
        """Inicializa calculadora de métricas de decil."""
        # Carregar taxas de conversão esperadas do modelo
        self.expected_rates = CONVERSION_RATES

    def calculate_decile_performance(
        self,
        matched_df: pd.DataFrame,
        product_value: float
    ) -> pd.DataFrame:
        """
        Calcula métricas reais por decil separando Guru vs Guru+TMB.

        Args:
            matched_df: DataFrame com matching realizado
            product_value: Valor do produto

        Returns:
            DataFrame com métricas por decil:
            - decile: D1-D10
            - leads: Total de leads
            - conversions_guru: Conversões Guru
            - conversions_total: Conversões Total (Guru+TMB)
            - conversion_rate_guru: Taxa conversão Guru (%)
            - conversion_rate_total: Taxa conversão Total (%)
            - expected_conversion_rate: Taxa esperada do modelo (%)
            - performance_ratio_guru: Guru / Esperado
            - performance_ratio_total: Total / Esperado
            - revenue_guru: Receita Guru
            - revenue_total: Receita Total
        """
        logger.info("📈 Calculando performance por decil...")

        # Filtrar apenas leads com decil definido
        df_with_decile = matched_df[matched_df['decile'].notna()].copy()

        if len(df_with_decile) == 0:
            logger.warning("⚠️ Nenhum lead com decil definido")
            return pd.DataFrame()

        logger.info(f"   {len(df_with_decile)} leads com decil definido")

        decile_metrics = []

        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_df = df_with_decile[df_with_decile['decile'] == decile]

            # Total de leads
            leads = len(decile_df)

            if leads == 0:
                # Pular decil sem leads
                continue

            # Conversões separadas por origem
            conversions_guru = len(decile_df[
                (decile_df['converted'] == True) &
                (decile_df['sale_origin'] == 'guru')
            ])

            conversions_total = len(decile_df[decile_df['converted'] == True])

            # Taxas de conversão
            conversion_rate_guru = (conversions_guru / leads * 100) if leads > 0 else 0
            conversion_rate_total = (conversions_total / leads * 100) if leads > 0 else 0

            # Taxa esperada do modelo (em %)
            expected_rate = self.expected_rates.get(decile, 0) * 100

            # Performance ratios (real / esperado)
            performance_ratio_guru = (
                (conversion_rate_guru / expected_rate) if expected_rate > 0 else 0
            )
            performance_ratio_total = (
                (conversion_rate_total / expected_rate) if expected_rate > 0 else 0
            )

            # Receitas
            revenue_guru = conversions_guru * product_value
            revenue_total = conversions_total * product_value

            decile_metrics.append({
                'decile': decile,
                'leads': leads,
                'conversions_guru': conversions_guru,
                'conversions_total': conversions_total,
                'conversion_rate_guru': round(conversion_rate_guru, 2),
                'conversion_rate_total': round(conversion_rate_total, 2),
                'expected_conversion_rate': round(expected_rate, 2),
                'performance_ratio_guru': round(performance_ratio_guru, 2),
                'performance_ratio_total': round(performance_ratio_total, 2),
                'revenue_guru': round(revenue_guru, 2),
                'revenue_total': round(revenue_total, 2),
            })

        df_metrics = pd.DataFrame(decile_metrics)

        if len(df_metrics) > 0:
            logger.info(f"   ✅ Métricas calculadas para {len(df_metrics)} decis")

            # Log summary
            total_guru = df_metrics['revenue_guru'].sum()
            total_all = df_metrics['revenue_total'].sum()
            logger.info(f"      Receita Guru: R$ {total_guru:,.2f}")
            logger.info(f"      Receita Total: R$ {total_all:,.2f}")
        else:
            logger.warning("   ⚠️ Nenhum decil com leads suficientes")

        return df_metrics


def compare_ml_vs_non_ml(campaign_metrics: pd.DataFrame) -> Dict:
    """
    Compara agregado de campanhas COM_ML vs SEM_ML.

    Args:
        campaign_metrics: DataFrame retornado por CampaignMetricsCalculator

    Returns:
        Dicionário com comparação:
        {
            'com_ml': {leads, conversions, conversion_rate, spend, cpl, roas, margin},
            'sem_ml': {leads, conversions, conversion_rate, spend, cpl, roas, margin},
            'difference': {conversion_rate_diff, roas_diff, margin_diff}
        }
    """
    logger.info("⚖️ Comparando COM_ML vs SEM_ML...")

    # Separar por tipo
    com_ml = campaign_metrics[campaign_metrics['ml_type'] == 'COM_ML']
    sem_ml = campaign_metrics[campaign_metrics['ml_type'] == 'SEM_ML']

    def aggregate_metrics(df: pd.DataFrame) -> Dict:
        """Agrega métricas de múltiplas campanhas."""
        if len(df) == 0:
            return {
                'leads': 0,
                'conversions': 0,
                'conversion_rate': 0,
                'spend': 0,
                'cpl': 0,
                'roas': 0,
                'margin': 0,
            }

        total_leads = df['leads'].sum()
        total_conversions = df['conversions'].sum()
        total_revenue = df['total_revenue'].sum()
        total_spend = df['spend'].sum()

        conversion_rate = (total_conversions / total_leads * 100) if total_leads > 0 else 0
        cpl = (total_spend / total_leads) if total_leads > 0 else 0
        roas = (total_revenue / total_spend) if total_spend > 0 else 0
        margin = total_revenue - total_spend

        return {
            'leads': int(total_leads),
            'conversions': int(total_conversions),
            'conversion_rate': round(conversion_rate, 2),
            'revenue': round(total_revenue, 2),
            'spend': round(total_spend, 2),
            'cpl': round(cpl, 2),
            'roas': round(roas, 2),
            'margin': round(margin, 2),
        }

    com_ml_agg = aggregate_metrics(com_ml)
    sem_ml_agg = aggregate_metrics(sem_ml)

    # Calcular diferenças percentuais
    def calc_diff(com, sem, key):
        """Calcula diferença percentual."""
        if sem == 0:
            return 0
        return round(((com - sem) / sem * 100), 2)

    difference = {
        'conversion_rate_diff': calc_diff(
            com_ml_agg['conversion_rate'],
            sem_ml_agg['conversion_rate'],
            'conversion_rate'
        ),
        'roas_diff': calc_diff(com_ml_agg['roas'], sem_ml_agg['roas'], 'roas'),
        'margin_diff': calc_diff(com_ml_agg['margin'], sem_ml_agg['margin'], 'margin'),
    }

    logger.info(f"   COM_ML: {com_ml_agg['leads']} leads, {com_ml_agg['conversions']} conversões, ROAS {com_ml_agg['roas']:.2f}x")
    logger.info(f"   SEM_ML: {sem_ml_agg['leads']} leads, {sem_ml_agg['conversions']} conversões, ROAS {sem_ml_agg['roas']:.2f}x")

    if com_ml_agg['roas'] > sem_ml_agg['roas']:
        improvement = difference['roas_diff']
        logger.info(f"   🏆 VENCEDOR: COM_ML (ROAS {improvement:.1f}% maior)")
    elif sem_ml_agg['roas'] > com_ml_agg['roas']:
        decline = abs(difference['roas_diff'])
        logger.warning(f"   ⚠️ SEM_ML performou {decline:.1f}% melhor")
    else:
        logger.info(f"   ➖ Empate técnico")

    return {
        'com_ml': com_ml_agg,
        'sem_ml': sem_ml_agg,
        'difference': difference
    }


def calculate_overall_stats(matched_df: pd.DataFrame, campaign_metrics: pd.DataFrame) -> Dict:
    """
    Calcula estatísticas gerais do sistema.

    Args:
        matched_df: DataFrame com matching
        campaign_metrics: DataFrame com métricas de campanhas

    Returns:
        Dicionário com estatísticas gerais
    """
    total_leads = len(matched_df)
    total_conversions = len(matched_df[matched_df['converted'] == True])
    total_revenue = matched_df[matched_df['converted'] == True]['sale_value'].sum()
    total_spend = campaign_metrics['spend'].sum()

    conversion_rate = (total_conversions / total_leads * 100) if total_leads > 0 else 0
    roas = (total_revenue / total_spend) if total_spend > 0 else 0
    margin = total_revenue - total_spend

    # Por origem
    conversions_guru = len(matched_df[
        (matched_df['converted'] == True) &
        (matched_df['sale_origin'] == 'guru')
    ])
    conversions_tmb = len(matched_df[
        (matched_df['converted'] == True) &
        (matched_df['sale_origin'] == 'tmb')
    ])

    return {
        'total_leads': total_leads,
        'total_conversions': total_conversions,
        'conversion_rate': round(conversion_rate, 2),
        'total_revenue': round(total_revenue, 2),
        'total_spend': round(total_spend, 2),
        'roas': round(roas, 2),
        'margin': round(margin, 2),
        'conversions_guru': conversions_guru,
        'conversions_tmb': conversions_tmb,
    }
