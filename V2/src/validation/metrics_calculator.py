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
        period_end: str,
        global_tracking_rate: float = None,
        costs_hierarchy_consolidated: Dict = None
    ) -> pd.DataFrame:
        """
        Calcula métricas completas por campanha.

        NOVO: Suporta custos pré-carregados de múltiplas contas Meta.

        Args:
            matched_df: DataFrame com matching realizado (leads + vendas)
            account_id: ID da conta Meta (act_XXXXXXXXX) ou lista de IDs separados por vírgula
            period_start: Data início (YYYY-MM-DD)
            period_end: Data fim (YYYY-MM-DD)
            global_tracking_rate: Taxa de trackeamento global (%) - opcional
            costs_hierarchy_consolidated: Dicionário com custos pré-carregados de múltiplas contas (opcional)

        Returns:
            DataFrame com métricas por campanha:
            - ml_type: COM_ML ou SEM_ML
            - campaign: Nome da campanha
            - leads: Total de leads
            - conversions: Total de conversões
            - conversion_rate: Taxa de conversão (%)
            - total_revenue: Receita total
            - spend: Gasto total (Meta API)
            - budget: Orçamento total (CBO ou soma ABO)
            - num_creatives: Número de criativos
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

        # 2. Buscar custos via Meta API (se não fornecidos)
        if costs_hierarchy_consolidated:
            logger.info("   Usando custos pré-carregados de múltiplas contas")
            costs_hierarchy = costs_hierarchy_consolidated
        else:
            logger.info("   Buscando custos via Meta API...")
            try:
                costs_hierarchy = self.meta_api.get_costs_hierarchy(
                    account_id=account_id,
                    since_date=period_start,
                    until_date=period_end
                )
            except Exception as e:
                logger.error(f"   ❌ Erro ao buscar custos Meta API: {e}")
                logger.warning("   Usando spend = 0 para todas as campanhas")
                costs_hierarchy = {'campaigns': {}}

        # Mapear custos para campanhas (sempre, independente da fonte)
        if costs_hierarchy and costs_hierarchy.get('campaigns'):
            campaign_stats['spend'] = campaign_stats['campaign'].apply(
                lambda camp: self._get_campaign_spend(camp, costs_hierarchy)
            )

            # Adicionar budget e número de criativos
            campaign_stats['budget'] = campaign_stats['campaign'].apply(
                lambda camp: self._get_campaign_budget(camp, costs_hierarchy)
            )
            campaign_stats['num_creatives'] = campaign_stats['campaign'].apply(
                lambda camp: self._get_campaign_num_creatives(camp, costs_hierarchy)
            )

            total_spend = campaign_stats['spend'].sum()
            logger.info(f"   ✅ Custos obtidos: R$ {total_spend:,.2f}")
        else:
            campaign_stats['spend'] = 0.0
            campaign_stats['budget'] = 0.0
            campaign_stats['num_creatives'] = 0

        # 2.5. Filtrar campanhas sem spend (não ativas no período)
        campaigns_before_filter = len(campaign_stats)
        campaign_stats = campaign_stats[campaign_stats['spend'] > 0]
        campaigns_filtered = campaigns_before_filter - len(campaign_stats)

        if campaigns_filtered > 0:
            logger.info(f"   ⚠️ {campaigns_filtered} campanhas removidas (spend = 0, não ativas no período)")

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

    def _extract_campaign_id(self, campaign_name: str) -> str:
        """
        Extrai o Campaign ID do final do nome da campanha.

        Formato esperado: "NOME DA CAMPANHA|CAMPAIGN_ID"
        Exemplo: "DEVLF | CAP | FRIO | FASE 01 | ABERTO ADV+ | PG2 | 2025-04-15|120220370119870390"

        Args:
            campaign_name: Nome completo da campanha com ID

        Returns:
            Campaign ID (string) ou None se não encontrar
        """
        if not campaign_name or pd.isna(campaign_name):
            return None

        # Dividir por "|" e pegar o último elemento
        parts = str(campaign_name).split('|')

        if len(parts) < 2:
            return None

        # O último elemento deve ser apenas números (Campaign ID)
        last_part = parts[-1].strip()

        if last_part.isdigit() and len(last_part) > 10:  # IDs do Meta têm ~18 dígitos
            return last_part

        return None

    def _get_campaign_spend(self, campaign_name: str, costs_hierarchy: Dict) -> float:
        """
        Busca o gasto de uma campanha específica na hierarquia de custos.

        NOVO: Usa Campaign ID extraído do nome para matching preciso.

        Args:
            campaign_name: Nome da campanha (pode incluir |ID no final)
            costs_hierarchy: Dicionário retornado por get_costs_hierarchy()

        Returns:
            Valor gasto (float)
        """
        if not costs_hierarchy:
            return 0.0

        campaigns = costs_hierarchy.get('campaigns', {})
        if not campaigns:
            return 0.0

        # MÉTODO 1: Tentar match por Campaign ID (mais preciso)
        campaign_id = self._extract_campaign_id(campaign_name)

        if campaign_id and campaign_id in campaigns:
            spend = float(campaigns[campaign_id].get('spend', 0))
            logger.debug(f"   ✅ Match por ID: {campaign_id} → R$ {spend:.2f}")
            return spend

        # MÉTODO 2: Fallback - match por nome (para campanhas sem ID no nome)
        # Remover ID do final para comparação
        campaign_name_clean = campaign_name
        if campaign_id:
            campaign_name_clean = '|'.join(campaign_name.split('|')[:-1]).strip()

        # Procurar por nome exato
        for camp_id, camp_data in campaigns.items():
            if camp_data.get('name', '').strip() == campaign_name_clean.strip():
                spend = float(camp_data.get('spend', 0))
                logger.debug(f"   ✅ Match por nome: {campaign_name_clean} → R$ {spend:.2f}")
                return spend

        # Se não encontrou exato, tentar match parcial (normalizado)
        campaign_lower = campaign_name_clean.lower().strip()
        for camp_id, camp_data in campaigns.items():
            camp_name_lower = camp_data.get('name', '').lower().strip()
            if campaign_lower in camp_name_lower or camp_name_lower in campaign_lower:
                spend = float(camp_data.get('spend', 0))
                logger.debug(f"   ⚠️ Match parcial: {campaign_name_clean} → R$ {spend:.2f}")
                return spend

        # Não encontrou
        logger.debug(f"   ❌ Campanha não encontrada: {campaign_name}")
        return 0.0

    def _get_campaign_budget(self, campaign_name: str, costs_hierarchy: Dict) -> float:
        """
        Busca o orçamento de uma campanha específica na hierarquia de custos.
        Para campanhas ABO (budget=0), soma os budgets dos adsets.

        NOVO: Usa Campaign ID extraído do nome para matching preciso.

        Args:
            campaign_name: Nome da campanha (pode incluir |ID no final)
            costs_hierarchy: Dicionário retornado por get_costs_hierarchy()

        Returns:
            Valor do orçamento (float) - prioriza daily_budget, senão lifetime_budget
        """
        if not costs_hierarchy:
            return 0.0

        campaigns = costs_hierarchy.get('campaigns', {})
        if not campaigns:
            return 0.0

        # MÉTODO 1: Tentar match por Campaign ID (mais preciso)
        campaign_id = self._extract_campaign_id(campaign_name)
        camp_data = None

        if campaign_id and campaign_id in campaigns:
            camp_data = campaigns[campaign_id]
        else:
            # MÉTODO 2: Fallback - match por nome
            campaign_name_clean = campaign_name
            if campaign_id:
                campaign_name_clean = '|'.join(campaign_name.split('|')[:-1]).strip()

            # Procurar por nome exato
            for camp_id, data in campaigns.items():
                if data.get('name', '').strip() == campaign_name_clean.strip():
                    camp_data = data
                    break

        if camp_data:
            # Priorizar daily_budget, senão lifetime_budget
            daily = float(camp_data.get('daily_budget', 0) or 0)
            lifetime = float(camp_data.get('lifetime_budget', 0) or 0)
            campaign_budget = daily if daily > 0 else lifetime

            # Se budget da campanha é 0, pode ser ABO - somar budgets dos adsets
            if campaign_budget == 0 and not camp_data.get('has_campaign_budget', False):
                adsets = camp_data.get('adsets', {})
                if adsets and self.meta_api:
                    # Buscar budget dos adsets via Meta API
                    total_adset_budget = 0.0
                    for adset_id in adsets.keys():
                        try:
                            budget_info = self.meta_api.get_adset_budget_info(adset_id)
                            adset_daily = float(budget_info.get('daily_budget', 0) or 0)
                            adset_lifetime = float(budget_info.get('lifetime_budget', 0) or 0)
                            adset_budget = adset_daily if adset_daily > 0 else adset_lifetime
                            total_adset_budget += adset_budget
                        except Exception as e:
                            logger.debug(f"Erro ao buscar budget do adset {adset_id}: {e}")
                            continue
                    if total_adset_budget > 0:
                        return total_adset_budget

            return campaign_budget

        return 0.0

    def _get_campaign_num_creatives(self, campaign_name: str, costs_hierarchy: Dict) -> int:
        """
        Busca o número de criativos (ads) de uma campanha específica na hierarquia de custos.
        Conta todos os ads em todos os adsets da campanha.

        NOVO: Usa Campaign ID extraído do nome para matching preciso.

        Args:
            campaign_name: Nome da campanha (pode incluir |ID no final)
            costs_hierarchy: Dicionário retornado por get_costs_hierarchy()

        Returns:
            Número de criativos (int)
        """
        if not costs_hierarchy:
            return 0

        campaigns = costs_hierarchy.get('campaigns', {})
        if not campaigns:
            return 0

        # MÉTODO 1: Tentar match por Campaign ID (mais preciso)
        campaign_id = self._extract_campaign_id(campaign_name)
        camp_data = None

        if campaign_id and campaign_id in campaigns:
            camp_data = campaigns[campaign_id]
        else:
            # MÉTODO 2: Fallback - match por nome
            campaign_name_clean = campaign_name
            if campaign_id:
                campaign_name_clean = '|'.join(campaign_name.split('|')[:-1]).strip()

            # Procurar por nome exato
            for camp_id, data in campaigns.items():
                if data.get('name', '').strip() == campaign_name_clean.strip():
                    camp_data = data
                    break

        if camp_data:
            # Contar todos os ads em todos os adsets
            adsets = camp_data.get('adsets', {})
            total_ads = 0
            for adset_id, adset_data in adsets.items():
                ads = adset_data.get('ads', {})
                total_ads += len(ads)
            return total_ads

        return 0


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
            'com_ml': {leads, conversions, conversion_rate, revenue, spend, cpl, roas, margin},
            'sem_ml': {leads, conversions, conversion_rate, revenue, spend, cpl, roas, margin},
            'difference': {leads_diff, conversions_diff, conversion_rate_diff, revenue_diff, spend_diff, cpl_diff, roas_diff, margin_diff}
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
        'leads_diff': calc_diff(com_ml_agg['leads'], sem_ml_agg['leads'], 'leads'),
        'conversions_diff': calc_diff(com_ml_agg['conversions'], sem_ml_agg['conversions'], 'conversions'),
        'conversion_rate_diff': calc_diff(
            com_ml_agg['conversion_rate'],
            sem_ml_agg['conversion_rate'],
            'conversion_rate'
        ),
        'revenue_diff': calc_diff(com_ml_agg['revenue'], sem_ml_agg['revenue'], 'revenue'),
        'spend_diff': calc_diff(com_ml_agg['spend'], sem_ml_agg['spend'], 'spend'),
        'cpl_diff': calc_diff(com_ml_agg['cpl'], sem_ml_agg['cpl'], 'cpl'),
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


def calculate_overall_stats(
    matched_df: pd.DataFrame,
    campaign_metrics: pd.DataFrame,
    lead_period: tuple = None,
    sales_period: tuple = None,
    sales_df: pd.DataFrame = None,
    product_value: float = 2000.0
) -> Dict:
    """
    Calcula estatísticas gerais do sistema.

    IMPORTANTE: Métricas de receita/conversão usam TODAS as vendas do período,
    não apenas as identificadas/matched. Apenas a taxa de tracking usa vendas matched.

    Args:
        matched_df: DataFrame com matching (para calcular tracking rate)
        campaign_metrics: DataFrame com métricas de campanhas
        lead_period: Tupla (start_date, end_date) do período de captação
        sales_period: Tupla (start_date, end_date) do período de vendas
        sales_df: DataFrame com TODAS as vendas do período (não apenas matched)
        product_value: Valor do produto (para calcular receita se sale_value não disponível)

    Returns:
        Dicionário com estatísticas gerais
    """
    total_leads = len(matched_df)
    total_spend = campaign_metrics['spend'].sum()

    # Conversões IDENTIFICADAS (matched)
    matched_conversions = len(matched_df[matched_df['converted'] == True])

    # Se sales_df fornecido, usar TODAS as vendas do período
    # Caso contrário, fallback para vendas matched
    if sales_df is not None and not sales_df.empty:
        # TOTAL de vendas do período (não apenas matched)
        total_conversions = len(sales_df)

        # Conversões por origem (TODAS, não apenas matched)
        # Coluna 'origem' vem do data_loader.py (linhas 291 e 398)
        if 'origem' in sales_df.columns:
            conversions_guru_total = len(sales_df[sales_df['origem'] == 'guru'])
            conversions_tmb_total = len(sales_df[sales_df['origem'] == 'tmb'])
        else:
            logger.warning("⚠️ Coluna 'origem' não encontrada em sales_df")
            conversions_guru_total = 0
            conversions_tmb_total = 0

        # Conversões IDENTIFICADAS por origem (somente matched)
        if 'sale_origin' in matched_df.columns:
            conversions_guru_matched = len(matched_df[
                (matched_df['converted'] == True) &
                (matched_df['sale_origin'] == 'guru')
            ])
            conversions_tmb_matched = len(matched_df[
                (matched_df['converted'] == True) &
                (matched_df['sale_origin'] == 'tmb')
            ])
        else:
            logger.warning("⚠️ Coluna 'sale_origin' não encontrada em matched_df")
            conversions_guru_matched = 0
            conversions_tmb_matched = 0

        # Receita TOTAL do período
        if 'sale_value' in sales_df.columns:
            total_revenue = sales_df['sale_value'].sum()
        else:
            # Se não tiver sale_value, usar product_value * quantidade
            total_revenue = len(sales_df) * product_value

    else:
        # Fallback: usar apenas vendas matched
        logger.warning("⚠️ sales_df não fornecido, usando apenas vendas matched para estatísticas gerais")
        total_conversions = matched_conversions
        total_revenue = matched_df[matched_df['converted'] == True]['sale_value'].sum()

        conversions_guru_total = len(matched_df[
            (matched_df['converted'] == True) &
            (matched_df['sale_origin'] == 'guru')
        ])
        conversions_tmb_total = len(matched_df[
            (matched_df['converted'] == True) &
            (matched_df['sale_origin'] == 'tmb')
        ])
        conversions_guru_matched = conversions_guru_total
        conversions_tmb_matched = conversions_tmb_total

    # Métricas gerais (baseadas em TODAS as vendas)
    conversion_rate = (total_conversions / total_leads * 100) if total_leads > 0 else 0
    roas = (total_revenue / total_spend) if total_spend > 0 else 0
    margin = total_revenue - total_spend

    result = {
        'total_leads': total_leads,
        'total_conversions': total_conversions,  # TODAS as vendas do período
        'matched_conversions': matched_conversions,  # Apenas vendas identificadas
        'conversion_rate': round(conversion_rate, 2),
        'total_revenue': round(total_revenue, 2),
        'total_spend': round(total_spend, 2),
        'roas': round(roas, 2),
        'margin': round(margin, 2),
        'conversions_guru_total': conversions_guru_total,          # Total Guru (todas)
        'conversions_guru_matched': conversions_guru_matched,      # Guru identificadas
        'conversions_tmb_total': conversions_tmb_total,            # Total TMB (todas)
        'conversions_tmb_matched': conversions_tmb_matched,        # TMB identificadas
    }

    # Adicionar períodos se fornecidos
    if lead_period:
        result['lead_period_start'] = lead_period[0]
        result['lead_period_end'] = lead_period[1]
    if sales_period:
        result['sales_period_start'] = sales_period[0]
        result['sales_period_end'] = sales_period[1]

    return result


def calculate_comparison_group_metrics(
    matched_df: pd.DataFrame,
    campaign_metrics: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula métricas agregadas por comparison_group (ML, Fair Control, Other).

    Args:
        matched_df: DataFrame com matching e coluna 'comparison_group'
        campaign_metrics: DataFrame com métricas e custos por campanha

    Returns:
        DataFrame com métricas por grupo:
        - comparison_group: ML, Fair Control, Other
        - leads: Total de leads
        - conversions: Total de conversões
        - conversion_rate: Taxa de conversão (%)
        - total_revenue: Receita total
        - spend: Gasto total
        - cpl: Custo por lead
        - roas: Return on Ad Spend
        - margin: Margem de contribuição
    """
    # Verificar se a coluna comparison_group existe
    if 'comparison_group' not in matched_df.columns:
        logger.warning("⚠️ Coluna 'comparison_group' não encontrada. Retornando DataFrame vazio.")
        return pd.DataFrame()

    logger.info("📊 Calculando métricas por grupo de comparação...")

    # Criar mapeamento campaign → spend
    campaign_spend_map = dict(zip(
        campaign_metrics['campaign'],
        campaign_metrics['spend']
    ))

    groups_metrics = []

    for group in ['ML', 'Fair Control', 'Other']:
        group_df = matched_df[matched_df['comparison_group'] == group]

        if len(group_df) == 0:
            continue

        # Métricas básicas
        leads = len(group_df)
        conversions = len(group_df[group_df['converted'] == True])
        conversion_rate = (conversions / leads * 100) if leads > 0 else 0
        total_revenue = group_df[group_df['converted'] == True]['sale_value'].sum()

        # Calcular spend total do grupo
        group_campaigns = group_df['campaign'].unique()
        spend = sum(campaign_spend_map.get(camp, 0) for camp in group_campaigns)

        # Métricas derivadas
        cpl = (spend / leads) if leads > 0 else 0
        roas = (total_revenue / spend) if spend > 0 else 0
        margin = total_revenue - spend

        groups_metrics.append({
            'comparison_group': group,
            'leads': leads,
            'conversions': conversions,
            'conversion_rate': round(conversion_rate, 2),
            'total_revenue': round(total_revenue, 2),
            'spend': round(spend, 2),
            'cpl': round(cpl, 2),
            'roas': round(roas, 2),
            'margin': round(margin, 2),
        })

    df_result = pd.DataFrame(groups_metrics)

    if len(df_result) > 0:
        logger.info("   ✅ Métricas calculadas por grupo:")
        for _, row in df_result.iterrows():
            logger.info(f"      {row['comparison_group']}: {row['leads']} leads, "
                       f"{row['conversions']} conversões ({row['conversion_rate']:.2f}%), "
                       f"ROAS {row['roas']:.2f}x")

    return df_result
