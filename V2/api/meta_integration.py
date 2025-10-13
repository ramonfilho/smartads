"""
Integração com Meta Ads API
Busca dados de custo para enriquecer análise UTM
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class MetaAdsIntegration:
    """Cliente para integração com Meta Ads API"""

    def __init__(self, access_token: str, api_version: str = "v18.0"):
        self.access_token = access_token
        self.api_version = api_version
        self.base_url = f"https://graph.facebook.com/{api_version}"

    def get_insights(
        self,
        account_id: str,
        level: str = "campaign",
        days: int = 7,
        fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Busca insights (métricas) de uma conta de anúncios

        Args:
            account_id: ID da conta (formato: act_XXXXXXXXX)
            level: Nível de agregação (campaign, adset, ad)
            days: Número de dias para buscar dados
            fields: Campos a retornar (padrão: campaign_name, spend, impressions, clicks, actions)

        Returns:
            Lista de dicts com dados de cada campanha/adset/ad
        """
        if fields is None:
            fields = ['campaign_name', 'adset_name', 'ad_name', 'spend', 'impressions', 'clicks', 'actions']

        url = f"{self.base_url}/{account_id}/insights"

        # Calcular período
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        until = datetime.now().strftime('%Y-%m-%d')

        params = {
            'access_token': self.access_token,
            'level': level,
            'fields': ','.join(fields),
            'time_range': f'{{"since":"{since}","until":"{until}"}}',
            'limit': 1000
        }

        logger.info(f"Buscando insights: account={account_id}, level={level}, days={days}")

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            results = data.get('data', [])

            logger.info(f"✅ Insights obtidos: {len(results)} registros")
            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro ao buscar insights: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            return []

    def get_costs_by_utm(
        self,
        account_id: str,
        days: int = 7
    ) -> Dict[str, Dict[str, float]]:
        """
        Busca custos agregados por dimensões UTM

        Returns:
            {
                'campaign': {'campaign_name1': spend1, ...},
                'adset': {'adset_name1': spend1, ...},
                'ad': {'ad_name1': spend1, ...}
            }
        """
        results = {
            'campaign': {},
            'adset': {},
            'ad': {}
        }

        # Buscar em cada nível
        for level in ['campaign', 'adset', 'ad']:
            insights = self.get_insights(account_id, level=level, days=days)

            for item in insights:
                name_key = f"{level}_name"
                name = item.get(name_key)
                spend = float(item.get('spend', 0))

                if name:
                    # Agregar custos (pode ter múltiplas entradas)
                    if name in results[level]:
                        results[level][name] += spend
                    else:
                        results[level][name] = spend

        return results

    def get_costs_multiple_periods(
        self,
        account_id: str,
        periods: List[int] = [1, 3, 7]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Busca custos para múltiplos períodos

        Args:
            periods: Lista de períodos em dias (ex: [1, 3, 7])

        Returns:
            {
                '1D': {'campaign': {...}, 'adset': {...}, 'ad': {...}},
                '3D': {'campaign': {...}, 'adset': {...}, 'ad': {...}},
                '7D': {'campaign': {...}, 'adset': {...}, 'ad': {...}}
            }
        """
        results = {}

        for days in periods:
            period_key = f"{days}D"
            logger.info(f"📅 Buscando custos para período: {period_key}")
            results[period_key] = self.get_costs_by_utm(account_id, days=days)

        # Adicionar período total (último ano)
        logger.info("📅 Buscando custos para período: Total")
        results['Total'] = self.get_costs_by_utm(account_id, days=365)

        return results


def match_campaign_name(meta_name: str, utm_campaign: str) -> bool:
    """
    Faz matching entre nome da campanha do Meta e UTM campaign

    Meta pode ter sufixo: "Campaign Name | 2025-04-15|120220370119870390"
    UTM pode ser: "Campaign Name"

    Remove sufixos antes de comparar
    """
    # Remover sufixo do Meta (tudo após '|')
    meta_clean = meta_name.split('|')[0].strip()
    utm_clean = utm_campaign.strip()

    return meta_clean.lower() == utm_clean.lower()


def enrich_utm_analysis_with_costs(
    utm_analysis_df: pd.DataFrame,
    costs_data: Dict[str, Dict[str, float]],
    dimension: str
) -> pd.DataFrame:
    """
    Enriquece análise UTM com dados de custo do Meta

    Args:
        utm_analysis_df: DataFrame com análise UTM (colunas: dimension_value, leads, %D10, etc)
        costs_data: Dict com custos por dimensão (do get_costs_by_utm)
        dimension: Dimensão sendo analisada (campaign, adset, ad, medium, term)

    Returns:
        DataFrame enriquecido com coluna 'spend'
    """
    df = utm_analysis_df.copy()

    # Mapear dimensão para nível do Meta
    meta_level_map = {
        'campaign': 'campaign',
        'adset': 'adset',
        'ad': 'ad',
        'medium': None,  # Não tem correspondente direto
        'term': None,
        'content': 'ad'  # Content pode ser mapeado para Ad
    }

    meta_level = meta_level_map.get(dimension)

    if meta_level is None or meta_level not in costs_data:
        # Se não tem correspondente, adiciona coluna com 0
        df['spend'] = 0.0
        logger.warning(f"⚠️ Dimensão '{dimension}' não tem correspondente no Meta")
        return df

    # Buscar custo para cada valor da dimensão
    spend_values = []
    for value in df['value']:
        # Buscar custo exato ou fazer matching
        spend = costs_data[meta_level].get(value, 0.0)

        # Se não encontrou exato, tentar matching parcial
        if spend == 0.0:
            for meta_name, meta_spend in costs_data[meta_level].items():
                if match_campaign_name(meta_name, value):
                    spend = meta_spend
                    break

        spend_values.append(spend)

    df['spend'] = spend_values

    logger.info(f"✅ Custos adicionados: {dimension} - Total: ${sum(spend_values):.2f}")

    return df
