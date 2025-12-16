"""
Módulo para criar grupo de controle justo para comparação com campanhas ML.

Filtra campanhas não-ML que tiveram:
- Orçamento similar (±15% de tolerância)
- Mesmos criativos (creative IDs)
- Mesmo período de execução
- Mesmas características de targeting

Isso permite uma comparação apples-to-apples entre campanhas ML e não-ML.
"""

import os
import sys
import logging
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import time

# Adicionar path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Meta Ads API
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.campaign import Campaign
from facebook_business.adobjects.adset import AdSet
from facebook_business.adobjects.ad import Ad

# Token fixo do projeto
from api.meta_config import META_CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAÇÃO: NÍVEIS DE COMPARAÇÃO - EVENTO ML
# ============================================================================

# NÍVEL 1: Evento ML (adsets iguais)
# Comparação rigorosa: mesma estrutura de adsets, budget similar
ADSETS_IGUAIS_CONFIG = {
    'name': 'Evento ML (adsets iguais)',
    'ml_campaigns': [
        '120236428684840390',  # ADV ML - CBO R$ 300/dia
        '120236428684850390',  # ADV ML - CBO R$ 300/dia
    ],
    'control_campaigns': [
        '120224064762630390',  # ADV Controle - CBO R$ 390/dia
        '120224064761980390',  # ADV Controle - CBO R$ 390/dia
    ],
    'matched_ads': [
        'AD0013', 'AD0014', 'AD0017', 'AD0018', 'AD0022', 'AD0033'
    ],
    'adset_names': [
        'ADV | Linguagem de programação',
        'ADV | Lookalike 1% Cadastrados - DEV 2.0 + Interesse Ciência da Computação',
        'ADV | Lookalike 2% Cadastrados - DEV 2.0 + Interesses',
        'ADV | Lookalike 2% Alunos + Interesse Linguagem de Programação',
    ],
    'filter_by_adset': True,
    'budget_tolerance': 0.30,  # 30%
}

# NÍVEL 2: Evento ML (todos)
# Comparação geral: todas campanhas Evento ML, estruturas variadas
TODOS_CONFIG = {
    'name': 'Evento ML (todos)',
    'ml_campaigns': [
        '120236428684090390',  # ABERTO ML - CBO R$ 550/dia
        '120236428684840390',  # ADV ML - CBO R$ 300/dia
        '120236428684850390',  # ADV ML - CBO R$ 300/dia
    ],
    'control_campaigns': [
        '120220370119870390',  # ABERTO Controle - ABO
        '120224064762630390',  # ADV Controle - CBO R$ 390/dia
        '120224064761980390',  # ADV Controle - CBO R$ 390/dia
        '120224064762010390',  # ADV Controle
        '120224064762600390',  # ADV Controle
        '120228073033890390',  # ADV Controle
        '120230454190910390',  # ADV Controle
    ],
    'matched_ads': [
        'AD0004', 'AD0013', 'AD0014', 'AD0017', 'AD0018', 'AD0022', 'AD0027', 'AD0033'
    ],
    'filter_by_adset': False,
    'budget_tolerance': None,  # Sem restrição
}


def get_comparison_config(comparison_level: str = 'adsets_iguais') -> Dict:
    """
    Retorna configuração do nível de comparação desejado.

    Args:
        comparison_level: 'adsets_iguais' ou 'todos' (ou nomes antigos para retrocompatibilidade)

    Returns:
        Dict com configuração do nível
    """
    # Mapeamento de nomes (incluindo retrocompatibilidade)
    level_map = {
        'adsets_iguais': ADSETS_IGUAIS_CONFIG,
        'todos': TODOS_CONFIG,
        # Retrocompatibilidade
        'ultra_fair': ADSETS_IGUAIS_CONFIG,
        'fair': TODOS_CONFIG,
    }

    if comparison_level not in level_map:
        raise ValueError(
            f"Nível de comparação inválido: {comparison_level}. "
            f"Use 'adsets_iguais' ou 'todos'"
        )

    return level_map[comparison_level]


def filter_campaigns_by_level(
    campaigns_df: pd.DataFrame,
    ml_type: str,
    comparison_level: str = 'ultra_fair'
) -> pd.DataFrame:
    """
    Filtra campanhas de acordo com o nível de comparação.

    Args:
        campaigns_df: DataFrame com todas as campanhas
        ml_type: 'eventos_ml' ou 'controle'
        comparison_level: 'ultra_fair' ou 'fair'

    Returns:
        DataFrame filtrado
    """
    config = get_comparison_config(comparison_level)

    if ml_type == 'eventos_ml':
        campaign_ids = config['ml_campaigns']
    elif ml_type == 'controle':
        campaign_ids = config['control_campaigns']
    else:
        raise ValueError(f"ml_type inválido: {ml_type}. Use 'eventos_ml' ou 'controle'")

    # Filtrar campanhas
    filtered = campaigns_df[campaigns_df['campaign_id'].isin(campaign_ids)].copy()

    logger.info(
        f"Filtro {config['name']} ({ml_type}): "
        f"{len(filtered)} de {len(campaigns_df)} campanhas"
    )

    return filtered


def filter_ads_by_level(
    ads_df: pd.DataFrame,
    comparison_level: str = 'ultra_fair'
) -> pd.DataFrame:
    """
    Filtra anúncios de acordo com o nível de comparação.

    Para Ultra Fair: apenas matched ads do nível
    Para Fair: todos matched ads

    Args:
        ads_df: DataFrame com todos os anúncios
        comparison_level: 'ultra_fair' ou 'fair'

    Returns:
        DataFrame filtrado
    """
    config = get_comparison_config(comparison_level)
    matched_ads = config['matched_ads']

    # Filtrar por ad_code
    if 'ad_code' in ads_df.columns:
        filtered = ads_df[ads_df['ad_code'].isin(matched_ads)].copy()
    else:
        logger.warning("Coluna 'ad_code' não encontrada. Retornando DataFrame original.")
        return ads_df

    logger.info(
        f"Filtro {config['name']}: "
        f"{len(filtered)} de {len(ads_df)} anúncios matched"
    )

    return filtered


def filter_ads_by_adset(
    ads_df: pd.DataFrame,
    comparison_level: str = 'ultra_fair'
) -> pd.DataFrame:
    """
    Filtra anúncios que aparecem nos mesmos adsets (apenas para Ultra Fair).

    Args:
        ads_df: DataFrame com anúncios (deve ter colunas 'adset_name' e 'ml_type')
        comparison_level: 'ultra_fair' ou 'fair'

    Returns:
        DataFrame filtrado
    """
    config = get_comparison_config(comparison_level)

    # Se não filtrar por adset, retornar tudo
    if not config.get('filter_by_adset', False):
        return ads_df

    # Verificar colunas necessárias
    required_cols = ['adset_name', 'ml_type', 'ad_code']
    missing_cols = [col for col in required_cols if col not in ads_df.columns]
    if missing_cols:
        logger.warning(f"Colunas faltando para filtro de adset: {missing_cols}")
        return ads_df

    # Adsets válidos
    valid_adsets = config.get('adset_names', [])
    if not valid_adsets:
        logger.warning("Nenhum adset configurado para filtro. Retornando tudo.")
        return ads_df

    # Para cada matched ad, verificar se aparece em ambos ML e Controle no mesmo adset
    matched_ads_in_same_adset = set()

    for ad_code in config['matched_ads']:
        ad_data = ads_df[ads_df['ad_code'] == ad_code]

        if len(ad_data) == 0:
            continue

        # Adsets onde o ad aparece em ML
        ml_adsets = set(ad_data[ad_data['ml_type'] == 'eventos_ml']['adset_name'].unique())

        # Adsets onde o ad aparece em Controle
        control_adsets = set(ad_data[ad_data['ml_type'] == 'controle']['adset_name'].unique())

        # Interseção: adsets em ambos
        common_adsets = ml_adsets & control_adsets & set(valid_adsets)

        if common_adsets:
            matched_ads_in_same_adset.add(ad_code)
            logger.debug(f"Ad {ad_code} em adsets comuns: {common_adsets}")

    # Filtrar apenas ads que aparecem nos mesmos adsets
    filtered = ads_df[ads_df['ad_code'].isin(matched_ads_in_same_adset)].copy()

    logger.info(
        f"Filtro de adset ({config['name']}): "
        f"{len(matched_ads_in_same_adset)} matched ads em adsets comuns"
    )

    return filtered


def extract_campaign_base_structure(campaign_name: str) -> str:
    """
    Extrai a estrutura base da campanha (sem tipo específico e data).

    Ex: "DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2 | 2025-05-28"
    → "DEVLF | CAP | FRIO | FASE 04 | ADV"

    Args:
        campaign_name: Nome completo da campanha

    Returns:
        Estrutura base (primeiros 5 segmentos)
    """
    parts = campaign_name.split('|')
    # Pegar primeiros 5 segmentos (estrutura base antes do tipo)
    base = '|'.join(parts[:5]).strip() if len(parts) >= 5 else campaign_name
    return base


def extract_campaign_creation_date(campaign_name: str) -> Optional[str]:
    """
    Extrai a data de criação da campanha do nome.

    Ex: "... | PG2 | 2025-05-28|120234..."
    → "2025-05-28"

    Args:
        campaign_name: Nome completo da campanha

    Returns:
        Data no formato YYYY-MM-DD ou None
    """
    import re
    # Procurar padrão YYYY-MM-DD no nome
    match = re.search(r'(\d{4}-\d{2}-\d{2})', campaign_name)
    if match:
        return match.group(1)
    return None


def normalize_targeting_for_comparison(targeting: Dict) -> Dict:
    """
    Normaliza targeting spec para comparação consistente.

    Remove campos que podem variar mas não afetam o targeting real,
    e ordena listas para comparação determinística.
    Converte objetos do SDK Meta para dicts serializáveis.

    Args:
        targeting: Dict com targeting spec da Meta API

    Returns:
        Dict normalizado para comparação
    """
    if not targeting:
        return {}

    def convert_to_dict(obj):
        """Converte objetos do SDK Meta para dict recursivamente."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_dict(item) for item in obj]
        elif hasattr(obj, 'export_all_data'):
            # Objeto do SDK Meta - exportar dados
            return obj.export_all_data()
        elif hasattr(obj, '__dict__'):
            # Objeto genérico - converter atributos para dict
            return {k: convert_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        else:
            # Fallback: converter para string
            return str(obj)

    # Converter targeting completo para dict serializável
    targeting_dict = convert_to_dict(targeting)

    # Criar cópia para não modificar o original
    normalized = {}

    # Campos importantes para comparação 100% igual
    important_fields = [
        'age_min',
        'age_max',
        'genders',
        'geo_locations',
        'interests',
        'behaviors',
        'custom_audiences',
        'excluded_custom_audiences',
        'flexible_spec',
        'exclusions',
        'publisher_platforms',
        'facebook_positions',
        'instagram_positions',
        'device_platforms',
        'locales'
    ]

    for field in important_fields:
        if field in targeting_dict:
            value = targeting_dict[field]

            # Se for lista, ordenar para comparação consistente
            if isinstance(value, list):
                # Para listas de dicts (como interests), ordenar por ID
                if value and isinstance(value[0], dict):
                    normalized[field] = sorted(value, key=lambda x: str(x.get('id', x.get('name', ''))))
                else:
                    normalized[field] = sorted(value)
            else:
                normalized[field] = value

    return normalized


def compare_targeting_specs(targeting1: Dict, targeting2: Dict) -> Tuple[bool, float]:
    """
    Compara dois targeting specs para igualdade 100%.

    Args:
        targeting1: Targeting spec da primeira campanha
        targeting2: Targeting spec da segunda campanha

    Returns:
        Tuple (is_equal: bool, similarity_score: float)
        - is_equal: True se 100% iguais
        - similarity_score: 0.0 a 1.0 indicando similaridade
    """
    norm1 = normalize_targeting_for_comparison(targeting1)
    norm2 = normalize_targeting_for_comparison(targeting2)

    # Se ambos vazios, considerar iguais
    if not norm1 and not norm2:
        return True, 1.0

    # Se apenas um vazio, diferentes
    if not norm1 or not norm2:
        return False, 0.0

    # Verificar se todos os campos são exatamente iguais
    import json

    # Converter para JSON string para comparação (garante ordem consistente)
    json1 = json.dumps(norm1, sort_keys=True)
    json2 = json.dumps(norm2, sort_keys=True)

    is_equal = (json1 == json2)

    # Calcular score de similaridade (mesmo que não seja 100% igual)
    all_keys = set(norm1.keys()) | set(norm2.keys())
    if not all_keys:
        return False, 0.0

    matching_keys = sum(1 for k in all_keys if norm1.get(k) == norm2.get(k))
    similarity = matching_keys / len(all_keys)

    return is_equal, similarity


class FairCampaignMatcher:
    """
    Encontra campanhas de controle justo para comparação com campanhas ML.

    Usa Meta Ads API para identificar campanhas não-ML que sejam comparáveis
    às campanhas ML em termos de spend, criativos e período.
    """

    def __init__(self, account_id: str):
        """
        Inicializa o matcher com credenciais Meta API.

        Args:
            account_id: Meta Ads account ID (e.g., 'act_188005769808959')
        """
        self.account_id = account_id if account_id.startswith('act_') else f'act_{account_id}'
        self.access_token = META_CONFIG['access_token']

        # Cache para targeting (evitar chamadas repetidas à API)
        self._targeting_cache = {}
        self._api_calls_count = 0
        self._cache_hits_count = 0

        try:
            FacebookAdsApi.init(access_token=self.access_token)
            self.account = AdAccount(self.account_id)
            self.api_available = True
            logger.info(f"✅ Meta API inicializada: {self.account_id}")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar Meta API: {e}")
            self.api_available = False

    def get_ml_campaign_metadata(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, Dict]:
        """
        Obtém metadata das campanhas ML (referência para comparação).

        Args:
            start_date: Data início (YYYY-MM-DD)
            end_date: Data fim (YYYY-MM-DD)

        Returns:
            Dict com campaign_id → {name, spend, creative_ids, impressions, ...}
        """
        if not self.api_available:
            logger.warning("⚠️ Meta API não disponível")
            return {}

        logger.info(f"🔍 Buscando campanhas ML ({start_date} a {end_date})...")

        try:
            params = {
                'time_range': {'since': start_date, 'until': end_date},
                'level': 'campaign',
                'fields': [
                    'campaign_id',
                    'campaign_name',
                    'spend',
                    'impressions',
                    'clicks',
                    'actions',
                ],
                'filtering': [
                    {
                        'field': 'campaign.name',
                        'operator': 'CONTAIN',
                        'value': 'MACHINE LEARNING'
                    }
                ]
            }

            insights = self.account.get_insights(params=params)

            ml_campaigns = {}
            for insight in insights:
                campaign_id = insight.get('campaign_id')
                campaign_name = insight.get('campaign_name')

                # Verificar se é realmente ML (MACHINE LEARNING ou | ML |)
                if not ('MACHINE LEARNING' in campaign_name.upper() or '| ML |' in campaign_name.upper()):
                    continue

                # Buscar budget e creative IDs desta campanha
                campaign_obj = Campaign(campaign_id)
                campaign_data = campaign_obj.api_get(fields=[
                    Campaign.Field.daily_budget,
                    Campaign.Field.lifetime_budget,
                    Campaign.Field.name
                ])

                # Pegar budget (priorizar daily, senão lifetime)
                # Meta API retorna em centavos - converter para reais
                budget_cents = float(campaign_data.get('daily_budget', 0) or campaign_data.get('lifetime_budget', 0) or 0)
                budget = budget_cents / 100 if budget_cents > 0 else 0

                # Se budget é 0, tentar buscar dos ad sets (CBO)
                if budget == 0:
                    try:
                        adsets = campaign_obj.get_ad_sets(fields=[
                            AdSet.Field.daily_budget,
                            AdSet.Field.lifetime_budget
                        ])
                        adset_budgets = []
                        for adset in adsets:
                            # Meta API retorna em centavos - converter para reais
                            adset_budget_cents = float(adset.get('daily_budget', 0) or adset.get('lifetime_budget', 0) or 0)
                            adset_budget = adset_budget_cents / 100 if adset_budget_cents > 0 else 0
                            if adset_budget > 0:
                                adset_budgets.append(adset_budget)

                        # Somar budgets dos ad sets
                        if adset_budgets:
                            budget = sum(adset_budgets)
                            logger.debug(f"Budget da campanha {campaign_id} obtido dos ad sets: R$ {budget:.2f}")
                    except Exception as e:
                        logger.debug(f"Erro ao buscar budget dos ad sets: {e}")

                # Buscar creative IDs desta campanha
                creative_ids = self._get_campaign_creative_ids(campaign_id, start_date, end_date)

                ml_campaigns[campaign_id] = {
                    'name': campaign_name,
                    'spend': float(insight.get('spend', 0)),
                    'budget': budget,
                    'impressions': int(insight.get('impressions', 0)),
                    'clicks': int(insight.get('clicks', 0)),
                    'creative_ids': creative_ids,
                }

            logger.info(f"   ✅ {len(ml_campaigns)} campanhas ML encontradas")
            for cid, data in ml_campaigns.items():
                logger.info(f"      {data['name']}: Budget R$ {data['budget']:.2f}, Spend R$ {data['spend']:.2f}, {len(data['creative_ids'])} criativos")

            return ml_campaigns

        except Exception as e:
            logger.error(f"❌ Erro ao buscar campanhas ML: {e}")
            return {}

    def find_fair_control_campaigns(
        self,
        ml_campaign_metadata: Dict[str, Dict],
        min_creative_overlap: float = 0.8,
        budget_tolerance: float = 0.30,
        start_date: str = None,
        end_date: str = None
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Encontra campanhas de controle com características similares.

        CRITÉRIOS REFINADOS (após análise manual):
        - Budget: ±30% tolerância (não precisa ser exato)
        - Criativos: 80%+ overlap (mínimo)
        - Targeting: NÃO verificado (evita rate limits, não essencial)

        Args:
            ml_campaign_metadata: Metadata das campanhas ML (de get_ml_campaign_metadata)
            min_creative_overlap: Sobreposição mínima de criativos (0.8 = 80%+)
            budget_tolerance: Tolerância de budget (0.30 = ±30%)
            start_date: Data início (YYYY-MM-DD)
            end_date: Data fim (YYYY-MM-DD)

        Returns:
            Tuple de (fair_control_map, control_id_to_name):
            - fair_control_map: Dict com ml_campaign_id → [fair_control_campaign_ids]
            - control_id_to_name: Dict com campaign_id → campaign_name (para controles)
        """
        if not self.api_available or not ml_campaign_metadata:
            return {}, {}

        logger.info(f"🔍 Buscando campanhas de controle justo (critérios: Budget ±{budget_tolerance*100:.0f}%, Criativos {min_creative_overlap*100:.0f}%+ iguais)...")

        fair_matches = {}
        control_id_to_name = {}

        try:
            # Buscar TODAS as campanhas do período (não apenas ML)
            params = {
                'time_range': {'since': start_date, 'until': end_date},
                'level': 'campaign',
                'fields': ['campaign_id', 'campaign_name', 'spend'],
            }

            all_insights = self.account.get_insights(params=params)

            # Converter para dict - extrair campanhas não-ML
            all_campaigns = {}
            for insight in all_insights:
                campaign_id = insight.get('campaign_id')
                campaign_name = insight.get('campaign_name')

                # Excluir campanhas ML
                if 'MACHINE LEARNING' in campaign_name.upper() or '| ML |' in campaign_name.upper():
                    continue

                # Buscar budget e criativos
                budget = self._get_campaign_budget(campaign_id)
                creative_ids = self._get_campaign_creative_ids(campaign_id, start_date, end_date)

                all_campaigns[campaign_id] = {
                    'name': campaign_name,
                    'spend': float(insight.get('spend', 0)),
                    'budget': budget,
                    'creative_ids': creative_ids,
                }

            logger.info(f"   ✅ {len(all_campaigns)} campanhas não-ML encontradas para comparação")

            # Para cada campanha ML, encontrar matches usando critérios refinados
            for ml_id, ml_data in ml_campaign_metadata.items():
                ml_budget = ml_data.get('budget', 0)
                ml_creatives = set(ml_data['creative_ids'])
                ml_name = ml_data['name']

                logger.info(f"\n   🔍 Analisando ML: {ml_name[:80]}...")
                logger.info(f"      Budget: R$ {ml_budget:.2f}, {len(ml_creatives)} criativos")

                matches = []
                rejected_budget = 0
                rejected_creatives = 0

                for ctrl_id, ctrl_data in all_campaigns.items():
                    ctrl_budget = ctrl_data.get('budget', 0)
                    ctrl_creatives = set(ctrl_data['creative_ids'])
                    ctrl_name = ctrl_data['name']

                    # Critério 1: Budget dentro da tolerância (±30%)
                    if ml_budget > 0:
                        budget_diff = abs(ml_budget - ctrl_budget) / ml_budget
                        if budget_diff > budget_tolerance:
                            rejected_budget += 1
                            if rejected_budget <= 3:  # Mostrar primeiras 3
                                logger.info(f"      ❌ Budget: {ctrl_name[:60]} (R$ {ctrl_budget:.2f}, diff {budget_diff*100:.0f}% > {budget_tolerance*100:.0f}%)")
                            continue

                    # Critério 2: Sobreposição de criativos (80%+)
                    if len(ml_creatives) == 0 or len(ctrl_creatives) == 0:
                        rejected_creatives += 1
                        continue

                    overlap = len(ml_creatives & ctrl_creatives)
                    overlap_pct_ml = overlap / len(ml_creatives)
                    overlap_pct_ctrl = overlap / len(ctrl_creatives)
                    min_overlap_pct = min(overlap_pct_ml, overlap_pct_ctrl)

                    if min_overlap_pct < min_creative_overlap:
                        rejected_creatives += 1
                        if rejected_creatives <= 3:  # Mostrar primeiras 3
                            logger.info(f"      ❌ Criativos: {ctrl_name[:60]} ({min_overlap_pct*100:.0f}% < {min_creative_overlap*100:.0f}%)")
                        continue

                    # ✅ MATCH ENCONTRADO! Budget ±30% e Criativos 80%+
                    logger.info(f"      ✅ MATCH: {ctrl_name[:60]}")
                    logger.info(f"         Budget: R$ {ctrl_budget:.2f} (diff {budget_diff*100:.0f}%), Criativos: {overlap}/{len(ml_creatives)} ({min_overlap_pct*100:.0f}%)")

                    match_info = {
                        'id': ctrl_id,
                        'name': ctrl_data['name'],
                        'spend': ctrl_data['spend'],
                        'budget': ctrl_budget,
                        'creative_overlap': overlap,
                        'creative_overlap_pct': min_overlap_pct * 100,
                        'match_score': (1 - budget_diff) * min_overlap_pct  # Score combinado
                    }

                    matches.append(match_info)

                # Mostrar resumo de filtros
                total_checked = rejected_budget + rejected_creatives + len(matches)
                logger.info(f"\n      📊 Resumo de filtros para esta ML:")
                logger.info(f"         Total verificadas: {total_checked}")
                logger.info(f"         ❌ Rejeitadas por Budget: {rejected_budget}")
                logger.info(f"         ❌ Rejeitadas por Criativos: {rejected_creatives}")
                logger.info(f"         ✅ Aprovadas (Fair Control): {len(matches)}")

                # Ordenar por match score (melhor combinação de budget + criativos)
                matches.sort(key=lambda x: -x['match_score'])

                fair_matches[ml_id] = [m['id'] for m in matches]

                # Adicionar ao mapeamento id → name
                for m in matches:
                    control_id_to_name[m['id']] = m['name']

                if matches:
                    logger.info(f"\n   ✅ {ml_name}:")
                    logger.info(f"      🎯 Encontradas {len(matches)} campanhas Fair Control:")
                    for m in matches[:3]:  # Top 3
                        logger.info(f"         • {m['name'][:60]}")
                        logger.info(f"           Budget: R$ {m['budget']:.2f}, Criativos: {m['creative_overlap']} ({m['creative_overlap_pct']:.0f}%), Score: {m['match_score']:.2f}")
                else:
                    logger.warning(f"   ⚠️ Nenhuma campanha Fair Control encontrada")

            return fair_matches, control_id_to_name

        except Exception as e:
            logger.error(f"❌ Erro ao buscar campanhas de controle: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}, {}

    def _get_campaign_budget(self, campaign_id: str) -> float:
        """
        Extrai budget de uma campanha (daily ou lifetime).

        Args:
            campaign_id: ID da campanha

        Returns:
            Budget em R$ (convertido de centavos)
        """
        try:
            campaign_obj = Campaign(campaign_id)
            campaign_data = campaign_obj.api_get(fields=[
                Campaign.Field.daily_budget,
                Campaign.Field.lifetime_budget
            ])
            # Meta API retorna em centavos - converter para reais
            budget_cents = float(campaign_data.get('daily_budget', 0) or campaign_data.get('lifetime_budget', 0) or 0)
            budget = budget_cents / 100 if budget_cents > 0 else 0

            # Se budget é 0, tentar buscar dos ad sets (CBO)
            if budget == 0:
                try:
                    adsets = campaign_obj.get_ad_sets(fields=[
                        AdSet.Field.daily_budget,
                        AdSet.Field.lifetime_budget
                    ])
                    adset_budgets = []
                    for adset in adsets:
                        adset_budget_cents = float(adset.get('daily_budget', 0) or adset.get('lifetime_budget', 0) or 0)
                        adset_budget = adset_budget_cents / 100 if adset_budget_cents > 0 else 0
                        if adset_budget > 0:
                            adset_budgets.append(adset_budget)

                    if adset_budgets:
                        budget = sum(adset_budgets)
                        logger.debug(f"Budget da campanha {campaign_id} obtido dos ad sets: R$ {budget:.2f}")
                except Exception as e_adset:
                    logger.debug(f"Erro ao buscar budget dos ad sets: {e_adset}")

            return budget

        except Exception as e:
            logger.debug(f"Erro ao buscar budget da campanha {campaign_id}: {e}")
            return 0.0

    def _get_campaign_creative_ids(
        self,
        campaign_id: str,
        start_date: str,
        end_date: str
    ) -> List[str]:
        """
        Extrai creative IDs de uma campanha.

        Args:
            campaign_id: ID da campanha
            start_date: Data início
            end_date: Data fim

        Returns:
            Lista de creative IDs únicos
        """
        try:
            campaign = Campaign(campaign_id)

            # Buscar ads da campanha (sem filtro de time_range, buscar todos)
            ads = campaign.get_ads(
                fields=[Ad.Field.creative, Ad.Field.name, Ad.Field.status],
                params={
                    'effective_status': ['ACTIVE', 'PAUSED', 'ARCHIVED'],
                }
            )

            creative_ids = set()
            ad_count = 0
            for ad in ads:
                ad_count += 1
                creative = ad.get(Ad.Field.creative)

                if creative:
                    # Pode vir como dict ou como objeto AdCreative
                    if isinstance(creative, dict):
                        creative_id = creative.get('id')
                    else:
                        # É um objeto AdCreative, pegar o ID dele
                        creative_id = getattr(creative, 'get_id', lambda: None)() or creative.get('id', None)

                    if creative_id:
                        creative_ids.add(str(creative_id))

            logger.debug(f"Campanha {campaign_id}: {ad_count} ads encontrados, {len(creative_ids)} criativos únicos")
            return list(creative_ids)

        except Exception as e:
            logger.debug(f"Erro ao buscar criativos da campanha {campaign_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    def create_comparison_groups(
        self,
        leads_df: pd.DataFrame,
        ml_campaign_metadata: Dict[str, Dict],
        fair_control_map: Dict[str, List[str]],
        control_id_to_name: Dict[str, str],
        campaign_hierarchy: Dict[str, Dict] = None,
        campaigns_with_custom_events: Set[str] = None
    ) -> pd.DataFrame:
        """
        Adiciona coluna 'comparison_group' aos leads baseado nos matches.

        Args:
            leads_df: DataFrame de leads com coluna 'campaign'
            ml_campaign_metadata: Metadata das campanhas ML (com nomes)
            fair_control_map: Mapeamento ML campaign_id → [control campaign_ids]
            control_id_to_name: Mapeamento control campaign_id → campaign_name
            campaign_hierarchy: Hierarquia de campanhas com optimization_goal (opcional)
            campaigns_with_custom_events: Set de campaign_ids que usam eventos customizados

        Returns:
            DataFrame com coluna 'comparison_group' adicionada:
            - 'Eventos ML': Campanha com eventos customizados
            - 'Otimização ML': Campanha com ML mas eventos padrão
            - 'Controle': Campanha de controle justo
            - 'Outro': Outras campanhas
        """
        df = leads_df.copy()

        # Debug: verificar colunas disponíveis
        has_lq = 'LeadQualified' in df.columns
        has_lqhq = 'LeadQualifiedHighQuality' in df.columns
        logger.info(f"   🔍 Colunas de eventos customizados: LeadQualified={has_lq}, LeadQualifiedHighQuality={has_lqhq}")
        if has_lq:
            lq_sum = df['LeadQualified'].sum()
            lq_campaigns = df[df['LeadQualified'] > 0]['campaign'].unique()
            logger.info(f"   📊 Total LeadQualified: {lq_sum} em {len(lq_campaigns)} campanhas")
            if len(lq_campaigns) > 0:
                logger.info(f"   📋 Campanhas com LeadQualified: {lq_campaigns[:3]}")

        # Criar set de nomes COMPLETOS de campanhas de controle justo (incluindo ID)
        fair_control_names = set(control_id_to_name.values())

        # Enriquecer com optimization_goal se não existir
        if 'optimization_goal' not in df.columns:
            # Criar mapeamento campaign_id → optimization_goal da hierarquia
            if campaign_hierarchy:
                logger.info(f"   🔍 Enriquecendo leads com optimization_goal ({len(campaign_hierarchy)} campanhas na hierarquia)")
                opt_goal_map = {}
                for campaign_id, campaign_data in campaign_hierarchy.items():
                    # Verificar TODOS os adsets para detectar eventos customizados
                    # Se qualquer adset usa evento customizado, a campanha é "Eventos ML"
                    adsets = campaign_data.get('adsets', {})
                    optimization_goals = []
                    for adset_id, adset_data in adsets.items():
                        opt_goal = adset_data.get('optimization_goal', 'Lead')
                        optimization_goals.append(opt_goal)

                    # Se qualquer adset usa LeadQualified ou LeadQualifiedHighQuality, usar isso
                    if 'LeadQualified' in optimization_goals:
                        opt_goal_map[campaign_id] = 'LeadQualified'
                    elif 'LeadQualifiedHighQuality' in optimization_goals:
                        opt_goal_map[campaign_id] = 'LeadQualifiedHighQuality'
                    else:
                        # Usar o primeiro adset como antes
                        opt_goal_map[campaign_id] = optimization_goals[0] if optimization_goals else 'Lead'

                logger.info(f"   ✅ Mapeamento criado para {len(opt_goal_map)} campanhas")

                # Extrair campaign_id da coluna campaign (formato: "nome|campaign_id")
                def get_campaign_id(campaign_str):
                    if pd.isna(campaign_str) or not isinstance(campaign_str, str):
                        return None
                    if '|' in campaign_str:
                        parts = campaign_str.rsplit('|', 1)
                        if len(parts) == 2 and parts[1].strip().isdigit():
                            return parts[1].strip()
                    return None

                df['_campaign_id'] = df['campaign'].apply(get_campaign_id)
                df['optimization_goal'] = df['_campaign_id'].map(opt_goal_map).fillna('Lead')
                df = df.drop(columns=['_campaign_id'])

                # Log de campanhas COM_ML e seus optimization_goals
                com_ml_campaigns = df[df['ml_type'] == 'COM_ML']['campaign'].unique()
                if len(com_ml_campaigns) > 0:
                    logger.info(f"   📊 Campanhas COM_ML e seus eventos:")
                    for camp in com_ml_campaigns[:5]:  # Primeiras 5
                        camp_rows = df[df['campaign'] == camp]
                        if len(camp_rows) > 0:
                            opt_goal = camp_rows.iloc[0]['optimization_goal']
                            logger.info(f"      • {camp[:60]}: {opt_goal}")
            else:
                logger.warning("   ⚠️ Hierarquia de campanhas não fornecida - usando 'Lead' padrão")

        # Classificar
        def classify_group(row):
            campaign_name = row.get('campaign', '')
            optimization_goal = row.get('optimization_goal', '')

            # Normalizar: remover sufixo |campaign_id se presente
            campaign_name_base = campaign_name
            campaign_id = None
            if '|' in campaign_name:
                parts = campaign_name.rsplit('|', 1)
                if len(parts) == 2 and parts[1].strip().isdigit():
                    campaign_name_base = parts[0].strip()
                    campaign_id = parts[1].strip()

            # Critério 1: Eventos ML (COM_ML + usa eventos customizados)
            if row.get('ml_type') == 'COM_ML':
                # Verificar se usa eventos customizados através de múltiplas fontes:
                # 1. Optimization goal da hierarquia
                # 2. Set explícito de campanhas com eventos customizados
                # 3. Heurística por data: campanhas criadas a partir de 25/11 usam eventos customizados
                uses_custom_events_by_goal = optimization_goal in ['LeadQualified', 'LeadQualifiedHighQuality']
                uses_custom_events_by_set = campaigns_with_custom_events and campaign_id in campaigns_with_custom_events

                # Heurística por data: extrair data do nome da campanha (formato: | YYYY-MM-DD)
                uses_custom_events_by_date = False
                import re
                date_match = re.search(r'\| (\d{4})-(\d{2})-(\d{2})', campaign_name)
                if date_match:
                    year, month, day = date_match.groups()
                    campaign_date = f"{year}-{month}-{day}"
                    # Campanhas Eventos ML foram criadas a partir de 2025-11-25 (conta 1880)
                    # ou 2025-11-11 (conta 7867 - teste com LeadQualified)
                    if campaign_date >= "2025-11-11":
                        uses_custom_events_by_date = True

                if uses_custom_events_by_goal or uses_custom_events_by_set or uses_custom_events_by_date:
                    return 'Eventos ML'
                else:
                    return 'Otimização ML'

            # Critério 2: É campanha de controle justo?
            # Check both full name and base name (without ID suffix)
            elif row.get('ml_type') == 'SEM_ML' and (
                campaign_name in fair_control_names or
                campaign_name_base in fair_control_names
            ):
                return 'Controle'

            # Outras campanhas
            else:
                return 'Outro'

        df['comparison_group'] = df.apply(classify_group, axis=1)

        # Log da distribuição
        logger.info(f"\n   📊 Distribuição de grupos de comparação:")
        for group in ['Eventos ML', 'Otimização ML', 'Controle', 'Outro']:
            count = len(df[df['comparison_group'] == group])
            pct = count / len(df) * 100 if len(df) > 0 else 0
            logger.info(f"      {group}: {count} leads ({pct:.1f}%)")

        return df


# =============================================================================
# AD-LEVEL COMPARISON - Comparação Automática por Anúncio
# =============================================================================

def identify_matched_ad_pairs(
    campaigns_df: pd.DataFrame,
    ml_campaign_ids: List[str],
    control_campaign_ids: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Identifica anúncios que aparecem tanto em campanhas ML quanto controle.

    Args:
        campaigns_df: DataFrame com campanhas e ads
        ml_campaign_ids: IDs das campanhas ML
        control_campaign_ids: IDs das campanhas controle

    Returns:
        Tuple (matched_ads, exclusive_ml_ads)
    """
    logger.info("🔍 Identificando matched pairs de anúncios...")

    # Anúncios em campanhas ML
    ml_ads = set()
    for campaign_id in ml_campaign_ids:
        camp_row = campaigns_df[campaigns_df['campaign_id'] == campaign_id]
        if not camp_row.empty:
            # Extrair TODOS os ad_codes desta campanha (não apenas a primeira linha)
            ads = camp_row['ad_code'].dropna().unique().tolist()
            ml_ads.update(ads)

    # Anúncios em campanhas controle
    control_ads = set()
    for campaign_id in control_campaign_ids:
        camp_row = campaigns_df[campaigns_df['campaign_id'] == campaign_id]
        if not camp_row.empty:
            # Extrair TODOS os ad_codes desta campanha (não apenas a primeira linha)
            ads = camp_row['ad_code'].dropna().unique().tolist()
            control_ads.update(ads)

    # Matched pairs
    matched = list(ml_ads.intersection(control_ads))
    exclusive_ml = list(ml_ads - control_ads)

    logger.info(f"   ✅ {len(matched)} anúncios matched")
    logger.info(f"   ⚠️  {len(exclusive_ml)} anúncios exclusivos ML")

    return matched, exclusive_ml


def get_ad_level_metrics(
    account_id: str,
    campaign_ids: List[str],
    since_date: str,
    until_date: str,
    access_token: str
) -> pd.DataFrame:
    """
    Busca métricas detalhadas por anúncio via Meta Ads API.

    Args:
        account_id: ID da conta Meta Ads
        campaign_ids: Lista de IDs de campanhas
        since_date: Data início (YYYY-MM-DD)
        until_date: Data fim (YYYY-MM-DD)
        access_token: Token de acesso Meta API

    Returns:
        DataFrame com métricas por anúncio
    """
    logger.info(f"📊 Buscando métricas por anúncio (período: {since_date} a {until_date})...")

    FacebookAdsApi.init(access_token=access_token)
    account = AdAccount(account_id)

    try:
        insights = account.get_insights(
            fields=[
                'ad_id',
                'ad_name',
                'campaign_id',
                'campaign_name',
                'adset_id',
                'adset_name',
                'spend',
                'impressions',
                'clicks',
                'ctr',
                'cpm',
                'cpc'
            ],
            params={
                'time_range': {'since': since_date, 'until': until_date},
                'level': 'ad',
                'filtering': [
                    {
                        'field': 'campaign.id',
                        'operator': 'IN',
                        'value': campaign_ids
                    }
                ],
                'limit': 500
            }
        )

        insights_list = list(insights)

        if len(insights_list) == 0:
            logger.warning("   ⚠️  Nenhum dado de anúncio encontrado no período")
            return pd.DataFrame()

        df = pd.DataFrame(insights_list)

        # Extrair código do anúncio (AD0XXX)
        df['ad_code'] = df['ad_name'].str.extract(r'(AD0\d+)', expand=False)

        # Converter métricas para tipos corretos
        numeric_cols = ['spend', 'impressions', 'clicks', 'ctr', 'cpm', 'cpc']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"   ✅ {len(df)} anúncios com dados no período")

        return df

    except Exception as e:
        logger.error(f"   ❌ Erro ao buscar insights de anúncios: {e}")
        return pd.DataFrame()


def compare_ad_performance(
    ad_metrics_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    ml_type_map: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    """
    Compara performance de anúncios entre categorias (EVENTOS ML, OTIM ML, CONTROLE).

    Args:
        ad_metrics_df: DataFrame com métricas de anúncios (da Meta API)
        matched_df: DataFrame com conversões matched (leads→vendas)
        ml_type_map: Mapeamento campaign_id → ml_type

    Returns:
        Dict com 4 DataFrames:
        - 'campaign_matches': Campanhas 100% matched
        - 'aggregated_matched': Agregação de matched pairs
        - 'detailed_matched': Detalhamento anúncio-a-anúncio matched
        - 'all_ads': Todos os anúncios (incluindo exclusivos)
    """
    logger.info("📊 Comparando performance de anúncios...")

    # Adicionar ml_type aos ad_metrics
    ad_metrics_df['ml_type'] = ad_metrics_df['campaign_id'].map(ml_type_map)

    # Calcular conversões por CAMPANHA (não temos granularidade por anúncio)
    conversions_by_campaign = matched_df[matched_df['converted'] == True].groupby(
        'campaign_id'
    ).size().reset_index(name='campaign_conversions')

    # Merge conversões da campanha
    ad_full = ad_metrics_df.merge(
        conversions_by_campaign,
        on='campaign_id',
        how='left'
    )
    ad_full['campaign_conversions'] = ad_full['campaign_conversions'].fillna(0)

    # Distribuir conversões proporcionalmente ao spend de cada anúncio
    ad_full['spend_pct'] = ad_full.groupby('campaign_id')['spend'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )
    ad_full['conversions'] = (ad_full['campaign_conversions'] * ad_full['spend_pct']).fillna(0)

    # Calcular CPL e ROAS (assumindo product_value fixo)
    # TODO: Buscar de business_config
    PRODUCT_VALUE = 2000.0

    ad_full['leads'] = ad_full.groupby(['campaign_id', 'ad_code'])['ad_id'].transform('count')
    ad_full['cpl'] = ad_full['spend'] / ad_full['leads'].replace(0, 1)
    ad_full['conversion_rate'] = (ad_full['conversions'] / ad_full['leads'].replace(0, 1)) * 100
    ad_full['revenue'] = ad_full['conversions'] * PRODUCT_VALUE
    ad_full['roas'] = ad_full['revenue'] / ad_full['spend'].replace(0, 1)

    # 1. Campaign Matches (100% matched)
    # TODO: Implementar lógica de identificação automática

    # 2. Aggregated Matched Pairs
    matched_ads, exclusive_ads = identify_matched_ad_pairs(
        campaigns_df=ad_full,
        ml_campaign_ids=ad_full[ad_full['ml_type'] == 'COM_ML']['campaign_id'].unique().tolist(),
        control_campaign_ids=ad_full[ad_full['ml_type'] != 'COM_ML']['campaign_id'].unique().tolist()
    )

    matched_pairs_df = ad_full[ad_full['ad_code'].isin(matched_ads)]

    aggregated = matched_pairs_df.groupby('ml_type').agg({
        'ad_code': 'nunique',
        'spend': 'sum',
        'cpl': 'mean',
        'conversion_rate': 'mean',
        'roas': 'mean'
    }).reset_index()

    # 3. Detailed Matched (anúncio-a-anúncio)
    detailed = matched_pairs_df.groupby(['ad_code', 'ml_type']).agg({
        'spend': 'sum',
        'leads': 'sum',
        'cpl': 'mean',
        'conversions': 'sum',
        'conversion_rate': 'mean',
        'roas': 'mean'
    }).reset_index()

    # 4. All Ads (incluindo exclusivos)
    all_ads = ad_full.copy()
    all_ads['is_matched'] = all_ads['ad_code'].isin(matched_ads)

    logger.info("   ✅ Comparações calculadas")

    return {
        'campaign_matches': pd.DataFrame(),  # TODO
        'aggregated_matched': aggregated,
        'detailed_matched': detailed,
        'all_ads': all_ads
    }


def prepare_ad_comparison_for_excel(
    comparisons: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Prepara DataFrames de comparação por anúncio para Excel.

    Retorna DataFrames prontos para serem salvos como abas no Excel.

    Args:
        comparisons: Dict com os 4 DataFrames de comparação

    Returns:
        Dict com DataFrames formatados para Excel:
        - 'aggregated': Agregação de matched pairs
        - 'detailed': Detalhamento anúncio-a-anúncio
        - 'all_summary': Resumo de todos os anúncios
    """
    logger.info("📝 Preparando comparações por anúncio para Excel...")

    excel_dfs = {}

    # 1. Aggregated Matched Pairs
    if not comparisons['aggregated_matched'].empty:
        df = comparisons['aggregated_matched'].copy()
        df.columns = ['Categoria', 'Anúncios Únicos', 'Gasto Total (R$)',
                      'CPL Médio (R$)', 'Taxa Conversão (%)', 'ROAS']
        excel_dfs['aggregated'] = df

    # 2. Detailed Matched (top 20 por ROAS)
    if not comparisons['detailed_matched'].empty:
        df = comparisons['detailed_matched'].copy()

        # Pivotar para ter categorias como colunas
        pivot = df.pivot_table(
            index='ad_code',
            columns='ml_type',
            values=['spend', 'cpl', 'roas'],
            aggfunc='mean'
        )

        # Flatten multi-index columns
        pivot.columns = [f'{col[1]}_{col[0]}' for col in pivot.columns]
        pivot = pivot.reset_index()
        pivot.columns = ['Anúncio'] + [col.replace('_', ' ') for col in pivot.columns[1:]]

        # Ordenar por ROAS médio
        roas_cols = [col for col in pivot.columns if 'roas' in col.lower()]
        if roas_cols:
            pivot['ROAS_Médio'] = pivot[roas_cols].mean(axis=1)
            pivot = pivot.nlargest(20, 'ROAS_Médio')
            pivot = pivot.drop('ROAS_Médio', axis=1)

        excel_dfs['detailed'] = pivot

    # 3. All Ads Summary
    if not comparisons['all_ads'].empty:
        df = comparisons['all_ads'].copy()

        summary = df.groupby(['ml_type', 'is_matched']).agg({
            'ad_code': 'nunique',
            'spend': 'sum',
            'cpl': 'mean',
            'roas': 'mean',
            'conversion_rate': 'mean'
        }).reset_index()

        summary['is_matched'] = summary['is_matched'].map({
            True: 'Matched',
            False: 'Exclusivos'
        })

        summary.columns = ['Categoria', 'Tipo', 'Anúncios', 'Gasto (R$)',
                          'CPL (R$)', 'ROAS', 'Taxa Conversão (%)']

        excel_dfs['all_summary'] = summary

    logger.info(f"   ✅ {len(excel_dfs)} abas preparadas para Excel")

    return excel_dfs
