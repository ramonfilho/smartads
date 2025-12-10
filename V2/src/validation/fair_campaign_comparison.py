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

    def get_campaign_adset_targeting(self, campaign_id: str) -> Dict[str, Dict]:
        """
        Busca targeting specs de todos os adsets de uma campanha (com cache).

        Args:
            campaign_id: ID da campanha

        Returns:
            Dict com adset_id → targeting_spec
            Ex: {
                '123456': {'age_min': 18, 'age_max': 65, 'genders': [1,2], 'interests': [...]},
                '123457': {'age_min': 25, 'age_max': 55, 'genders': [2], 'custom_audiences': [...]}
            }
        """
        # Verificar cache primeiro
        if campaign_id in self._targeting_cache:
            self._cache_hits_count += 1
            logger.debug(f"   ✓ Cache hit para campanha {campaign_id} (total hits: {self._cache_hits_count})")
            return self._targeting_cache[campaign_id]

        if not self.api_available:
            return {}

        try:
            # Delay para evitar rate limiting (1.5 segundos entre chamadas)
            self._api_calls_count += 1
            logger.info(f"   🔄 Buscando targeting da campanha {campaign_id} (chamada API #{self._api_calls_count})...")
            time.sleep(1.5)

            campaign = Campaign(campaign_id)
            adsets = campaign.get_ad_sets(fields=[
                AdSet.Field.id,
                AdSet.Field.name,
                AdSet.Field.targeting
            ])

            targeting_map = {}
            for adset in adsets:
                adset_id = adset.get('id')
                targeting = adset.get('targeting', {})
                if adset_id:
                    targeting_map[adset_id] = targeting

            # Armazenar no cache
            self._targeting_cache[campaign_id] = targeting_map
            logger.debug(f"   ✓ Targeting obtido: {len(targeting_map)} adsets")

            return targeting_map

        except Exception as e:
            logger.warning(f"⚠️ Erro ao buscar targeting dos adsets da campanha {campaign_id}: {e}")
            # Cachear resultado vazio para evitar retry repetido
            self._targeting_cache[campaign_id] = {}
            return {}

    def compare_campaign_targeting(self, campaign1_id: str, campaign2_id: str) -> Tuple[bool, float]:
        """
        Compara o targeting de duas campanhas (agregado de todos os adsets).

        Critério: TODAS as combinações de adsets devem ter targeting 100% igual.

        Args:
            campaign1_id: ID da primeira campanha
            campaign2_id: ID da segunda campanha

        Returns:
            Tuple (is_equal: bool, avg_similarity: float)
        """
        targeting1 = self.get_campaign_adset_targeting(campaign1_id)
        targeting2 = self.get_campaign_adset_targeting(campaign2_id)

        # Se não conseguiu buscar targeting de alguma campanha, considerar diferentes
        if not targeting1 or not targeting2:
            return False, 0.0

        # Se número de adsets diferente, já são diferentes
        if len(targeting1) != len(targeting2):
            return False, 0.0

        # Comparar cada adset com cada adset (melhor match)
        # Para campanhas duplicadas, os adsets devem ter targeting idêntico
        similarities = []
        for t1 in targeting1.values():
            best_match = 0.0
            for t2 in targeting2.values():
                is_equal, similarity = compare_targeting_specs(t1, t2)
                best_match = max(best_match, similarity)
            similarities.append(best_match)

        if not similarities:
            return False, 0.0

        avg_similarity = sum(similarities) / len(similarities)

        # Para ser considerado match, TODOS os adsets devem ter 100% de match
        is_equal = all(s >= 1.0 for s in similarities)

        return is_equal, avg_similarity

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
        start_date: str = None,
        end_date: str = None
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Encontra campanhas de controle com características similares.

        Args:
            ml_campaign_metadata: Metadata das campanhas ML (de get_ml_campaign_metadata)
            min_creative_overlap: Sobreposição mínima de criativos (0.8 = 80%+)
            start_date: Data início (YYYY-MM-DD)
            end_date: Data fim (YYYY-MM-DD)

        Returns:
            Tuple de (fair_control_map, control_id_to_name):
            - fair_control_map: Dict com ml_campaign_id → [fair_control_campaign_ids]
            - control_id_to_name: Dict com campaign_id → campaign_name (para controles)
        """
        if not self.api_available or not ml_campaign_metadata:
            return {}, {}

        logger.info(f"🔍 Buscando campanhas de controle justo (critérios: Budget 100% igual, Criativos {min_creative_overlap*100:.0f}%+ iguais, Targeting 100% igual)...")

        fair_matches = {}
        control_id_to_name = {}  # Mapeamento campaign_id → campaign_name para controles

        try:
            # Buscar TODAS as campanhas do período (não apenas ML)
            params = {
                'time_range': {'since': start_date, 'until': end_date},
                'level': 'campaign',
                'fields': [
                    'campaign_id',
                    'campaign_name',
                    'spend',
                    'impressions',
                    'clicks',
                ],
            }

            all_insights = self.account.get_insights(params=params)

            # Converter para dict para facilitar busca
            all_campaigns = {}
            for insight in all_insights:
                campaign_id = insight.get('campaign_id')
                campaign_name = insight.get('campaign_name')

                # Excluir campanhas ML
                if 'MACHINE LEARNING' in campaign_name.upper() or '| ML |' in campaign_name.upper():
                    continue

                # Buscar budget
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
                                # Meta API retorna em centavos - converter para reais
                                adset_budget_cents = float(adset.get('daily_budget', 0) or adset.get('lifetime_budget', 0) or 0)
                                adset_budget = adset_budget_cents / 100 if adset_budget_cents > 0 else 0
                                if adset_budget > 0:
                                    adset_budgets.append(adset_budget)

                            # Somar budgets dos ad sets
                            if adset_budgets:
                                budget = sum(adset_budgets)
                                logger.debug(f"Budget da campanha {campaign_id} obtido dos ad sets: R$ {budget:.2f}")
                        except Exception as e_adset:
                            logger.debug(f"Erro ao buscar budget dos ad sets: {e_adset}")

                except Exception as e:
                    logger.debug(f"Erro ao buscar budget da campanha {campaign_id}: {e}")
                    budget = 0

                # Buscar creative IDs
                creative_ids = self._get_campaign_creative_ids(campaign_id, start_date, end_date)

                all_campaigns[campaign_id] = {
                    'name': campaign_name,
                    'spend': float(insight.get('spend', 0)),
                    'budget': budget,
                    'impressions': int(insight.get('impressions', 0)),
                    'clicks': int(insight.get('clicks', 0)),
                    'creative_ids': creative_ids,
                }

            logger.info(f"   ✅ {len(all_campaigns)} campanhas não-ML encontradas para comparação")

            # Para cada campanha ML, encontrar matches
            for ml_id, ml_data in ml_campaign_metadata.items():
                ml_spend = ml_data['spend']
                ml_budget = ml_data.get('budget', 0)  # Daily ou lifetime budget
                ml_creatives = set(ml_data['creative_ids'])
                ml_name = ml_data['name']

                logger.info(f"\n   🔍 Analisando ML: {ml_name[:80]}...")
                logger.info(f"      Budget: R$ {ml_budget:.2f}, {len(ml_creatives)} criativos")

                matches = []
                rejected_budget = 0
                rejected_creatives = 0
                rejected_targeting = 0

                for ctrl_id, ctrl_data in all_campaigns.items():
                    ctrl_spend = ctrl_data['spend']
                    ctrl_budget = ctrl_data.get('budget', 0)
                    ctrl_creatives = set(ctrl_data['creative_ids'])
                    ctrl_name = ctrl_data['name']

                    # Critério 1: Budget 100% igual (deve ser exatamente o mesmo)
                    if ml_budget != ctrl_budget:
                        rejected_budget += 1
                        if rejected_budget <= 3:  # Mostrar primeiras 3
                            logger.info(f"      ❌ Budget: {ctrl_name[:60]} (R$ {ctrl_budget:.2f} ≠ R$ {ml_budget:.2f})")
                        continue

                    # Critério 2: Alta sobreposição de criativos (80%+)
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

                    # ✅ Passou Budget e Criativos! Agora verificar targeting
                    logger.info(f"      ✅ Budget+Criativos OK: {ctrl_name[:60]}")
                    logger.info(f"         Verificando targeting...")

                    # Critério 3: Targeting 100% igual (AdSet level)
                    # Comparar targeting specs de todos os adsets
                    targeting_equal, targeting_similarity = self.compare_campaign_targeting(ml_id, ctrl_id)

                    if not targeting_equal:
                        # Targeting não é 100% igual, rejeitar
                        rejected_targeting += 1
                        logger.info(f"         ❌ Targeting: {ctrl_name[:60]} ({targeting_similarity*100:.0f}% similar)")
                        continue

                    # ✅ MATCH ENCONTRADO! Budget, criativos E targeting são 100% iguais
                    logger.info(f"         ✅ MATCH! Targeting 100% igual!")
                    match_info = {
                        'id': ctrl_id,
                        'name': ctrl_data['name'],
                        'spend': ctrl_spend,
                        'budget': ctrl_budget,
                        'creative_overlap': overlap,
                        'creative_overlap_pct': min_overlap_pct * 100,
                        'targeting_similarity': targeting_similarity * 100,
                    }

                    matches.append(match_info)

                # Mostrar resumo de rejeições
                total_checked = rejected_budget + rejected_creatives + rejected_targeting + len(matches)
                logger.info(f"\n      📊 Resumo de filtros para esta ML:")
                logger.info(f"         Total verificadas: {total_checked}")
                logger.info(f"         ❌ Rejeitadas por Budget: {rejected_budget}")
                logger.info(f"         ❌ Rejeitadas por Criativos: {rejected_creatives}")
                logger.info(f"         ❌ Rejeitadas por Targeting: {rejected_targeting}")
                logger.info(f"         ✅ Aprovadas (Fair Control): {len(matches)}")

                # Ordenar por melhor match (maior sobreposição de criativos)
                matches.sort(key=lambda x: -x['creative_overlap'])

                fair_matches[ml_id] = [m['id'] for m in matches]

                # Adicionar ao mapeamento id → name APENAS as campanhas selecionadas
                for m in matches:
                    control_id_to_name[m['id']] = m['name']

                if matches:
                    logger.info(f"\n   ✅ {ml_data['name']}:")
                    logger.info(f"      Budget: R$ {ml_budget:.2f} | Spend: R$ {ml_spend:.2f}")
                    logger.info(f"      🎯 Encontradas {len(matches)} campanhas Fair Control (Budget + Criativos + Targeting 100% iguais):")
                    for m in matches[:3]:  # Top 3
                        logger.info(f"         • {m['name']}: Budget R$ {m['budget']:.2f}, {m['creative_overlap']} criativos ({m['creative_overlap_pct']:.0f}% overlap), Targeting {m['targeting_similarity']:.0f}% similar")
                else:
                    logger.warning(f"   ⚠️ Nenhuma campanha Fair Control encontrada")

            # Log resumo de performance da API
            total_checks = self._api_calls_count + self._cache_hits_count
            if total_checks > 0:
                cache_rate = (self._cache_hits_count / total_checks) * 100
                logger.info(f"\n   📊 Resumo de chamadas à API:")
                logger.info(f"      🔄 Chamadas API: {self._api_calls_count}")
                logger.info(f"      ✓ Cache hits: {self._cache_hits_count}")
                logger.info(f"      📈 Taxa de cache: {cache_rate:.1f}%")

            return fair_matches, control_id_to_name

        except Exception as e:
            logger.error(f"❌ Erro ao buscar campanhas de controle: {e}")
            # Log resumo mesmo em caso de erro
            total_checks = self._api_calls_count + self._cache_hits_count
            if total_checks > 0:
                logger.info(f"   📊 Chamadas antes do erro - API: {self._api_calls_count}, Cache: {self._cache_hits_count}")
            return {}, {}

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
        control_id_to_name: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Adiciona coluna 'comparison_group' aos leads baseado nos matches.

        Args:
            leads_df: DataFrame de leads com coluna 'campaign'
            ml_campaign_metadata: Metadata das campanhas ML (com nomes)
            fair_control_map: Mapeamento ML campaign_id → [control campaign_ids]
            control_id_to_name: Mapeamento control campaign_id → campaign_name

        Returns:
            DataFrame com coluna 'comparison_group' adicionada:
            - 'ML': Campanha ML
            - 'Fair Control': Campanha não-ML similar
            - 'Other': Outras campanhas
        """
        df = leads_df.copy()

        # Criar set de nomes COMPLETOS de campanhas de controle justo (incluindo ID)
        fair_control_names = set(control_id_to_name.values())

        # Classificar
        def classify_group(row):
            campaign_name = row.get('campaign', '')

            # Normalizar: remover sufixo |campaign_id se presente
            campaign_name_base = campaign_name
            if '|' in campaign_name:
                parts = campaign_name.rsplit('|', 1)
                if len(parts) == 2 and parts[1].strip().isdigit():
                    campaign_name_base = parts[0].strip()

            # Critério 1: É campanha ML?
            if row.get('ml_type') == 'COM_ML':
                return 'ML'

            # Critério 2: É campanha de controle justo?
            # Check both full name and base name (without ID suffix)
            elif row.get('ml_type') == 'SEM_ML' and (
                campaign_name in fair_control_names or
                campaign_name_base in fair_control_names
            ):
                return 'Fair Control'

            # Outras campanhas
            else:
                return 'Other'

        df['comparison_group'] = df.apply(classify_group, axis=1)

        # Log da distribuição
        logger.info(f"\n   📊 Distribuição de grupos de comparação:")
        for group in ['ML', 'Fair Control', 'Other']:
            count = len(df[df['comparison_group'] == group])
            pct = count / len(df) * 100 if len(df) > 0 else 0
            logger.info(f"      {group}: {count} leads ({pct:.1f}%)")

        return df
