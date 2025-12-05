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
                budget = float(campaign_data.get('daily_budget', 0) or campaign_data.get('lifetime_budget', 0) or 0)

                # Se budget é 0, tentar buscar dos ad sets (CBO)
                if budget == 0:
                    try:
                        adsets = campaign_obj.get_ad_sets(fields=[
                            AdSet.Field.daily_budget,
                            AdSet.Field.lifetime_budget
                        ])
                        adset_budgets = []
                        for adset in adsets:
                            adset_budget = float(adset.get('daily_budget', 0) or adset.get('lifetime_budget', 0) or 0)
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

        logger.info(f"🔍 Buscando campanhas de controle justo (critérios rigorosos: budget 100% igual, {min_creative_overlap*100:.0f}%+ criativos iguais)...")

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
                    budget = float(campaign_data.get('daily_budget', 0) or campaign_data.get('lifetime_budget', 0) or 0)

                    # Se budget é 0, tentar buscar dos ad sets (CBO)
                    if budget == 0:
                        try:
                            adsets = campaign_obj.get_ad_sets(fields=[
                                AdSet.Field.daily_budget,
                                AdSet.Field.lifetime_budget
                            ])
                            adset_budgets = []
                            for adset in adsets:
                                adset_budget = float(adset.get('daily_budget', 0) or adset.get('lifetime_budget', 0) or 0)
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

                matches = []

                for ctrl_id, ctrl_data in all_campaigns.items():
                    ctrl_spend = ctrl_data['spend']
                    ctrl_budget = ctrl_data.get('budget', 0)
                    ctrl_creatives = set(ctrl_data['creative_ids'])

                    # Critério 1: Budget 100% igual (deve ser exatamente o mesmo)
                    if ml_budget != ctrl_budget:
                        continue

                    # Critério 2: Alta sobreposição de criativos (80%+)
                    if len(ml_creatives) == 0 or len(ctrl_creatives) == 0:
                        continue

                    overlap = len(ml_creatives & ctrl_creatives)
                    overlap_pct_ml = overlap / len(ml_creatives)
                    overlap_pct_ctrl = overlap / len(ctrl_creatives)
                    min_overlap_pct = min(overlap_pct_ml, overlap_pct_ctrl)

                    if min_overlap_pct < min_creative_overlap:
                        continue

                    # Match encontrado!
                    matches.append({
                        'id': ctrl_id,
                        'name': ctrl_data['name'],
                        'spend': ctrl_spend,
                        'budget': ctrl_budget,
                        'creative_overlap': overlap,
                        'creative_overlap_pct': min_overlap_pct * 100,
                    })

                    # Adicionar ao mapeamento id → name
                    control_id_to_name[ctrl_id] = ctrl_data['name']

                # Ordenar por melhor match (maior sobreposição de criativos)
                matches.sort(key=lambda x: -x['creative_overlap'])

                fair_matches[ml_id] = [m['id'] for m in matches]

                if matches:
                    logger.info(f"\n   ✅ {ml_data['name']}:")
                    logger.info(f"      Budget: R$ {ml_budget:.2f} | Spend: R$ {ml_spend:.2f}")
                    logger.info(f"      Encontrados {len(matches)} matches:")
                    for m in matches[:3]:  # Top 3
                        logger.info(f"         • {m['name']}: Budget R$ {m['budget']:.2f}, {m['creative_overlap']} criativos ({m['creative_overlap_pct']:.0f}% sobreposição)")
                else:
                    logger.warning(f"   ⚠️ Nenhum match encontrado para {ml_data['name']}")

            return fair_matches, control_id_to_name

        except Exception as e:
            logger.error(f"❌ Erro ao buscar campanhas de controle: {e}")
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

        # Criar set de nomes de campanhas de controle justo
        fair_control_names = set(control_id_to_name.values())

        # Classificar
        def classify_group(row):
            campaign_name = row.get('campaign', '')

            # Normalizar nome: remover sufixo |campaign_id se presente
            # Formato CSV: "CAMPAIGN_NAME|120224064762600390"
            # Formato API: "CAMPAIGN_NAME"
            campaign_name_clean = campaign_name
            if '|' in campaign_name:
                parts = campaign_name.rsplit('|', 1)
                if len(parts) == 2 and parts[1].strip().isdigit():
                    campaign_name_clean = parts[0].strip()

            # Critério 1: É campanha ML?
            if row.get('ml_type') == 'COM_ML':
                return 'ML'

            # Critério 2: É campanha de controle justo? (match exato por nome)
            elif row.get('ml_type') == 'SEM_ML' and campaign_name_clean in fair_control_names:
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
