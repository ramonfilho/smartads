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
import requests
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

    def __init__(self, meta_api_integration: MetaAdsIntegration, product_value: float, use_cache: bool = True):
        """
        Args:
            meta_api_integration: Cliente da Meta Ads API
            product_value: Valor do produto em R$
            use_cache: Se True, usa cache em arquivo para evitar chamadas repetidas à API
        """
        self.meta_api = meta_api_integration
        self.product_value = product_value
        self.use_cache = use_cache
        self.cache_dir = Path(__file__).parent.parent.parent / 'files' / 'validation' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, prefix: str, account_id: str, period_start: str, period_end: str) -> str:
        """Gera chave única para cache baseado nos parâmetros"""
        import hashlib
        key = f"{prefix}_{account_id}_{period_start}_{period_end}"
        return hashlib.md5(key.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Dict:
        """Carrega dados do cache se existir"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            import json
            with open(cache_file, 'r') as f:
                data = json.load(f)
                logger.info(f"   💾 Cache HIT: {cache_file.name}")
                return data
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Salva dados no cache"""
        import json
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"   💾 Cache SAVED: {cache_file.name}")

    def _get_campaign_leads_from_meta(
        self,
        account_id: str,
        period_start: str,
        period_end: str
    ) -> Dict[str, Dict[str, int]]:
        """
        Busca eventos 'lead' e eventos personalizados por campanha via Meta API.

        Eventos buscados:
        - 'lead': Total de cadastros (padrão)
        - 'offsite_conversion.fb_pixel_custom.LeadQualified': Leads qualificados
        - 'offsite_conversion.fb_pixel_custom.LeadQualifiedHighQuality': Leads alta qualidade
        - 'offsite_conversion.fb_pixel_custom.Faixa A': Leads Faixa A

        Args:
            account_id: ID da conta (formato: act_XXXXXXXXX) ou lista de IDs separados por vírgula
            period_start: Data início (formato YYYY-MM-DD)
            period_end: Data fim (formato YYYY-MM-DD)

        Returns:
            Dict mapeando campaign_id → {evento: contagem}
            Ex: {'120220370119870390': {'lead': 289, 'LeadQualified': 150, 'LeadQualifiedHighQuality': 80}}
        """
        logger.info("   🔍 Buscando eventos 'lead' da Meta API...")

        # Tentar carregar do cache primeiro
        if self.use_cache:
            cache_key = self._get_cache_key('leads', account_id, period_start, period_end)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        # Estrutura: {campaign_id: {evento: contagem}}
        campaign_events = {}

        # Eventos personalizados para buscar
        CUSTOM_EVENTS = [
            'offsite_conversion.fb_pixel_custom.LeadQualified',
            'offsite_conversion.fb_pixel_custom.LeadQualifiedHighQuality',
            'offsite_conversion.fb_pixel_custom.Faixa A'
        ]

        # Suportar múltiplas contas
        if isinstance(account_id, str):
            account_ids = [acc.strip() for acc in account_id.split(',')]
        else:
            account_ids = [account_id]

        for acc_id in account_ids:
            # Buscar no nível 'adset' para obter eventos personalizados separadamente
            fields = ['campaign_id', 'campaign_name', 'adset_id', 'adset_name', 'actions', 'action_values', 'impressions']

            try:
                logger.info(f"   🔍 Buscando no nível 'adset' para capturar eventos por adset...")
                logger.info(f"   🎯 Usando janela de atribuição padrão da conta (não especificada na API)")
                # Usar action_breakdowns para separar eventos individuais
                insights = self.meta_api.get_insights(
                    account_id=acc_id,
                    level='adset',
                    fields=fields,
                    since_date=period_start,
                    until_date=period_end,
                    action_breakdowns=['action_type']
                )

                # Log primeiros insights com datas para debug
                if len(insights) > 0:
                    logger.info(f"   📅 DEBUG - Primeiros 3 insights com datas:")
                    for i, insight in enumerate(insights[:3]):
                        date_start = insight.get('date_start', 'N/A')
                        date_stop = insight.get('date_stop', 'N/A')
                        camp_id = insight.get('campaign_id', 'N/A')[:15]
                        logger.info(f"      Insight #{i+1}: Campaign {camp_id}..., Período: {date_start} a {date_stop}")

                # Processar actions para extrair eventos
                for adset_data in insights:
                    campaign_id = adset_data.get('campaign_id')
                    campaign_name = adset_data.get('campaign_name', '')
                    adset_id = adset_data.get('adset_id')
                    adset_name = adset_data.get('adset_name', 'Unknown')
                    actions = adset_data.get('actions', [])

                    # Log date_start e date_stop para debug
                    date_start = adset_data.get('date_start')
                    date_stop = adset_data.get('date_stop')

                    # Inicializar dicionário de eventos para esta campanha
                    if campaign_id not in campaign_events:
                        campaign_events[campaign_id] = {}

                    # DEBUG: Log all custom event actions for ML campaign
                    if '120234062599950534' in campaign_id and actions:
                        logger.info(f"   📋 DEBUG - Adset {adset_id[:15]}... ({adset_name[:50]}):")
                        for action in actions:
                            action_type = action.get('action_type')
                            value = action.get('value', 0)
                            if 'custom' in action_type or 'Lead' in action_type:
                                logger.info(f"      {action_type}: {value}")

                    # Buscar todos os eventos (lead + custom events)
                    for action in actions:
                        action_type = action.get('action_type')
                        value = int(action.get('value', 0))

                        # Evento 'lead' padrão
                        if action_type == 'lead':
                            if 'lead' in campaign_events[campaign_id]:
                                campaign_events[campaign_id]['lead'] += value
                            else:
                                campaign_events[campaign_id]['lead'] = value

                        # Eventos personalizados genéricos - precisam ser mapeados
                        elif action_type == 'offsite_conversion.fb_pixel_custom' and adset_id:
                            # Meta API retorna evento genérico, buscar o custom_event deste adset
                            # Log para debug: adset_id e adset_name
                            adset_name = adset_data.get('adset_name', 'Unknown')

                            try:
                                # Buscar promoted_object do adset específico
                                adset_url = f"{self.meta_api.base_url}/{adset_id}"
                                adset_response = requests.get(adset_url, params={
                                    'access_token': self.meta_api.access_token,
                                    'fields': 'promoted_object'
                                }, timeout=3)

                                if adset_response.status_code == 200:
                                    adset_info = adset_response.json()
                                    promoted_obj = adset_info.get('promoted_object', {})
                                    custom_event = promoted_obj.get('custom_event_str')

                                    if custom_event and custom_event in ['LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A']:
                                        # DEBUG: Log detalhado com nome do adset e impressions
                                        impressions = adset_data.get('impressions', 0)
                                        logger.info(f"   🎯 Camp {campaign_id[:15]}..., Adset {adset_id[:15]}... ({adset_name[:50]}): {custom_event} = {value}, Impressions = {impressions}")

                                        if custom_event in campaign_events[campaign_id]:
                                            campaign_events[campaign_id][custom_event] += value
                                        else:
                                            campaign_events[campaign_id][custom_event] = value
                            except Exception as e:
                                logger.debug(f"      Erro ao buscar custom_event do adset {adset_id}: {e}")

            except Exception as e:
                logger.error(f"   ❌ Erro ao buscar leads da conta {acc_id}: {e}")
                continue

        # Estatísticas de resumo
        total_leads = sum(events.get('lead', 0) for events in campaign_events.values())
        total_lead_qualified = sum(events.get('LeadQualified', 0) for events in campaign_events.values())
        total_lead_qualified_hq = sum(events.get('LeadQualifiedHighQuality', 0) for events in campaign_events.values())
        total_faixa_a = sum(events.get('Faixa A', 0) for events in campaign_events.values())

        logger.info(f"   ✅ {len(campaign_events)} campanhas encontradas")
        logger.info(f"      • Leads: {total_leads}")
        logger.info(f"      • LeadQualified: {total_lead_qualified}")
        logger.info(f"      • LeadQualifiedHighQuality: {total_lead_qualified_hq}")
        logger.info(f"      • Faixa A: {total_faixa_a}")

        # DEBUG: Mostrar detalhes das campanhas com eventos personalizados
        logger.info(f"   📋 DEBUG - Campanhas com eventos personalizados:")
        for campaign_id, events in campaign_events.items():
            if any(e in events for e in ['LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A']):
                # Buscar nome da campanha
                try:
                    camp_url = f"{self.meta_api.base_url}/{campaign_id}"
                    camp_response = requests.get(camp_url, params={
                        'access_token': self.meta_api.access_token,
                        'fields': 'name'
                    }, timeout=2)
                    if camp_response.status_code == 200:
                        camp_name = camp_response.json().get('name', 'Unknown')[:60]
                        logger.info(f"      • {campaign_id}: {camp_name}")
                        logger.info(f"        {events}")
                    else:
                        logger.info(f"      • {campaign_id}: {events}")
                except:
                    logger.info(f"      • {campaign_id}: {events}")

        # Salvar no cache para próxima execução
        if self.use_cache:
            cache_key = self._get_cache_key('leads', account_id, period_start, period_end)
            self._save_to_cache(cache_key, campaign_events)

        return campaign_events

    def _get_campaign_lead_count(self, campaign_name: str, campaign_events: Dict[str, Dict[str, int]]) -> int:
        """
        Busca o lead count de uma campanha específica no dicionário retornado pela Meta API.

        Args:
            campaign_name: Nome da campanha (pode incluir |ID no final)
            campaign_events: Dicionário {campaign_id: {evento: contagem}}

        Returns:
            Número de leads (int)
        """
        if not campaign_events:
            return 0

        # Extrair Campaign ID do nome
        campaign_id = self._extract_campaign_id(campaign_name)

        if campaign_id and campaign_id in campaign_events:
            return campaign_events[campaign_id].get('lead', 0)

        return 0

    def _get_campaign_custom_event_count(self, campaign_name: str, campaign_events: Dict[str, Dict[str, int]], event_name: str) -> int:
        """
        Busca a contagem de um evento personalizado para uma campanha específica.

        Args:
            campaign_name: Nome da campanha (pode incluir |ID no final)
            campaign_events: Dicionário {campaign_id: {evento: contagem}}
            event_name: Nome do evento (ex: 'LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A')

        Returns:
            Contagem do evento (int)
        """
        if not campaign_events:
            return 0

        # Extrair Campaign ID do nome
        campaign_id = self._extract_campaign_id(campaign_name)

        if campaign_id and campaign_id in campaign_events:
            return campaign_events[campaign_id].get(event_name, 0)

        return 0

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

        # FIX: Extrair campaign_id ANTES do groupby para consolidar variações de nome
        logger.info("   🔧 Extraindo campaign_id para consolidar variações de nome...")
        matched_df['campaign_id_extracted'] = matched_df['campaign'].apply(
            lambda camp: self._extract_campaign_id(camp)
        )

        # DEBUG: Ver campanhas que não conseguiram extrair ID
        campaigns_no_id = matched_df[matched_df['campaign_id_extracted'].isna()]['campaign'].unique()
        if len(campaigns_no_id) > 0:
            logger.warning(f"   ⚠️  {len(campaigns_no_id)} campanhas SEM ID extraído (serão tratadas separadamente):")
            for camp in list(campaigns_no_id)[:3]:  # Mostrar apenas 3
                logger.warning(f"      - {camp[:80]}")

        # Verificar se há variações de nome para o mesmo ID
        from collections import defaultdict
        campaigns_by_id = defaultdict(list)

        for _, row in matched_df[['campaign', 'campaign_id_extracted']].drop_duplicates().iterrows():
            camp_id = row['campaign_id_extracted']
            camp_name = row['campaign']
            if camp_id:
                campaigns_by_id[camp_id].append(camp_name)

        variations_found = {cid: names for cid, names in campaigns_by_id.items() if len(names) > 1}

        if variations_found:
            logger.warning(f"   ⚠️  {len(variations_found)} IDs têm múltiplas variações de nome!")
            logger.warning(f"   Consolidando por campaign_id para evitar duplicação de leads...")
            for camp_id, names in list(variations_found.items())[:5]:
                logger.warning(f"      • ID {camp_id}: {len(names)} variações sendo consolidadas")
                for name in names:
                    count = len(matched_df[matched_df['campaign'] == name])
                    logger.warning(f"         - {count} respostas: {name[:70]}")

        # Usar o nome mais completo (maior comprimento) para cada campaign_id
        def get_best_campaign_name(camp_id):
            """Retorna o nome mais completo (maior) para um campaign_id"""
            if not camp_id or camp_id not in campaigns_by_id:
                return None
            names = campaigns_by_id[camp_id]
            return max(names, key=len) if names else None

        # Substituir nomes por versão consolidada
        matched_df['campaign_consolidated'] = matched_df['campaign_id_extracted'].apply(
            lambda cid: get_best_campaign_name(cid) if cid else None
        )

        # Se não conseguiu extrair ID, manter nome original
        matched_df['campaign_consolidated'] = matched_df['campaign_consolidated'].fillna(matched_df['campaign'])

        # Groupby usando nome consolidado
        campaign_stats = matched_df.groupby(['ml_type', 'campaign_consolidated']).agg({
            'email': 'count',  # respostas na pesquisa (leads que responderam)
            'converted': 'sum',  # conversions
            'sale_value': 'sum'  # revenue total
        }).reset_index()

        # Renomear coluna de volta para 'campaign'
        campaign_stats.columns = ['ml_type', 'campaign', 'respostas_pesquisa', 'conversions', 'total_revenue']

        # Limpar colunas auxiliares do matched_df
        matched_df = matched_df.drop(['campaign_id_extracted', 'campaign_consolidated'], axis=1)

        # Calcular taxa de conversão (baseada em respostas da pesquisa)
        campaign_stats['conversion_rate'] = (
            campaign_stats['conversions'] / campaign_stats['respostas_pesquisa'] * 100
        ).round(2)

        # Se sale_value não estava disponível, calcular receita baseado em product_value
        if campaign_stats['total_revenue'].sum() == 0:
            campaign_stats['total_revenue'] = campaign_stats['conversions'] * self.product_value

        logger.info(f"   {len(campaign_stats)} campanhas agregadas")

        # 1.5. Buscar leads e eventos personalizados da Meta API
        logger.info("   Buscando eventos 'lead' e eventos personalizados da Meta API...")
        try:
            campaign_events_meta = self._get_campaign_leads_from_meta(
                account_id=account_id,
                period_start=period_start,
                period_end=period_end
            )
        except Exception as e:
            logger.error(f"   ❌ Erro ao buscar leads da Meta: {e}")
            campaign_events_meta = {}

        # Mapear leads da Meta para campanhas
        campaign_stats['leads'] = campaign_stats['campaign'].apply(
            lambda camp: self._get_campaign_lead_count(camp, campaign_events_meta)
        )

        # Mapear eventos personalizados para campanhas
        campaign_stats['LeadQualified'] = campaign_stats['campaign'].apply(
            lambda camp: self._get_campaign_custom_event_count(camp, campaign_events_meta, 'LeadQualified')
        )
        campaign_stats['LeadQualifiedHighQuality'] = campaign_stats['campaign'].apply(
            lambda camp: self._get_campaign_custom_event_count(camp, campaign_events_meta, 'LeadQualifiedHighQuality')
        )
        campaign_stats['Faixa A'] = campaign_stats['campaign'].apply(
            lambda camp: self._get_campaign_custom_event_count(camp, campaign_events_meta, 'Faixa A')
        )

        # DEBUG: Verificar campanha específica 120224064762600390
        debug_campaign_id = '120224064762600390'
        debug_campaigns = campaign_stats[campaign_stats['campaign'].str.contains(debug_campaign_id, na=False)]
        if len(debug_campaigns) > 0:
            logger.info(f"   🔍 DEBUG - Campanha {debug_campaign_id}:")
            for _, row in debug_campaigns.iterrows():
                logger.info(f"      Nome: {row['campaign'][:70]}")
                logger.info(f"      Respostas: {row['respostas_pesquisa']}")
                logger.info(f"      Leads (Meta): {row['leads']}")
        else:
            logger.warning(f"   ⚠️  Campanha {debug_campaign_id} NÃO encontrada no campaign_stats")
            logger.warning(f"   Isso significa que não há respostas da pesquisa para esta campanha.")
            # Verificar se está nos lead_counts_meta
            if debug_campaign_id in lead_counts_meta:
                logger.warning(f"   Mas a Meta API retornou {lead_counts_meta[debug_campaign_id]} leads para ela!")

        # Detectar duplicação: múltiplas campanhas com o mesmo ID
        campaign_stats['campaign_id_extracted'] = campaign_stats['campaign'].apply(
            lambda camp: self._extract_campaign_id(camp)
        )

        # Verificar se há IDs duplicados
        id_counts = campaign_stats['campaign_id_extracted'].value_counts()
        duplicated_ids = id_counts[id_counts > 1]

        if len(duplicated_ids) > 0:
            logger.warning(f"   ⚠️  ATENÇÃO: {len(duplicated_ids)} IDs de campanha aparecem múltiplas vezes!")
            logger.warning(f"   Isso causa DUPLICAÇÃO de leads. IDs afetados:")
            for camp_id, count in duplicated_ids.items():
                if camp_id:  # Ignorar None
                    campaigns_with_id = campaign_stats[campaign_stats['campaign_id_extracted'] == camp_id]['campaign'].tolist()
                    leads_for_id = lead_counts_meta.get(camp_id, 0)
                    total_leads_duplicated = leads_for_id * count
                    logger.warning(f"      • ID {camp_id}: aparece {count}x, {leads_for_id} leads cada = {total_leads_duplicated} total")
                    for camp_name in campaigns_with_id:
                        logger.warning(f"         - {camp_name[:80]}")

        # Remover coluna auxiliar
        campaign_stats = campaign_stats.drop('campaign_id_extracted', axis=1)

        # Calcular taxa de resposta: (respostas_pesquisa / leads) * 100
        campaign_stats['taxa_resposta'] = campaign_stats.apply(
            lambda row: (row['respostas_pesquisa'] / row['leads'] * 100) if row['leads'] > 0 else 0,
            axis=1
        ).round(2)

        # IMPORTANTE: Salvar total de leads ANTES de filtrar campanhas com spend=0
        # Isso garante que o relatório mostre o total correto de leads da Meta
        self.total_leads_meta_before_filter = campaign_stats['leads'].sum()

        logger.info(f"   ✅ Total de leads (Meta): {self.total_leads_meta_before_filter}")
        logger.info(f"   ✅ Total de respostas: {campaign_stats['respostas_pesquisa'].sum()}")
        taxa_media = campaign_stats['respostas_pesquisa'].sum() / campaign_stats['leads'].sum() * 100 if campaign_stats['leads'].sum() > 0 else 0
        logger.info(f"   ✅ Taxa de resposta média: {taxa_media:.2f}%")

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

            # Adicionar optimization_goals (eventos de conversão customizados)
            campaign_stats['optimization_goal'] = campaign_stats['campaign'].apply(
                lambda camp: self._get_campaign_optimization_goals(camp, costs_hierarchy)
            )

            total_spend = campaign_stats['spend'].sum()
            logger.info(f"   ✅ Custos obtidos: R$ {total_spend:,.2f}")
        else:
            campaign_stats['spend'] = 0.0
            campaign_stats['budget'] = 0.0
            campaign_stats['num_creatives'] = 0
            campaign_stats['optimization_goal'] = "-"

        # 2.5. Filtrar campanhas sem spend (não ativas no período)
        # IMPORTANTE: Só remover se temos leads=0 E spend=0
        # Se temos leads mas spend=0, pode ser erro na Meta API - manter campanha
        campaigns_before_filter = len(campaign_stats)
        campaign_stats = campaign_stats[
            (campaign_stats['spend'] > 0) | (campaign_stats['leads'] > 0)
        ]
        campaigns_filtered = campaigns_before_filter - len(campaign_stats)

        if campaigns_filtered > 0:
            logger.info(f"   ⚠️ {campaigns_filtered} campanhas removidas (spend = 0 E leads = 0, não ativas no período)")

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
        Extrai o Campaign ID do nome da campanha.

        Formato esperado: "NOME DA CAMPANHA|CAMPAIGN_ID"
        Exemplo: "DEVLF | CAP | FRIO | FASE 01 | ABERTO ADV+ | PG2 | 2025-04-15|120220370119870390"

        MELHORIA: Agora busca o ID em qualquer parte do nome (não apenas no final)
        para lidar com IDs truncados no Google Sheets.

        Args:
            campaign_name: Nome completo da campanha com ID

        Returns:
            Campaign ID (string) ou None se não encontrar
        """
        if not campaign_name or pd.isna(campaign_name):
            return None

        import re

        # Tentar extrair o ID do final primeiro (método padrão)
        parts = str(campaign_name).split('|')
        if len(parts) >= 2:
            last_part = parts[-1].strip()
            if last_part.isdigit() and len(last_part) > 10:  # IDs do Meta têm ~18 dígitos
                return last_part

        # Fallback: Buscar sequência de 15+ dígitos em QUALQUER lugar do nome
        # IDs do Meta têm 18 dígitos geralmente, mas aceitar 15+ para IDs truncados
        match = re.search(r'1\d{14,}', str(campaign_name))  # Começa com '1' e tem 15+ dígitos
        if match:
            return match.group(0)

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
            # Retornar apenas daily_budget (não usar lifetime_budget pois é total, não diário)
            daily = float(camp_data.get('daily_budget', 0) or 0)
            campaign_budget = daily

            # Se budget da campanha é 0, pode ser ABO - somar budgets dos adsets
            if campaign_budget == 0 and not camp_data.get('has_campaign_budget', False):
                adsets = camp_data.get('adsets', {})
                if adsets and self.meta_api:
                    # Buscar budget dos adsets via Meta API (apenas daily_budget)
                    total_adset_budget = 0.0
                    for adset_id in adsets.keys():
                        try:
                            budget_info = self.meta_api.get_adset_budget_info(adset_id)
                            adset_daily = float(budget_info.get('daily_budget', 0) or 0)
                            # Somar apenas daily_budget (não usar lifetime_budget)
                            total_adset_budget += adset_daily
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

    def _get_campaign_optimization_goals(self, campaign_name: str, costs_hierarchy: Dict) -> str:
        """
        Busca os optimization_goals dos adsets de uma campanha específica.
        Como uma campanha pode ter múltiplos adsets com diferentes goals, retorna todos únicos.

        NOVO: Usa Campaign ID extraído do nome para matching preciso.

        Args:
            campaign_name: Nome da campanha (pode incluir |ID no final)
            costs_hierarchy: Dicionário retornado por get_costs_hierarchy()

        Returns:
            String com optimization_goals únicos separados por vírgula
            Ex: "LeadQualifiedHighQuality", "LeadQualified, LEAD", etc.
            Retorna "-" se não encontrar nenhum goal
        """
        if not costs_hierarchy:
            return "-"

        campaigns = costs_hierarchy.get('campaigns', {})
        if not campaigns:
            return "-"

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
            # Coletar optimization_goals únicos de todos os adsets
            adsets = camp_data.get('adsets', {})
            optimization_goals = set()

            for adset_id, adset_data in adsets.items():
                goal = adset_data.get('optimization_goal')
                if goal:
                    # Mapear OFFSITE_CONVERSIONS para "Lead" quando não for evento personalizado
                    if goal == 'OFFSITE_CONVERSIONS':
                        goal = 'Lead'
                    optimization_goals.add(goal)

            if optimization_goals:
                # Ordenar para consistência e retornar como string
                return ", ".join(sorted(optimization_goals))

        return "-"


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
                'revenue': 0,
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
    product_value: float = 2000.0,
    excluded_leads: int = 0,
    campaign_calc: 'CampaignMetricsCalculator' = None,
    lead_source_stats: Dict = None
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

    # Calcular total de leads da Meta (soma de todas as campanhas)
    # IMPORTANTE: Usar valor salvo ANTES de filtrar campanhas com spend=0
    # Isso garante que o total esteja correto mesmo se campanhas foram removidas
    total_leads_meta = 0

    # Tentar obter do CampaignMetricsCalculator (salvo antes do filtro)
    if campaign_calc and hasattr(campaign_calc, 'total_leads_meta_before_filter'):
        total_leads_meta = campaign_calc.total_leads_meta_before_filter
        logger.info(f"   📊 Usando total de leads salvo antes do filtro: {total_leads_meta}")
    # Fallback: usar campaign_metrics (pode estar incorreto se campanhas foram filtradas)
    elif 'leads' in campaign_metrics.columns and not campaign_metrics.empty:
        total_leads_meta = campaign_metrics['leads'].sum()
    else:
        logger.warning("⚠️ 'leads' não encontrado ou campaign_metrics vazio")

    # Métricas gerais (baseadas em TODAS as vendas)
    conversion_rate = (total_conversions / total_leads * 100) if total_leads > 0 else 0
    roas = (total_revenue / total_spend) if total_spend > 0 else 0
    margin = total_revenue - total_spend

    result = {
        'total_leads_meta': int(total_leads_meta),  # Leads da Meta (cadastros)
        'total_leads': total_leads,  # Respostas da pesquisa (apenas com UTM Meta válida)
        'total_leads_including_excluded': total_leads + excluded_leads,  # TODAS as respostas (incluindo sem UTM)
        'excluded_leads': excluded_leads,  # Leads sem UTM válida (excluídos)
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

    # Adicionar estatísticas de fonte de leads se fornecidas
    if lead_source_stats:
        result['survey_leads'] = lead_source_stats.get('survey_leads', 0)
        result['capi_leads_total'] = lead_source_stats.get('capi_leads_total', 0)
        result['capi_leads_extras'] = lead_source_stats.get('capi_leads_extras', 0)

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
