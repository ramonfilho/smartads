"""
Módulo para carregar dados dos relatórios Excel exportados do Meta Ads.

Lê os arquivos de campanhas, conjuntos de anúncios e anúncios exportados
manualmente do Meta Ads Manager.
"""

import os
import sys
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from glob import glob

logger = logging.getLogger(__name__)


def normalize_unicode(text: str) -> str:
    """
    Normaliza texto Unicode para NFC (composed form).

    Isso resolve problemas onde "ú" pode estar como:
    - NFC: "ú" (1 caractere)
    - NFD: "u" + "´" (2 caracteres: base + combining accent)
    """
    return unicodedata.normalize('NFC', text)


class MetaReportsLoader:
    """
    Carrega dados dos relatórios Excel do Meta Ads.

    Estrutura esperada dos arquivos:
    - Ads---[Conta]-Campanhas-[período].xlsx
    - Ads---[Conta]-Conjuntos-de-anúncios-[período].xlsx
    - Ads---[Conta]-Anúncios-[período].xlsx
    """

    def __init__(self, reports_dir: str):
        """
        Inicializa o loader.

        Args:
            reports_dir: Diretório contendo os relatórios Excel
        """
        self.reports_dir = Path(reports_dir)
        if not self.reports_dir.exists():
            raise FileNotFoundError(f"Diretório de relatórios não encontrado: {reports_dir}")

    def load_all_reports(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Carrega todos os relatórios (campanhas, adsets, ads) de todas as contas.

        Args:
            start_date: Data início (YYYY-MM-DD)
            end_date: Data fim (YYYY-MM-DD)

        Returns:
            Dict com DataFrames:
            - 'campaigns': DataFrame consolidado de campanhas
            - 'adsets': DataFrame consolidado de adsets
            - 'ads': DataFrame consolidado de ads
        """
        logger.info(f"📂 Carregando relatórios Meta de {self.reports_dir}...")

        # Buscar arquivos por padrão (recursivamente em subpastas)
        # Usar listagem manual para evitar problemas com Unicode em glob
        all_files = list(self.reports_dir.rglob('*.xlsx'))

        # Normalizar nomes de arquivos para resolver problemas de encoding Unicode
        campaign_files = [f for f in all_files
                         if 'Campanhas' in normalize_unicode(f.name) or 'campanhas' in normalize_unicode(f.name)]

        adset_files = [f for f in all_files
                      if 'Conjuntos' in normalize_unicode(f.name)
                      and ('anúncios' in normalize_unicode(f.name) or 'anuncios' in normalize_unicode(f.name))]

        # Para ads: deve ter "Anúncios" MAS NÃO "Conjuntos"
        ad_files = [f for f in all_files
                   if ('Anúncios' in normalize_unicode(f.name) or 'Anuncios' in normalize_unicode(f.name))
                   and 'Conjuntos' not in normalize_unicode(f.name)]

        logger.info(f"   Campanhas: {len(campaign_files)} arquivo(s)")
        logger.info(f"   Adsets: {len(adset_files)} arquivo(s)")
        logger.info(f"   Anúncios: {len(ad_files)} arquivo(s)")

        # Carregar e consolidar
        campaigns_df = self._load_and_consolidate(campaign_files, 'campaign')
        adsets_df = self._load_and_consolidate(adset_files, 'adset')
        ads_df = self._load_and_consolidate(ad_files, 'ad')

        return {
            'campaigns': campaigns_df,
            'adsets': adsets_df,
            'ads': ads_df
        }

    def _load_and_consolidate(
        self,
        file_paths: List[Path],
        report_type: str
    ) -> pd.DataFrame:
        """
        Carrega e consolida múltiplos arquivos Excel.

        Args:
            file_paths: Lista de caminhos dos arquivos
            report_type: 'campaign', 'adset' ou 'ad'

        Returns:
            DataFrame consolidado
        """
        if not file_paths:
            logger.warning(f"   ⚠️  Nenhum arquivo encontrado para {report_type}")
            return pd.DataFrame()

        all_dfs = []

        for file_path in file_paths:
            try:
                # Detectar conta pelo nome do arquivo
                account_name = self._extract_account_name(file_path.name)

                # Ler Excel
                df = pd.read_excel(file_path)

                # Adicionar identificação da conta
                df['_source_file'] = file_path.name
                df['_account_name'] = account_name

                # Normalizar nomes de colunas
                df = self._normalize_column_names(df, report_type)

                all_dfs.append(df)

                logger.info(f"      ✅ {file_path.name}: {len(df)} linhas")

            except Exception as e:
                logger.error(f"      ❌ Erro ao ler {file_path.name}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        # Consolidar todos os DataFrames
        consolidated = pd.concat(all_dfs, ignore_index=True)

        logger.info(f"   ✅ Total consolidado: {len(consolidated)} linhas de {report_type}")

        return consolidated

    def _extract_account_name(self, filename: str) -> str:
        """
        Extrai nome da conta do nome do arquivo.

        Args:
            filename: Nome do arquivo (ex: "Ads---Rodolfo-Mori-Campanhas-...")

        Returns:
            Nome da conta (ex: "Rodolfo Mori")
        """
        # Padrão: Ads---[Nome-da-Conta]-[Tipo]-[Período]
        parts = filename.split('---')
        if len(parts) >= 2:
            # Pegar segunda parte e remover tipo do relatório
            account_part = parts[1].split('-Campanhas-')[0]
            account_part = account_part.split('-Conjuntos-')[0]
            account_part = account_part.split('-Anúncios-')[0]
            return account_part.replace('-', ' ')
        return 'Unknown'

    def _normalize_column_names(self, df: pd.DataFrame, report_type: str) -> pd.DataFrame:
        """
        Normaliza nomes de colunas para padrão consistente.

        Args:
            df: DataFrame original
            report_type: 'campaign', 'adset' ou 'ad'

        Returns:
            DataFrame com colunas normalizadas
        """
        # Mapeamento de nomes do Excel → nomes padronizados
        column_mapping = {
            # Comum a todos
            'Nome da campanha': 'campaign_name',
            'Valor usado (BRL)': 'spend',
            'Resultados': 'results',
            'Indicador de resultados': 'optimization_goal',
            'Identificação da campanha': 'campaign_id',

            # Budget - campanhas e adsets
            'Orçamento da campanha': 'budget',  # CBO (Campaign Budget Optimization)
            'Orçamento do conjunto de anúncios': 'budget',  # ABO (Ad Set Budget Optimization)
            'Tipo de orçamento do conjunto de anúncios': 'budget_type',
            'Tipo de orçamento da campanha': 'budget_type',

            # Adsets
            'Nome do conjunto de anúncios': 'adset_name',
            'Identificação do conjunto de anúncios': 'adset_id',

            # Anúncios
            'Nome do anúncio': 'ad_name',
            'Identificação do anúncio': 'ad_id',

            # Eventos personalizados (colunas diretas nos novos relatórios)
            'Leads': 'leads_standard',
            'LeadQualified': 'lead_qualified',
            'LeadQualifiedHighQuality': 'lead_qualified_hq',
            'Faixa A': 'faixa_a',

            # Indicador de resultados (objetivo de otimização) - IMPORTANTE para classificação ML
            'Indicador de resultados': 'optimization_goal_indicator',
        }

        df = df.rename(columns=column_mapping)

        # Converter spend para numérico
        if 'spend' in df.columns:
            df['spend'] = pd.to_numeric(df['spend'].astype(str).str.replace(',', ''), errors='coerce')

        # Converter results para numérico
        if 'results' in df.columns:
            df['results'] = pd.to_numeric(df['results'], errors='coerce')

        # Converter budget para numérico
        if 'budget' in df.columns:
            df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

        # Converter IDs para string (para evitar notação científica)
        for id_col in ['campaign_id', 'adset_id', 'ad_id']:
            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str).str.replace('.0', '', regex=False)

        # Converter colunas de eventos para numérico
        for event_col in ['leads_standard', 'lead_qualified', 'lead_qualified_hq', 'faixa_a']:
            if event_col in df.columns:
                df[event_col] = pd.to_numeric(df[event_col], errors='coerce').fillna(0)

        # Extrair AD code do nome do anúncio (ex: "DEV-AD0033-vid" → "AD0033")
        if report_type == 'ad' and 'ad_name' in df.columns:
            df['ad_code'] = df['ad_name'].str.extract(r'(AD0\d+)', expand=False)

        # Simplificar optimization_goal_indicator (extrair apenas o tipo de evento)
        # Ex: "actions:offsite_conversion.fb_pixel_lead" → "Lead"
        # Ex: "conversions:offsite_conversion.fb_pixel_custom.LeadQualifiedHighQuality" → "LeadQualifiedHighQuality"
        if 'optimization_goal_indicator' in df.columns:
            def simplify_optimization_goal(val):
                if pd.isna(val):
                    return 'Lead'
                val_str = str(val).lower()

                # Verificar eventos customizados CAPI (ordem importa - mais específico primeiro)
                if 'leadqualifiedhighquality' in val_str or 'lead_qualified_high_quality' in val_str:
                    return 'LeadQualifiedHighQuality'
                elif 'leadqualified' in val_str or 'lead_qualified' in val_str:
                    return 'LeadQualified'
                elif 'faixa a' in val_str or 'faixa_a' in val_str:
                    return 'Faixa A'
                elif 'lead' in val_str:
                    return 'Lead'

                return str(val)

            df['optimization_goal'] = df['optimization_goal_indicator'].apply(simplify_optimization_goal)

        return df

    def build_costs_hierarchy(
        self,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Constrói estrutura costs_hierarchy a partir dos relatórios Excel.

        Args:
            start_date: Data início (YYYY-MM-DD)
            end_date: Data fim (YYYY-MM-DD)

        Returns:
            Dict no formato costs_hierarchy esperado pelo metrics_calculator:
            {
                'campaigns': {
                    campaign_id: {
                        'name': campaign_name,
                        'account_id': account_id,
                        'spend': total_spend,
                        'daily_budget': budget,
                        'num_creatives': num_ads,
                        'optimization_goals': set([...])
                    }
                }
            }
        """
        reports = self.load_all_reports(start_date, end_date)
        campaigns_df = reports['campaigns']
        adsets_df = reports['adsets']
        ads_df = reports['ads']

        costs_hierarchy = {'campaigns': {}}

        # NOVO: Criar mapeamento campaign_name → campaign_id dos relatórios de Campanhas
        # (quando disponível - arquivos novos com coluna "Identificação da campanha")
        campaign_name_to_id = {}
        skipped_campaigns = []

        if not campaigns_df.empty and 'campaign_id' in campaigns_df.columns:
            for _, row in campaigns_df.iterrows():
                camp_name = row.get('campaign_name')
                camp_id = row.get('campaign_id')

                # Debug: verificar valores
                if pd.isna(camp_id):
                    skipped_campaigns.append((camp_name, 'ID is NaN'))
                    continue
                if pd.isna(camp_name):
                    skipped_campaigns.append((str(camp_id), 'Name is NaN'))
                    continue

                # Converter ID
                camp_id_str = str(int(camp_id)) if isinstance(camp_id, float) else str(camp_id)
                campaign_name_to_id[camp_name] = camp_id_str

            if campaign_name_to_id:
                logger.info(f"   ✅ {len(campaign_name_to_id)} IDs de campanha carregados dos relatórios de Campanhas")

            if skipped_campaigns:
                logger.warning(f"   ⚠️  {len(skipped_campaigns)} campanhas ignoradas no mapeamento:")
                for name, reason in skipped_campaigns[:5]:
                    logger.warning(f"      • {name[:60]}: {reason}")

        # Validar adsets
        if adsets_df.empty or 'campaign_id' not in adsets_df.columns:
            logger.error("   ❌ Adsets vazios ou sem campaign_id")
            return costs_hierarchy

        # Agrupar por campaign_id extraído dos adsets
        for campaign_id in adsets_df['campaign_id'].dropna().unique():
            if pd.isna(campaign_id) or campaign_id == 'nan':
                continue

            # Buscar adsets desta campanha
            campaign_adsets = adsets_df[adsets_df['campaign_id'] == campaign_id]
            if campaign_adsets.empty:
                continue

            # Nome da campanha vem do primeiro adset
            campaign_name = campaign_adsets.iloc[0]['campaign_name']

            # NOTA: NÃO sobrescrever campaign_id do adset com mapeamento por nome
            # porque pode haver campanhas com mesmo nome mas IDs diferentes.
            # O campaign_id do adset já é correto (vem da coluna "Identificação da campanha")

            # CORREÇÃO: Usar spend dos ADSETS (que tem campaign_id correto)
            # em vez de campaigns_df (que só tem nome e causa duplicação)
            total_spend = campaign_adsets['spend'].sum() if not campaign_adsets.empty else 0.0

            # Buscar ads desta campanha
            campaign_ads = ads_df[ads_df['campaign_id'] == campaign_id] if not ads_df.empty else pd.DataFrame()

            # Agregar budget dos adsets (ABO - Ad Set Budget Optimization)
            budget = campaign_adsets['budget'].sum() if not campaign_adsets.empty else 0.0

            # Se budget dos adsets é 0, pode ser CBO (Campaign Budget Optimization)
            # Buscar budget do nível da campanha
            if budget == 0:
                # Buscar do campaigns_df por nome (único lugar onde budget de campanha CBO pode estar)
                campaign_rows = campaigns_df[campaigns_df['campaign_name'] == campaign_name]
                if not campaign_rows.empty and 'budget' in campaign_rows.columns:
                    campaign_budget = campaign_rows['budget'].iloc[0] if len(campaign_rows) > 0 else 0
                    if campaign_budget > 0:
                        budget = campaign_budget

            # Número de criativos (ads únicos)
            num_creatives = len(campaign_ads) if not campaign_ads.empty else 0

            # Account name do primeiro adset
            account_name = campaign_adsets.iloc[0].get('_account_name', 'Unknown')

            # NOVO: Extrair eventos diretamente das colunas dos adsets
            # Os novos relatórios têm colunas separadas para cada tipo de evento
            total_leads_standard = 0
            lead_qualified = 0
            lead_qualified_hq = 0
            faixa_a = 0

            # Somar eventos de todos os adsets da campanha
            if 'leads_standard' in campaign_adsets.columns:
                total_leads_standard = campaign_adsets['leads_standard'].sum()
            if 'lead_qualified' in campaign_adsets.columns:
                lead_qualified = campaign_adsets['lead_qualified'].sum()
            if 'lead_qualified_hq' in campaign_adsets.columns:
                lead_qualified_hq = campaign_adsets['lead_qualified_hq'].sum()
            if 'faixa_a' in campaign_adsets.columns:
                faixa_a = campaign_adsets['faixa_a'].sum()

            # Total de leads = apenas leads padrão (não somar eventos customizados, pois são subsets)
            # Eventos customizados (LQ, LQHQ, Faixa A) são subconjuntos dos leads, não leads adicionais
            total_leads = total_leads_standard

            # Coletar optimization_goals únicos dos adsets desta campanha
            # Agora usa a coluna "Indicador de resultados" (já simplificada para "optimization_goal")
            optimization_goals = set()
            if 'optimization_goal' in campaign_adsets.columns:
                for goal in campaign_adsets['optimization_goal'].dropna().unique():
                    if goal and goal != 'nan':
                        optimization_goals.add(str(goal))

            # Se não tiver optimization_goal, usar Lead como fallback
            if not optimization_goals:
                optimization_goals.add('Lead')

            # Construir estrutura de adsets para essa campanha
            adsets_dict = {}
            for _, adset_row in campaign_adsets.iterrows():
                adset_id = adset_row.get('adset_id', 'unknown')
                adset_name = adset_row.get('adset_name', 'Unknown')

                # Usar optimization_goal do adset (já simplificado na normalização)
                adset_opt_goal = adset_row.get('optimization_goal', 'Lead')
                if pd.isna(adset_opt_goal) or adset_opt_goal == 'nan':
                    adset_opt_goal = 'Lead'

                adsets_dict[str(adset_id)] = {
                    'name': adset_name,
                    'optimization_goal': adset_opt_goal,
                    'spend': float(adset_row.get('spend', 0)),
                    'budget': float(adset_row.get('budget', 0))
                }

            # Construir entrada
            costs_hierarchy['campaigns'][campaign_id] = {
                'name': campaign_name,
                'account_id': account_name,
                'spend': float(total_spend) if not pd.isna(total_spend) else 0.0,
                'daily_budget': float(budget) if not pd.isna(budget) else 0.0,
                'num_creatives': num_creatives,
                'optimization_goals': optimization_goals,
                'leads': int(total_leads) if not pd.isna(total_leads) else 0,
                'LeadQualified': int(lead_qualified) if not pd.isna(lead_qualified) else 0,
                'LeadQualifiedHighQuality': int(lead_qualified_hq) if not pd.isna(lead_qualified_hq) else 0,
                'Faixa A': int(faixa_a) if not pd.isna(faixa_a) else 0,
                'adsets': adsets_dict  # NOVO: Adicionar adsets individuais
            }

        logger.info(f"   ✅ Costs hierarchy construída: {len(costs_hierarchy['campaigns'])} campanhas")

        # DEBUG: Verificar se leads foram extraídos
        total_leads_extracted = sum(camp.get('leads', 0) for camp in costs_hierarchy['campaigns'].values())
        logger.info(f"   📊 Total de leads extraídos: {total_leads_extracted}")

        # DEBUG: Verificar gasto total
        total_spend_extracted = sum(camp.get('spend', 0) for camp in costs_hierarchy['campaigns'].values())
        logger.info(f"   💰 Gasto total extraído: R$ {total_spend_extracted:,.2f}")

        # DEBUG: Comparar com total nos relatórios
        total_campaigns_in_df = len(campaign_name_to_id) if campaign_name_to_id else 0
        total_adsets_unique_campaigns = len(adsets_df['campaign_id'].dropna().unique()) if not adsets_df.empty else 0
        if total_campaigns_in_df != len(costs_hierarchy['campaigns']):
            logger.warning(f"   ⚠️  Discrepância: {total_campaigns_in_df} IDs nos Campanhas.xlsx, mas apenas {len(costs_hierarchy['campaigns'])} processadas")
            logger.warning(f"   ⚠️  {total_adsets_unique_campaigns} IDs únicos de campanha encontrados nos Adsets")

        # DEBUG DETALHADO: Estatísticas por conta
        logger.info("")
        logger.info("=" * 100)
        logger.info("📊 DEBUG: ESTATÍSTICAS POR CONTA")
        logger.info("=" * 100)

        # Agrupar campanhas por account
        from collections import defaultdict
        stats_by_account = defaultdict(lambda: {'campaigns': 0, 'adsets': 0, 'ads': 0, 'spend': 0.0})

        # Contar campanhas por conta
        for camp_id, camp_data in costs_hierarchy['campaigns'].items():
            account = camp_data.get('account_id', 'Unknown')
            stats_by_account[account]['campaigns'] += 1
            stats_by_account[account]['spend'] += camp_data.get('spend', 0.0)

        # Contar adsets por conta (usar account_name dos relatórios)
        if not adsets_df.empty and 'account_name' in adsets_df.columns:
            adsets_by_account = adsets_df.groupby('account_name').size().to_dict()
            for account, count in adsets_by_account.items():
                stats_by_account[account]['adsets'] = count

        # Contar ads por conta (usar account_name dos relatórios)
        if not ads_df.empty and 'account_name' in ads_df.columns:
            ads_by_account = ads_df.groupby('account_name').size().to_dict()
            for account, count in ads_by_account.items():
                stats_by_account[account]['ads'] = count

        # Exibir estatísticas
        for account in sorted(stats_by_account.keys()):
            stats = stats_by_account[account]
            logger.info("")
            logger.info(f"🏢 Conta: {account}")
            logger.info(f"   📊 Campanhas: {stats['campaigns']}")
            logger.info(f"   📊 Adsets: {stats['adsets']}")
            logger.info(f"   📊 Ads: {stats['ads']}")
            logger.info(f"   💰 Gasto Total: R$ {stats['spend']:,.2f}")

        logger.info("")
        logger.info("=" * 100)
        logger.info("")

        return costs_hierarchy


    def load_adsets_for_comparison(
        self,
        ml_campaign_ids: List[str],
        control_campaign_ids: List[str]
    ) -> pd.DataFrame:
        """
        Carrega adsets de campanhas ML e Controle para comparação.

        Args:
            ml_campaign_ids: IDs das campanhas ML
            control_campaign_ids: IDs das campanhas controle

        Returns:
            DataFrame com adsets filtrados e marcados (ml_type)
        """
        reports = self.load_all_reports(start_date='2025-11-18', end_date='2025-12-01')
        adsets_df = reports['adsets']

        if adsets_df.empty:
            return pd.DataFrame()

        # Filtrar apenas campanhas relevantes
        all_campaign_ids = ml_campaign_ids + control_campaign_ids
        adsets_filtered = adsets_df[adsets_df['campaign_id'].isin(all_campaign_ids)].copy()

        # Marcar tipo (ML ou Controle)
        adsets_filtered['ml_type'] = adsets_filtered['campaign_id'].apply(
            lambda x: 'COM_ML' if x in ml_campaign_ids else 'SEM_ML'
        )

        return adsets_filtered

    def load_ads_for_comparison(
        self,
        ml_campaign_ids: List[str],
        control_campaign_ids: List[str]
    ) -> pd.DataFrame:
        """
        Carrega ads de campanhas ML e Controle para comparação.

        Args:
            ml_campaign_ids: IDs das campanhas ML
            control_campaign_ids: IDs das campanhas controle

        Returns:
            DataFrame com ads filtrados e marcados (ml_type)
        """
        reports = self.load_all_reports(start_date='2025-11-18', end_date='2025-12-01')
        ads_df = reports['ads']

        if ads_df.empty:
            return pd.DataFrame()

        # Extrair AD code do nome do anúncio
        ads_df['ad_code'] = ads_df['ad_name'].str.extract(r'(AD0\d+)', expand=False)

        # Filtrar apenas campanhas relevantes
        all_campaign_ids = ml_campaign_ids + control_campaign_ids
        ads_filtered = ads_df[ads_df['campaign_id'].isin(all_campaign_ids)].copy()

        # Marcar tipo (ML ou Controle)
        ads_filtered['ml_type'] = ads_filtered['campaign_id'].apply(
            lambda x: 'COM_ML' if x in ml_campaign_ids else 'SEM_ML'
        )

        return ads_filtered
