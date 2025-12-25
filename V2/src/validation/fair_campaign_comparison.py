"""
Módulo para comparação fair control em nível de ADSETS e ADS.

Compara campanhas ML vs Controle no nível de conjuntos de anúncios e anúncios individuais.
Foco em métricas de negócio: ROAS, CPA, Margem de Contribuição, Taxa de Conversão.

Princípio: "Maçãs com Maçãs"
- Compara MESMOS adsets (mesmo targeting/público)
- Compara MESMOS ads (mesmo criativo)
- Filtro de gasto mínimo: R$ 200 por adset/ad
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


# ============================================================================
# HELPER: CRIAR MAPEAMENTO REFINADO DE CAMPANHAS
# ============================================================================

def create_refined_campaign_map(
    campaigns_df: pd.DataFrame,
    ml_campaign_ids: List[str],
    control_campaign_ids: List[str]
) -> Dict[str, str]:
    """
    Cria mapeamento refinado: campaign_id → comparison_group.

    Distingue entre:
    - 'Eventos ML': Campanhas ML que usam eventos CAPI customizados (LeadQualified/LQHQ)
    - 'Otimização ML': Campanhas ML que NÃO usam eventos customizados
    - 'Controle': Campanhas sem ML

    Args:
        campaigns_df: DataFrame com campanhas (deve ter optimization_goal)
        ml_campaign_ids: Lista de IDs de campanhas ML
        control_campaign_ids: Lista de IDs de campanhas Controle

    Returns:
        Dict mapping campaign_id → comparison_group ('Eventos ML', 'Otimização ML', ou 'Controle')
    """
    refined_map = {}

    # DEBUG: Mostrar colunas disponíveis
    logger.info(f"   🔍 DEBUG create_refined_campaign_map:")
    logger.info(f"      Colunas disponíveis: {list(campaigns_df.columns)[:10]}")
    logger.info(f"      Total de linhas: {len(campaigns_df)}")

    # Criar lookup de campaign_id → optimization_goal
    if 'optimization_goal' in campaigns_df.columns and 'campaign_id' in campaigns_df.columns:
        # Limpar campaign_id (primeiros 15 dígitos)
        campaigns_df['campaign_id_clean'] = campaigns_df['campaign_id'].astype(str).str[:15]

        opt_goal_map = {}
        for idx, row in campaigns_df.iterrows():
            cid = str(row['campaign_id_clean'])
            opt_goal = str(row.get('optimization_goal', ''))
            opt_goal_map[cid] = opt_goal
    else:
        opt_goal_map = {}

    # Classificar campanhas ML
    for cid in ml_campaign_ids:
        cid_clean = str(cid)[:15]
        opt_goal = opt_goal_map.get(cid_clean, '')

        # Verificar se usa eventos customizados CAPI
        uses_custom_events = any(custom in opt_goal for custom in ['LeadQualified', 'LeadQualifiedHighQuality'])

        # DEBUG: Log das campanhas problemáticas
        if '120234062599950' in cid_clean or '120234748179990' in cid_clean:
            logger.info(f"   🔍 DEBUG - Classificando campanha ML:")
            logger.info(f"      ID (15): {cid_clean}")
            logger.info(f"      ID (completo): {cid}")
            logger.info(f"      optimization_goal encontrado: '{opt_goal}'")
            logger.info(f"      uses_custom_events: {uses_custom_events}")
            logger.info(f"      Grupo: {'Eventos ML' if uses_custom_events else 'Otimização ML'}")

        if uses_custom_events:
            refined_map[cid_clean] = 'Eventos ML'  # USAR 15 DÍGITOS como chave
        else:
            refined_map[cid_clean] = 'Otimização ML'  # USAR 15 DÍGITOS como chave

    # Classificar campanhas Controle
    for cid in control_campaign_ids:
        cid_clean = str(cid)[:15]
        refined_map[cid_clean] = 'Controle'  # USAR 15 DÍGITOS como chave

    logger.info(f"   📊 Mapeamento refinado criado:")
    eventos_ml = sum(1 for v in refined_map.values() if v == 'Eventos ML')
    otimiz_ml = sum(1 for v in refined_map.values() if v == 'Otimização ML')
    controle = sum(1 for v in refined_map.values() if v == 'Controle')
    logger.info(f"      Eventos ML: {eventos_ml}, Otimização ML: {otimiz_ml}, Controle: {controle}")

    return refined_map


# ============================================================================
# CONFIGURAÇÃO: ADSETS E ADS MATCHED
# ============================================================================

# Definição dos matched adsets (aparecem em ML E Controle)
# IMPORTANTE: Espaçamento deve ser EXATO como no Excel!
MATCHED_ADSETS = [
    'ABERTO |  AD0022',  # ATENÇÃO: 2 espaços entre | e AD0022
    'ABERTO | AD0027',   # 1 espaço (este está correto)
    'ADV | Linguagem de programação',
    'ADV | Lookalike 1% Cadastrados - DEV 2.0 + Interesse Ciência da Computação',
    'ADV | Lookalike 2% Cadastrados - DEV 2.0 + Interesses',
]

# Definição dos matched ads (aparecem em ML E Controle)
MATCHED_ADS = [
    'AD0013', 'AD0014', 'AD0017', 'AD0018', 'AD0022', 'AD0027', 'AD0033'
]

# Gasto mínimo para incluir adset/ad na comparação (R$)
MIN_SPEND = 200.0


# ============================================================================
# FUNÇÕES DE IDENTIFICAÇÃO DE MATCHED PAIRS
# ============================================================================

def identify_matched_adset_pairs(
    adsets_df: pd.DataFrame,
    ml_campaign_ids: List[str],
    control_campaign_ids: List[str],
    min_spend: float = MIN_SPEND
) -> Tuple[List[str], pd.DataFrame]:
    """
    Identifica adsets que aparecem tanto em campanhas ML quanto controle.

    Args:
        adsets_df: DataFrame com adsets e suas métricas
        ml_campaign_ids: IDs das campanhas ML
        control_campaign_ids: IDs das campanhas controle
        min_spend: Gasto mínimo para incluir adset (default: R$ 200)

    Returns:
        Tuple (matched_adsets, adsets_metrics_df)
        - matched_adsets: Lista de nomes de adsets matched
        - adsets_metrics_df: DataFrame com métricas por adset
    """
    logger.info("🔍 Identificando matched pairs de adsets...")

    # Adsets em campanhas ML - AGREGAR primeiro, DEPOIS filtrar
    ml_adsets_all = adsets_df[adsets_df['campaign_id'].isin(ml_campaign_ids)]
    if not ml_adsets_all.empty:
        # Agregar spend por adset_name
        ml_adsets_agg = ml_adsets_all.groupby('adset_name')['spend'].sum().reset_index()
        # Filtrar por gasto mínimo agregado
        ml_adsets_filtered = ml_adsets_agg[ml_adsets_agg['spend'] >= min_spend]
        ml_adsets = set(ml_adsets_filtered['adset_name'].dropna().unique().tolist())
    else:
        ml_adsets = set()

    # Adsets em campanhas controle - AGREGAR primeiro, DEPOIS filtrar
    ctrl_adsets_all = adsets_df[adsets_df['campaign_id'].isin(control_campaign_ids)]
    if not ctrl_adsets_all.empty:
        # Agregar spend por adset_name
        ctrl_adsets_agg = ctrl_adsets_all.groupby('adset_name')['spend'].sum().reset_index()
        # Filtrar por gasto mínimo agregado
        ctrl_adsets_filtered = ctrl_adsets_agg[ctrl_adsets_agg['spend'] >= min_spend]
        control_adsets = set(ctrl_adsets_filtered['adset_name'].dropna().unique().tolist())
    else:
        control_adsets = set()

    # Matched pairs (interseção)
    matched = list(ml_adsets.intersection(control_adsets))

    # Filtrar apenas adsets na lista MATCHED_ADSETS
    matched_final = [adset for adset in matched if adset in MATCHED_ADSETS]

    logger.info(f"   ✅ {len(matched_final)} adsets matched (de {len(MATCHED_ADSETS)} esperados)")
    logger.info(f"      ML adsets: {len(ml_adsets)}, Controle adsets: {len(control_adsets)}")

    # Criar DataFrame com métricas por adset
    adsets_metrics = adsets_df[adsets_df['adset_name'].isin(matched_final)].copy()

    # Renomear 'leads_standard' para 'leads' (vem do MetaReportsLoader)
    if 'leads_standard' in adsets_metrics.columns:
        adsets_metrics['leads'] = adsets_metrics['leads_standard']

    return matched_final, adsets_metrics


def identify_matched_ad_pairs(
    ads_df: pd.DataFrame,
    ml_campaign_ids: List[str],
    control_campaign_ids: List[str],
    min_spend: float = MIN_SPEND
) -> Tuple[List[str], pd.DataFrame]:
    """
    Identifica anúncios que aparecem tanto em campanhas ML quanto controle.

    Args:
        ads_df: DataFrame com ads e suas métricas
        ml_campaign_ids: IDs das campanhas ML
        control_campaign_ids: IDs das campanhas controle
        min_spend: Gasto mínimo para incluir ad (default: R$ 200)

    Returns:
        Tuple (matched_ads, ads_metrics_df)
        - matched_ads: Lista de AD codes matched
        - ads_metrics_df: DataFrame com métricas por ad
    """
    logger.info("🔍 Identificando matched pairs de anúncios...")

    # Anúncios em campanhas ML - AGREGAR primeiro, DEPOIS filtrar
    ml_ads_all = ads_df[ads_df['campaign_id'].isin(ml_campaign_ids)]
    if not ml_ads_all.empty:
        # Agregar spend por ad_code
        ml_ads_agg = ml_ads_all.groupby('ad_code')['spend'].sum().reset_index()
        # Filtrar por gasto mínimo agregado
        ml_ads_filtered = ml_ads_agg[ml_ads_agg['spend'] >= min_spend]
        ml_ads = set(ml_ads_filtered['ad_code'].dropna().unique().tolist())
    else:
        ml_ads = set()

    # Anúncios em campanhas controle - AGREGAR primeiro, DEPOIS filtrar
    ctrl_ads_all = ads_df[ads_df['campaign_id'].isin(control_campaign_ids)]
    if not ctrl_ads_all.empty:
        # Agregar spend por ad_code
        ctrl_ads_agg = ctrl_ads_all.groupby('ad_code')['spend'].sum().reset_index()
        # Filtrar por gasto mínimo agregado
        ctrl_ads_filtered = ctrl_ads_agg[ctrl_ads_agg['spend'] >= min_spend]
        control_ads = set(ctrl_ads_filtered['ad_code'].dropna().unique().tolist())
    else:
        control_ads = set()

    # Matched pairs (interseção)
    matched = list(ml_ads.intersection(control_ads))

    # Filtrar apenas ads na lista MATCHED_ADS
    matched_final = [ad for ad in matched if ad in MATCHED_ADS]

    logger.info(f"   ✅ {len(matched_final)} anúncios matched (de {len(MATCHED_ADS)} esperados)")
    logger.info(f"      ML ads: {len(ml_ads)}, Controle ads: {len(control_ads)}")

    # Criar DataFrame com métricas por ad
    ads_metrics = ads_df[ads_df['ad_code'].isin(matched_final)].copy()

    return matched_final, ads_metrics


# ============================================================================
# FUNÇÕES DE COMPARAÇÃO DE PERFORMANCE
# ============================================================================

def compare_all_adsets_performance(
    adsets_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    comparison_group_map: Dict[str, str],
    product_value: float = 2000.0,
    min_spend: float = 0.0
) -> pd.DataFrame:
    """
    Compara performance de TODOS os adsets (Eventos ML vs Controle), sem filtrar por matched pairs.

    Args:
        adsets_df: DataFrame com TODOS os adsets do Excel (sem filtro de matched pairs)
        matched_df: DataFrame com conversões matched (leads→vendas)
        comparison_group_map: Mapeamento campaign_id → comparison_group
        product_value: Valor do produto em R$
        min_spend: Gasto mínimo para incluir adset (default: R$ 0)

    Returns:
        DataFrame agregado com métricas por grupo (Eventos ML vs Controle)
    """
    logger.info("📊 Comparando performance de TODOS os adsets (Eventos ML vs Controle)...")

    # Calcular conversões E RECEITA REAL por CAMPAIGN + ADSET usando 'campaign' e 'medium' do matched_df
    if 'converted' in matched_df.columns and 'medium' in matched_df.columns and 'campaign' in matched_df.columns:
        # Conversões e receita por campanha + adset
        converted_leads = matched_df[matched_df['converted'] == True].copy()

        conversions_by_campaign_adset = converted_leads.groupby(
            ['campaign', 'medium']
        ).agg({
            'email': 'nunique',  # Conversões únicas
            'sale_value': 'sum'   # Receita real total
        }).reset_index()

        conversions_by_campaign_adset.columns = ['campaign_name', 'adset_name', 'conversions', 'revenue']

        # Extrair Campaign ID do final do nome
        def extract_campaign_id(campaign_name):
            if pd.isna(campaign_name):
                return None
            parts = str(campaign_name).split('|')
            if len(parts) >= 2:
                campaign_id = parts[-1].strip()
                if campaign_id.isdigit() and len(campaign_id) >= 15:
                    return campaign_id[:15]
            return None

        conversions_by_campaign_adset['campaign_id_from_utm'] = conversions_by_campaign_adset['campaign_name'].apply(extract_campaign_id)

        # Verificar se há IDs extraídos
        total_convs = len(conversions_by_campaign_adset)
        convs_with_id = conversions_by_campaign_adset['campaign_id_from_utm'].notna().sum()
        total_revenue = conversions_by_campaign_adset['revenue'].sum()
        logger.info(f"   ✅ Conversões e receita calculadas por campanha + adset (via 'campaign' + 'medium')")
        logger.info(f"   Total de combinações campanha+adset com conversões: {total_convs}")
        logger.info(f"   Campaign IDs extraídos dos UTMs: {convs_with_id}/{total_convs}")
        logger.info(f"   Receita total real: R$ {total_revenue:,.2f}")
    else:
        conversions_by_campaign_adset = pd.DataFrame(columns=['campaign_name', 'adset_name', 'conversions', 'revenue'])

    # Preparar campaign_id para merge (primeiros 15 dígitos)
    adsets_df['campaign_id_clean'] = adsets_df['campaign_id'].astype(str).str[:15]
    conversions_by_campaign_adset['campaign_id_clean'] = conversions_by_campaign_adset['campaign_id_from_utm'].astype(str).str[:15]

    # Merge conversões com dados do Excel
    adsets_full = adsets_df.merge(
        conversions_by_campaign_adset,
        on=['campaign_id_clean', 'adset_name'],
        how='left',
        suffixes=('', '_conv')
    )

    adsets_full['conversions'] = adsets_full['conversions'].fillna(0)
    adsets_full['revenue'] = adsets_full['revenue'].fillna(0)  # Receita real do matched_df

    # Renomear 'leads_standard' para 'leads' (vem do MetaReportsLoader)
    if 'leads_standard' in adsets_full.columns:
        adsets_full['leads'] = adsets_full['leads_standard']
    elif 'leads' not in adsets_full.columns:
        logger.warning("   ⚠️ Coluna 'leads' não encontrada, usando 0")
        adsets_full['leads'] = 0

    # AJUSTE ESPECIAL: Adsets da campanha edge case que não dispara evento Lead
    # Aplicar mesma lógica de "forjar" leads que foi feita no nível de campanha
    campaign_special_id_prefix = '120234062599950'

    # Verificar se há coluna LeadQualified
    if 'lead_qualified' in adsets_full.columns or 'LeadQualified' in adsets_full.columns:
        lq_col = 'lead_qualified' if 'lead_qualified' in adsets_full.columns else 'LeadQualified'

        # Calcular proporção média LQ/Leads dos adsets normais (excluindo campanha especial)
        adsets_normal = adsets_full[
            ~adsets_full['campaign_id_clean'].astype(str).str.startswith(campaign_special_id_prefix)
        ]

        total_leads_normal = adsets_normal['leads'].sum()
        total_lq_normal = adsets_normal[lq_col].sum()

        if total_leads_normal > 0 and total_lq_normal > 0:
            avg_ratio = total_lq_normal / total_leads_normal

            # Ajustar adsets da campanha especial
            for idx in adsets_full.index:
                camp_id = str(adsets_full.at[idx, 'campaign_id_clean'])
                if camp_id.startswith(campaign_special_id_prefix):
                    lq = adsets_full.at[idx, lq_col]
                    current_leads = adsets_full.at[idx, 'leads']

                    # Se tem LQ mas não tem leads, calcular leads artificiais
                    if pd.notna(lq) and lq > 0 and (pd.isna(current_leads) or current_leads == 0):
                        leads_artificial = int(lq / avg_ratio)
                        adsets_full.at[idx, 'leads'] = leads_artificial
                        logger.info(f"   🔧 Adset especial ajustado: {adsets_full.at[idx, 'adset_name'][:50]}")
                        logger.info(f"      LeadQualified: {lq:.0f} → Leads artificial: {leads_artificial} (proporção {avg_ratio:.2%})")

    # Adicionar comparison_group
    adsets_full['comparison_group'] = adsets_full['campaign_id_clean'].map(comparison_group_map)

    # Filtrar apenas Eventos ML e Controle (remover Otimização ML e outros)
    # IMPORTANTE: Mostrar TODOS os adsets, independente de gasto, leads ou conversões
    # Isso garante que os totais batam com a tabela de Campanhas
    adsets_filtered = adsets_full[adsets_full['comparison_group'].isin(['Eventos ML', 'Controle'])].copy()

    logger.info(f"   📊 Todos os adsets das campanhas Eventos ML + Controle: {len(adsets_filtered)}")

    # Calcular métricas de negócio
    # NOTA: 'revenue' já vem do matched_df com valores reais de venda, não usar product_value fixo
    # adsets_filtered['revenue'] já foi preenchida no merge acima
    adsets_filtered['cpl'] = adsets_filtered['spend'] / adsets_filtered['leads'].replace(0, 1)  # Evitar divisão por zero
    adsets_filtered['roas'] = adsets_filtered['revenue'] / adsets_filtered['spend'].replace(0, 1)  # Evitar divisão por zero
    adsets_filtered['margin'] = adsets_filtered['revenue'] - adsets_filtered['spend']
    adsets_filtered['conversion_rate'] = adsets_filtered['conversions'] / adsets_filtered['leads'].replace(0, 1)  # Evitar divisão por zero

    # Agregar por comparison_group
    aggregated = adsets_filtered.groupby('comparison_group').agg({
        'leads': 'sum',
        'conversions': 'sum',
        'spend': 'sum',
        'revenue': 'sum',
        'margin': 'sum'
    }).reset_index()

    # Recalcular métricas agregadas
    aggregated['conversion_rate'] = aggregated['conversions'] / aggregated['leads']
    aggregated['cpl'] = aggregated['spend'] / aggregated['leads']
    aggregated['roas'] = aggregated['revenue'] / aggregated['spend']

    logger.info(f"   ✅ Comparação completa de adsets calculada")
    logger.info(f"      Eventos ML: {aggregated[aggregated['comparison_group']=='Eventos ML']['conversions'].sum():.0f} conversões")
    logger.info(f"      Controle: {aggregated[aggregated['comparison_group']=='Controle']['conversions'].sum():.0f} conversões")

    return aggregated


def compare_adset_performance(
    adsets_metrics_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    ml_type_map: Dict[str, str],
    product_value: float = 2000.0,
    comparison_group_map: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compara performance de adsets entre ML e Controle.

    Args:
        adsets_metrics_df: DataFrame com métricas de adsets (da Meta API)
        matched_df: DataFrame com conversões matched (leads→vendas)
        ml_type_map: Mapeamento campaign_id → ml_type (DEPRECATED - usar comparison_group_map)
        product_value: Valor do produto em R$
        comparison_group_map: Mapeamento campaign_id → comparison_group ('Eventos ML', 'Otimização ML', 'Controle')

    Returns:
        Dict com DataFrames:
        - 'aggregated': Agregação ML vs Controle
        - 'detailed': Detalhamento adset-a-adset (CADA CAMPANHA SEPARADA)
    """
    logger.info("📊 Comparando performance de adsets...")

    # Adicionar ml_type aos adsets (para compatibilidade)
    adsets_metrics_df['ml_type'] = adsets_metrics_df['campaign_id'].map(ml_type_map)

    # Se temos comparison_group_map, usar ele diretamente (novo comportamento)
    if comparison_group_map:
        adsets_metrics_df['comparison_group_from_campaign'] = adsets_metrics_df['campaign_id'].map(comparison_group_map)

    # NOVO: Calcular conversões E RECEITA REAL por CAMPAIGN + ADSET usando 'campaign' e 'medium' do matched_df
    # A coluna 'medium' contém o NOME DO ADSET que gerou o lead
    # A coluna 'campaign' contém o NOME DA CAMPANHA
    # IMPORTANTE: Precisamos das DUAS para matching preciso!
    if 'converted' in matched_df.columns and 'medium' in matched_df.columns and 'campaign' in matched_df.columns:
        # Conversões e receita real por campanha + adset (usando campaign + medium)
        # CRÍTICO: Contar emails únicos, não agregação de linhas (evita duplicatas)
        converted_leads = matched_df[matched_df['converted'] == True].copy()

        conversions_by_campaign_adset = converted_leads.groupby(
            ['campaign', 'medium']  # campaign + medium = identificação única
        ).agg({
            'email': 'nunique',  # Conversões únicas
            'sale_value': 'sum'   # Receita real total
        }).reset_index()

        conversions_by_campaign_adset.columns = ['campaign_name', 'adset_name', 'conversions', 'revenue']

        # IMPORTANTE: Extrair Campaign ID do final do nome para fazer matching preciso
        # Exemplo: "CAMPAIGN | 2025-04-15|120220370119870390" → ID = "120220370119870390"
        def extract_campaign_id(campaign_name):
            """Extrai o Campaign ID do final do nome da campanha"""
            if pd.isna(campaign_name):
                return None
            parts = str(campaign_name).split('|')
            if len(parts) > 1:
                last_part = parts[-1].strip()
                # Se último elemento é numérico e tem 18+ dígitos, é um Campaign ID
                if last_part.isdigit() and len(last_part) >= 18:
                    return last_part
            return None

        conversions_by_campaign_adset['campaign_id_from_utm'] = conversions_by_campaign_adset['campaign_name'].apply(extract_campaign_id)

        # DEBUG: Verificar quantos IDs foram extraídos
        ids_extracted = conversions_by_campaign_adset['campaign_id_from_utm'].notna().sum()
        logger.info(f"   ✅ Conversões calculadas por campanha + adset (via 'campaign' + 'medium')")
        logger.info(f"   Total de combinações campanha+adset com conversões: {len(conversions_by_campaign_adset)}")
        logger.info(f"   Campaign IDs extraídos dos UTMs: {ids_extracted}/{len(conversions_by_campaign_adset)}")

        if ids_extracted < len(conversions_by_campaign_adset):
            logger.warning(f"   ⚠️ {len(conversions_by_campaign_adset) - ids_extracted} conversões SEM Campaign ID no UTM")
            # Mostrar exemplos
            sem_id = conversions_by_campaign_adset[conversions_by_campaign_adset['campaign_id_from_utm'].isna()]
            for idx, row in sem_id.head(3).iterrows():
                logger.warning(f"      • Campaign: {row['campaign_name'][:70]}")
                logger.warning(f"        Adset: {row['adset_name'][:50]}")
    else:
        conversions_by_campaign_adset = pd.DataFrame(columns=['campaign_name', 'adset_name', 'conversions', 'revenue'])
        if 'medium' not in matched_df.columns:
            logger.warning("   ⚠️ Coluna 'medium' não encontrada em matched_df - conversões não podem ser atribuídas aos adsets!")

    # Merge conversões por campanha + adset
    # IMPORTANTE: Usar Campaign ID + Adset Name para matching preciso
    # (evita ambiguidade quando há múltiplas campanhas com mesmo nome)

    # Preparar campaign_id para merge (primeiros 15 dígitos - parte comum)
    # UTMs têm 18 dígitos, Excel tem 21 (18 + "000"), primeiros 15 são a parte comum
    adsets_metrics_df['campaign_id_clean'] = adsets_metrics_df['campaign_id'].astype(str).str[:15]
    conversions_by_campaign_adset['campaign_id_clean'] = conversions_by_campaign_adset['campaign_id_from_utm'].astype(str).str[:15]

    # DEBUG: Verificar matching antes do merge
    logger.info(f"\n   🔍 DEBUG - Preparando merge:")
    logger.info(f"      IDs únicos no Excel: {adsets_metrics_df['campaign_id_clean'].nunique()}")
    logger.info(f"      IDs únicos nas conversões: {conversions_by_campaign_adset['campaign_id_clean'].nunique()}")

    # Verificar se há algum match possível
    ids_excel = set(adsets_metrics_df['campaign_id_clean'].unique())
    ids_conversions = set(conversions_by_campaign_adset['campaign_id_clean'].dropna().unique())
    matching_ids = ids_excel & ids_conversions
    logger.info(f"      IDs que fazem match: {len(matching_ids)}")

    if len(matching_ids) == 0:
        logger.warning(f"      ⚠️ NENHUM ID FAZ MATCH! Vamos comparar:")
        logger.warning(f"         Excel (primeiros 3): {list(ids_excel)[:3]}")
        logger.warning(f"         Conversões (primeiros 3): {list(ids_conversions)[:3]}")

    # DEBUG: Identificar conversões que NÃO fazem match com Excel
    # Fazer merge inverso: quais conversões não encontram adset no Excel?
    conversions_not_in_excel = conversions_by_campaign_adset.merge(
        adsets_metrics_df[['campaign_id_clean', 'adset_name']].drop_duplicates(),
        on=['campaign_id_clean', 'adset_name'],
        how='left',
        indicator=True
    )
    unmatched_conversions = conversions_not_in_excel[conversions_not_in_excel['_merge'] == 'left_only']

    if len(unmatched_conversions) > 0:
        total_unmatched_convs = unmatched_conversions['conversions'].sum()
        logger.warning(f"\n   ⚠️ CONVERSÕES NÃO ENCONTRADAS NO EXCEL META:")
        logger.warning(f"      Total de adsets não encontrados: {len(unmatched_conversions)}")
        logger.warning(f"      Total de conversões perdidas: {total_unmatched_convs:.0f}")
        logger.warning(f"\n      Detalhes dos adsets não encontrados:")

        # Para cada adset não encontrado, buscar os emails correspondentes em matched_df
        for idx, row in unmatched_conversions.iterrows():
            logger.warning(f"      • Campaign: {row['campaign_name'][:60]}")
            logger.warning(f"        Adset: {row['adset_name'][:50]}")
            logger.warning(f"        Campaign ID (15): {row['campaign_id_clean']}")
            logger.warning(f"        Conversões: {row['conversions']:.0f}")

            # Buscar emails específicos deste adset em matched_df
            if 'campaign' in matched_df.columns and 'medium' in matched_df.columns:
                matching_rows = matched_df[
                    (matched_df['campaign'] == row['campaign_name']) &
                    (matched_df['medium'] == row['adset_name']) &
                    (matched_df['converted'] == True)
                ]
                if len(matching_rows) > 0:
                    emails = matching_rows['email'].unique()
                    logger.warning(f"        Emails: {', '.join(emails[:3])}{' ...' if len(emails) > 3 else ''}")
                    # Verificar comparison_group destes leads
                    if 'comparison_group' in matching_rows.columns:
                        groups = matching_rows['comparison_group'].unique()
                        logger.warning(f"        Grupos: {', '.join(groups)}")
        logger.warning("")

    # MELHORADO: Merge com matching mais flexível de nomes
    # 1. Tentar merge exato primeiro
    adsets_full = adsets_metrics_df.merge(
        conversions_by_campaign_adset,
        on=['campaign_id_clean', 'adset_name'],
        how='left',
        suffixes=('', '_conv')
    )

    # 2. Para conversões que não tiveram match exato, tentar matching flexível
    # (útil quando nomes no UTM são truncados)
    if len(unmatched_conversions) > 0:
        logger.info(f"   🔧 Tentando matching flexível para {len(unmatched_conversions)} conversões não encontradas...")
        matches_found = 0

        # Para cada conversão que não encontrou adset no Excel
        for conv_idx, conv_row in unmatched_conversions.iterrows():
            utm_name = str(conv_row['adset_name']).strip()
            utm_campaign_id = str(conv_row['campaign_id_clean'])

            # Normalizar nome do UTM (remover espaços extras, lowercase)
            utm_name_normalized = ' '.join(utm_name.split()).lower()

            # DEBUG: Log do adset que estamos tentando encontrar
            logger.info(f"      🔍 Buscando adset: '{utm_name}' (campanha {utm_campaign_id})")

            # Procurar adsets da mesma campanha no Excel
            same_campaign_excel = adsets_metrics_df[
                adsets_metrics_df['campaign_id_clean'] == utm_campaign_id
            ]

            logger.info(f"         Encontrados {len(same_campaign_excel)} adsets da mesma campanha no Excel")

            # Tentar matching por substring/similaridade
            for excel_idx, excel_row in same_campaign_excel.iterrows():
                excel_name = str(excel_row['adset_name']).strip()
                excel_name_normalized = ' '.join(excel_name.split()).lower()

                # DEBUG: Comparar nomes
                logger.info(f"         Comparando com: '{excel_name}'")
                logger.info(f"            UTM norm: '{utm_name_normalized}'")
                logger.info(f"            Excel norm: '{excel_name_normalized}'")
                logger.info(f"            Exato match: {utm_name_normalized == excel_name_normalized}")

                # Estratégia 1: Match exato (redundante mas útil para debug)
                if utm_name_normalized == excel_name_normalized:
                    # Encontrar a linha correspondente em adsets_full
                    match_mask = (
                        (adsets_full['campaign_id_clean'] == utm_campaign_id) &
                        (adsets_full['adset_name'] == excel_name)
                    )
                    if match_mask.any():
                        adsets_full.loc[match_mask, 'conversions'] = conv_row['conversions']
                        matches_found += 1
                        logger.info(f"      ✅ Match EXATO encontrado: '{utm_name[:50]}' → '{excel_name[:50]}'")
                        break

                # Estratégia 2: Substring/similaridade (70% dos caracteres)
                min_len = min(len(utm_name_normalized), len(excel_name_normalized))

                if min_len >= 20:  # Só tentar se os nomes forem razoavelmente longos
                    # Verificar se os primeiros 70% dos caracteres são iguais
                    check_len = int(min_len * 0.7)
                    if utm_name_normalized[:check_len] == excel_name_normalized[:check_len]:
                        # Encontrar a linha correspondente em adsets_full
                        match_mask = (
                            (adsets_full['campaign_id_clean'] == utm_campaign_id) &
                            (adsets_full['adset_name'] == excel_name)
                        )
                        if match_mask.any():
                            adsets_full.loc[match_mask, 'conversions'] = conv_row['conversions']
                            matches_found += 1
                            logger.info(f"      ✅ Match flexível ({check_len} chars): '{utm_name[:50]}' → '{excel_name[:50]}'")
                            break

        if matches_found > 0:
            logger.info(f"   ✅ Recuperadas {matches_found} conversões via matching flexível!")
        else:
            logger.info(f"   ⚠️ Nenhum match flexível encontrado")

    adsets_full['conversions'] = adsets_full['conversions'].fillna(0)
    adsets_full['revenue'] = adsets_full['revenue'].fillna(0)  # Receita real do matched_df

    # Remover colunas temporárias
    adsets_full = adsets_full.drop(columns=['campaign_id_clean', 'campaign_id_from_utm', 'campaign_name_conv'], errors='ignore')

    # Calcular métricas de negócio
    # IMPORTANTE: NÃO sobrescrever 'leads' - o valor já vem correto do Excel!
    # Apenas garantir que leads esteja preenchido (fallback para casos sem dados)
    if 'leads' not in adsets_full.columns or adsets_full['leads'].isna().all():
        logger.warning("   ⚠️ Coluna 'leads' não encontrada ou vazia, usando count como fallback")
        adsets_full['leads'] = adsets_full.groupby(['campaign_id', 'adset_name'])['adset_id'].transform('count')
    else:
        # Preencher NaN com 0
        adsets_full['leads'] = adsets_full['leads'].fillna(0)

    # DEBUG: Verificar adsets com conversões mas sem leads
    weird_adsets = adsets_full[
        (adsets_full['conversions'] > 0) &
        (adsets_full['leads'] == 0)
    ]

    if len(weird_adsets) > 0:
        logger.warning(f"\n   ⚠️ ATENÇÃO: {len(weird_adsets)} adset(s) com conversões mas 0 leads:")
        for idx, row in weird_adsets.iterrows():
            logger.warning(f"      • {row['campaign_name'][:60]}")
            logger.warning(f"        Adset: {row['adset_name'][:50]}")
            logger.warning(f"        Conversões: {row['conversions']:.0f} | Leads (Excel): {row['leads']:.0f}")
            logger.warning(f"        Isso indica discrepância entre dados CAPI/CSV e relatório Meta")
        logger.warning("")

    # AJUSTE ESPECIAL: Adsets da campanha edge case que não dispara evento Lead
    # Aplicar mesma lógica de "forjar" leads que foi feita no nível de campanha
    campaign_special_id_prefix = '120234062599950'

    # Verificar se há coluna LeadQualified
    if 'lead_qualified' in adsets_full.columns or 'LeadQualified' in adsets_full.columns:
        lq_col = 'lead_qualified' if 'lead_qualified' in adsets_full.columns else 'LeadQualified'

        # Calcular proporção média LQ/Leads dos adsets normais (excluindo campanha especial)
        adsets_normal = adsets_full[
            ~adsets_full['campaign_id'].astype(str).str.startswith(campaign_special_id_prefix)
        ]

        total_leads_normal = adsets_normal['leads'].sum()
        total_lq_normal = adsets_normal[lq_col].sum()

        if total_leads_normal > 0 and total_lq_normal > 0:
            avg_ratio = total_lq_normal / total_leads_normal

            # Ajustar adsets da campanha especial
            for idx in adsets_full.index:
                camp_id = str(adsets_full.at[idx, 'campaign_id'])
                if camp_id.startswith(campaign_special_id_prefix):
                    lq = adsets_full.at[idx, lq_col]
                    current_leads = adsets_full.at[idx, 'leads']

                    # Se tem LQ mas não tem leads, calcular leads artificiais
                    if pd.notna(lq) and lq > 0 and (pd.isna(current_leads) or current_leads == 0):
                        leads_artificial = int(lq / avg_ratio)
                        adsets_full.at[idx, 'leads'] = leads_artificial
                        logger.info(f"   🔧 Adset especial ajustado (Matched): {adsets_full.at[idx, 'adset_name'][:50]}")
                        logger.info(f"      LeadQualified: {lq:.0f} → Leads artificial: {leads_artificial} (proporção {avg_ratio:.2%})")

    # Filtrar adsets com gasto 0 E leads 0 (sem atividade)
    # IMPORTANTE: Manter adsets com conversões mesmo se spend/leads = 0
    adsets_full = adsets_full[
        (adsets_full['spend'] > 0) |
        (adsets_full['leads'] > 0) |
        (adsets_full['conversions'] > 0)
    ].copy()

    logger.info(f"   📊 Adsets após filtro (removidos com spend=0 e leads=0): {len(adsets_full)}")

    adsets_full['cpl'] = adsets_full['spend'] / adsets_full['leads'].replace(0, 1)
    adsets_full['cpa'] = adsets_full['spend'] / adsets_full['conversions'].replace(0, 1)
    adsets_full['conversion_rate'] = (adsets_full['conversions'] / adsets_full['leads'].replace(0, 1)) * 100
    # NOTA: 'revenue' já vem do matched_df com valores reais de venda, não usar product_value fixo
    # adsets_full['revenue'] já foi preenchida no merge acima
    adsets_full['roas'] = adsets_full['revenue'] / adsets_full['spend'].replace(0, 1)
    adsets_full['margin'] = adsets_full['revenue'] - adsets_full['spend']
    adsets_full['margin_pct'] = (adsets_full['margin'] / adsets_full['revenue'].replace(0, 1)) * 100

    # Agregação ML vs Controle
    aggregated = adsets_full.groupby('ml_type').agg({
        'adset_name': 'nunique',
        'spend': 'sum',
        'leads': 'sum',
        'conversions': 'sum',
        'cpl': 'mean',
        'cpa': 'mean',
        'conversion_rate': 'mean',
        'roas': 'mean',
        'margin': 'sum',
        'margin_pct': 'mean'
    }).reset_index()

    # Detalhamento adset-a-adset (CADA CAMPANHA SEPARADA - NÃO AGREGAR)
    # MUDANÇA IMPORTANTE: Não agrupar por adset_name
    # Cada linha representa um adset de uma campanha específica
    # Exemplo: "ADV | Lookalike 1%" na campanha A é diferente de "ADV | Lookalike 1%" na campanha B
    detailed = adsets_full[['campaign_name', 'campaign_id', 'adset_name', 'adset_id', 'ml_type',
                             'spend', 'leads', 'conversions', 'cpl', 'cpa',
                             'conversion_rate', 'roas', 'revenue', 'margin', 'margin_pct']].copy()

    # Adicionar account_id baseado no adset_id
    # O account_name vem do MetaReportsLoader (extraído do nome do arquivo Excel)
    if '_account_name' in adsets_metrics_df.columns:
        account_map = adsets_metrics_df[['adset_id', '_account_name']].drop_duplicates().set_index('adset_id')['_account_name'].to_dict()
        detailed['account_id'] = detailed['adset_id'].map(account_map)
    elif 'account_id' in adsets_metrics_df.columns:
        account_map = adsets_metrics_df[['adset_id', 'account_id']].drop_duplicates().set_index('adset_id')['account_id'].to_dict()
        detailed['account_id'] = detailed['adset_id'].map(account_map)
    else:
        detailed['account_id'] = None

    # IMPORTANTE: Adsets herdam a classificação da CAMPANHA PAI
    # Não reclassificamos por optimization_goal do adset para manter consistência
    # A classificação já foi feita no nível de campanha considerando optimization_goal

    # Buscar optimization_goal apenas para referência (não para reclassificar)
    if 'optimization_goal' in adsets_metrics_df.columns:
        optimization_map = adsets_metrics_df[['adset_id', 'optimization_goal']].drop_duplicates().set_index('adset_id')['optimization_goal'].to_dict()
        detailed['optimization_goal'] = detailed['adset_id'].map(optimization_map)
    else:
        detailed['optimization_goal'] = None

    # Adicionar comparison_group HERDADO da campanha
    if comparison_group_map:
        # NOVO: Usar mapeamento refinado direto (já distingue Eventos ML vs Otimização ML)
        # IMPORTANTE: Usar primeiros 15 dígitos para matching (mesma lógica do comparison_group_map)
        detailed['campaign_id_15'] = detailed['campaign_id'].astype(str).str[:15]

        # DEBUG: Verificar mapeamento antes de aplicar
        logger.info(f"\n   🔍 DEBUG ADSETS - Verificando mapeamento comparison_group:")
        logger.info(f"      Total de IDs no mapa: {len(comparison_group_map)}")
        logger.info(f"      Total de adsets: {len(detailed)}")
        logger.info(f"      IDs únicos nos adsets (15 dig): {detailed['campaign_id_15'].nunique()}")

        # DEBUG: Mostrar mapeamento das campanhas ML
        logger.info(f"\n      Mapeamento de campanhas ML:")
        for id_15, group in comparison_group_map.items():
            if 'ML' in group or group == 'Eventos ML':
                logger.info(f"         {id_15} → {group}")

        # Verificar se há IDs que não fazem match
        ids_in_detailed = set(detailed['campaign_id_15'].unique())
        ids_in_map = set(comparison_group_map.keys())
        ids_not_in_map = ids_in_detailed - ids_in_map
        if ids_not_in_map:
            logger.warning(f"      ⚠️ {len(ids_not_in_map)} IDs de adsets NÃO encontrados no mapa:")
            for id_val in list(ids_not_in_map)[:5]:
                # Mostrar o nome da campanha correspondente
                sample_adset = detailed[detailed['campaign_id_15'] == id_val].iloc[0]
                logger.warning(f"         • ID 15: {id_val} → {sample_adset['campaign_name'][:50]}")

        detailed['comparison_group'] = detailed['campaign_id_15'].map(comparison_group_map)

        # DEBUG: Verificar se há NaN após mapeamento
        unmapped_count = detailed['comparison_group'].isna().sum()
        if unmapped_count > 0:
            logger.warning(f"      ⚠️ {unmapped_count} adsets sem grupo após mapeamento (NaN)")

        logger.info("   ✅ Usando mapeamento refinado (Eventos ML / Otimização ML / Controle)")
    else:
        # LEGACY: Converter ml_type para comparison_group (sem distinção Eventos/Otimização)
        def classify_comparison_group_from_ml_type(row):
            """Conversão legacy: ml_type → comparison_group (sem refinamento)"""
            ml_type = row['ml_type']
            if ml_type == 'COM_ML':
                return 'Eventos ML'  # Assume todos ML são Eventos (não ideal)
            elif ml_type == 'SEM_ML':
                return 'Controle'
            else:
                return 'Outro'

        detailed['comparison_group'] = detailed.apply(classify_comparison_group_from_ml_type, axis=1)
        logger.warning("   ⚠️ Usando mapeamento legacy (sem distinção Eventos ML vs Otimização ML)")

    # Filtrar apenas "Eventos ML" vs "Controle" (excluir "Otimização ML" e "Outro")
    before_filter = len(detailed)
    conversions_before_filter = detailed['conversions'].sum()

    # DEBUG: Verificar quais conversões serão removidas pelo filtro
    removed_by_filter = detailed[~detailed['comparison_group'].isin(['Eventos ML', 'Controle'])].copy()
    if len(removed_by_filter) > 0:
        convs_removed = removed_by_filter['conversions'].sum()
        logger.warning(f"\n   🔍 CONVERSÕES REMOVIDAS PELO FILTRO (Otimização ML / Outro):")
        logger.warning(f"      Adsets removidos: {len(removed_by_filter)}")
        logger.warning(f"      Conversões removidas: {convs_removed:.0f}")

        # Mostrar breakdown por comparison_group
        by_group = removed_by_filter.groupby('comparison_group').agg({
            'adset_name': 'count',
            'conversions': 'sum'
        }).reset_index()
        by_group.columns = ['Grupo', 'Adsets', 'Conversões']
        logger.warning(f"\n      Breakdown por grupo:")
        for _, row in by_group.iterrows():
            logger.warning(f"         {row['Grupo']}: {row['Adsets']} adsets, {row['Conversões']:.0f} conversões")

        # Mostrar detalhes de cada adset removido
        logger.warning(f"\n      Detalhes dos adsets removidos:")
        for idx, row in removed_by_filter.iterrows():
            logger.warning(f"      • [{row['comparison_group']}] {row['campaign_name'][:60]}")
            logger.warning(f"        Adset: {row['adset_name'][:60]}")
            logger.warning(f"        Conversões: {row['conversions']:.0f}")
            logger.warning(f"        Campaign ID (completo): {row['campaign_id']}")
            logger.warning(f"        Campaign ID (15 dígitos): {str(row['campaign_id'])[:15]}")

            # Buscar emails específicos deste adset em matched_df
            if 'campaign' in matched_df.columns and 'medium' in matched_df.columns:
                # Construir o nome completo da campanha como aparece no matched_df
                campaign_variations = [
                    row['campaign_name'],
                    f"{row['campaign_name']}|{row['campaign_id']}",
                    f"{row['campaign_name'].rstrip('|')}|{row['campaign_id']}"
                ]

                matching_rows = pd.DataFrame()
                for camp_var in campaign_variations:
                    matches = matched_df[
                        (matched_df['campaign'] == camp_var) &
                        (matched_df['medium'] == row['adset_name']) &
                        (matched_df['converted'] == True)
                    ]
                    if len(matches) > 0:
                        matching_rows = matches
                        break

                if len(matching_rows) > 0:
                    emails = matching_rows['email'].unique()
                    logger.warning(f"        Emails ({len(emails)}): {', '.join(emails[:5])}{' ...' if len(emails) > 5 else ''}")
                else:
                    logger.warning(f"        ⚠️  Não encontrei os emails correspondentes em matched_df")
        logger.warning("")

    detailed = detailed[detailed['comparison_group'].isin(['Eventos ML', 'Controle'])].copy()
    after_filter = len(detailed)
    conversions_after_filter = detailed['conversions'].sum()

    if before_filter != after_filter:
        logger.info(f"   🔍 Filtrados {before_filter - after_filter} adsets (Otimização ML ou Outro)")
        logger.info(f"   📊 Conversões: {conversions_before_filter:.0f} → {conversions_after_filter:.0f} ({conversions_before_filter - conversions_after_filter:.0f} removidas)")

    logger.info("   ✅ Comparações de adsets calculadas")
    logger.info(f"      Adsets após filtro (Eventos ML + Controle): {len(detailed)}")

    return {
        'aggregated': aggregated,
        'detailed': detailed
    }


def compare_ad_performance(
    ad_metrics_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    ml_type_map: Dict[str, str],
    product_value: float = 2000.0,
    comparison_group_map: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compara performance de anúncios entre ML e Controle.

    IMPORTANTE: Usa matching preciso por (Campaign ID + ad_code).
    Mesma lógica dos adsets, mas usando utm_content em vez de utm_medium.

    Args:
        ad_metrics_df: DataFrame com métricas de anúncios (da Meta API)
        matched_df: DataFrame com conversões matched (leads→vendas)
        ml_type_map: Mapeamento campaign_id → ml_type (DEPRECATED - usar comparison_group_map)
        product_value: Valor do produto em R$
        comparison_group_map: Mapeamento campaign_id → comparison_group ('Eventos ML', 'Otimização ML', 'Controle')

    Returns:
        Dict com DataFrames:
        - 'aggregated': Agregação ML vs Controle
        - 'detailed': Detalhamento anúncio-a-anúncio
    """
    logger.info("📊 Comparando performance de anúncios...")

    # Adicionar ml_type aos anúncios (para compatibilidade)
    ad_metrics_df['ml_type'] = ad_metrics_df['campaign_id'].map(ml_type_map)

    # Se temos comparison_group_map, usar ele diretamente (novo comportamento)
    if comparison_group_map:
        ad_metrics_df['comparison_group_from_campaign'] = ad_metrics_df['campaign_id'].map(comparison_group_map)

    # Criar coluna account_id se não existir (pode vir como _account_name)
    if 'account_id' not in ad_metrics_df.columns and '_account_name' in ad_metrics_df.columns:
        ad_metrics_df['account_id'] = ad_metrics_df['_account_name']

    # =========================================================================
    # MATCHING PRECISO DE ANÚNCIOS (mesma lógica dos adsets)
    # =========================================================================

    # Função para extrair Campaign ID (reutilizar a mesma dos adsets)
    def extract_campaign_id(campaign_name):
        """Extrai o Campaign ID do final do nome da campanha"""
        if pd.isna(campaign_name):
            return None
        parts = str(campaign_name).split('|')
        if len(parts) > 1:
            last_part = parts[-1].strip()
            if last_part.isdigit() and len(last_part) >= 18:
                return last_part
        return None

    # Calcular conversões por ANÚNCIO (matching preciso!)
    if 'converted' in matched_df.columns:
        conversions_df = matched_df[matched_df['converted'] == True].copy()

        # 1. Extrair Campaign ID do utm_campaign
        conversions_df['campaign_id_from_utm'] = conversions_df['campaign'].apply(extract_campaign_id)

        # 2. Extrair ad_code do utm_content (padrão: AD0\d+)
        conversions_df['ad_code_from_utm'] = conversions_df['content'].str.extract(r'(AD0\d+)', expand=False)

        # DEBUG
        content_filled = conversions_df['content'].notna().sum()
        ad_code_extracted = conversions_df['ad_code_from_utm'].notna().sum()
        logger.info(f"\n   🔍 Extração de ad_code das conversões:")
        logger.info(f"      Total conversões: {len(conversions_df)}")
        logger.info(f"      utm_content preenchido: {content_filled} ({content_filled/len(conversions_df)*100:.1f}%)")
        logger.info(f"      ad_code extraído: {ad_code_extracted} ({ad_code_extracted/len(conversions_df)*100:.1f}%)")

        # 3. Preparar campaign_id_clean (primeiros 15 dígitos)
        conversions_df['campaign_id_clean'] = conversions_df['campaign_id_from_utm'].astype(str).str[:15]

        # 4. Agrupar por (campaign_id_clean, ad_code) - MATCHING PRECISO E calcular RECEITA REAL
        # CRÍTICO: Contar emails únicos para conversões, somar sale_value para receita
        conversions_by_campaign_ad = conversions_df.groupby(
            ['campaign_id_clean', 'ad_code_from_utm']
        ).agg({
            'email': 'nunique',  # Conversões únicas
            'sale_value': 'sum'   # Receita real total
        }).reset_index()
        conversions_by_campaign_ad.columns = ['campaign_id_clean', 'ad_code_from_utm', 'conversions', 'revenue']

        logger.info(f"      Agrupadas por (Campaign ID + ad_code): {len(conversions_by_campaign_ad)}")

        # 5. Calcular LEADS por anúncio (usar TODO o matched_df, não só conversões)
        logger.info(f"\n   📊 Calculando leads por anúncio:")

        # Extrair ad_code de TODOS os leads
        all_leads_df = matched_df.copy()
        all_leads_df['campaign_id_from_utm'] = all_leads_df['campaign'].apply(extract_campaign_id)
        all_leads_df['ad_code_from_utm'] = all_leads_df['content'].str.extract(r'(AD0\d+)', expand=False)
        all_leads_df['campaign_id_clean'] = all_leads_df['campaign_id_from_utm'].astype(str).str[:15]

        # Agrupar por (campaign_id_clean, ad_code) para contar leads
        leads_by_campaign_ad = all_leads_df[all_leads_df['ad_code_from_utm'].notna()].groupby(
            ['campaign_id_clean', 'ad_code_from_utm']
        ).size().reset_index(name='leads')

        logger.info(f"      Total de leads com ad_code: {leads_by_campaign_ad['leads'].sum()}")
        logger.info(f"      Combinações únicas (Campaign ID + ad_code): {len(leads_by_campaign_ad)}")

        # 6. Preparar ad_metrics_df para merge
        ad_metrics_df['campaign_id_clean'] = ad_metrics_df['campaign_id'].astype(str).str[:15]

        # 7. AGREGAÇÃO: Consolidar anúncios com mesmo (campaign_id, ad_code)
        #    Mesmo ad_code pode ter múltiplos ad_ids no Excel → somar spend
        logger.info(f"\n   📊 Agregando anúncios com mesmo (Campaign ID + ad_code):")
        logger.info(f"      Total de linhas no Excel: {len(ad_metrics_df)}")

        ad_metrics_aggregated = ad_metrics_df.groupby(['campaign_id_clean', 'ad_code', 'ml_type'], dropna=False).agg({
            'spend': 'sum',
            'campaign_id': 'first',  # Manter ID original
            'campaign_name': 'first',  # Manter nome
            'ad_name': 'first',  # Manter nome do primeiro ad
            'adset_name': 'first',  # Nome do adset
            'account_id': 'first'  # Account ID
        }).reset_index()

        logger.info(f"      Total após agregação: {len(ad_metrics_aggregated)}")
        logger.info(f"      Anúncios consolidados: {len(ad_metrics_df) - len(ad_metrics_aggregated)}")

        # 8. Merge LEADS por (campaign_id_clean, ad_code)
        ad_full = ad_metrics_aggregated.merge(
            leads_by_campaign_ad,
            left_on=['campaign_id_clean', 'ad_code'],
            right_on=['campaign_id_clean', 'ad_code_from_utm'],
            how='left',
            suffixes=('', '_leads')
        )

        ad_full['leads'] = ad_full['leads'].fillna(0)

        # 9. Merge CONVERSÕES E RECEITA REAL por (campaign_id_clean, ad_code)
        ad_full = ad_full.merge(
            conversions_by_campaign_ad,
            left_on=['campaign_id_clean', 'ad_code'],
            right_on=['campaign_id_clean', 'ad_code_from_utm'],
            how='left',
            suffixes=('', '_conv')
        )

        ad_full['conversions'] = ad_full['conversions'].fillna(0)
        ad_full['revenue'] = ad_full['revenue'].fillna(0)  # Receita real do matched_df

        logger.info(f"   ✅ Anúncios com leads: {(ad_full['leads'] > 0).sum()}")
        logger.info(f"   ✅ Anúncios com conversões: {(ad_full['conversions'] > 0).sum()}")
        logger.info(f"   📊 Total leads: {ad_full['leads'].sum():.0f}")
        logger.info(f"   📊 Total conversões atribuídas: {ad_full['conversions'].sum():.0f}")

    else:
        ad_metrics_df['campaign_id_clean'] = ad_metrics_df['campaign_id'].astype(str).str[:15]
        ad_full = ad_metrics_df.copy()
        ad_full['conversions'] = 0
        ad_full['leads'] = 0
        ad_full['revenue'] = 0

    # =========================================================================
    # CALCULAR MÉTRICAS DE NEGÓCIO
    # =========================================================================

    # Calcular métricas
    ad_full['cpl'] = ad_full['spend'] / ad_full['leads'].replace(0, 1)
    ad_full['cpa'] = ad_full['spend'] / ad_full['conversions'].replace(0, 1)
    ad_full['conversion_rate'] = (ad_full['conversions'] / ad_full['leads'].replace(0, 1)) * 100
    # NOTA: 'revenue' já vem do matched_df com valores reais de venda, não usar product_value fixo
    # ad_full['revenue'] já foi preenchida no merge acima
    ad_full['roas'] = ad_full['revenue'] / ad_full['spend'].replace(0, 1)
    ad_full['margin'] = ad_full['revenue'] - ad_full['spend']
    ad_full['margin_pct'] = (ad_full['margin'] / ad_full['revenue'].replace(0, 1)) * 100

    # CORREÇÃO: Adicionar comparison_group ANTES de criar agregações
    # Preparar campaign_id_15 para matching
    ad_full['campaign_id_15'] = ad_full['campaign_id'].astype(str).str[:15]

    # Aplicar comparison_group_map
    if comparison_group_map:
        ad_full['comparison_group'] = ad_full['campaign_id_15'].map(comparison_group_map)
        logger.info("   ✅ Usando mapeamento refinado (Eventos ML / Otimização ML / Controle)")
    else:
        # Fallback para ml_type
        ad_full['comparison_group'] = ad_full['ml_type'].map({
            'COM_ML': 'Eventos ML',
            'SEM_ML': 'Controle'
        })
        logger.warning("   ⚠️ comparison_group_map não disponível, usando classificação simples")

    # FILTRAR antes de agregar: apenas Eventos ML e Controle
    before_filter_count = len(ad_full)
    ad_full_filtered = ad_full[ad_full['comparison_group'].isin(['Eventos ML', 'Controle'])].copy()
    after_filter_count = len(ad_full_filtered)

    if before_filter_count != after_filter_count:
        logger.info(f"   🔍 Filtrados {before_filter_count - after_filter_count} ads (Otimização ML ou Outro) ANTES da agregação")

    # Agregação por comparison_group (NÃO ml_type)
    aggregated = ad_full_filtered.groupby('comparison_group').agg({
        'ad_code': 'nunique',
        'spend': 'sum',
        'leads': 'sum',
        'conversions': 'sum',
        'cpl': 'mean',
        'cpa': 'mean',
        'conversion_rate': 'mean',
        'roas': 'mean',
        'margin': 'sum',
        'margin_pct': 'mean'
    }).reset_index()

    # Detalhamento anúncio-a-anúncio (incluir informações contextuais)
    # USAR ad_full_filtered para garantir que apenas Eventos ML e Controle sejam incluídos
    detailed = ad_full_filtered.groupby(['ad_code', 'comparison_group']).agg({
        'campaign_name': 'first',  # Nome da campanha
        'campaign_id': 'first',  # ID da campanha (para buscar optimization_goal)
        'account_id': 'first',  # Account ID
        'adset_name': 'first',  # Nome do adset
        'ad_name': 'first',  # Nome do anúncio
        'ml_type': 'first',  # Manter ml_type para compatibilidade
        'spend': 'sum',
        'leads': 'sum',
        'conversions': 'sum',
        'cpl': 'mean',
        'cpa': 'mean',
        'conversion_rate': 'mean',
        'roas': 'mean',
        'revenue': 'sum',
        'margin': 'sum',
        'margin_pct': 'mean'
    }).reset_index()

    # IMPORTANTE: Ads herdam a classificação da CAMPANHA PAI
    # Não reclassificamos por optimization_goal do ad para manter consistência
    # A classificação já foi feita no nível de campanha considerando optimization_goal

    # Buscar optimization_goal apenas para referência (não para reclassificar)
    if 'optimization_goal' in ad_metrics_df.columns:
        optimization_map = ad_metrics_df[['campaign_id', 'optimization_goal']].drop_duplicates().set_index('campaign_id')['optimization_goal'].to_dict()
        detailed['optimization_goal'] = detailed['campaign_id'].map(optimization_map)
    else:
        detailed['optimization_goal'] = None

    # NOTA: comparison_group já foi adicionado ao ad_full nas linhas 1086-1095 e
    # o DataFrame foi filtrado (apenas Eventos ML e Controle) nas linhas 1097-1099.
    # O 'detailed' foi criado a partir do ad_full_filtered já filtrado (linhas 1118-1137),
    # então não é necessário adicionar comparison_group ou filtrar novamente aqui.

    logger.info(f"   ✅ Comparações de anúncios calculadas")
    logger.info(f"      Anúncios após filtro (Eventos ML + Controle): {len(detailed)}")

    return {
        'aggregated': aggregated,
        'detailed': detailed
    }


def compare_ads_in_matched_adsets(
    ad_metrics_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    ml_type_map: Dict[str, str],
    product_value: float = 2000.0,
    comparison_group_map: Optional[Dict[str, str]] = None,
    filtered_matched_adsets: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compara performance de anúncios que pertencem APENAS aos adsets matched.

    DIFERENÇA vs compare_ad_performance:
    - compare_ad_performance: todos os ads da lista MATCHED_ADS
    - compare_ads_in_matched_adsets: apenas ads cujo adset pai está em MATCHED_ADSETS

    Args:
        ad_metrics_df: DataFrame com métricas de anúncios (da Meta API)
        matched_df: DataFrame com conversões matched (leads→vendas)
        ml_type_map: Mapeamento campaign_id → ml_type
        product_value: Valor do produto em R$
        comparison_group_map: Mapeamento campaign_id → comparison_group
        filtered_matched_adsets: Lista de adsets matched que passaram nos filtros (comparison_group).
                                 Se None, usa a lista hardcoded MATCHED_ADSETS.

    Returns:
        Dict com DataFrames:
        - 'aggregated': Agregação ML vs Controle
        - 'detailed': Detalhamento anúncio-a-anúncio
    """
    logger.info("📊 Comparando performance de anúncios EM adsets matched...")

    # CORREÇÃO: Usar lista filtrada de adsets em vez da hardcoded
    # Isso garante que apenas ads de adsets "Eventos ML" e "Controle" sejam incluídos
    # (excluindo "Otimização ML" e outros)
    adsets_to_use = filtered_matched_adsets if filtered_matched_adsets is not None else MATCHED_ADSETS

    logger.info(f"   📋 Usando {len(adsets_to_use)} adsets matched (filtrados por comparison_group)")

    # Filtrar apenas ads cujo adset pai está na lista filtrada
    ads_in_matched_adsets = ad_metrics_df[
        ad_metrics_df['adset_name'].isin(adsets_to_use)
    ].copy()

    logger.info(f"   📋 Ads em adsets matched: {len(ads_in_matched_adsets)}")
    logger.info(f"   📋 Ad codes únicos: {ads_in_matched_adsets['ad_name'].nunique()}")

    if len(ads_in_matched_adsets) == 0:
        logger.warning("   ⚠️ Nenhum ad encontrado nos adsets matched!")
        return {
            'aggregated': pd.DataFrame(),
            'detailed': pd.DataFrame()
        }

    # Usar a mesma lógica de compare_ad_performance, mas com o DataFrame filtrado
    return compare_ad_performance(
        ads_in_matched_adsets,
        matched_df,
        ml_type_map,
        product_value,
        comparison_group_map
    )


def compare_matched_ads_in_matched_adsets(
    ad_metrics_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    ml_type_map: Dict[str, str],
    product_value: float = 2000.0,
    comparison_group_map: Optional[Dict[str, str]] = None,
    filtered_matched_adsets: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compara performance de anúncios MATCHED que pertencem APENAS aos adsets matched.

    Combina dois filtros:
    1. Apenas adsets matched (aparecem em ML e Controle)
    2. Dentro desses adsets, apenas ads matched (ad_code aparece em ML e Controle)

    Args:
        ad_metrics_df: DataFrame com métricas de anúncios
        matched_df: DataFrame com dados de matching leads-vendas
        ml_type_map: Dict mapeando campaign_id para ml_type (COM_ML/SEM_ML)
        product_value: Valor do produto para cálculo de receita
        comparison_group_map: Dict mapeando campaign_id (15 dígitos) para grupo de comparação
        filtered_matched_adsets: Lista de adsets matched que passaram nos filtros

    Returns:
        Dict com 'aggregated' e 'detailed' DataFrames
    """
    logger.info("📊 Comparando ads MATCHED em adsets MATCHED...")

    # PASSO 1: Filtrar apenas ads dos adsets matched
    adsets_to_use = filtered_matched_adsets if filtered_matched_adsets is not None else MATCHED_ADSETS

    logger.info(f"   📋 Usando {len(adsets_to_use)} adsets matched (filtrados por comparison_group)")

    ads_in_matched_adsets = ad_metrics_df[
        ad_metrics_df['adset_name'].isin(adsets_to_use)
    ].copy()

    logger.info(f"   📋 Ads em adsets matched: {len(ads_in_matched_adsets)}")

    if len(ads_in_matched_adsets) == 0:
        logger.warning("   ⚠️ Nenhum ad encontrado nos adsets matched!")
        return {
            'aggregated': pd.DataFrame(),
            'detailed': pd.DataFrame()
        }

    # PASSO 2: Identificar ad_codes matched (aparecem em ML E Controle)
    # Preparar campaign_id_15 e comparison_group
    ads_in_matched_adsets['campaign_id_15'] = ads_in_matched_adsets['campaign_id'].astype(str).str[:15]

    if comparison_group_map:
        ads_in_matched_adsets['comparison_group'] = ads_in_matched_adsets['campaign_id_15'].map(comparison_group_map)
    else:
        ads_in_matched_adsets['comparison_group'] = ads_in_matched_adsets['campaign_id'].map(ml_type_map).map({
            'COM_ML': 'Eventos ML',
            'SEM_ML': 'Controle'
        })

    # Filtrar apenas Eventos ML e Controle
    ads_filtered = ads_in_matched_adsets[
        ads_in_matched_adsets['comparison_group'].isin(['Eventos ML', 'Controle'])
    ].copy()

    logger.info(f"   📋 Ads após filtro de comparison_group: {len(ads_filtered)}")

    # Identificar quais ad_codes aparecem em AMBOS os grupos
    ad_codes_by_group = ads_filtered.groupby('comparison_group')['ad_code'].unique()

    if 'Eventos ML' not in ad_codes_by_group or 'Controle' not in ad_codes_by_group:
        logger.warning("   ⚠️ Não há ad_codes em ambos os grupos!")
        return {
            'aggregated': pd.DataFrame(),
            'detailed': pd.DataFrame()
        }

    eventos_ml_codes = set(ad_codes_by_group['Eventos ML'])
    controle_codes = set(ad_codes_by_group['Controle'])
    matched_ad_codes = eventos_ml_codes & controle_codes

    logger.info(f"   📋 Ad codes matched (aparecem em ML E Controle): {len(matched_ad_codes)}")
    logger.info(f"      Eventos ML: {len(eventos_ml_codes)} códigos")
    logger.info(f"      Controle: {len(controle_codes)} códigos")
    logger.info(f"      Interseção: {len(matched_ad_codes)} códigos")

    if len(matched_ad_codes) == 0:
        logger.warning("   ⚠️ Nenhum ad_code matched encontrado!")
        return {
            'aggregated': pd.DataFrame(),
            'detailed': pd.DataFrame()
        }

    # PASSO 3: Filtrar apenas ads com ad_codes matched
    matched_ads_only = ads_filtered[
        ads_filtered['ad_code'].isin(matched_ad_codes)
    ].copy()

    logger.info(f"   📋 Ads finais (matched em adsets matched): {len(matched_ads_only)}")

    # PASSO 4: Calcular métricas usando a mesma lógica de compare_ad_performance
    return compare_ad_performance(
        matched_ads_only,
        matched_df,
        ml_type_map,
        product_value,
        comparison_group_map
    )


# ============================================================================
# FUNÇÕES DE FORMATAÇÃO PARA EXCEL
# ============================================================================

def prepare_adset_comparison_for_excel(
    comparisons: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Prepara DataFrames de comparação por adset para Excel.
    Formato similar à aba "Comparação por Campanhas".

    Args:
        comparisons: Dict com 'aggregated' e 'detailed'

    Returns:
        Dict com DataFrames formatados para Excel
    """
    logger.info("📝 Preparando comparações por adset para Excel...")

    excel_dfs = {}

    # Usar apenas 'detailed' com formato similar à aba Campanhas
    if not comparisons['detailed'].empty:
        df = comparisons['detailed'].copy()

        # Renomear e reordenar colunas para formato similar à aba Campanhas
        df = df.rename(columns={
            'account_id': 'Conta',
            'comparison_group': 'Grupo',
            'campaign_name': 'Campanha',
            'adset_name': 'Adset',
            'adset_id': 'Adset ID',
            'leads': 'Leads',
            'conversions': 'Vendas',
            'conversion_rate': 'Taxa de conversão',
            'spend': 'Valor gasto',
            'cpl': 'CPL',
            'roas': 'ROAS',
            'revenue': 'Receita Total',
            'margin': 'Margem de contribuição'
        })

        # Mapear account_id para nomes amigáveis
        if 'Conta' in df.columns:
            def clean_account_name(x):
                if pd.isna(x):
                    return 'N/A'
                x = str(x)
                # Se é um ID de conta
                if x.startswith('act_'):
                    account_ids = {
                        'act_188005769808959': 'Rodolfo Mori',
                        'act_786790755803474': 'Gestor de IA'
                    }
                    return account_ids.get(x, x)
                # Se é nome de arquivo, extrair nome da conta
                if 'Rodolfo Mori' in x:
                    return 'Rodolfo Mori'
                elif 'Gestor de IA' in x:
                    return 'Gestor de IA'
                return x

            df['Conta'] = df['Conta'].apply(clean_account_name)

        # Calcular receita total se não existir
        if 'Receita Total' not in df.columns and 'Vendas' in df.columns:
            df['Receita Total'] = df['Vendas'] * 2000.0  # product_value

        # Selecionar e ordenar colunas (similar à aba Campanhas, com Conta primeiro)
        columns_order = [
            'Conta', 'Campanha', 'Adset', 'Adset ID', 'Grupo', 'Leads', 'Vendas',
            'Taxa de conversão', 'Valor gasto', 'CPL', 'ROAS',
            'Receita Total', 'Margem de contribuição'
        ]

        # Incluir apenas colunas que existem
        available_columns = [col for col in columns_order if col in df.columns]
        df = df[available_columns]

        # Ordenar por ROAS descendente
        if 'ROAS' in df.columns:
            df = df.sort_values('ROAS', ascending=False)

        excel_dfs['comparacao_adsets'] = df

    logger.info(f"   ✅ {len(excel_dfs)} abas preparadas para Excel")

    return excel_dfs


def prepare_ad_comparison_for_excel(
    comparisons: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Prepara DataFrames de comparação por anúncio para Excel.
    Formato similar à aba "Comparação por Adsets".

    Args:
        comparisons: Dict com 'aggregated' e 'detailed'

    Returns:
        Dict com DataFrames formatados para Excel
    """
    logger.info("📝 Preparando comparações por anúncio para Excel...")

    excel_dfs = {}

    # Usar 'detailed' com formato similar à aba Adsets
    if 'detailed' in comparisons and not comparisons['detailed'].empty:
        df = comparisons['detailed'].copy()

        # Renomear colunas (removidas: Conta, Campanha, Adset - são agregadas e enganosas)
        df = df.rename(columns={
            'comparison_group': 'Grupo',
            'ad_code': 'Ad Code',
            'ad_name': 'Nome do Anúncio',
            'leads': 'Leads',
            'conversions': 'Vendas',
            'conversion_rate': 'Taxa de conversão',
            'spend': 'Valor gasto',
            'cpl': 'CPL',
            'roas': 'ROAS',
            'revenue': 'Receita Total',
            'margin': 'Margem de contribuição'
        })

        # Calcular receita total se não existir
        if 'Receita Total' not in df.columns and 'Vendas' in df.columns:
            df['Receita Total'] = df['Vendas'] * 2000.0  # product_value

        # Selecionar e ordenar colunas (removidas: Conta, Campanha, Adset)
        columns_order = [
            'Ad Code', 'Nome do Anúncio', 'Grupo',
            'Leads', 'Vendas', 'Taxa de conversão', 'Valor gasto', 'CPL', 'ROAS',
            'Receita Total', 'Margem de contribuição'
        ]

        # Incluir apenas colunas que existem
        available_columns = [col for col in columns_order if col in df.columns]
        df = df[available_columns]

        # Ordenar por ROAS descendente
        if 'ROAS' in df.columns:
            df = df.sort_values('ROAS', ascending=False)

        excel_dfs['comparacao_ads'] = df

    logger.info(f"   ✅ {len(excel_dfs)} abas preparadas para Excel")

    return excel_dfs
