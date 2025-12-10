#!/usr/bin/env python3
"""
Script CLI para Validação de Performance do Modelo de ML de Lead Scoring.

Compara campanhas COM ML vs SEM ML e valida performance por decil D1-D10.

Uso:
    python scripts/validate_ml_performance.py \
        --periodo periodo_1 \
        --account-id act_XXXXXXXXX

    python scripts/validate_ml_performance.py \
        --start-date 2025-11-11 \
        --end-date 2025-12-01 \
        --account-id act_XXXXXXXXX \
        --product-value 2000
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from glob import glob
import yaml
import logging
import time
import pandas as pd
from tabulate import tabulate

# Adicionar V2/ ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports dos módulos de validação
from src.validation.data_loader import LeadDataLoader, SalesDataLoader, CAPILeadDataLoader
from src.validation.campaign_classifier import add_ml_classification
from src.validation.matching import (
    match_leads_to_sales,
    get_matching_stats,
    filter_by_period
)
from src.validation.metrics_calculator import (
    CampaignMetricsCalculator,
    DecileMetricsCalculator,
    compare_ml_vs_non_ml,
    calculate_overall_stats,
    calculate_comparison_group_metrics
)
from src.validation.report_generator import ValidationReportGenerator
from src.validation.visualization import ValidationVisualizer
from src.validation.fair_campaign_comparison import FairCampaignMatcher
from src.validation.period_calculator import PeriodCalculator

# Imports de integrações existentes
from api.meta_integration import MetaAdsIntegration
from api.meta_config import META_CONFIG

# Para exibição de tabelas no terminal
try:
    from tabulate import tabulate
except ImportError:
    print("⚠️ Biblioteca 'tabulate' não encontrada. Instale com: pip install tabulate")
    sys.exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse argumentos da linha de comando.

    Returns:
        Namespace com argumentos
    """
    parser = argparse.ArgumentParser(
        description='Sistema de Validação de Performance ML - Lead Scoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Usar período pré-configurado
  python scripts/validate_ml_performance.py --periodo periodo_1 --account-id act_123456789

  # Usar datas customizadas
  python scripts/validate_ml_performance.py \\
    --start-date 2025-11-11 \\
    --end-date 2025-12-01 \\
    --account-id act_123456789

  # Sobrescrever parâmetros do config
  python scripts/validate_ml_performance.py \\
    --periodo periodo_1 \\
    --account-id act_123456789 \\
    --product-value 2500 \\
    --max-match-days 45
        """
    )

    # Período
    period_group = parser.add_mutually_exclusive_group()
    period_group.add_argument(
        '--periodo',
        type=str,
        help='Período pré-configurado (periodo_1, periodo_2, periodo_3)'
    )
    period_group.add_argument(
        '--start-date',
        type=str,
        help='Data início (YYYY-MM-DD) - usa com --end-date'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='Data fim (YYYY-MM-DD) - usa com --start-date'
    )

    # Período de vendas (opcional, separado do período de captação)
    parser.add_argument(
        '--sales-start-date',
        type=str,
        help='Data início das vendas para matching (YYYY-MM-DD) - opcional'
    )

    parser.add_argument(
        '--sales-end-date',
        type=str,
        help='Data fim das vendas para matching (YYYY-MM-DD) - opcional'
    )

    # Meta Ads API
    parser.add_argument(
        '--account-id',
        type=str,
        nargs='+',
        required=True,
        help='IDs das contas Meta Ads, separados por espaço (ex: act_123456789 act_987654321)'
    )

    # Caminhos
    parser.add_argument(
        '--leads-path',
        type=str,
        help='Caminho para CSV de leads (default: files/validation/leads/leads_completo.csv)'
    )

    parser.add_argument(
        '--vendas-path',
        type=str,
        help='Caminho para pasta com arquivos de vendas (default: files/validation/vendas/)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Diretório de saída (default: files/validation/resultados/)'
    )

    # Configurações
    parser.add_argument(
        '--config',
        type=str,
        default='configs/validation_config.yaml',
        help='Caminho para arquivo de configuração YAML'
    )

    parser.add_argument(
        '--product-value',
        type=float,
        help='Valor do produto em R$ (sobrescreve config)'
    )

    parser.add_argument(
        '--max-match-days',
        type=int,
        help='Janela máxima para matching em dias (sobrescreve config)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Desabilita cache de chamadas à Meta API (força buscar dados novos)'
    )

    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Limpa todo o cache antes de executar'
    )

    # Meta Access Token
    parser.add_argument(
        '--meta-token',
        type=str,
        help='Token de acesso Meta API (sobrescreve config)'
    )

    # Fair Comparison (HABILITADO POR PADRÃO)
    parser.add_argument(
        '--disable-fair-comparison',
        action='store_true',
        help='Desabilita comparação justa (usa comparação total COM ML vs SEM ML)'
    )

    args = parser.parse_args()

    # Validações
    if args.start_date and not args.end_date:
        parser.error("--start-date requer --end-date")
    if args.end_date and not args.start_date:
        parser.error("--end-date requer --start-date")

    if not args.periodo and not args.start_date:
        parser.error("É necessário especificar --periodo OU --start-date/--end-date")

    return args


def load_config(config_path: str) -> dict:
    """
    Carrega configuração do arquivo YAML.

    Args:
        config_path: Caminho para validation_config.yaml

    Returns:
        Dicionário com configurações
    """
    if not Path(config_path).exists():
        logger.error(f"❌ Arquivo de configuração não encontrado: {config_path}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def print_summary_table(ml_comparison: dict):
    """
    Exibe tabela de comparação ML vs Não-ML no terminal.

    Args:
        ml_comparison: Dict retornado por compare_ml_vs_non_ml()
    """
    com_ml = ml_comparison.get('com_ml', {})
    sem_ml = ml_comparison.get('sem_ml', {})
    diff = ml_comparison.get('difference', {})

    data = [
        ['Total de Leads', f"{com_ml.get('leads', 0):,}", f"{sem_ml.get('leads', 0):,}"],
        ['Conversões', f"{com_ml.get('conversions', 0):,}", f"{sem_ml.get('conversions', 0):,}"],
        ['Taxa Conversão', f"{com_ml.get('conversion_rate', 0):.2f}%", f"{sem_ml.get('conversion_rate', 0):.2f}%"],
        ['Receita Total', f"R$ {com_ml.get('revenue', 0):,.2f}", f"R$ {sem_ml.get('revenue', 0):,.2f}"],
        ['Gasto Total', f"R$ {com_ml.get('spend', 0):,.2f}", f"R$ {sem_ml.get('spend', 0):,.2f}"],
        ['CPL', f"R$ {com_ml.get('cpl', 0):,.2f}", f"R$ {sem_ml.get('cpl', 0):,.2f}"],
        ['ROAS', f"{com_ml.get('roas', 0):.2f}x", f"{sem_ml.get('roas', 0):.2f}x"],
        ['Margem Contrib.', f"R$ {com_ml.get('margin', 0):,.2f}", f"R$ {sem_ml.get('margin', 0):,.2f}"],
    ]

    headers = ['Métrica', 'COM ML', 'SEM ML']
    print(tabulate(data, headers=headers, tablefmt='grid'), flush=True)

    # Mostrar vencedor
    print(flush=True)
    if com_ml.get('roas', 0) > sem_ml.get('roas', 0):
        improvement = diff.get('roas_diff', 0)
        print(f"🏆 VENCEDOR: COM ML (ROAS {improvement:.1f}% maior)", flush=True)
    elif sem_ml.get('roas', 0) > com_ml.get('roas', 0):
        decline = abs(diff.get('roas_diff', 0))
        print(f"⚠️ VENCEDOR: SEM ML (ROAS {decline:.1f}% maior)", flush=True)
    else:
        print("➖ Empate técnico em ROAS", flush=True)


def print_decile_table(decile_metrics):
    """
    Exibe tabela de performance por decil no terminal (Guru vs Guru+TMB).

    Args:
        decile_metrics: DataFrame retornado por DecileMetricsCalculator
    """
    if decile_metrics.empty:
        print("⚠️ Nenhuma métrica de decil disponível", flush=True)
        return

    # Formatar dados para exibição
    table_data = []
    for _, row in decile_metrics.iterrows():
        table_data.append([
            row['decile'],
            row['leads'],
            row['conversions_guru'],
            row['conversions_total'],
            f"{row['conversion_rate_guru']:.2f}%",
            f"{row['conversion_rate_total']:.2f}%",
            f"{row['expected_conversion_rate']:.2f}%",
            f"{row['performance_ratio_guru']:.2f}x",
            f"{row['performance_ratio_total']:.2f}x",
            f"R$ {row['revenue_guru']:,.0f}",
            f"R$ {row['revenue_total']:,.0f}"
        ])

    headers = [
        'Decil', 'Leads',
        'Conv\nGuru', 'Conv\nTotal',
        'Taxa\nGuru', 'Taxa\nTotal',
        'Taxa\nEsperada',
        'Perf\nGuru', 'Perf\nTotal',
        'Receita\nGuru', 'Receita\nTotal'
    ]
    print(tabulate(table_data, headers=headers, tablefmt='grid'), flush=True)

    # Resumo de performance
    total_guru = decile_metrics['revenue_guru'].sum()
    total_tmb_only = decile_metrics['revenue_total'].sum() - total_guru
    print(flush=True)
    print(f"💰 Receita Total Guru: R$ {total_guru:,.2f}", flush=True)
    print(f"💰 Receita Total TMB: R$ {total_tmb_only:,.2f}", flush=True)
    print(f"💰 Receita Total (Guru+TMB): R$ {decile_metrics['revenue_total'].sum():,.2f}", flush=True)


def enrich_campaign_ids(leads_df: pd.DataFrame, account_ids: list, access_token: str) -> pd.DataFrame:
    """
    Enriquece IDs de campanha/adset com nomes reais da Meta API.

    Identifica linhas onde a coluna 'campaign' contém apenas um ID numérico
    e busca o nome real da campanha ou adset na Meta API.

    Args:
        leads_df: DataFrame com leads
        account_ids: Lista de IDs das contas Meta
        access_token: Token de acesso Meta API

    Returns:
        DataFrame com nomes de campanha enriquecidos
    """
    logger.info("   🔍 Procurando IDs de campanha/adset sem nomes...")

    # Identificar linhas com apenas ID numérico
    def is_numeric_id(value):
        if pd.isna(value):
            return False
        value_str = str(value).strip()
        return value_str.isdigit() and len(value_str) > 10  # IDs Meta têm 15+ dígitos

    mask = leads_df['campaign'].apply(is_numeric_id)
    ids_to_enrich = leads_df.loc[mask, 'campaign'].unique()

    if len(ids_to_enrich) == 0:
        logger.info("   ✅ Nenhum ID sem nome encontrado")
        return leads_df

    logger.info(f"   📋 Encontrados {len(ids_to_enrich)} IDs únicos para enriquecer ({mask.sum()} respostas)")

    # Inicializar Meta API
    meta_api = MetaAdsIntegration(access_token=access_token)

    # Mapa ID → Nome
    id_to_name = {}

    for campaign_id in ids_to_enrich:
        # Evitar conversão para float que perde precisão em IDs grandes
        campaign_id_str = str(campaign_id).strip()
        # Remover .0 se houver
        if campaign_id_str.endswith('.0'):
            campaign_id_str = campaign_id_str[:-2]

        try:
            import requests

            # Buscar nome via API direta
            url = f"{meta_api.base_url}/{campaign_id_str}"
            params = {
                'access_token': access_token,
                'fields': 'name'
            }

            response = requests.get(url, params=params, timeout=1)  # Timeout reduzido para 1s

            if response.status_code == 200:
                data = response.json()
                name = data.get('name', campaign_id_str)
                id_to_name[campaign_id] = name
                logger.info(f"      ✅ {campaign_id_str[:15]}... → {name[:60]}...")
            else:
                logger.info(f"      ⚠️ ID {campaign_id_str}: status {response.status_code} (pode ser adset ou campanha de outra conta)")
                id_to_name[campaign_id] = campaign_id_str

        except Exception as e:
            logger.info(f"      ⚠️ Erro ao buscar {campaign_id_str}: {e}")
            id_to_name[campaign_id] = campaign_id_str

    # Atualizar DataFrame
    enriched_count = 0
    for old_id, new_name in id_to_name.items():
        # Converter old_id para string sem perder precisão
        old_id_str = str(old_id).strip()
        if old_id_str.endswith('.0'):
            old_id_str = old_id_str[:-2]

        if new_name != old_id_str:  # Se mudou
            leads_df.loc[leads_df['campaign'] == old_id, 'campaign'] = new_name
            enriched_count += 1

    logger.info(f"   ✅ {enriched_count}/{len(ids_to_enrich)} IDs enriquecidos com sucesso")

    return leads_df


def main():
    """
    Função principal do CLI.
    """
    start_time = time.time()

    print("=" * 80, flush=True)
    print("🚀 SISTEMA DE VALIDAÇÃO DE PERFORMANCE ML - LEAD SCORING", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    # 1. Parse argumentos
    args = parse_args()

    # 1.5. Gerenciar cache se solicitado
    if args.clear_cache:
        import shutil
        cache_dir = Path(__file__).parent.parent.parent / 'files' / 'validation' / 'cache'
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("🗑️  Cache limpo com sucesso!", flush=True)
            print(flush=True)
        else:
            print("⚠️  Nenhum cache encontrado para limpar", flush=True)
            print(flush=True)

    # 2. Carregar configuração
    logger.info(f"⚙️ Carregando configuração de {args.config}...")
    config = load_config(args.config)

    # Sobrescrever com argumentos CLI
    if args.product_value:
        config['product_value'] = args.product_value
    if args.max_match_days:
        config['max_match_days'] = args.max_match_days
    if args.meta_token:
        config['meta_access_token'] = args.meta_token

    # Determinar período
    if args.periodo:
        if args.periodo not in config.get('periodos', {}):
            logger.error(f"❌ Período '{args.periodo}' não encontrado no config")
            sys.exit(1)
        period_config = config['periodos'][args.periodo]
        start_date = period_config['start_date']
        end_date = period_config['end_date']
        period_name = period_config['name']
        logger.info(f"   Período: {period_name} ({start_date} a {end_date})")

        # Usar sales dates do config se não foram especificados via CLI
        if not args.sales_start_date and 'sales_start_date' in period_config:
            args.sales_start_date = period_config['sales_start_date']
            logger.info(f"   📅 Período de vendas do config: {args.sales_start_date} a {period_config.get('sales_end_date')}")
        if not args.sales_end_date and 'sales_end_date' in period_config:
            args.sales_end_date = period_config['sales_end_date']
    else:
        start_date = args.start_date
        end_date = args.end_date
        period_name = f"Período {start_date} a {end_date}"
        logger.info(f"   Período customizado: {start_date} a {end_date}")

    # Determinar caminhos
    if args.leads_path:
        leads_path = args.leads_path
    else:
        # Buscar o arquivo mais recente de leads automaticamente
        leads_dir = Path(config['paths']['leads'])
        pattern = str(leads_dir / '*Pesquisa*.csv')
        matching_files = glob(pattern)

        if not matching_files:
            logger.error(f"❌ Nenhum arquivo de leads encontrado em: {leads_dir}")
            sys.exit(1)

        # Pegar o arquivo mais recente
        leads_path = max(matching_files, key=os.path.getmtime)
        logger.info(f"   📄 Arquivo de leads detectado automaticamente: {Path(leads_path).name}")

    vendas_path = args.vendas_path or config['paths']['vendas']
    output_dir = args.output_dir or 'files/validation/resultados'

    logger.info(f"   Leads: {leads_path}")
    logger.info(f"   Vendas: {vendas_path}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Valor do produto: R$ {config['product_value']:,.2f}")
    logger.info(f"   Janela de matching: {config['max_match_days']} dias")
    print(flush=True)

    # 3. Carregar dados
    print("📂 CARREGANDO DADOS...", flush=True)
    print(flush=True)

    # Leads - usar CAPI + Pesquisa combinados
    capi_loader = CAPILeadDataLoader()
    if not Path(leads_path).exists():
        logger.error(f"❌ Arquivo de leads não encontrado: {leads_path}")
        sys.exit(1)

    # Usar loader combinado que busca Pesquisa + CAPI
    # start_date e end_date já são datetime objects
    leads_df, lead_source_stats = capi_loader.load_combined_leads(
        csv_path=leads_path,
        start_date=start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d'),
        end_date=end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d')
    )
    logger.info(f"   ✅ {len(leads_df)} leads carregados")
    logger.info(f"   📊 Estatísticas: {lead_source_stats['survey_leads']} pesquisa + {lead_source_stats['capi_leads_extras']} CAPI extras")

    # Vendas
    sales_loader = SalesDataLoader()
    # Buscar arquivos Guru com qualquer capitalização: guru_, Guru, GURU
    guru_files = sorted(glob(f"{vendas_path}/guru_*.xlsx")) + sorted(glob(f"{vendas_path}/Guru*.xlsx")) + sorted(glob(f"{vendas_path}/GURU*.xlsx"))
    # Buscar arquivos TMB com qualquer capitalização: tmb_, Tmb, TMB
    tmb_files = sorted(glob(f"{vendas_path}/tmb_*.xlsx")) + sorted(glob(f"{vendas_path}/Tmb*.xlsx")) + sorted(glob(f"{vendas_path}/TMB*.xlsx"))

    logger.info(f"   Arquivos Guru encontrados: {len(guru_files)}")
    logger.info(f"   Arquivos TMB encontrados: {len(tmb_files)}")

    sales_df = sales_loader.combine_sales(
        guru_paths=guru_files if guru_files else None,
        tmb_paths=tmb_files if tmb_files else None
    )

    if sales_df.empty:
        logger.error("❌ Nenhuma venda carregada. Verifique os arquivos de vendas.")
        sys.exit(1)

    logger.info(f"   ✅ {len(sales_df)} vendas carregadas (Guru + TMB)")
    print(flush=True)

    # 4. Filtrar por período
    # Período de vendas pode ser diferente do período de captação
    # Se não foram fornecidos, calcular usando a lógica documentada (3 semanas)
    if args.sales_start_date and args.sales_end_date:
        sales_start = args.sales_start_date
        sales_end = args.sales_end_date
        logger.info(f"   📅 Usando período de vendas customizado: {sales_start} a {sales_end}")
    else:
        # Usar PeriodCalculator para calcular o período de vendas correto
        period_calc = PeriodCalculator()
        calculated_periods = period_calc.calculate_periods(start_date)
        sales_start = calculated_periods['sales']['start']
        sales_end = calculated_periods['sales']['end']
        logger.info(f"   📅 Período de vendas calculado automaticamente: {sales_start} a {sales_end}")

    print(flush=True)
    print(f"📅 FILTRANDO DADOS...", flush=True)
    print(f"   Período de Captação (Leads/Campanhas): {start_date} a {end_date}", flush=True)
    print(f"   Período de Vendas (Matching): {sales_start} a {sales_end}", flush=True)
    print(flush=True)
    leads_df = filter_by_period(leads_df, start_date, end_date, 'data_captura')
    sales_df = filter_by_period(sales_df, sales_start, sales_end, 'sale_date')

    if leads_df.empty:
        logger.error("❌ Nenhum lead no período especificado")
        sys.exit(1)

    # 4.5. Enriquecer IDs de campanha/adset com nomes reais
    print("🔗 ENRIQUECENDO NOMES DE CAMPANHA...", flush=True)
    print(flush=True)
    leads_df = enrich_campaign_ids(leads_df, args.account_id, META_CONFIG['access_token'])

    # 5. Classificar campanhas
    print("🏷️ CLASSIFICANDO CAMPANHAS...", flush=True)
    print(flush=True)
    leads_df, excluded_count = add_ml_classification(leads_df, campaign_col='campaign')

    com_ml_count = len(leads_df[leads_df['ml_type'] == 'COM_ML'])
    sem_ml_count = len(leads_df[leads_df['ml_type'] == 'SEM_ML'])
    logger.info(f"   ✅ COM ML: {com_ml_count} leads ({com_ml_count/len(leads_df)*100:.1f}%)")
    logger.info(f"   ✅ SEM ML: {sem_ml_count} leads ({sem_ml_count/len(leads_df)*100:.1f}%)")
    print(flush=True)

    # 5.5. Fair Campaign Comparison (opcional)
    fair_control_map = {}
    control_id_to_name = {}
    ml_metadata = {}
    if not args.disable_fair_comparison:
        print("🎯 COMPARAÇÃO JUSTA - BUSCANDO CAMPANHAS DE CONTROLE...", flush=True)
        print(flush=True)

        # Usar primeira conta como referência
        primary_account_id = args.account_id[0] if isinstance(args.account_id, list) else args.account_id

        try:
            matcher = FairCampaignMatcher(primary_account_id)

            # Buscar metadata das campanhas ML
            ml_metadata = matcher.get_ml_campaign_metadata(start_date, end_date)

            if ml_metadata:
                # Encontrar campanhas de controle justo
                fair_control_map, control_id_to_name = matcher.find_fair_control_campaigns(
                    ml_metadata,
                    min_creative_overlap=0.8,
                    start_date=start_date,
                    end_date=end_date
                )

                total_matches = sum(len(matches) for matches in fair_control_map.values())
                logger.info(f"   ✅ {len(fair_control_map)} campanhas ML com controles justos")
                logger.info(f"   ✅ {total_matches} campanhas de controle encontradas no total")

                # Adicionar comparison_group aos leads
                if total_matches > 0:
                    leads_df = matcher.create_comparison_groups(
                        leads_df,
                        ml_metadata,
                        fair_control_map,
                        control_id_to_name
                    )
            else:
                logger.warning("   ⚠️ Nenhuma campanha ML encontrada no período")
        except Exception as e:
            logger.error(f"   ❌ Erro na comparação justa: {e}")
            import traceback
            traceback.print_exc()

        print(flush=True)

    # 6. Matching
    print("🔗 VINCULANDO LEADS COM VENDAS...", flush=True)
    print(flush=True)
    matched_df = match_leads_to_sales(
        leads_df,
        sales_df,
        use_temporal_validation=False  # Results analysis mode - match against full history
    )
    matching_stats = get_matching_stats(matched_df, total_sales=len(sales_df))

    logger.info(f"   ✅ Conversões: {matching_stats['total_conversions']}")
    logger.info(f"   ✅ Taxa de conversão geral: {matching_stats['conversion_rate']:.2f}%")
    logger.info(f"   ✅ Match por email: {matching_stats['matched_by_email']}")
    logger.info(f"   ✅ Match por telefone: {matching_stats['matched_by_phone']}")
    print(flush=True)

    # 7. Buscar custos Meta de múltiplas contas
    print("💰 BUSCANDO CUSTOS DAS CAMPANHAS (META API)...", flush=True)
    print(flush=True)

    # Usar o mesmo META_CONFIG que o FairCampaignMatcher usa
    if not META_CONFIG.get('access_token') or META_CONFIG['access_token'] == 'YOUR_ACCESS_TOKEN_HERE':
        logger.warning("⚠️ Meta access token não configurado. Usando spend=0 para todas as campanhas")
        logger.warning("   Configure o token em api/meta_config.py")
        meta_api = None
        costs_hierarchy_consolidated = {'campaigns': {}}
    else:
        try:
            meta_api = MetaAdsIntegration(access_token=META_CONFIG['access_token'])
            logger.info("   ✅ Meta API configurada")

            # Buscar custos de todas as contas e consolidar
            account_ids = args.account_id if isinstance(args.account_id, list) else [args.account_id]
            logger.info(f"   📊 Buscando custos de {len(account_ids)} conta(s) Meta...")

            costs_hierarchy_consolidated = {'campaigns': {}}

            for account_id in account_ids:
                logger.info(f"   🔍 Conta: {account_id}")
                try:
                    costs = meta_api.get_costs_hierarchy(
                        account_id=account_id,
                        since_date=start_date,
                        until_date=end_date
                    )

                    # Consolidar campanhas
                    campaigns_from_account = costs.get('campaigns', {})
                    if campaigns_from_account:
                        costs_hierarchy_consolidated['campaigns'].update(campaigns_from_account)
                        logger.info(f"      ✅ {len(campaigns_from_account)} campanhas encontradas")
                    else:
                        logger.warning(f"      ⚠️ Nenhuma campanha encontrada nesta conta")

                except Exception as e:
                    logger.error(f"      ❌ Erro ao buscar custos da conta {account_id}: {e}")

            total_campaigns = len(costs_hierarchy_consolidated['campaigns'])
            logger.info(f"   ✅ Total consolidado: {total_campaigns} campanhas de todas as contas")

        except Exception as e:
            logger.warning(f"⚠️ Erro ao configurar Meta API: {e}")
            logger.warning("   Usando spend=0 para todas as campanhas")
            meta_api = None
            costs_hierarchy_consolidated = {'campaigns': {}}

    print(flush=True)

    # 8. Calcular métricas
    print("📊 CALCULANDO MÉTRICAS...", flush=True)
    print(flush=True)

    # Por campanha
    use_cache = not args.no_cache  # Usar cache por padrão, desabilitar se --no-cache
    campaign_calc = CampaignMetricsCalculator(
        meta_api if meta_api else None,
        config['product_value'],
        use_cache=use_cache
    )

    if not use_cache:
        logger.info("   ⚠️ Cache desabilitado - forçando busca de dados novos da Meta API")

    # Usar TODAS as contas para buscar leads (não apenas a primeira)
    all_account_ids = ','.join(args.account_id) if isinstance(args.account_id, list) else args.account_id

    campaign_metrics = campaign_calc.calculate_campaign_metrics(
        matched_df,
        all_account_ids,
        start_date,
        end_date,
        global_tracking_rate=matching_stats.get('tracking_rate', 100.0),
        costs_hierarchy_consolidated=costs_hierarchy_consolidated
    )
    logger.info(f"   ✅ Métricas calculadas para {len(campaign_metrics)} campanhas")

    # Adicionar comparison_group ao campaign_metrics (se disponível)
    if 'comparison_group' in matched_df.columns:
        # Criar mapeamento campanha → comparison_group
        campaign_to_group = matched_df.groupby('campaign')['comparison_group'].first().to_dict()
        campaign_metrics['comparison_group'] = campaign_metrics['campaign'].map(campaign_to_group)
        logger.info(f"   ✅ Grupos de comparação adicionados às métricas de campanha")

    # Por decil
    decile_calc = DecileMetricsCalculator()
    decile_metrics = decile_calc.calculate_decile_performance(
        matched_df,
        config['product_value']
    )
    logger.info(f"   ✅ Performance calculada para todos os decis (D1-D10)")

    # Comparação ML
    ml_comparison = compare_ml_vs_non_ml(campaign_metrics)

    # Estatísticas gerais (usando TODAS as vendas do período, não apenas matched)
    overall_stats = calculate_overall_stats(
        matched_df,
        campaign_metrics,
        lead_period=(start_date, end_date),
        sales_period=(sales_start, sales_end),
        sales_df=sales_df,  # Todas as vendas do período
        product_value=config['product_value'],
        excluded_leads=excluded_count,
        campaign_calc=campaign_calc,  # Para acessar total_leads_meta_before_filter
        lead_source_stats=lead_source_stats  # Estatísticas de pesquisa vs CAPI
    )

    # Comparação por grupo (quando fair comparison estiver habilitado)
    comparison_group_metrics = None
    fair_comparison_info = None
    if not args.disable_fair_comparison and 'comparison_group' in matched_df.columns:
        comparison_group_metrics = calculate_comparison_group_metrics(matched_df, campaign_metrics)

        # Preparar informações dos matches para o relatório
        fair_comparison_info = {
            'ml_metadata': ml_metadata,
            'fair_control_map': fair_control_map,
            'control_id_to_name': control_id_to_name
        }

    print(flush=True)

    # 9. EXIBIR RESUMO NO TERMINAL
    print("=" * 80, flush=True)
    print("📊 RESUMO EXECUTIVO - COMPARAÇÃO ML vs NÃO-ML", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    print_summary_table(ml_comparison)

    print(flush=True)
    print("=" * 80, flush=True)
    print("📈 PERFORMANCE POR DECIL (Real vs Esperado)", flush=True)
    print("=" * 80, flush=True)
    print("IMPORTANTE: Modelo treinado APENAS com vendas Guru", flush=True)
    print("→ Guru = Dados de treinamento | Total = Guru + TMB (generalização)", flush=True)
    print(flush=True)
    print_decile_table(decile_metrics)

    print(flush=True)

    # 9.5. Exibir métricas por campanha
    print("=" * 80, flush=True)
    print("📊 MÉTRICAS DETALHADAS POR CAMPANHA", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    # Formatar nome das campanhas
    def format_campaign_name(row):
        campaign = str(row['campaign'])

        # Identificador ML/não-ML
        if 'MACHINE LEARNING' in campaign:
            prefix = '[ML]'
        elif 'ESCALA SCORE' in campaign:
            prefix = '[ESCALA]'
        elif 'FAIXA A' in campaign:
            prefix = '[FAIXA-A]'
        elif 'FAIXA B' in campaign:
            prefix = '[FAIXA-B]'
        elif 'FAIXA C' in campaign:
            prefix = '[FAIXA-C]'
        else:
            prefix = '[OUTRO]'

        # Data
        parts = campaign.split('|')
        date_part = parts[-1].strip()[:10] if len(parts) > 1 and '2025' in parts[-1] else ''

        # Tipo/Temperatura
        tipo = 'CAP' if 'CAP' in campaign else 'RET' if 'RET' in campaign else ''
        temp = 'FRIO' if 'FRIO' in campaign else 'MORNO' if 'MORNO' in campaign else ''

        desc = f'{tipo}/{temp}' if tipo and temp else tipo if tipo else temp if temp else ''

        return f'{prefix:10} {desc:10} {date_part:10}'.strip()

    campaign_display = campaign_metrics.copy()
    campaign_display['brief_name'] = campaign_display.apply(format_campaign_name, axis=1)

    # Ordenar por ROAS
    campaign_display = campaign_display.sort_values('roas', ascending=False)

    # Preparar dados para exibição
    display_data = []
    for _, row in campaign_display.iterrows():
        display_data.append([
            row['ml_type'],
            row['brief_name'],
            f"{row['leads']:,}",
            f"{row['conversions']:,}",
            f"{row['conversion_rate']:.2f}%",  # Já está em porcentagem
            f"R$ {row['total_revenue']:,.0f}",
            f"R$ {row['spend']:,.0f}",
            f"R$ {row['cpl']:.2f}",
            f"{row['roas']:.2f}x",
            f"R$ {row['contribution_margin']:,.0f}"
        ])

    headers = ['Tipo', 'Campanha', 'Leads', 'Conv', 'Taxa', 'Receita', 'Gasto', 'CPL', 'ROAS', 'Margem']
    print(tabulate(display_data, headers=headers, tablefmt='grid'), flush=True)
    print(flush=True)
    print(f"Total de campanhas: {len(campaign_display)}", flush=True)
    print(flush=True)

    # 10. Gerar relatório Excel
    print("📄 Gerando relatório Excel...", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    # Sempre usar nome com datas (sobrescreve se mesmo período)
    excel_filename = f"validation_report_{start_date}_to_{end_date}.xlsx"
    excel_path = str(Path(output_dir) / excel_filename)

    if Path(excel_path).exists():
        logger.info(f"   📌 Sobrescrevendo relatório existente: {excel_filename}")
    else:
        logger.info(f"   📌 Criando novo relatório: {excel_filename}")

    # Formatar account IDs para exibição
    account_ids_display = ', '.join(args.account_id) if isinstance(args.account_id, list) else args.account_id

    config_params = {
        'Período': period_name,
        'Data Início': start_date,
        'Data Fim': end_date,
        'Valor do Produto': f"R$ {config['product_value']:,.2f}",
        'Janela de Matching': f"{config['max_match_days']} dias",
        'Account IDs': account_ids_display,
        'Total de Leads': len(leads_df),
        'Total de Conversões': matching_stats['total_conversions'],
        'Gerado em': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }

    report_gen = ValidationReportGenerator()
    report_gen.generate_excel_report(
        campaign_metrics,
        decile_metrics,
        ml_comparison,
        matching_stats,
        overall_stats,
        config_params,
        excel_path,
        comparison_group_metrics=comparison_group_metrics,
        fair_comparison_info=fair_comparison_info
    )
    print(f"   ✅ Excel salvo: {excel_path}", flush=True)
    print(flush=True)

    # 11. Gerar gráficos
    # DESABILITADO: Gerando apenas análise em console até finalizar formato
    # print("📈 Gerando visualizações...")
    # viz = ValidationVisualizer()
    # viz.generate_all_charts(
    #     campaign_metrics,
    #     decile_metrics,
    #     ml_comparison,
    #     output_dir
    # )
    # print()

    # 12. Finalização
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("=" * 80, flush=True)
    print("✅ VALIDAÇÃO CONCLUÍDA COM SUCESSO!", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    print(f"📊 Análise exibida no console acima", flush=True)
    print(f"📄 Excel atualizado: {excel_path}", flush=True)
    print(f"⏱️  Tempo de execução: {elapsed_time:.1f} segundos ({elapsed_time/60:.1f} minutos)", flush=True)
    print(flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Operação cancelada pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
