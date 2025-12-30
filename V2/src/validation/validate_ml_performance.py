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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
from src.validation.period_calculator import PeriodCalculator
from src.validation.meta_reports_loader import MetaReportsLoader

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

    # Nível de Comparação - Evento ML
    parser.add_argument(
        '--comparison-level',
        type=str,
        choices=['adsets_iguais', 'todos', 'both'],
        default='both',
        help='Nível de comparação: adsets_iguais (apenas ADV estrutura idêntica), todos (todas campanhas ML), both (gera ambos) - default: both'
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
    # Buscar arquivos Guru com qualquer capitalização e formato: guru*, Guru*, GURU*
    guru_files = sorted(glob(f"{vendas_path}/[Gg][Uu][Rr][Uu]*.xlsx"))
    # Buscar arquivos TMB com qualquer capitalização e formato: tmb*, Tmb*, TMB*
    tmb_files = sorted(glob(f"{vendas_path}/[Tt][Mm][Bb]*.xlsx"))

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
    # DESABILITADO: Usando MetaReportsLoader ao invés de API
    # print("🔗 ENRIQUECENDO NOMES DE CAMPANHA...", flush=True)
    # print(flush=True)
    # leads_df = enrich_campaign_ids(leads_df, args.account_id, META_CONFIG['access_token'])

    # 5. Classificar campanhas
    print("🏷️ CLASSIFICANDO CAMPANHAS...", flush=True)
    print(flush=True)
    leads_df, excluded_count = add_ml_classification(leads_df, campaign_col='campaign')

    com_ml_count = len(leads_df[leads_df['ml_type'] == 'COM_ML'])
    sem_ml_count = len(leads_df[leads_df['ml_type'] == 'SEM_ML'])
    logger.info(f"   ✅ COM ML: {com_ml_count} leads ({com_ml_count/len(leads_df)*100:.1f}%)")
    logger.info(f"   ✅ SEM ML: {sem_ml_count} leads ({sem_ml_count/len(leads_df)*100:.1f}%)")
    print(flush=True)

    # 5.5. Carregar relatórios Meta para criar grupos de comparação refinados
    print("💰 CARREGANDO RELATÓRIOS META PARA CLASSIFICAÇÃO...", flush=True)
    print(flush=True)

    # Carregar relatórios Meta locais
    # IMPORTANTE: Usar pasta específica com relatórios oficiais do período (não adsets_analysis)
    reports_dir = 'files/validation/meta_reports/02:12 - 08:12'
    loader = MetaReportsLoader(reports_dir)
    costs_hierarchy_temp = loader.build_costs_hierarchy(start_date, end_date)

    # Obter DataFrame de campanhas
    reports = loader.load_all_reports(start_date, end_date)
    campaigns_df = reports.get('campaigns', pd.DataFrame())

    # 5.6. Criar grupos de comparação REFINADOS (distingue Eventos ML vs Otimização ML)
    print("🎯 CRIANDO GRUPOS DE COMPARAÇÃO...", flush=True)
    print(flush=True)

    comparison_group_map_15 = {}  # Mapa com IDs de 15 dígitos

    if 'ml_type' in leads_df.columns and not campaigns_df.empty:
        # Identificar campanhas ML vs Controle dinamicamente dos relatórios
        # IMPORTANTE: Também incluir "| ML |" para nomes truncados (ex: "ADV | ML | S/ ABERTO")
        ml_campaigns = campaigns_df[campaigns_df['campaign_name'].str.contains('MACHINE LEARNING|\\| ML \\|', case=False, na=False, regex=True)]
        ml_campaign_ids = ml_campaigns['campaign_id'].unique().tolist()

        control_campaigns = campaigns_df[
            campaigns_df['campaign_name'].str.contains('ESCALA SCORE|FAIXA A|\\bSCORE\\b', case=False, na=False, regex=True)
        ]
        control_campaign_ids = control_campaigns['campaign_id'].unique().tolist()

        if ml_campaign_ids and control_campaign_ids:
            # Usar função refinada que distingue Eventos ML vs Otimização ML
            from src.validation.fair_campaign_comparison import create_refined_campaign_map

            comparison_group_map_15 = create_refined_campaign_map(
                campaigns_df=campaigns_df,
                ml_campaign_ids=ml_campaign_ids,
                control_campaign_ids=control_campaign_ids
            )

            # Mapear leads para grupos refinados usando campaign_id (primeiros 15 dígitos)
            def map_to_refined_group(row):
                # Tentar obter campaign_id de várias fontes
                campaign_id = None

                # Fonte 1: campaign_id_meta (leads CAPI)
                if pd.notna(row.get('campaign_id_meta')):
                    campaign_id = str(row['campaign_id_meta'])

                # Fonte 2: Extrair do nome da campanha (formato: "nome|ID")
                elif pd.notna(row.get('campaign')):
                    campaign_str = str(row['campaign'])
                    # Procurar por ID de 18 dígitos após o último "|"
                    if '|' in campaign_str:
                        parts = campaign_str.split('|')
                        last_part = parts[-1].strip()
                        # Verificar se é um ID numérico de 18 dígitos
                        if last_part.isdigit() and len(last_part) == 18:
                            campaign_id = last_part

                # Se conseguimos um campaign_id, mapear usando os primeiros 15 dígitos
                if campaign_id:
                    cid_15 = campaign_id[:15]
                    grupo = comparison_group_map_15.get(cid_15)
                    if grupo:
                        return grupo

                # Fallback para ml_type
                if row.get('ml_type') == 'SEM_ML':
                    return 'Controle'
                elif row.get('ml_type') == 'COM_ML':
                    return 'Eventos ML'  # Apenas se não conseguimos o ID
                else:
                    return 'Outro'

            leads_df['comparison_group'] = leads_df.apply(map_to_refined_group, axis=1)

            group_counts = leads_df['comparison_group'].value_counts()
            logger.info(f"   ✅ Grupos refinados criados:")
            for group, count in group_counts.items():
                logger.info(f"      {group}: {count} leads")
        else:
            # Fallback: usar mapeamento simples
            logger.warning("   ⚠️ Não foi possível criar mapeamento refinado, usando simples")
            leads_df['comparison_group'] = leads_df['ml_type'].map({
                'COM_ML': 'Eventos ML',
                'SEM_ML': 'Controle'
            }).fillna('Outro')
    else:
        logger.warning("   ⚠️ Coluna ml_type não encontrada, pulando criação de grupos")

    print(flush=True)

    # 6. Matching
    print("🔗 VINCULANDO LEADS COM VENDAS...", flush=True)
    print(flush=True)
    matched_df = match_leads_to_sales(
        leads_df,
        sales_df,
        use_temporal_validation=False  # Results analysis mode - match against full history
    )

    # 6.1. Filtrar conversões por período de captura
    print("📅 FILTRANDO CONVERSÕES POR PERÍODO DE CAPTURA...", flush=True)
    print(flush=True)
    from src.validation.matching import filter_conversions_by_capture_period
    matched_df = filter_conversions_by_capture_period(
        matched_df,
        period_start=start_date,
        period_end=end_date
    )

    # 6.2. Remover duplicatas artificiais
    print("🧹 REMOVENDO DUPLICATAS ARTIFICIAIS...", flush=True)
    print(flush=True)
    from src.validation.matching import deduplicate_conversions
    matched_df = deduplicate_conversions(matched_df)

    matching_stats = get_matching_stats(matched_df, total_sales=len(sales_df))

    logger.info(f"   ✅ Conversões: {matching_stats['total_conversions']}")
    logger.info(f"   ✅ Taxa de conversão geral: {matching_stats['conversion_rate']:.2f}%")
    logger.info(f"   ✅ Match por email: {matching_stats['matched_by_email']}")
    logger.info(f"   ✅ Match por telefone: {matching_stats['matched_by_phone']}")
    print(flush=True)

    # 7. Reutilizar custos dos relatórios Meta já carregados
    print("💰 REUTILIZANDO CUSTOS DOS RELATÓRIOS META...", flush=True)
    print(flush=True)

    meta_api = None  # Não usar API, apenas relatórios locais
    costs_hierarchy_consolidated = costs_hierarchy_temp  # Reutilizar dados já carregados

    num_campaigns = len(costs_hierarchy_consolidated.get('campaigns', {}))
    if num_campaigns > 0:
        logger.info(f"   ✅ {num_campaigns} campanhas reutilizadas dos relatórios")
    else:
        logger.warning("   ⚠️ Nenhuma campanha encontrada nos relatórios")

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
    if 'comparison_group' in matched_df.columns and len(campaign_metrics) > 0:
        # Criar mapeamento campanha → comparison_group
        campaign_to_group = matched_df.groupby('campaign')['comparison_group'].first().to_dict()
        campaign_metrics['comparison_group'] = campaign_metrics['campaign'].map(campaign_to_group)
        logger.info(f"   ✅ Grupos de comparação adicionados às métricas de campanha")
    elif len(campaign_metrics) == 0:
        logger.warning("   ⚠️ Nenhuma métrica de campanha disponível - DataFrame vazio")

    # Por decil
    decile_calc = DecileMetricsCalculator()
    decile_metrics = decile_calc.calculate_decile_performance(
        matched_df,
        config['product_value']
    )
    logger.info(f"   ✅ Performance calculada para todos os decis (D1-D10)")

    # Comparação ML
    ml_comparison = compare_ml_vs_non_ml(campaign_metrics) if len(campaign_metrics) > 0 else None

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

    # Comparação por grupo
    comparison_group_metrics = None
    if 'comparison_group' in matched_df.columns and len(campaign_metrics) > 0:
        comparison_group_metrics = calculate_comparison_group_metrics(matched_df, campaign_metrics)
        logger.info(f"   ✅ Métricas calculadas por grupo de comparação")

    # Fair comparison info (legacy - não usado mais)
    fair_comparison_info = None

    # 8.5. COMPARAÇÕES DE ADSETS E ADS (usando relatórios locais)
    all_adsets_comparison = None
    adset_level_comparisons = None
    ad_level_comparisons = None
    ad_in_matched_adsets_comparisons = None
    matched_ads_in_matched_adsets_comparisons = None

    if not args.disable_fair_comparison and len(campaign_metrics) > 0:
        try:
            from src.validation.fair_campaign_comparison import (
                compare_all_adsets_performance,
                identify_matched_adset_pairs,
                compare_adset_performance,
                compare_ads_in_matched_adsets,
                compare_matched_ads_in_matched_adsets
            )

            print("\n📊 COMPARAÇÃO DE ADSETS E ADS (relatórios locais)...", flush=True)
            print(flush=True)

            # Carregar adsets e ads dos relatórios Meta locais
            reports = loader.load_all_reports(start_date, end_date)
            adsets_df = reports.get('adsets', pd.DataFrame())
            ads_df = reports.get('ads', pd.DataFrame())

            if not adsets_df.empty:
                logger.info(f"   ✅ {len(adsets_df)} adsets carregados dos relatórios")

                # DEBUG: Verificar se total_spend existe
                if 'total_spend' in adsets_df.columns:
                    total_spend_sum = adsets_df['total_spend'].sum()
                    spend_sum = adsets_df['spend'].sum()
                    logger.info(f"   📊 DEBUG: total_spend existe em adsets_df")
                    logger.info(f"      Total spend (histórico): R$ {total_spend_sum:,.2f}")
                    logger.info(f"      Total spend (filtrado): R$ {spend_sum:,.2f}")
                else:
                    logger.warning(f"   ⚠️ DEBUG: total_spend NÃO existe em adsets_df")

                # Extrair campaign IDs do matched_df
                def extract_campaign_id(campaign_name):
                    if pd.isna(campaign_name) or not isinstance(campaign_name, str):
                        return None
                    parts = campaign_name.split('|')
                    if len(parts) >= 2:
                        campaign_id = parts[-1].strip()
                        # Validar que é numérico e tem tamanho de ID da Meta
                        if campaign_id.isdigit() and len(campaign_id) >= 15:
                            return campaign_id
                    return None

                matched_df['campaign_id'] = matched_df['campaign'].apply(extract_campaign_id)

                # Reutilizar comparison_group_map criado anteriormente
                comparison_group_map = comparison_group_map_15  # Já criado na seção 5.6

                # Extrair IDs de campanhas por grupo (Eventos ML, Otimização ML, Controle)
                eventos_ml_campaign_ids = []
                otimizacao_ml_campaign_ids = []
                control_campaign_ids = []

                if comparison_group_map:
                    for cid_15, group in comparison_group_map.items():
                        # Buscar o ID completo (18 dígitos) nos relatórios
                        matching_campaigns = campaigns_df[campaigns_df['campaign_id'].astype(str).str.startswith(cid_15)]
                        if not matching_campaigns.empty:
                            full_id = matching_campaigns.iloc[0]['campaign_id']
                            if group == 'Eventos ML':
                                eventos_ml_campaign_ids.append(full_id)
                            elif group == 'Otimização ML':
                                otimizacao_ml_campaign_ids.append(full_id)
                            elif group == 'Controle':
                                control_campaign_ids.append(full_id)

                    logger.info(f"   📊 Campanhas por grupo:")
                    logger.info(f"      Eventos ML: {len(eventos_ml_campaign_ids)}")
                    logger.info(f"      Otimização ML: {len(otimizacao_ml_campaign_ids)}")
                    logger.info(f"      Controle: {len(control_campaign_ids)}")

                    # Criar ml_type_map para compatibilidade (COM_ML para Eventos e Otimização)
                    ml_type_map = {}
                    for cid_15, group in comparison_group_map.items():
                        if group in ['Eventos ML', 'Otimização ML']:
                            ml_type_map[cid_15] = 'COM_ML'
                        elif group == 'Controle':
                            ml_type_map[cid_15] = 'SEM_ML'
                else:
                    ml_type_map = {}
                    logger.warning("   ⚠️ comparison_group_map vazio, não será possível fazer comparação")

                # 1. Comparação de TODOS os adsets (Eventos ML vs Controle)
                all_adsets_comparison = compare_all_adsets_performance(
                    adsets_df=adsets_df,
                    matched_df=matched_df,
                    comparison_group_map=comparison_group_map,
                    product_value=config['product_value'],
                    min_spend=0.0,
                    config=config
                )
                logger.info(f"   ✅ Comparação de todos adsets concluída")

                # 2. Identificar matched adset pairs (Eventos ML vs Controle apenas)
                # IMPORTANTE: Usar apenas eventos_ml_campaign_ids (excluir Otimização ML)
                matched_adsets, matched_adsets_df = identify_matched_adset_pairs(
                    adsets_df=adsets_df,
                    ml_campaign_ids=eventos_ml_campaign_ids,  # Apenas Eventos ML!
                    control_campaign_ids=control_campaign_ids,
                    min_spend=0.0
                )

                if matched_adsets:
                    logger.info(f"   ✅ {len(matched_adsets)} adsets matched identificados (Eventos ML vs Controle)")

                    # IMPORTANTE: Usar matched_adsets_df retornado por identify_matched_adset_pairs
                    # Este DataFrame já tem a coluna 'leads' criada a partir de 'leads_standard'
                    logger.info(f"\n   🔍 DEBUG - Usando matched_adsets_df com 'leads' column:")
                    logger.info(f"      Total de adsets: {len(matched_adsets_df)}")
                    logger.info(f"      Tem 'leads'? {'leads' in matched_adsets_df.columns}")
                    if 'leads' in matched_adsets_df.columns:
                        logger.info(f"      Total de leads: {matched_adsets_df['leads'].sum():.0f}")

                    if not matched_adsets_df.empty:
                        adset_level_comparisons = compare_adset_performance(
                            adsets_metrics_df=matched_adsets_df,  # Usar DF com 'leads' já criado!
                            matched_df=matched_df,
                            ml_type_map=ml_type_map,
                            product_value=config['product_value'],
                            comparison_group_map=comparison_group_map
                        )
                        logger.info(f"   ✅ Comparação de matched adsets concluída ({len(matched_adsets_df)} adsets)")
                    else:
                        logger.warning("   ⚠️ Nenhum adset matched encontrado após filtragem")
                        adset_level_comparisons = None
                else:
                    logger.warning("   ⚠️ Nenhum matched adset identificado")
                    adset_level_comparisons = None

                # COMENTADO: Comparação de ads desabilitada temporariamente
                # # 3. Comparar TODOS os ads (se houver ads_df)
                # if not ads_df.empty:
                #     from src.validation.fair_campaign_comparison import compare_ad_performance
                #
                #     # Preparar ads_df: adicionar ml_type usando primeiros 15 dígitos
                #     ads_df_prep = ads_df.copy()
                #     ads_df_prep['campaign_id_15'] = ads_df_prep['campaign_id'].astype(str).str[:15]
                #     ads_df_prep['ml_type'] = ads_df_prep['campaign_id_15'].map(ml_type_map)
                #
                #     # Criar ml_type_map expandido com IDs completos (18 dígitos)
                #     # para compatibilidade com compare_ad_performance
                #     ml_type_map_full = {}
                #     for _, row in ads_df_prep[['campaign_id', 'ml_type']].drop_duplicates().iterrows():
                #         if pd.notna(row['ml_type']):
                #             ml_type_map_full[row['campaign_id']] = row['ml_type']
                #
                #     ad_level_comparisons = compare_ad_performance(
                #         ad_metrics_df=ads_df,
                #         matched_df=matched_df,
                #         ml_type_map=ml_type_map_full
                #     )
                #
                #     logger.info(f"   ✅ Comparação de ads concluída ({len(ad_level_comparisons.get('detailed_matched', pd.DataFrame()))} linhas)")
                #
                # # 4. Comparar ads dentro dos matched adsets (reutilizar matched_adsets já identificado)
                # if matched_adsets and not ads_df.empty:
                #     try:
                #         ad_in_matched_adsets_comparisons = compare_ads_in_matched_adsets(
                #             ad_metrics_df=ads_df,
                #             matched_df=matched_df,
                #             ml_type_map=ml_type_map_full if 'ml_type_map_full' in locals() else ml_type_map,
                #             product_value=config['product_value'],
                #             comparison_group_map=comparison_group_map,
                #             filtered_matched_adsets=matched_adsets
                #         )
                #         logger.info(f"   ✅ Comparação de ads em adsets matched concluída")
                #     except Exception as e:
                #         logger.warning(f"   ⚠️  Erro na comparação de ads em adsets matched: {e}")
                #         ad_in_matched_adsets_comparisons = None
                #
                #     # 6. Comparar matched ads dentro dos matched adsets
                #     try:
                #         matched_ads_in_matched_adsets_comparisons = compare_matched_ads_in_matched_adsets(
                #             ad_metrics_df=ads_df,
                #             matched_df=matched_df,
                #             ml_type_map=ml_type_map_full if 'ml_type_map_full' in locals() else ml_type_map,
                #             product_value=config['product_value'],
                #             comparison_group_map=comparison_group_map,
                #             filtered_matched_adsets=matched_adsets
                #         )
                #         logger.info(f"   ✅ Comparação de matched ads em adsets matched concluída")
                #     except Exception as e:
                #         logger.warning(f"   ⚠️  Erro na comparação de matched ads em adsets matched: {e}")
                #         matched_ads_in_matched_adsets_comparisons = None
                # else:
                #     logger.warning("   ⚠️ Nenhum adset matched encontrado (comparações específicas não disponíveis)")
            else:
                logger.warning("   ⚠️ Nenhum adset carregado dos relatórios")

        except Exception as e:
            logger.error(f"   ❌ Erro na comparação de adsets/ads: {e}")
            import traceback
            traceback.print_exc()

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
        fair_comparison_info=fair_comparison_info,
        matched_df=matched_df,
        sales_df=sales_df,
        all_adsets_comparison=all_adsets_comparison,
        adset_level_comparisons=adset_level_comparisons,
        ad_level_comparisons=ad_level_comparisons,
        ad_in_matched_adsets_comparisons=ad_in_matched_adsets_comparisons,
        matched_ads_in_matched_adsets_comparisons=matched_ads_in_matched_adsets_comparisons
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
