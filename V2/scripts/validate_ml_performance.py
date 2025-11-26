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

# Adicionar V2/ ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports dos módulos de validação
from src.validation.data_loader import LeadDataLoader, SalesDataLoader
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
    calculate_overall_stats
)
from src.validation.report_generator import ValidationReportGenerator
from src.validation.visualization import ValidationVisualizer

# Imports de integrações existentes
from api.meta_integration import MetaAdsIntegration

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

    # Meta Ads API
    parser.add_argument(
        '--account-id',
        type=str,
        required=True,
        help='ID da conta Meta Ads (ex: act_123456789)'
    )

    # Caminhos
    parser.add_argument(
        '--leads-path',
        type=str,
        help='Caminho para CSV de leads (default: validation/leads/leads_completo.csv)'
    )

    parser.add_argument(
        '--vendas-path',
        type=str,
        help='Caminho para pasta com arquivos de vendas (default: validation/vendas/)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Diretório de saída (default: validation/resultados/)'
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

    # Meta Access Token
    parser.add_argument(
        '--meta-token',
        type=str,
        help='Token de acesso Meta API (sobrescreve config)'
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
    print(tabulate(data, headers=headers, tablefmt='grid'))

    # Mostrar vencedor
    print()
    if com_ml.get('roas', 0) > sem_ml.get('roas', 0):
        improvement = diff.get('roas_diff', 0)
        print(f"🏆 VENCEDOR: COM ML (ROAS {improvement:.1f}% maior)")
    elif sem_ml.get('roas', 0) > com_ml.get('roas', 0):
        decline = abs(diff.get('roas_diff', 0))
        print(f"⚠️ VENCEDOR: SEM ML (ROAS {decline:.1f}% maior)")
    else:
        print("➖ Empate técnico em ROAS")


def print_decile_table(decile_metrics):
    """
    Exibe tabela de performance por decil no terminal (Guru vs Guru+TMB).

    Args:
        decile_metrics: DataFrame retornado por DecileMetricsCalculator
    """
    if decile_metrics.empty:
        print("⚠️ Nenhuma métrica de decil disponível")
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
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Resumo de performance
    total_guru = decile_metrics['revenue_guru'].sum()
    total_tmb_only = decile_metrics['revenue_total'].sum() - total_guru
    print()
    print(f"💰 Receita Total Guru: R$ {total_guru:,.2f}")
    print(f"💰 Receita Total TMB: R$ {total_tmb_only:,.2f}")
    print(f"💰 Receita Total (Guru+TMB): R$ {decile_metrics['revenue_total'].sum():,.2f}")


def main():
    """
    Função principal do CLI.
    """
    print("=" * 80)
    print("🚀 SISTEMA DE VALIDAÇÃO DE PERFORMANCE ML - LEAD SCORING")
    print("=" * 80)
    print()

    # 1. Parse argumentos
    args = parse_args()

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
    else:
        start_date = args.start_date
        end_date = args.end_date
        period_name = f"Período {start_date} a {end_date}"
        logger.info(f"   Período customizado: {start_date} a {end_date}")

    # Determinar caminhos
    leads_path = args.leads_path or str(Path(config['paths']['leads']) / 'leads_completo.csv')
    vendas_path = args.vendas_path or config['paths']['vendas']
    output_dir = args.output_dir or 'validation/resultados'

    logger.info(f"   Leads: {leads_path}")
    logger.info(f"   Vendas: {vendas_path}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Valor do produto: R$ {config['product_value']:,.2f}")
    logger.info(f"   Janela de matching: {config['max_match_days']} dias")
    print()

    # 3. Carregar dados
    print("📂 CARREGANDO DADOS...")
    print()

    # Leads
    lead_loader = LeadDataLoader()
    if not Path(leads_path).exists():
        logger.error(f"❌ Arquivo de leads não encontrado: {leads_path}")
        sys.exit(1)
    leads_df = lead_loader.load_leads_csv(leads_path)
    logger.info(f"   ✅ {len(leads_df)} leads carregados")

    # Vendas
    sales_loader = SalesDataLoader()
    guru_files = sorted(glob(f"{vendas_path}/guru_*.xlsx")) + sorted(glob(f"{vendas_path}/GURU*.xlsx"))
    tmb_files = sorted(glob(f"{vendas_path}/tmb_*.xlsx")) + sorted(glob(f"{vendas_path}/TMB*.xlsx"))

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
    print()

    # 4. Filtrar por período
    print(f"📅 FILTRANDO POR PERÍODO ({start_date} a {end_date})...")
    print()
    leads_df = filter_by_period(leads_df, start_date, end_date, 'data_captura')
    sales_df = filter_by_period(sales_df, start_date, end_date, 'sale_date')

    if leads_df.empty:
        logger.error("❌ Nenhum lead no período especificado")
        sys.exit(1)

    # 5. Classificar campanhas
    print("🏷️ CLASSIFICANDO CAMPANHAS...")
    print()
    leads_df = add_ml_classification(leads_df, campaign_col='campaign')

    com_ml_count = len(leads_df[leads_df['ml_type'] == 'COM_ML'])
    sem_ml_count = len(leads_df[leads_df['ml_type'] == 'SEM_ML'])
    logger.info(f"   ✅ COM ML: {com_ml_count} leads ({com_ml_count/len(leads_df)*100:.1f}%)")
    logger.info(f"   ✅ SEM ML: {sem_ml_count} leads ({sem_ml_count/len(leads_df)*100:.1f}%)")
    print()

    # 6. Matching
    print("🔗 VINCULANDO LEADS COM VENDAS...")
    print()
    matched_df = match_leads_to_sales(
        leads_df,
        sales_df,
        max_days_window=config['max_match_days']
    )
    matching_stats = get_matching_stats(matched_df)

    logger.info(f"   ✅ Conversões: {matching_stats['total_conversions']}")
    logger.info(f"   ✅ Taxa de conversão geral: {matching_stats['conversion_rate']:.2f}%")
    logger.info(f"   ✅ Match por email: {matching_stats['matched_by_email']}")
    logger.info(f"   ✅ Match por telefone: {matching_stats['matched_by_phone']}")
    print()

    # 7. Buscar custos Meta
    print("💰 BUSCANDO CUSTOS DAS CAMPANHAS (META API)...")
    print()

    if not config.get('meta_access_token') or config['meta_access_token'] == 'EAAV...':
        logger.warning("⚠️ Meta access token não configurado. Usando spend=0 para todas as campanhas")
        logger.warning("   Configure o token em configs/validation_config.yaml")
        meta_api = None
    else:
        try:
            meta_api = MetaAdsIntegration(access_token=config['meta_access_token'])
            logger.info("   ✅ Meta API configurada")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao configurar Meta API: {e}")
            logger.warning("   Usando spend=0 para todas as campanhas")
            meta_api = None

    print()

    # 8. Calcular métricas
    print("📊 CALCULANDO MÉTRICAS...")
    print()

    # Por campanha
    campaign_calc = CampaignMetricsCalculator(
        meta_api if meta_api else None,
        config['product_value']
    )
    campaign_metrics = campaign_calc.calculate_campaign_metrics(
        matched_df,
        args.account_id,
        start_date,
        end_date
    )
    logger.info(f"   ✅ Métricas calculadas para {len(campaign_metrics)} campanhas")

    # Por decil
    decile_calc = DecileMetricsCalculator()
    decile_metrics = decile_calc.calculate_decile_performance(
        matched_df,
        config['product_value']
    )
    logger.info(f"   ✅ Performance calculada para todos os decis (D1-D10)")

    # Comparação ML
    ml_comparison = compare_ml_vs_non_ml(campaign_metrics)

    # Estatísticas gerais
    overall_stats = calculate_overall_stats(matched_df, campaign_metrics)

    print()

    # 9. EXIBIR RESUMO NO TERMINAL
    print("=" * 80)
    print("📊 RESUMO EXECUTIVO - COMPARAÇÃO ML vs NÃO-ML")
    print("=" * 80)
    print()
    print_summary_table(ml_comparison)

    print()
    print("=" * 80)
    print("📈 PERFORMANCE POR DECIL (Real vs Esperado)")
    print("=" * 80)
    print("IMPORTANTE: Modelo treinado APENAS com vendas Guru")
    print("→ Guru = Dados de treinamento | Total = Guru + TMB (generalização)")
    print()
    print_decile_table(decile_metrics)

    print()

    # 10. Gerar relatório Excel
    print("📄 Gerando relatório Excel...")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"validation_report_{timestamp}.xlsx"
    excel_path = str(Path(output_dir) / excel_filename)

    config_params = {
        'Período': period_name,
        'Data Início': start_date,
        'Data Fim': end_date,
        'Valor do Produto': f"R$ {config['product_value']:,.2f}",
        'Janela de Matching': f"{config['max_match_days']} dias",
        'Account ID': args.account_id,
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
        excel_path
    )
    logger.info(f"   ✅ Excel salvo: {excel_path}")
    print()

    # 11. Gerar gráficos
    print("📈 Gerando visualizações...")
    viz = ValidationVisualizer()
    viz.generate_all_charts(
        campaign_metrics,
        decile_metrics,
        ml_comparison,
        output_dir
    )
    print()

    # 12. Finalização
    print("=" * 80)
    print("✅ VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 80)
    print()
    print(f"📁 Arquivos gerados em: {output_dir}/")
    print(f"   - {excel_filename}")
    print(f"   - conversion_rate_comparison.png")
    print(f"   - roas_comparison.png")
    print(f"   - decile_performance.png")
    print(f"   - cumulative_revenue_by_decile.png")
    print(f"   - contribution_margin_by_campaign.png")
    print()


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
