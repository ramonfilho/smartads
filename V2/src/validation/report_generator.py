"""
Módulo para geração de relatórios Excel de validação de performance ML.

Gera Excel com 3-4 abas:
1. Resumo Executivo - Comparação COM_ML vs SEM_ML
2. Métricas por Campanha - Detalhamento por campanha
3. Performance por Decil - Real vs Esperado (Guru vs Guru+TMB)
4. Comparação Justa - (Opcional) Fair comparison entre campanhas
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ValidationReportGenerator:
    """
    Gera relatórios Excel formatados para validação de performance ML.

    IMPORTANTE: Inclui métricas separadas para Guru (treinamento) vs
    Guru+TMB (generalização), pois o modelo foi treinado apenas com Guru.
    """

    def __init__(self):
        """Inicializa gerador de relatórios."""
        pass

    def generate_excel_report(
        self,
        campaign_metrics: pd.DataFrame,
        decile_metrics: pd.DataFrame,
        ml_comparison: Dict,
        matching_stats: Dict,
        overall_stats: Dict,
        config_params: Dict,
        output_path: str,
        comparison_group_metrics: Optional[pd.DataFrame] = None,
        fair_comparison_info: Optional[Dict] = None,
        matched_df: Optional[pd.DataFrame] = None,
        sales_df: Optional[pd.DataFrame] = None,
        all_adsets_comparison: Optional[pd.DataFrame] = None,
        adset_level_comparisons: Optional[Dict] = None,
        ad_level_comparisons: Optional[Dict] = None,
        ad_in_matched_adsets_comparisons: Optional[Dict] = None,
        matched_ads_in_matched_adsets_comparisons: Optional[Dict] = None
    ) -> str:
        """
        Gera relatório Excel completo com 5-6 abas.

        Args:
            campaign_metrics: DataFrame de CampaignMetricsCalculator
            decile_metrics: DataFrame de DecileMetricsCalculator
            ml_comparison: Dict de compare_ml_vs_non_ml()
            matching_stats: Dict de get_matching_stats()
            overall_stats: Dict de calculate_overall_stats()
            config_params: Dicionário com parâmetros da análise
            output_path: Caminho completo para salvar Excel
            comparison_group_metrics: (Opcional) DataFrame com métricas por comparison_group
            fair_comparison_info: (Opcional) Dict com informações dos matches de campanhas

        Returns:
            Caminho do arquivo gerado
        """
        logger.info(f"📄 Gerando relatório Excel: {output_path}")

        # Criar diretório se não existir
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Criar ExcelWriter
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        workbook = writer.book

        # Definir formatos
        formats = self._create_formats(workbook)

        # Aba 1: Performance Geral
        logger.info("   Gerando aba: Performance Geral")
        self._write_performance_geral(writer, overall_stats, matching_stats, campaign_metrics, formats)

        # Aba 2: Performance por Campanha - REMOVIDA conforme solicitação
        # logger.info("   Gerando aba: Performance por Campanha")
        # self._write_performance_campanhas(writer, campaign_metrics, formats)

        # Aba 2: Comparação por Campanhas (antiga "Comparação Justa")
        # SEMPRE gerar aba (mesmo se vazia), para ter consistência no relatório
        logger.info("   Gerando aba: Comparação por Campanhas")
        self._write_fair_comparison(writer, campaign_metrics, comparison_group_metrics, fair_comparison_info, formats)

        # Guardar o caminho do arquivo para ler de volta as abas formatadas
        # (para garantir que temos as mesmas colunas, especialmente 'Grupo')
        temp_output_path = output_path

        # Aba 3: Comparação por Adsets (MOVIDA ANTES DE COMPARAÇÃO ML)
        # Guardar DataFrames para consolidação na aba Comparação ML
        # IMPORTANTE: Usar campaign_metrics agregado por comparison_group (não comparison_group_metrics)
        # comparison_group_metrics vem do matched_df que pode ter mais leads do que os relatórios Meta
        # Para consistência, usar leads dos relatórios Meta (campaign_metrics)

        campanhas_df = None
        if 'comparison_group' in campaign_metrics.columns and not campaign_metrics.empty:
            # Se configurado, agrupar "Otimização ML" com "Controle"
            campaign_for_filtering = campaign_metrics.copy()
            merge_otimizacao = config_params.get('merge_otimizacao_ml_with_controle', False)

            if merge_otimizacao:
                logger.info("   ⚙️ Agrupando 'Otimização ML' com 'Controle' (merge_otimizacao_ml_with_controle=true)")
                # Substituir 'Otimização ML' por 'Controle' antes da agregação
                campaign_for_filtering.loc[
                    campaign_for_filtering['comparison_group'] == 'Otimização ML',
                    'comparison_group'
                ] = 'Controle'

            # Filtrar apenas "Eventos ML" e "Controle" para consistência com adsets
            campaign_filtered = campaign_for_filtering[
                campaign_for_filtering['comparison_group'].isin(['Eventos ML', 'Controle'])
            ].copy()

            # Agregar por comparison_group
            agg_dict = {
                'leads': 'sum',
                'conversions': 'sum',
                'conversion_rate': lambda x: (campaign_filtered[campaign_filtered['comparison_group'] == x.name]['conversions'].sum() / campaign_filtered[campaign_filtered['comparison_group'] == x.name]['leads'].sum() * 100) if campaign_filtered[campaign_filtered['comparison_group'] == x.name]['leads'].sum() > 0 else 0,
                'total_revenue': 'sum' if 'total_revenue' in campaign_filtered.columns else lambda x: 0,
                'spend': 'sum',
                'cpl': lambda x: (campaign_filtered[campaign_filtered['comparison_group'] == x.name]['spend'].sum() / campaign_filtered[campaign_filtered['comparison_group'] == x.name]['leads'].sum()) if campaign_filtered[campaign_filtered['comparison_group'] == x.name]['leads'].sum() > 0 else 0,
                'roas': lambda x: (campaign_filtered[campaign_filtered['comparison_group'] == x.name]['total_revenue'].sum() / campaign_filtered[campaign_filtered['comparison_group'] == x.name]['spend'].sum()) if campaign_filtered[campaign_filtered['comparison_group'] == x.name]['spend'].sum() > 0 else 0 if 'total_revenue' in campaign_filtered.columns else 0,
                'contribution_margin': 'sum' if 'contribution_margin' in campaign_filtered.columns else lambda x: 0
            }

            campanhas_df = campaign_filtered.groupby('comparison_group', as_index=False).agg({
                'leads': 'sum',
                'conversions': 'sum',
                'total_revenue': 'sum' if 'total_revenue' in campaign_filtered.columns else lambda x: 0,
                'spend': 'sum',
                'contribution_margin': 'sum' if 'contribution_margin' in campaign_filtered.columns else lambda x: 0
            })

            # Calcular métricas derivadas
            campanhas_df['conversion_rate'] = (campanhas_df['conversions'] / campanhas_df['leads'] * 100).fillna(0)
            campanhas_df['cpl'] = (campanhas_df['spend'] / campanhas_df['leads']).fillna(0)
            if 'total_revenue' in campanhas_df.columns:
                campanhas_df['roas'] = (campanhas_df['total_revenue'] / campanhas_df['spend']).fillna(0)
            else:
                campanhas_df['roas'] = 0

            campanhas_df['margin'] = campanhas_df.get('contribution_margin', 0)
        adsets_df = None
        ads_df = None

        try:
            from src.validation.fair_campaign_comparison import (
                prepare_adset_comparison_for_excel,
                prepare_ad_comparison_for_excel
            )

            # ADSETS - formato similar à aba Campanhas
            if adset_level_comparisons is not None:
                logger.info("   Gerando aba: Comparação por Adsets")
                logger.info(f"   🔍 DEBUG - adset_level_comparisons keys: {list(adset_level_comparisons.keys())}")
                excel_dfs_adsets = prepare_adset_comparison_for_excel(adset_level_comparisons)

                logger.info(f"   🔍 DEBUG - excel_dfs_adsets keys: {list(excel_dfs_adsets.keys())}")
                if 'comparacao_adsets' in excel_dfs_adsets:
                    logger.info(f"   🔍 DEBUG - comparacao_adsets shape: {excel_dfs_adsets['comparacao_adsets'].shape}")

                if 'comparacao_adsets' in excel_dfs_adsets and not excel_dfs_adsets['comparacao_adsets'].empty:
                    adsets_df = excel_dfs_adsets['comparacao_adsets']
                    self._write_adsets_comparison(writer, adsets_df, formats)
            else:
                logger.warning("   ⚠️ adset_level_comparisons is None, pulando aba Comparação por Adsets")

            # COMENTADO: Aba de comparação por ads desabilitada temporariamente
            # # ADS - formato similar à aba Campanhas e Adsets
            # if ad_level_comparisons is not None:
            #     logger.info("   Gerando aba: Comparação por Ads")
            #     excel_dfs_ads = prepare_ad_comparison_for_excel(ad_level_comparisons)
            #
            #     if 'comparacao_ads' in excel_dfs_ads and not excel_dfs_ads['comparacao_ads'].empty:
            #         ads_df = excel_dfs_ads['comparacao_ads']
            #         self._write_ads_comparison(writer, ads_df, formats)

            # ADS EM ADSETS MATCHED - nova comparação
            ads_in_adsets_df = None
            if ad_in_matched_adsets_comparisons is not None:
                from src.validation.fair_campaign_comparison import prepare_ad_comparison_for_excel
                excel_dfs_ads_in_adsets = prepare_ad_comparison_for_excel(ad_in_matched_adsets_comparisons)

                if 'comparacao_ads' in excel_dfs_ads_in_adsets and not excel_dfs_ads_in_adsets['comparacao_ads'].empty:
                    ads_in_adsets_df = excel_dfs_ads_in_adsets['comparacao_ads']

            # ADS MATCHED EM ADSETS MATCHED - nova comparação (Tabela 6)
            matched_ads_in_adsets_df = None
            if matched_ads_in_matched_adsets_comparisons is not None:
                from src.validation.fair_campaign_comparison import prepare_ad_comparison_for_excel
                excel_dfs_matched_ads_in_adsets = prepare_ad_comparison_for_excel(matched_ads_in_matched_adsets_comparisons)

                if 'comparacao_ads' in excel_dfs_matched_ads_in_adsets and not excel_dfs_matched_ads_in_adsets['comparacao_ads'].empty:
                    matched_ads_in_adsets_df = excel_dfs_matched_ads_in_adsets['comparacao_ads']

        except Exception as e:
            logger.warning(f"   ⚠️ Erro ao gerar abas de comparação (adsets/ads): {e}")
            import traceback
            traceback.print_exc()

        # Aba 4: Comparação ML (resumo da comparação com 4 tabelas consolidadas)
        logger.info("   Gerando aba: Comparação ML")
        logger.info(f"   🔍 DEBUG - DataFrames status:")
        logger.info(f"      campanhas_df: {'OK' if campanhas_df is not None else 'None'}")
        logger.info(f"      all_adsets_comparison: {'OK' if all_adsets_comparison is not None else 'None'} {f'({len(all_adsets_comparison)} rows)' if all_adsets_comparison is not None else ''}")
        logger.info(f"      adsets_df (matched): {'OK' if adsets_df is not None else 'None'} {f'({len(adsets_df)} rows)' if adsets_df is not None else ''}")
        logger.info(f"      ads_df: {'OK' if ads_df is not None else 'None'}")
        logger.info(f"      ads_in_adsets_df: {'OK' if ads_in_adsets_df is not None else 'None'}")
        logger.info(f"      matched_ads_in_adsets_df: {'OK' if matched_ads_in_adsets_df is not None else 'None'}")
        self._write_comparacao_ml(writer, ml_comparison, campanhas_df, all_adsets_comparison, adsets_df, ads_df, ads_in_adsets_df, matched_ads_in_adsets_df, formats)

        # Aba 5: Comparação Faixa A (Eventos ML vs Faixa A - sistema legado)
        if campaign_metrics is not None and not campaign_metrics.empty and 'comparison_group' in campaign_metrics.columns:
            logger.info("   Gerando aba: Comparação Faixa A")
            self._write_comparacao_faixa_a(writer, campaign_metrics, formats)

        # Aba FINAL: Detalhes das Conversões (movida para última posição)
        if sales_df is not None:
            logger.info("   Gerando aba: Detalhes das Conversões")
            self._write_conversions_detail(writer, matched_df, sales_df, formats)

        # Salvar Excel
        writer.close()

        logger.info(f"   ✅ Excel salvo com sucesso ({Path(output_path).stat().st_size / 1024:.1f} KB)")

        return output_path

    def _create_formats(self, workbook) -> Dict:
        """
        Cria formatos de célula para o Excel.

        Returns:
            Dicionário com formatos para aplicar
        """
        formats = {
            'header': workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter',
                'text_wrap': True
            }),
            'header_green': workbook.add_format({
                'bold': True,
                'bg_color': '#70AD47',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            }),
            'header_red': workbook.add_format({
                'bold': True,
                'bg_color': '#E74C3C',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            }),
            'currency': workbook.add_format({
                'num_format': 'R$ #,##0.00',
                'border': 1
            }),
            'percent': workbook.add_format({
                'num_format': '0.00%',
                'border': 1
            }),
            'number': workbook.add_format({
                'num_format': '#,##0',
                'border': 1
            }),
            'decimal': workbook.add_format({
                'num_format': '0.00',
                'border': 1
            }),
            'text': workbook.add_format({
                'border': 1,
                'align': 'left',
                'valign': 'vcenter'
            }),
            'title': workbook.add_format({
                'bold': True,
                'font_size': 14,
                'font_color': '#2E4053'
            }),
            'subtitle': workbook.add_format({
                'bold': True,
                'font_size': 11,
                'font_color': '#34495E'
            }),
            'positive': workbook.add_format({
                'bg_color': '#D5F4E6',
                'border': 1
            }),
            'negative': workbook.add_format({
                'bg_color': '#FADBD8',
                'border': 1
            })
        }

        return formats

    def _write_performance_geral(
        self,
        writer: pd.ExcelWriter,
        overall_stats: Dict,
        matching_stats: Dict,
        campaign_metrics: pd.DataFrame,
        formats: Dict
    ):
        """
        Escreve aba 'Performance Geral' com estatísticas agregadas do período.
        """
        worksheet = workbook = writer.book.add_worksheet('Performance Geral')

        # Título
        worksheet.write(0, 0, 'PERFORMANCE GERAL - VALIDAÇÃO DE PERFORMANCE ML', formats['title'])
        worksheet.write(1, 0, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", formats['subtitle'])

        # Períodos de Aferição
        row = 3
        worksheet.write(row, 0, '📅 PERÍODOS DE AFERIÇÃO', formats['subtitle'])
        row += 1

        # Extrair datas dos config_params se disponíveis
        lead_start = overall_stats.get('lead_period_start', 'N/A')
        lead_end = overall_stats.get('lead_period_end', 'N/A')
        sales_start = overall_stats.get('sales_period_start', 'N/A')
        sales_end = overall_stats.get('sales_period_end', 'N/A')

        worksheet.write(row, 0, 'Período de Captação', formats['text'])
        worksheet.write(row, 1, f"{lead_start} a {lead_end}", formats['text'])
        row += 1

        worksheet.write(row, 0, 'Período de Vendas', formats['text'])
        worksheet.write(row, 1, f"{sales_start} a {sales_end}", formats['text'])
        row += 1

        # Estatísticas Gerais
        row += 1
        worksheet.write(row, 0, '📊 ESTATÍSTICAS GERAIS', formats['subtitle'])
        row += 1

        # Calcular tracking rate
        total_conv = overall_stats.get('total_conversions', 0)
        matched_conv = overall_stats.get('matched_conversions', 0)
        tracking_rate = (matched_conv / total_conv) if total_conv > 0 else 0

        # Métricas conforme definido pelo usuário
        # 1. Leads Meta - eventos "Lead" das campanhas
        total_leads_meta = overall_stats.get('total_leads_meta', 0)

        # 2. Pessoas únicas CAPI - pessoas únicas no banco CAPI
        capi_leads_total = overall_stats.get('capi_leads_total', 0)

        # 3. Respostas na pesquisa - da Google Sheets
        survey_leads = overall_stats.get('survey_leads', 0)

        # 4. Vendas - total no período
        total_vendas = total_conv

        # 5. Vendas identificadas - com matching
        vendas_identificadas = matched_conv

        # 6. % de trackeamento - vendas identificadas / total vendas
        pct_trackeamento = tracking_rate

        general_data = [
            ['Leads Meta', total_leads_meta],
            ['Pessoas únicas (CAPI)', capi_leads_total],
            ['Respostas na pesquisa', survey_leads],
            ['Vendas no Período', total_vendas],
            ['Vendas identificadas', vendas_identificadas],
            ['% de trackeamento', pct_trackeamento],
            ['Gasto Total', overall_stats.get('total_spend', 0)],
        ]

        for metric, value in general_data:
            worksheet.write(row, 0, metric, formats['text'])
            if '% de trackeamento' in metric or 'Taxa' in metric:
                worksheet.write(row, 1, value, formats['percent'])
            elif 'Receita' in metric or 'Gasto' in metric or 'Margem' in metric:
                worksheet.write(row, 1, value, formats['currency'])
            elif 'ROAS' in metric:
                worksheet.write(row, 1, value, formats['decimal'])
            else:
                worksheet.write(row, 1, value, formats['number'])
            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 25)
        worksheet.set_column(1, 3, 18)

    def _write_performance_campanhas(
        self,
        writer: pd.ExcelWriter,
        campaign_metrics: pd.DataFrame,
        formats: Dict
    ):
        """
        Escreve aba 'Performance por Campanha' com duas tabelas separadas por conta.
        """
        if campaign_metrics.empty:
            # Criar sheet vazia com mensagem
            worksheet = writer.book.add_worksheet('Performance por Campanha')
            worksheet.write(0, 0, 'Nenhuma métrica de campanha disponível', formats['subtitle'])
            return

        # Criar coluna de leads consolidada (usar total_conversion_events quando disponível)
        # Mesma lógica da aba "Comparação Justa"
        campaign_metrics = campaign_metrics.copy()
        campaign_metrics['leads_display'] = campaign_metrics.apply(
            lambda row: int(row['total_conversion_events']) if row.get('total_conversion_events', 0) > 0 else int(row.get('leads', 0)),
            axis=1
        )

        # Reorganizar e renomear colunas
        column_mapping = {
            'comparison_group': 'Grupo',
            'campaign': 'Campanha',
            'optimization_goal': 'Evento de conversão',
            'leads_display': 'Leads',
            'LeadQualified': 'LeadQualified',
            'LeadQualifiedHighQuality': 'LeadQualifiedHighQuality',
            'Faixa A': 'Faixa A',
            'conversions': 'Vendas',
            'conversion_rate': 'Taxa de conversão',
            'budget': 'Orçamento',
            'spend': 'Valor gasto',
            'cpl': 'CPL',
            'roas': 'ROAS',
            'total_revenue': 'Receita Total',
            'contribution_margin': 'Margem de contribuição',
        }

        # Ordem das colunas (campaign após comparison_group, custom events após leads)
        # IMPORTANTE: Margem de contribuição vai para o final DEPOIS das colunas restantes
        column_order = [
            'comparison_group', 'campaign', 'optimization_goal', 'leads_display',
            'LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A',
            'conversions', 'conversion_rate', 'budget',
            'spend', 'cpl', 'roas', 'total_revenue'
        ]

        # Colunas a excluir (incluindo total_conversion_events e num_creatives)
        exclude_cols = ['ml_type', 'margin_percent', 'account_id', 'total_conversion_events', 'num_creatives', 'leads']

        # Adicionar colunas restantes que não estão na lista (exceto as excluídas)
        remaining_cols = [
            col for col in campaign_metrics.columns
            if col not in column_order and col not in exclude_cols and col != 'contribution_margin'
        ]

        # Montar ordem final: colunas principais + restantes + Margem de Contribuição por último
        final_column_order = column_order + remaining_cols + ['contribution_margin']

        # Reordenar DataFrame mantendo apenas colunas que existem
        existing_cols = [col for col in final_column_order if col in campaign_metrics.columns]

        # Criar worksheet
        worksheet = writer.book.add_worksheet('Performance por Campanha')

        # Título principal
        worksheet.write(0, 0, '📊 PERFORMANCE DETALHADA POR CAMPANHA', formats['title'])

        # Separar por conta
        account_ids = campaign_metrics['account_id'].unique()

        # Mapear account_id para nomes amigáveis
        account_names = {
            'act_188005769808959': 'Ads - Rodolfo Mori',
            'act_786790755803474': 'Ads - Gestor de IA'
        }

        current_row = 2

        for account_id in sorted(account_ids):
            if not account_id:  # Pular campanhas sem account_id
                continue

            # Filtrar campanhas desta conta
            account_campaigns = campaign_metrics[campaign_metrics['account_id'] == account_id].copy()

            # DEBUG: Verificar vendas antes de excluir campanhas não-captação
            vendas_antes = account_campaigns['conversions'].sum()
            # Filtrar apenas campanhas não-captação (VENDA, CPL, BLACK, etc.)
            excluir_campanhas = account_campaigns[
                account_campaigns['ml_type'] == 'EXCLUIR'
            ]
            excluir_vendas = excluir_campanhas['conversions'].sum()

            if excluir_vendas > 0:
                logger.warning(f"   ⚠️  {int(excluir_vendas)} vendas em campanhas não-captação (não mostradas na aba):")
                for _, row in excluir_campanhas[excluir_campanhas['conversions'] > 0].iterrows():
                    logger.warning(f"      • {int(row['conversions'])} vendas: {row['campaign'][:70]}")

            # IMPORTANTE: Excluir apenas campanhas não-captação (ml_type == 'EXCLUIR')
            # Isso mantém todas as campanhas de captação (ML e não-ML) independente de serem Fair Control
            account_campaigns = account_campaigns[
                account_campaigns['ml_type'] != 'EXCLUIR'
            ].copy()

            vendas_depois = account_campaigns['conversions'].sum()
            if excluir_vendas > 0:
                logger.warning(f"      Total: {int(vendas_antes)} vendas antes → {int(vendas_depois)} vendas na aba (diff: {int(excluir_vendas)})")

            if account_campaigns.empty:
                continue

            # Ordenar e preparar
            account_campaigns_ordered = account_campaigns[existing_cols].copy()
            account_campaigns_ordered.rename(columns=column_mapping, inplace=True)

            # Subtítulo da conta
            account_name = account_names.get(account_id, account_id)
            worksheet.write(current_row, 0, f'🏢 {account_name}', formats['subtitle'])
            current_row += 1

            # Headers
            for col_num, col_name in enumerate(account_campaigns_ordered.columns):
                worksheet.write(current_row, col_num, col_name, formats['header'])
            current_row += 1

            # Dados
            for row_num in range(len(account_campaigns_ordered)):
                for col_num, col_name in enumerate(account_campaigns_ordered.columns):
                    value = account_campaigns_ordered.iloc[row_num, col_num]

                    # Tratamento de valores NaN/None/Inf
                    import math
                    if pd.isna(value) or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                        value = '' if col_name in ['Grupo', 'Campanha', 'Evento de conversão'] else 0

                    # Escolher formato baseado no nome da coluna
                    if col_name in ['Taxa de conversão']:
                        worksheet.write(current_row, col_num, value / 100 if value else 0, formats['percent'])
                    elif col_name in ['Valor gasto', 'Orçamento', 'CPL', 'Receita Total', 'Margem de contribuição']:
                        worksheet.write(current_row, col_num, value if value else 0, formats['currency'])
                    elif col_name in ['ROAS']:
                        worksheet.write(current_row, col_num, value if value else 0, formats['decimal'])
                    elif col_name in ['Leads', 'Vendas', 'LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A']:
                        worksheet.write(current_row, col_num, int(value) if value else 0, formats['number'])
                    else:
                        worksheet.write(current_row, col_num, str(value) if value else '', formats['text'])
                current_row += 1

            # Espaço entre tabelas
            current_row += 2

        # Ajustar larguras
        worksheet.set_column(0, 0, 18)  # Tipo de campanha
        worksheet.set_column(1, 1, 30)  # Campanha
        worksheet.set_column(2, len(existing_cols) - 1, 15)

    def _write_conversions_detail(
        self,
        writer: pd.ExcelWriter,
        matched_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        formats: Dict
    ):
        """
        Escreve aba 'Detalhes das Conversões' mostrando TODOS os compradores do período.
        Compradores com UTM trackeada aparecem primeiro.
        """
        if sales_df.empty:
            # Criar sheet vazia com mensagem
            worksheet = writer.book.add_worksheet('Detalhes das Conversões')
            worksheet.write(0, 0, 'Nenhuma venda encontrada no período', formats['subtitle'])
            return

        # Criar índice de leads convertidos (com UTM trackeada)
        # Usar tanto email quanto telefone como chaves, e permitir múltiplas vendas por contato
        tracked_by_email = {}
        tracked_by_phone = {}

        if matched_df is not None and not matched_df.empty:
            conversions = matched_df[matched_df['converted'] == True].copy()
            for idx, conv in conversions.iterrows():
                # Extrair campaign_id do nome da campanha (formato: "Nome|ID")
                campaign_name = str(conv.get('campaign', ''))
                campaign_id = ''
                if '|' in campaign_name and len(campaign_name.split('|')) >= 2:
                    campaign_id = campaign_name.split('|')[-1].strip()

                tracking_data = {
                    'campaign_id': campaign_id,
                    'campaign': campaign_name,
                    'comparison_group': conv.get('comparison_group', ''),
                    'data_captura': conv.get('data_captura', ''),
                    'sale_date': conv.get('sale_date', ''),
                    'sale_origin': conv.get('sale_origin', ''),
                    'match_method': conv.get('match_method', '')
                }

                # Indexar por email
                email = str(conv.get('email', '')).strip().lower()
                if email and email != 'nan' and email != '':
                    if email not in tracked_by_email:
                        tracked_by_email[email] = []
                    tracked_by_email[email].append(tracking_data)

                # Indexar por telefone
                phone = str(conv.get('telefone', '')).strip()
                if phone and phone != 'nan' and phone != '':
                    if phone not in tracked_by_phone:
                        tracked_by_phone[phone] = []
                    tracked_by_phone[phone].append(tracking_data)

        # Criar lista de todas as vendas com informação de tracking
        all_sales = []
        for idx, sale in sales_df.iterrows():
            email = str(sale.get('email', '')).strip().lower()
            phone = str(sale.get('telefone', '')).strip()

            # Verificar se essa venda tem UTM trackeada (por email ou telefone)
            tracking_data = None
            if email and email != 'nan' and email in tracked_by_email:
                # Se houver múltiplas conversões para o mesmo email, pegar a primeira
                tracking_data = tracked_by_email[email][0]
            elif phone and phone != 'nan' and phone in tracked_by_phone:
                tracking_data = tracked_by_phone[phone][0]

            is_tracked = tracking_data is not None

            sale_data = {
                'trackeado': 'Sim' if is_tracked else 'Não',
                'email': sale.get('email', ''),
                'telefone': sale.get('telefone', ''),
                'sale_date': tracking_data['sale_date'] if is_tracked else sale.get('sale_date', ''),
                'sale_value': sale.get('sale_value', 0),
                'sale_origin': tracking_data['sale_origin'] if is_tracked else sale.get('origem', ''),
                'campaign_id': tracking_data['campaign_id'] if is_tracked else '',
                'campaign': tracking_data['campaign'] if is_tracked else '',
                'comparison_group': tracking_data['comparison_group'] if is_tracked else '',
                'data_captura': tracking_data['data_captura'] if is_tracked else ''
            }
            all_sales.append(sale_data)

        # Converter para DataFrame
        all_sales_df = pd.DataFrame(all_sales)

        # CRÍTICO: Remover duplicatas por email (manter apenas primeira ocorrência)
        # Isso garante consistência com as outras abas que usam deduplicação
        all_sales_df_original_count = len(all_sales_df)
        all_sales_df = all_sales_df.drop_duplicates(subset=['email'], keep='first')
        duplicates_removed = all_sales_df_original_count - len(all_sales_df)

        if duplicates_removed > 0:
            logger.info(f"   🧹 Duplicatas removidas na aba Detalhes das Conversões: {duplicates_removed}")

        # Ordenar: Trackeados primeiro (Sim antes de Não), depois por data de venda
        all_sales_df['sort_key'] = all_sales_df['trackeado'].map({'Sim': 0, 'Não': 1})
        all_sales_df = all_sales_df.sort_values(['sort_key', 'sale_date']).drop('sort_key', axis=1)

        # Reordenar colunas
        all_sales_df = all_sales_df[[
            'trackeado',
            'email',
            'telefone',
            'campaign_id',
            'campaign',
            'comparison_group',
            'data_captura',
            'sale_date',
            'sale_value',
            'sale_origin'
        ]]

        # Criar worksheet
        worksheet = writer.book.add_worksheet('Detalhes das Conversões')

        # Título
        tracked_count = len(all_sales_df[all_sales_df['trackeado'] == 'Sim'])
        total_count = len(all_sales_df)
        worksheet.write(0, 0, f'🎯 TODAS AS {total_count} VENDAS DO PERÍODO ({tracked_count} trackeadas)', formats['title'])

        # Cabeçalhos
        headers = [
            'Trackeado',
            'E-mail Comprador',
            'Telefone',
            'ID Campanha',
            'Nome Campanha',
            'Grupo',
            'Data Captura',
            'Data Venda',
            'Valor Venda',
            'Fonte Venda'
        ]

        for col_num, header in enumerate(headers):
            worksheet.write(2, col_num, header, formats['header'])

        # Escrever dados
        for row_num, (idx, row) in enumerate(all_sales_df.iterrows(), start=3):
            worksheet.write(row_num, 0, row['trackeado'], formats['text'])
            worksheet.write(row_num, 1, row['email'] if row['email'] else '', formats['text'])
            worksheet.write(row_num, 2, row['telefone'] if row['telefone'] else '', formats['text'])
            worksheet.write(row_num, 3, row['campaign_id'] if row['campaign_id'] else '', formats['text'])
            worksheet.write(row_num, 4, row['campaign'] if row['campaign'] else '', formats['text'])
            worksheet.write(row_num, 5, row['comparison_group'] if row['comparison_group'] else '', formats['text'])
            worksheet.write(row_num, 6, str(row['data_captura']) if row['data_captura'] else '', formats['text'])
            worksheet.write(row_num, 7, str(row['sale_date']) if row['sale_date'] else '', formats['text'])
            worksheet.write(row_num, 8, row['sale_value'] if row['sale_value'] else 0, formats['currency'])
            worksheet.write(row_num, 9, row['sale_origin'] if row['sale_origin'] else '', formats['text'])

        # Ajustar larguras
        worksheet.set_column(0, 0, 12)  # Trackeado
        worksheet.set_column(1, 1, 30)  # E-mail
        worksheet.set_column(2, 2, 18)  # Telefone
        worksheet.set_column(3, 3, 20)  # ID Campanha
        worksheet.set_column(4, 4, 50)  # Nome Campanha
        worksheet.set_column(5, 5, 12)  # Tipo ML
        worksheet.set_column(6, 7, 18)  # Datas
        worksheet.set_column(8, 8, 15)  # Valor
        worksheet.set_column(9, 9, 15)  # Fonte

    def _write_comparacao_ml(
        self,
        writer: pd.ExcelWriter,
        ml_comparison: Dict,
        campanhas_df: pd.DataFrame,
        all_adsets_df: pd.DataFrame,
        adsets_matched_df: pd.DataFrame,
        ads_df: pd.DataFrame,
        ads_in_adsets_df: pd.DataFrame,
        matched_ads_in_adsets_df: pd.DataFrame,
        formats: Dict
    ):
        """
        Escreve aba 'Comparação ML' com 4 tabelas consolidadas:
        1. Comparação por Campanhas (todas)
        2. Comparação por Adsets (todos - Eventos ML vs Controle)
        3. Comparação por Adsets Matched (apenas matched pairs)
        4. Comparação por Ads MATCHED em Adsets Matched (interseção mais rigorosa)
        """
        worksheet = writer.book.add_worksheet('Comparação ML')

        current_row = 0

        # TABELA 1: Comparação por Campanhas
        worksheet.write(current_row, 0, '📊 COMPARAÇÃO POR CAMPANHAS (All vs All)', formats['title'])
        current_row += 1
        worksheet.write(current_row, 0, 'Todas as campanhas Eventos ML vs Controle', formats['subtitle'])
        current_row += 2

        if campanhas_df is not None and not campanhas_df.empty:
            current_row = self._write_consolidated_table(
                worksheet, campanhas_df, formats, current_row,
                label='Campanhas'
            )
        else:
            worksheet.write(current_row, 0, 'Dados indisponíveis', formats['text'])
            current_row += 1

        current_row += 2  # Espaçamento

        # TABELA 2: Comparação por TODOS os Adsets (Eventos ML vs Controle)
        worksheet.write(current_row, 0, '📊 COMPARAÇÃO POR ADSETS (All vs All)', formats['title'])
        current_row += 1
        worksheet.write(current_row, 0, 'Todos os adsets das campanhas Eventos ML vs Controle (sem filtros)', formats['subtitle'])
        current_row += 2

        if all_adsets_df is not None and not all_adsets_df.empty:
            current_row = self._write_consolidated_table(
                worksheet, all_adsets_df, formats, current_row,
                label='Adsets (Todos)'
            )
        else:
            worksheet.write(current_row, 0, 'Dados indisponíveis', formats['text'])
            current_row += 1

        current_row += 2  # Espaçamento

        # TABELA 3: Comparação por Adsets Matched Pairs
        # Importar lista de matched adsets para exibir no título
        from src.validation.fair_campaign_comparison import MATCHED_ADSETS
        matched_adsets_list = ', '.join(MATCHED_ADSETS)

        worksheet.write(current_row, 0, '📊 COMPARAÇÃO POR ADSETS MATCHED (Matched Pairs)', formats['title'])
        current_row += 1
        worksheet.write(current_row, 0, 'Apenas adsets que aparecem em Eventos ML E Controle (R$ 200+ gasto)', formats['subtitle'])
        current_row += 1
        worksheet.write(current_row, 0, f'Matched Adsets: {matched_adsets_list}', formats['text'])
        current_row += 2

        if adsets_matched_df is not None and not adsets_matched_df.empty:
            current_row = self._write_consolidated_table(
                worksheet, adsets_matched_df, formats, current_row,
                label='Adsets (Matched)'
            )
        else:
            worksheet.write(current_row, 0, 'Dados indisponíveis', formats['text'])
            current_row += 1

        current_row += 2  # Espaçamento

        # COMENTADO: Tabela de ads matched desabilitada temporariamente
        # # TABELA 4: Comparação por Ads MATCHED EM Adsets Matched
        # worksheet.write(current_row, 0, '📊 COMPARAÇÃO POR ADS MATCHED EM ADSETS MATCHED', formats['title'])
        # current_row += 1
        # worksheet.write(current_row, 0, 'Apenas ads matched (mesmo ad_code) que pertencem aos adsets matched (R$ 200+ gasto)', formats['subtitle'])
        # current_row += 2
        #
        # if matched_ads_in_adsets_df is not None and not matched_ads_in_adsets_df.empty:
        #     current_row = self._write_consolidated_table(
        #         worksheet, matched_ads_in_adsets_df, formats, current_row,
        #         label='Ads Matched em Adsets Matched'
        #     )
        # else:
        #     worksheet.write(current_row, 0, 'Dados indisponíveis', formats['text'])
        #     current_row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 25)
        worksheet.set_column(1, 3, 18)

    def _write_comparacao_faixa_a(
        self,
        writer: pd.ExcelWriter,
        campaign_metrics: pd.DataFrame,
        formats: Dict
    ):
        """
        Escreve aba 'Comparação Faixa A' comparando Eventos ML vs Faixa A (sistema legado).
        Usa o mesmo formato das tabelas da aba 'Comparação ML'.

        Args:
            writer: Excel writer
            campaign_metrics: DataFrame com métricas de campanhas
            formats: Formatos do Excel
        """
        worksheet = writer.book.add_worksheet('Comparação Faixa A')
        current_row = 0

        # Título principal
        worksheet.write(current_row, 0, '📊 EVENTOS ML vs FAIXA A (Sistema Legado)', formats['title'])
        current_row += 1
        worksheet.write(current_row, 0, 'Comparação entre campanhas com eventos customizados CAPI vs sistema legado Faixa A', formats['subtitle'])
        current_row += 2

        # Verificar qual nome de coluna existe para Faixa A
        faixa_a_col = None
        for col in campaign_metrics.columns:
            if col.lower().replace(' ', '_') == 'faixa_a' or col == 'Faixa A':
                faixa_a_col = col
                break

        # Preparar DataFrame para _write_consolidated_table
        # Filtrar Eventos ML (TODAS as campanhas Eventos ML, independente de ter Faixa A)
        eventos_ml = campaign_metrics[
            campaign_metrics['comparison_group'] == 'Eventos ML'
        ].copy()

        # Faixa A: campanhas com Faixa A > 0
        if faixa_a_col:
            faixa_a = campaign_metrics[campaign_metrics[faixa_a_col] > 0].copy()
        else:
            faixa_a = pd.DataFrame()

        # Adicionar coluna 'Grupo' com nomes personalizados
        if not eventos_ml.empty:
            eventos_ml['Grupo'] = 'Eventos ML'
        if not faixa_a.empty:
            faixa_a['Grupo'] = 'Faixa A (Legado)'

        # Combinar
        df_combined = pd.concat([eventos_ml, faixa_a], ignore_index=True)

        if df_combined.empty:
            worksheet.write(current_row, 0, 'Sem dados disponíveis para comparação', formats['text'])
            return

        # Padronizar nomes de colunas
        column_mapping = {
            'leads': 'Leads',
            'conversions': 'Vendas',
            'spend': 'Valor gasto',
            'total_revenue': 'Receita Total',
            'contribution_margin': 'Margem de contribuição'
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df_combined.columns and new_name not in df_combined.columns:
                df_combined[new_name] = df_combined[old_name]

        # Usar função customizada para escrever tabela com labels personalizados
        current_row = self._write_faixa_a_table(
            worksheet, df_combined, formats, current_row,
            label='Campanhas Eventos ML vs Campanhas Faixa A'
        )

        # Ajustar larguras
        worksheet.set_column(0, 0, 25)
        worksheet.set_column(1, 3, 18)

    def _write_faixa_a_table(
        self,
        worksheet,
        df: pd.DataFrame,
        formats: Dict,
        start_row: int,
        label: str
    ) -> int:
        """
        Escreve uma tabela consolidada comparando Eventos ML vs Faixa A (Legado).
        Similar a _write_consolidated_table mas aceita labels personalizados.

        Args:
            worksheet: Worksheet do Excel
            df: DataFrame com dados
            formats: Formatos do Excel
            start_row: Linha inicial para escrever
            label: Label para identificação

        Returns:
            Próxima linha disponível após a tabela
        """
        # Identificar coluna de grupo
        group_col = 'Grupo' if 'Grupo' in df.columns else 'comparison_group'

        # Filtrar apenas Eventos ML e Faixa A (Legado)
        df_filtered = df[df[group_col].isin(['Eventos ML', 'Faixa A (Legado)'])].copy()

        if df_filtered.empty:
            worksheet.write(start_row, 0, 'Nenhum dado encontrado', formats['text'])
            return start_row + 1

        # Preparar colunas para agregação
        agg_dict = {}

        # Mapear colunas
        if 'Leads' in df_filtered.columns:
            agg_dict['Leads'] = 'sum'
        if 'Vendas' in df_filtered.columns:
            agg_dict['Vendas'] = 'sum'
        if 'Valor gasto' in df_filtered.columns:
            agg_dict['Valor gasto'] = 'sum'
        if 'Receita Total' in df_filtered.columns:
            agg_dict['Receita Total'] = 'sum'
        if 'Margem de contribuição' in df_filtered.columns:
            agg_dict['Margem de contribuição'] = 'sum'

        if not agg_dict:
            worksheet.write(start_row, 0, 'Colunas necessárias não encontradas', formats['text'])
            return start_row + 1

        # Agregar métricas por Grupo
        aggregated = df_filtered.groupby(group_col).agg(agg_dict).reset_index()

        # Calcular métricas derivadas
        aggregated['Taxa de conversão'] = (aggregated['Vendas'] / aggregated['Leads']) * 100
        aggregated['CPL'] = aggregated['Valor gasto'] / aggregated['Leads']
        aggregated['ROAS'] = aggregated['Receita Total'] / aggregated['Valor gasto']

        # Substituir NaN/Inf por 0
        aggregated = aggregated.fillna(0)
        aggregated = aggregated.replace([float('inf'), float('-inf')], 0)

        # Extrair dados de Eventos ML e Faixa A
        ml_data = aggregated[aggregated[group_col] == 'Eventos ML']
        faixa_a_data = aggregated[aggregated[group_col] == 'Faixa A (Legado)']

        if ml_data.empty and faixa_a_data.empty:
            worksheet.write(start_row, 0, 'Sem dados para comparação', formats['text'])
            return start_row + 1

        # Extrair métricas
        ml_metrics = {
            'leads': ml_data['Leads'].iloc[0] if not ml_data.empty else 0,
            'conversions': ml_data['Vendas'].iloc[0] if not ml_data.empty else 0,
            'conversion_rate': ml_data['Taxa de conversão'].iloc[0] if not ml_data.empty else 0,
            'spend': ml_data['Valor gasto'].iloc[0] if not ml_data.empty else 0,
            'revenue': ml_data['Receita Total'].iloc[0] if not ml_data.empty else 0,
            'cpl': ml_data['CPL'].iloc[0] if not ml_data.empty else 0,
            'roas': ml_data['ROAS'].iloc[0] if not ml_data.empty else 0,
            'margin': ml_data['Margem de contribuição'].iloc[0] if not ml_data.empty else 0,
        }

        faixa_a_metrics = {
            'leads': faixa_a_data['Leads'].iloc[0] if not faixa_a_data.empty else 0,
            'conversions': faixa_a_data['Vendas'].iloc[0] if not faixa_a_data.empty else 0,
            'conversion_rate': faixa_a_data['Taxa de conversão'].iloc[0] if not faixa_a_data.empty else 0,
            'spend': faixa_a_data['Valor gasto'].iloc[0] if not faixa_a_data.empty else 0,
            'revenue': faixa_a_data['Receita Total'].iloc[0] if not faixa_a_data.empty else 0,
            'cpl': faixa_a_data['CPL'].iloc[0] if not faixa_a_data.empty else 0,
            'roas': faixa_a_data['ROAS'].iloc[0] if not faixa_a_data.empty else 0,
            'margin': faixa_a_data['Margem de contribuição'].iloc[0] if not faixa_a_data.empty else 0,
        }

        # Escrever tabela
        row = start_row

        # Cabeçalhos
        headers = ['Métrica', 'Eventos ML', 'Faixa A (Legado)', 'Diferença %']
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, formats['header'])
        row += 1

        # Função auxiliar para calcular diferença %
        def calc_diff_pct(ml_val, fa_val):
            if fa_val == 0:
                return 0
            return ((ml_val - fa_val) / fa_val) * 100

        # Dados de comparação
        comparison_data = [
            ('Leads', ml_metrics['leads'], faixa_a_metrics['leads'], 'number'),
            ('Vendas', ml_metrics['conversions'], faixa_a_metrics['conversions'], 'number'),
            ('Taxa de conversão', ml_metrics['conversion_rate'] / 100, faixa_a_metrics['conversion_rate'] / 100, 'percent'),
            ('Valor gasto', ml_metrics['spend'], faixa_a_metrics['spend'], 'currency'),
            ('CPL', ml_metrics['cpl'], faixa_a_metrics['cpl'], 'currency'),
            ('ROAS', ml_metrics['roas'], faixa_a_metrics['roas'], 'decimal'),
            ('Receita Total', ml_metrics['revenue'], faixa_a_metrics['revenue'], 'currency'),
            ('Margem Contribuição', ml_metrics['margin'], faixa_a_metrics['margin'], 'currency'),
        ]

        for metric, ml_value, fa_value, fmt_type in comparison_data:
            worksheet.write(row, 0, metric, formats['text'])
            worksheet.write(row, 1, ml_value, formats[fmt_type])
            worksheet.write(row, 2, fa_value, formats[fmt_type])

            # Calcular diferença %
            diff_pct = calc_diff_pct(ml_value, fa_value) / 100 if fmt_type != 'percent' else calc_diff_pct(ml_value * 100, fa_value * 100) / 100
            if diff_pct != 0:
                cell_format = formats['positive'] if diff_pct > 0 else formats['negative']
                worksheet.write(row, 3, diff_pct, cell_format)
            else:
                worksheet.write(row, 3, '-', formats['text'])
            row += 1

        # Vencedor
        row += 1
        if ml_metrics['roas'] > faixa_a_metrics['roas']:
            diff_pct = calc_diff_pct(ml_metrics['roas'], faixa_a_metrics['roas'])
            winner_text = f"🏆 VENCEDOR: Eventos ML (ROAS {diff_pct:.1f}% maior)"
            worksheet.write(row, 0, winner_text, formats['header_green'])
        elif faixa_a_metrics['roas'] > ml_metrics['roas']:
            diff_pct = abs(calc_diff_pct(ml_metrics['roas'], faixa_a_metrics['roas']))
            winner_text = f"⚠️ VENCEDOR: Faixa A (ROAS {diff_pct:.1f}% maior)"
            worksheet.write(row, 0, winner_text, formats['header_red'])
        else:
            worksheet.write(row, 0, "➖ Empate técnico em ROAS", formats['header'])

        return row + 2

    def _write_consolidated_table(
        self,
        worksheet,
        df: pd.DataFrame,
        formats: Dict,
        start_row: int,
        label: str
    ) -> int:
        """
        Escreve uma tabela consolidada agregando por Grupo (Eventos ML vs Controle).

        Args:
            worksheet: Worksheet do Excel
            df: DataFrame com dados (pode ser campanhas, adsets ou ads)
            formats: Formatos do Excel
            start_row: Linha inicial para escrever
            label: Label para identificação (Campanhas/Adsets/Ads)

        Returns:
            Próxima linha disponível após a tabela
        """
        # Identificar coluna de grupo (pode ser 'Grupo' ou 'comparison_group')
        group_col = None
        if 'Grupo' in df.columns:
            group_col = 'Grupo'
        elif 'comparison_group' in df.columns:
            group_col = 'comparison_group'
        else:
            worksheet.write(start_row, 0, f'Coluna de grupo não encontrada', formats['text'])
            return start_row + 1

        # Filtrar apenas Eventos ML e Controle
        df_filtered = df[df[group_col].isin(['Eventos ML', 'Controle'])].copy()

        if df_filtered.empty:
            worksheet.write(start_row, 0, 'Nenhum dado de Eventos ML ou Controle encontrado', formats['text'])
            return start_row + 1

        # Preparar colunas para agregação (verificar quais existem)
        agg_dict = {}

        # Mapear: Leads
        if 'Leads' in df_filtered.columns:
            agg_dict['Leads'] = 'sum'
        elif 'leads' in df_filtered.columns:
            df_filtered['Leads'] = df_filtered['leads']
            agg_dict['Leads'] = 'sum'

        # Mapear: Vendas/Conversões
        if 'Vendas' in df_filtered.columns:
            agg_dict['Vendas'] = 'sum'
        elif 'conversions' in df_filtered.columns:
            df_filtered['Vendas'] = df_filtered['conversions']
            agg_dict['Vendas'] = 'sum'

        # Mapear: Valor gasto
        if 'Valor gasto' in df_filtered.columns:
            agg_dict['Valor gasto'] = 'sum'
        elif 'spend' in df_filtered.columns:
            df_filtered['Valor gasto'] = df_filtered['spend']
            agg_dict['Valor gasto'] = 'sum'

        # Mapear: Receita Total
        if 'Receita Total' in df_filtered.columns:
            agg_dict['Receita Total'] = 'sum'
        elif 'total_revenue' in df_filtered.columns:
            df_filtered['Receita Total'] = df_filtered['total_revenue']
            agg_dict['Receita Total'] = 'sum'
        elif 'revenue' in df_filtered.columns:
            df_filtered['Receita Total'] = df_filtered['revenue']
            agg_dict['Receita Total'] = 'sum'

        # Mapear: Margem de contribuição
        if 'Margem de contribuição' in df_filtered.columns:
            agg_dict['Margem de contribuição'] = 'sum'
        elif 'contribution_margin' in df_filtered.columns:
            df_filtered['Margem de contribuição'] = df_filtered['contribution_margin']
            agg_dict['Margem de contribuição'] = 'sum'
        elif 'margin' in df_filtered.columns:
            df_filtered['Margem de contribuição'] = df_filtered['margin']
            agg_dict['Margem de contribuição'] = 'sum'

        if not agg_dict:
            worksheet.write(start_row, 0, 'Colunas necessárias não encontradas', formats['text'])
            return start_row + 1

        # Agregar métricas por Grupo
        aggregated = df_filtered.groupby(group_col).agg(agg_dict).reset_index()

        # Calcular métricas derivadas
        aggregated['Taxa de conversão'] = (aggregated['Vendas'] / aggregated['Leads']) * 100
        aggregated['CPL'] = aggregated['Valor gasto'] / aggregated['Leads']
        aggregated['ROAS'] = aggregated['Receita Total'] / aggregated['Valor gasto']

        # Substituir NaN/Inf por 0
        aggregated = aggregated.fillna(0)
        aggregated = aggregated.replace([float('inf'), float('-inf')], 0)

        # Extrair dados de ML e Controle
        ml_data = aggregated[aggregated[group_col] == 'Eventos ML']
        ctrl_data = aggregated[aggregated[group_col] == 'Controle']

        if ml_data.empty or ctrl_data.empty:
            worksheet.write(start_row, 0, 'Dados incompletos (falta ML ou Controle)', formats['text'])
            return start_row + 1

        ml_row = ml_data.iloc[0]
        ctrl_row = ctrl_data.iloc[0]

        # Calcular diferenças percentuais
        def calc_diff_pct(ml_val, ctrl_val):
            if ctrl_val == 0:
                return 0
            return ((ml_val - ctrl_val) / ctrl_val)

        # Headers
        row = start_row
        worksheet.write(row, 0, 'Métrica', formats['header'])
        worksheet.write(row, 1, 'Eventos ML', formats['header_green'])
        worksheet.write(row, 2, 'Controle', formats['header_red'])
        worksheet.write(row, 3, 'Diferença %', formats['header'])
        row += 1

        # Dados da tabela
        comparison_data = [
            ('Total de Leads', ml_row['Leads'], ctrl_row['Leads'], 'number'),
            ('Conversões', ml_row['Vendas'], ctrl_row['Vendas'], 'number'),
            ('Taxa Conversão', ml_row['Taxa de conversão'] / 100, ctrl_row['Taxa de conversão'] / 100, 'percent'),
            ('Receita Total', ml_row['Receita Total'], ctrl_row['Receita Total'], 'currency'),
            ('Gasto Total', ml_row['Valor gasto'], ctrl_row['Valor gasto'], 'currency'),
            ('CPL', ml_row['CPL'], ctrl_row['CPL'], 'currency'),
            ('ROAS', ml_row['ROAS'], ctrl_row['ROAS'], 'decimal'),
            ('Margem Contribuição', ml_row['Margem de contribuição'], ctrl_row['Margem de contribuição'], 'currency'),
        ]

        for metric, ml_value, ctrl_value, fmt_type in comparison_data:
            worksheet.write(row, 0, metric, formats['text'])
            worksheet.write(row, 1, ml_value, formats[fmt_type])
            worksheet.write(row, 2, ctrl_value, formats[fmt_type])

            # Calcular diferença %
            diff_pct = calc_diff_pct(ml_value, ctrl_value)
            if diff_pct != 0:
                cell_format = formats['positive'] if diff_pct > 0 else formats['negative']
                worksheet.write(row, 3, diff_pct, cell_format)
            else:
                worksheet.write(row, 3, '-', formats['text'])
            row += 1

        # Vencedor
        row += 1
        if ml_row['ROAS'] > ctrl_row['ROAS']:
            diff_pct = calc_diff_pct(ml_row['ROAS'], ctrl_row['ROAS']) * 100
            winner_text = f"🏆 VENCEDOR: Eventos ML (ROAS {diff_pct:.1f}% maior)"
            worksheet.write(row, 0, winner_text, formats['header_green'])
        elif ctrl_row['ROAS'] > ml_row['ROAS']:
            diff_pct = abs(calc_diff_pct(ml_row['ROAS'], ctrl_row['ROAS'])) * 100
            winner_text = f"⚠️ VENCEDOR: Controle (ROAS {diff_pct:.1f}% maior)"
            worksheet.write(row, 0, winner_text, formats['header_red'])
        else:
            worksheet.write(row, 0, "➖ Empate técnico em ROAS", formats['header'])
        row += 1

        return row

    def _write_total_comparison_table(
        self,
        worksheet,
        ml_comparison: Dict,
        formats: Dict,
        start_row: int = 2
    ):
        """Escreve tabela de comparação total (COM ML vs SEM ML)."""
        # Headers
        row = start_row
        worksheet.write(row, 0, 'Métrica', formats['header'])
        worksheet.write(row, 1, 'COM ML', formats['header_green'])
        worksheet.write(row, 2, 'SEM ML', formats['header_red'])
        worksheet.write(row, 3, 'Diferença %', formats['header'])
        row += 1

        com_ml = ml_comparison.get('com_ml', {})
        sem_ml = ml_comparison.get('sem_ml', {})
        diff = ml_comparison.get('difference', {})

        comparison_data = [
            ('Total de Leads', com_ml.get('leads', 0), sem_ml.get('leads', 0), diff.get('leads_diff', 0) / 100, 'number'),
            ('Conversões', com_ml.get('conversions', 0), sem_ml.get('conversions', 0), diff.get('conversions_diff', 0) / 100, 'number'),
            ('Taxa Conversão', com_ml.get('conversion_rate', 0) / 100, sem_ml.get('conversion_rate', 0) / 100, diff.get('conversion_rate_diff', 0) / 100, 'percent'),
            ('Receita Total', com_ml.get('revenue', 0), sem_ml.get('revenue', 0), diff.get('revenue_diff', 0) / 100, 'currency'),
            ('Gasto Total', com_ml.get('spend', 0), sem_ml.get('spend', 0), diff.get('spend_diff', 0) / 100, 'currency'),
            ('CPL', com_ml.get('cpl', 0), sem_ml.get('cpl', 0), diff.get('cpl_diff', 0) / 100, 'currency'),
            ('ROAS', com_ml.get('roas', 0), sem_ml.get('roas', 0), diff.get('roas_diff', 0) / 100, 'decimal'),
            ('Margem Contribuição', com_ml.get('margin', 0), sem_ml.get('margin', 0), diff.get('margin_diff', 0) / 100, 'currency'),
        ]

        for metric, com_value, sem_value, diff_value, fmt_type in comparison_data:
            worksheet.write(row, 0, metric, formats['text'])
            worksheet.write(row, 1, com_value, formats[fmt_type])
            worksheet.write(row, 2, sem_value, formats[fmt_type])

            if diff_value is not None:
                cell_format = formats['positive'] if diff_value > 0 else formats['negative']
                worksheet.write(row, 3, diff_value, cell_format)
            else:
                worksheet.write(row, 3, '-', formats['text'])
            row += 1

        # Vencedor
        row += 1
        if com_ml.get('roas', 0) > sem_ml.get('roas', 0):
            winner_text = f"🏆 VENCEDOR: COM ML (ROAS {diff.get('roas_diff', 0):.1f}% maior)"
            worksheet.write(row, 0, winner_text, formats['header_green'])
        elif sem_ml.get('roas', 0) > com_ml.get('roas', 0):
            winner_text = f"⚠️ VENCEDOR: SEM ML (ROAS {abs(diff.get('roas_diff', 0)):.1f}% maior)"
            worksheet.write(row, 0, winner_text, formats['header_red'])
        else:
            worksheet.write(row, 0, "➖ Empate técnico em ROAS", formats['header'])

        # Ajustar larguras
        worksheet.set_column(0, 0, 25)
        worksheet.set_column(1, 3, 18)

    def _write_fair_comparison_table(
        self,
        worksheet,
        comparison_group_metrics: pd.DataFrame,
        formats: Dict,
        start_row: int = 3
    ):
        """Escreve tabela de comparação justa (Eventos ML vs Controle)."""
        # Filtrar apenas Eventos ML e Controle
        ml_data = comparison_group_metrics[comparison_group_metrics['comparison_group'] == 'Eventos ML']
        fc_data = comparison_group_metrics[comparison_group_metrics['comparison_group'] == 'Controle']

        if ml_data.empty or fc_data.empty:
            worksheet.write(start_row, 0, 'Dados insuficientes para comparação justa', formats['subtitle'])
            return

        # Extrair métricas
        ml_metrics = ml_data.iloc[0]
        fc_metrics = fc_data.iloc[0]

        # Calcular diferenças percentuais
        def calc_diff_pct(ml_val, fc_val):
            if fc_val == 0:
                return 0
            return ((ml_val - fc_val) / fc_val) * 100

        # Headers
        row = start_row
        worksheet.write(row, 0, 'Métrica', formats['header'])
        worksheet.write(row, 1, 'Eventos ML', formats['header_green'])
        worksheet.write(row, 2, 'Controle', formats['header_red'])
        worksheet.write(row, 3, 'Diferença %', formats['header'])
        row += 1

        # Preparar dados de comparação
        comparison_data = [
            ('Total de Leads', ml_metrics.get('leads', 0), fc_metrics.get('leads', 0), 'number'),
            ('Conversões', ml_metrics.get('conversions', 0), fc_metrics.get('conversions', 0), 'number'),
            ('Taxa Conversão', ml_metrics.get('conversion_rate', 0) / 100, fc_metrics.get('conversion_rate', 0) / 100, 'percent'),
            ('Receita Total', ml_metrics.get('total_revenue', 0), fc_metrics.get('total_revenue', 0), 'currency'),
            ('Gasto Total', ml_metrics.get('spend', 0), fc_metrics.get('spend', 0), 'currency'),
            ('CPL', ml_metrics.get('cpl', 0), fc_metrics.get('cpl', 0), 'currency'),
            ('ROAS', ml_metrics.get('roas', 0), fc_metrics.get('roas', 0), 'decimal'),
            ('Margem Contribuição', ml_metrics.get('margin', 0), fc_metrics.get('margin', 0), 'currency'),
        ]

        for metric, ml_value, fc_value, fmt_type in comparison_data:
            worksheet.write(row, 0, metric, formats['text'])
            worksheet.write(row, 1, ml_value, formats[fmt_type])
            worksheet.write(row, 2, fc_value, formats[fmt_type])

            # Calcular diferença %
            diff_pct = calc_diff_pct(ml_value, fc_value) / 100
            if diff_pct is not None and diff_pct != 0:
                cell_format = formats['positive'] if diff_pct > 0 else formats['negative']
                worksheet.write(row, 3, diff_pct, cell_format)
            else:
                worksheet.write(row, 3, '-', formats['text'])
            row += 1

        # Vencedor
        row += 1
        ml_roas = ml_metrics.get('roas', 0)
        fc_roas = fc_metrics.get('roas', 0)

        if ml_roas > fc_roas:
            diff_pct = calc_diff_pct(ml_roas, fc_roas)
            winner_text = f"🏆 VENCEDOR: Eventos ML (ROAS {diff_pct:.1f}% maior)"
            worksheet.write(row, 0, winner_text, formats['header_green'])
        elif fc_roas > ml_roas:
            diff_pct = abs(calc_diff_pct(ml_roas, fc_roas))
            winner_text = f"⚠️ VENCEDOR: Controle (ROAS {diff_pct:.1f}% maior)"
            worksheet.write(row, 0, winner_text, formats['header_red'])
        else:
            worksheet.write(row, 0, "➖ Empate técnico em ROAS", formats['header'])

        # Ajustar larguras
        worksheet.set_column(0, 0, 25)
        worksheet.set_column(1, 3, 18)

    def _write_matching_stats(
        self,
        writer: pd.ExcelWriter,
        matching_stats: Dict,
        formats: Dict
    ):
        """
        Escreve aba 'Matching Stats' com estatísticas de vinculação.
        """
        worksheet = writer.book.add_worksheet('Matching Stats')

        # Título
        worksheet.write(0, 0, '🔗 ESTATÍSTICAS DE MATCHING (Leads ↔ Vendas)', formats['title'])

        # Dados
        row = 2
        stats_data = [
            ('Total de Leads', matching_stats.get('total_leads', 0), 'number'),
            ('Total de Conversões', matching_stats.get('total_conversions', 0), 'number'),
            ('Taxa de Conversão Geral', matching_stats.get('conversion_rate', 0) / 100, 'percent'),
            ('', '', 'text'),  # Separador
            ('Match por Email', matching_stats.get('matched_by_email', 0), 'number'),
            ('Match por Telefone', matching_stats.get('matched_by_phone', 0), 'number'),
            ('Taxa Match Email', matching_stats.get('match_rate_email', 0) / 100, 'percent'),
            ('Taxa Match Telefone', matching_stats.get('match_rate_phone', 0) / 100, 'percent'),
            ('', '', 'text'),  # Separador
            ('Receita Total', matching_stats.get('total_revenue', 0), 'currency'),
            ('Ticket Médio', matching_stats.get('avg_ticket', 0), 'currency'),
            ('', '', 'text'),  # Separador
            ('Conversões Guru', matching_stats.get('conversions_guru', 0), 'number'),
            ('Conversões TMB', matching_stats.get('conversions_tmb', 0), 'number'),
        ]

        for metric, value, fmt_type in stats_data:
            if metric:  # Não escrever linha vazia
                worksheet.write(row, 0, metric, formats['text'])
                worksheet.write(row, 1, value, formats[fmt_type])
            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 30)
        worksheet.set_column(1, 1, 20)

    def _write_configuracao(
        self,
        writer: pd.ExcelWriter,
        config_params: Dict,
        formats: Dict
    ):
        """
        Escreve aba 'Configuração' com parâmetros da análise.
        """
        worksheet = writer.book.add_worksheet('Configuração')

        # Título
        worksheet.write(0, 0, '⚙️ PARÂMETROS DE CONFIGURAÇÃO', formats['title'])

        # Dados
        row = 2
        for param, value in config_params.items():
            worksheet.write(row, 0, param, formats['text'])
            worksheet.write(row, 1, str(value), formats['text'])
            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 30)
        worksheet.set_column(1, 1, 50)

    def _write_fair_comparison(
        self,
        writer: pd.ExcelWriter,
        campaign_metrics: pd.DataFrame,
        comparison_group_metrics: pd.DataFrame,
        fair_comparison_info: Optional[Dict],
        formats: Dict
    ):
        """
        Escreve aba 'Comparação por Campanhas' com lista detalhada de campanhas ML vs Fair Control matched.
        Usa o mesmo matching que já funciona nas outras abas (via comparison_group).
        """
        worksheet = writer.book.add_worksheet('Comparação por Campanhas')

        # Título
        worksheet.write(0, 0, '🎯 COMPARAÇÃO POR CAMPANHAS - EVENTOS ML vs CONTROLE', formats['title'])
        worksheet.write(1, 0, 'Lista de campanhas matched com MESMO budget e criativos', formats['subtitle'])

        row = 3

        # Verificar se temos métricas de comparação
        if campaign_metrics.empty or 'comparison_group' not in campaign_metrics.columns:
            # Mensagem simples quando não há matches
            worksheet.write(row, 0, 'Nenhuma campanha de controle encontrada no período.', formats['text'])
            return

        # Mapear account_id para nomes amigáveis
        account_names = {
            'act_188005769808959': 'Rodolfo Mori',
            'act_786790755803474': 'Gestor de IA'
        }

        # Cabeçalhos (Conta como primeira coluna, adicionar Campaign ID)
        headers = [
            'Conta', 'Campanha', 'Campaign ID', 'Grupo',
            'Leads', 'LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A',
            'Vendas', 'Taxa de conversão',
            'Orçamento', 'Valor gasto', 'CPL', 'ROAS', 'Receita Total', 'Margem de contribuição'
        ]
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, formats['header'])
        row += 1

        # Filtrar campanhas ML (Eventos ML + Otimização ML) e Controle
        # Incluir AMBOS os tipos de ML para comparação completa
        fair_campaigns = campaign_metrics[
            campaign_metrics['comparison_group'].isin(['Eventos ML', 'Otimização ML', 'Controle'])
        ].sort_values(['comparison_group', 'campaign'])

        # Escrever linhas das campanhas
        for _, campaign_row in fair_campaigns.iterrows():
            col_idx = 0
            # Conta (primeira coluna)
            account_id = campaign_row.get('account_id', '')
            account_name = account_names.get(account_id, account_id if account_id else 'N/A')
            worksheet.write(row, col_idx, account_name, formats['text'])
            col_idx += 1
            # Campanha
            worksheet.write(row, col_idx, campaign_row['campaign'], formats['text'])
            col_idx += 1
            # Campaign ID (extrair do nome da campanha que tem formato "NOME|ID")
            campaign_name = campaign_row['campaign']
            campaign_id = 'N/A'
            if '|' in campaign_name:
                parts = campaign_name.split('|')
                # O ID está na última parte (após o último |)
                potential_id = parts[-1].strip()
                # Verificar se é um número de 18 dígitos
                if potential_id.isdigit() and len(potential_id) == 18:
                    campaign_id = potential_id
            worksheet.write(row, col_idx, campaign_id, formats['text'])
            col_idx += 1
            # Grupo
            worksheet.write(row, col_idx, campaign_row['comparison_group'], formats['text'])
            col_idx += 1

            # Leads - usar diretamente o campo 'leads'
            # Este valor já foi ajustado com leads artificiais para a campanha especial
            leads = int(campaign_row.get('leads', 0))
            worksheet.write(row, col_idx, leads, formats['number'])
            col_idx += 1

            # Custom events
            worksheet.write(row, col_idx, int(campaign_row.get('LeadQualified', 0)), formats['number'])
            col_idx += 1
            worksheet.write(row, col_idx, int(campaign_row.get('LeadQualifiedHighQuality', 0)), formats['number'])
            col_idx += 1
            worksheet.write(row, col_idx, int(campaign_row.get('Faixa A', 0)), formats['number'])
            col_idx += 1

            worksheet.write(row, col_idx, int(campaign_row['conversions']), formats['number'])
            col_idx += 1
            worksheet.write(row, col_idx, campaign_row['conversion_rate'] / 100, formats['percent'])
            col_idx += 1
            worksheet.write(row, col_idx, campaign_row.get('budget', 0), formats['currency'])
            col_idx += 1
            worksheet.write(row, col_idx, campaign_row['spend'], formats['currency'])
            col_idx += 1
            worksheet.write(row, col_idx, campaign_row['cpl'], formats['currency'])
            col_idx += 1
            worksheet.write(row, col_idx, campaign_row['roas'], formats['decimal'])
            col_idx += 1
            worksheet.write(row, col_idx, campaign_row.get('total_revenue', 0), formats['currency'])
            col_idx += 1
            worksheet.write(row, col_idx, campaign_row.get('contribution_margin', 0), formats['currency'])
            row += 1

        # Ajustar larguras (ajustado para nova estrutura de colunas)
        worksheet.set_column(0, 0, 18)  # Conta
        worksheet.set_column(1, 1, 60)  # Campanha
        worksheet.set_column(2, 2, 20)  # Campaign ID
        worksheet.set_column(3, 3, 18)  # Grupo
        worksheet.set_column(4, 17, 15)  # Outras métricas (Leads, LeadQualified, etc.)

    def _write_adsets_comparison(
        self,
        writer: pd.ExcelWriter,
        adsets_df: pd.DataFrame,
        formats: Dict
    ):
        """
        Escreve aba 'Comparação por Adsets' com formato similar à aba Campanhas.
        """
        worksheet = writer.book.add_worksheet('Comparação por Adsets')

        # Título
        worksheet.write(0, 0, '📊 COMPARAÇÃO POR ADSETS - MATCHED', formats['title'])
        worksheet.write(1, 0, 'Adsets com mesmo targeting e criativos', formats['subtitle'])

        row = 3

        # Cabeçalhos
        headers = [
            'Conta', 'Campanha', 'Campaign ID', 'Adset', 'Adset ID', 'Grupo',
            'Leads', 'Vendas', 'Taxa de conversão',
            'Valor gasto', 'CPL', 'ROAS',
            'Receita Total', 'Margem de contribuição'
        ]
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, formats['header'])
        row += 1

        # Escrever linhas dos adsets
        for _, adset_row in adsets_df.iterrows():
            col_idx = 0

            # Conta
            worksheet.write(row, col_idx, adset_row.get('Conta', ''), formats['text'])
            col_idx += 1

            # Campanha
            worksheet.write(row, col_idx, adset_row.get('Campanha', ''), formats['text'])
            col_idx += 1

            # Campaign ID
            worksheet.write(row, col_idx, str(adset_row.get('Campaign ID', '')), formats['text'])
            col_idx += 1

            # Adset
            worksheet.write(row, col_idx, adset_row.get('Adset', ''), formats['text'])
            col_idx += 1

            # Adset ID
            worksheet.write(row, col_idx, str(adset_row.get('Adset ID', '')), formats['text'])
            col_idx += 1

            # Grupo
            worksheet.write(row, col_idx, adset_row.get('Grupo', ''), formats['text'])
            col_idx += 1

            # Leads
            worksheet.write(row, col_idx, int(adset_row.get('Leads', 0)), formats['number'])
            col_idx += 1

            # Vendas
            worksheet.write(row, col_idx, int(adset_row.get('Vendas', 0)), formats['number'])
            col_idx += 1

            # Taxa de conversão
            # IMPORTANTE: O valor vem como percentual (1.5 = 1.5%)
            # Para formato Excel percent, sempre dividir por 100
            taxa = adset_row.get('Taxa de conversão', 0)
            worksheet.write(row, col_idx, taxa / 100 if taxa else 0, formats['percent'])
            col_idx += 1

            # Valor gasto
            worksheet.write(row, col_idx, adset_row.get('Valor gasto', 0), formats['currency'])
            col_idx += 1

            # CPL
            worksheet.write(row, col_idx, adset_row.get('CPL', 0), formats['currency'])
            col_idx += 1

            # ROAS
            worksheet.write(row, col_idx, adset_row.get('ROAS', 0), formats['decimal'])
            col_idx += 1

            # Receita Total
            worksheet.write(row, col_idx, adset_row.get('Receita Total', 0), formats['currency'])
            col_idx += 1

            # Margem de contribuição
            worksheet.write(row, col_idx, adset_row.get('Margem de contribuição', 0), formats['currency'])
            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 18)  # Conta
        worksheet.set_column(1, 1, 50)  # Campanha
        worksheet.set_column(2, 2, 20)  # Campaign ID
        worksheet.set_column(3, 3, 40)  # Adset
        worksheet.set_column(4, 4, 20)  # Adset ID
        worksheet.set_column(5, 5, 18)  # Grupo
        worksheet.set_column(6, 13, 15)  # Outras métricas

    def _write_ads_comparison(
        self,
        writer: pd.ExcelWriter,
        ads_df: pd.DataFrame,
        formats: Dict
    ):
        """
        Escreve aba 'Comparação por Ads' com formato similar à aba Adsets.
        """
        worksheet = writer.book.add_worksheet('Comparação por Ads')

        # Título
        worksheet.write(0, 0, '📊 COMPARAÇÃO POR ADS - MATCHED', formats['title'])
        worksheet.write(1, 0, 'Anúncios (criativos) com mesmo ad_code', formats['subtitle'])

        row = 3

        # Cabeçalhos (adicionar Campaign ID e Adset ID)
        headers = [
            'Campaign ID', 'Adset ID', 'Ad Code', 'Nome do Anúncio', 'Grupo',
            'Leads', 'Vendas', 'Taxa de conversão',
            'Valor gasto', 'CPL', 'ROAS',
            'Receita Total', 'Margem de contribuição'
        ]
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, formats['header'])
        row += 1

        # Escrever linhas dos ads
        for _, ad_row_series in ads_df.iterrows():
            # Convert Series to dict for safer access
            ad_row = ad_row_series.to_dict()

            col_idx = 0

            # Campaign ID
            worksheet.write(row, col_idx, str(ad_row.get('Campaign ID', '')), formats['text'])
            col_idx += 1

            # Adset ID
            worksheet.write(row, col_idx, str(ad_row.get('Adset ID', '')), formats['text'])
            col_idx += 1

            # Ad Code
            worksheet.write(row, col_idx, ad_row.get('Ad Code', ''), formats['text'])
            col_idx += 1

            # Nome do Anúncio
            worksheet.write(row, col_idx, ad_row.get('Nome do Anúncio', ''), formats['text'])
            col_idx += 1

            # Grupo
            worksheet.write(row, col_idx, ad_row.get('Grupo', ''), formats['text'])
            col_idx += 1

            # Leads
            worksheet.write(row, col_idx, int(ad_row.get('Leads', 0)), formats['number'])
            col_idx += 1

            # Vendas
            worksheet.write(row, col_idx, int(ad_row.get('Vendas', 0)), formats['number'])
            col_idx += 1

            # Taxa de conversão
            taxa = ad_row.get('Taxa de conversão', 0)
            worksheet.write(row, col_idx, taxa / 100 if taxa else 0, formats['percent'])
            col_idx += 1

            # Valor gasto
            worksheet.write(row, col_idx, ad_row.get('Valor gasto', 0), formats['currency'])
            col_idx += 1

            # CPL
            worksheet.write(row, col_idx, ad_row.get('CPL', 0), formats['currency'])
            col_idx += 1

            # ROAS
            worksheet.write(row, col_idx, ad_row.get('ROAS', 0), formats['decimal'])
            col_idx += 1

            # Receita Total
            worksheet.write(row, col_idx, ad_row.get('Receita Total', 0), formats['currency'])
            col_idx += 1

            # Margem de contribuição
            worksheet.write(row, col_idx, ad_row.get('Margem de contribuição', 0), formats['currency'])
            row += 1

        # Ajustar larguras (adicionar Campaign ID e Adset ID)
        worksheet.set_column(0, 0, 20)  # Campaign ID
        worksheet.set_column(1, 1, 20)  # Adset ID
        worksheet.set_column(2, 2, 12)  # Ad Code
        worksheet.set_column(3, 3, 40)  # Nome do Anúncio
        worksheet.set_column(4, 4, 18)  # Grupo
        worksheet.set_column(5, 12, 15)  # Outras métricas

    def _write_adset_aggregated(
        self,
        writer: pd.ExcelWriter,
        aggregated_df: pd.DataFrame,
        formats: Dict,
        sheet_name: str = 'Comparação Adsets'
    ):
        """
        Escreve aba 'Comparação Adsets' com comparação agregada de adsets matched.
        """
        worksheet = writer.book.add_worksheet(sheet_name)

        # Título
        worksheet.write(0, 0, '📊 COMPARAÇÃO AGREGADA - ADSETS MATCHED', formats['title'])
        worksheet.write(1, 0, 'Apenas adsets que aparecem em ML E controle (R$ 200+ gasto)', formats['subtitle'])

        # Cabeçalhos
        row = 3
        for col_num, col_name in enumerate(aggregated_df.columns):
            worksheet.write(row, col_num, col_name, formats['header'])
        row += 1

        # Dados
        for _, data_row in aggregated_df.iterrows():
            for col_num, col_name in enumerate(aggregated_df.columns):
                value = data_row[col_name]

                # Aplicar formato baseado no nome da coluna
                if 'Taxa Conversão' in col_name or '%' in col_name:
                    worksheet.write(row, col_num, value / 100 if value > 0 else 0, formats['percent'])
                elif any(term in col_name for term in ['Gasto', 'CPL', 'CPA', 'R$', 'Margem']):
                    worksheet.write(row, col_num, value if value else 0, formats['currency'])
                elif 'ROAS' in col_name:
                    worksheet.write(row, col_num, value if value else 0, formats['decimal'])
                elif any(term in col_name for term in ['Adsets', 'Leads', 'Vendas']):
                    worksheet.write(row, col_num, int(value) if value else 0, formats['number'])
                else:
                    worksheet.write(row, col_num, value, formats['text'])
            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 20)  # Tipo
        worksheet.set_column(1, len(aggregated_df.columns) - 1, 18)

    def _write_adset_detailed(
        self,
        writer: pd.ExcelWriter,
        detailed_df: pd.DataFrame,
        formats: Dict,
        sheet_name: str = 'Detalhes Adsets'
    ):
        """
        Escreve aba 'Detalhes Adsets' com comparação adset-a-adset.
        """
        worksheet = writer.book.add_worksheet(sheet_name)

        # Título
        worksheet.write(0, 0, '🔬 COMPARAÇÃO DETALHADA - ADSETS', formats['title'])
        worksheet.write(1, 0, 'Comparação lado-a-lado (ML vs Controle)', formats['subtitle'])

        # Cabeçalhos
        row = 3
        for col_num, col_name in enumerate(detailed_df.columns):
            worksheet.write(row, col_num, col_name, formats['header'])
        row += 1

        # Dados
        for _, data_row in detailed_df.iterrows():
            for col_num, col_name in enumerate(detailed_df.columns):
                value = data_row[col_name]

                # Aplicar formato baseado no nome da coluna
                if 'Taxa Conversão' in col_name or 'Conv %' in col_name or '%' in col_name:
                    worksheet.write(row, col_num, value / 100 if pd.notna(value) and value > 0 else 0, formats['percent'])
                elif any(term in col_name for term in ['Gasto', 'CPL', 'CPA', 'R$', 'Margem']):
                    worksheet.write(row, col_num, value if pd.notna(value) else 0, formats['currency'])
                elif 'ROAS' in col_name:
                    worksheet.write(row, col_num, value if pd.notna(value) else 0, formats['decimal'])
                elif any(term in col_name for term in ['Vendas', 'Leads']):
                    worksheet.write(row, col_num, int(value) if pd.notna(value) else 0, formats['number'])
                else:
                    worksheet.write(row, col_num, value if pd.notna(value) else '', formats['text'])
            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 60)  # Adset name
        worksheet.set_column(1, len(detailed_df.columns) - 1, 16)

    def _write_ad_aggregated(
        self,
        writer: pd.ExcelWriter,
        aggregated_df: pd.DataFrame,
        formats: Dict,
        sheet_name: str = 'Comparação Anúncios'
    ):
        """
        Escreve aba 'Comparação Anúncios' com comparação agregada de anúncios matched.
        """
        worksheet = writer.book.add_worksheet(sheet_name)

        # Título
        worksheet.write(0, 0, '📊 COMPARAÇÃO AGREGADA - ANÚNCIOS MATCHED', formats['title'])
        worksheet.write(1, 0, 'Apenas anúncios que aparecem em ML E controle', formats['subtitle'])

        # Cabeçalhos
        row = 3
        for col_num, col_name in enumerate(aggregated_df.columns):
            worksheet.write(row, col_num, col_name, formats['header'])
        row += 1

        # Dados
        for _, data_row in aggregated_df.iterrows():
            for col_num, col_name in enumerate(aggregated_df.columns):
                value = data_row[col_name]

                # Aplicar formato baseado no nome da coluna
                if 'Taxa Conversão' in col_name:
                    worksheet.write(row, col_num, value / 100, formats['percent'])
                elif any(term in col_name for term in ['Gasto', 'CPL']):
                    worksheet.write(row, col_num, value, formats['currency'])
                elif 'ROAS' in col_name:
                    worksheet.write(row, col_num, value, formats['decimal'])
                elif 'Anúncios' in col_name:
                    worksheet.write(row, col_num, int(value), formats['number'])
                else:
                    worksheet.write(row, col_num, value, formats['text'])
            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 20)  # Categoria
        worksheet.set_column(1, len(aggregated_df.columns) - 1, 18)

    def _write_ad_detailed(
        self,
        writer: pd.ExcelWriter,
        detailed_df: pd.DataFrame,
        formats: Dict,
        sheet_name: str = 'Detalhamento Anúncios'
    ):
        """
        Escreve aba 'Detalhamento Anúncios' com comparação anúncio-a-anúncio (top 20 por ROAS).
        """
        worksheet = writer.book.add_worksheet(sheet_name)

        # Título
        worksheet.write(0, 0, '🔬 COMPARAÇÃO DETALHADA - TOP 20 ANÚNCIOS', formats['title'])
        worksheet.write(1, 0, 'Ordenado por ROAS (ML)', formats['subtitle'])

        # Cabeçalhos
        row = 3
        for col_num, col_name in enumerate(detailed_df.columns):
            worksheet.write(row, col_num, col_name, formats['header'])
        row += 1

        # Dados
        for _, data_row in detailed_df.iterrows():
            for col_num, col_name in enumerate(detailed_df.columns):
                value = data_row[col_name]

                # Aplicar formato baseado no nome da coluna
                if 'Taxa Conversão' in col_name or 'Conv %' in col_name:
                    worksheet.write(row, col_num, value / 100, formats['percent'])
                elif any(term in col_name for term in ['Gasto', 'CPL', 'R$']):
                    worksheet.write(row, col_num, value, formats['currency'])
                elif 'ROAS' in col_name:
                    worksheet.write(row, col_num, value, formats['decimal'])
                elif any(term in col_name for term in ['Vendas', 'Leads']):
                    worksheet.write(row, col_num, int(value) if pd.notna(value) else 0, formats['number'])
                else:
                    worksheet.write(row, col_num, value, formats['text'])
            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 15)  # Ad Code
        worksheet.set_column(1, len(detailed_df.columns) - 1, 16)

    def _write_ad_all_summary(
        self,
        writer: pd.ExcelWriter,
        all_summary_df: pd.DataFrame,
        formats: Dict,
        sheet_name: str = 'Resumo Todos Anúncios'
    ):
        """
        Escreve aba 'Resumo Todos Anúncios' com comparação incluindo anúncios exclusivos.
        """
        worksheet = writer.book.add_worksheet(sheet_name)

        # Título
        worksheet.write(0, 0, '📋 RESUMO COMPLETO - TODOS OS ANÚNCIOS', formats['title'])
        worksheet.write(1, 0, 'Incluindo anúncios matched e exclusivos do ML', formats['subtitle'])

        # Cabeçalhos
        row = 3
        for col_num, col_name in enumerate(all_summary_df.columns):
            worksheet.write(row, col_num, col_name, formats['header'])
        row += 1

        # Dados
        for _, data_row in all_summary_df.iterrows():
            for col_num, col_name in enumerate(all_summary_df.columns):
                value = data_row[col_name]

                # Aplicar formato baseado no nome da coluna
                if 'Taxa Conversão' in col_name or '% Conv' in col_name:
                    worksheet.write(row, col_num, value / 100, formats['percent'])
                elif any(term in col_name for term in ['Gasto', 'CPL']):
                    worksheet.write(row, col_num, value, formats['currency'])
                elif 'ROAS' in col_name:
                    worksheet.write(row, col_num, value, formats['decimal'])
                elif any(term in col_name for term in ['Anúncios', 'Vendas', 'Count']):
                    worksheet.write(row, col_num, int(value) if pd.notna(value) else 0, formats['number'])
                else:
                    worksheet.write(row, col_num, value, formats['text'])
            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 20)  # Categoria/Tipo
        worksheet.set_column(1, len(all_summary_df.columns) - 1, 18)
