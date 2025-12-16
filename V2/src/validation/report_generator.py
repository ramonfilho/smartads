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
        ad_level_comparisons: Optional[Dict] = None,
        ad_level_comparisons_adsets_iguais: Optional[Dict] = None,
        ad_level_comparisons_todos: Optional[Dict] = None,
        comparison_level: str = 'both'
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

        # Aba 2: Performance por Campanha
        logger.info("   Gerando aba: Performance por Campanha")
        self._write_performance_campanhas(writer, campaign_metrics, formats)

        # Aba 3: Detalhes das Conversões
        if sales_df is not None:
            logger.info("   Gerando aba: Detalhes das Conversões")
            self._write_conversions_detail(writer, matched_df, sales_df, formats)

        # Aba 4: Comparação Justa (PENÚLTIMA - detalhes dos matches ML vs Fair Control)
        # SEMPRE gerar aba (mesmo se vazia), para ter consistência no relatório
        logger.info("   Gerando aba: Comparação Justa")
        self._write_fair_comparison(writer, campaign_metrics, comparison_group_metrics, fair_comparison_info, formats)

        # Aba 4: Comparação ML (ÚLTIMA - resumo da comparação)
        logger.info("   Gerando aba: Comparação ML")
        self._write_comparacao_ml(writer, ml_comparison, comparison_group_metrics, formats)

        # Abas de comparação por anúncio (se disponíveis)
        try:
            from src.validation.fair_campaign_comparison import prepare_ad_comparison_for_excel

            # Se comparison_level == 'both', gerar abas separadas
            if comparison_level == 'both':
                # Adsets Iguais
                if ad_level_comparisons_adsets_iguais is not None:
                    logger.info("   Gerando abas: Evento ML (adsets iguais)")
                    excel_dfs_iguais = prepare_ad_comparison_for_excel(ad_level_comparisons_adsets_iguais)

                    if 'aggregated' in excel_dfs_iguais and not excel_dfs_iguais['aggregated'].empty:
                        logger.info("      - Agregação Matched Pairs (adsets iguais)")
                        self._write_ad_aggregated(writer, excel_dfs_iguais['aggregated'], formats,
                                                 sheet_name='📊 Adsets Iguais - Agregação')

                    if 'detailed' in excel_dfs_iguais and not excel_dfs_iguais['detailed'].empty:
                        logger.info("      - Detalhamento Anúncios (adsets iguais)")
                        self._write_ad_detailed(writer, excel_dfs_iguais['detailed'], formats,
                                              sheet_name='📋 Adsets Iguais - Detalhes')

                    if 'all_summary' in excel_dfs_iguais and not excel_dfs_iguais['all_summary'].empty:
                        logger.info("      - Resumo Todos Anúncios (adsets iguais)")
                        self._write_ad_all_summary(writer, excel_dfs_iguais['all_summary'], formats,
                                                  sheet_name='📝 Adsets Iguais - Resumo')

                # Todos
                if ad_level_comparisons_todos is not None:
                    logger.info("   Gerando abas: Evento ML (todos)")
                    excel_dfs_todos = prepare_ad_comparison_for_excel(ad_level_comparisons_todos)

                    if 'aggregated' in excel_dfs_todos and not excel_dfs_todos['aggregated'].empty:
                        logger.info("      - Agregação Matched Pairs (todos)")
                        self._write_ad_aggregated(writer, excel_dfs_todos['aggregated'], formats,
                                                sheet_name='📊 Todos - Agregação')

                    if 'detailed' in excel_dfs_todos and not excel_dfs_todos['detailed'].empty:
                        logger.info("      - Detalhamento Anúncios (todos)")
                        self._write_ad_detailed(writer, excel_dfs_todos['detailed'], formats,
                                             sheet_name='📋 Todos - Detalhes')

                    if 'all_summary' in excel_dfs_todos and not excel_dfs_todos['all_summary'].empty:
                        logger.info("      - Resumo Todos Anúncios (todos)")
                        self._write_ad_all_summary(writer, excel_dfs_todos['all_summary'], formats,
                                                 sheet_name='📝 Todos - Resumo')

            # Se apenas um nível, gerar abas padrão
            elif ad_level_comparisons is not None:
                level_name = "Evento ML (adsets iguais)" if comparison_level == 'adsets_iguais' else "Evento ML (todos)"
                logger.info(f"   Gerando abas: Comparação por Anúncio - {level_name}")
                excel_dfs = prepare_ad_comparison_for_excel(ad_level_comparisons)

                if 'aggregated' in excel_dfs and not excel_dfs['aggregated'].empty:
                    logger.info("      - Agregação Matched Pairs")
                    self._write_ad_aggregated(writer, excel_dfs['aggregated'], formats)

                if 'detailed' in excel_dfs and not excel_dfs['detailed'].empty:
                    logger.info("      - Detalhamento Anúncios")
                    self._write_ad_detailed(writer, excel_dfs['detailed'], formats)

                if 'all_summary' in excel_dfs and not excel_dfs['all_summary'].empty:
                    logger.info("      - Resumo Todos Anúncios")
                    self._write_ad_all_summary(writer, excel_dfs['all_summary'], formats)

        except Exception as e:
            logger.warning(f"   ⚠️ Erro ao gerar abas de comparação por anúncio: {e}")

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
        # 1. Leads Meta - da API do Meta
        total_leads_meta = overall_stats.get('total_leads_meta', 0)

        # 2. Leads no banco CAPI - total no PostgreSQL
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
            ['Leads no banco CAPI', capi_leads_total],
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
            'ml_type': 'Tipo de campanha',
            'campaign': 'Campanha',
            'optimization_goal': 'Evento de conversão',
            'leads_display': 'Leads',
            'LeadQualified': 'LeadQualified',
            'LeadQualifiedHighQuality': 'LeadQualifiedHighQuality',
            'Faixa A': 'Faixa A',
            'respostas_pesquisa': 'Respostas pesquisa',
            'taxa_resposta': '% de resposta',
            'conversions': 'Vendas',
            'conversion_rate': 'Taxa de conversão',
            'budget': 'Orçamento',
            'spend': 'Valor gasto',
            'cpl': 'CPL',
            'roas': 'ROAS',
            'total_revenue': 'Receita Total',
            'contribution_margin': 'Margem de contribuição',
        }

        # Ordem das colunas (campaign após ml_type, custom events após leads)
        # IMPORTANTE: Margem de contribuição vai para o final DEPOIS das colunas restantes
        column_order = [
            'ml_type', 'campaign', 'optimization_goal', 'leads_display',
            'LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A',
            'respostas_pesquisa', 'taxa_resposta', 'conversions', 'conversion_rate', 'budget',
            'spend', 'cpl', 'roas', 'total_revenue'
        ]

        # Colunas a excluir (incluindo total_conversion_events e num_creatives)
        exclude_cols = ['comparison_group', 'margin_percent', 'account_id', 'total_conversion_events', 'num_creatives', 'leads']

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

            # DEBUG: Verificar vendas antes de excluir campanhas EXCLUIR
            vendas_antes = account_campaigns['conversions'].sum()
            excluir_campanhas = account_campaigns[account_campaigns['ml_type'] == 'EXCLUIR']
            excluir_vendas = excluir_campanhas['conversions'].sum()

            if excluir_vendas > 0:
                logger.warning(f"   ⚠️  {int(excluir_vendas)} vendas em campanhas EXCLUIR (não mostradas na aba):")
                for _, row in excluir_campanhas[excluir_campanhas['conversions'] > 0].iterrows():
                    logger.warning(f"      • {int(row['conversions'])} vendas: {row['campaign'][:70]}")

            # IMPORTANTE: Excluir campanhas do tipo EXCLUIR (não são campanhas de captação)
            account_campaigns = account_campaigns[account_campaigns['ml_type'] != 'EXCLUIR'].copy()

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

                    # Escolher formato baseado no nome da coluna
                    if col_name in ['Taxa de conversão', '% de resposta']:
                        worksheet.write(current_row, col_num, value / 100, formats['percent'])
                    elif col_name in ['Valor gasto', 'Orçamento', 'CPL', 'Receita Total', 'Margem de contribuição']:
                        worksheet.write(current_row, col_num, value, formats['currency'])
                    elif col_name in ['ROAS']:
                        worksheet.write(current_row, col_num, value, formats['decimal'])
                    elif col_name in ['Leads', 'Respostas pesquisa', 'Vendas', 'LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A']:
                        worksheet.write(current_row, col_num, value, formats['number'])
                    else:
                        worksheet.write(current_row, col_num, value, formats['text'])
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
                    'ml_type': conv.get('ml_type', ''),
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
                'ml_type': tracking_data['ml_type'] if is_tracked else '',
                'data_captura': tracking_data['data_captura'] if is_tracked else ''
            }
            all_sales.append(sale_data)

        # Converter para DataFrame
        all_sales_df = pd.DataFrame(all_sales)

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
            'ml_type',
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
            'Tipo ML',
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
            worksheet.write(row_num, 5, row['ml_type'] if row['ml_type'] else '', formats['text'])
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
        comparison_group_metrics: pd.DataFrame,
        formats: Dict
    ):
        """
        Escreve aba 'Comparação ML' com tabela de comparação.

        Se comparison_group_metrics estiver disponível, usa comparação justa (ML vs Fair Control).
        Caso contrário, usa comparação total (COM ML vs SEM ML).
        """
        worksheet = writer.book.add_worksheet('Comparação ML')

        # Verificar se temos dados de comparação justa
        use_fair_comparison = (
            comparison_group_metrics is not None and
            not comparison_group_metrics.empty and
            'Eventos ML' in comparison_group_metrics['comparison_group'].values and
            'Controle' in comparison_group_metrics['comparison_group'].values
        )

        if use_fair_comparison:
            # COMPARAÇÃO JUSTA: Eventos ML vs Controle (campanhas duplicadas com mesmo setup)
            worksheet.write(0, 0, '⚖️ COMPARAÇÃO JUSTA: Eventos ML vs Controle', formats['title'])
            worksheet.write(1, 0, 'Apenas campanhas com MESMO budget e criativos (duplicadas)', formats['subtitle'])
            self._write_fair_comparison_table(worksheet, comparison_group_metrics, formats, start_row=3)
        else:
            # COMPARAÇÃO TOTAL: COM ML vs SEM ML (todas as campanhas)
            worksheet.write(0, 0, '⚖️ COMPARAÇÃO: COM ML vs SEM ML', formats['title'])
            self._write_total_comparison_table(worksheet, ml_comparison, formats, start_row=2)

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
        Escreve aba 'Comparação Justa' com lista detalhada de campanhas ML vs Fair Control matched.
        Usa o mesmo matching que já funciona nas outras abas (via comparison_group).
        """
        worksheet = writer.book.add_worksheet('Comparação Justa')

        # Título
        worksheet.write(0, 0, '🎯 COMPARAÇÃO JUSTA - EVENTOS ML vs CONTROLE', formats['title'])
        worksheet.write(1, 0, 'Lista de campanhas matched com MESMO budget e criativos', formats['subtitle'])

        row = 3

        # Verificar se temos métricas de comparação
        if campaign_metrics.empty or 'comparison_group' not in campaign_metrics.columns:
            # Mensagem simples quando não há matches
            worksheet.write(row, 0, 'Nenhuma campanha de controle encontrada no período.', formats['text'])
            return

        # Cabeçalhos (ordem consistente com aba Performance por Campanha)
        headers = [
            'Campanha', 'Grupo', 'Evento de conversão',
            'Leads', 'LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A',
            'Respostas pesquisa', '% de resposta',
            'Vendas', 'Taxa de conversão',
            'Orçamento', 'Valor gasto', 'CPL', 'ROAS', 'Receita Total', 'Margem de contribuição'
        ]
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, formats['header'])
        row += 1

        # Filtrar apenas campanhas ML (Eventos ML + Otimização ML) e Controle, e ordenar por grupo e nome
        fair_campaigns = campaign_metrics[
            campaign_metrics['comparison_group'].isin(['Eventos ML', 'Otimização ML', 'Controle'])
        ].sort_values(['comparison_group', 'campaign'])

        # Escrever linhas das campanhas
        for _, campaign_row in fair_campaigns.iterrows():
            col_idx = 0
            worksheet.write(row, col_idx, campaign_row['campaign'], formats['text'])
            col_idx += 1
            worksheet.write(row, col_idx, campaign_row['comparison_group'], formats['text'])
            col_idx += 1
            worksheet.write(row, col_idx, campaign_row.get('optimization_goal', '-'), formats['text'])
            col_idx += 1

            # Leads - usar total_conversion_events para campanhas com eventos customizados
            # Isso mostra o total correto (ex: 612 em vez de apenas 4)
            total_events = campaign_row.get('total_conversion_events', 0)
            standard_leads = campaign_row.get('leads', 0)
            # Se total_events existe e é maior que leads, usar total_events
            leads_to_show = int(total_events) if total_events > 0 else int(standard_leads)
            worksheet.write(row, col_idx, leads_to_show, formats['number'])
            col_idx += 1

            # Custom events
            worksheet.write(row, col_idx, int(campaign_row.get('LeadQualified', 0)), formats['number'])
            col_idx += 1
            worksheet.write(row, col_idx, int(campaign_row.get('LeadQualifiedHighQuality', 0)), formats['number'])
            col_idx += 1
            worksheet.write(row, col_idx, int(campaign_row.get('Faixa A', 0)), formats['number'])
            col_idx += 1

            worksheet.write(row, col_idx, int(campaign_row.get('respostas_pesquisa', 0)), formats['number'])
            col_idx += 1
            # taxa_resposta vem como número (65.67), dividir por 100 antes do formato percent
            worksheet.write(row, col_idx, campaign_row.get('taxa_resposta', 0) / 100, formats['percent'])
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

        # Ajustar larguras
        worksheet.set_column(0, 0, 60)  # Campanha
        worksheet.set_column(1, 1, 15)  # Grupo
        worksheet.set_column(2, 2, 30)  # Evento de conversão
        worksheet.set_column(3, 12, 15)  # Outras métricas

    def _write_ad_aggregated(
        self,
        writer: pd.ExcelWriter,
        aggregated_df: pd.DataFrame,
        formats: Dict,
        sheet_name: str = 'Agregação Matched Pairs'
    ):
        """
        Escreve aba 'Agregação Matched Pairs' com comparação agregada de anúncios matched.
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
