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
        fair_comparison_info: Optional[Dict] = None
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

        # Aba 3: Comparação Justa (PENÚLTIMA - detalhes dos matches ML vs Fair Control)
        # SEMPRE gerar aba (mesmo se vazia), para ter consistência no relatório
        logger.info("   Gerando aba: Comparação Justa")
        self._write_fair_comparison(writer, campaign_metrics, comparison_group_metrics, fair_comparison_info, formats)

        # Aba 4: Comparação ML (ÚLTIMA - resumo da comparação)
        logger.info("   Gerando aba: Comparação ML")
        self._write_comparacao_ml(writer, ml_comparison, comparison_group_metrics, formats)

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

        worksheet.write(row, 0, 'Período de Captação (Leads/Campanhas)', formats['text'])
        worksheet.write(row, 1, f"{lead_start} a {lead_end}", formats['text'])
        row += 1

        worksheet.write(row, 0, 'Período de Vendas (Matching)', formats['text'])
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
            ['Vendas', total_vendas],
            ['Vendas identificadas', vendas_identificadas],
            ['% de trackeamento', pct_trackeamento],
            ['Taxa de Conversão Geral', overall_stats.get('conversion_rate', 0) / 100],
            ['Receita Total', overall_stats.get('total_revenue', 0)],
            ['Gasto Total', overall_stats.get('total_spend', 0)],
            ['ROAS Geral', overall_stats.get('roas', 0)],
            ['Margem Total', overall_stats.get('margin', 0)],
            ['Conversões Guru', overall_stats.get('conversions_guru_total', 0)],
            ['Conversões Identificadas Guru', overall_stats.get('conversions_guru_matched', 0)],
            ['Conversões TMB', overall_stats.get('conversions_tmb_total', 0)],
            ['Conversões Identificadas TMB', overall_stats.get('conversions_tmb_matched', 0)],
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
        Escreve aba 'Performance por Campanha' com tabela detalhada.
        """
        if campaign_metrics.empty:
            # Criar sheet vazia com mensagem
            worksheet = writer.book.add_worksheet('Performance por Campanha')
            worksheet.write(0, 0, 'Nenhuma métrica de campanha disponível', formats['subtitle'])
            return

        # Reorganizar e renomear colunas
        column_mapping = {
            'ml_type': 'Tipo de campanha',
            'campaign': 'Campanha',
            'optimization_goal': 'Evento de conversão',
            'leads': 'Leads',
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
            'contribution_margin': 'Margem de contribuição',
        }

        # Ordem das colunas (campaign após ml_type, custom events após leads)
        column_order = [
            'ml_type', 'campaign', 'optimization_goal', 'leads',
            'LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A',
            'respostas_pesquisa', 'taxa_resposta', 'conversions', 'conversion_rate', 'budget',
            'spend', 'cpl', 'roas', 'contribution_margin'
        ]

        # Colunas a excluir
        exclude_cols = ['comparison_group', 'margin_percent']

        # Adicionar colunas restantes que não estão na lista (exceto as excluídas)
        remaining_cols = [
            col for col in campaign_metrics.columns
            if col not in column_order and col not in exclude_cols
        ]
        final_column_order = column_order + remaining_cols

        # Reordenar DataFrame mantendo apenas colunas que existem
        existing_cols = [col for col in final_column_order if col in campaign_metrics.columns]
        campaign_metrics_ordered = campaign_metrics[existing_cols].copy()

        # Renomear colunas para nomes em português
        campaign_metrics_ordered.rename(columns=column_mapping, inplace=True)

        # Escrever DataFrame
        campaign_metrics_ordered.to_excel(writer, sheet_name='Performance por Campanha', index=False, startrow=1)

        worksheet = writer.sheets['Performance por Campanha']

        # Título
        worksheet.write(0, 0, '📊 PERFORMANCE DETALHADA POR CAMPANHA', formats['title'])

        # Aplicar formatos aos headers (row 1)
        for col_num, col_name in enumerate(campaign_metrics_ordered.columns):
            worksheet.write(1, col_num, col_name, formats['header'])

        # Aplicar formatos às células de dados
        for row_num in range(len(campaign_metrics_ordered)):
            for col_num, col_name in enumerate(campaign_metrics_ordered.columns):
                value = campaign_metrics_ordered.iloc[row_num, col_num]

                # Escolher formato baseado no nome da coluna
                if col_name in ['Taxa de conversão', '% de resposta']:
                    # Converter % para decimal
                    worksheet.write(row_num + 2, col_num, value / 100, formats['percent'])
                elif col_name in ['Valor gasto', 'Orçamento', 'CPL', 'Margem de contribuição']:
                    worksheet.write(row_num + 2, col_num, value, formats['currency'])
                elif col_name in ['ROAS']:
                    worksheet.write(row_num + 2, col_num, value, formats['decimal'])
                elif col_name in ['Leads', 'Respostas pesquisa', 'Vendas', 'LeadQualified', 'LeadQualifiedHighQuality', 'Faixa A']:
                    # Sempre mostrar o número, mesmo que seja 0
                    worksheet.write(row_num + 2, col_num, value, formats['number'])
                else:
                    worksheet.write(row_num + 2, col_num, value, formats['text'])

        # Ajustar larguras
        worksheet.set_column(0, 0, 18)  # Tipo de campanha
        worksheet.set_column(1, 1, 30)  # Evento de conversão

        worksheet.set_column(2, len(campaign_metrics_ordered.columns) - 1, 15)

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
            'ML' in comparison_group_metrics['comparison_group'].values and
            'Fair Control' in comparison_group_metrics['comparison_group'].values
        )

        if use_fair_comparison:
            # COMPARAÇÃO JUSTA: ML vs Fair Control (campanhas duplicadas com mesmo setup)
            worksheet.write(0, 0, '⚖️ COMPARAÇÃO JUSTA: ML vs Fair Control', formats['title'])
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
        """Escreve tabela de comparação justa (ML vs Fair Control)."""
        # Filtrar apenas ML e Fair Control
        ml_data = comparison_group_metrics[comparison_group_metrics['comparison_group'] == 'ML']
        fc_data = comparison_group_metrics[comparison_group_metrics['comparison_group'] == 'Fair Control']

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
        worksheet.write(row, 1, 'ML', formats['header_green'])
        worksheet.write(row, 2, 'Fair Control', formats['header_red'])
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
            winner_text = f"🏆 VENCEDOR: ML (ROAS {diff_pct:.1f}% maior)"
            worksheet.write(row, 0, winner_text, formats['header_green'])
        elif fc_roas > ml_roas:
            diff_pct = abs(calc_diff_pct(ml_roas, fc_roas))
            winner_text = f"⚠️ VENCEDOR: Fair Control (ROAS {diff_pct:.1f}% maior)"
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
        worksheet.write(0, 0, '🎯 COMPARAÇÃO JUSTA - ML vs FAIR CONTROL', formats['title'])
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
            'Orçamento', 'Valor gasto', 'CPL', 'ROAS', 'Margem de contribuição'
        ]
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, formats['header'])
        row += 1

        # Filtrar apenas campanhas ML e Fair Control, e ordenar por grupo e nome
        fair_campaigns = campaign_metrics[
            campaign_metrics['comparison_group'].isin(['ML', 'Fair Control'])
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

            # Leads - sempre mostrar o número
            leads_val = campaign_row.get('leads', 0)
            worksheet.write(row, col_idx, int(leads_val), formats['number'])
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
            worksheet.write(row, col_idx, campaign_row.get('contribution_margin', 0), formats['currency'])
            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 60)  # Campanha
        worksheet.set_column(1, 1, 15)  # Grupo
        worksheet.set_column(2, 2, 30)  # Evento de conversão
        worksheet.set_column(3, 12, 15)  # Outras métricas
