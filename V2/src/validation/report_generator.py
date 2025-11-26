"""
Módulo para geração de relatórios Excel de validação de performance ML.

Gera Excel com 6 abas:
1. Resumo Executivo - Comparação COM_ML vs SEM_ML
2. Métricas por Campanha - Detalhamento por campanha
3. Performance por Decil - Real vs Esperado (Guru vs Guru+TMB)
4. Comparação ML vs Não-ML - Agregado consolidado
5. Matching Stats - Estatísticas de vinculação
6. Configuração - Parâmetros utilizados
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
        output_path: str
    ) -> str:
        """
        Gera relatório Excel completo com 6 abas.

        Args:
            campaign_metrics: DataFrame de CampaignMetricsCalculator
            decile_metrics: DataFrame de DecileMetricsCalculator
            ml_comparison: Dict de compare_ml_vs_non_ml()
            matching_stats: Dict de get_matching_stats()
            overall_stats: Dict de calculate_overall_stats()
            config_params: Dicionário com parâmetros da análise
            output_path: Caminho completo para salvar Excel

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

        # Aba 1: Resumo Executivo
        logger.info("   Gerando aba: Resumo Executivo")
        self._write_resumo_executivo(writer, ml_comparison, overall_stats, formats)

        # Aba 2: Métricas por Campanha
        logger.info("   Gerando aba: Métricas por Campanha")
        self._write_metricas_campanhas(writer, campaign_metrics, formats)

        # Aba 3: Performance por Decil (Guru vs Guru+TMB)
        logger.info("   Gerando aba: Performance por Decil")
        self._write_performance_decis(writer, decile_metrics, formats)

        # Aba 4: Comparação ML vs Não-ML
        logger.info("   Gerando aba: Comparação ML")
        self._write_comparacao_ml(writer, ml_comparison, formats)

        # Aba 5: Matching Stats
        logger.info("   Gerando aba: Matching Stats")
        self._write_matching_stats(writer, matching_stats, formats)

        # Aba 6: Configuração
        logger.info("   Gerando aba: Configuração")
        self._write_configuracao(writer, config_params, formats)

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

    def _write_resumo_executivo(
        self,
        writer: pd.ExcelWriter,
        ml_comparison: Dict,
        overall_stats: Dict,
        formats: Dict
    ):
        """
        Escreve aba 'Resumo Executivo' com KPIs principais.
        """
        worksheet = workbook = writer.book.add_worksheet('Resumo Executivo')

        # Título
        worksheet.write(0, 0, 'RESUMO EXECUTIVO - VALIDAÇÃO DE PERFORMANCE ML', formats['title'])
        worksheet.write(1, 0, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", formats['subtitle'])

        # Estatísticas Gerais
        row = 3
        worksheet.write(row, 0, '📊 ESTATÍSTICAS GERAIS', formats['subtitle'])
        row += 1

        general_data = [
            ['Total de Leads', overall_stats.get('total_leads', 0)],
            ['Total de Conversões', overall_stats.get('total_conversions', 0)],
            ['Taxa de Conversão Geral', overall_stats.get('conversion_rate', 0) / 100],
            ['Receita Total', overall_stats.get('total_revenue', 0)],
            ['Gasto Total', overall_stats.get('total_spend', 0)],
            ['ROAS Geral', overall_stats.get('roas', 0)],
            ['Margem Total', overall_stats.get('margin', 0)],
            ['Conversões Guru', overall_stats.get('conversions_guru', 0)],
            ['Conversões TMB', overall_stats.get('conversions_tmb', 0)],
        ]

        for metric, value in general_data:
            worksheet.write(row, 0, metric, formats['text'])
            if 'Taxa' in metric:
                worksheet.write(row, 1, value, formats['percent'])
            elif 'Receita' in metric or 'Gasto' in metric or 'Margem' in metric:
                worksheet.write(row, 1, value, formats['currency'])
            elif 'ROAS' in metric:
                worksheet.write(row, 1, value, formats['decimal'])
            else:
                worksheet.write(row, 1, value, formats['number'])
            row += 1

        # Comparação COM ML vs SEM ML
        row += 2
        worksheet.write(row, 0, '⚖️ COMPARAÇÃO: COM ML vs SEM ML', formats['subtitle'])
        row += 1

        # Headers
        worksheet.write(row, 0, 'Métrica', formats['header'])
        worksheet.write(row, 1, 'COM ML', formats['header_green'])
        worksheet.write(row, 2, 'SEM ML', formats['header_red'])
        worksheet.write(row, 3, 'Diferença %', formats['header'])
        row += 1

        com_ml = ml_comparison.get('com_ml', {})
        sem_ml = ml_comparison.get('sem_ml', {})
        diff = ml_comparison.get('difference', {})

        comparison_data = [
            ('Total de Leads', com_ml.get('leads', 0), sem_ml.get('leads', 0), None, 'number'),
            ('Conversões', com_ml.get('conversions', 0), sem_ml.get('conversions', 0), None, 'number'),
            ('Taxa Conversão', com_ml.get('conversion_rate', 0) / 100, sem_ml.get('conversion_rate', 0) / 100, diff.get('conversion_rate_diff', 0) / 100, 'percent'),
            ('Receita Total', com_ml.get('revenue', 0), sem_ml.get('revenue', 0), None, 'currency'),
            ('Gasto Total', com_ml.get('spend', 0), sem_ml.get('spend', 0), None, 'currency'),
            ('CPL', com_ml.get('cpl', 0), sem_ml.get('cpl', 0), None, 'currency'),
            ('ROAS', com_ml.get('roas', 0), sem_ml.get('roas', 0), diff.get('roas_diff', 0) / 100, 'decimal'),
            ('Margem Contribuição', com_ml.get('margin', 0), sem_ml.get('margin', 0), diff.get('margin_diff', 0) / 100, 'currency'),
        ]

        for metric, com_value, sem_value, diff_value, fmt_type in comparison_data:
            worksheet.write(row, 0, metric, formats['text'])

            # COM ML
            worksheet.write(row, 1, com_value, formats[fmt_type])

            # SEM ML
            worksheet.write(row, 2, sem_value, formats[fmt_type])

            # Diferença
            if diff_value is not None:
                cell_format = formats['positive'] if diff_value > 0 else formats['negative']
                if fmt_type == 'percent':
                    worksheet.write(row, 3, diff_value, cell_format)
                else:
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

    def _write_metricas_campanhas(
        self,
        writer: pd.ExcelWriter,
        campaign_metrics: pd.DataFrame,
        formats: Dict
    ):
        """
        Escreve aba 'Métricas por Campanha' com tabela detalhada.
        """
        if campaign_metrics.empty:
            # Criar sheet vazia com mensagem
            worksheet = writer.book.add_worksheet('Métricas por Campanha')
            worksheet.write(0, 0, 'Nenhuma métrica de campanha disponível', formats['subtitle'])
            return

        # Escrever DataFrame
        campaign_metrics.to_excel(writer, sheet_name='Métricas por Campanha', index=False, startrow=1)

        worksheet = writer.sheets['Métricas por Campanha']

        # Título
        worksheet.write(0, 0, '📊 MÉTRICAS DETALHADAS POR CAMPANHA', formats['title'])

        # Aplicar formatos aos headers (row 1)
        for col_num, col_name in enumerate(campaign_metrics.columns):
            worksheet.write(1, col_num, col_name, formats['header'])

        # Aplicar formatos às células de dados
        for row_num in range(len(campaign_metrics)):
            for col_num, col_name in enumerate(campaign_metrics.columns):
                value = campaign_metrics.iloc[row_num, col_num]

                # Escolher formato baseado no nome da coluna
                if col_name in ['conversion_rate', 'margin_percent']:
                    # Converter % para decimal
                    worksheet.write(row_num + 2, col_num, value / 100, formats['percent'])
                elif col_name in ['spend', 'cpl', 'total_revenue', 'contribution_margin']:
                    worksheet.write(row_num + 2, col_num, value, formats['currency'])
                elif col_name in ['roas']:
                    worksheet.write(row_num + 2, col_num, value, formats['decimal'])
                elif col_name in ['leads', 'conversions']:
                    worksheet.write(row_num + 2, col_num, value, formats['number'])
                else:
                    worksheet.write(row_num + 2, col_num, value, formats['text'])

        # Ajustar larguras
        worksheet.set_column(0, 0, 12)  # ml_type
        worksheet.set_column(1, 1, 50)  # campaign
        worksheet.set_column(2, len(campaign_metrics.columns) - 1, 15)

    def _write_performance_decis(
        self,
        writer: pd.ExcelWriter,
        decile_metrics: pd.DataFrame,
        formats: Dict
    ):
        """
        Escreve aba 'Performance por Decil' com métricas Guru vs Guru+TMB.

        IMPORTANTE: Inclui colunas separadas para Guru (treinamento) e Total (generalização).
        """
        if decile_metrics.empty:
            worksheet = writer.book.add_worksheet('Performance por Decil')
            worksheet.write(0, 0, 'Nenhuma métrica de decil disponível', formats['subtitle'])
            return

        # Escrever DataFrame
        decile_metrics.to_excel(writer, sheet_name='Performance por Decil', index=False, startrow=3)

        worksheet = writer.sheets['Performance por Decil']

        # Título e contexto
        worksheet.write(0, 0, '📈 PERFORMANCE POR DECIL (D1-D10)', formats['title'])
        worksheet.write(1, 0, 'IMPORTANTE: Modelo treinado APENAS com vendas Guru', formats['subtitle'])
        worksheet.write(2, 0, '→ Guru = Dados de treinamento | Total = Guru + TMB (generalização)', formats['text'])

        # Aplicar formatos aos headers (row 3)
        for col_num, col_name in enumerate(decile_metrics.columns):
            # Usar cores diferentes para Guru vs Total
            if 'guru' in col_name.lower() and col_name != 'decile':
                worksheet.write(3, col_num, col_name, formats['header_green'])
            elif 'total' in col_name.lower():
                worksheet.write(3, col_num, col_name, formats['header_red'])
            else:
                worksheet.write(3, col_num, col_name, formats['header'])

        # Aplicar formatos às células de dados
        for row_num in range(len(decile_metrics)):
            for col_num, col_name in enumerate(decile_metrics.columns):
                value = decile_metrics.iloc[row_num, col_num]

                # Escolher formato baseado no nome da coluna
                if 'conversion_rate' in col_name or 'expected' in col_name:
                    # Converter % para decimal
                    worksheet.write(row_num + 4, col_num, value / 100, formats['percent'])
                elif 'performance_ratio' in col_name:
                    worksheet.write(row_num + 4, col_num, value, formats['decimal'])
                elif 'revenue' in col_name:
                    worksheet.write(row_num + 4, col_num, value, formats['currency'])
                elif col_name in ['leads', 'conversions_guru', 'conversions_total']:
                    worksheet.write(row_num + 4, col_num, value, formats['number'])
                else:
                    worksheet.write(row_num + 4, col_num, value, formats['text'])

        # Adicionar linha de totais
        total_row = len(decile_metrics) + 5
        worksheet.write(total_row, 0, 'TOTAL', formats['header'])

        # Somar colunas relevantes
        worksheet.write(total_row, 1, decile_metrics['leads'].sum(), formats['number'])
        worksheet.write(total_row, 2, decile_metrics['conversions_guru'].sum(), formats['number'])
        worksheet.write(total_row, 3, decile_metrics['conversions_total'].sum(), formats['number'])
        worksheet.write(total_row, 9, decile_metrics['revenue_guru'].sum(), formats['currency'])
        worksheet.write(total_row, 10, decile_metrics['revenue_total'].sum(), formats['currency'])

        # Ajustar larguras
        worksheet.set_column(0, 0, 10)  # decile
        worksheet.set_column(1, len(decile_metrics.columns) - 1, 16)

    def _write_comparacao_ml(
        self,
        writer: pd.ExcelWriter,
        ml_comparison: Dict,
        formats: Dict
    ):
        """
        Escreve aba 'Comparação ML' com tabela agregada.
        """
        worksheet = writer.book.add_worksheet('Comparação ML')

        # Título
        worksheet.write(0, 0, '⚖️ COMPARAÇÃO AGREGADA: COM ML vs SEM ML', formats['title'])

        # Headers
        row = 2
        worksheet.write(row, 0, 'Métrica', formats['header'])
        worksheet.write(row, 1, 'COM ML', formats['header_green'])
        worksheet.write(row, 2, 'SEM ML', formats['header_red'])
        worksheet.write(row, 3, 'Diferença', formats['header'])
        worksheet.write(row, 4, 'Diferença %', formats['header'])
        row += 1

        com_ml = ml_comparison.get('com_ml', {})
        sem_ml = ml_comparison.get('sem_ml', {})
        diff = ml_comparison.get('difference', {})

        comparison_data = [
            ('Total de Leads', com_ml.get('leads', 0), sem_ml.get('leads', 0), None, None, 'number'),
            ('Conversões', com_ml.get('conversions', 0), sem_ml.get('conversions', 0), None, None, 'number'),
            ('Taxa Conversão (%)', com_ml.get('conversion_rate', 0), sem_ml.get('conversion_rate', 0),
             com_ml.get('conversion_rate', 0) - sem_ml.get('conversion_rate', 0),
             diff.get('conversion_rate_diff', 0), 'decimal'),
            ('Receita Total', com_ml.get('revenue', 0), sem_ml.get('revenue', 0),
             com_ml.get('revenue', 0) - sem_ml.get('revenue', 0), None, 'currency'),
            ('Gasto Total', com_ml.get('spend', 0), sem_ml.get('spend', 0),
             com_ml.get('spend', 0) - sem_ml.get('spend', 0), None, 'currency'),
            ('CPL', com_ml.get('cpl', 0), sem_ml.get('cpl', 0),
             com_ml.get('cpl', 0) - sem_ml.get('cpl', 0), None, 'currency'),
            ('ROAS', com_ml.get('roas', 0), sem_ml.get('roas', 0),
             com_ml.get('roas', 0) - sem_ml.get('roas', 0),
             diff.get('roas_diff', 0), 'decimal'),
            ('Margem Contribuição', com_ml.get('margin', 0), sem_ml.get('margin', 0),
             com_ml.get('margin', 0) - sem_ml.get('margin', 0),
             diff.get('margin_diff', 0), 'currency'),
        ]

        for metric, com_value, sem_value, diff_value, diff_pct, fmt_type in comparison_data:
            worksheet.write(row, 0, metric, formats['text'])
            worksheet.write(row, 1, com_value, formats[fmt_type])
            worksheet.write(row, 2, sem_value, formats[fmt_type])

            if diff_value is not None:
                cell_format = formats['positive'] if diff_value > 0 else formats['negative']
                worksheet.write(row, 3, diff_value, cell_format)
            else:
                worksheet.write(row, 3, '-', formats['text'])

            if diff_pct is not None:
                cell_format = formats['positive'] if diff_pct > 0 else formats['negative']
                worksheet.write(row, 4, diff_pct / 100, cell_format)
            else:
                worksheet.write(row, 4, '-', formats['text'])

            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 25)
        worksheet.set_column(1, 4, 18)

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
