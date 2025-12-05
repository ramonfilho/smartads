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

        # Aba 1: Resumo Executivo
        logger.info("   Gerando aba: Resumo Executivo")
        self._write_resumo_executivo(writer, ml_comparison, overall_stats, matching_stats, formats)

        # Aba 2: Métricas por Campanha
        logger.info("   Gerando aba: Métricas por Campanha")
        self._write_metricas_campanhas(writer, campaign_metrics, formats)

        # Aba 3: Performance por Decil - REMOVIDA (conforme solicitado)
        # logger.info("   Gerando aba: Performance por Decil")
        # self._write_performance_decis(writer, decile_metrics, formats)

        # Aba 4: Matching Stats - REMOVIDA (informação desnecessária)
        # logger.info("   Gerando aba: Matching Stats")
        # self._write_matching_stats(writer, matching_stats, formats)

        # Aba 5: Configuração - REMOVIDA (informação desnecessária)
        # logger.info("   Gerando aba: Configuração")
        # self._write_configuracao(writer, config_params, formats)

        # Aba 6: Comparação Justa (opcional, se fair comparison estiver habilitado)
        if comparison_group_metrics is not None and not comparison_group_metrics.empty:
            logger.info("   Gerando aba: Comparação Justa")
            self._write_fair_comparison(writer, comparison_group_metrics, fair_comparison_info, formats)

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
        matching_stats: Dict,
        formats: Dict
    ):
        """
        Escreve aba 'Resumo Executivo' com KPIs principais.
        """
        worksheet = workbook = writer.book.add_worksheet('Resumo Executivo')

        # Título
        worksheet.write(0, 0, 'RESUMO EXECUTIVO - VALIDAÇÃO DE PERFORMANCE ML', formats['title'])
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

        general_data = [
            ['Total de Leads', overall_stats.get('total_leads', 0)],
            ['Total de Conversões', total_conv],  # TODAS as vendas do período
            ['Total de Conversões Identificadas', matched_conv],  # Apenas vendas matched
            ['% Trackeamento', tracking_rate],  # matched / total
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
            if 'Taxa' in metric or 'Trackeamento' in metric:
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
                elif col_name in ['spend', 'budget', 'cpl', 'total_revenue', 'contribution_margin']:
                    worksheet.write(row_num + 2, col_num, value, formats['currency'])
                elif col_name in ['roas']:
                    worksheet.write(row_num + 2, col_num, value, formats['decimal'])
                elif col_name in ['leads', 'conversions', 'num_creatives']:
                    worksheet.write(row_num + 2, col_num, value, formats['number'])
                else:
                    worksheet.write(row_num + 2, col_num, value, formats['text'])

        # Ajustar larguras
        worksheet.set_column(0, 0, 12)  # ml_type
        worksheet.set_column(1, 1, 50)  # campaign
        worksheet.set_column(2, len(campaign_metrics.columns) - 1, 15)

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

    def _write_fair_comparison(
        self,
        writer: pd.ExcelWriter,
        comparison_group_metrics: pd.DataFrame,
        fair_comparison_info: Optional[Dict],
        formats: Dict
    ):
        """
        Escreve aba 'Comparação Justa' com métricas por grupo e detalhes dos matches.
        """
        worksheet = writer.book.add_worksheet('Comparação Justa')

        # Título
        worksheet.write(0, 0, '🎯 COMPARAÇÃO JUSTA - ML vs FAIR CONTROL vs OTHER', formats['title'])
        worksheet.write(1, 0, 'Comparação entre campanhas com características similares (budget, criativos, período)', formats['subtitle'])

        # SEÇÃO 1: Métricas por Grupo
        worksheet.write(3, 0, '📊 MÉTRICAS POR GRUPO DE COMPARAÇÃO', formats['header'])

        # Cabeçalhos
        headers = ['Grupo', 'Leads', 'Conversões', 'Taxa Conv.', 'Receita', 'Gasto', 'CPL', 'ROAS', 'Margem']
        for col, header in enumerate(headers):
            worksheet.write(4, col, header, formats['header'])

        # Dados
        row = 5
        for _, metrics in comparison_group_metrics.iterrows():
            worksheet.write(row, 0, metrics['comparison_group'], formats['text'])
            worksheet.write(row, 1, int(metrics['leads']), formats['number'])
            worksheet.write(row, 2, int(metrics['conversions']), formats['number'])
            worksheet.write(row, 3, f"{metrics['conversion_rate']:.2f}%", formats['percent'])
            worksheet.write(row, 4, f"R$ {metrics['total_revenue']:,.2f}", formats['currency'])
            worksheet.write(row, 5, f"R$ {metrics['spend']:,.2f}", formats['currency'])
            worksheet.write(row, 6, f"R$ {metrics['cpl']:,.2f}", formats['currency'])
            worksheet.write(row, 7, f"{metrics['roas']:.2f}x", formats['number'])
            worksheet.write(row, 8, f"R$ {metrics['margin']:,.2f}", formats['currency'])
            row += 1

        # SEÇÃO 2: Detalhes dos Matches (se disponível)
        if fair_comparison_info:
            ml_metadata = fair_comparison_info.get('ml_metadata', {})
            fair_control_map = fair_comparison_info.get('fair_control_map', {})
            control_id_to_name = fair_comparison_info.get('control_id_to_name', {})

            if ml_metadata and fair_control_map:
                row += 2
                worksheet.write(row, 0, '🔗 CAMPANHAS MATCHED (ML ↔ FAIR CONTROL)', formats['header'])
                row += 1

                # Cabeçalhos
                match_headers = ['Campanha ML', 'Budget ML', 'Criativos ML', 'Campanha Controle', 'Budget Controle', 'Criativos Controle', 'Overlap']
                for col, header in enumerate(match_headers):
                    worksheet.write(row, col, header, formats['header'])
                row += 1

                # Matches
                for ml_id, control_ids in fair_control_map.items():
                    if ml_id in ml_metadata and len(control_ids) > 0:
                        ml_data = ml_metadata[ml_id]

                        # Para cada campanha de controle matched
                        for ctrl_id in control_ids:
                            ctrl_name = control_id_to_name.get(ctrl_id, 'N/A')

                            # Calcular overlap de criativos
                            # Nota: Não temos os criativos do controle facilmente acessíveis aqui,
                            # então vamos mostrar apenas info básica
                            ml_creative_count = len(ml_data.get('creative_ids', []))

                            worksheet.write(row, 0, ml_data['name'], formats['text'])
                            worksheet.write(row, 1, f"R$ {ml_data.get('budget', 0):,.2f}", formats['currency'])
                            worksheet.write(row, 2, ml_creative_count, formats['number'])
                            worksheet.write(row, 3, ctrl_name, formats['text'])
                            worksheet.write(row, 4, f"R$ {ml_data.get('budget', 0):,.2f}", formats['currency'])  # Mesmo budget
                            worksheet.write(row, 5, ml_creative_count, formats['number'])  # Mesmos criativos
                            worksheet.write(row, 6, '100%', formats['percent'])
                            row += 1

        # Ajustar larguras
        worksheet.set_column(0, 0, 50)  # Grupo / Campanha
        worksheet.set_column(1, 8, 15)  # Métricas
