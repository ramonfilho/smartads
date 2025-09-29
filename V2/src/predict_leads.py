#!/usr/bin/env python
"""
Script para executar predições de lead scoring.
Processa arquivo Excel e gera arquivo de saída com scores.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import LeadScoringPipeline


def main():
    # Parser de argumentos
    parser = argparse.ArgumentParser(description='Executar predições de Lead Scoring')
    parser.add_argument(
        'input_file',
        type=str,
        help='Arquivo Excel de entrada com os leads'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Arquivo Excel de saída (default: input_file_scored.xlsx)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='v1_devclub_rf_temporal',
        help='Nome do modelo a usar (default: v1_devclub_rf_temporal)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='Salvar apenas os top N leads com maior score'
    )

    args = parser.parse_args()

    # Validar arquivo de entrada
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Erro: Arquivo não encontrado: {input_path}")
        return 1

    # Definir arquivo de saída
    if args.output:
        output_path = Path(args.output)
    else:
        # Adicionar _scored antes da extensão
        output_path = input_path.parent / f"{input_path.stem}_scored.xlsx"

    print(f"📁 Arquivo de entrada: {input_path}")
    print(f"📁 Arquivo de saída: {output_path}")
    print(f"🤖 Modelo: {args.model}")
    print("-" * 50)

    try:
        # Inicializar pipeline
        print("🔧 Inicializando pipeline...")
        pipeline = LeadScoringPipeline(model_name=args.model)

        # Executar pipeline com predições
        print("🚀 Processando dados e fazendo predições...")
        result_df = pipeline.run(str(input_path), with_predictions=True)

        # Ordenar por score (maior primeiro)
        result_df = result_df.sort_values('lead_score', ascending=False)

        # Filtrar top N se solicitado
        if args.top_n:
            print(f"📊 Selecionando top {args.top_n} leads...")
            result_df = result_df.head(args.top_n)

        # Salvar resultado
        print(f"💾 Salvando resultado em {output_path}...")

        # Criar writer do Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Aba principal com todos os dados
            result_df.to_excel(writer, sheet_name='Leads Scored', index=False)

            # Aba de resumo por decil
            decil_summary = result_df.groupby('decil').agg({
                'lead_score': ['count', 'mean', 'min', 'max']
            }).round(4)
            decil_summary.columns = ['Quantidade', 'Score Médio', 'Score Mínimo', 'Score Máximo']
            decil_summary.to_excel(writer, sheet_name='Resumo por Decil')

            # Aba de metadados
            metadata = pd.DataFrame({
                'Informação': [
                    'Data de Processamento',
                    'Modelo Utilizado',
                    'Total de Leads Processados',
                    'Score Médio (%)',
                    'Score Mínimo (%)',
                    'Score Máximo (%)',
                    'Leads no Decil 10 (Melhor)',
                    'Leads no Decil 1 (Pior)'
                ],
                'Valor': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    args.model,
                    len(result_df),
                    f"{result_df['lead_score'].mean()*100:.2f}%",
                    f"{result_df['lead_score'].min()*100:.2f}%",
                    f"{result_df['lead_score'].max()*100:.2f}%",
                    len(result_df[result_df['decil'] == 10]),
                    len(result_df[result_df['decil'] == 1])
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadados', index=False)

        print("✅ Processamento concluído com sucesso!")
        print(f"📊 Total de leads processados: {len(result_df)}")
        print(f"📈 Score médio: {result_df['lead_score'].mean():.4f}")
        print(f"🎯 Leads no decil 10 (melhor): {len(result_df[result_df['decil'] == 10])}")

        return 0

    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())