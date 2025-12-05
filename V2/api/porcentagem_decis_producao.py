#!/usr/bin/env python3
"""
Análise temporal da distribuição de decis D10 em produção vs treino
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

ARQUIVO_PRODUCAO = Path('/Users/ramonmoreira/Desktop/smart_ads/V2/files/validation/leads/[LF] Pesquisa - Mai25 - [LF] Pesquisa-16.csv')

# Baseline do treino
TREINO_TOTAL_LEADS = 32610
TREINO_D10_LEADS = 3261
TREINO_D10_PERCENTAGE = (TREINO_D10_LEADS / TREINO_TOTAL_LEADS) * 100  # 10%

# =============================================================================
# ANÁLISE
# =============================================================================

def analisar_distribuicao_temporal():
    """Analisa distribuição de D10 ao longo do tempo"""

    print("="*80)
    print("ANÁLISE DE DISTRIBUIÇÃO DE DECIS D10 - PRODUÇÃO VS TREINO")
    print("="*80)

    # Carregar dados de produção
    print(f"\n📊 Carregando dados de produção...")
    df = pd.read_csv(ARQUIVO_PRODUCAO)

    print(f"   Total de registros: {len(df):,}")

    # Converter coluna Data para datetime
    df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Filtrar apenas leads com decil preenchido
    df_com_decil = df[df['decil'].notna()].copy()
    print(f"   Leads com decil: {len(df_com_decil):,}")

    # Filtrar dados desde novembro 2024
    df_nov_onwards = df_com_decil[df_com_decil['Data'] >= '2024-11-01'].copy()
    print(f"   Leads desde Nov/2024: {len(df_nov_onwards):,}")

    # Adicionar coluna de semana
    df_nov_onwards['ano_semana'] = df_nov_onwards['Data'].dt.to_period('W').astype(str)
    df_nov_onwards['semana_inicio'] = df_nov_onwards['Data'].dt.to_period('W').apply(lambda r: r.start_time)

    # Agrupar por semana
    print(f"\n📈 DISTRIBUIÇÃO SEMANAL DE DECIS D10 (Nov/2024 até hoje)")
    print("="*80)

    semanas = df_nov_onwards.groupby(['ano_semana', 'semana_inicio']).agg({
        'decil': 'count',
        'Data': 'min'
    }).rename(columns={'decil': 'total_leads', 'Data': 'primeira_data'}).reset_index()

    # Contar D10 por semana
    d10_por_semana = df_nov_onwards[df_nov_onwards['decil'] == 'D10'].groupby('ano_semana').size().reset_index(name='d10_leads')

    # Merge
    semanas = semanas.merge(d10_por_semana, on='ano_semana', how='left')
    semanas['d10_leads'] = semanas['d10_leads'].fillna(0).astype(int)

    # Calcular percentual
    semanas['d10_percentage'] = (semanas['d10_leads'] / semanas['total_leads'] * 100).round(2)

    # Calcular diferença vs baseline treino
    semanas['diff_vs_treino'] = (semanas['d10_percentage'] - TREINO_D10_PERCENTAGE).round(2)

    # Ordenar por data
    semanas = semanas.sort_values('semana_inicio')

    # Imprimir tabela
    print(f"\n{'Semana':<20} {'Total Leads':<15} {'D10 Leads':<12} {'% D10':<10} {'Diff Treino':<12}")
    print("-"*80)

    for _, row in semanas.iterrows():
        semana_str = row['semana_inicio'].strftime('%Y-%m-%d')
        total = f"{int(row['total_leads']):,}"
        d10 = f"{int(row['d10_leads']):,}"
        pct = f"{row['d10_percentage']:.2f}%"
        diff = f"{row['diff_vs_treino']:+.2f}pp"

        # Marcar semanas com diferença significativa (>2pp)
        alerta = " ⚠️" if abs(row['diff_vs_treino']) > 2 else ""

        print(f"{semana_str:<20} {total:<15} {d10:<12} {pct:<10} {diff:<12}{alerta}")

    # Estatísticas gerais
    print("\n" + "="*80)
    print("📊 ESTATÍSTICAS GERAIS (Nov/2024 até hoje)")
    print("="*80)

    total_producao = len(df_nov_onwards)
    total_d10_producao = len(df_nov_onwards[df_nov_onwards['decil'] == 'D10'])
    pct_d10_producao = (total_d10_producao / total_producao * 100)

    print(f"\n🔹 TREINO (Test Set):")
    print(f"   Total de leads: {TREINO_TOTAL_LEADS:,}")
    print(f"   Leads D10: {TREINO_D10_LEADS:,}")
    print(f"   % D10: {TREINO_D10_PERCENTAGE:.2f}%")

    print(f"\n🔹 PRODUÇÃO (Nov/2024 até hoje):")
    print(f"   Total de leads: {total_producao:,}")
    print(f"   Leads D10: {total_d10_producao:,}")
    print(f"   % D10: {pct_d10_producao:.2f}%")

    print(f"\n🔹 COMPARAÇÃO:")
    diff_total = pct_d10_producao - TREINO_D10_PERCENTAGE
    print(f"   Diferença: {diff_total:+.2f} pontos percentuais")

    if abs(diff_total) <= 1:
        status = "✅ NORMAL - Distribuição similar ao treino"
    elif abs(diff_total) <= 2:
        status = "⚠️  ATENÇÃO - Pequena variação"
    else:
        status = "🚨 ALERTA - Variação significativa"

    print(f"   Status: {status}")

    # Distribuição completa de decis
    print(f"\n📊 DISTRIBUIÇÃO COMPLETA DE DECIS (Nov/2024 até hoje)")
    print("="*80)

    dist_decis = df_nov_onwards['decil'].value_counts().sort_index()

    print(f"\n{'Decil':<10} {'Total':<12} {'%':<10}")
    print("-"*40)

    for decil in sorted(dist_decis.index):
        count = dist_decis[decil]
        pct = (count / total_producao * 100)
        print(f"{decil:<10} {count:<12,} {pct:>6.2f}%")

    print("\n" + "="*80)

if __name__ == '__main__':
    analisar_distribuicao_temporal()
