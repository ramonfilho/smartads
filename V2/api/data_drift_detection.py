#!/usr/bin/env python3
"""
Análise de Feature Drift: Comparação entre período de treino e produção
"""
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

ARQUIVO_PRODUCAO = Path('/Users/ramonmoreira/Desktop/smart_ads/V2/files/validation/leads/[LF] Pesquisa - Mai25 - [LF] Pesquisa-16.csv')

# Datas de corte
CUT_DATE_TREINO = '2025-09-24'  # Dados até essa data = treino
PERIODO_PRODUCAO_INICIO = '2024-11-01'  # Nov/Dez 2024

# Features para analisar
FEATURES_CATEGORICAS = [
    'O seu gênero:',
    'Qual a sua idade?',
    'O que você faz atualmente?',
    'Atualmente, qual a sua faixa salarial?',
    'Você possui cartão de crédito?',
    'Já estudou programação?',
    'Você já fez/faz/pretende fazer faculdade?',
    'Já investiu em algum curso online para aprender uma nova forma de ganhar dinheiro?',
    'O que mais te chama atenção na profissão de Programador?',
    'O que mais você quer ver no evento?',
    'Source',
    'Medium'
]

# =============================================================================
# FUNÇÕES
# =============================================================================

def calcular_divergencia_kl(dist1, dist2):
    """Calcula divergência KL entre duas distribuições (simplificada)"""
    # Adicionar pequeno epsilon para evitar log(0)
    epsilon = 1e-10
    dist1 = np.array(dist1) + epsilon
    dist2 = np.array(dist2) + epsilon

    # Normalizar
    dist1 = dist1 / dist1.sum()
    dist2 = dist2 / dist2.sum()

    kl = np.sum(dist1 * np.log(dist1 / dist2))
    return kl

def comparar_distribuicoes(df_treino, df_producao, feature, top_n=None):
    """Compara distribuição de uma feature entre treino e produção"""

    # Contar valores em cada período
    dist_treino = df_treino[feature].value_counts(normalize=True).sort_index()
    dist_producao = df_producao[feature].value_counts(normalize=True).sort_index()

    # Criar DataFrame de comparação
    df_comp = pd.DataFrame({
        'Treino (%)': (dist_treino * 100).round(2),
        'Produção (%)': (dist_producao * 100).round(2)
    })

    # Calcular diferença
    df_comp['Diff (pp)'] = (df_comp['Produção (%)'] - df_comp['Treino (%)']).round(2)
    df_comp['Diff Abs (pp)'] = df_comp['Diff (pp)'].abs()

    # Preencher NaN com 0 para valores que não existem em um dos períodos
    df_comp = df_comp.fillna(0)

    # Ordenar por diferença absoluta
    df_comp = df_comp.sort_values('Diff Abs (pp)', ascending=False)

    # Limitar top N se especificado
    if top_n:
        df_comp = df_comp.head(top_n)

    return df_comp

def analisar_feature_drift():
    """Análise completa de feature drift"""

    print("="*100)
    print("ANÁLISE DE FEATURE DRIFT: TREINO VS PRODUÇÃO")
    print("="*100)

    # Carregar dados
    print(f"\n📊 Carregando dados...")
    df = pd.read_csv(ARQUIVO_PRODUCAO, low_memory=False)
    df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    print(f"   Total de registros: {len(df):,}")

    # Separar períodos
    df_treino = df[df['Data'] <= CUT_DATE_TREINO].copy()
    df_producao = df[
        (df['Data'] >= PERIODO_PRODUCAO_INICIO) &
        (df['decil'].notna())
    ].copy()

    print(f"\n📅 PERÍODOS:")
    print(f"   TREINO: Até {CUT_DATE_TREINO}")
    print(f"      Total de leads: {len(df_treino):,}")
    print(f"      Data mín: {df_treino['Data'].min()}")
    print(f"      Data máx: {df_treino['Data'].max()}")

    print(f"\n   PRODUÇÃO: Nov/Dez 2024 (desde {PERIODO_PRODUCAO_INICIO})")
    print(f"      Total de leads: {len(df_producao):,}")
    print(f"      Data mín: {df_producao['Data'].min()}")
    print(f"      Data máx: {df_producao['Data'].max()}")

    # Analisar cada feature
    print(f"\n" + "="*100)
    print("COMPARAÇÃO DE FEATURES")
    print("="*100)

    drift_summary = []

    for feature in FEATURES_CATEGORICAS:
        if feature not in df.columns:
            print(f"\n⚠️  Feature '{feature}' não encontrada nos dados")
            continue

        print(f"\n{'='*100}")
        print(f"📊 {feature.upper()}")
        print(f"{'='*100}")

        # Comparar distribuições
        top_n = 10 if feature in ['Medium', 'Campaign'] else None
        df_comp = comparar_distribuicoes(df_treino, df_producao, feature, top_n=top_n)

        # Imprimir tabela
        print(f"\n{df_comp.to_string()}")

        # Calcular drift score (máxima diferença absoluta)
        max_drift = df_comp['Diff Abs (pp)'].max()
        mean_drift = df_comp['Diff Abs (pp)'].mean()

        drift_summary.append({
            'Feature': feature,
            'Max Drift (pp)': max_drift,
            'Mean Drift (pp)': mean_drift,
            'Categoria Mais Afetada': df_comp.index[0] if len(df_comp) > 0 else 'N/A',
            'Diff Mais Afetada (pp)': df_comp['Diff (pp)'].iloc[0] if len(df_comp) > 0 else 0
        })

        # Alerta se drift significativo
        if max_drift > 5:
            print(f"\n🚨 ALERTA: Drift significativo detectado! (Max: {max_drift:.2f}pp)")
        elif max_drift > 2:
            print(f"\n⚠️  Drift moderado detectado (Max: {max_drift:.2f}pp)")
        else:
            print(f"\n✅ Drift baixo (Max: {max_drift:.2f}pp)")

    # Resumo geral de drift
    print(f"\n" + "="*100)
    print("📊 RESUMO DE FEATURE DRIFT")
    print("="*100)

    df_drift_summary = pd.DataFrame(drift_summary)
    df_drift_summary = df_drift_summary.sort_values('Max Drift (pp)', ascending=False)

    print(f"\n{df_drift_summary.to_string(index=False)}")

    # Top 5 features com mais drift
    print(f"\n" + "="*100)
    print("🔝 TOP 5 FEATURES COM MAIOR DRIFT")
    print("="*100)

    for i, row in df_drift_summary.head(5).iterrows():
        status = "🚨" if row['Max Drift (pp)'] > 5 else "⚠️" if row['Max Drift (pp)'] > 2 else "✅"
        print(f"\n{status} {row['Feature']}")
        print(f"   Max Drift: {row['Max Drift (pp)']:.2f}pp")
        print(f"   Mean Drift: {row['Mean Drift (pp)']:.2f}pp")
        print(f"   Categoria mais afetada: {row['Categoria Mais Afetada']} ({row['Diff Mais Afetada (pp)']:+.2f}pp)")

    # Análise de decis por período
    print(f"\n" + "="*100)
    print("📊 DISTRIBUIÇÃO DE DECIS POR PERÍODO")
    print("="*100)

    # Para treino, calcular os decis do test_set_predictions.csv
    print(f"\n🔹 PRODUÇÃO (Nov/Dez 2024):")
    dist_decis_prod = df_producao['decil'].value_counts(normalize=True).sort_index() * 100

    for decil, pct in dist_decis_prod.items():
        print(f"   {decil}: {pct:.2f}%")

    print(f"\n" + "="*100)

if __name__ == '__main__':
    analisar_feature_drift()
