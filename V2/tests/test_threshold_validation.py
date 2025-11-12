"""
Test de validação de thresholds fixos vs pd.qcut.

Usa o pipeline de produção existente para:
1. Processar CSV de leads
2. Obter scores do modelo
3. Comparar distribuição de decis: pd.qcut (atual) vs thresholds fixos (proposto)
4. Anal isar para: hoje, ontem, últimos 7 dias
"""
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_pipeline import LeadScoringPipeline
from src.model.decil_thresholds import atribuir_decis_batch, comparar_distribuicoes


def carregar_thresholds():
    """Carrega thresholds do metadata do modelo."""
    metadata_path = "files/20251111_212345/model_metadata_v1_devclub_rf_temporal_single.json"

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata['decil_thresholds']['thresholds']


def processar_leads(csv_path, model_path):
    """Processa leads e retorna DataFrame com scores."""
    print(f"\n{'='*80}")
    print(f"PROCESSANDO: {csv_path}")
    print(f"{'='*80}")

    # Carregar CSV diretamente com encoding especificado
    print("Carregando CSV...")
    df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
    print(f"✓ {len(df)} leads carregados")

    # Salvar coluna Data antes do preprocessing (será removida)
    df_dates = df[['Data']].copy()
    df_dates['Data'] = pd.to_datetime(df_dates['Data'])

    # Usar pipeline de produção
    pipeline = LeadScoringPipeline(
        model_name="v1_devclub_rf_temporal_single",
        model_path=model_path
    )

    # Processar com dados já carregados
    pipeline.data = df
    pipeline.original_data = df.copy()
    pipeline.preprocess()
    df_with_scores = pipeline.predict()

    # Adicionar coluna Data de volta
    df_with_scores['Data'] = df_dates['Data'].values

    return df_with_scores


def comparar_metodos(df, thresholds, periodo_nome):
    """Compara pd.qcut vs thresholds fixos."""
    print(f"\n{'='*80}")
    print(f"ANÁLISE: {periodo_nome.upper()}")
    print(f"{'='*80}")

    n_leads = len(df)
    scores = df['lead_score'].values

    print(f"Total de leads: {n_leads}")
    print(f"Score min: {scores.min():.4f}")
    print(f"Score max: {scores.max():.4f}")
    print(f"Score mean: {scores.mean():.4f}")
    print(f"Valores únicos: {len(set(scores))}")

    # Método 1: pd.qcut (atual - dinâmico)
    print(f"\n📊 MÉTODO 1: pd.qcut (DINÂMICO - ATUAL)")
    print("-" * 80)

    try:
        decis_qcut = pd.qcut(
            scores,
            q=10,
            labels=[f'D{i}' for i in range(1, 11)],
            duplicates='drop'
        )

        dist_qcut = decis_qcut.value_counts().sort_index()
        print("Distribuição:")
        for decil, count in dist_qcut.items():
            pct = count / n_leads * 100
            print(f"  {decil}: {count:5d} ({pct:5.1f}%)")

    except Exception as e:
        print(f"❌ ERRO: {e}")
        decis_qcut = None

    # Método 2: Thresholds fixos (proposto)
    print(f"\n📊 MÉTODO 2: THRESHOLDS FIXOS (PROPOSTO)")
    print("-" * 80)

    decis_threshold = atribuir_decis_batch(scores, thresholds)
    dist_threshold = pd.Series(decis_threshold).value_counts().sort_index()

    print("Distribuição:")
    for decil in [f'D{i}' for i in range(1, 11)]:
        count = dist_threshold.get(decil, 0)
        pct = count / n_leads * 100
        print(f"  {decil}: {count:5d} ({pct:5.1f}%)")

    # Análise de desigualdade
    print(f"\n📈 ANÁLISE DE DESIGUALDADE")
    print("-" * 80)

    cv = dist_threshold.std() / dist_threshold.mean()
    top3_pct = dist_threshold.tail(3).sum() / n_leads * 100
    max_pct = dist_threshold.max() / n_leads * 100
    decis_vazios = [f'D{i}' for i in range(1, 11) if f'D{i}' not in dist_threshold.index]

    print(f"Coeficiente de variação: {cv:.3f}")
    print(f"Concentração top 3: {top3_pct:.1f}%")
    print(f"Max em um decil: {max_pct:.1f}% ({dist_threshold.idxmax()})")
    print(f"Decis vazios: {len(decis_vazios)}")

    if decis_vazios:
        print(f"  ⚠️  Decis sem leads: {', '.join(decis_vazios)}")

    # Comparação
    if decis_qcut is not None:
        print(f"\n📊 COMPARAÇÃO: pd.qcut vs Thresholds")
        print("-" * 80)
        comparar_distribuicoes(decis_qcut, decis_threshold, verbose=True)

    return {
        'periodo': periodo_nome,
        'n_leads': n_leads,
        'cv': cv,
        'top3_pct': top3_pct,
        'max_pct': max_pct,
        'decis_vazios': len(decis_vazios)
    }


def main():
    print("="*80)
    print("VALIDAÇÃO DE THRESHOLDS FIXOS - DADOS REAIS DE PRODUÇÃO")
    print("="*80)

    # Paths
    csv_path = "files/[LF] Pesquisa - Mai25 - [LF] Pesquisa-2.csv"
    model_path = "files/20251111_212345"

    # Carregar thresholds
    print("\n📁 Carregando thresholds do modelo...")
    thresholds = carregar_thresholds()
    print(f"✓ {len(thresholds)} decis carregados")

    # Processar leads
    print("\n🔄 Processando leads com pipeline de produção...")
    df = processar_leads(csv_path, model_path)

    # Coluna Data já está em datetime
    hoje = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Análises por período
    resultados = []

    # Todos os dados
    print("\n" + "="*80)
    print("TODOS OS DADOS (25,730 leads)")
    print("="*80)
    resultado = comparar_metodos(df, thresholds, "Todos os dados")
    resultados.append(resultado)

    # Últimos 7 dias
    df_7d = df[df['Data'] >= (hoje - timedelta(days=7))]
    if len(df_7d) > 0:
        resultado = comparar_metodos(df_7d, thresholds, "Últimos 7 dias")
        resultados.append(resultado)
    else:
        print("\n⚠️  Sem leads dos últimos 7 dias")

    # Ontem
    df_ontem = df[(df['Data'] >= (hoje - timedelta(days=1))) & (df['Data'] < hoje)]
    if len(df_ontem) > 0:
        resultado = comparar_metodos(df_ontem, thresholds, "Ontem")
        resultados.append(resultado)
    else:
        print("\n⚠️  Sem leads de ontem")

    # Hoje
    df_hoje = df[df['Data'] >= hoje]
    if len(df_hoje) > 0:
        resultado = comparar_metodos(df_hoje, thresholds, "Hoje")
        resultados.append(resultado)
    else:
        print("\n⚠️  Sem leads de hoje")

    # Resumo final
    print("\n" + "="*80)
    print("RESUMO FINAL - VIABILIDADE DOS THRESHOLDS FIXOS")
    print("="*80)

    print(f"\n{'Período':<20} {'Leads':>10} {'CV':>8} {'Top3%':>8} {'Max%':>8} {'Vazios':>8}")
    print("-"*80)
    for r in resultados:
        print(f"{r['periodo']:<20} {r['n_leads']:>10} {r['cv']:>8.3f} {r['top3_pct']:>7.1f}% {r['max_pct']:>7.1f}% {r['decis_vazios']:>8}")

    # Decisão
    print("\n" + "="*80)
    print("DECISÃO:")
    print("="*80)

    # Critérios
    max_cv = max(r['cv'] for r in resultados)
    max_top3 = max(r['top3_pct'] for r in resultados)
    max_max = max(r['max_pct'] for r in resultados)
    max_vazios = max(r['decis_vazios'] for r in resultados)

    viavel = (
        max_cv <= 0.5 and
        max_top3 <= 60 and
        max_max <= 50 and
        max_vazios <= 2
    )

    if viavel:
        print("✅ THRESHOLDS FIXOS SÃO VIÁVEIS!")
        print("\nPróximos passos:")
        print("  1. Implementar na API")
        print("  2. Habilitar batching de alta frequência (2-3h)")
        print("  3. Monitorar em produção")
    else:
        print("❌ THRESHOLDS FIXOS NÃO SÃO VIÁVEIS")
        print("\nProblemas identificados:")
        if max_cv > 0.5:
            print(f"  - CV muito alto: {max_cv:.3f} (limite: 0.5)")
        if max_top3 > 60:
            print(f"  - Top 3 concentra muito: {max_top3:.1f}% (limite: 60%)")
        if max_max > 50:
            print(f"  - Um decil concentra muito: {max_max:.1f}% (limite: 50%)")
        if max_vazios > 2:
            print(f"  - Muitos decis vazios: {max_vazios} (limite: 2)")

    print("="*80)


if __name__ == "__main__":
    main()
