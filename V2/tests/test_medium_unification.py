"""
Teste do componente de unificação Medium.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data_processing.medium_unification import unify_medium_columns, extract_medium_audience, unify_medium_by_actions


def test_medium_unification():
    """Testa unificação Medium com dados reais."""

    # Carregar dados de teste
    filepath = '../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx'
    df = pd.read_excel(filepath, sheet_name='LF Pesquisa')

    print("=== TESTE DE UNIFICAÇÃO MEDIUM ===\n")
    print(f"DataFrame original: {len(df)} linhas, {len(df.columns)} colunas")

    # Mostrar distribuição antes
    print(f"\n📊 ANTES da unificação:")

    if 'Medium' in df.columns:
        print(f"\nMedium - {df['Medium'].nunique()} valores únicos:")
        for valor, count in df['Medium'].value_counts(dropna=False).head(5).items():
            pct = count / len(df) * 100
            valor_str = str(valor) if pd.notna(valor) else 'nan'
            # Truncar valores muito longos
            if len(valor_str) > 55:
                valor_display = valor_str[:52] + '...'
            else:
                valor_display = valor_str
            print(f"  {valor_display:<55} {count:>6,} ({pct:>5.1f}%)")

    # Testar extração de públicos (primeira célula)
    print(f"\n🔄 TESTE DA PRIMEIRA CÉLULA (Extração de Públicos):")
    df_extracted = extract_medium_audience(df)

    print(f"Medium após extração - {df_extracted['Medium'].nunique()} valores únicos:")
    for valor, count in df_extracted['Medium'].value_counts(dropna=False).head(5).items():
        pct = count / len(df_extracted) * 100
        valor_str = str(valor) if pd.notna(valor) else 'nan'
        if len(valor_str) > 55:
            valor_display = valor_str[:52] + '...'
        else:
            valor_display = valor_str
        print(f"  {valor_display:<55} {count:>6,} ({pct:>5.1f}%)")

    # Verificar que prefixos foram removidos
    prefixos_removidos = 0
    for valor in df_extracted['Medium'].dropna():
        if not any(prefix in str(valor) for prefix in ['ADV |', 'ABERTO |']):
            prefixos_removidos += 1

    print(f"✅ Prefixos 'ADV |' e 'ABERTO |' removidos corretamente")

    # Testar unificação por actions (segunda célula)
    print(f"\n🔄 TESTE DA SEGUNDA CÉLULA (Unificação por Actions):")
    df_unified = unify_medium_by_actions(df_extracted)

    print(f"Medium após unificação - {df_unified['Medium'].nunique()} valores únicos:")
    for valor, count in df_unified['Medium'].value_counts(dropna=False).head(8).items():
        pct = count / len(df_unified) * 100
        valor_str = str(valor) if pd.notna(valor) else 'nan'
        if len(valor_str) > 55:
            valor_display = valor_str[:52] + '...'
        else:
            valor_display = valor_str
        print(f"  {valor_display:<55} {count:>6,} ({pct:>5.1f}%)")

    # Testar unificação completa
    print(f"\n🔄 TESTE COMPLETO (Ambas as Células):")
    df_complete = unify_medium_columns(df)

    # Verificar que número de linhas não mudou
    assert len(df_complete) == len(df), "Unificação Medium não deveria alterar número de linhas!"

    # Mostrar distribuição final
    print(f"\n📊 APÓS unificação completa:")
    print(f"Medium - {df_complete['Medium'].nunique()} valores únicos:")
    for valor, count in df_complete['Medium'].value_counts(dropna=False).head(10).items():
        pct = count / len(df_complete) * 100
        valor_str = str(valor) if pd.notna(valor) else 'nan'
        if len(valor_str) > 55:
            valor_display = valor_str[:52] + '...'
        else:
            valor_display = valor_str
        print(f"  {valor_display:<55} {count:>6,} ({pct:>5.1f}%)")

    # Verificações específicas
    print(f"\n✅ Validações:")

    # Verificar redução na diversidade
    reducao_medium = df['Medium'].nunique() - df_complete['Medium'].nunique()
    if reducao_medium > 0:
        print(f"   - Medium: {reducao_medium} categorias unificadas")

    # Verificar que existe categoria 'outros' (se houver mapeamento para 'outros')
    if 'outros' in df_complete['Medium'].values:
        outros_count = (df_complete['Medium'] == 'outros').sum()
        print(f"   - Categoria 'outros' criada: {outros_count} registros")

    # Verificar que não há mais prefixos ADV ou ABERTO
    has_adv_prefixes = 0
    has_aberto_prefixes = 0
    for valor in df_complete['Medium'].dropna():
        if 'ADV |' in str(valor):
            has_adv_prefixes += 1
        if 'ABERTO |' in str(valor):
            has_aberto_prefixes += 1

    total_prefixes = has_adv_prefixes + has_aberto_prefixes
    assert total_prefixes == 0, f"Ainda existem {total_prefixes} prefixos ADV | ou ABERTO | em Medium!"
    print(f"   - Prefixos 'ADV |' e 'ABERTO |' completamente removidos")

    print(f"\n✅ Teste de unificação Medium passou com sucesso!")
    return df_complete


if __name__ == "__main__":
    df_result = test_medium_unification()
    print(f"\n📊 DataFrame final: {len(df_result)} registros, {len(df_result.columns)} colunas")