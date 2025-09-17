"""
Teste do componente de unificação UTM (Source e Term).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data.utm_unification import unify_utm_columns, get_utm_summary


def test_utm_unification():
    """Testa unificação UTM com dados reais."""

    # Carregar dados de teste
    filepath = '../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx'
    df = pd.read_excel(filepath, sheet_name='LF Pesquisa')

    print("=== TESTE DE UNIFICAÇÃO UTM ===\n")
    print(f"DataFrame original: {len(df)} linhas, {len(df.columns)} colunas")

    # Mostrar distribuição antes
    print(f"\n📊 ANTES da unificação:")

    if 'Source' in df.columns:
        print(f"\nSource - {df['Source'].nunique()} valores únicos:")
        for valor, count in df['Source'].value_counts(dropna=False).head(5).items():
            pct = count / len(df) * 100
            valor_str = str(valor) if pd.notna(valor) else 'nan'
            print(f"  {valor_str:<25} {count:>6,} ({pct:>5.1f}%)")

    if 'Term' in df.columns:
        print(f"\nTerm - {df['Term'].nunique()} valores únicos:")
        for valor, count in df['Term'].value_counts(dropna=False).head(5).items():
            pct = count / len(df) * 100
            valor_str = str(valor) if pd.notna(valor) else 'nan'
            print(f"  {valor_str:<35} {count:>6,} ({pct:>5.1f}%)")

    # Aplicar unificação
    df_unified = unify_utm_columns(df)

    # Verificar que número de linhas não mudou
    assert len(df_unified) == len(df), "Unificação UTM não deveria alterar número de linhas!"

    # Mostrar distribuição depois
    print(f"\n📊 APÓS a unificação:")

    if 'Source' in df_unified.columns:
        print(f"\nSource - {df_unified['Source'].nunique()} valores únicos:")
        for valor, count in df_unified['Source'].value_counts(dropna=False).items():
            pct = count / len(df_unified) * 100
            valor_str = str(valor) if pd.notna(valor) else 'nan'
            print(f"  {valor_str:<25} {count:>6,} ({pct:>5.1f}%)")

    if 'Term' in df_unified.columns:
        print(f"\nTerm - {df_unified['Term'].nunique()} valores únicos:")
        for valor, count in df_unified['Term'].value_counts(dropna=False).items():
            pct = count / len(df_unified) * 100
            valor_str = str(valor) if pd.notna(valor) else 'nan'
            print(f"  {valor_str:<25} {count:>6,} ({pct:>5.1f}%)")

    # Verificações específicas
    print(f"\n✅ Validações:")

    # Verificar unificações específicas
    if 'Source' in df_unified.columns:
        # Verificar que 'fb' foi convertido para 'outros'
        assert 'fb' not in df_unified['Source'].values, "Valor 'fb' em Source não foi unificado!"
        print(f"   - Source 'fb' convertido para 'outros'")

    if 'Term' in df_unified.columns:
        # Verificar que 'ig' foi convertido para 'instagram'
        assert 'ig' not in df_unified['Term'].values, "Valor 'ig' em Term não foi unificado!"
        print(f"   - Term 'ig' convertido para 'instagram'")

        # Verificar que 'fb' foi convertido para 'facebook'
        assert 'fb' not in df_unified['Term'].values, "Valor 'fb' em Term não foi unificado!"
        print(f"   - Term 'fb' convertido para 'facebook'")

        # Verificar que IDs com -- viraram 'outros'
        ids_com_tracos = df_unified['Term'].str.contains('--', na=False).sum()
        assert ids_com_tracos == 0, "IDs com -- ainda existem em Term!"
        print(f"   - IDs com '--' convertidos para 'outros'")

    # Verificar redução na diversidade
    if 'Source' in df.columns and 'Source' in df_unified.columns:
        reducao_source = df['Source'].nunique() - df_unified['Source'].nunique()
        if reducao_source > 0:
            print(f"   - Source: {reducao_source} categorias unificadas")

    if 'Term' in df.columns and 'Term' in df_unified.columns:
        reducao_term = df['Term'].nunique() - df_unified['Term'].nunique()
        if reducao_term > 0:
            print(f"   - Term: {reducao_term} categorias unificadas")

    print(f"\n✅ Teste de unificação UTM passou com sucesso!")
    return df_unified


if __name__ == "__main__":
    df_result = test_utm_unification()
    print(f"\n📊 DataFrame final: {len(df_result)} registros, {len(df_result.columns)} colunas")