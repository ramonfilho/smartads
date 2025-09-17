"""
Teste do componente de limpeza de colunas.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data.preprocessing import clean_columns, get_columns_to_remove


def test_clean_columns():
    """Testa limpeza de colunas com dados reais."""

    # Carregar dados de teste
    filepath = '../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx'
    df = pd.read_excel(filepath, sheet_name='LF Pesquisa')

    print("=== TESTE DE LIMPEZA DE COLUNAS ===\n")
    print(f"DataFrame original: {len(df)} linhas, {len(df.columns)} colunas")

    # Verificar colunas a remover presentes
    columns_to_remove = get_columns_to_remove()
    columns_present = [col for col in columns_to_remove if col in df.columns]
    print(f"\nColunas marcadas para remoção presentes no DataFrame:")
    for col in columns_present:
        print(f"  - {col}")

    # Aplicar limpeza
    df_clean = clean_columns(df)

    # Verificar resultados
    print(f"\nDataFrame após limpeza: {len(df_clean)} linhas, {len(df_clean.columns)} colunas")
    print(f"Colunas removidas: {len(df.columns) - len(df_clean.columns)}")

    # Verificar que colunas foram removidas
    for col in columns_present:
        assert col not in df_clean.columns, f"Coluna '{col}' não foi removida!"

    # Verificar que não há colunas Unnamed
    unnamed_cols = [col for col in df_clean.columns if col.startswith('Unnamed:')]
    assert len(unnamed_cols) == 0, f"Ainda existem colunas Unnamed: {unnamed_cols}"

    # Verificar que a coluna importante de faixa salarial NÃO foi removida
    assert 'Atualmente, qual a sua faixa salarial?' in df_clean.columns, \
        "Coluna de faixa salarial foi removida incorretamente!"

    print("\n✅ Teste de limpeza de colunas passou com sucesso!")
    print(f"   - {len(columns_present)} colunas de score/faixa removidas")
    print(f"   - Nenhuma coluna Unnamed presente")
    print(f"   - Coluna de faixa salarial preservada corretamente")

    return df_clean


if __name__ == "__main__":
    df_result = test_clean_columns()
    print(f"\n📊 DataFrame final disponível com {len(df_result.columns)} colunas")