"""
Teste do componente de remoção de features de campanha.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data_processing.preprocessing import remove_campaign_features, get_campaign_features_to_remove


def test_remove_campaign_features():
    """Testa remoção de features de campanha com dados reais."""

    # Carregar dados de teste
    filepath = '../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx'
    df = pd.read_excel(filepath, sheet_name='LF Pesquisa')

    print("=== TESTE DE REMOÇÃO DE FEATURES DE CAMPANHA ===\n")
    print(f"DataFrame original: {len(df)} linhas, {len(df.columns)} colunas")

    # Verificar features de campanha presentes
    campaign_features = get_campaign_features_to_remove()
    features_present = [feat for feat in campaign_features if feat in df.columns]

    print(f"\nFeatures de campanha presentes no DataFrame:")
    for feat in features_present:
        print(f"  - {feat}")

    if not features_present:
        print("  (Nenhuma feature de campanha encontrada)")

    # Aplicar remoção de features de campanha
    df_clean = remove_campaign_features(df)

    # Verificar resultados
    print(f"\nDataFrame após remoção: {len(df_clean)} linhas, {len(df_clean.columns)} colunas")
    print(f"Features removidas: {len(df.columns) - len(df_clean.columns)}")

    # Verificar que features de campanha foram removidas
    for feat in features_present:
        assert feat not in df_clean.columns, f"Feature '{feat}' não foi removida!"

    # Verificar que nenhuma linha foi perdida
    assert len(df_clean) == len(df), "Linhas foram perdidas durante remoção de features!"

    # Verificar samples das features removidas (se existiram)
    if features_present:
        print(f"\nSamples das features removidas:")
        for feat in features_present:
            unique_values = df[feat].nunique()
            sample_values = df[feat].value_counts().head(3)
            print(f"\n  {feat}:")
            print(f"    - Valores únicos: {unique_values}")
            print(f"    - Top 3 valores:")
            for val, count in sample_values.items():
                print(f"      '{val}': {count} ocorrências")

    print(f"\n✅ Teste de remoção de features de campanha passou com sucesso!")
    print(f"   - {len(features_present)} features de campanha removidas")
    print(f"   - Nenhuma linha perdida")
    print(f"   - DataFrame: {len(df_clean)} registros, {len(df_clean.columns)} colunas")

    return df_clean


if __name__ == "__main__":
    df_result = test_remove_campaign_features()
    print(f"\n📊 DataFrame final disponível com {len(df_result.columns)} colunas")