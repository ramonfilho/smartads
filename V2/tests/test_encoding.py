"""
Teste do componente de encoding categórico.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.features.encoding import apply_categorical_encoding, get_encoding_summary


def test_categorical_encoding():
    """Testa encoding categórico com dados reais."""

    # Carregar dados de teste
    filepath = '../../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx'
    df = pd.read_excel(filepath, sheet_name='LF Pesquisa')

    print("=== TESTE DE ENCODING CATEGÓRICO ===\n")
    print(f"DataFrame original: {len(df)} linhas, {len(df.columns)} colunas")

    # Simular dados após feature engineering (precisa ter dia_semana)
    df_fe = df.copy()
    if 'Data' in df_fe.columns:
        df_fe['dia_semana'] = pd.to_datetime(df_fe['Data']).dt.dayofweek

    # Mostrar distribuição antes do encoding
    print(f"\n📊 ANTES do encoding:")

    # Verificar colunas ordinais
    colunas_ordinais = ['Qual a sua idade?', 'Atualmente, qual a sua faixa salarial?', 'dia_semana']
    for col in colunas_ordinais:
        if col in df_fe.columns:
            unique_vals = df_fe[col].nunique()
            print(f"  {col}: {unique_vals} valores únicos")

    # Verificar algumas colunas categóricas
    categorical_sample = ['Source', 'Term', 'Medium']
    for col in categorical_sample:
        if col in df_fe.columns:
            unique_vals = df_fe[col].nunique()
            dtype = df_fe[col].dtype
            print(f"  {col} ({dtype}): {unique_vals} valores únicos")

    # Aplicar encoding categórico
    print(f"\n🔄 APLICANDO ENCODING CATEGÓRICO:")
    df_original = df_fe.copy()
    df_encoded = apply_categorical_encoding(df_fe)

    # Verificar que número de linhas não mudou
    assert len(df_encoded) == len(df_original), "Encoding não deveria alterar número de linhas!"

    # Mostrar resultado final
    print(f"\n📊 APÓS encoding categórico:")
    print(f"DataFrame: {len(df_encoded)} linhas, {len(df_encoded.columns)} colunas")

    # Verificar encoding ordinal
    print(f"\nVerificação do encoding ordinal:")
    for col in ['Qual a sua idade?', 'Atualmente, qual a sua faixa salarial?']:
        if col in df_encoded.columns:
            unique_vals = sorted(df_encoded[col].dropna().unique())
            print(f"  {col}: {unique_vals} (deve ser números sequenciais)")

    # Verificar one-hot encoding
    print(f"\nVerificação do one-hot encoding:")

    # Contar colunas binárias (0 e 1 apenas)
    binary_cols = 0
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'int64' or df_encoded[col].dtype == 'int32':
            unique_vals = set(df_encoded[col].dropna().unique())
            if unique_vals.issubset({0, 1}):
                binary_cols += 1

    print(f"  Colunas binárias (0/1): {binary_cols}")

    # Verificar tipos de dados
    tipos_dados = df_encoded.dtypes.value_counts()
    print(f"\nTipos de dados finais:")
    for tipo, count in tipos_dados.items():
        print(f"  {tipo}: {count} colunas")

    # Gerar resumo
    summary = get_encoding_summary(df_original, df_encoded)
    print(f"\n📋 RESUMO:")
    print(f"  Colunas originais: {summary['original_columns']}")
    print(f"  Colunas após encoding: {summary['encoded_columns']}")
    print(f"  Colunas adicionadas: {summary['columns_added']}")

    # Verificações específicas
    print(f"\n✅ Validações:")

    # Verificar que colunas aumentaram (one-hot deve criar muitas)
    assert len(df_encoded.columns) > len(df_original.columns), "One-hot encoding deveria adicionar colunas!"
    cols_added = len(df_encoded.columns) - len(df_original.columns)
    print(f"   - {cols_added} colunas adicionadas pelo one-hot encoding")

    # Verificar que não há mais colunas object (exceto se não foram processadas)
    object_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"   ⚠️  Colunas object restantes: {object_cols}")
    else:
        print(f"   - Todas colunas categóricas foram encodadas")

    # Verificar que encoding ordinal funcionou
    if 'Qual a sua idade?' in df_encoded.columns:
        idade_values = set(df_encoded['Qual a sua idade?'].dropna().unique())
        expected_values = set(range(6))  # 0 a 5
        if idade_values.issubset(expected_values):
            print(f"   - Encoding ordinal de idade funcionou corretamente")

    print(f"\n✅ Teste de encoding categórico passou com sucesso!")
    return df_encoded


if __name__ == "__main__":
    df_result = test_categorical_encoding()
    print(f"\n📊 DataFrame final: {len(df_result)} registros, {len(df_result.columns)} colunas")