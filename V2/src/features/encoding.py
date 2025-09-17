"""
Módulo de encoding categórico para o pipeline de lead scoring.
Mantém a lógica EXATA do notebook original para garantir reprodutibilidade.
"""

import pandas as pd
from typing import Dict


def apply_categorical_encoding(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica encoding em um dataset específico.

    Função EXATA copiada da Seção 20 do notebook original.
    """
    df = df_original.copy()

    print(f"Colunas antes do encoding: {len(df.columns)}")

    # 1. ENCODING ORDINAL para variáveis com ordem natural
    variaveis_ordinais = {
        'Qual a sua idade?': ['Menos de 18 anos', '18 - 24 anos', '25 - 34 anos',
                              '35 - 44 anos', '45 - 54 anos', 'Mais de 55 anos'],
        'Atualmente, qual a sua faixa salarial?': ['Não tenho renda', 'Entre R$1.000 a R$2.000 reais ao mês',
                                                   'Entre R$2.001 a R$3.000 reais ao mês',
                                                   'Entre R$3.001 a R$5.000 reais ao mês',
                                                   'Mais de R$5.001 reais ao mês'],
        'dia_semana': [0, 1, 2, 3, 4, 5, 6]  # Já é numérico
    }

    print(f"\nAplicando ORDINAL ENCODING:")
    for var, ordem in variaveis_ordinais.items():
        if var in df.columns:
            if var == 'dia_semana':
                # Já é numérico, apenas reportar
                print(f"  {var}: mantido como numérico (0-6)")
            else:
                # Criar mapeamento ordinal
                mapeamento = {categoria: i for i, categoria in enumerate(ordem)}
                df[var] = df[var].map(mapeamento)
                print(f"  {var}: {len(ordem)} categorias → 0-{len(ordem)-1}")

    # 2. ONE-HOT ENCODING para variáveis categóricas nominais
    variaveis_one_hot = []

    # Identificar variáveis categóricas (excluindo ordinais já processadas e target)
    for col in df.columns:
        if col not in ['target'] and col not in variaveis_ordinais and col != 'nome_comprimento':
            # Verificar se é categórica (object ou poucos valores únicos)
            if df[col].dtype == 'object' or df[col].nunique() <= 20:
                variaveis_one_hot.append(col)

    print(f"\nAplicando ONE-HOT ENCODING para {len(variaveis_one_hot)} variáveis:")

    # Aplicar one-hot encoding
    df_encoded = pd.get_dummies(df, columns=variaveis_one_hot, prefix_sep='_', dtype=int)

    # Reportar criação de colunas
    colunas_criadas = len(df_encoded.columns) - len(df.columns)
    for var in variaveis_one_hot:
        categorias_unicas = df[var].nunique()
        print(f"  {var}: {categorias_unicas} categorias → {categorias_unicas} colunas binárias")

    print(f"\nResultado:")
    print(f"  Colunas one-hot originais: {len(variaveis_one_hot)}")
    print(f"  Colunas binárias criadas: {colunas_criadas}")
    print(f"  Total de colunas final: {len(df_encoded.columns)}")

    # Verificar tipos de dados finais
    tipos_dados = df_encoded.dtypes.value_counts()
    print(f"\nTipos de dados no dataset final:")
    for tipo, count in tipos_dados.items():
        print(f"  {tipo}: {count} colunas")

    return df_encoded


def get_encoding_summary(df_original: pd.DataFrame, df_encoded: pd.DataFrame) -> Dict:
    """
    Gera resumo do processo de encoding.

    Args:
        df_original: DataFrame original antes do encoding
        df_encoded: DataFrame após encoding

    Returns:
        Dicionário com estatísticas do encoding
    """
    summary = {
        'original_columns': len(df_original.columns),
        'encoded_columns': len(df_encoded.columns),
        'columns_added': len(df_encoded.columns) - len(df_original.columns),
        'rows': len(df_encoded)
    }

    # Tipos de dados
    original_types = df_original.dtypes.value_counts()
    encoded_types = df_encoded.dtypes.value_counts()

    summary['original_types'] = {str(tipo): int(count) for tipo, count in original_types.items()}
    summary['encoded_types'] = {str(tipo): int(count) for tipo, count in encoded_types.items()}

    return summary