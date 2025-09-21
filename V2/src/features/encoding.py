"""
Módulo de encoding categórico para o pipeline de lead scoring.
Mantém a lógica EXATA do notebook original para garantir reprodutibilidade.
"""

import pandas as pd
from typing import Dict


def apply_categorical_encoding(df_original: pd.DataFrame, versao: str = "v1") -> pd.DataFrame:
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


    # REMOVER DUPLICATAS DE COLUNAS (se houver) - CRÍTICO para evitar features extras
    colunas_antes_duplicatas = len(df_encoded.columns)
    df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]
    duplicatas_removidas = colunas_antes_duplicatas - len(df_encoded.columns)
    if duplicatas_removidas > 0:
        print(f"⚠️  Duplicatas removidas: {duplicatas_removidas} colunas")

    # Reportar criação de colunas
    colunas_criadas = len(df_encoded.columns) - len(df.columns)
    for var in variaveis_one_hot:
        categorias_unicas = df[var].nunique()
        print(f"  {var}: {categorias_unicas} categorias → {categorias_unicas} colunas binárias")

    print(f"\nResultado:")
    print(f"  Colunas one-hot originais: {len(variaveis_one_hot)}")
    print(f"  Colunas binárias criadas: {colunas_criadas}")

    # NORMALIZAÇÃO DOS NOMES DAS COLUNAS (linhas 4976-4978 do notebook)
    # CRÍTICO: Esta etapa estava faltando e causava incompatibilidade com o modelo
    print(f"\nNormalizando nomes das colunas...")

    # Guardar nomes originais para comparação
    colunas_antes = list(df_encoded.columns)

    # Aplicar normalização EXATA do notebook
    df_encoded.columns = df_encoded.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    df_encoded.columns = df_encoded.columns.str.replace('__+', '_', regex=True)
    df_encoded.columns = df_encoded.columns.str.strip('_')

    # MAPEAMENTOS ESPECÍFICOS para manter consistência com arquivo de features
    mapeamentos_especificos = {
        'O_que_voc_faz_atualmente_Sou_autonomo': 'O_que_voc_faz_atualmente_Sou_aut_nomo',
        'Tem_computador_notebook_SIM': 'Tem_computador_notebook_Sim',
        'Medium_outros': 'Medium_Outros'  # Corrigir capitalização
    }

    # Aplicar mapeamentos
    df_encoded.columns = [mapeamentos_especificos.get(col, col) for col in df_encoded.columns]

    # Contar quantas colunas foram alteradas
    colunas_alteradas = sum(1 for antes, depois in zip(colunas_antes, list(df_encoded.columns))
                            if antes != depois)
    print(f"  Colunas normalizadas: {colunas_alteradas}")

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