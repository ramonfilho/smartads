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
    # Print do cabeçalho para comparação com notebook
    print("ENCODING ESTRATÉGICO DOS 4 DATASETS")
    print("=" * 45)

    df = df_original.copy()

    print(f"\nProcessando dataset...")
    print(f"Colunas antes do encoding: {len(df.columns)}")

    # DEBUG: Listar colunas exatas que chegam no encoding
    print(f"\nColunas que chegam no encoding (total: {len(df.columns)}):")
    for i, col in enumerate(sorted(df.columns), 1):
        print(f"{i:2d}. {col}")
    print()

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
    print(f"ORDEM das variáveis one-hot:")
    for i, var in enumerate(variaveis_one_hot, 1):
        print(f"  {i:2d}. {var}")

    # Aplicar one-hot encoding
    df_encoded = pd.get_dummies(df, columns=variaveis_one_hot, prefix_sep='_', dtype=int)

    # REMOVER telefone_comprimento_8 (EXATO do notebook - linha 5076-5078)
    if 'telefone_comprimento_8' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['telefone_comprimento_8'])
        print(f"  ⚠️  telefone_comprimento_8 removida (conforme notebook)")

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

    # REORDENAR COLUNAS PARA ORDEM ESPERADA PELOS MODELOS (HARDCODED)
    # Ordem exata baseada nos arquivos features_ordenadas_*.json dos modelos treinados
    print(f"\nReordenando colunas para compatibilidade com modelos...")

    ordem_esperada = [
        "Qual_a_sua_idade",
        "Atualmente_qual_a_sua_faixa_salarial",
        "dia_semana",
        "nome_comprimento",
        "O_seu_g_nero_Feminino",
        "O_seu_g_nero_Masculino",
        "O_que_voc_faz_atualmente_N_o_trabalho_e_nem_estudo",
        "O_que_voc_faz_atualmente_Sou_CLT_Funcion_rio_P_blico",
        "O_que_voc_faz_atualmente_Sou_apenas_estudante",
        "O_que_voc_faz_atualmente_Sou_aposentado",
        "O_que_voc_faz_atualmente_Sou_aut_nomo",
        "Source_facebook_ads",
        "Source_google_ads",
        "Source_outros",
        "Medium_Aberto",
        "Medium_Interesse_Programa_o",
        "Medium_Linguagem_de_programa_o",
        "Medium_Lookalike_1_Cadastrados_DEV_2_0_Interesse_Ci_ncia_da_Computa_o",
        "Medium_Lookalike_2_Alunos_Interesse_Linguagem_de_Programa_o",
        "Medium_Lookalike_2_Cadastrados_DEV_2_0_Interesses",
        "Medium_Outros",
        "Medium_dgen",
        "Term_facebook",
        "Term_instagram",
        "Term_outros",
        "Voc_possui_cart_o_de_cr_dito_N_o",
        "Voc_possui_cart_o_de_cr_dito_Sim",
        "O_que_mais_voc_quer_ver_no_evento_A_aula_com_a_recrutadora",
        "O_que_mais_voc_quer_ver_no_evento_Fazer_freelancer_como_programador",
        "O_que_mais_voc_quer_ver_no_evento_Fazer_transi_o_de_carreira_e_conseguir_meu_primeiro_emprego_na_rea",
        "O_que_mais_voc_quer_ver_no_evento_Fazer_um_projeto_na_pr_tica",
        "O_que_mais_voc_quer_ver_no_evento_Quero_saber_se_para_mim",
        "Tem_computador_notebook_N_o",
        "Tem_computador_notebook_Sim",
        "J_estudou_programa_o_N_o",
        "J_estudou_programa_o_Sim",
        "Voc_j_fez_faz_pretende_fazer_faculdade_N_o",
        "Voc_j_fez_faz_pretende_fazer_faculdade_Sim",
        "investiu_curso_online_N_o",
        "investiu_curso_online_Sim",
        "interesse_programacao_A_ideia_de_nunca_faltar_emprego_na_rea",
        "interesse_programacao_A_possibilidade_de_ganhar_altos_sal_rios",
        "interesse_programacao_Poder_trabalhar_de_qualquer_lugar_do_mundo",
        "interesse_programacao_Todas_as_alternativas",
        "interesse_programacao_Trabalhar_para_outros_pa_ses_e_ganhar_em_outra_moeda",
        "nome_tem_sobrenome_False",
        "nome_tem_sobrenome_True",
        "nome_valido_False",
        "nome_valido_True",
        "email_valido_False",
        "email_valido_True",
        "telefone_valido_False",
        "telefone_valido_True",
        "telefone_comprimento_4",
        "telefone_comprimento_9",
        "telefone_comprimento_10",
        "telefone_comprimento_11"
    ]

    # Verificar se todas as colunas esperadas existem
    colunas_faltando = [col for col in ordem_esperada if col not in df_encoded.columns]
    colunas_extras = [col for col in df_encoded.columns if col not in ordem_esperada]

    if colunas_faltando:
        print(f"  ⚠️  ERRO: {len(colunas_faltando)} colunas esperadas estão faltando!")
        for col in colunas_faltando[:5]:
            print(f"    - {col}")

    if colunas_extras:
        print(f"  ⚠️  ERRO: {len(colunas_extras)} colunas extras encontradas!")
        for col in colunas_extras[:5]:
            print(f"    + {col}")

    if not colunas_faltando and not colunas_extras:
        # Reordenar DataFrame para ordem exata esperada pelos modelos
        df_encoded = df_encoded[ordem_esperada]
        print(f"  ✅ Colunas reordenadas para ordem dos modelos: {len(ordem_esperada)} features")
    else:
        print(f"  ❌ Mantendo ordem original devido a incompatibilidades")

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