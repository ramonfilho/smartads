"""
Teste do componente de engenharia de features.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.features.engineering import (
    create_derived_features,
    normalizar_telefone_robusto,
    validar_email_robusto,
    validar_nome_robusto,
    get_feature_engineering_summary,
    get_created_features_list,
    get_removed_columns_list
)


def test_feature_engineering():
    """Testa engenharia de features com dados reais."""

    # Carregar dados de teste
    filepath = '../../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx'
    df = pd.read_excel(filepath, sheet_name='LF Pesquisa')

    print("=== TESTE DE ENGENHARIA DE FEATURES ===\n")
    print(f"DataFrame original: {len(df)} linhas, {len(df.columns)} colunas")

    # Mostrar colunas relevantes antes
    print(f"\n📊 ANTES da engenharia de features:")

    # Verificar colunas que serão utilizadas
    colunas_relevantes = ['Data', 'Nome Completo', 'E-mail', 'Telefone']
    for col in colunas_relevantes:
        if col in df.columns:
            print(f"  {col}: {df[col].notna().sum()}/{len(df)} não nulos ({df[col].notna().mean()*100:.1f}%)")
        else:
            print(f"  {col}: COLUNA NÃO ENCONTRADA")

    # Testar validação de telefone
    print(f"\n🔄 TESTE DA VALIDAÇÃO DE TELEFONE:")
    test_phones = ['11999887766', '11 99988-7766', '5511999887766', '99988-7766', '123', None, 1.2345e+10]
    for phone in test_phones:
        resultado = normalizar_telefone_robusto(phone)
        print(f"  {str(phone):<15} -> {resultado}")

    # Testar validação de email
    print(f"\n🔄 TESTE DA VALIDAÇÃO DE EMAIL:")
    test_emails = ['test@gmail.com', 'invalid@', '@invalid.com', 'test', 'a@b.co', None]
    for email in test_emails:
        resultado = validar_email_robusto(email)
        print(f"  {str(email):<20} -> {resultado}")

    # Testar validação de nome
    print(f"\n🔄 TESTE DA VALIDAÇÃO DE NOME:")
    test_names = ['João Silva', 'Ana', '123456', 'João123', '', None, 'A B']
    for name in test_names:
        resultado = validar_nome_robusto(name)
        print(f"  {str(name):<15} -> {resultado}")

    # Aplicar engenharia de features
    print(f"\n🔄 APLICANDO ENGENHARIA DE FEATURES:")
    df_original = df.copy()
    df_fe = create_derived_features(df)

    # Verificar que número de linhas não mudou
    assert len(df_fe) == len(df), "Engenharia de features não deveria alterar número de linhas!"

    # Mostrar features criadas
    print(f"\nFeatures criadas:")
    features_esperadas = get_created_features_list()
    for feature in features_esperadas:
        if feature in df_fe.columns:
            print(f"  ✅ {feature}: criada")
        else:
            print(f"  ❌ {feature}: NÃO criada")

    # Mostrar colunas removidas
    print(f"\nColunas removidas:")
    colunas_removidas = get_removed_columns_list()
    for col in colunas_removidas:
        if col not in df_fe.columns and col in df_original.columns:
            print(f"  ✅ {col}: removida")
        elif col in df_fe.columns and col in df_original.columns:
            print(f"  ❌ {col}: NÃO removida")
        else:
            print(f"  ⚠️  {col}: não existia no dataset original")

    # Estatísticas das features criadas
    print(f"\n📊 APÓS engenharia de features:")
    print(f"DataFrame: {len(df_fe)} linhas, {len(df_fe.columns)} colunas")

    # Mostrar distribuição das features temporais
    if 'dia_semana' in df_fe.columns:
        print(f"\nDistribuição por dia da semana:")
        dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        for dia_num, count in df_fe['dia_semana'].value_counts().sort_index().items():
            if pd.notna(dia_num):
                dia_nome = dias[int(dia_num)] if int(dia_num) < 7 else f"Dia {int(dia_num)}"
                pct = count / len(df_fe) * 100
                print(f"  {dia_nome:<8}: {count:>4} ({pct:>5.1f}%)")

    # Mostrar estatísticas de qualidade
    qualidade_features = ['nome_valido', 'email_valido', 'telefone_valido']
    print(f"\nEstatísticas de qualidade:")
    for feature in qualidade_features:
        if feature in df_fe.columns:
            validos = df_fe[feature].sum()
            total = len(df_fe)
            pct = validos / total * 100
            print(f"  {feature:<15}: {validos:>4}/{total} ({pct:>5.1f}%)")

    # Gerar resumo
    summary = get_feature_engineering_summary(df_original, df_fe)
    print(f"\n📋 RESUMO:")
    print(f"  Colunas originais: {summary['original_columns']}")
    print(f"  Colunas após FE:    {summary['fe_columns']}")
    print(f"  Colunas adicionadas: {summary['columns_added']}")

    # Verificações específicas
    print(f"\n✅ Validações:")

    # Verificar que features temporais foram criadas corretamente
    if 'dia_semana' in df_fe.columns:
        dias_unicos = df_fe['dia_semana'].dropna().nunique()
        assert dias_unicos <= 7, f"dia_semana deveria ter no máximo 7 valores únicos, mas tem {dias_unicos}!"
        print(f"   - Feature temporal 'dia_semana' criada corretamente ({dias_unicos} dias)")

    # Verificar que features de qualidade foram criadas
    features_qualidade_criadas = 0
    for feature in ['nome_valido', 'email_valido', 'telefone_valido']:
        if feature in df_fe.columns:
            features_qualidade_criadas += 1

    assert features_qualidade_criadas >= 1, "Pelo menos uma feature de qualidade deveria ter sido criada!"
    print(f"   - {features_qualidade_criadas} features de qualidade criadas")

    # Verificar que colunas originais foram removidas
    colunas_que_deveriam_ser_removidas = ['Nome Completo', 'E-mail', 'Telefone']
    colunas_ainda_presentes = [col for col in colunas_que_deveriam_ser_removidas if col in df_fe.columns]

    if not colunas_ainda_presentes:
        print(f"   - Colunas originais removidas corretamente")
    else:
        print(f"   ⚠️  Colunas ainda presentes: {colunas_ainda_presentes}")

    print(f"\n✅ Teste de engenharia de features passou com sucesso!")
    return df_fe


if __name__ == "__main__":
    df_result = test_feature_engineering()
    print(f"\n📊 DataFrame final: {len(df_result)} registros, {len(df_result.columns)} colunas")