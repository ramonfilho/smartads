"""
Script para analisar categorias duplicadas nas colunas categóricas do arquivo LF24.
Verifica se ainda há necessidade de unificação de categorias em produção.
"""

import pandas as pd
import numpy as np
from collections import Counter


def analyze_categorical_columns():
    """Analisa categorias duplicadas nas colunas categóricas."""

    # Carregar dados de produção
    filepath = '../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx'
    df = pd.read_excel(filepath, sheet_name='LF Pesquisa')

    print("=== ANÁLISE DE CATEGORIAS DUPLICADAS - ARQUIVO LF24 ===\n")
    print(f"Total de registros: {len(df)}")

    # Colunas categóricas para analisar
    categorical_columns = [
        'O seu gênero:',
        'Qual estado você mora?',
        'Qual a sua idade?',
        'O que você faz atualmente?',
        'Atualmente, qual a sua faixa salarial?',
        'Você possui cartão de crédito?',
        'Já estudou programação?',
        'Você já fez/faz/pretende fazer faculdade?',
        'Já investiu em algum curso online para aprender uma nova forma de ganhar dinheiro?',
        'O que mais te chama atenção na profissão de Programador?',
        'O que mais você quer ver no evento?'
    ]

    # Armazenar resultados da análise
    duplicates_found = {}

    for col in categorical_columns:
        if col in df.columns:
            print(f"\n📊 Analisando: {col}")
            print("-" * 50)

            # Obter valores únicos (excluindo NaN)
            values = df[col].dropna().astype(str).str.strip()
            unique_values = values.unique()
            value_counts = values.value_counts()

            print(f"Total de categorias únicas: {len(unique_values)}")

            # Detectar possíveis duplicatas (case-insensitive, espaços extras, etc.)
            potential_duplicates = []

            # Agrupar por versão normalizada (lowercase, sem espaços extras)
            normalized_groups = {}
            for val in unique_values:
                normalized = val.lower().strip()
                normalized = ' '.join(normalized.split())  # Remove espaços múltiplos

                if normalized not in normalized_groups:
                    normalized_groups[normalized] = []
                normalized_groups[normalized].append(val)

            # Identificar grupos com mais de uma variação
            for norm_val, variations in normalized_groups.items():
                if len(variations) > 1:
                    potential_duplicates.append(variations)
                    print(f"\n  ⚠️ Possíveis duplicatas encontradas:")
                    for var in variations:
                        count = value_counts[var]
                        print(f"    - '{var}' ({count} ocorrências)")

            # Detectar variações similares (ex: "Sim" vs "sim", "SIM")
            if not potential_duplicates:
                # Verificar variações com pequenas diferenças
                for i, val1 in enumerate(unique_values):
                    for val2 in unique_values[i+1:]:
                        # Comparar versões normalizadas
                        if val1.lower().strip() == val2.lower().strip() and val1 != val2:
                            print(f"\n  ⚠️ Variações de caso encontradas:")
                            print(f"    - '{val1}' ({value_counts[val1]} ocorrências)")
                            print(f"    - '{val2}' ({value_counts[val2]} ocorrências)")
                            potential_duplicates.append([val1, val2])

            if potential_duplicates:
                duplicates_found[col] = potential_duplicates
            else:
                print("  ✅ Nenhuma duplicata óbvia encontrada")

            # Mostrar top 5 categorias mais frequentes
            print(f"\n  Top 5 categorias mais frequentes:")
            for val, count in value_counts.head(5).items():
                pct = (count / len(values)) * 100
                print(f"    - '{val}': {count} ({pct:.1f}%)")

        else:
            print(f"\n❌ Coluna não encontrada: {col}")

    # Resumo final
    print("\n" + "=" * 60)
    print("RESUMO DA ANÁLISE")
    print("=" * 60)

    if duplicates_found:
        print(f"\n⚠️ Colunas com possíveis duplicatas: {len(duplicates_found)}")
        for col, dups in duplicates_found.items():
            print(f"\n  {col}:")
            print(f"    - {len(dups)} grupos de duplicatas encontrados")
    else:
        print("\n✅ Nenhuma duplicata significativa encontrada nas colunas categóricas!")
        print("   A Seção 7 de unificação de categorias pode não ser necessária.")

    return duplicates_found


def check_specific_unifications():
    """Verifica unificações específicas mencionadas no notebook original."""

    filepath = '../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx'
    df = pd.read_excel(filepath, sheet_name='LF Pesquisa')

    print("\n\n=== VERIFICAÇÃO DE UNIFICAÇÕES ESPECÍFICAS DO NOTEBOOK ===\n")

    # Verificar algumas unificações específicas mencionadas no notebook
    checks = {
        'O que você faz atualmente?': {
            'patterns': [
                ('CLT', 'clt', 'Clt'),
                ('Freelancer', 'freelancer', 'FREELANCER'),
                ('Desempregado', 'desempregado', 'DESEMPREGADO'),
                ('Autônomo', 'autônomo', 'AUTÔNOMO')
            ]
        },
        'Qual estado você mora?': {
            'patterns': [
                ('SP', 'sp', 'São Paulo', 'SÃO PAULO'),
                ('RJ', 'rj', 'Rio de Janeiro', 'RIO DE JANEIRO'),
                ('MG', 'mg', 'Minas Gerais', 'MINAS GERAIS')
            ]
        }
    }

    for col, check_data in checks.items():
        if col in df.columns:
            print(f"\n📌 {col}")
            values = df[col].dropna().astype(str)

            for pattern_group in check_data['patterns']:
                found = []
                for pattern in pattern_group:
                    count = (values == pattern).sum()
                    if count > 0:
                        found.append(f"'{pattern}' ({count})")

                if len(found) > 1:
                    print(f"  ⚠️ Variações encontradas: {', '.join(found)}")
                elif len(found) == 1:
                    print(f"  ✓ Apenas uma forma encontrada: {found[0]}")


if __name__ == "__main__":
    # Executar análise
    duplicates = analyze_categorical_columns()
    check_specific_unifications()

    print("\n\n" + "=" * 60)
    print("CONCLUSÃO")
    print("=" * 60)

    if duplicates:
        print("\n⚠️ RECOMENDAÇÃO: Implementar componente de unificação de categorias")
        print("   Foram encontradas variações que podem afetar o modelo.")
    else:
        print("\n✅ RECOMENDAÇÃO: Pular Seção 7 de unificação de categorias")
        print("   Os dados de produção (LF24) parecem estar bem padronizados.")