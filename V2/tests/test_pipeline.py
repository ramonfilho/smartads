"""
Teste de integração do pipeline completo.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import LeadScoringPipeline
import pandas as pd


def test_pipeline():
    """Testa o pipeline completo com dados reais."""

    print("=== TESTE DO PIPELINE COMPLETO ===\n")

    # Inicializar pipeline
    pipeline = LeadScoringPipeline()

    # Arquivo de teste conforme especificado no PROJECT_GUIDE.md
    # Usar o arquivo correto: Lead score LF 24.xlsx (com espaços)
    test_file = '../../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx'

    # Verificar se arquivo existe
    if not os.path.exists(test_file):
        print(f"❌ Arquivo não encontrado: {test_file}")
        print("Certifique-se de que o arquivo de teste está no local correto.")
        return None

    # Carregar dados originais da aba correta para comparação
    original_df = pd.read_excel(test_file, sheet_name='LF Pesquisa')
    print(f"Dados originais (aba 'LF Pesquisa'): {len(original_df)} linhas, {len(original_df.columns)} colunas")

    # Executar pipeline - ele deve usar a aba 'LF Pesquisa' por padrão
    result = pipeline.run(test_file)

    # Verificar resultados básicos
    print(f"\n📊 Resultados do Pipeline:")
    print(f"   - Linhas originais: {len(original_df)}")
    print(f"   - Linhas processadas: {len(result)}")
    print(f"   - Duplicatas removidas: {len(original_df) - len(result)}")
    print(f"   - Colunas originais: {len(original_df.columns)}")
    print(f"   - Colunas finais: {len(result.columns)}")
    print(f"   - Variação de colunas: {len(result.columns) - len(original_df.columns):+d}")

    # Identificar colunas que foram removidas e adicionadas (cálculo real)
    original_cols = set(original_df.columns)
    final_cols = set(result.columns)

    removed_cols = original_cols - final_cols
    added_cols = final_cols - original_cols

    print(f"\n🔍 Análise Real das Colunas:")
    print(f"   - Colunas removidas: {len(removed_cols)}")
    if removed_cols:
        print(f"     • {', '.join(sorted(removed_cols))}")

    print(f"   - Colunas adicionadas: {len(added_cols)}")
    if added_cols:
        print(f"     • {', '.join(sorted(added_cols))}")

    print(f"   - Resultado líquido: {len(added_cols) - len(removed_cols):+d} colunas")

    # Verificar que duplicatas foram processadas
    assert len(result) <= len(original_df), "Pipeline não deveria adicionar linhas"
    print(f"   - Número de linhas consistente (não foram adicionadas linhas)")

    # Verificar que encoding expandiu as features categóricas
    assert len(result.columns) > len(original_df.columns), "One-hot encoding deveria adicionar colunas"
    print(f"   - Colunas expandidas pelo encoding conforme esperado (+{len(result.columns) - len(original_df.columns)})")

    # Verificar que a coluna de faixa salarial foi preservada (é uma feature, não resultado)
    if 'Atualmente, qual a sua faixa salarial?' in original_df.columns:
        assert 'Atualmente, qual a sua faixa salarial?' in result.columns, \
            "❌ Coluna de faixa salarial foi removida incorretamente!"
        print(f"   - Coluna de faixa salarial preservada (é uma feature)")

    print(f"\n✅ Pipeline executado com sucesso!")
    return result


if __name__ == "__main__":
    df_result = test_pipeline()
    if df_result is not None:
        print(f"\n📊 Pipeline testado com sucesso!")
        print(f"DataFrame final: {len(df_result)} registros, {len(df_result.columns)} colunas")
    else:
        print("\n❌ Teste falhou - verifique o caminho do arquivo")
        sys.exit(1)