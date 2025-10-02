"""
Teste unitário da função read_excel_files() do módulo ingestion.

Valida que a função executa sem erros e retorna a estrutura esperada.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import yaml
from src.data_processing.ingestion import read_excel_files

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test_read_excel_files():
    """Testa leitura de múltiplos arquivos Excel."""

    print("\n" + "=" * 80)
    print("TESTE UNITÁRIO: read_excel_files()")
    print("=" * 80)

    # Carregar config
    import glob
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'devclub.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['ingestion']['training_data_dir']
    test_files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))

    # Verificar se arquivos existem
    missing_files = [f for f in test_files if not os.path.exists(f)]
    if missing_files:
        print("\n⚠️  Arquivos não encontrados:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nAjuste os caminhos no teste conforme sua estrutura de dados.")
        return

    print(f"\n📂 Testando leitura de {len(test_files)} arquivo(s)...\n")

    # Executar função
    result = read_excel_files(test_files)

    # Validações básicas
    print("\n" + "=" * 80)
    print("RESULTADOS DO TESTE UNITÁRIO:")
    print("=" * 80)

    print(f"\n✅ Total de arquivos lidos: {len(result)}")

    for filename, sheets in result.items():
        print(f"\n📄 Arquivo: {filename}")
        print(f"   Número de abas: {len(sheets)}")

        for sheet_name, df in sheets.items():
            print(f"     • Aba '{sheet_name}': {len(df):,} linhas × {len(df.columns)} colunas")

    # Asserções
    assert isinstance(result, dict), "Resultado deve ser um dicionário"
    assert len(result) == len(test_files), f"Deve ter lido {len(test_files)} arquivos"

    for filename, sheets in result.items():
        assert isinstance(sheets, dict), f"Abas de {filename} devem ser um dicionário"
        assert len(sheets) > 0, f"{filename} deve ter pelo menos uma aba"

    print("\n" + "=" * 80)
    print("TESTE UNITÁRIO CONCLUÍDO")
    print("=" * 80)
    print("\n⚠️  AGUARDANDO VALIDAÇÃO DO USUÁRIO (única fonte da verdade)")


if __name__ == "__main__":
    test_read_excel_files()
