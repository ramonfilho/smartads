"""
Pipeline de treino - Reproduz notebook DevClub célula por célula.

Integra funções modularizadas conforme são aprovadas.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import yaml
import glob
import logging
from src.data_processing.ingestion import read_excel_files, filter_sheets, remove_duplicates_per_sheet

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Executa pipeline de treino completo."""

    print("\n" + "=" * 80)
    print("PIPELINE DE TREINO")
    print("=" * 80)

    # Carregar configuração
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/devclub.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # === CÉLULA 1: Upload/Leitura de arquivos ===
    print("\n📤 CÉLULA 1: LEITURA DE ARQUIVOS")
    data_dir = config['ingestion']['training_data_dir']
    filepaths = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))

    print(f"Total de arquivos: {len(filepaths)}")
    for f in filepaths:
        print(f"  - {os.path.basename(f)}")

    # Ler arquivos
    all_data = read_excel_files(filepaths)

    # === CÉLULA 2: Filtragem + Remoção de Duplicatas ===
    print("\n🔄 CÉLULA 2: FILTRAGEM DE ABAS + REMOÇÃO DE DUPLICATAS")
    print("=" * 60)

    # Filtrar abas
    filtered_data, filter_report = filter_sheets(
        all_data,
        termos_manter=config['ingestion']['termos_manter'],
        termos_remover=config['ingestion']['termos_remover'],
        min_linhas=config['ingestion']['min_linhas']
    )

    # Remover duplicatas
    clean_data, dup_stats = remove_duplicates_per_sheet(filtered_data)

    # === RELATÓRIO (linhas 96-127 do notebook) ===
    print(f"\n📊 ABAS MANTIDAS E PROCESSADAS")
    print("=" * 80)
    print(f"{'ARQUIVO':<35} {'ABA':<20} {'ORIGINAL':>10} {'FINAL':>10} {'REMOVIDAS':>10}")
    print("-" * 80)

    total_original = 0
    total_final = 0
    total_duplicatas = 0

    for item in filter_report:
        if item['status'] == 'MANTIDA':
            filename = item['arquivo']
            sheet_name = item['aba']
            linhas_original = item['linhas_original']

            # Pegar estatísticas de duplicatas
            duplicatas = dup_stats.get(filename, {}).get(sheet_name, 0)
            linhas_final = linhas_original - duplicatas

            print(f"{filename[:34]:<35} {sheet_name[:19]:<20} "
                  f"{linhas_original:>10,} {linhas_final:>10,} {duplicatas:>10,}")

            total_original += linhas_original
            total_final += linhas_final
            total_duplicatas += duplicatas

    print("-" * 80)
    print(f"{'TOTAL':<35} {'':<20} {total_original:>10,} {total_final:>10,} {total_duplicatas:>10,}")

    # Resumo final
    abas_mantidas = sum(1 for item in filter_report if item['status'] == 'MANTIDA')
    abas_removidas = len(filter_report) - abas_mantidas

    print(f"\n📈 RESUMO FINAL:")
    print(f"Arquivos processados: {len(clean_data)}")
    print(f"Abas mantidas: {abas_mantidas}")
    print(f"Abas removidas: {abas_removidas}")
    print(f"Linhas totais após processamento: {total_final:,}")
    print(f"Duplicatas removidas: {total_duplicatas:,}")
    if total_original > 0:
        print(f"Redução por duplicatas: {(total_duplicatas/total_original*100):.2f}%")

    print(f"\n✅ Dados processados disponíveis na variável 'arquivos_filtrados'")
    print("=" * 80)


if __name__ == "__main__":
    main()
