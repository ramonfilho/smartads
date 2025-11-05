"""
Pipeline de treino - Reproduz notebook DevClub célula por célula.

Integra funções modularizadas conforme são aprovadas.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import glob
import logging
import argparse
from src.data_processing.ingestion import (
    read_excel_files,
    filter_sheets,
    remove_duplicates_per_sheet,
    remove_unnecessary_columns,
    consolidate_datasets
)
from src.data_processing.column_unification import unificar_colunas_datasets
from src.data_processing.category_unification import unificar_categorias_completo, gerar_relatorio_final_categorias
from src.data_processing.feature_removal import remover_features_desnecessarias, listar_colunas_restantes
from src.data_processing.utm_training import unificar_utm_source_term, verificar_consistencia_utm
from src.data_processing.medium_training import extrair_publico_medium, relatorio_final_medium
from src.data_processing.medium_production_training import unificar_medium_para_producao, relatorio_unificacao_producao
from src.data_processing.dataset_versioning_training import criar_dataset_pos_cutoff, disponibilizar_dataset
from src.matching.matching_training import fazer_matching_robusto as fazer_matching_variantes
from src.matching.matching_robusto import fazer_matching_robusto
from src.matching.matching_email_only import fazer_matching_email_only
from src.data_processing.devclub_filtering_training import criar_dataset_devclub
from src.features.feature_engineering_training import criar_features_derivadas
from src.features.encoding_training import aplicar_encoding_estrategico
from src.model.training_model import registrar_features_e_modelo_devclub

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main(initial_matching='email_only'):
    """Executa pipeline de treino completo.

    Args:
        initial_matching: Método de matching inicial na célula 15
                         ('email_only', 'variantes' ou 'robusto')
    """

    print("\n" + "=" * 80)
    print("PIPELINE DE TREINO")
    print("=" * 80)
    print(f"\n🔧 CONFIGURAÇÃO:")
    print(f"   Método de matching inicial (célula 15): {initial_matching}")
    print("=" * 80)

    # Carregar configuração
    config_path = os.path.join(os.path.dirname(__file__), '../configs/devclub.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # === CÉLULA 1: Upload/Leitura de arquivos ===
    print("\n📤 CÉLULA 1: LEITURA DE ARQUIVOS")
    data_dir = config['ingestion']['training_data_dir']

    # Custom sorting para replicar ordem do notebook
    # No notebook, arquivos foram carregados via upload do Colab que preserva
    # a ordem do file picker (macOS/Linux), onde "[" vem antes de letras
    def notebook_sort_key(filepath):
        """Ordena arquivos para replicar a ordem do notebook."""
        basename = os.path.basename(filepath).lower()
        # Converter '[' para um caractere que vem antes de letras na ordenação
        # Usar '!' que tem ASCII 33, bem antes de letras
        return basename.replace('[', '!')

    filepaths = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")), key=notebook_sort_key)

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

    # === CÉLULA 3: Remoção de colunas desnecessárias ===
    print("\n🧹 CÉLULA 3: REMOÇÃO DE COLUNAS DESNECESSÁRIAS")
    print("=" * 60)

    clean_data_cols, cols_report = remove_unnecessary_columns(
        clean_data,
        colunas_remover=config['cleaning']['colunas_remover']
    )

    print(f"\n📊 COLUNAS REMOVIDAS POR ABA")
    print("=" * 80)
    print(f"{'ARQUIVO':<35} {'ABA':<20} {'ANTES':>10} {'DEPOIS':>10} {'REMOVIDAS':>10}")
    print("-" * 80)

    total_antes = 0
    total_depois = 0
    total_removidas_cols = 0

    for item in cols_report:
        print(f"{item['arquivo'][:34]:<35} {item['aba'][:19]:<20} "
              f"{item['colunas_antes']:>10} {item['colunas_depois']:>10} {item['removidas']:>10}")
        total_antes += item['colunas_antes']
        total_depois += item['colunas_depois']
        total_removidas_cols += item['removidas']

    print("-" * 80)
    print(f"{'TOTAL':<35} {'':<20} {total_antes:>10} {total_depois:>10} {total_removidas_cols:>10}")

    print(f"\n📈 RESUMO:")
    print(f"Total de colunas removidas: {total_removidas_cols}")
    print(f"\n✅ Dados sem colunas desnecessárias disponíveis")
    print("=" * 80)

    # === CÉLULA 4: Consolidação de datasets ===
    print("\nCONSOLIDAÇÃO DE DATASETS - PESQUISA E VENDAS")
    print("=" * 45)

    df_pesquisa, df_vendas = consolidate_datasets(
        clean_data_cols,
        pesquisa_keywords=config['consolidation']['pesquisa_keywords'],
        vendas_keywords=config['consolidation']['vendas_keywords']
    )

    # Função para gerar relatório de colunas (igual ao notebook)
    def gerar_relatorio_colunas(df, nome_dataset):
        """Gera relatório detalhado das colunas de um dataset"""

        print(f"\n{nome_dataset.upper()} - {len(df)} registros")
        print("=" * 70)
        print(f"{'COLUNA':<35} {'ÚNICOS':>10} {'% AUSENTES':>12} {'TOTAL':>10}")
        print("-" * 70)

        for col in df.columns:
            valores_unicos = df[col].nunique()
            valores_ausentes = df[col].isnull().sum()
            pct_ausentes = (valores_ausentes / len(df)) * 100 if len(df) > 0 else 0
            total_registros = len(df)

            print(f"{col[:34]:<35} {valores_unicos:>10,} {pct_ausentes:>11.1f}% {total_registros:>10,}")

    # Gerar relatórios
    gerar_relatorio_colunas(df_pesquisa, "DATASET PESQUISA")
    gerar_relatorio_colunas(df_vendas, "DATASET VENDAS")

    print(f"\nRESUMO:")
    print(f"Dataset Pesquisa: {len(df_pesquisa):,} registros, {len(df_pesquisa.columns)} colunas")
    print(f"Dataset Vendas: {len(df_vendas):,} registros, {len(df_vendas.columns)} colunas")

    print(f"\nDatasets consolidados disponíveis nas variáveis:")
    print(f"- dataset_pesquisa_final")
    print(f"- dataset_vendas_final")

    # === CÉLULA 5: Unificação de colunas duplicadas ===
    print("\nUNIFICAÇÃO DE COLUNAS DUPLICADAS")
    print("=" * 32)

    df_pesquisa_final, df_vendas_final = unificar_colunas_datasets(df_pesquisa, df_vendas)

    print(f"\nRESULTADO:")
    print(f"Pesquisa: {len(df_pesquisa_final)} registros, {len(df_pesquisa_final.columns)} colunas")
    print(f"Vendas: {len(df_vendas_final)} registros, {len(df_vendas_final.columns)} colunas")

    # Gerar relatórios finais
    gerar_relatorio_colunas(df_pesquisa_final, "DATASET PESQUISA")
    gerar_relatorio_colunas(df_vendas_final, "DATASET VENDAS")

    # === CÉLULA 7: Unificação completa de categorias ===
    print("\nUNIFICAÇÃO COMPLETA DE CATEGORIAS - NOVO CÓDIGO")
    print("=" * 52)

    df_pesquisa_final_unificado = unificar_categorias_completo(df_pesquisa_final)

    # Gerar relatório final
    gerar_relatorio_final_categorias(df_pesquisa_final_unificado)

    # === CÉLULA 8: Remoção de features desnecessárias ===
    print("\nREMOÇÃO DE FEATURES DESNECESSÁRIAS")
    print("=" * 38)

    df_features_removidas = remover_features_desnecessarias(df_pesquisa_final_unificado)

    # Listar colunas restantes
    listar_colunas_restantes(df_features_removidas)

    # === CÉLULA 10: Unificação de UTM Source e Term ===
    print("\nUNIFICAÇÃO DE UTM SOURCE E TERM")
    print("=" * 35)

    df_utm_unificado = unificar_utm_source_term(df_features_removidas)

    # Verificar consistência
    verificar_consistencia_utm(df_utm_unificado)

    # === CÉLULA 11: Unificação de UTM Medium - Extração de Públicos ===
    print("\nUNIFICAÇÃO DE UTM MEDIUM - EXTRAÇÃO DE PÚBLICOS")
    print("=" * 52)

    df_medium_unificado = extrair_publico_medium(df_utm_unificado)

    # Gerar relatório final
    relatorio_final_medium(df_medium_unificado)

    # === CÉLULA 11.1: Unificação de Medium para Produção ===
    print("\nUNIFICAÇÃO DE UTM MEDIUM BASEADA EM ACTIONS + TRATAMENTO DE PRODUÇÃO")
    print("=" * 72)

    print("Iniciando processo de unificação para produção...")
    df_original = df_medium_unificado.copy()
    df_medium_producao = unificar_medium_para_producao(df_medium_unificado)

    # Gerar relatório
    relatorio_unificacao_producao(df_original, df_medium_producao)

    print(f"\nProcesso concluído!")
    print(f"Dataset final disponível em: pesquisa_medium_producao_unificado")
    print(f"Este dataset está pronto para o pipeline de produção e não gerará incompatibilidades!")

    # === CÉLULA 13: Criação de versão do dataset por missing rate ===
    print("\nCRIAÇÃO DE VERSÕES DO DATASET POR MISSING RATE")
    print("=" * 50)

    print("Iniciando criação das versões...")
    df_pos_cutoff = criar_dataset_pos_cutoff(df_medium_producao)

    # Disponibilizar dataset
    disponibilizar_dataset(df_pos_cutoff)

    print(f"\nProcesso concluído!")
    print(f"Duas versões do dataset criadas com sucesso.")

    # === CÉLULA 15: Matching robusto por email e telefone ===
    if initial_matching == 'email_only':
        dataset_v1_final = fazer_matching_email_only(df_pos_cutoff, df_vendas_final)
    elif initial_matching == 'variantes':
        dataset_v1_final = fazer_matching_variantes(df_pos_cutoff, df_vendas_final)
    elif initial_matching == 'robusto':
        dataset_v1_final = fazer_matching_robusto(df_pos_cutoff, df_vendas_final)
    else:
        raise ValueError(f"Método de matching inicial inválido: {initial_matching}. Use 'email_only', 'variantes' ou 'robusto'")

    # === CÉLULA 17: Filtragem DevClub ===
    dataset_v1_devclub = criar_dataset_devclub(dataset_v1_final, df_vendas_final)

    # === CÉLULA 18: Feature Engineering ===
    dataset_v1_devclub_fe = criar_features_derivadas(dataset_v1_devclub)

    # === CÉLULA 20: Encoding Estratégico ===
    dataset_v1_devclub_encoded = aplicar_encoding_estrategico(dataset_v1_devclub_fe)

    # === CÉLULA MODELAGEM: Treino e Registro do Modelo ===
    resultado_registro_devclub = registrar_features_e_modelo_devclub(dataset_v1_devclub_encoded, dataset_v1_devclub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline de treino DevClub')
    parser.add_argument(
        '--initial-matching',
        type=str,
        choices=['email_only', 'variantes', 'robusto'],
        default='email_only',
        help='Método de matching inicial (célula 15) - padrão: email_only (100%% monotonia, máxima precisão)'
    )

    args = parser.parse_args()
    main(initial_matching=args.initial_matching)
