"""
Módulo de ingestão de dados para lead scoring.

Funções:
- read_excel_files(): Leitura de múltiplos arquivos Excel
- filter_sheets(): Filtragem de abas por critérios configuráveis
- remove_duplicates_per_sheet(): Remoção de duplicatas por aba

Extraído do notebook DevClub e tornado configurável.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def read_excel_files(filepaths: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Lê múltiplos arquivos Excel e retorna estrutura organizada.

    Esta função reproduz a lógica das linhas 38-45 do notebook DevClub:
    - Itera sobre múltiplos arquivos Excel
    - Lê todas as abas de cada arquivo
    - Retorna estrutura {filename: {sheet_name: DataFrame}}

    Args:
        filepaths: Lista de caminhos para arquivos Excel (.xlsx ou .xls)

    Returns:
        Dicionário com estrutura:
        {
            'arquivo1.xlsx': {
                'aba1': DataFrame,
                'aba2': DataFrame
            },
            'arquivo2.xlsx': {
                'aba1': DataFrame
            }
        }

    Raises:
        FileNotFoundError: Se algum arquivo não existir
        ValueError: Se a lista de arquivos estiver vazia

    Example:
        >>> files = ['data/LF19.xlsx', 'data/LF20.xlsx']
        >>> data = read_excel_files(files)
        >>> print(data.keys())  # ['LF19.xlsx', 'LF20.xlsx']
    """
    if not filepaths:
        raise ValueError("Lista de arquivos não pode estar vazia")

    logger.info(f"📂 Lendo {len(filepaths)} arquivo(s) Excel...")

    all_data = {}

    for filepath in filepaths:
        # Verificar se arquivo existe
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        filename = Path(filepath).name
        logger.info(f"  Processando: {filename}")

        try:
            # Ler arquivo Excel
            xl_file = pd.ExcelFile(filepath)
            file_data = {}

            # Ler todas as abas
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(xl_file, sheet_name=sheet_name)
                file_data[sheet_name] = df
                logger.debug(f"    ✅ Aba '{sheet_name}': {len(df)} linhas, {len(df.columns)} colunas")

            all_data[filename] = file_data
            logger.info(f"    Total: {len(file_data)} aba(s) lida(s)")

        except Exception as e:
            logger.error(f"    ❌ Erro ao ler {filename}: {e}")
            raise

    logger.info(f"✅ Total de arquivos lidos: {len(all_data)}")

    return all_data


def filter_sheets(
    files_data: Dict[str, Dict[str, pd.DataFrame]],
    termos_manter: List[str],
    termos_remover: List[str],
    min_linhas: int
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], List[Dict]]:
    """
    Filtra abas de múltiplos arquivos Excel baseado em critérios configuráveis.

    Reproduz a lógica de filtragem das linhas 48-59 do notebook DevClub.

    Args:
        files_data: Dicionário {filename: {sheet_name: DataFrame}}
        termos_manter: Lista de termos que as abas devem conter para serem mantidas
        termos_remover: Lista de termos que, se presentes, fazem a aba ser removida
        min_linhas: Número mínimo de linhas para manter uma aba

    Returns:
        Tupla (arquivos_filtrados, relatório):
        - arquivos_filtrados: Dict com abas que passaram nos critérios
        - relatório: Lista de dicts com informações sobre cada aba (mantida ou removida)

    Example:
        >>> filtered, report = filter_sheets(
        ...     data,
        ...     termos_manter=["Pesquisa", "Vendas"],
        ...     termos_remover=["Pontuação", "Lead Score"],
        ...     min_linhas=230
        ... )
    """
    logger.info("🔍 Filtrando abas por critérios...")

    arquivos_filtrados = {}
    relatorio = []

    for filename, sheets in files_data.items():
        abas_filtradas = {}

        for sheet_name, df in sheets.items():
            linhas_original = len(df)

            # APLICAR CRITÉRIOS DE FILTRAGEM (linhas 48-59 do notebook)
            deve_remover_por_termo = any(termo.lower() in sheet_name.lower() for termo in termos_remover)
            tem_termo_permitido = any(termo.lower() in sheet_name.lower() for termo in termos_manter)
            tem_linhas_suficientes = len(df) >= min_linhas
            nao_esta_vazia = len(df) > 0 and not df.empty

            # Critério específico DevClub: remover abas TMB/Guru de arquivos LF (exceto LF06)
            eh_lf_com_vendas = (
                'LF' in filename and
                any(vendas_termo.lower() in sheet_name.lower() for vendas_termo in ['tmb', 'guru']) and
                'LF06' not in filename
            )

            # Decidir se mantém a aba
            if (nao_esta_vazia and not deve_remover_por_termo and not eh_lf_com_vendas and
                (tem_termo_permitido or tem_linhas_suficientes)):

                # Aba MANTIDA
                abas_filtradas[sheet_name] = df

                relatorio.append({
                    'arquivo': filename,
                    'aba': sheet_name,
                    'linhas_original': linhas_original,
                    'status': 'MANTIDA'
                })
                logger.debug(f"  ✅ {filename} - {sheet_name}: {linhas_original:,} linhas (MANTIDA)")

            else:
                # Aba REMOVIDA
                relatorio.append({
                    'arquivo': filename,
                    'aba': sheet_name,
                    'linhas_original': linhas_original,
                    'status': 'REMOVIDA'
                })
                logger.debug(f"  ❌ {filename} - {sheet_name}: removida pelos critérios")

        # Salvar arquivo se tiver abas válidas
        if abas_filtradas:
            arquivos_filtrados[filename] = abas_filtradas

    abas_mantidas = sum(1 for item in relatorio if item['status'] == 'MANTIDA')
    abas_removidas = len(relatorio) - abas_mantidas

    logger.info(f"  Abas mantidas: {abas_mantidas}")
    logger.info(f"  Abas removidas: {abas_removidas}")

    return arquivos_filtrados, relatorio


def remove_duplicates_per_sheet(
    files_data: Dict[str, Dict[str, pd.DataFrame]]
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, int]]]:
    """
    Remove duplicatas de cada aba de cada arquivo.

    Reproduz a lógica da linha 62 do notebook DevClub.

    Args:
        files_data: Dicionário {filename: {sheet_name: DataFrame}}

    Returns:
        Tupla (arquivos_limpos, estatísticas):
        - arquivos_limpos: Dict com DataFrames sem duplicatas
        - estatísticas: Dict {filename: {sheet_name: duplicatas_removidas}}

    Example:
        >>> clean_data, stats = remove_duplicates_per_sheet(data)
    """
    logger.info("🧹 Removendo duplicatas...")

    arquivos_limpos = {}
    estatisticas = {}

    for filename, sheets in files_data.items():
        abas_limpas = {}
        stats_arquivo = {}

        for sheet_name, df in sheets.items():
            linhas_antes = len(df)

            # REMOVER DUPLICATAS (linha 62 do notebook)
            df_limpo = df.drop_duplicates(keep='first')
            linhas_depois = len(df_limpo)
            duplicatas_removidas = linhas_antes - linhas_depois

            abas_limpas[sheet_name] = df_limpo
            stats_arquivo[sheet_name] = duplicatas_removidas

            if duplicatas_removidas > 0:
                logger.debug(f"  {filename} - {sheet_name}: {duplicatas_removidas} duplicatas removidas")

        arquivos_limpos[filename] = abas_limpas
        estatisticas[filename] = stats_arquivo

    total_duplicatas = sum(sum(stats.values()) for stats in estatisticas.values())
    logger.info(f"  Total de duplicatas removidas: {total_duplicatas:,}")

    return arquivos_limpos, estatisticas
