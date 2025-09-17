"""
Pipeline principal de lead scoring em produção.
APENAS orquestra componentes, sem conter lógica própria.
"""

import pandas as pd
import logging
from .data.preprocessing import remove_duplicates, clean_columns, remove_campaign_features
from .data.utm_unification import unify_utm_columns
from .data.medium_unification import unify_medium_columns
from .features.engineering import create_derived_features
from .features.encoding import apply_categorical_encoding

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LeadScoringPipeline:
    """Pipeline de produção para lead scoring."""

    def __init__(self):
        """Inicializa o pipeline."""
        self.data = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega arquivo de leads no formato Excel.

        Args:
            filepath: Caminho para o arquivo Excel

        Returns:
            DataFrame com os dados carregados
        """
        logger.info(f"Carregando arquivo: {filepath}")
        self.data = pd.read_excel(filepath)
        logger.info(f"Arquivo carregado: {len(self.data)} linhas, {len(self.data.columns)} colunas")
        return self.data

    def preprocess(self) -> pd.DataFrame:
        """
        Aplica pré-processamento aos dados.

        Returns:
            DataFrame pré-processado
        """
        if self.data is None:
            raise ValueError("Dados não carregados. Use load_data() primeiro.")

        initial_rows = len(self.data)
        initial_cols = len(self.data.columns)

        logger.info(f"📊 INÍCIO DO PIPELINE: {initial_rows} linhas, {initial_cols} colunas")

        # 1. Remover duplicatas (usando componente importado)
        logger.info("🔄 [1/6] Removendo duplicatas...")
        self.data = remove_duplicates(self.data)

        duplicates_removed = initial_rows - len(self.data)
        logger.info(f"   ➤ Duplicatas removidas: {duplicates_removed}")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 2. Limpar colunas desnecessárias (usando componente importado)
        logger.info("🔄 [2/6] Removendo colunas score/faixa...")
        cols_before_clean = len(self.data.columns)
        self.data = clean_columns(self.data)

        columns_removed = cols_before_clean - len(self.data.columns)
        logger.info(f"   ➤ Colunas de score/faixa removidas: {columns_removed}")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 3. Remover features de campanha (usando componente importado)
        logger.info("🔄 [3/6] Removendo features de campanha...")
        cols_before_campaign = len(self.data.columns)
        self.data = remove_campaign_features(self.data)

        campaign_cols_removed = cols_before_campaign - len(self.data.columns)
        logger.info(f"   ➤ Features de campanha removidas: {campaign_cols_removed}")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 4. Unificar categorias UTM (usando componente importado)
        logger.info("🔄 [4/6] Unificando categorias UTM...")
        utm_source_before = self.data['Source'].nunique() if 'Source' in self.data.columns else 0
        utm_term_before = self.data['Term'].nunique() if 'Term' in self.data.columns else 0

        self.data = unify_utm_columns(self.data)

        utm_source_after = self.data['Source'].nunique() if 'Source' in self.data.columns else 0
        utm_term_after = self.data['Term'].nunique() if 'Term' in self.data.columns else 0
        logger.info(f"   ➤ Source: {utm_source_before}→{utm_source_after} categorias")
        logger.info(f"   ➤ Term: {utm_term_before}→{utm_term_after} categorias")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 5. Unificar categorias Medium (usando componente importado)
        logger.info("🔄 [5/6] Unificando categorias Medium...")
        medium_before = self.data['Medium'].nunique() if 'Medium' in self.data.columns else 0

        self.data = unify_medium_columns(self.data)

        medium_after = self.data['Medium'].nunique() if 'Medium' in self.data.columns else 0
        logger.info(f"   ➤ Medium: {medium_before}→{medium_after} categorias")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 6. Engenharia de features (usando componente importado)
        logger.info("🔄 [6/7] Aplicando engenharia de features...")
        cols_before_fe = len(self.data.columns)

        # Verificar se colunas necessárias existem
        fe_input_cols = ['Data', 'Nome Completo', 'E-mail', 'Telefone']
        available_fe_cols = [col for col in fe_input_cols if col in self.data.columns]
        logger.info(f"   ➤ Colunas disponíveis para FE: {available_fe_cols}")

        self.data = create_derived_features(self.data)

        cols_added = len(self.data.columns) - cols_before_fe
        logger.info(f"   ➤ Features criadas: {cols_added} novas colunas")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 7. Encoding categórico (usando componente importado)
        logger.info("🔄 [7/7] Aplicando encoding categórico...")
        cols_before_encoding = len(self.data.columns)

        self.data = apply_categorical_encoding(self.data)

        encoding_cols_added = len(self.data.columns) - cols_before_encoding
        logger.info(f"   ➤ Colunas adicionadas pelo encoding: {encoding_cols_added}")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # Resumo final
        final_rows = len(self.data)
        final_cols = len(self.data.columns)
        total_rows_removed = initial_rows - final_rows
        net_cols_change = final_cols - initial_cols

        logger.info(f"📊 RESUMO FINAL:")
        logger.info(f"   ➤ Linhas: {initial_rows}→{final_rows} (removidas: {total_rows_removed})")
        logger.info(f"   ➤ Colunas: {initial_cols}→{final_cols} (variação: {net_cols_change:+d})")

        return self.data

    def run(self, filepath: str) -> pd.DataFrame:
        """
        Executa o pipeline completo.

        Args:
            filepath: Caminho para o arquivo de entrada

        Returns:
            DataFrame processado
        """
        logger.info("=== Iniciando Pipeline de Lead Scoring ===")

        # Carregar dados
        self.load_data(filepath)

        # Pré-processar
        self.preprocess()

        logger.info("=== Pipeline concluído ===")
        return self.data