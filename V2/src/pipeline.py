"""
Pipeline principal de lead scoring em produção.
APENAS orquestra componentes, sem conter lógica própria.
Reproduz EXATAMENTE a lógica do notebook com parâmetros configuráveis.
"""

import pandas as pd
import logging
from .data.preprocessing import remove_duplicates, clean_columns, remove_campaign_features, remove_technical_fields, rename_long_column_names
from .data.utm_unification import unify_utm_columns
from .data.medium_unification import unify_medium_columns
from .features.engineering import create_derived_features
from .features.encoding import apply_categorical_encoding
from .model.prediction import LeadScoringPredictor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LeadScoringPipeline:
    """
    Pipeline de produção para lead scoring.

    Reproduz EXATAMENTE a lógica do notebook com parâmetros configuráveis.
    """

    def __init__(self, model_name: str = "v1_devclub_rf_temporal_single"):
        """
        Inicializa o pipeline com configuração fixa.

        Args:
            model_name: Nome do modelo a usar para predições (default: v1_devclub_rf_temporal_single)

        Configuração:
        - Mantém features UTM (com_utm=True)
        - Dataset V1 (versao="v1")
        - Sem cutoff temporal
        """
        self.data = None
        self.original_data = None  # Preservar dados originais
        self.predictor = LeadScoringPredictor(model_name)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega arquivo de leads no formato Excel ou CSV.

        Args:
            filepath: Caminho para o arquivo Excel ou CSV

        Returns:
            DataFrame com os dados carregados
        """
        logger.info(f"Carregando arquivo: {filepath}")

        # Detectar formato do arquivo pela extensão
        if filepath.lower().endswith('.csv'):
            self.data = pd.read_csv(filepath)
        elif filepath.lower().endswith(('.xlsx', '.xls')):
            self.data = pd.read_excel(filepath)
        else:
            # Tentar CSV primeiro, depois Excel
            try:
                self.data = pd.read_csv(filepath)
            except:
                self.data = pd.read_excel(filepath)

        self.original_data = self.data.copy()  # Preservar cópia original
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
        logger.info("🔄 [1/10] Removendo duplicatas...")
        self.data = remove_duplicates(self.data)

        duplicates_removed = initial_rows - len(self.data)
        logger.info(f"   ➤ Duplicatas removidas: {duplicates_removed}")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 2. Limpar colunas desnecessárias (usando componente importado)
        logger.info("🔄 [2/10] Removendo colunas score/faixa...")
        cols_before_clean = len(self.data.columns)
        self.data = clean_columns(self.data)

        columns_removed = cols_before_clean - len(self.data.columns)
        logger.info(f"   ➤ Colunas de score/faixa removidas: {columns_removed}")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 3. Remover features de campanha (usando componente importado)
        logger.info("🔄 [3/10] Removendo features de campanha...")
        cols_before_campaign = len(self.data.columns)
        self.data = remove_campaign_features(self.data)

        campaign_cols_removed = cols_before_campaign - len(self.data.columns)
        logger.info(f"   ➤ Features de campanha removidas: {campaign_cols_removed}")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 4. Unificar categorias UTM (usando componente importado)
        logger.info("🔄 [4/10] Unificando categorias UTM...")
        utm_source_before = self.data['Source'].nunique() if 'Source' in self.data.columns else 0
        utm_term_before = self.data['Term'].nunique() if 'Term' in self.data.columns else 0

        self.data = unify_utm_columns(self.data)

        utm_source_after = self.data['Source'].nunique() if 'Source' in self.data.columns else 0
        utm_term_after = self.data['Term'].nunique() if 'Term' in self.data.columns else 0
        logger.info(f"   ➤ Source: {utm_source_before}→{utm_source_after} categorias")
        logger.info(f"   ➤ Term: {utm_term_before}→{utm_term_after} categorias")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 5. Unificar categorias Medium (usando componente importado)
        logger.info("🔄 [5/10] Unificando categorias Medium...")
        medium_before = self.data['Medium'].nunique() if 'Medium' in self.data.columns else 0

        self.data = unify_medium_columns(self.data)

        medium_after = self.data['Medium'].nunique() if 'Medium' in self.data.columns else 0
        logger.info(f"   ➤ Medium: {medium_before}→{medium_after} categorias")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 6. Remover campos técnicos (usando componente importado)
        logger.info("🔄 [6/10] Removendo campos técnicos...")
        cols_before_tech = len(self.data.columns)
        self.data = remove_technical_fields(self.data)

        tech_cols_removed = cols_before_tech - len(self.data.columns)
        logger.info(f"   ➤ Campos técnicos removidos: {tech_cols_removed}")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 7. Renomear colunas longas (usando componente importado)
        logger.info("🔄 [7/10] Renomeando colunas longas...")
        self.data = rename_long_column_names(self.data)

        # Número de colunas deveria permanecer o mesmo (renomeação não adiciona/remove)
        logger.info(f"   ➤ Colunas renomeadas (mantém total): {len(self.data.columns)}")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 8. Engenharia de features (usando componente importado)
        logger.info("🔄 [8/10] Aplicando engenharia de features...")
        cols_before_fe = len(self.data.columns)

        # Verificar se colunas necessárias existem
        fe_input_cols = ['Data', 'Nome Completo', 'E-mail', 'Telefone']
        available_fe_cols = [col for col in fe_input_cols if col in self.data.columns]
        logger.info(f"   ➤ Colunas disponíveis para FE: {available_fe_cols}")

        self.data = create_derived_features(self.data)


        cols_added = len(self.data.columns) - cols_before_fe
        logger.info(f"   ➤ Features criadas/processadas: {cols_added} novas colunas")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 9. Encoding categórico (usando componente importado)
        logger.info("🔄 [9/10] Aplicando encoding categórico...")
        cols_before_encoding = len(self.data.columns)

        self.data = apply_categorical_encoding(self.data, versao="v1")

        encoding_cols_added = len(self.data.columns) - cols_before_encoding
        logger.info(f"   ➤ Colunas adicionadas pelo encoding: {encoding_cols_added}")
        logger.info(f"   ➤ Estado atual: {len(self.data)} linhas, {len(self.data.columns)} colunas")

        # 10. Manter features UTM (configuração fixa)
        logger.info("🔄 [10/10] Mantendo features UTM")

        # Resumo final
        final_rows = len(self.data)
        final_cols = len(self.data.columns)
        total_rows_removed = initial_rows - final_rows
        net_cols_change = final_cols - initial_cols

        logger.info(f"📊 RESUMO FINAL (v1, com UTM):")
        logger.info(f"   ➤ Linhas: {initial_rows}→{final_rows} (removidas: {total_rows_removed})")
        logger.info(f"   ➤ Colunas: {initial_cols}→{final_cols} (variação: {net_cols_change:+d})")

        return self.data

    def predict(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Realiza predições no DataFrame processado.

        Args:
            df: DataFrame a ser usado (se None, usa self.data)

        Returns:
            DataFrame original com scores de predição
        """
        if df is None:
            if self.data is None:
                raise ValueError("Nenhum dado disponível para predição. Execute preprocess() primeiro.")
            df = self.data

        logger.info("=== Iniciando Predições ===")
        # Passar tanto o DataFrame processado quanto o original
        result = self.predictor.predict(df, self.original_data)
        logger.info("=== Predições Concluídas ===")

        return result

    def run(self, filepath: str, with_predictions: bool = False) -> pd.DataFrame:
        """
        Executa o pipeline completo.

        Args:
            filepath: Caminho para o arquivo de entrada
            with_predictions: Se True, inclui predições no resultado

        Returns:
            DataFrame processado (com predições se solicitado)
        """
        logger.info("=== Iniciando Pipeline de Lead Scoring ===")

        # Carregar dados
        self.load_data(filepath)

        # Pré-processar
        self.preprocess()

        # Fazer predições se solicitado
        if with_predictions:
            self.data = self.predict()

        logger.info("=== Pipeline concluído ===")
        return self.data