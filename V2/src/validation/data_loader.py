"""
Módulo para carregamento e normalização de dados de leads e vendas.

Este módulo fornece classes para carregar dados de:
- Google Sheets (CSV com leads e scores)
- Guru (Excel com vendas)
- TMB (Excel com vendas)

Todas as funções normalizam emails, telefones e datas para garantir
matching consistente.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path
import logging
import re

# Importar funções de normalização existentes
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.matching.matching_email_telefone import normalizar_email, normalizar_telefone_robusto

logger = logging.getLogger(__name__)


class LeadDataLoader:
    """
    Carrega e normaliza dados de leads do Google Sheets.

    CSV esperado contém:
    - Data: Timestamp da captura
    - E-mail: Email do lead
    - Nome Completo: Nome completo
    - Telefone: Telefone com DDD
    - Campaign: Nome da campanha UTM
    - lead_score: Score do modelo (0-1)
    - Source, Medium, Term, Content: UTMs
    """

    def __init__(self):
        self.required_columns = ['Data', 'E-mail', 'Campaign']
        self._thresholds_cache = None  # Cache dos thresholds do modelo

    def load_leads_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Carrega CSV de leads do Google Sheets e normaliza.

        Args:
            csv_path: Caminho para o CSV

        Returns:
            DataFrame normalizado com colunas:
            - email: Email normalizado
            - nome: Nome completo
            - telefone: Telefone normalizado
            - data_captura: Datetime da captura
            - campaign: Nome da campanha
            - lead_score: Score do modelo
            - decile: Decil (D1-D10)
            - source, medium, term, content: UTMs
        """
        logger.info(f"📂 Carregando leads de {csv_path}")

        # Ler CSV
        df = pd.read_csv(csv_path)
        logger.info(f"   {len(df)} linhas lidas do CSV")

        # Verificar colunas obrigatórias
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        # Normalizar nomes de colunas
        df_norm = pd.DataFrame()

        # Email (normalizado)
        df_norm['email'] = df['E-mail'].apply(lambda x: normalizar_email(x) if pd.notna(x) else None)

        # Nome
        df_norm['nome'] = df.get('Nome Completo', np.nan)

        # Telefone (normalizado)
        if 'Telefone' in df.columns:
            df_norm['telefone'] = df['Telefone'].apply(
                lambda x: normalizar_telefone_robusto(str(x)) if pd.notna(x) else None
            )
        else:
            df_norm['telefone'] = None

        # Data de captura
        df_norm['data_captura'] = pd.to_datetime(df['Data'], errors='coerce')

        # Campanha e UTMs
        df_norm['campaign'] = df['Campaign']
        df_norm['source'] = df.get('Source', np.nan)
        df_norm['medium'] = df.get('Medium', np.nan)
        df_norm['term'] = df.get('Term', np.nan)
        df_norm['content'] = df.get('Content', np.nan)

        # Lead Score e Decil
        df_norm['lead_score'] = df.get('lead_score', np.nan)

        # Extrair decil: PRIORIZAR lead_score (ML) sobre Faixa (legacy)
        if 'lead_score' in df.columns and df['lead_score'].notna().any():
            # PRIORITY 1: ML model scores (53.3% coverage)
            df_norm['decile'] = df['lead_score'].apply(self._assign_decile_from_score)
            logger.info(f"   ✅ Decis atribuídos via lead_score: {df_norm['decile'].notna().sum()}/{len(df_norm)}")
        elif 'Faixa' in df.columns and df['Faixa'].notna().any():
            # FALLBACK: Legacy classification (4.3% coverage)
            df_norm['decile'] = df['Faixa']
            logger.info(f"   ⚠️ Decis atribuídos via Faixa (legacy): {df_norm['decile'].notna().sum()}/{len(df_norm)}")
        else:
            df_norm['decile'] = None
            logger.warning("⚠️ Nenhuma coluna de score/decil encontrada")

        # Remover linhas com email inválido
        before = len(df_norm)
        df_norm = df_norm[df_norm['email'].notna()].copy()
        after = len(df_norm)

        if before != after:
            logger.warning(f"⚠️ {before - after} leads removidos (email inválido)")

        logger.info(f"   ✅ {len(df_norm)} leads carregados e normalizados")

        return df_norm

    def _get_thresholds(self) -> dict:
        """
        Carrega thresholds do JSON do modelo (lazy loading com cache).

        Returns:
            Dict com thresholds por decil no formato:
            {'D1': {'threshold_min': ..., 'threshold_max': ...}, ...}
        """
        if self._thresholds_cache is None:
            import json

            # Caminho para o JSON canônico do modelo ativo
            metadata_path = Path(__file__).parent.parent.parent / \
                "files" / "20251111_212345" / \
                "model_metadata_v1_devclub_rf_temporal_single.json"

            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Arquivo de metadata do modelo não encontrado: {metadata_path}"
                )

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            self._thresholds_cache = metadata['decil_thresholds']['thresholds']
            logger.debug(f"✅ Thresholds carregados do modelo: {len(self._thresholds_cache)} decis")

        return self._thresholds_cache

    def _assign_decile_from_score(self, score) -> Optional[str]:
        """
        Atribui decil baseado no score usando módulo decil_thresholds.

        Usa thresholds do JSON canônico do modelo ativo:
        V2/files/20251111_212345/model_metadata_v1_devclub_rf_temporal_single.json

        Args:
            score: Lead score (0-1), pode ser string com vírgula

        Returns:
            Label do decil (D1-D10) ou None se score inválido
        """
        if pd.isna(score):
            return None

        # Convert string to float (handle comma decimal separator)
        if isinstance(score, str):
            try:
                # Replace comma with dot: "0,1572" → "0.1572"
                score_float = float(score.replace(',', '.'))
            except (ValueError, AttributeError):
                logger.warning(f"⚠️ Score inválido (não numérico): {score}")
                return None
        else:
            score_float = float(score)

        # Validate range
        if not (0 <= score_float <= 1):
            logger.warning(f"⚠️ Score fora do range [0,1]: {score_float}")
            return None

        # Importar função do módulo existente
        from src.model.decil_thresholds import atribuir_decil_por_threshold

        # Carregar thresholds (com cache)
        thresholds = self._get_thresholds()

        # Atribuir decil usando função do módulo
        return atribuir_decil_por_threshold(score_float, thresholds)


class SalesDataLoader:
    """
    Carrega e normaliza dados de vendas da Guru e TMB.

    Combina dados de ambas as plataformas em formato padronizado.
    """

    def __init__(self):
        pass

    def load_guru_sales(self, guru_paths: List[str]) -> pd.DataFrame:
        """
        Carrega arquivos Excel de vendas da Guru.

        Colunas esperadas:
        - email contato: Email do comprador
        - nome contato: Nome
        - valor venda: Valor da transação
        - utm_campaign: Campanha de origem
        - data pedido / data aprovacao: Data da compra

        Args:
            guru_paths: Lista de caminhos para arquivos Excel da Guru

        Returns:
            DataFrame normalizado com origem='guru'
        """
        if not guru_paths:
            logger.warning("⚠️ Nenhum arquivo Guru fornecido")
            return pd.DataFrame()

        logger.info(f"📂 Carregando vendas Guru de {len(guru_paths)} arquivo(s)")

        all_sales = []

        for path in guru_paths:
            try:
                df = pd.read_excel(path)
                logger.info(f"   {len(df)} vendas de {Path(path).name}")
                all_sales.append(df)
            except Exception as e:
                logger.error(f"❌ Erro ao ler {path}: {e}")
                continue

        if not all_sales:
            logger.warning("⚠️ Nenhuma venda Guru carregada")
            return pd.DataFrame()

        # Combinar todos os DataFrames
        df_combined = pd.concat(all_sales, ignore_index=True)

        # Normalizar colunas
        df_norm = pd.DataFrame()

        # Email (normalizado)
        df_norm['email'] = df_combined['email contato'].apply(
            lambda x: normalizar_email(x) if pd.notna(x) else None
        )

        # Nome
        df_norm['nome'] = df_combined.get('nome contato', np.nan)

        # Telefone (se disponível)
        if 'telefone contato' in df_combined.columns:
            df_norm['telefone'] = df_combined['telefone contato'].apply(
                lambda x: normalizar_telefone_robusto(str(x)) if pd.notna(x) else None
            )
        else:
            df_norm['telefone'] = None

        # Valor da venda
        df_norm['sale_value'] = pd.to_numeric(df_combined.get('valor venda', 0), errors='coerce')

        # Data da venda (tentar múltiplas colunas)
        date_col = None
        for col in ['data aprovacao', 'data pedido', 'data_aprovacao', 'data_pedido']:
            if col in df_combined.columns:
                date_col = col
                break

        if date_col:
            df_norm['sale_date'] = pd.to_datetime(df_combined[date_col], errors='coerce')
        else:
            df_norm['sale_date'] = None
            logger.warning("⚠️ Coluna de data não encontrada em vendas Guru")

        # UTM Campaign
        df_norm['utm_campaign'] = df_combined.get('utm_campaign', np.nan)

        # Origem
        df_norm['origem'] = 'guru'

        # Remover vendas sem email ou data
        before = len(df_norm)
        df_norm = df_norm[
            (df_norm['email'].notna()) &
            (df_norm['sale_date'].notna())
        ].copy()
        after = len(df_norm)

        if before != after:
            logger.warning(f"⚠️ {before - after} vendas Guru removidas (email/data inválido)")

        logger.info(f"   ✅ {len(df_norm)} vendas Guru carregadas e normalizadas")

        return df_norm

    def load_tmb_sales(self, tmb_paths: List[str]) -> pd.DataFrame:
        """
        Carrega arquivos Excel de vendas da TMB.

        Colunas esperadas:
        - Cliente Email: Email do comprador
        - Cliente Nome: Nome
        - Ticket (R$): Valor da transação
        - utm_campaign: Campanha de origem
        - Status: Status do pedido (filtrar apenas 'Efetivado')

        Args:
            tmb_paths: Lista de caminhos para arquivos Excel da TMB

        Returns:
            DataFrame normalizado com origem='tmb'
        """
        if not tmb_paths:
            logger.warning("⚠️ Nenhum arquivo TMB fornecido")
            return pd.DataFrame()

        logger.info(f"📂 Carregando vendas TMB de {len(tmb_paths)} arquivo(s)")

        all_sales = []

        for path in tmb_paths:
            try:
                df = pd.read_excel(path)
                logger.info(f"   {len(df)} vendas de {Path(path).name}")
                all_sales.append(df)
            except Exception as e:
                logger.error(f"❌ Erro ao ler {path}: {e}")
                continue

        if not all_sales:
            logger.warning("⚠️ Nenhuma venda TMB carregada")
            return pd.DataFrame()

        # Combinar todos os DataFrames
        df_combined = pd.concat(all_sales, ignore_index=True)

        # Filtrar apenas vendas efetivadas
        if 'Status' in df_combined.columns:
            before = len(df_combined)
            df_combined = df_combined[df_combined['Status'] == 'Efetivado'].copy()
            after = len(df_combined)
            logger.info(f"   Filtradas {after} vendas efetivadas de {before} total")

        # Normalizar colunas
        df_norm = pd.DataFrame()

        # Email (normalizado)
        email_col = 'Cliente Email' if 'Cliente Email' in df_combined.columns else 'Cliente E-mail'
        df_norm['email'] = df_combined[email_col].apply(
            lambda x: normalizar_email(x) if pd.notna(x) else None
        )

        # Nome
        df_norm['nome'] = df_combined.get('Cliente Nome', np.nan)

        # Telefone (se disponível)
        if 'Telefone' in df_combined.columns or 'Cliente Telefone' in df_combined.columns:
            phone_col = 'Telefone' if 'Telefone' in df_combined.columns else 'Cliente Telefone'
            df_norm['telefone'] = df_combined[phone_col].apply(
                lambda x: normalizar_telefone_robusto(str(x)) if pd.notna(x) else None
            )
        else:
            df_norm['telefone'] = None

        # Valor da venda
        ticket_col = 'Ticket (R$)' if 'Ticket (R$)' in df_combined.columns else 'Ticket'
        df_norm['sale_value'] = pd.to_numeric(df_combined.get(ticket_col, 0), errors='coerce')

        # Data da venda (tentar múltiplas colunas)
        date_col = None
        for col in ['Data Efetivado', 'Data Pedido', 'Data do Pedido', 'Data']:
            if col in df_combined.columns:
                date_col = col
                break

        if date_col:
            df_norm['sale_date'] = pd.to_datetime(df_combined[date_col], errors='coerce')
        else:
            df_norm['sale_date'] = None
            logger.warning("⚠️ Coluna de data não encontrada em vendas TMB")

        # UTM Campaign
        df_norm['utm_campaign'] = df_combined.get('utm_campaign', np.nan)

        # Origem
        df_norm['origem'] = 'tmb'

        # Remover vendas sem email ou data
        before = len(df_norm)
        df_norm = df_norm[
            (df_norm['email'].notna()) &
            (df_norm['sale_date'].notna())
        ].copy()
        after = len(df_norm)

        if before != after:
            logger.warning(f"⚠️ {before - after} vendas TMB removidas (email/data inválido)")

        logger.info(f"   ✅ {len(df_norm)} vendas TMB carregadas e normalizadas")

        return df_norm

    def combine_sales(self, guru_df: pd.DataFrame = None, tmb_df: pd.DataFrame = None,
                     guru_paths: List[str] = None, tmb_paths: List[str] = None) -> pd.DataFrame:
        """
        Combina vendas da Guru e TMB em um único DataFrame.

        Args:
            guru_df: DataFrame já carregado da Guru (opcional)
            tmb_df: DataFrame já carregado da TMB (opcional)
            guru_paths: Caminhos para arquivos Guru (se guru_df não fornecido)
            tmb_paths: Caminhos para arquivos TMB (se tmb_df não fornecido)

        Returns:
            DataFrame combinado e deduplicado (prioriza Guru em caso de conflito)
        """
        logger.info("🔗 Combinando vendas Guru + TMB")

        # Carregar se necessário
        if guru_df is None and guru_paths:
            guru_df = self.load_guru_sales(guru_paths)
        if tmb_df is None and tmb_paths:
            tmb_df = self.load_tmb_sales(tmb_paths)

        # Combinar DataFrames
        dfs = []
        if guru_df is not None and len(guru_df) > 0:
            dfs.append(guru_df)
        if tmb_df is not None and len(tmb_df) > 0:
            dfs.append(tmb_df)

        if not dfs:
            logger.warning("⚠️ Nenhuma venda para combinar")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        # Deduplicar (priorizar Guru se mesmo email+data)
        # Ordenar por origem (guru primeiro) e remover duplicatas
        combined['_priority'] = combined['origem'].apply(lambda x: 0 if x == 'guru' else 1)
        combined = combined.sort_values('_priority')

        before = len(combined)
        combined = combined.drop_duplicates(subset=['email', 'sale_date'], keep='first')
        combined = combined.drop(columns=['_priority'])
        after = len(combined)

        if before != after:
            logger.info(f"   Removidas {before - after} duplicatas (priorizando Guru)")

        logger.info(f"   ✅ {len(combined)} vendas únicas combinadas")
        logger.info(f"      Guru: {len(combined[combined['origem'] == 'guru'])}")
        logger.info(f"      TMB: {len(combined[combined['origem'] == 'tmb'])}")

        return combined
