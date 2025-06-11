# smart_ads_pipeline/data_handlers/data_loader.py

import os
import sys
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar funções existentes do pipeline unificado
from src.utils.local_storage import (
    connect_to_gcs, list_files_by_extension, categorize_files,
    load_csv_or_excel, load_csv_with_auto_delimiter, extract_launch_id
)
from src.utils.feature_naming import standardize_dataframe_columns
from src.preprocessing.email_processing import normalize_email

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Responsável por carregar e organizar os dados brutos.
    
    Esta classe encapsula toda a lógica de carregamento de arquivos,
    identificação de tipos (survey, buyer, utm) e organização por lançamento.
    """
    
    def __init__(self, data_path: str):
        """
        Inicializa o DataLoader.
        
        Args:
            data_path: Caminho para o diretório com os dados brutos
        """
        self.data_path = data_path
        self.bucket = None
        logger.info(f"DataLoader inicializado com path: {data_path}")
    
    def load_training_data(self) -> Dict[str, pd.DataFrame]:
        """
        Carrega todos os dados necessários para treino.
        
        Returns:
            Dicionário com DataFrames consolidados:
            {
                'surveys': DataFrame com todas as pesquisas,
                'buyers': DataFrame com todos os compradores,
                'utms': DataFrame com todos os dados UTM
            }
        """
        logger.info("Iniciando carregamento de dados de treino...")
        
        # Conectar ao armazenamento
        self.bucket = connect_to_gcs("local_bucket", data_path=self.data_path)
        
        # Listar e categorizar arquivos
        file_paths = list_files_by_extension(self.bucket, prefix="")
        logger.info(f"Encontrados {len(file_paths)} arquivos")
        
        # Categorizar arquivos
        survey_files, buyer_files, utm_files, all_files_by_launch = categorize_files(file_paths)
        
        logger.info(f"Arquivos categorizados:")
        logger.info(f"  - Surveys: {len(survey_files)}")
        logger.info(f"  - Buyers: {len(buyer_files)}")
        logger.info(f"  - UTMs: {len(utm_files)}")
        
        # Carregar cada tipo de arquivo
        surveys_df = self._load_survey_files(survey_files)
        buyers_df = self._load_buyer_files(buyer_files)
        utms_df = self._load_utm_files(utm_files)
        
        # Adicionar normalização de emails
        if not surveys_df.empty:
            surveys_df = self._normalize_emails(surveys_df)
        
        if not buyers_df.empty:
            buyers_df = self._normalize_emails(buyers_df)
            
        if not utms_df.empty:
            utms_df = self._normalize_emails(utms_df)
        
        result = {
            'surveys': surveys_df,
            'buyers': buyers_df,
            'utms': utms_df
        }
        
        logger.info("Carregamento de dados concluído")
        return result
    
    def load_prediction_data(self, data_source: Optional[str] = None,
                           dataframe: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Carrega dados para predição.
        
        Args:
            data_source: Caminho para arquivo CSV ou conexão com Google Sheets
            dataframe: DataFrame já carregado (se disponível)
            
        Returns:
            DataFrame com dados para predição
        """
        logger.info("Carregando dados para predição...")
        
        if dataframe is not None:
            logger.info("Usando DataFrame fornecido")
            df = dataframe.copy()
        elif data_source is not None:
            logger.info(f"Carregando dados de: {data_source}")
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            else:
                # TODO: Implementar carregamento do Google Sheets
                raise NotImplementedError("Carregamento do Google Sheets ainda não implementado")
        else:
            raise ValueError("Deve fornecer data_source ou dataframe")
        
        # Padronizar nomes de colunas
        df = standardize_dataframe_columns(df)
        
        # Adicionar email_norm se houver coluna de email
        df = self._normalize_emails(df)
        
        logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return df
    
    def _load_survey_files(self, survey_files: List[str]) -> pd.DataFrame:
        """Carrega e consolida arquivos de pesquisa."""
        if not survey_files:
            logger.warning("Nenhum arquivo de pesquisa encontrado")
            return pd.DataFrame()
        
        dfs = []
        for file_path in survey_files:
            try:
                df = self._load_file(file_path, file_type='survey')
                if df is not None:
                    # Adicionar identificador de lançamento
                    launch_id = extract_launch_id(file_path)
                    if launch_id:
                        df['lancamento'] = launch_id
                    dfs.append(df)
                    logger.debug(f"Carregado: {file_path} ({len(df)} linhas)")
            except Exception as e:
                logger.error(f"Erro ao carregar {file_path}: {e}")
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total de pesquisas carregadas: {len(result)} linhas")
            return result
        
        return pd.DataFrame()
    
    def _load_buyer_files(self, buyer_files: List[str]) -> pd.DataFrame:
        """Carrega e consolida arquivos de compradores."""
        if not buyer_files:
            logger.warning("Nenhum arquivo de compradores encontrado")
            return pd.DataFrame()
        
        dfs = []
        for file_path in buyer_files:
            try:
                df = self._load_file(file_path, file_type='buyer')
                if df is not None:
                    launch_id = extract_launch_id(file_path)
                    if launch_id:
                        df['lancamento'] = launch_id
                    dfs.append(df)
                    logger.debug(f"Carregado: {file_path} ({len(df)} linhas)")
            except Exception as e:
                logger.error(f"Erro ao carregar {file_path}: {e}")
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total de compradores carregados: {len(result)} linhas")
            return result
        
        return pd.DataFrame()
    
    def _load_utm_files(self, utm_files: List[str]) -> pd.DataFrame:
        """Carrega e consolida arquivos UTM."""
        if not utm_files:
            logger.warning("Nenhum arquivo UTM encontrado")
            return pd.DataFrame()
        
        dfs = []
        for file_path in utm_files:
            try:
                # UTMs usam função especial para detecção de delimitador
                df = self._load_file(file_path, file_type='utm')
                if df is not None:
                    launch_id = extract_launch_id(file_path)
                    if launch_id:
                        df['lancamento'] = launch_id
                    dfs.append(df)
                    logger.debug(f"Carregado: {file_path} ({len(df)} linhas)")
            except Exception as e:
                logger.error(f"Erro ao carregar {file_path}: {e}")
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total de UTMs carregados: {len(result)} linhas")
            return result
        
        return pd.DataFrame()
    
    def _load_file(self, file_path: str, file_type: str) -> Optional[pd.DataFrame]:
        """
        Carrega um arquivo individual.
        
        Args:
            file_path: Caminho do arquivo
            file_type: Tipo do arquivo ('survey', 'buyer', 'utm')
            
        Returns:
            DataFrame carregado ou None se houver erro
        """
        if file_type == 'utm':
            # UTMs precisam de detecção automática de delimitador
            return load_csv_with_auto_delimiter(self.bucket, file_path)
        else:
            # Outros tipos usam função padrão
            return load_csv_or_excel(self.bucket, file_path)
    
    def _normalize_emails(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona coluna email_norm ao DataFrame.
        
        Args:
            df: DataFrame para processar
            
        Returns:
            DataFrame com coluna email_norm adicionada
        """
        if df.empty:
            return df
        
        # Procurar colunas de email
        email_patterns = ['email', 'e_mail', 'mail', 'correo']
        email_cols = []
        
        for col in df.columns:
            if any(pattern in col.lower() for pattern in email_patterns):
                email_cols.append(col)
        
        if not email_cols:
            logger.debug("Nenhuma coluna de email encontrada")
            return df
        
        # Criar email_norm
        df['email_norm'] = pd.Series(dtype='object')
        
        for email_col in email_cols:
            mask = df['email_norm'].isna() & df[email_col].notna()
            if mask.any():
                df.loc[mask, 'email_norm'] = df.loc[mask, email_col].apply(normalize_email)
        
        non_null_emails = df['email_norm'].notna().sum()
        logger.debug(f"Emails normalizados: {non_null_emails}")
        
        return df