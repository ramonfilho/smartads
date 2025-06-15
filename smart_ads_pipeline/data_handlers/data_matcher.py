# smart_ads_pipeline/data_handlers/data_matcher.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

import numpy as np
from src.utils.feature_naming import standardize_feature_name
from smart_ads_pipeline.core.columns_config import INFERENCE_COLUMNS, TEMPORARY_COLUMNS

class DataMatcher:
    """
    Responsável por fazer o matching entre surveys e buyers e criar a variável target.
    """
    
    def __init__(self):
        """Inicializa o DataMatcher."""
        self.match_stats = {}
        logger.info("DataMatcher inicializado")
        
        # Usar a definição importada de columns_config
        self.INFERENCE_COLUMNS = INFERENCE_COLUMNS
    
    def match_and_create_target(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Executa o processo completo de matching e criação de target.
        
        Args:
            data_dict: Dicionário com DataFrames:
                - 'surveys': DataFrame de pesquisas
                - 'buyers': DataFrame de compradores
                - 'utms': DataFrame de UTMs
                
        Returns:
            DataFrame consolidado com variável target
        """
        logger.info("Iniciando processo de matching e criação de target...")
        
        surveys = data_dict.get('surveys', pd.DataFrame())
        buyers = data_dict.get('buyers', pd.DataFrame())
        utms = data_dict.get('utms', pd.DataFrame())
        
        # Validar dados
        if surveys.empty:
            raise ValueError("DataFrame de surveys está vazio")
        
        # 1. Fazer matching
        matches_df = self._match_surveys_with_buyers(surveys, buyers)
        
        # 2. Criar variável target
        surveys_with_target = self._create_target_variable(surveys, matches_df)
        
        # 3. Merge com UTMs
        final_df = self._merge_with_utms(surveys_with_target, utms)
        
        # 4. Preparar dataset final
        final_df = self._prepare_final_dataset(final_df)
        
        # Log estatísticas
        self._log_statistics(final_df)
        
        return final_df
    
    def _match_surveys_with_buyers(self, surveys: pd.DataFrame, 
                                  buyers: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza correspondência entre pesquisas e compradores.
        
        Args:
            surveys: DataFrame de pesquisas
            buyers: DataFrame de compradores
            
        Returns:
            DataFrame com matches encontrados
        """
        logger.info("Realizando matching entre surveys e buyers...")
        start_time = time.time()
        
        # Verificar se podemos fazer matching
        if (surveys.empty or buyers.empty or 
            'email_norm' not in surveys.columns or 
            'email_norm' not in buyers.columns):
            logger.warning("Não é possível fazer matching - dados insuficientes")
            return pd.DataFrame(columns=['buyer_id', 'survey_id', 'match_type', 'score'])
        
        # Criar dicionário de emails das pesquisas
        survey_emails_dict = defaultdict(list)
        for idx, row in surveys.iterrows():
            email_norm = row.get('email_norm')
            if pd.notna(email_norm):
                survey_emails_dict[email_norm].append(idx)
        
        survey_emails_set = set(survey_emails_dict.keys())
        
        # Fazer matching
        matches = []
        match_count = 0
        unique_surveys_matched = set()
        
        for idx, buyer in buyers.iterrows():
            buyer_email = buyer.get('email_norm')
            if pd.isna(buyer_email) or buyer_email not in survey_emails_set:
                continue
            
            # Criar match para cada survey com esse email
            survey_indices = survey_emails_dict[buyer_email]
            for survey_idx in survey_indices:
                match_data = {
                    'buyer_id': idx,
                    'survey_id': survey_idx,
                    'match_type': 'exact',
                    'score': 1.0
                }
                
                # Adicionar informação de lançamento se disponível
                if 'lancamento' in buyer and not pd.isna(buyer['lancamento']):
                    match_data['lancamento'] = buyer['lancamento']
                
                matches.append(match_data)
                unique_surveys_matched.add(survey_idx)
                match_count += 1
        
        # Criar DataFrame de matches
        matches_df = pd.DataFrame(matches)
        
        # Estatísticas
        elapsed_time = time.time() - start_time
        unique_buyers_matched = len(set(m['buyer_id'] for m in matches)) if matches else 0
        
        self.match_stats = {
            'total_matches': match_count,
            'unique_buyers_matched': unique_buyers_matched,
            'unique_surveys_matched': len(unique_surveys_matched),
            'total_buyers': len(buyers),
            'total_surveys': len(surveys),
            'match_time_seconds': elapsed_time
        }
        
        logger.info(f"Matching concluído em {elapsed_time:.2f}s")
        logger.info(f"  - Total de matches: {match_count}")
        logger.info(f"  - Buyers únicos matched: {unique_buyers_matched}/{len(buyers)} "
                   f"({unique_buyers_matched/len(buyers)*100:.1f}%)")
        logger.info(f"  - Surveys únicos matched: {len(unique_surveys_matched)}/{len(surveys)} "
                   f"({len(unique_surveys_matched)/len(surveys)*100:.1f}%)")
        
        return matches_df
    
    def _create_target_variable(self, surveys_df: pd.DataFrame, 
                               matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria a variável target baseada nos matches.
        
        Args:
            surveys_df: DataFrame de pesquisas
            matches_df: DataFrame com matches
            
        Returns:
            DataFrame com variável target adicionada
        """
        logger.info("Criando variável target...")
        
        if surveys_df.empty:
            return pd.DataFrame(columns=['target'])
        
        # Copiar DataFrame
        result_df = surveys_df.copy()
        
        # Inicializar target com 0
        result_df['target'] = 0
        
        # Se não há matches, todos são negativos
        if matches_df.empty:
            logger.warning("Nenhum match encontrado - target será todo 0")
            return result_df
        
        # Marcar positivos baseado nos matches
        matched_count = 0
        for _, match in matches_df.iterrows():
            survey_id = match['survey_id']
            if survey_id in result_df.index:
                result_df.loc[survey_id, 'target'] = 1
                matched_count += 1
        
        positive_count = result_df['target'].sum()
        positive_rate = (positive_count / len(result_df) * 100) if len(result_df) > 0 else 0
        
        logger.info(f"Target criado: {positive_count} positivos de {len(result_df)} "
                   f"({positive_rate:.2f}%)")
        
        return result_df
    
    def _merge_with_utms(self, surveys_df: pd.DataFrame, 
                        utm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Faz merge com dados UTM.
        
        Args:
            surveys_df: DataFrame de pesquisas com target
            utm_df: DataFrame de UTMs
            
        Returns:
            DataFrame merged
        """
        logger.info("Fazendo merge com dados UTM...")
        
        if utm_df.empty or 'email_norm' not in utm_df.columns:
            logger.warning("Dados UTM não disponíveis ou sem email_norm")
            return surveys_df
        
        # Remover duplicatas de UTMs
        logger.info(f"UTMs antes de deduplicação: {len(utm_df)}")
        utm_dedup = utm_df.drop_duplicates(subset=['email_norm'])
        logger.info(f"UTMs após deduplicação: {len(utm_dedup)}")
        
        # Fazer merge
        merged_df = pd.merge(
            surveys_df,
            utm_dedup,
            on='email_norm',
            how='left',
            suffixes=('', '_utm')
        )
        
        logger.info(f"Merge concluído: {merged_df.shape[0]} linhas, {merged_df.shape[1]} colunas")
        
        # Consolidar colunas de email se necessário
        if 'e_mail_utm' in merged_df.columns and 'e_mail' in merged_df.columns:
            mask = merged_df['e_mail'].isna() & merged_df['e_mail_utm'].notna()
            merged_df.loc[mask, 'e_mail'] = merged_df.loc[mask, 'e_mail_utm']
        
        return merged_df
    
    def _prepare_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara o dataset final removendo colunas temporárias e 
        garantindo compatibilidade com produção.
        
        Args:
            df: DataFrame para preparar
            
        Returns:
            DataFrame preparado e filtrado para produção
        """
        logger.info("Preparando dataset final...")
        logger.info(f"   Dataset original: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Guardar colunas originais para relatório
        original_columns = set(df.columns)
        
        # Passo 1: Remover colunas temporárias específicas
        temp_columns = ['email_norm', 'email']
        for col in temp_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.debug(f"Coluna temporária '{col}' removida")
        
        # Passo 2: Identificar colunas válidas para produção
        # Incluir apenas colunas que estão em INFERENCE_COLUMNS ou é target
        valid_columns = []
        for col in df.columns:
            if col in self.INFERENCE_COLUMNS or col == 'target':
                valid_columns.append(col)
        
        # Aplicar filtro mantendo apenas colunas válidas
        df_filtered = df[valid_columns].copy()
        
        # Passo 3: Adicionar colunas faltantes de INFERENCE_COLUMNS com NaN
        missing_columns = []
        for col in self.INFERENCE_COLUMNS:
            if col not in df_filtered.columns:
                df_filtered[col] = np.nan
                missing_columns.append(col)
        
        # Passo 4: Reordenar colunas conforme INFERENCE_COLUMNS
        # Manter ordem consistente: primeiro INFERENCE_COLUMNS, depois target se existir
        column_order = [col for col in self.INFERENCE_COLUMNS if col in df_filtered.columns]
        if 'target' in df_filtered.columns:
            column_order.append('target')
        
        df_filtered = df_filtered[column_order]
        
        # Relatório de colunas removidas
        final_columns = set(df_filtered.columns)
        removed_columns = original_columns - final_columns
        
        logger.info(f"\n   COLUMNS REMOVED FOR PRODUCTION COMPATIBILITY:")
        logger.info(f"   Total columns removed: {len(removed_columns)}")
        if removed_columns:
            logger.info(f"   Removed columns list:")
            # Ordenar para consistência no log
            for col in sorted(removed_columns):
                logger.info(f"      - {col}")
        else:
            logger.info(f"   No additional columns removed")
        
        if missing_columns:
            logger.info(f"\n   Added {len(missing_columns)} missing INFERENCE_COLUMNS with NaN values")
            if len(missing_columns) <= 10:
                logger.info(f"   Missing columns: {missing_columns}")
            else:
                logger.info(f"   Missing columns: {missing_columns[:5]} ... and {len(missing_columns)-5} more")
        
        logger.info(f"\n   Dataset final: {df_filtered.shape[0]} linhas, {df_filtered.shape[1]} colunas")
        
        # Salvar estatísticas para relatório posterior
        self.match_stats['removed_columns'] = list(removed_columns)
        self.match_stats['missing_columns_added'] = missing_columns
        self.match_stats['final_shape'] = df_filtered.shape
        self.match_stats['final_columns'] = list(df_filtered.columns)
        
        return df_filtered
    
    def validate_production_compatibility(self, df: pd.DataFrame) -> tuple[bool, dict]:
        """
        Valida se o DataFrame é compatível com produção.
        
        Args:
            df: DataFrame para validar
            
        Returns:
            Tupla (is_compatible, validation_report)
        """
        validation_report = {
            'is_compatible': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Colunas esperadas em produção (sem target)
        production_columns = self.INFERENCE_COLUMNS
        
        # Verificar colunas obrigatórias
        df_columns = set(df.columns)
        missing_required = set(production_columns) - df_columns
        extra_columns = df_columns - set(production_columns) - {'target'}
        
        if missing_required:
            validation_report['is_compatible'] = False
            validation_report['errors'].append(f"Missing required columns: {list(missing_required)}")
        
        if extra_columns:
            validation_report['warnings'].append(f"Extra columns found: {list(extra_columns)}")
        
        # Verificar ordem das colunas
        expected_order = [col for col in production_columns if col in df.columns]
        actual_order = [col for col in df.columns if col in production_columns]
        
        if expected_order != actual_order:
            validation_report['warnings'].append("Column order differs from expected")
        
        # Verificar colunas críticas de dados
        critical_columns = ['cual_es_tu_e_mail', 'e_mail', 'data', 'marca_temporal']
        for col in critical_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                total_count = len(df)
                coverage_pct = (non_null_count / total_count * 100) if total_count > 0 else 0
                
                validation_report['info'].append(f"{col}: {coverage_pct:.1f}% coverage ({non_null_count}/{total_count})")
                
                if coverage_pct < 10:
                    validation_report['warnings'].append(f"{col} has very low coverage: {coverage_pct:.1f}%")
        
        # Verificar target se presente
        if 'target' in df.columns:
            unique_values = df['target'].dropna().unique()
            valid_targets = set(unique_values).issubset({0, 1, 0.0, 1.0})
            
            if not valid_targets:
                validation_report['errors'].append(f"Invalid target values found: {list(unique_values)}")
                validation_report['is_compatible'] = False
            else:
                positive_rate = (df['target'] == 1).sum() / len(df) * 100 if len(df) > 0 else 0
                validation_report['info'].append(f"Target positive rate: {positive_rate:.2f}%")
        
        return validation_report['is_compatible'], validation_report
    
    def _log_statistics(self, df: pd.DataFrame) -> None:
        """
        Loga estatísticas do dataset final.
        
        Args:
            df: DataFrame final
        """
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            positive_rate = (target_counts.get(1, 0) / len(df) * 100) if len(df) > 0 else 0
            
            logger.info("\nEstatísticas do dataset final:")
            logger.info(f"  Total de registros: {len(df):,}")
            logger.info(f"  Negativos (target=0): {target_counts.get(0, 0):,} ({100-positive_rate:.2f}%)")
            logger.info(f"  Positivos (target=1): {target_counts.get(1, 0):,} ({positive_rate:.2f}%)")
            
            if hasattr(self, 'match_stats') and self.match_stats:
                logger.info("\nEstatísticas de matching:")
                for key, value in self.match_stats.items():
                    logger.info(f"  {key}: {value}")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Retorna estatísticas do processo de matching.
        
        Returns:
            Dicionário com estatísticas
        """
        return self.match_stats.copy()