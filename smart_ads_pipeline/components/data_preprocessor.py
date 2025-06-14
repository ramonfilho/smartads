# smart_ads_pipeline/components/data_preprocessor.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import sys

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smart_ads_pipeline.core import BaseComponent, ExtendedParameterManager
from src.utils.feature_naming import standardize_feature_name

logger = logging.getLogger(__name__)


class DataPreprocessor(BaseComponent):
    """
    Componente responsável pelo pré-processamento dos dados.
    
    Responsabilidades:
    - Consolidar colunas de qualidade
    - Tratar valores ausentes
    - Tratar outliers
    - Normalizar valores numéricos
    - Converter tipos de dados
    """
    
    def __init__(self):
        super().__init__(name="data_preprocessor")
        
        # Parâmetros do componente
        self.columns_to_drop = []
        self.columns_to_convert = {}
        self.fillna_values = {}
        self.normalize_params = {}
        self.inference_columns = []  # Será carregado dos parâmetros
        
        # Colunas permitidas para inferência (do pipeline original)
        self.INFERENCE_COLUMNS = [
            'cual_es_tu_genero', 'cual_es_tu_edad', 'cual_es_tu_pais',
            'hace_quanto_tiempo_me_conoces', 'cual_es_tu_disponibilidad_de_tiempo_para_estudiar_ingles',
            'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida_que_oportunidades_se_abriran_para_ti',
            'cual_es_tu_sueldo_anual_en_dolares', 'cuanto_te_gustaria_ganar_al_ano',
            'crees_que_aprender_ingles_te_acercaria_mas_al_salario_que_mencionaste_anteriormente',
            'crees_que_aprender_ingles_puede_ayudarte_en_el_trabajo_o_en_tu_vida_diaria',
            'que_esperas_aprender_en_el_evento_cero_a_ingles_fluido', 'dejame_un_mensaje',
            'qualidade_numero', 'utm_campaing', 'utm_source', 'utm_medium',
            'utm_content', 'utm_term'
        ]
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """
        Aprende parâmetros de pré-processamento.
        
        Args:
            X: DataFrame de treino
            y: Target (não usado)
            
        Returns:
            self
        """
        self._validate_input(X)
        logger.info(f"{self.name}: Iniciando fit com shape {X.shape}")
        
        # Fazer cópia para não modificar original
        X_work = X.copy()
        
        # 1. Consolidar colunas de qualidade
        X_work, quality_params = self._consolidate_quality_columns_fit(X_work)
        self.quality_params = quality_params
        
        # 2. Tratar missing values
        X_work, missing_params = self._handle_missing_values_fit(X_work)
        self.missing_params = missing_params
        
        # 3. Tratar outliers
        X_work, outlier_params = self._handle_outliers_fit(X_work)
        self.outlier_params = outlier_params
        
        # 4. Normalização
        X_work, norm_params = self._normalize_values_fit(X_work)
        self.normalization_params = norm_params
        
        self.is_fitted = True
        logger.info(f"{self.name}: Fit concluído")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma dados usando parâmetros aprendidos.
        
        Args:
            X: DataFrame para transformar
            
        Returns:
            DataFrame transformado
        """
        self._check_is_fitted()
        self._validate_input(X)
        logger.info(f"{self.name}: Iniciando transform com shape {X.shape}")
        
        # Fazer cópia
        X_transformed = X.copy()
        
        # 1. Consolidar colunas de qualidade
        X_transformed = self._consolidate_quality_columns_transform(
            X_transformed, self.quality_params
        )
        
        # 2. Converter tipos de dados (antes de outros processamentos)
        X_transformed = self._convert_data_types(X_transformed)
        
        # 3. Tratar missing values
        X_transformed = self._handle_missing_values_transform(
            X_transformed, self.missing_params
        )
        
        # 4. Tratar outliers
        X_transformed = self._handle_outliers_transform(
            X_transformed, self.outlier_params
        )
        
        # 5. Normalização
        X_transformed = self._normalize_values_transform(
            X_transformed, self.normalization_params
        )
        
        logger.info(f"{self.name}: Transform concluído com shape {X_transformed.shape}")
        
        return X_transformed
    
    def _consolidate_quality_columns_fit(self, df: pd.DataFrame) -> tuple:
        """Consolida colunas de qualidade durante fit."""
        params = {}
        df_result = df.copy()
        
        # Variantes conhecidas das colunas de qualidade
        quality_variants = [
            'Qualidade (Nome)', 'Qualidade (Número)', 'Qualidade (Numero)', 
            'Qualidade (nome)', 'Qualidade (número)', 'Qualidade (Nombre)', 
            'Qualidade', 'Qualidade (Número) ', 'Qualidade (Nome) '
        ]
        
        # Nomes padronizados
        numeric_col_name = standardize_feature_name('qualidade_numerica')
        text_col_name = standardize_feature_name('qualidade_textual')
        
        # Verificar quais existem
        existing_variants = [col for col in quality_variants if col in df.columns]
        
        if not existing_variants:
            # Criar colunas vazias se não existem
            df_result[numeric_col_name] = np.nan
            df_result[text_col_name] = None
            params['found_variants'] = False
        else:
            # Separar numéricas e textuais
            numeric_cols = []
            text_cols = []
            
            for col in existing_variants:
                # Testar se é numérica
                try:
                    numeric_test = pd.to_numeric(df[col], errors='coerce')
                    not_null = df[col].notna().sum()
                    if not_null > 0 and numeric_test.notna().sum() / not_null >= 0.5:
                        numeric_cols.append(col)
                    else:
                        text_cols.append(col)
                except:
                    text_cols.append(col)
            
            params['numeric_cols'] = numeric_cols
            params['text_cols'] = text_cols
            params['found_variants'] = True
            
            # Consolidar
            if numeric_cols:
                df_result[numeric_col_name] = np.nan
                for col in sorted(numeric_cols, key=lambda x: df[x].notna().sum(), reverse=True):
                    mask = df_result[numeric_col_name].isna() & df[col].notna()
                    if mask.sum() > 0:
                        df_result.loc[mask, numeric_col_name] = pd.to_numeric(
                            df.loc[mask, col], errors='coerce'
                        )
            else:
                df_result[numeric_col_name] = np.nan
            
            if text_cols:
                df_result[text_col_name] = None
                for col in sorted(text_cols, key=lambda x: df[x].notna().sum(), reverse=True):
                    mask = df_result[text_col_name].isna() & df[col].notna()
                    if mask.sum() > 0:
                        df_result.loc[mask, text_col_name] = df.loc[mask, col]
            else:
                df_result[text_col_name] = None
            
            # Remover colunas originais
            df_result = df_result.drop(columns=existing_variants)
        
        params['numeric_col_name'] = numeric_col_name
        params['text_col_name'] = text_col_name
        
        logger.debug(f"Colunas de qualidade consolidadas: {params}")
        
        return df_result, params
    
    def _consolidate_quality_columns_transform(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Consolida colunas de qualidade durante transform."""
        df_result = df.copy()
        
        numeric_col_name = params['numeric_col_name']
        text_col_name = params['text_col_name']
        
        # Se as colunas já existem, não fazer nada
        if numeric_col_name in df_result.columns and text_col_name in df_result.columns:
            return df_result
        
        # Criar colunas se não existem
        if numeric_col_name not in df_result.columns:
            df_result[numeric_col_name] = np.nan
        if text_col_name not in df_result.columns:
            df_result[text_col_name] = None
        
        # Remover variantes se ainda existirem
        if params.get('found_variants', False):
            all_variants = params.get('numeric_cols', []) + params.get('text_cols', [])
            existing_to_remove = [col for col in all_variants if col in df_result.columns]
            if existing_to_remove:
                df_result = df_result.drop(columns=existing_to_remove)
        
        return df_result
    
    def _handle_missing_values_fit(self, df: pd.DataFrame) -> tuple:
        """Trata valores ausentes durante fit."""
        params = {}
        df_result = df.copy()
        
        # 1. Remover colunas com muitos missing (>95%)
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_pct[missing_pct > 95].index.tolist()
        params['high_missing_cols'] = high_missing_cols
        
        if high_missing_cols:
            df_result = df_result.drop(columns=high_missing_cols)
            logger.debug(f"Removidas {len(high_missing_cols)} colunas com >=95% missing")
        
        # 2. Preencher UTMs
        utm_cols = [col for col in df_result.columns if col.startswith('utm_') or 'utm' in col.lower()]
        for col in utm_cols:
            df_result[col] = df_result[col].fillna('unknown')
        
        # 3. Preencher categóricas conhecidas
        categorical_fills = {
            'cual_es_tu_genero': 'desconhecido',
            'cual_es_tu_edad': 'desconhecido',
            'cual_es_tu_pais': 'desconhecido',
            'hace_quanto_tiempo_me_conoces': 'desconhecido',
            'cual_es_tu_disponibilidad_de_tiempo_para_estudiar_ingles': 'desconhecido',
            'cual_es_tu_sueldo_anual_en_dolares': 'desconhecido',
            'cuanto_te_gustaria_ganar_al_ano': 'desconhecido',
            'crees_que_aprender_ingles_te_acercaria_mas_al_salario_que_mencionaste_anteriormente': 'desconhecido',
            'crees_que_aprender_ingles_puede_ayudarte_en_el_trabajo_o_en_tu_vida_diaria': 'desconhecido',
            'qualidade': 'desconhecido',
            'lancamento': 'desconhecido'
        }
        
        for col, fill_value in categorical_fills.items():
            if col in df_result.columns:
                df_result[col] = df_result[col].fillna(fill_value)
        
        # 4. Preencher textos com string vazia
        text_patterns = [
            'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida',
            'que_esperas_aprender_en_la_semana_de_cero_a_ingles_fluido',
            'que_esperas_aprender_en_el_evento_cero_a_ingles_fluido',
            'dejame_un_mensaje',
            'que_esperas_aprender_en_la_inmersion_desbloquea_tu_ingles_en_72_horas'
        ]
        for pattern in text_patterns:
            for col in df_result.columns:
                if pattern in col.lower():
                    df_result[col] = df_result[col].fillna('')
        
        # 5. Calcular medianas para numéricas
        numeric_cols = df_result.select_dtypes(include=['number']).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        for col in numeric_cols:
            if df_result[col].isna().sum() > 0:
                median_value = df_result[col].median()
                params[f'median_{col}'] = median_value
                df_result[col] = df_result[col].fillna(median_value)
        
        return df_result, params
    
    def _handle_missing_values_transform(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Trata valores ausentes durante transform."""
        df_result = df.copy()
        
        # 1. Remover colunas com alto missing (se existirem)
        high_missing_cols = params.get('high_missing_cols', [])
        cols_to_drop = [col for col in high_missing_cols if col in df_result.columns]
        if cols_to_drop:
            df_result = df_result.drop(columns=cols_to_drop)
        
        # 2. Aplicar mesmos preenchimentos
        # UTMs
        utm_cols = [col for col in df_result.columns if col.startswith('utm_') or 'utm' in col.lower()]
        for col in utm_cols:
            df_result[col] = df_result[col].fillna('unknown')
        
        # Categóricas
        categorical_fills = {
            'cual_es_tu_genero': 'desconhecido',
            'cual_es_tu_edad': 'desconhecido',
            'cual_es_tu_pais': 'desconhecido',
            'hace_quanto_tiempo_me_conoces': 'desconhecido',
            'cual_es_tu_disponibilidad_de_tiempo_para_estudiar_ingles': 'desconhecido',
            'cual_es_tu_sueldo_anual_en_dolares': 'desconhecido',
            'cuanto_te_gustaria_ganar_al_ano': 'desconhecido',
            'lancamento': 'desconhecido'
        }
        
        for col, fill_value in categorical_fills.items():
            if col in df_result.columns:
                df_result[col] = df_result[col].fillna(fill_value)
        
        # Textos
        text_patterns = ['cuando_hables_ingles', 'que_esperas_aprender', 'dejame_un_mensaje']
        for col in df_result.columns:
            if any(pattern in col.lower() for pattern in text_patterns):
                df_result[col] = df_result[col].fillna('')
        
        # Numéricas com medianas salvas
        for col in df_result.select_dtypes(include=['number']).columns:
            if col != 'target' and df_result[col].isna().sum() > 0:
                median_key = f'median_{col}'
                if median_key in params:
                    df_result[col] = df_result[col].fillna(params[median_key])
                else:
                    # Fallback: usar mediana atual
                    df_result[col] = df_result[col].fillna(df_result[col].median())
        
        return df_result
    
    def _handle_outliers_fit(self, df: pd.DataFrame) -> tuple:
        """Trata outliers durante fit."""
        params = {}
        df_result = df.copy()
        
        numeric_cols = df_result.select_dtypes(include=['number']).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        # Identificar colunas de qualidade numéricas
        quality_numeric_cols = [col for col in numeric_cols 
                               if 'qualidade' in col.lower() or 'qualidad' in col.lower()]
        
        for col in numeric_cols:
            if df_result[col].nunique() > 10:  # Só tratar se tem variabilidade
                if col in quality_numeric_cols:
                    # Tratamento especial para qualidade
                    lower_bound = df_result[col].quantile(0.01)
                    upper_bound = df_result[col].quantile(0.99)
                else:
                    # Tratamento padrão com IQR
                    q1 = df_result[col].quantile(0.25)
                    q3 = df_result[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                
                params[f'{col}_bounds'] = (float(lower_bound), float(upper_bound))
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_result, params
    
    def _handle_outliers_transform(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Trata outliers durante transform."""
        df_result = df.copy()
        
        for col in df_result.select_dtypes(include=['number']).columns:
            if col != 'target' and f'{col}_bounds' in params:
                lower_bound, upper_bound = params[f'{col}_bounds']
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_result
    
    def _normalize_values_fit(self, df: pd.DataFrame) -> tuple:
        """Normaliza valores durante fit."""
        params = {}
        df_result = df.copy()
        
        # Selecionar colunas numéricas com variabilidade
        numeric_cols = df_result.select_dtypes(include=['number']).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        cols_with_std = []
        for col in numeric_cols:
            if df_result[col].std() > 0:
                cols_with_std.append(col)
        
        params['cols_with_std'] = cols_with_std
        params['means'] = {}
        params['stds'] = {}
        
        # Normalizar
        for col in cols_with_std:
            mean = df_result[col].mean()
            std = df_result[col].std()
            params['means'][col] = float(mean)
            params['stds'][col] = float(std)
            df_result[col] = (df_result[col] - mean) / std
        
        return df_result, params
    
    def _normalize_values_transform(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Normaliza valores durante transform."""
        df_result = df.copy()
        
        cols_with_std = params.get('cols_with_std', [])
        means = params.get('means', {})
        stds = params.get('stds', {})
        
        for col in cols_with_std:
            if col in df_result.columns and col in means and col in stds:
                mean = means[col]
                std = stds[col]
                df_result[col] = (df_result[col] - mean) / std
        
        return df_result
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte tipos de dados."""
        df_result = df.copy()
        
        # Converter colunas temporais
        temporal_cols = ['marca_temporal', 'data']
        for col in temporal_cols:
            if col in df_result.columns:
                df_result[col] = pd.to_datetime(df_result[col], errors='coerce')
        
        return df_result
    
    def _save_component_params(self, param_manager: ExtendedParameterManager) -> None:
        """Salva parâmetros do componente."""
        params = {
            'quality_params': self.quality_params,
            'missing_params': self.missing_params,
            'outlier_params': self.outlier_params,
            'normalization_params': self.normalization_params
        }
        param_manager.save_component_params(self.name, params)
    
    def _load_component_params(self, param_manager: ExtendedParameterManager) -> None:
        """Carrega parâmetros do componente."""
        params = param_manager.get_component_params(self.name)
        self.quality_params = params.get('quality_params', {})
        self.missing_params = params.get('missing_params', {})
        self.outlier_params = params.get('outlier_params', {})
        self.normalization_params = params.get('normalization_params', {})