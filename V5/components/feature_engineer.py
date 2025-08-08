# smart_ads_pipeline/components/feature_engineer.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import sys
from sklearn.preprocessing import LabelEncoder

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smart_ads_pipeline.core import BaseComponent, ExtendedParameterManager
from src.utils.feature_naming import standardize_feature_name

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseComponent):
    """
    Componente responsável pela engenharia de features básicas.
    
    Responsabilidades:
    - Criar features de identidade
    - Criar features temporais
    - Encodar variáveis categóricas
    - Criar features de interação
    
    Este componente NÃO inclui:
    - Processamento de texto (TextProcessor)
    - Features profissionais/motivacionais (ProfessionalFeatures)
    """
    
    def __init__(self):
        super().__init__(name="feature_engineer")
        
        # Parâmetros aprendidos durante fit
        self.categorical_params = {}
        self.columns_to_remove = []
        self.preserved_columns = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Aprende parâmetros para feature engineering.
        
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
        
        # 1. Criar features de identidade (não precisa fit)
        X_work = self._create_identity_features(X_work)
        
        # 2. Criar features temporais (não precisa fit)
        X_work = self._create_temporal_features(X_work)
        
        # 3. Encodar categóricas (precisa fit)
        X_work, categorical_params = self._encode_categorical_features_fit(X_work)
        self.categorical_params = categorical_params
        
        # 4. Definir colunas a remover (mas preservar para processamento posterior)
        self.columns_to_remove = [
            'como_te_llamas',
            'cual_es_tu_instagram', 
            'cual_es_tu_profesion',
            'cual_es_tu_telefono'
        ]
        
        # Preservar colunas para processamento de texto posterior
        for col in self.columns_to_remove:
            if col in X_work.columns:
                self.preserved_columns[col] = X_work[col].copy()
        
        self.is_fitted = True
        logger.info(f"{self.name}: Fit concluído")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma dados criando features.
        
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
        
        # 1. Criar features de identidade
        X_transformed = self._create_identity_features(X_transformed)
        
        # 2. Criar features temporais
        X_transformed = self._create_temporal_features(X_transformed)
        
        # 3. Encodar categóricas
        X_transformed = self._encode_categorical_features_transform(
            X_transformed, self.categorical_params
        )
        
        # 4. Remover colunas originais
        cols_to_drop = [col for col in self.columns_to_remove if col in X_transformed.columns]
        if cols_to_drop:
            X_transformed = X_transformed.drop(columns=cols_to_drop)
            logger.debug(f"Removidas {len(cols_to_drop)} colunas originais")
        
        logger.info(f"{self.name}: Transform concluído com shape {X_transformed.shape}")
        
        return X_transformed
    
    def _create_identity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features baseadas nos campos de identidade do usuário."""
        df_result = df.copy()
        
        # Feature de comprimento do nome
        nome_col = standardize_feature_name('¿Cómo te llamas?')
        if nome_col in df_result.columns:
            df_result[standardize_feature_name('name_length')] = df_result[nome_col].str.len()
            df_result[standardize_feature_name('name_word_count')] = df_result[nome_col].str.split().str.len()
        
        # Feature de validade do telefone
        tel_col = standardize_feature_name('¿Cual es tu telefono?')
        if tel_col in df_result.columns:
            df_result[standardize_feature_name('valid_phone')] = (
                df_result[tel_col].str.replace(r'\D', '', regex=True).str.len() >= 8
            ).astype(int)
        
        # Feature de presença de instagram
        insta_col = standardize_feature_name('¿Cuál es tu instagram?')
        if insta_col in df_result.columns:
            df_result[standardize_feature_name('has_instagram')] = (
                df_result[insta_col].notna() & (df_result[insta_col] != '')
            ).astype(int)
        
        return df_result
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features baseadas em informações temporais."""
        df_result = df.copy()
        
        # Procurar coluna marca temporal
        marca_temporal_col = None
        for possible_name in ['marca_temporal', 'Marca temporal', 'Marca_temporal']:
            if possible_name in df_result.columns:
                marca_temporal_col = possible_name
                break
        
        if marca_temporal_col and pd.api.types.is_datetime64_any_dtype(df_result[marca_temporal_col]):
            # Extrair componentes básicos
            df_result[standardize_feature_name('hour')] = df_result[marca_temporal_col].dt.hour
            df_result[standardize_feature_name('day_of_week')] = df_result[marca_temporal_col].dt.dayofweek
            df_result[standardize_feature_name('month')] = df_result[marca_temporal_col].dt.month
            df_result[standardize_feature_name('year')] = df_result[marca_temporal_col].dt.year
            
            # Features cíclicas
            hour_col = standardize_feature_name('hour')
            day_col = standardize_feature_name('day_of_week')
            
            df_result[standardize_feature_name('hour_sin')] = np.sin(2 * np.pi * df_result[hour_col] / 24)
            df_result[standardize_feature_name('hour_cos')] = np.cos(2 * np.pi * df_result[hour_col] / 24)
            df_result[standardize_feature_name('day_sin')] = np.sin(2 * np.pi * df_result[day_col] / 7)
            df_result[standardize_feature_name('day_cos')] = np.cos(2 * np.pi * df_result[day_col] / 7)
            
            # Período do dia
            df_result[standardize_feature_name('period_of_day')] = pd.cut(
                df_result[hour_col], 
                bins=[0, 6, 12, 18, 24], 
                labels=['madrugada', 'manha', 'tarde', 'noite']
            )
        
        # Processar DATA se existir
        data_col = None
        for possible_name in ['data', 'DATA', 'Data']:
            if possible_name in df_result.columns:
                data_col = possible_name
                break
        
        if data_col and pd.api.types.is_datetime64_any_dtype(df_result[data_col]):
            df_result[standardize_feature_name('utm_hour')] = df_result[data_col].dt.hour
            df_result[standardize_feature_name('utm_day_of_week')] = df_result[data_col].dt.dayofweek
            df_result[standardize_feature_name('utm_month')] = df_result[data_col].dt.month
            df_result[standardize_feature_name('utm_year')] = df_result[data_col].dt.year
        
        return df_result
    
    def _encode_categorical_features_fit(self, df: pd.DataFrame) -> tuple:
        """Codifica variáveis categóricas durante fit."""
        params = {}
        df_result = df.copy()
        
        # 1. Mapas para variáveis ordinais
        age_map = {
            '18 años a 24 años': 1,
            '25 años a 34 años': 2,
            '35 años a 44 años': 3,
            '45 años a 54 años': 4,
            'Mas de 54': 5,
            'desconhecido': 0
        }
        
        time_map = {
            'Te acabo de conocer a través del anuncio.': 0,
            'Te sigo desde hace 1 mes': 1,
            'Te sigo desde hace 3 meses': 2,
            'Te sigo desde hace más de 5 meses': 3,
            'Te sigo desde hace 1 año': 4,
            'Te sigo hace más de 1 año': 5,
            'Te sigo hace más de 2 años': 6,
            'desconhecido': -1
        }
        
        availability_map = {
            'Menos de 1 hora al día': 0,
            '1 hora al día': 1,
            '2 horas al día': 2,
            '3 horas al día': 3,
            'Más de 3 horas al día': 4,
            'desconhecido': -1
        }
        
        salary_map = {
            'Menos de US$3000': 1,
            'US$3000 a US$5000': 2,
            'US$5000 o más': 3,
            'US$10000 o más': 4,
            'US$20000 o más': 5,
            'desconhecido': 0
        }
        
        desired_salary_map = {
            'Al menos US$ 3000 por año': 1,
            'Más de US$5000 por año': 2,
            'Más de US$10000 por año': 3,
            'Más de US$20000 por año': 4,
            'desconhecido': 0
        }
        
        belief_map = {
            'Creo que no...': 0,
            'Tal vez': 1,
            '¡Sí, sin duda!': 2,
            '¡Si por su puesto!': 2,
            'desconhecido': -1
        }
        
        gender_map = {
            'Mujer': 1, 
            'Hombre': 0, 
            'desconhecido': -1
        }
        
        # Guardar mapas
        params['age_map'] = age_map
        params['time_map'] = time_map
        params['availability_map'] = availability_map
        params['salary_map'] = salary_map
        params['desired_salary_map'] = desired_salary_map
        params['belief_map'] = belief_map
        params['gender_map'] = gender_map
        
        # 2. Aplicar mapas
        
        # Idade
        age_col = standardize_feature_name('¿Cuál es tu edad?')
        if age_col in df_result.columns:
            df_result[standardize_feature_name('age_encoded')] = df_result[age_col].map(age_map).fillna(0)
        
        # Tempo conhecido
        time_col = standardize_feature_name('¿Hace quánto tiempo me conoces?')
        if time_col in df_result.columns:
            df_result[standardize_feature_name('time_known_encoded')] = df_result[time_col].map(time_map).fillna(-1)
        
        # Disponibilidade
        avail_col = standardize_feature_name('¿Cuál es tu disponibilidad de tiempo para estudiar inglés?')
        if avail_col in df_result.columns:
            df_result[standardize_feature_name('availability_encoded')] = df_result[avail_col].map(availability_map).fillna(-1)
        
        # Salário atual
        salary_col = standardize_feature_name('¿Cuál es tu sueldo anual? (en dólares)')
        if salary_col in df_result.columns:
            df_result[standardize_feature_name('current_salary_encoded')] = df_result[salary_col].map(salary_map).fillna(0)
        
        # Salário desejado
        desired_col = standardize_feature_name('¿Cuánto te gustaría ganar al año?')
        if desired_col in df_result.columns:
            df_result[standardize_feature_name('desired_salary_encoded')] = df_result[desired_col].map(desired_salary_map).fillna(0)
        
        # Crenças
        belief_salary_col = standardize_feature_name('¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?')
        if belief_salary_col in df_result.columns:
            df_result[standardize_feature_name('belief_salary_encoded')] = df_result[belief_salary_col].map(belief_map).fillna(-1)
        
        belief_work_col = standardize_feature_name('¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?')
        if belief_work_col in df_result.columns:
            df_result[standardize_feature_name('belief_work_encoded')] = df_result[belief_work_col].map(belief_map).fillna(-1)
        
        # Gênero
        gender_col = standardize_feature_name('¿Cuál es tu género?')
        if gender_col in df_result.columns:
            df_result[standardize_feature_name('gender_encoded')] = df_result[gender_col].map(gender_map).fillna(-1)
        
        # 3. Encoding para variáveis de alta cardinalidade
        
        # País
        country_col = standardize_feature_name('¿Cual es tu país?')
        if country_col not in df_result.columns:
            country_col = standardize_feature_name('¿Cuál es tu país?')
        
        if country_col in df_result.columns:
            # Frequency Encoding
            freq_map = df_result[country_col].value_counts(normalize=True).to_dict()
            params[f'{country_col}_freq_map'] = freq_map
            df_result[standardize_feature_name('country_freq')] = df_result[country_col].map(freq_map).fillna(0)
            
            # Agrupar categorias raras
            threshold = 0.01
            rare_categories = [cat for cat, freq in freq_map.items() if freq < threshold]
            params[f'{country_col}_rare_categories'] = rare_categories
            
            # Label Encoding para categoria agrupada
            grouped_col = country_col + '_grouped'
            df_result[grouped_col] = df_result[country_col].apply(
                lambda x: 'Rare' if x in rare_categories else x
            )
            
            le = LabelEncoder()
            df_result[grouped_col] = df_result[grouped_col].fillna('Unknown')
            df_result[standardize_feature_name('country_encoded')] = le.fit_transform(df_result[grouped_col])
            params[f'{country_col}_label_mapping'] = dict(zip(le.classes_, range(len(le.classes_))))
            
            # Remover coluna temporária
            df_result = df_result.drop(columns=[grouped_col])
        
        # 4. Tratamento de UTMs
        utm_cols = [col for col in df_result.columns if 'utm_' in col.lower()]
        
        for col in utm_cols:
            if col in df_result.columns:
                cardinality = df_result[col].nunique()
                
                if cardinality <= 10:  # Baixa cardinalidade
                    le = LabelEncoder()
                    # Garantir que valores nulos sejam tratados
                    values = df_result[col].fillna('unknown').astype(str)
                    df_result[standardize_feature_name(f'{col}_encoded')] = le.fit_transform(values)
                    params[f'{col}_label_mapping'] = dict(zip(le.classes_, range(len(le.classes_))))
                else:  # Alta cardinalidade
                    freq_map = df_result[col].value_counts(normalize=True).to_dict()
                    params[f'{col}_freq_map'] = freq_map
                    df_result[standardize_feature_name(f'{col}_freq')] = df_result[col].map(freq_map).fillna(0)
        
        # 5. GCLID como indicador binário
        gclid_cols = [col for col in df_result.columns if 'gclid' in col.lower()]
        if gclid_cols:
            gclid_col = gclid_cols[0]
            df_result[standardize_feature_name('has_gclid')] = df_result[gclid_col].notna().astype(int)
        
        return df_result, params
    
    def _encode_categorical_features_transform(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Codifica variáveis categóricas durante transform."""
        df_result = df.copy()
        
        # 1. Aplicar mapas ordinais
        mappings = [
            ('¿Cuál es tu edad?', 'age_encoded', 'age_map', 0),
            ('¿Hace quánto tiempo me conoces?', 'time_known_encoded', 'time_map', -1),
            ('¿Cuál es tu disponibilidad de tiempo para estudiar inglés?', 'availability_encoded', 'availability_map', -1),
            ('¿Cuál es tu sueldo anual? (en dólares)', 'current_salary_encoded', 'salary_map', 0),
            ('¿Cuánto te gustaría ganar al año?', 'desired_salary_encoded', 'desired_salary_map', 0),
            ('¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?', 'belief_salary_encoded', 'belief_map', -1),
            ('¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?', 'belief_work_encoded', 'belief_map', -1),
            ('¿Cuál es tu género?', 'gender_encoded', 'gender_map', -1),
        ]
        
        for orig_col, new_col, map_name, default in mappings:
            col = standardize_feature_name(orig_col)
            if col in df_result.columns and map_name in params:
                mapping = params[map_name]
                df_result[standardize_feature_name(new_col)] = df_result[col].map(mapping).fillna(default)
        
        # 2. País
        country_col = standardize_feature_name('¿Cual es tu país?')
        if country_col not in df_result.columns:
            country_col = standardize_feature_name('¿Cuál es tu país?')
        
        if country_col in df_result.columns:
            # Frequency encoding
            freq_map_key = f'{country_col}_freq_map'
            if freq_map_key in params:
                freq_map = params[freq_map_key]
                df_result[standardize_feature_name('country_freq')] = df_result[country_col].map(freq_map).fillna(0)
            
            # Label encoding
            rare_key = f'{country_col}_rare_categories'
            label_key = f'{country_col}_label_mapping'
            if rare_key in params and label_key in params:
                rare_categories = params[rare_key]
                label_mapping = params[label_key]
                
                grouped_col = country_col + '_grouped'
                df_result[grouped_col] = df_result[country_col].apply(
                    lambda x: 'Rare' if x in rare_categories else x
                )
                df_result[grouped_col] = df_result[grouped_col].fillna('Unknown')
                df_result[standardize_feature_name('country_encoded')] = df_result[grouped_col].map(label_mapping).fillna(-1).astype(int)
                df_result = df_result.drop(columns=[grouped_col])
        
        # 3. UTMs
        utm_cols = [col for col in df_result.columns if 'utm_' in col.lower()]
        
        for col in utm_cols:
            # Label encoding
            label_key = f'{col}_label_mapping'
            if label_key in params:
                mapping = params[label_key]
                values = df_result[col].fillna('unknown').astype(str)
                df_result[standardize_feature_name(f'{col}_encoded')] = values.map(mapping).fillna(-1).astype(int)
            
            # Frequency encoding
            freq_key = f'{col}_freq_map'
            if freq_key in params:
                freq_map = params[freq_key]
                df_result[standardize_feature_name(f'{col}_freq')] = df_result[col].map(freq_map).fillna(0)
        
        # 4. GCLID
        gclid_cols = [col for col in df_result.columns if 'gclid' in col.lower()]
        if gclid_cols:
            gclid_col = gclid_cols[0]
            df_result[standardize_feature_name('has_gclid')] = df_result[gclid_col].notna().astype(int)
        
        return df_result
    
    def _save_component_params(self, param_manager: ExtendedParameterManager) -> None:
        """Salva parâmetros do componente."""
        params = {
            'categorical_params': self.categorical_params,
            'columns_to_remove': self.columns_to_remove,
            'preserved_columns': {k: v.to_dict() for k, v in self.preserved_columns.items()}
        }
        param_manager.save_component_params(self.name, params)
    
    def _load_component_params(self, param_manager: ExtendedParameterManager) -> None:
        """Carrega parâmetros do componente."""
        params = param_manager.get_component_params(self.name)
        self.categorical_params = params.get('categorical_params', {})
        self.columns_to_remove = params.get('columns_to_remove', [])
        # Não carregar preserved_columns no transform - só relevante no fit
        self.preserved_columns = {}