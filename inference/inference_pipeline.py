#!/usr/bin/env python
"""
Pipeline de inferência para o projeto Smart Ads.
Este script implementa todas as transformações necessárias para
processar novos dados e fazer predições usando o modelo GMM calibrado.
"""

import os
import pandas as pd
import numpy as np
import joblib
import re
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
from typing import Dict, List, Union, Optional, Any

class GMM_Wrapper:
    """
    Classe wrapper para o GMM que implementa a API sklearn para calibração.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pca_model = pipeline['pca_model']
        self.gmm_model = pipeline['gmm_model']
        self.scaler_model = pipeline['scaler_model']
        self.cluster_models = pipeline['cluster_models']
        self.n_clusters = pipeline['n_clusters']
        self.threshold = pipeline.get('threshold')
        
        # Adicionar atributos necessários para a API sklearn
        self.classes_ = np.array([0, 1])  # Classes binárias
        self._fitted = True  # Marcar como já ajustado
        self._estimator_type = "classifier"  # Indicar explicitamente que é um classificador
        
    def fit(self, X, y):
        # Como o modelo já está treinado, apenas verificamos as classes
        self.classes_ = np.unique(y)
        self._fitted = True
        return self
        
    def predict_proba(self, X):
        # Preparar os dados para o modelo GMM
        X_numeric = X.select_dtypes(include=['number'])
        
        # Substituir valores NaN por 0
        X_numeric = X_numeric.fillna(0)
        
        # Aplicar o scaler
        if hasattr(self.scaler_model, 'feature_names_in_'):
            # Garantir que temos exatamente as features esperadas pelo scaler
            scaler_features = self.scaler_model.feature_names_in_
            
            # Remover features extras e adicionar as que faltam
            features_to_remove = [col for col in X_numeric.columns if col not in scaler_features]
            X_numeric = X_numeric.drop(columns=features_to_remove, errors='ignore')
            
            for col in scaler_features:
                if col not in X_numeric.columns:
                    X_numeric[col] = 0.0
            
            # Garantir a ordem correta das colunas
            X_numeric = X_numeric[scaler_features]
        
        # Verificar novamente por NaNs após o ajuste de colunas
        X_numeric = X_numeric.fillna(0)
        
        X_scaled = self.scaler_model.transform(X_numeric)
        
        # Verificar por NaNs no array após scaling
        if np.isnan(X_scaled).any():
            # Se ainda houver NaNs, substitua-os por zeros
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # Aplicar PCA
        X_pca = self.pca_model.transform(X_scaled)
        
        # Aplicar GMM para obter cluster labels e probabilidades
        cluster_labels = self.gmm_model.predict(X_pca)
        cluster_probs = self.gmm_model.predict_proba(X_pca)
        
        # Inicializar array de probabilidades
        n_samples = len(X)
        y_pred_proba = np.zeros((n_samples, 2), dtype=float)
        
        # Para cada cluster, fazer previsões
        for cluster_id, model_info in self.cluster_models.items():
            # Converter cluster_id para inteiro se for string
            cluster_id_int = int(cluster_id) if isinstance(cluster_id, str) else cluster_id
            
            # Selecionar amostras deste cluster
            cluster_mask = (cluster_labels == cluster_id_int)
            
            if not any(cluster_mask):
                continue
            
            # Obter modelo específico do cluster
            model = model_info['model']
            
            # Detectar features necessárias
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                
                # Criar um DataFrame temporário com as features corretas
                X_temp = X.copy()
                
                # Lidar com features ausentes ou extras
                missing_features = [col for col in expected_features if col not in X.columns]
                for col in missing_features:
                    X_temp[col] = 0.0
                
                # Garantir a ordem correta das colunas
                features_to_use = [col for col in expected_features if col in X_temp.columns]
                X_cluster = X_temp.loc[cluster_mask, features_to_use].astype(float)
                
                # Substituir NaNs por zeros
                X_cluster = X_cluster.fillna(0)
            else:
                # Usar todas as features numéricas disponíveis
                X_cluster = X.loc[cluster_mask].select_dtypes(include=['number']).fillna(0)
            
            if len(X_cluster) > 0:
                # Fazer previsões
                try:
                    proba = model.predict_proba(X_cluster)
                    
                    # Armazenar resultados
                    y_pred_proba[cluster_mask] = proba
                except Exception as e:
                    print(f"ERRO ao fazer previsões para o cluster {cluster_id_int}: {e}")
                    # Em caso de erro, usar probabilidades default
                    y_pred_proba[cluster_mask, 0] = 0.9  # classe negativa (majoritária)
                    y_pred_proba[cluster_mask, 1] = 0.1  # classe positiva (minoritária)
        
        return y_pred_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

# Ignorar warnings
warnings.filterwarnings('ignore')

# Configuração de caminhos para parâmetros e modelos
PARAMS_DIR = "/Users/ramonmoreira/desktop/smart_ads/inference/params"

# Dicionário para armazenar os parâmetros e modelos carregados
MODELS = {
    'preprocessing_params': None,
    'tfidf_vectorizers': None,
    'lda_models': None,
    'script03_params': None,
    'pca_model': None,
    'scaler_model': None,
    'gmm_calibrated': None,
    'threshold': None
}

def load_models():
    """
    Carrega todos os modelos e parâmetros necessários para a pipeline.
    Deve ser chamado antes de iniciar o processamento.
    """
    global MODELS
    
    try:
        # Carregar parâmetros de pré-processamento (Script 02)
        preprocessing_path = os.path.join(PARAMS_DIR, "02_all_preprocessing_params.joblib")
        MODELS['preprocessing_params'] = joblib.load(preprocessing_path)
        print(f"✓ Parâmetros de pré-processamento carregados: {preprocessing_path}")
        
        # Carregar modelos de NLP (Script 03)
        tfidf_path = os.path.join(PARAMS_DIR, "03_tfidf_vectorizers.joblib")
        lda_path = os.path.join(PARAMS_DIR, "03_lda_models.joblib")
        MODELS['tfidf_vectorizers'] = joblib.load(tfidf_path)
        MODELS['lda_models'] = joblib.load(lda_path)
        print(f"✓ Modelos de NLP carregados: {tfidf_path}, {lda_path}")
        
        # Carregar parâmetros de features avançadas de texto (Script 04)
        script03_path = os.path.join(PARAMS_DIR, "04_script03_params.joblib")
        MODELS['script03_params'] = joblib.load(script03_path)
        print(f"✓ Parâmetros de features avançadas carregados: {script03_path}")
        
        # Carregar componentes de preparação para o modelo (Script 08)
        scaler_path = os.path.join(PARAMS_DIR, "08_scaler_model.joblib")
        pca_path = os.path.join(PARAMS_DIR, "08_pca_model.joblib")
        MODELS['scaler_model'] = joblib.load(scaler_path)
        MODELS['pca_model'] = joblib.load(pca_path)
        print(f"✓ Modelos de preparação carregados: {scaler_path}, {pca_path}")
        
        # Carregar modelo calibrado e threshold (Script 10)
        gmm_path = os.path.join(PARAMS_DIR, "10_gmm_calibrated.joblib")
        threshold_path = os.path.join(PARAMS_DIR, "10_threshold.txt")
        MODELS['gmm_calibrated'] = joblib.load(gmm_path)
        
        with open(threshold_path, 'r') as f:
            MODELS['threshold'] = float(f.read().strip())
        print(f"✓ Modelo calibrado carregado: {gmm_path}, threshold: {MODELS['threshold']}")
        
        return True
    
    except Exception as e:
        print(f"Erro ao carregar modelos: {str(e)}")
        return False

# Funções do Script 1: Normalização de emails
def normalize_email(email):
    """Normaliza endereços de email para facilitar correspondência.
    """
    if pd.isna(email) or email is None:
        return None
    
    email_str = str(email).lower().strip().replace(" ", "")
    if '@' not in email_str:
        return None
    
    username, domain = email_str.split('@', 1)
    
    # Gmail-specific normalization
    if domain == 'gmail.com':
        username = username.replace('.', '')
    
    # Common domain corrections
    domain_corrections = {
        'gmial.com': 'gmail.com', 'gmail.con': 'gmail.com', 'hotmial.com': 'hotmail.com',
        'outlook.con': 'outlook.com', 'yahoo.con': 'yahoo.com'
    }
    
    if domain in domain_corrections:
        domain = domain_corrections[domain]
    
    return f"{username}@{domain}"

def normalize_emails_in_dataframe(df, email_col='email'):
    """Adiciona uma coluna de emails normalizados a um DataFrame.
    """
    df_result = df.copy()
    if email_col in df_result.columns:
        df_result['email_norm'] = df_result[email_col].apply(normalize_email)
    return df_result

# Funções do Script 2: Pré-processamento

def detect_quality_columns(df):
    """Detecta colunas relacionadas à qualidade no DataFrame.
    """
    # Verificar se as colunas consolidadas já existem
    has_numeric = 'qualidade_numerica' in df.columns
    has_textual = 'qualidade_textual' in df.columns
    
    # Listar todas as variantes das colunas de qualidade
    quality_variants = [
        'Qualidade (Nome)', 'Qualidade (Número)', 'Qualidade (Numero)', 
        'Qualidade (nome)', 'Qualidade (número)', 'Qualidade (Nombre)', 
        'Qualidade', 'Qualidade (Número) ', 'Qualidade (Nome) '
    ]
    
    # Checar quais variantes existem no dataset
    existing_variants = [col for col in quality_variants if col in df.columns]
    
    # Verificar possíveis colunas renomeadas
    if len(existing_variants) == 0 and not (has_numeric or has_textual):
        possible_renamed = [col for col in df.columns if 'qual' in col.lower()]
    else:
        possible_renamed = []
    
    return {
        'has_numeric': has_numeric,
        'has_textual': has_textual,
        'existing_variants': existing_variants,
        'possible_renamed': possible_renamed
    }

def consolidate_quality_columns(df):
    """Consolida múltiplas colunas de qualidade em colunas normalizadas.
    """
    # Cria uma cópia do dataframe para não modificar o original
    df_result = df.copy()
    
    # Detectar colunas de qualidade
    quality_info = detect_quality_columns(df_result)
    
    params = MODELS['preprocessing_params']
    
    # No modo transform, só criar colunas se elas não existirem
    if not quality_info['has_numeric']:
        df_result['qualidade_numerica'] = np.nan
    if not quality_info['has_textual']:
        df_result['qualidade_textual'] = None
        
    # Remover colunas originais se ainda existirem
    variants_to_remove = [col for col in quality_info['existing_variants'] 
                         if col in df_result.columns]
    if variants_to_remove:
        df_result = df_result.drop(columns=variants_to_remove)
    
    return df_result

def handle_missing_values(df):
    """Trata valores ausentes com estratégias específicas por tipo de coluna.
    """
    # Criar cópia para não modificar o original
    df_result = df.copy()
    
    params = MODELS['preprocessing_params']
    missing_params = params.get('missing_values', {})
    
    # 1. Remover colunas com alta ausência (>95%)
    high_missing_cols = missing_params.get('high_missing_cols', [])
    
    # Remover apenas as colunas que existem no DataFrame
    cols_to_drop = [col for col in high_missing_cols if col in df_result.columns]
    if cols_to_drop:
        df_result = df_result.drop(columns=cols_to_drop)
    
    # 2. Colunas UTM (tratar para análise de marketing)
    utm_cols = [col for col in df_result.columns if col.startswith('UTM_') or 'utm' in col.lower()]
    for col in utm_cols:
        df_result[col] = df_result[col].fillna('unknown')
    
    # 3. Dados categóricos (preencher com 'desconhecido')
    cat_cols = [
        '¿Cuál es tu género?', '¿Cuál es tu edad?', '¿Cual es tu país?',
        '¿Hace quánto tiempo me conoces?', '¿Cuál es tu disponibilidad de tiempo para estudiar inglés?',
        '¿Cuál es tu sueldo anual? (en dólares)', '¿Cuánto te gustaría ganar al año?',
        '¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?',
        '¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?',
        'Qualidade', 'lançamento'
    ]
    
    for col in cat_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna('desconhecido')
    
    # 4. Colunas de texto livre (preencher com string vazia)
    text_cols = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
        'Déjame un mensaje',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
    ]
    
    for col in text_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna('')
    
    # 5. Colunas de qualidade (tratamento específico)
    quality_cols = [col for col in df_result.columns if 'qualidade' in col.lower() or 'qualidad' in col.lower()]
    quality_numeric_cols = []
    
    # Verificar quais colunas de qualidade são numéricas
    for col in quality_cols:
        if col in df_result.columns:
            try:
                non_null_values = df_result[col].dropna()
                pd.to_numeric(non_null_values, errors='raise')
                quality_numeric_cols.append(col)
            except:
                # Se não for numérica, tratar como categórica
                if df_result[col].isna().sum() > 0:
                    df_result[col] = df_result[col].fillna('desconhecido')
    
    # Processar colunas de qualidade numéricas
    for col in quality_numeric_cols:
        if col in df_result.columns:
            # Converter para numérico forçando valores não numéricos para NaN
            df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
            
            # Usar mediana armazenada ou calcular se não existir
            median_value = missing_params.get(f'median_{col}')
            if median_value is None:
                median_value = df_result[col].median()
            
            # Preencher valores ausentes
            df_result[col] = df_result[col].fillna(median_value)
    
    # 6. Outras colunas numéricas (preencher com mediana)
    numeric_cols = df_result.select_dtypes(include=['number']).columns.tolist()
    other_numeric_cols = [col for col in numeric_cols if col not in ['target'] + quality_numeric_cols]
    
    for col in other_numeric_cols:
        if col in df_result.columns and df_result[col].isna().sum() > 0:
            # Usar mediana armazenada ou calcular se não existir
            median_value = missing_params.get(f'median_{col}')
            if median_value is None:
                median_value = df_result[col].median()
            
            # Preencher valores ausentes
            df_result[col] = df_result[col].fillna(median_value)
    
    return df_result

def handle_outliers(df):
    """Trata outliers com estratégias diferentes para colunas de qualidade.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    params = MODELS['preprocessing_params']
    outlier_params = params.get('outliers', {})
    
    # Identificar colunas numéricas
    numeric_cols = df_result.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['target']]
    
    # Identificar colunas de qualidade numéricas
    quality_numeric_cols = [col for col in numeric_cols 
                           if 'qualidade' in col.lower() or 'qualidad' in col.lower()]
    
    # Processar cada coluna numérica
    for col in numeric_cols:
        if col in df_result.columns and df_result[col].nunique() > 10:
            # Tratamento específico para colunas de qualidade
            if col in quality_numeric_cols:
                # Usar limites armazenados
                if f'{col}_bounds' in outlier_params:
                    lower_bound, upper_bound = outlier_params[f'{col}_bounds']
                else:
                    # Fallback
                    lower_bound = df_result[col].quantile(0.01)
                    upper_bound = df_result[col].quantile(0.99)
                
                # Aplicar capping
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
            
            else:
                # Tratamento padrão para outras colunas numéricas
                # Usar limites armazenados
                if f'{col}_bounds' in outlier_params:
                    lower_bound, upper_bound = outlier_params[f'{col}_bounds']
                else:
                    # Fallback
                    q1 = df_result[col].quantile(0.25)
                    q3 = df_result[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                
                # Aplicar capping
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_result

def normalize_values(df):
    """Normaliza valores numéricos usando estatísticas do treinamento.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    params = MODELS['preprocessing_params']
    norm_params = params.get('normalization', {})
    
    # Selecionar colunas numéricas que não são target
    numeric_cols = df_result.select_dtypes(include=['number']).columns.tolist()
    cols_to_normalize = [col for col in numeric_cols if col != 'target' and col in df_result.columns]
    
    # No modo transform, usar as colunas identificadas no fit
    cols_with_std = norm_params.get('cols_with_std', [])
    cols_with_std = [col for col in cols_with_std if col in cols_to_normalize]
    
    # Se temos colunas para normalizar
    if cols_with_std:
        # Usar estatísticas armazenadas para normalizar
        if 'means' in norm_params and 'stds' in norm_params:
            # Criar um array para os dados normalizados
            df_normalized = np.zeros((len(df_result), len(cols_with_std)))
            
            # Normalizar cada coluna separadamente
            for i, col in enumerate(cols_with_std):
                if col in norm_params['means'] and col in norm_params['stds']:
                    mean = norm_params['means'][col]
                    std = norm_params['stds'][col]
                    df_normalized[:, i] = (df_result[col].values - mean) / std
                else:
                    # Se não temos as estatísticas, usar os dados originais
                    df_normalized[:, i] = df_result[col].values
        else:
            # Fallback: normalizar usando scaler
            df_normalized = MODELS['scaler_model'].transform(df_result[cols_with_std])
        
        # Atualizar o DataFrame com os valores normalizados
        df_result[cols_with_std] = df_normalized
    
    return df_result

def convert_data_types(df):
    """Converte tipos de dados para formatos apropriados.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Data e hora
    if 'Marca temporal' in df_result.columns:
        df_result['Marca temporal'] = pd.to_datetime(df_result['Marca temporal'], errors='coerce')
    
    if 'DATA' in df_result.columns:
        df_result['DATA'] = pd.to_datetime(df_result['DATA'], errors='coerce')
    
    return df_result

def create_identity_features(df):
    """Cria features baseadas nos campos de identidade do usuário.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Feature de comprimento do nome
    if '¿Cómo te llamas?' in df_result.columns:
        df_result['name_length'] = df_result['¿Cómo te llamas?'].str.len()
        df_result['name_word_count'] = df_result['¿Cómo te llamas?'].str.split().str.len()
    
    # Feature de validade do telefone
    if '¿Cual es tu telefono?' in df_result.columns:
        df_result['valid_phone'] = df_result['¿Cual es tu telefono?'].str.replace(r'\D', '', regex=True).str.len() >= 8
    
    # Feature de presença de instagram
    if '¿Cuál es tu instagram?' in df_result.columns:
        df_result['has_instagram'] = df_result['¿Cuál es tu instagram?'].notna() & (df_result['¿Cuál es tu instagram?'] != '')
    
    return df_result

def create_temporal_features(df):
    """Cria features baseadas em informações temporais.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Features de tempo baseadas em 'Marca temporal'
    if 'Marca temporal' in df_result.columns:
        # Converter para datetime se ainda não for
        if not pd.api.types.is_datetime64_dtype(df_result['Marca temporal']):
            df_result['Marca temporal'] = pd.to_datetime(df_result['Marca temporal'], errors='coerce')
        
        # Extrair componentes básicos
        df_result['hour'] = df_result['Marca temporal'].dt.hour
        df_result['day_of_week'] = df_result['Marca temporal'].dt.dayofweek
        df_result['month'] = df_result['Marca temporal'].dt.month
        df_result['year'] = df_result['Marca temporal'].dt.year
        
        # Features cíclicas para hora e dia da semana
        df_result['hour_sin'] = np.sin(2 * np.pi * df_result['hour'] / 24)
        df_result['hour_cos'] = np.cos(2 * np.pi * df_result['hour'] / 24)
        df_result['day_sin'] = np.sin(2 * np.pi * df_result['day_of_week'] / 7)
        df_result['day_cos'] = np.cos(2 * np.pi * df_result['day_of_week'] / 7)
        
        # Período do dia
        df_result['period_of_day'] = pd.cut(
            df_result['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['madrugada', 'manha', 'tarde', 'noite']
        )
    
    # Repetir processo para DATA se existir
    if 'DATA' in df_result.columns:
        if not pd.api.types.is_datetime64_dtype(df_result['DATA']):
            df_result['DATA'] = pd.to_datetime(df_result['DATA'], errors='coerce')
        
        # Extrair componentes apenas se conversão foi bem-sucedida
        if not df_result['DATA'].isna().all():
            df_result['utm_hour'] = df_result['DATA'].dt.hour
            df_result['utm_day_of_week'] = df_result['DATA'].dt.dayofweek
            df_result['utm_month'] = df_result['DATA'].dt.month
            df_result['utm_year'] = df_result['DATA'].dt.year
    
    return df_result

def encode_categorical_features(df):
    """Codifica variáveis categóricas usando diferentes estratégias.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    params = MODELS['preprocessing_params']
    categorical_params = params.get('categorical_encoding', {})
    
    # 2. Aplicar mapas para variáveis ordinais
    if '¿Cuál es tu edad?' in df_result.columns:
        df_result['age_encoded'] = df_result['¿Cuál es tu edad?'].map(categorical_params.get('age_map', {}))
    
    if '¿Hace quánto tiempo me conoces?' in df_result.columns:
        df_result['time_known_encoded'] = df_result['¿Hace quánto tiempo me conoces?'].map(categorical_params.get('time_map', {}))
    
    if '¿Cuál es tu disponibilidad de tiempo para estudiar inglés?' in df_result.columns:
        df_result['availability_encoded'] = df_result['¿Cuál es tu disponibilidad de tiempo para estudiar inglés?'].map(categorical_params.get('availability_map', {}))
    
    if '¿Cuál es tu sueldo anual? (en dólares)' in df_result.columns:
        df_result['current_salary_encoded'] = df_result['¿Cuál es tu sueldo anual? (en dólares)'].map(categorical_params.get('salary_map', {}))
    
    if '¿Cuánto te gustaría ganar al año?' in df_result.columns:
        df_result['desired_salary_encoded'] = df_result['¿Cuánto te gustaría ganar al año?'].map(categorical_params.get('desired_salary_map', {}))
    
    # Ambas colunas de crença
    for col in ['¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?', 
                '¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?']:
        if col in df_result.columns:
            new_col = 'belief_salary_encoded' if 'salario' in col else 'belief_work_encoded'
            df_result[new_col] = df_result[col].map(categorical_params.get('belief_map', {}))
    
    # Gênero
    if '¿Cuál es tu género?' in df_result.columns:
        df_result['gender_encoded'] = df_result['¿Cuál es tu género?'].map(categorical_params.get('gender_map', {}))
    
    # 3. Encoding para variáveis nominais de alta cardinalidade
    nominal_high_cardinality = ['¿Cual es tu país?', '¿Cuál es tu profesión?']
    
    for col in nominal_high_cardinality:
        if col in df_result.columns:
            # Frequency Encoding
            freq_map = categorical_params.get(f'{col}_freq_map', {})
            
            col_name = 'country_freq' if 'país' in col else 'profession_freq'
            df_result[col_name] = df_result[col].map(freq_map)
            
            # Agrupar categorias raras
            rare_categories = categorical_params.get(f'{col}_rare_categories', [])
            
            # Criar variável agrupada
            grouped_col = col + '_grouped'
            df_result[grouped_col] = df_result[col].apply(
                lambda x: 'Rare' if x in rare_categories else x
            )
            
            # Label Encoding para categoria agrupada
            encoded_col = 'country_encoded' if 'país' in col else 'profession_encoded'
            
            # Converter usando o mapeamento aprendido
            mapping = categorical_params.get(f'{col}_label_mapping', {})
            df_result[encoded_col] = df_result[grouped_col].map(mapping).fillna(-1).astype(int)
    
    # 4. Tratamento de UTMs
    utm_cols = [col for col in df_result.columns if 'UTM_' in col or 'utm_' in col]
    
    for col in utm_cols:
        if col in df_result.columns:
            # Verificar cardinalidade
            cardinality = df_result[col].nunique()
            
            if cardinality <= 10:  # Baixa cardinalidade
                # Label Encoding
                mapping = categorical_params.get(f'{col}_label_mapping', {})
                df_result[f'{col}_encoded'] = df_result[col].fillna('unknown').astype(str).map(mapping).fillna(-1).astype(int)
            else:  # Alta cardinalidade
                # Frequency Encoding
                freq_map = categorical_params.get(f'{col}_freq_map', {})
                df_result[f'{col}_freq'] = df_result[col].map(freq_map)
    
    # 5. GCLID como indicador binário
    if 'GCLID' in df_result.columns:
        df_result['has_gclid'] = df_result['GCLID'].notna().astype(int)
    
    return df_result

def feature_engineering(df):
    """Executa o pipeline de engenharia de features não-textuais.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # 1. Colunas a remover (exceto texto)
    cols_to_remove = [
        '¿Cómo te llamas?',
        '¿Cual es tu telefono?',
        '¿Cuál es tu instagram?'
    ]
    
    # Verificar quais colunas existem no dataframe
    cols_to_remove = [col for col in cols_to_remove if col in df_result.columns]
    
    # 2. Criar features
    df_result = create_identity_features(df_result)
    df_result = create_temporal_features(df_result)
    df_result = encode_categorical_features(df_result)
    
    # 3. Remover colunas originais após criação das features
    df_result = df_result.drop(columns=cols_to_remove, errors='ignore')
    
    return df_result

# Funções do Script 3: Processamento básico de NLP

def preprocess_text(text, language='spanish'):
    """
    Realiza pré-processamento básico de texto:
    - Converte para minúsculas
    - Remove caracteres especiais e números
    - Remove stopwords
    - Lematização básica
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Converter para minúsculas
    text = text.lower()

    # Remover URLs e emails
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Remover caracteres especiais e números (preservando letras acentuadas)
    text = re.sub(r'[^\w\s\áéíóúñü]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenização simplificada (sem dependência de parâmetro de idioma)
    tokens = text.split()

    # Remover stopwords (espanhol)
    try:
        stop_words = set(stopwords.words(language))
        tokens = [token for token in tokens if token not in stop_words]
    except:
        # Fallback para caso não tenhamos NLTK disponível
        common_stopwords = ['a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra',
                         'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e', 'el', 'ella']
        tokens = [token for token in tokens if token not in common_stopwords]

    # Lematização simplificada (sem dependência de parâmetro de idioma)
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except:
        pass  # Se não tiver o lemmatizer, mantém os tokens originais

    # Remover tokens muito curtos
    tokens = [token for token in tokens if len(token) > 2]

    # Juntar tokens novamente
    processed_text = ' '.join(tokens)

    return processed_text

def extract_basic_features(text):
    """
    Extrai features básicas de uma string de texto.
    """
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'word_count': 0,
            'char_count': 0,
            'avg_word_length': 0,
            'has_question': 0,
            'has_exclamation': 0
        }

    # Limpar e tokenizar o texto
    text = text.strip()
    words = text.split()

    # Extrair features
    word_count = len(words)
    char_count = len(text)
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    has_question = 1 if '?' in text else 0
    has_exclamation = 1 if '!' in text else 0

    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'has_question': has_question,
        'has_exclamation': has_exclamation
    }

def extract_tfidf_features(texts, vectorizer=None):
    """
    Extrai features TF-IDF com refinamento de pesos.
    """
    # Criar array para rastrear índices de documentos não vazios
    valid_indices = []
    filtered_texts = []

    # Filtrar textos vazios mantendo controle dos índices
    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text.strip()) > 0:
            filtered_texts.append(text)
            valid_indices.append(i)

    # Inicializar DataFrame de resultado para todos os documentos
    total_docs = len(texts)

    # Transformar usando o vetorizador existente
    if vectorizer and len(filtered_texts) > 0:
        tfidf_matrix = vectorizer.transform(filtered_texts)
        feature_names = vectorizer.get_feature_names_out()

        # Inicializar matriz para todos os documentos com zeros
        result_matrix = np.zeros((total_docs, len(feature_names)))

        # Preencher apenas para documentos válidos
        matrix_array = tfidf_matrix.toarray()
        for i, orig_idx in enumerate(valid_indices):
            result_matrix[orig_idx] = matrix_array[i]

        return result_matrix, feature_names
    else:
        # Retornar matriz vazia se não houver textos válidos ou vetorizador
        return np.zeros((total_docs, 0)), []

def extract_topics_lda(texts, n_topics=5, vectorizer=None, lda_model=None):
    """
    Extrai tópicos latentes usando LDA e preserva a indexação original.
    """
    # Criar array para rastrear índices de documentos não vazios
    valid_indices = []
    filtered_texts = []

    # Filtrar textos vazios mantendo controle dos índices
    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text.strip()) > 0:
            filtered_texts.append(text)
            valid_indices.append(i)

    if len(filtered_texts) < 10:
        # Retornar matriz vazia se não houver textos suficientes
        empty_matrix = np.zeros((len(texts), n_topics))
        return empty_matrix

    # Verificar se temos vectorizador e modelo
    if vectorizer is None or lda_model is None:
        return np.zeros((len(texts), n_topics))
        
    # Transformar com vectorizador existente
    dtm = vectorizer.transform(filtered_texts)
    
    # Obter distribuição de tópicos
    doc_topic_dist = lda_model.transform(dtm)
    
    # Criar matriz do tamanho original
    topic_dist = np.zeros((len(texts), n_topics))
    
    # Preencher apenas posições com textos válidos
    for i, orig_idx in enumerate(valid_indices):
        topic_dist[orig_idx] = doc_topic_dist[i]
        
    return topic_dist

def process_text_columns(df, text_columns):
    """
    Processa colunas de texto para extrair features básicas, TF-IDF e LDA.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    tfidf_vectorizers = MODELS['tfidf_vectorizers']
    lda_models = MODELS['lda_models']
    
    # Inicializar DataFrame para armazenar features
    result_features = pd.DataFrame(index=df_result.index)
    
    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df_result.columns:
            continue
            
        # Criar nome de coluna limpo
        col_key = col.replace(' ', '_').replace('?', '').replace('¿', '')
        col_key = re.sub(r'[^\w]', '', col_key)[:30]
        
        # 1. Pré-processar textos
        processed_texts = df_result[col].apply(lambda x: preprocess_text(x))
        
        # 2. Extrair features básicas
        basic_features = processed_texts.apply(extract_basic_features)
        
        # Adicionar features básicas ao DataFrame
        for feature in ['word_count', 'char_count', 'avg_word_length', 'has_question', 'has_exclamation']:
            result_features[f"{col_key}_{feature}"] = basic_features.apply(lambda x: x.get(feature, 0))
        
        # 3. Extrair features TF-IDF
        vectorizer = tfidf_vectorizers.get(col_key)
        if vectorizer:
            tfidf_matrix, feature_names = extract_tfidf_features(
                processed_texts,
                vectorizer=vectorizer
            )
            
            # Adicionar features TF-IDF - Corrigindo a verificação do array
            if isinstance(feature_names, (list, np.ndarray)) and len(feature_names) > 0:
                for i, term in enumerate(feature_names):
                    result_features[f"{col_key}_tfidf_{term}"] = tfidf_matrix[:, i]
        
        # 4. Extrair tópicos com LDA
        model_info = lda_models.get(col_key, {})
        cv = model_info.get('vectorizer')
        lda = model_info.get('lda_model')
        
        if cv and lda:
            topic_dist = extract_topics_lda(
                processed_texts,
                n_topics=3,
                vectorizer=cv,
                lda_model=lda
            )
            
            # Adicionar features de tópicos
            for i in range(min(3, topic_dist.shape[1])):
                result_features[f"{col_key}_topic_{i}_prob"] = topic_dist[:, i]
                
            # Adicionar tópico dominante
            if topic_dist.shape[1] > 0:
                result_features[f"{col_key}_dominant_topic"] = np.argmax(topic_dist, axis=1)
    
    # Combinar com o DataFrame original
    result_df = pd.concat([df_result, result_features], axis=1)
    
    return result_df

# Funções do Script 4: Features avançadas de texto
def process_professional_motivation(df, text_columns):
    """
    Aplica processamento de motivação profissional às colunas de texto.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Obter parâmetros do script 4
    script03_params = MODELS['script03_params']
    
    # Inicializar features
    professional_features = pd.DataFrame(index=df_result.index)
    
    # Palavras-chave relacionadas à motivação profissional com pesos
    work_keywords = script03_params.get('work_keywords', {})
    
    # Inicializar score array
    motivation_scores = np.zeros(len(df_result))
    keyword_counts = np.zeros(len(df_result))
    
    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df_result.columns:
            continue
            
        # Limpar e normalizar texto
        for idx, text in enumerate(df_result[col]):
            text = text.lower().strip() if isinstance(text, str) else ""
            if not text:
                continue
                
            # Contar ocorrências ponderadas de palavras-chave
            col_score = 0
            col_count = 0
            
            for keyword, weight in work_keywords.items():
                occurrences = text.count(keyword)
                if occurrences > 0:
                    col_score += occurrences * weight
                    col_count += occurrences
            
            # Adicionar aos scores totais
            motivation_scores[idx] += col_score
            keyword_counts[idx] += col_count
    
    # Normalizar scores (0-1 range)
    max_score = np.max(motivation_scores) if np.max(motivation_scores) > 0 else 1
    normalized_scores = motivation_scores / max_score
    
    # Adicionar features ao DataFrame
    professional_features['professional_motivation_score'] = normalized_scores
    professional_features['career_keyword_count'] = keyword_counts
    
    # Combinar com o DataFrame original
    df_result = pd.concat([df_result, professional_features], axis=1)
    
    return df_result

def process_aspiration_sentiment(df, text_columns):
    """
    Implementa análise de sentimento específica para linguagem de aspiração.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Obter parâmetros do script 4
    script03_params = MODELS['script03_params']
    
    # Frases de aspiração para detectar
    aspiration_phrases = script03_params.get('aspiration_phrases', [])
    
    # Inicializar features
    sentiment_features = pd.DataFrame(index=df_result.index)
    
    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df_result.columns:
            continue
            
        col_clean = col.replace(' ', '_').replace('?', '').replace('¿', '')
        col_clean = re.sub(r'[^\w]', '', col_clean)[:30]
        
        # Listas para armazenar resultados
        sentiment_pos = []
        sentiment_neg = []
        sentiment_compound = []
        aspiration_counts = []
        
        # Processar cada texto
        for text in df_result[col]:
            text = text.lower().strip() if isinstance(text, str) else ""
            
            # Análise básica para sentimento
            sentiment_pos.append(0.5)  # Valores default
            sentiment_neg.append(0.2)
            sentiment_compound.append(0.3)
            
            # Contar frases de aspiração
            count = sum(text.count(phrase) for phrase in aspiration_phrases)
            aspiration_counts.append(count)
        
        # Adicionar features ao DataFrame
        sentiment_features[f"{col_clean}_sentiment_pos"] = sentiment_pos
        sentiment_features[f"{col_clean}_sentiment_neg"] = sentiment_neg
        sentiment_features[f"{col_clean}_sentiment_compound"] = sentiment_compound
        sentiment_features[f"{col_clean}_aspiration_count"] = aspiration_counts
        
        # Score de aspiração
        sentiment_features[f"{col_clean}_aspiration_score"] = np.array(aspiration_counts) * np.array(sentiment_pos)
    
    # Combinar com o DataFrame original
    df_result = pd.concat([df_result, sentiment_features], axis=1)
    
    return df_result

def process_commitment_expressions(df, text_columns):
    """
    Cria features para expressões de compromisso e determinação.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Obter parâmetros do script 4
    script03_params = MODELS['script03_params']
    
    # Frases de compromisso com pesos
    commitment_phrases = script03_params.get('commitment_phrases', {})
    
    # Inicializar features
    commitment_features = pd.DataFrame(index=df_result.index)
    
    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df_result.columns:
            continue
            
        col_clean = col.replace(' ', '_').replace('?', '').replace('¿', '')
        col_clean = re.sub(r'[^\w]', '', col_clean)[:30]
        
        # Listas para armazenar features de compromisso
        commitment_scores = []
        has_commitment = []
        commitment_counts = []
        
        # Processar cada texto
        for text in df_result[col]:
            text = text.lower().strip() if isinstance(text, str) else ""
            
            # Contar e pontuar frases de compromisso
            score = 0
            count = 0
            
            for phrase, weight in commitment_phrases.items():
                occurrences = text.count(phrase)
                if occurrences > 0:
                    score += occurrences * weight
                    count += occurrences
            
            commitment_scores.append(score)
            has_commitment.append(1 if count > 0 else 0)
            commitment_counts.append(count)
        
        # Adicionar features ao DataFrame
        commitment_features[f"{col_clean}_commitment_score"] = commitment_scores
        commitment_features[f"{col_clean}_has_commitment"] = has_commitment
        commitment_features[f"{col_clean}_commitment_count"] = commitment_counts
    
    # Combinar com o DataFrame original
    df_result = pd.concat([df_result, commitment_features], axis=1)
    
    return df_result

def process_career_terms(df, text_columns):
    """
    Cria um detector para termos relacionados à carreira com pesos de importância.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Obter parâmetros do script 4
    script03_params = MODELS['script03_params']
    
    # Termos relacionados à carreira com pesos
    career_terms = script03_params.get('career_terms', {})
    
    # Inicializar features
    career_features = pd.DataFrame(index=df_result.index)
    
    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df_result.columns:
            continue
            
        col_clean = col.replace(' ', '_').replace('?', '').replace('¿', '')
        col_clean = re.sub(r'[^\w]', '', col_clean)[:30]
        
        # Listas para features de termos de carreira
        career_scores = []
        has_career_terms = []
        career_term_counts = []
        
        # Processar cada texto
        for text in df_result[col]:
            text = text.lower().strip() if isinstance(text, str) else ""
            
            # Contar e pontuar termos de carreira
            score = 0
            count = 0
            
            for term, weight in career_terms.items():
                occurrences = text.count(term)
                if occurrences > 0:
                    score += occurrences * weight
                    count += occurrences
            
            career_scores.append(score)
            has_career_terms.append(1 if count > 0 else 0)
            career_term_counts.append(count)
        
        # Adicionar features ao DataFrame
        career_features[f"{col_clean}_career_term_score"] = career_scores
        career_features[f"{col_clean}_has_career_terms"] = has_career_terms
        career_features[f"{col_clean}_career_term_count"] = career_term_counts
    
    # Combinar com o DataFrame original
    df_result = pd.concat([df_result, career_features], axis=1)
    
    return df_result

def process_advanced_text_features(df, text_columns):
    """
    Aplica processamento avançado de texto para melhorar predição.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Aplicar cada tipo de processamento
    df_result = process_professional_motivation(df_result, text_columns)
    df_result = process_aspiration_sentiment(df_result, text_columns)
    df_result = process_commitment_expressions(df_result, text_columns)
    df_result = process_career_terms(df_result, text_columns)
    
    return df_result

# Funções do Script 8: Preparação para o modelo
def prepare_for_model(df):
    """
    Prepara os dados para o modelo aplicando scaler e PCA.
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Obter scaler e PCA
    scaler_model = MODELS['scaler_model']
    pca_model = MODELS['pca_model']
    
    # Identificar features numéricas
    numeric_cols = []
    for col in df_result.columns:
        try:
            df_result[col].astype(float)
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    # Extrair features numéricas
    X_numeric = df_result[numeric_cols].copy().astype(float)
    
    # Substituir valores ausentes
    X_numeric.fillna(0, inplace=True)
    
    # Aplicar scaler
    X_scaled = scaler_model.transform(X_numeric)
    
    # Aplicar PCA
    X_pca = pca_model.transform(X_scaled)
    
    return X_pca

# Funções do Script 10: Predição com modelo calibrado
def predict(X_pca):
    """
    Faz a predição usando o modelo GMM calibrado e o threshold otimizado.
    """
    # Obter modelo calibrado e threshold
    gmm_calibrated = MODELS['gmm_calibrated']
    threshold = MODELS['threshold']
    
    # Calcular probabilidades
    probabilities = gmm_calibrated.predict_proba(X_pca)[:, 1]
    
    # Aplicar threshold
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities

# Função principal de inferência
def process_data(df):
    """
    Função principal que aplica todo o pipeline de inferência.
    
    Args:
        df: DataFrame com dados para inferência
        
    Returns:
        predictions: Classificação binária (0/1)
        probabilities: Probabilidades de classe positiva
        processed_df: DataFrame processado (opcional, para debug)
    """
    # Definir colunas de texto
    text_columns = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
        'Déjame un mensaje',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
    ]
    
    # 1. Normalizar emails
    print("1. Normalizando emails...")
    df_processed = normalize_emails_in_dataframe(df)
    
    # 2. Aplicar pipeline de pré-processamento
    print("2. Aplicando pré-processamento...")
    df_processed = consolidate_quality_columns(df_processed)
    df_processed = handle_missing_values(df_processed)
    df_processed = handle_outliers(df_processed)
    df_processed = normalize_values(df_processed)
    df_processed = convert_data_types(df_processed)
    
    # 3. Aplicar feature engineering não-textual
    print("3. Aplicando feature engineering não-textual...")
    df_processed = feature_engineering(df_processed)
    
    # 4. Processar features textuais básicas
    print("4. Processando features textuais básicas...")
    df_processed = process_text_columns(df_processed, text_columns)
    
    # 5. Processar features textuais avançadas
    print("5. Processando features textuais avançadas...")
    df_processed = process_advanced_text_features(df_processed, text_columns)
    
    # 6. Preparar para o modelo
    print("6. Preparando dados para o modelo...")
    X_pca = prepare_for_model(df_processed)
    
    # 7. Fazer predição
    print("7. Fazendo predição...")
    predictions, probabilities = predict(X_pca)
    
    print("Pipeline de inferência concluída!")
    
    # Retornar resultados
    return predictions, probabilities, df_processed

# Função para testar a pipeline com um arquivo
def test_pipeline(file_path):
    """
    Testa a pipeline de inferência com um arquivo CSV.
    
    Args:
        file_path: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com dados originais, predições e probabilidades
    """
    print(f"Carregando arquivo de teste: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Arquivo carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Carregar modelos se ainda não foram carregados
    if MODELS['preprocessing_params'] is None:
        if not load_models():
            return None
    
    # Aplicar pipeline
    print("Aplicando pipeline de inferência...")
    predictions, probabilities, _ = process_data(df)
    
    # Adicionar resultados ao DataFrame
    df_result = df.copy()
    df_result['prediction'] = predictions
    df_result['probability'] = probabilities
    
    print("Teste concluído!")
    
    return df_result

# Caso o script seja executado diretamente
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline de inferência para o projeto Smart Ads.')
    parser.add_argument('--test', type=str, default=os.path.join('/Users/ramonmoreira/desktop/smart_ads', 'data/01_split/test.csv'), 
                      help='Caminho para arquivo CSV de teste')
    parser.add_argument('--output', type=str, 
                      default=os.path.join('/Users/ramonmoreira/desktop/smart_ads', 'reports/inference/inference_results.csv'), 
                      help='Caminho para salvar resultados')
    
    args = parser.parse_args()
    
    # Garantir que o diretório de saída existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.test:
        # Carregar modelos
        load_models()
        
        # Testar pipeline
        results = test_pipeline(args.test)
        
        if results is not None:
            # Salvar resultados
            results.to_csv(args.output, index=False)
            print(f"Resultados salvos em: {args.output}")
            
            # Mostrar sumário
            positives = results['prediction'].sum()
            total = len(results)
            positive_rate = positives / total * 100
            print(f"\nSumário das predições:")
            print(f"Total de registros: {total}")
            print(f"Positivos (leads qualificados): {positives} ({positive_rate:.2f}%)")
            print(f"Negativos (leads não qualificados): {total - positives} ({100 - positive_rate:.2f}%)")
    else:
        print("Nenhum arquivo de teste especificado. Use --test para testar a pipeline.")