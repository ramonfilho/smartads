"""
Módulo para aplicar transformações equivalentes ao script 02_preprocessing.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings

# Adicionar o diretório raiz do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# Importar módulos necessários
from src.preprocessing.email_processing import normalize_emails_in_dataframe
from src.preprocessing.data_cleaning import (
    consolidate_quality_columns,
    handle_missing_values,
    handle_outliers,
    normalize_values,
    convert_data_types
)
from src.preprocessing.feature_engineering import feature_engineering
from src.preprocessing.text_processing import text_feature_engineering
from src.preprocessing.advanced_feature_engineering import advanced_feature_engineering

# Suprimir avisos
warnings.filterwarnings('ignore')

def apply_preprocessing_pipeline(df, params=None, fit=False, preserve_text=True):
    """
    Replica a função apply_preprocessing_pipeline do script 02_preprocessing.py
    """
    # Inicializar parâmetros se não fornecidos
    if params is None:
        params = {}
    
    print(f"Iniciando pipeline de pré-processamento para DataFrame: {df.shape}")
    
    # 1. Normalizar emails
    print("1. Normalizando emails...")
    df = normalize_emails_in_dataframe(df, email_col='email')
    
    # 2. Consolidar colunas de qualidade
    print("2. Consolidando colunas de qualidade...")
    quality_params = params.get('quality_columns', {})
    df, quality_params = consolidate_quality_columns(df, fit=fit, params=quality_params)
    
    # 3. Tratamento de valores ausentes
    print("3. Tratando valores ausentes...")
    missing_params = params.get('missing_values', {})
    df, missing_params = handle_missing_values(df, fit=fit, params=missing_params)
    
    # 4. Tratamento de outliers
    print("4. Tratando outliers...")
    outlier_params = params.get('outliers', {})
    df, outlier_params = handle_outliers(df, fit=fit, params=outlier_params)
    
    # 5. Normalização de valores
    print("5. Normalizando valores numéricos...")
    norm_params = params.get('normalization', {})
    df, norm_params = normalize_values(df, fit=fit, params=norm_params)
    
    # 6. Converter tipos de dados
    print("6. Convertendo tipos de dados...")
    df, _ = convert_data_types(df, fit=fit)
    
    # Identificar colunas de texto antes do processamento
    text_cols = [
        col for col in df.columns 
        if df[col].dtype == 'object' and any(term in col for term in [
            'mensaje', 'inglés', 'vida', 'oportunidades', 'esperas', 'aprender', 
            'Semana', 'Inmersión', 'Déjame', 'fluidez'
        ])
    ]
    print(f"Colunas de texto identificadas ({len(text_cols)}): {text_cols[:3] if len(text_cols) > 0 else []}")
    
    # Criar cópia das colunas de texto originais (com sufixo _original)
    if text_cols and preserve_text:
        print("Preservando colunas de texto originais...")
        for col in text_cols:
            df[f"{col}_original"] = df[col].copy()
    
    # 7. Feature engineering não-textual
    print("7. Aplicando feature engineering não-textual...")
    feature_params = params.get('feature_engineering', {})
    df, feature_params = feature_engineering(df, fit=fit, params=feature_params)
    
    # 8. Processamento de texto
    print("8. Processando features textuais...")
    text_params = params.get('text_processing', {})
    df, text_params = text_feature_engineering(df, fit=fit, params=text_params)
    
    # 9. Feature engineering avançada
    print("9. Aplicando feature engineering avançada...")
    advanced_params = params.get('advanced_features', {})
    df, advanced_params = advanced_feature_engineering(df, fit=fit, params=advanced_params)
    
    # 10. Compilar parâmetros atualizados
    updated_params = {
        'quality_columns': quality_params,
        'missing_values': missing_params,
        'outliers': outlier_params,
        'normalization': norm_params,
        'feature_engineering': feature_params,
        'text_processing': text_params,
        'advanced_features': advanced_params
    }
    
    print(f"Pipeline concluída! Dimensões finais: {df.shape}")
    return df, updated_params

def fix_discriminative_features(df, params):
    """
    Cria manualmente as features discriminativas que estão faltando.
    """
    print("\nCorrigindo features discriminativas...")
    
    if 'text_processing' not in params or 'discriminative_terms' not in params['text_processing']:
        print("Parâmetros de termos discriminativos não encontrados.")
        return df
    
    discr_params = params['text_processing']['discriminative_terms']
    
    # Para cada coluna de texto com termos discriminativos
    for col, terms_dict in discr_params.items():
        print(f"Processando coluna: {col}")
        
        # Extrair termos positivos e negativos
        pos_terms = terms_dict.get('positive', [])
        neg_terms = terms_dict.get('negative', [])
        
        # Verificar se temos a coluna de texto original
        col_original = f"{col}_original"
        if col_original not in df.columns and col in df.columns:
            col_original = col
        
        if col_original not in df.columns:
            print(f"  Coluna original não encontrada: {col_original}")
            continue
        
        # Criar features para termos positivos
        for term_info in pos_terms:
            try:
                term = term_info[0]  # O termo está na posição 0 da tupla (termo, lift)
                feature_name = f"{col}_high_conv_term_{term}"
                df[feature_name] = df[col_original].fillna("").str.contains(term, case=False, na=False).astype(int)
                print(f"  Criada feature: {feature_name}")
            except Exception as e:
                print(f"  Erro ao criar feature para termo positivo {term}: {e}")
        
        # Criar features para termos negativos
        for term_info in neg_terms:
            try:
                term = term_info[0]  # O termo está na posição 0 da tupla (termo, lift)
                feature_name = f"{col}_low_conv_term_{term}"
                df[feature_name] = df[col_original].fillna("").str.contains(term, case=False, na=False).astype(int)
                print(f"  Criada feature: {feature_name}")
            except Exception as e:
                print(f"  Erro ao criar feature para termo negativo {term}: {e}")
        
        # Criar features agregadas
        high_features = [f"{col}_high_conv_term_{term[0]}" for term in pos_terms if f"{col}_high_conv_term_{term[0]}" in df.columns]
        if high_features:
            df[f"{col}_has_any_high_conv_term"] = df[high_features].max(axis=1)
            df[f"{col}_num_high_conv_terms"] = df[high_features].sum(axis=1)
            print(f"  Criadas features agregadas para termos positivos")
        
        low_features = [f"{col}_low_conv_term_{term[0]}" for term in neg_terms if f"{col}_low_conv_term_{term[0]}" in df.columns]
        if low_features:
            df[f"{col}_has_any_low_conv_term"] = df[low_features].max(axis=1)
            df[f"{col}_num_low_conv_terms"] = df[low_features].sum(axis=1)
            print(f"  Criadas features agregadas para termos negativos")
    
    print(f"Correção de features discriminativas concluída. DataFrame: {df.shape}")
    return df

def apply_script2_transformations(df, params_path):
    """
    Função principal para aplicar transformações do script 2.
    
    Args:
        df: DataFrame de entrada
        params_path: Caminho para o arquivo de parâmetros
        
    Returns:
        DataFrame processado
    """
    print(f"\n=== Aplicando transformações do script 2 ===")
    
    # Carregar parâmetros
    print(f"Carregando parâmetros de: {params_path}")
    try:
        params = joblib.load(params_path)
        print(f"Parâmetros carregados com sucesso.")
    except Exception as e:
        print(f"ERRO ao carregar parâmetros: {str(e)}")
        sys.exit(1)
    
    # Verificar se há coluna target
    has_target = 'target' in df.columns
    target_values = None
    
    if has_target:
        print("Removendo coluna 'target' para processamento...")
        target_values = df['target'].copy()
        df = df.drop(columns=['target'])
    
    # Aplicar pré-processamento
    df, _ = apply_preprocessing_pipeline(df, params=params, fit=False, preserve_text=True)
    
    # Aplicar correção para features discriminativas
    df = fix_discriminative_features(df, params)
    
    print(f"Transformações do script 2 concluídas. Dimensões finais: {df.shape}")
    return df