# preprocessing_pipeline.py - Pipeline para aplicar transformações em CV/test

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Importar módulos do projeto
from src.preprocessing.data_cleaning import (
    handle_missing_values, 
    handle_duplicates, 
    convert_data_types,
    consolidate_quality_columns
)
from src.preprocessing.email_processing import normalize_emails_in_dataframe
from src.preprocessing.text_processing import (
    clean_text, 
    extract_basic_text_features,
    extract_discriminative_features
)
from src.preprocessing.feature_engineering import (
    encode_categorical_features,
    create_identity_features,
    create_temporal_features
)

def preprocess_data(dataframe, is_training=False):
    """
    Aplica todas as etapas de pré-processamento em um DataFrame.
    
    Args:
        dataframe: DataFrame original
        is_training: Se True, ajusta os transformadores. Se False, utiliza transformadores existentes.
    
    Returns:
        DataFrame processado
    """
    print(f"Iniciando pré-processamento {'(modo treinamento)' if is_training else '(modo inferência)'}")
    print(f"Shape inicial: {dataframe.shape}")
    
    # 1. Limpeza de dados básica
    print("Etapa 1: Limpeza de dados básica...")
    df_cleaned = handle_missing_values(dataframe)
    df_cleaned = handle_duplicates(df_cleaned)
    df_cleaned = convert_data_types(df_cleaned)
    print(f"Shape após limpeza básica: {df_cleaned.shape}")
    
    # 2. Processamento de e-mails
    print("Etapa 2: Processamento de e-mails...")
    email_cols = [col for col in df_cleaned.columns if 'email' in col.lower()]
    if email_cols:
        df_cleaned = normalize_emails_in_dataframe(df_cleaned, email_cols)
    print(f"Shape após processamento de e-mails: {df_cleaned.shape}")
    
    # 3. Processamento de texto
    print("Etapa 3: Processamento de texto...")
    text_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype == 'object' 
                and col not in email_cols
                and not col.lower().startswith('utm')]
    
    if text_cols:
        # Clean text columns
        for col in text_cols:
            if col in df_cleaned.columns:
                df_cleaned[f"{col}_cleaned"] = df_cleaned[col].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
        
        # Extract basic features
        for col in text_cols:
            if f"{col}_cleaned" in df_cleaned.columns:
                basic_features = df_cleaned[f"{col}_cleaned"].apply(
                    lambda x: extract_basic_text_features(x) if isinstance(x, str) else {}
                )
                
                # Expand the dictionaries into separate columns
                for feature in ['word_count', 'char_count', 'avg_word_length', 'has_question', 'has_exclamation']:
                    df_cleaned[f"{col}_{feature}"] = basic_features.apply(
                        lambda x: x.get(feature, np.nan) if isinstance(x, dict) else np.nan
                    )
        
        # For training mode, extract discriminative features
        if is_training:
            for col in text_cols:
                if f"{col}_cleaned" in df_cleaned.columns:
                    discriminative_features = extract_discriminative_features(
                        df_cleaned[f"{col}_cleaned"], 
                        save_path=f"models/tfidf_{col}.joblib"
                    )
                    
                    # Add discriminative features to dataframe
                    for feature_name, feature_values in discriminative_features.items():
                        df_cleaned[f"{col}_tfidf_{feature_name}"] = feature_values
        else:
            # For inference mode, load and apply pre-trained discriminative features
            for col in text_cols:
                if f"{col}_cleaned" in df_cleaned.columns and os.path.exists(f"models/tfidf_{col}.joblib"):
                    tfidf_model = joblib.load(f"models/tfidf_{col}.joblib")
                    discriminative_features = tfidf_model.transform(df_cleaned[f"{col}_cleaned"].fillna(''))
                    
                    # Get feature names from the model
                    feature_names = tfidf_model.get_feature_names_out()
                    
                    # Add top N features as columns
                    top_n = 10  # Number of top features to include
                    for i in range(min(top_n, len(feature_names))):
                        feature_name = feature_names[i]
                        df_cleaned[f"{col}_tfidf_{feature_name}"] = discriminative_features[:, i].toarray().flatten()
    
    print(f"Shape após processamento de texto: {df_cleaned.shape}")
    
    # 4. Feature Engineering
    print("Etapa 4: Feature Engineering...")
    # Encode categorical features
    categorical_cols = [col for col in df_cleaned.columns 
                       if df_cleaned[col].dtype == 'object' 
                       and not col.endswith('_cleaned')]
    
    if categorical_cols:
        if is_training:
            df_cleaned, encoders = encode_categorical_features(df_cleaned, categorical_cols, save_encoders=True)
        else:
            df_cleaned = encode_categorical_features(df_cleaned, categorical_cols, save_encoders=False, load_encoders=True)
    
    # Create identity features
    identity_cols = [col for col in df_cleaned.columns if any(x in col.lower() for x in ['country', 'profession', 'utm'])]
    if identity_cols:
        df_cleaned = create_identity_features(df_cleaned, identity_cols)
    
    # Create temporal features if date columns exist
    date_cols = [col for col in df_cleaned.columns if any(x in col.lower() for x in ['date', 'time', 'day', 'hour'])]
    if date_cols:
        df_cleaned = create_temporal_features(df_cleaned, date_cols)
    
    print(f"Shape após feature engineering: {df_cleaned.shape}")
    
    # 5. Consolidar colunas de qualidade (se existirem)
    quality_cols = [col for col in df_cleaned.columns if 'quality' in col.lower()]
    if quality_cols:
        df_cleaned = consolidate_quality_columns(df_cleaned, quality_cols)
        print(f"Shape após consolidação de colunas de qualidade: {df_cleaned.shape}")
    
    # 6. Normalizar features numéricas
    numeric_cols = [col for col in df_cleaned.columns 
                   if df_cleaned[col].dtype in ['int64', 'float64']
                   and not col.startswith('target')]
    
    if numeric_cols:
        if is_training:
            scaler = StandardScaler()
            df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols].fillna(0))
            joblib.dump(scaler, 'models/standard_scaler.joblib')
        else:
            try:
                scaler = joblib.load('models/standard_scaler.joblib')
                df_cleaned[numeric_cols] = scaler.transform(df_cleaned[numeric_cols].fillna(0))
            except FileNotFoundError:
                print("Arquivo do scaler não encontrado. Aplicando normalização padrão.")
                df_cleaned[numeric_cols] = (df_cleaned[numeric_cols].fillna(0) - df_cleaned[numeric_cols].fillna(0).mean()) / df_cleaned[numeric_cols].fillna(0).std()
    
    print(f"Shape final após pré-processamento: {df_cleaned.shape}")
    return df_cleaned

def process_data_for_baseline(train_data, validation_data, test_data):
    """
    Aplica o pipeline de pré-processamento nos conjuntos de dados para o modelo baseline.
    
    Args:
        train_data: DataFrame de treinamento já processado
        validation_data: DataFrame de validação original
        test_data: DataFrame de teste original
    
    Returns:
        X_cv_processed, X_test_processed: DataFrames processados para validação e teste
    """
    # O conjunto de treinamento já está processado
    print("Processando conjunto de validação...")
    X_cv_processed = preprocess_data(validation_data, is_training=False)
    
    print("Processando conjunto de teste...")
    X_test_processed = preprocess_data(test_data, is_training=False)
    
    # Alinhar colunas com o conjunto de treinamento
    train_cols = set(train_data.columns)
    cv_cols = set(X_cv_processed.columns)
    test_cols = set(X_test_processed.columns)
    
    # Adicionar colunas faltantes
    for col in train_cols - cv_cols:
        X_cv_processed[col] = 0
    
    for col in train_cols - test_cols:
        X_test_processed[col] = 0
    
    # Remover colunas extras
    X_cv_processed = X_cv_processed[[col for col in train_data.columns if col in X_cv_processed.columns]]
    X_test_processed = X_test_processed[[col for col in train_data.columns if col in X_test_processed.columns]]
    
    # Garantir que todas as colunas estejam na mesma ordem
    X_cv_processed = X_cv_processed[train_data.columns]
    X_test_processed = X_test_processed[train_data.columns]
    
    return X_cv_processed, X_test_processed

def apply_preprocessing_to_cv_and_test(train_df, cv_df, test_df, target_column='target'):
    """
    Função principal para aplicar o pré-processamento nos conjuntos de validação e teste.
    
    Args:
        train_df: DataFrame de treinamento já processado
        cv_df: DataFrame de validação original
        test_df: DataFrame de teste original
        target_column: Nome da coluna alvo
    
    Returns:
        X_cv_processed, y_cv, X_test_processed, y_test: Conjuntos processados
    """
    # Separar features e target
    X_train = train_df.drop(columns=[target_column]) if target_column in train_df.columns else train_df
    X_cv = cv_df.drop(columns=[target_column]) if target_column in cv_df.columns else cv_df
    X_test = test_df.drop(columns=[target_column]) if target_column in test_df.columns else test_df
    
    y_cv = cv_df[target_column] if target_column in cv_df.columns else None
    y_test = test_df[target_column] if target_column in test_df.columns else None
    
    # Aplicar o pipeline de pré-processamento
    X_cv_processed, X_test_processed = process_data_for_baseline(X_train, X_cv, X_test)
    
    return X_cv_processed, y_cv, X_test_processed, y_test