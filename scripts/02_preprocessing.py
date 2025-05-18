#!/usr/bin/env python
"""
Script para aplicar a pipeline de pré-processamento nos conjuntos de treino, validação e teste,
garantindo que as mesmas transformações sejam aplicadas em todos.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
import argparse

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

# Imports dos módulos de pré-processamento
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
from src.preprocessing.advanced_feature_engineering import (
    refine_tfidf_weights,
    create_text_embeddings_simple,
    perform_topic_modeling,
    create_salary_features,
    create_country_interaction_features,
    create_age_interaction_features,
    create_temporal_interaction_features,
    advanced_feature_engineering
)

def apply_preprocessing_pipeline(df, params=None, fit=False, preserve_text=True):
    """
    Aplica a pipeline completa de pré-processamento.
    
    Args:
        df: DataFrame a ser processado
        params: Parâmetros para transformações (None para começar do zero)
        fit: Se True, ajusta as transformações, se False, apenas aplica
        preserve_text: Se True, preserva as colunas de texto originais
        
    Returns:
        DataFrame processado e parâmetros atualizados
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
    print(f"Colunas de texto identificadas ({len(text_cols)}): {text_cols[:3]}...")
    
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
    
    # 9. Feature engineering avançada (nova etapa)
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

def ensure_column_consistency(train_df, test_df):
    """
    Garante que o DataFrame de teste tenha as mesmas colunas que o de treinamento.
    
    Args:
        train_df: DataFrame de treinamento
        test_df: DataFrame de teste para alinhar
        
    Returns:
        DataFrame de teste com colunas alinhadas
    """
    print("Alinhando colunas entre conjuntos de dados...")
    
    # Colunas presentes no treino, mas ausentes no teste
    missing_cols = set(train_df.columns) - set(test_df.columns)
    
    # Adicionar colunas faltantes com valores padrão
    for col in missing_cols:
        if col in train_df.select_dtypes(include=['number']).columns:
            test_df[col] = 0
        else:
            test_df[col] = None
        print(f"  Adicionada coluna ausente: {col}")
    
    # Remover colunas extras no teste não presentes no treino
    extra_cols = set(test_df.columns) - set(train_df.columns)
    if extra_cols:
        test_df = test_df.drop(columns=list(extra_cols))
        print(f"  Removidas colunas extras: {', '.join(list(extra_cols)[:5])}" + 
              (f" e mais {len(extra_cols)-5} outras" if len(extra_cols) > 5 else ""))
    
    # Garantir a mesma ordem de colunas
    test_df = test_df[train_df.columns]
    
    print(f"Alinhamento concluído: {len(missing_cols)} colunas adicionadas, {len(extra_cols)} removidas")
    return test_df

def process_datasets(input_dir, output_dir, params_dir=None, preserve_text=True):
    """
    Função principal que processa todos os conjuntos na ordem correta.
    
    Args:
        input_dir: Diretório contendo os arquivos de entrada
        output_dir: Diretório para salvar os arquivos processados
        params_dir: Diretório para salvar os parâmetros (opcional)
        preserve_text: Se True, preserva as colunas de texto originais
    
    Returns:
        Dicionário com os DataFrames processados e parâmetros
    """
    # 1. Definir caminhos dos datasets
    train_path = os.path.join(input_dir, "train.csv")
    cv_path = os.path.join(input_dir, "validation.csv")
    test_path = os.path.join(input_dir, "test.csv")
    
    # Garantir que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar se os arquivos existem
    print(f"Verificando existência dos arquivos de entrada:")
    print(f"  Train path: {train_path} - Existe: {os.path.exists(train_path)}")
    print(f"  CV path: {cv_path} - Existe: {os.path.exists(cv_path)}")
    print(f"  Test path: {test_path} - Existe: {os.path.exists(test_path)}")
    
    if not all([os.path.exists(train_path), os.path.exists(cv_path), os.path.exists(test_path)]):
        print("ERRO: Um ou mais arquivos de entrada não foram encontrados!")
        print("Por favor, verifique o caminho dos arquivos.")
        return None
    
    # 2. Carregar os datasets
    print(f"Carregando datasets de {input_dir}...")
    train_df = pd.read_csv(train_path)
    cv_df = pd.read_csv(cv_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Datasets carregados: treino {train_df.shape}, validação {cv_df.shape}, teste {test_df.shape}")
    
    # 3. Processar o conjunto de treinamento com fit=True para aprender parâmetros
    print("\n--- Processando conjunto de treinamento ---")
    train_processed, params = apply_preprocessing_pipeline(train_df, fit=True, preserve_text=preserve_text)
    
    # 4. Salvar parâmetros aprendidos
    if params_dir:
        os.makedirs(params_dir, exist_ok=True)
        params_path = os.path.join(params_dir, "02_params.joblib")
        joblib.dump(params, params_path)
        print(f"Parâmetros de pré-processamento salvos em {params_path}")
    
    # 5. Salvar conjunto de treino processado
    train_processed.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    print(f"Dataset de treino processado e salvo em {os.path.join(output_dir, 'train.csv')}")
    
    # 6. Processar o conjunto de validação com fit=False para aplicar parâmetros aprendidos
    print("\n--- Processando conjunto de validação ---")
    cv_processed, _ = apply_preprocessing_pipeline(cv_df, params=params, fit=False, preserve_text=preserve_text)
    
    # 7. Garantir consistência de colunas com o treino
    cv_processed = ensure_column_consistency(train_processed, cv_processed)
    
    # 8. Salvar conjunto de validação processado
    cv_processed.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    print(f"Dataset de validação processado e salvo em {os.path.join(output_dir, 'validation.csv')}")
    
    # 9. Processar o conjunto de teste com fit=False para aplicar parâmetros aprendidos
    print("\n--- Processando conjunto de teste ---")
    test_processed, _ = apply_preprocessing_pipeline(test_df, params=params, fit=False, preserve_text=preserve_text)
    
    # 10. Garantir consistência de colunas com o treino
    test_processed = ensure_column_consistency(train_processed, test_processed)
    
    # 11. Salvar conjunto de teste processado
    test_processed.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Dataset de teste processado e salvo em {os.path.join(output_dir, 'test.csv')}")
    
    print("\nPré-processamento dos conjuntos concluído com sucesso!")
    print(f"Os datasets processados foram salvos em {output_dir}/")
    
    return {
        'train': train_processed,
        'cv': cv_processed,
        'test': test_processed,
        'params': params
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplicar pipeline de pré-processamento nos conjuntos de dados.")
    parser.add_argument("--input-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/01_split"), 
                        help="Diretório contendo os arquivos de entrada (train.csv, validation.csv, test.csv)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/02_1_processed"), 
                        help="Diretório para salvar os arquivos processados")
    parser.add_argument("--params-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/src/preprocessing/preprocessing_params"), 
                        help="Diretório para salvar os parâmetros aprendidos")
    parser.add_argument("--preserve-text", action="store_true", default=True,
                        help="Preservar as colunas de texto originais (default: True)")
    
    args = parser.parse_args()
    
    # Chamada da função principal
    results = process_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        params_dir=args.params_dir,
        preserve_text=args.preserve_text
    )
    
    if results is None:
        sys.exit(1)  # Sair com código de erro