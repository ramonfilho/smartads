#!/usr/bin/env python
"""
Script unificado de pré-processamento que integra todas as transformações
de feature engineering, incluindo processamento de texto avançado.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from tqdm import tqdm
import time

# Adicionar diretório raiz ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Suprimir avisos
warnings.filterwarnings('ignore')

# Importar módulos de pré-processamento
from src.preprocessing.email_processing import normalize_emails_in_dataframe
from src.preprocessing.data_cleaning import (
    consolidate_quality_columns, handle_missing_values, 
    handle_outliers, normalize_values, convert_data_types
)
from src.preprocessing.feature_engineering import feature_engineering
from src.preprocessing.text_processing import text_feature_engineering
from src.preprocessing.professional_motivation_features import (
    create_professional_motivation_score, analyze_aspiration_sentiment,
    detect_commitment_expressions, create_career_term_detector, 
    enhance_tfidf_for_career_terms, enhance_professional_features
)

# Definir constantes
TEXT_COLUMNS = [
    'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
    '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?', 
    'Déjame un mensaje',
    '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
]

def apply_unified_pipeline(df, params=None, fit=False):
    """
    Aplica pipeline unificada de pré-processamento, combinando todas as transformações.
    
    Args:
        df: DataFrame a ser processado
        params: Parâmetros para transformações (None para começar do zero)
        fit: Se True, ajusta as transformações, se False, apenas aplica
        
    Returns:
        DataFrame processado e parâmetros atualizados
    """
    # 1. Inicializar parâmetros se não fornecidos
    if params is None:
        params = {}
    
    print(f"Iniciando pipeline unificada para DataFrame: {df.shape}")
    start_time = time.time()
    
    # 2. Normalizar emails - fundamental para matching correto
    print("1. Normalizando emails...")
    df = normalize_emails_in_dataframe(df, email_col='email')
    
    # 3. Consolidar colunas de qualidade
    print("2. Consolidando colunas de qualidade...")
    quality_params = params.get('quality_columns', {})
    df, quality_params = consolidate_quality_columns(df, fit=fit, params=quality_params)
    params['quality_columns'] = quality_params
    
    # 4. Tratamento de valores ausentes
    print("3. Tratando valores ausentes...")
    missing_params = params.get('missing_values', {})
    df, missing_params = handle_missing_values(df, fit=fit, params=missing_params)
    params['missing_values'] = missing_params
    
    # 5. Tratamento de outliers
    print("4. Tratando outliers...")
    outlier_params = params.get('outliers', {})
    df, outlier_params = handle_outliers(df, fit=fit, params=outlier_params)
    params['outliers'] = outlier_params
    
    # 6. Normalização de valores numéricos
    print("5. Normalizando valores numéricos...")
    norm_params = params.get('normalization', {})
    df, norm_params = normalize_values(df, fit=fit, params=norm_params)
    params['normalization'] = norm_params
    
    # 7. Converter tipos de dados
    print("6. Convertendo tipos de dados...")
    df, _ = convert_data_types(df, fit=fit)
    
    # 8. Preservar colunas de texto originais
    text_cols = [col for col in TEXT_COLUMNS if col in df.columns]
    print(f"7. Preservando {len(text_cols)} colunas de texto originais...")
    for col in text_cols:
        df[f"{col}_original"] = df[col].copy()
    
    # 9. Feature engineering básico (não-textual)
    print("8. Aplicando feature engineering não-textual...")
    feature_params = params.get('feature_engineering', {})
    df, feature_params = feature_engineering(df, fit=fit, params=feature_params)
    params['feature_engineering'] = feature_params
    
    # 10. Processamento de texto básico
    print("9. Processando features textuais básicas...")
    text_params = params.get('text_processing', {})
    df, text_params = text_feature_engineering(df, fit=fit, params=text_params)
    params['text_processing'] = text_params
    
    # 11. Features de motivação profissional (usando a função integrada do módulo)
    print("10. Aplicando engenharia de features profissionais...")
    prof_params = params.get('professional_features', {})
    df, prof_params = enhance_professional_features(df, text_cols, fit=fit, params=prof_params)
    params['professional_features'] = prof_params
    
    # 12. Remover colunas duplicadas
    original_cols = df.columns.tolist()
    df = df.loc[:, ~df.columns.duplicated()]
    if len(original_cols) > len(df.columns):
        print(f"Removidas {len(original_cols) - len(df.columns)} colunas duplicadas")
    
    elapsed_time = time.time() - start_time
    print(f"Pipeline concluída! Dimensões finais: {df.shape}, tempo: {elapsed_time:.2f}s")
    
    return df, params

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
    
    # Remover colunas extras no teste não presentes no treino
    extra_cols = set(test_df.columns) - set(train_df.columns)
    if extra_cols:
        test_df = test_df.drop(columns=list(extra_cols))
    
    # Garantir a mesma ordem de colunas
    test_df = test_df[train_df.columns]
    
    print(f"Alinhamento concluído: {len(missing_cols)} colunas adicionadas, {len(extra_cols)} removidas")
    return test_df

def process_datasets(input_dir, output_dir, params_path=None):
    """
    Função principal que processa todos os conjuntos na ordem correta.
    
    Args:
        input_dir: Diretório contendo os arquivos de entrada
        output_dir: Diretório para salvar os arquivos processados
        params_path: Caminho para salvar/carregar os parâmetros
    
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
    if not all([os.path.exists(train_path), os.path.exists(cv_path), os.path.exists(test_path)]):
        print("ERRO: Um ou mais arquivos de entrada não foram encontrados!")
        return None
    
    # 2. Carregar os datasets
    print(f"Carregando datasets de {input_dir}...")
    train_df = pd.read_csv(train_path)
    cv_df = pd.read_csv(cv_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Datasets carregados: treino {train_df.shape}, validação {cv_df.shape}, teste {test_df.shape}")
    
    # 3. Carregar parâmetros se existirem, ou criar novos
    params = None
    if params_path and os.path.exists(params_path):
        try:
            print(f"Carregando parâmetros de {params_path}...")
            params = joblib.load(params_path)
        except Exception as e:
            print(f"Erro ao carregar parâmetros: {e}")
            params = {}
    
    # 4. Processar o conjunto de treinamento com fit=True
    print("\n--- Processando conjunto de treinamento ---")
    train_processed, params = apply_unified_pipeline(train_df, params=params, fit=True)
    
    # 5. Salvar parâmetros aprendidos
    if params_path:
        os.makedirs(os.path.dirname(params_path), exist_ok=True)
        joblib.dump(params, params_path)
        print(f"Parâmetros de pré-processamento salvos em {params_path}")
    
    # 6. Salvar conjunto de treino processado
    train_out_path = os.path.join(output_dir, "train.csv")
    train_processed.to_csv(train_out_path, index=False)
    print(f"Dataset de treino processado e salvo em {train_out_path}")
    
    # 7. Processar o conjunto de validação com fit=False
    print("\n--- Processando conjunto de validação ---")
    cv_processed, _ = apply_unified_pipeline(cv_df, params=params, fit=False)
    
    # 8. Garantir consistência de colunas com o treino
    cv_processed = ensure_column_consistency(train_processed, cv_processed)
    
    # 9. Salvar conjunto de validação processado
    cv_out_path = os.path.join(output_dir, "validation.csv")
    cv_processed.to_csv(cv_out_path, index=False)
    print(f"Dataset de validação processado e salvo em {cv_out_path}")
    
    # 10. Processar o conjunto de teste com fit=False
    print("\n--- Processando conjunto de teste ---")
    test_processed, _ = apply_unified_pipeline(test_df, params=params, fit=False)
    
    # 11. Garantir consistência de colunas com o treino
    test_processed = ensure_column_consistency(train_processed, test_processed)
    
    # 12. Salvar conjunto de teste processado
    test_out_path = os.path.join(output_dir, "test.csv")
    test_processed.to_csv(test_out_path, index=False)
    print(f"Dataset de teste processado e salvo em {test_out_path}")
    
    print("\nPré-processamento dos conjuntos concluído com sucesso!")
    
    return {
        'train': train_processed,
        'cv': cv_processed,
        'test': test_processed,
        'params': params
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline unificada de pré-processamento.")
    parser.add_argument("--input-dir", type=str, required=True, 
                        help="Diretório contendo os arquivos de entrada (train.csv, validation.csv, test.csv)")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Diretório para salvar os arquivos processados")
    parser.add_argument("--params-path", type=str, 
                        help="Caminho para salvar/carregar os parâmetros")
    
    args = parser.parse_args()
    
    # Chamada da função principal
    results = process_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        params_path=args.params_path
    )
    
    if results is None:
        sys.exit(1)  # Sair com código de erro