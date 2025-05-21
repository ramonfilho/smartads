#!/usr/bin/env python
"""
Script adaptado para inferência, baseado no 02_preprocessing.py original.
Função principal apply() recebe um DataFrame e retorna o DataFrame processado.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings

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

def ensure_column_consistency(train_features, test_df):
    """
    Garante que o DataFrame de teste tenha as mesmas colunas que o de treinamento.
    
    Args:
        train_features: Lista de features do modelo treinado
        test_df: DataFrame de teste para alinhar
        
    Returns:
        DataFrame de teste com colunas alinhadas
    """
    print("Alinhando colunas com as features do modelo treinado...")
    
    # Converter para conjunto para operações mais rápidas
    test_cols = set(test_df.columns)
    train_cols = set(train_features)
    
    # Colunas presentes no treino, mas ausentes no teste
    missing_cols = train_cols - test_cols
    
    # Adicionar colunas faltantes com valores padrão
    for col in missing_cols:
        # Assumir valores numéricos para simplificar
        test_df[col] = 0
        print(f"  Adicionada coluna ausente: {col}")
    
    # Remover colunas extras no teste não presentes no treino
    extra_cols = test_cols - train_cols
    if extra_cols:
        test_df = test_df.drop(columns=list(extra_cols))
        print(f"  Removidas {len(extra_cols)} colunas extras")
    
    # Garantir a mesma ordem de colunas
    return test_df[train_features]

def apply(df, params=None):
    """
    Função principal para aplicar o pré-processamento em modo de inferência.
    
    Args:
        df: DataFrame com dados brutos de entrada
        params: Parâmetros pré-treinados (se None, serão carregados do disco)
        
    Returns:
        DataFrame processado com as mesmas transformações do conjunto de treino
    """
    # Carregar parâmetros se não fornecidos
    if params is None:
        params_path = "/Users/ramonmoreira/desktop/smart_ads/src/preprocessing/02_params/02_params.joblib"
        print(f"Carregando parâmetros de: {params_path}")
        try:
            params = joblib.load(params_path)
        except Exception as e:
            print(f"ERRO ao carregar parâmetros: {e}")
            print("Prosseguindo sem parâmetros pré-treinados")
            params = {}
    
    # Fazer cópia do DataFrame para não modificar o original
    df_copy = df.copy()
    
    # Aplicar pipeline de pré-processamento em modo transform
    df_processed, _ = apply_preprocessing_pipeline(
        df_copy, 
        params=params, 
        fit=False,  # Forçar modo transform
        preserve_text=True  # Manter colunas de texto para próximos scripts
    )
    
    # Verificar se temos informações sobre features esperadas pelo modelo
    model_features_path = "/Users/ramonmoreira/desktop/smart_ads/models/artifacts/gmm_optimized/model_features.joblib"
    if os.path.exists(model_features_path):
        print(f"Carregando lista de features do modelo de: {model_features_path}")
        try:
            model_features = joblib.load(model_features_path)
            # Alinhar colunas com as esperadas pelo modelo
            df_processed = ensure_column_consistency(model_features, df_processed)
        except Exception as e:
            print(f"AVISO: Não foi possível alinhar features com o modelo: {e}")
    
    print(f"Pré-processamento concluído. Dimensões finais: {df_processed.shape}")
    return df_processed

# Manter a funcionalidade original para uso direto do script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aplicar pré-processamento em modo de inferência.")
    parser.add_argument("--input", type=str, required=True, 
                       help="Caminho para o arquivo CSV de entrada")
    parser.add_argument("--output", type=str, required=True,
                       help="Caminho para salvar o arquivo CSV processado")
    
    args = parser.parse_args()
    
    # Carregar dados
    print(f"Carregando dados de: {args.input}")
    input_data = pd.read_csv(args.input)
    
    # Processar
    processed_data = apply(input_data)
    
    # Salvar resultado
    print(f"Salvando resultado em: {args.output}")
    processed_data.to_csv(args.output, index=False)
    
    print("Processamento concluído!")