#!/usr/bin/env python
"""
Script para criar features específicas para redução de falsos negativos,
baseado nos resultados da análise de erros.

Este script aplica transformações adicionais nos datasets processados
para melhorar a detecção de falsos negativos e aumentar o recall do modelo.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar o módulo de features para falsos negativos
from src.preprocessing.fn_targeted_features import enhance_features_for_false_negatives

def process_datasets(input_dir, output_dir, params_dir=None):
    """
    Processa os datasets (treino, validação, teste) aplicando as 
    transformações específicas para falsos negativos.
    
    Args:
        input_dir: Diretório contendo os datasets de entrada
        output_dir: Diretório para salvar os datasets processados
        params_dir: Diretório para salvar os parâmetros aprendidos
    
    Returns:
        Dict com os datasets processados e parâmetros
    """
    # 1. Definir caminhos dos datasets
    train_path = os.path.join(input_dir, "train.csv")
    cv_path = os.path.join(input_dir, "validation.csv")
    test_path = os.path.join(input_dir, "test.csv")
    
    # Garantir que o diretório de saída existe (usar pasta fixa em vez de adicionar timestamp)
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
    start_time = time.time()
    
    # Aplicar enhance_features_for_false_negatives
    train_processed, params = enhance_features_for_false_negatives(train_df, fit=True)
    
    train_time = time.time() - start_time
    print(f"Tempo de processamento do treino: {train_time:.2f} segundos")
    
    # 4. Salvar parâmetros aprendidos
    if params_dir:
        os.makedirs(params_dir, exist_ok=True)
        params_path = os.path.join(params_dir, "fn_features_params.joblib")
        joblib.dump(params, params_path)
        print(f"Parâmetros de features de falsos negativos salvos em {params_path}")
    
    # 5. Salvar conjunto de treino processado
    train_processed.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    print(f"Dataset de treino processado e salvo em {os.path.join(output_dir, 'train.csv')}")
    
    # 6. Processar o conjunto de validação com fit=False para aplicar parâmetros aprendidos
    print("\n--- Processando conjunto de validação ---")
    start_time = time.time()
    cv_processed, _ = enhance_features_for_false_negatives(cv_df, fit=False, params=params)
    cv_time = time.time() - start_time
    print(f"Tempo de processamento da validação: {cv_time:.2f} segundos")
    
    def ensure_column_consistency(train_df, other_df):
        """
        Garante que o DataFrame tenha as mesmas colunas que o de treinamento.
        
        Args:
            train_df: DataFrame de referência
            other_df: DataFrame para alinhar
            
        Returns:
            DataFrame com colunas alinhadas
        """
        print("Alinhando colunas entre conjuntos de dados...")
        
        # Colunas presentes no treino, mas ausentes no outro conjunto
        missing_cols = set(train_df.columns) - set(other_df.columns)
        
        # Adicionar colunas faltantes com valores padrão
        for col in missing_cols:
            if col in train_df.select_dtypes(include=['number']).columns:
                other_df[col] = 0.0
            else:
                other_df[col] = None
            print(f"  Adicionada coluna ausente: {col}")
        
        # Remover colunas extras não presentes no treino
        extra_cols = set(other_df.columns) - set(train_df.columns)
        if extra_cols:
            other_df = other_df.drop(columns=list(extra_cols))
            print(f"  Removidas colunas extras: {', '.join(list(extra_cols)[:5])}" + 
                  (f" e mais {len(extra_cols)-5} outras" if len(extra_cols) > 5 else ""))
        
        # Garantir a mesma ordem de colunas
        other_df = other_df[train_df.columns]
        
        print(f"Alinhamento concluído: {len(missing_cols)} colunas adicionadas, {len(extra_cols)} removidas")
        return other_df
    
    # 7. Garantir consistência de colunas para validação
    cv_processed = ensure_column_consistency(train_processed, cv_processed)
    
    # 8. Salvar conjunto de validação processado
    cv_processed.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    print(f"Dataset de validação processado e salvo em {os.path.join(output_dir, 'validation.csv')}")
    
    # 9. Processar o conjunto de teste com fit=False para aplicar parâmetros aprendidos
    print("\n--- Processando conjunto de teste ---")
    start_time = time.time()
    test_processed, _ = enhance_features_for_false_negatives(test_df, fit=False, params=params)
    test_time = time.time() - start_time
    print(f"Tempo de processamento do teste: {test_time:.2f} segundos")
    
    # 10. Garantir consistência de colunas para teste
    test_processed = ensure_column_consistency(train_processed, test_processed)
    
    # 11. Salvar conjunto de teste processado
    test_processed.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Dataset de teste processado e salvo em {os.path.join(output_dir, 'test.csv')}")
    
    # 12. Verificar e comparar nomes de colunas após o processamento
    print(f"\nVerificando compatibilidade de colunas:")
    integer_cols = [col for col in train_processed.columns if 'int' in str(train_processed[col].dtype)]
    if integer_cols:
        print(f"AVISO: Ainda existem {len(integer_cols)} colunas inteiras que podem causar problemas de compatibilidade.")
        print(f"Primeiras colunas inteiras: {integer_cols[:5]}")
    else:
        print(f"Compatibilidade de tipos: OK - Todas as colunas numéricas são float")
    
    # Verificar características problemáticas nos nomes de colunas
    problem_chars = [col for col in train_processed.columns if any(c in col for c in '?¿.,;:!')]
    if problem_chars:
        print(f"AVISO: {len(problem_chars)} colunas contêm caracteres problemáticos.")
        print(f"Exemplos: {problem_chars[:5]}")
    else:
        print(f"Compatibilidade de nomes: OK - Nenhum caractere problemático encontrado")
    
    print("\nProcessamento de features para falsos negativos concluído com sucesso!")
    print(f"Os datasets processados foram salvos em {output_dir}/")
    
    return {
        'train': train_processed,
        'cv': cv_processed,
        'test': test_processed,
        'params': params
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplicar features específicas para redução de falsos negativos.")
    parser.add_argument("--input-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/02_3_processed_text_code6"), 
                        help="Diretório contendo os arquivos de entrada (train.csv, validation.csv, test.csv)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/02_4_processed_text_fn_code7"), 
                        help="Diretório para salvar os arquivos processados")
    parser.add_argument("--params-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/src/preprocessing/preprocessing_params"), 
                        help="Diretório para salvar os parâmetros aprendidos")
    
    args = parser.parse_args()
    
    # Usar diretamente o diretório de saída especificado
    output_dir = args.output_dir
    
    # Chamada da função principal
    results = process_datasets(
        input_dir=args.input_dir,
        output_dir=output_dir,
        params_dir=args.params_dir
    )
    
    if results is None:
        sys.exit(1)  # Sair com código de erro