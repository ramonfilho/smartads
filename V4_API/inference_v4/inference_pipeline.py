#!/usr/bin/env python
"""
Pipeline de inferência para Smart Ads.

Este script coordena a execução de três módulos adaptados de pré-processamento
e feature engineering, mais a predição usando um modelo GMM calibrado,
para gerar previsões em novos dados.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import argparse
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Importar a classe GMM_Wrapper para garantir que esteja disponível durante o carregamento
# No início do arquivo inference_pipeline.py, após os imports
from src.modeling.gmm_wrapper import GMM_Wrapper
import builtins
builtins.GMM_Wrapper = GMM_Wrapper  # Adicionar ao namespace global para desserialização

# Importar os scripts adaptados para inferência
from inference_v4 import inference_02_preprocessing as step1
from inference_v4 import inference_03_feature_engineering_1 as step2
from inference_v4 import inference_04_feature_engineering_2 as step3
from inference_v4 import inference_05_gmm_predict as step4

# Caminhos para parâmetros
PARAMS_PATHS = {
    'step1': os.path.join(project_root, "src/preprocessing/02_params/02_params.joblib"),
    'step2_tfidf': os.path.join(project_root, "src/preprocessing/03_params/03_tfidf_vectorizers.joblib"),
    'step2_lda': os.path.join(project_root, "src/preprocessing/03_params/03_lda_models.joblib"),
    'step3': os.path.join(project_root, "src/preprocessing/04_params/04_params.joblib")
}

def load_parameters():
    """Carrega os parâmetros pré-treinados para cada etapa."""
    params = {}
    
    print("Carregando parâmetros...")
    try:
        # Carregar parâmetros para a etapa 1
        print(f"Carregando parâmetros da etapa 1 de: {PARAMS_PATHS['step1']}")
        params['step1'] = joblib.load(PARAMS_PATHS['step1'])
        
        # Carregar parâmetros para a etapa 2
        print(f"Carregando parâmetros TF-IDF da etapa 2 de: {PARAMS_PATHS['step2_tfidf']}")
        tfidf_vectorizers = joblib.load(PARAMS_PATHS['step2_tfidf'])
        
        print(f"Carregando parâmetros LDA da etapa 2 de: {PARAMS_PATHS['step2_lda']}")
        lda_models = joblib.load(PARAMS_PATHS['step2_lda'])
        
        params['step2'] = {
            'tfidf_vectorizers': tfidf_vectorizers,
            'lda_models': lda_models
        }
        
        # Carregar parâmetros para a etapa 3
        print(f"Carregando parâmetros da etapa 3 de: {PARAMS_PATHS['step3']}")
        params['step3'] = joblib.load(PARAMS_PATHS['step3'])
        
        print("Todos os parâmetros carregados com sucesso.")
    except Exception as e:
        print(f"ERRO ao carregar parâmetros: {e}")
        return None
    
    return params

def process_inference(input_data, params=None, return_proba=True, threshold=None, use_calibrated=True):
    """
    Aplica a pipeline completa de inferência em um DataFrame de entrada.
    
    Args:
        input_data: DataFrame com dados brutos para inferência
        params: Parâmetros pré-carregados (se None, serão carregados)
        return_proba: Se True, retorna probabilidades em vez de classes
        threshold: Limiar personalizado para classificação (se None, usa o do modelo)
        use_calibrated: Se True, usa o modelo GMM calibrado
        
    Returns:
        DataFrame original com previsões adicionadas
    """
    # Fazer cópia do dataframe de entrada
    df_original = input_data.copy()
    
    # Registrar tempo de início
    start_time = time.time()
    
    print("\n===== INICIANDO PIPELINE DE INFERÊNCIA =====")
    print(f"Data e hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dimensões dos dados de entrada: {df_original.shape}")
    print(f"Usando modelo calibrado: {'Sim' if use_calibrated else 'Não'}")
    
    # Carregar parâmetros se não fornecidos
    if params is None:
        params = load_parameters()
        if params is None:
            print("ERRO: Falha ao carregar parâmetros. Abortando pipeline.")
            return None
    
    # Etapa 1: Pré-processamento básico
    print("\n----- Etapa 1: Pré-processamento básico -----")
    etapa1_start = time.time()
    df_step1 = step1.apply(df_original, params=params['step1'])
    etapa1_time = time.time() - etapa1_start
    print(f"Etapa 1 concluída em {etapa1_time:.2f} segundos. Dimensões: {df_step1.shape}")
    
    # Etapa 2: Feature engineering - texto
    print("\n----- Etapa 2: Feature engineering de texto -----")
    etapa2_start = time.time()
    df_step2 = step2.apply(df_step1, params=params['step2'])
    etapa2_time = time.time() - etapa2_start
    print(f"Etapa 2 concluída em {etapa2_time:.2f} segundos. Dimensões: {df_step2.shape}")
    
    # Etapa 3: Feature engineering avançado
    print("\n----- Etapa 3: Feature engineering avançado -----")
    etapa3_start = time.time()
    df_step3 = step3.apply(df_step2, params=params['step3'])
    etapa3_time = time.time() - etapa3_start
    print(f"Etapa 3 concluída em {etapa3_time:.2f} segundos. Dimensões: {df_step3.shape}")
    
    # Etapa 4: Previsão com o modelo GMM calibrado
    print("\n----- Etapa 4: Previsão com o modelo GMM" + (" calibrado" if use_calibrated else "") + " -----")
    prediction_start = time.time()
    
    # Gerar previsões
    predictions = step4.apply(
        df_step3, 
        return_proba=return_proba, 
        threshold=threshold
    )
    
    # Adicionar previsões ao dataframe original
    if predictions is not None:
        df_original = step4.analyze_predictions(df_original, predictions, threshold)
    else:
        print("ERRO: Falha ao gerar previsões.")
        return None
    
    prediction_time = time.time() - prediction_start
    print(f"Previsões concluídas em {prediction_time:.2f} segundos.")
    
    # Calcular tempo total
    total_time = time.time() - start_time
    print(f"\nPipeline de inferência concluída em {total_time:.2f} segundos.")
    
    # Resumo das dimensões
    print("\nResumo das dimensões:")
    print(f"Entrada:   {input_data.shape}")
    print(f"Etapa 1:   {df_step1.shape}")
    print(f"Etapa 2:   {df_step2.shape}")
    print(f"Etapa 3:   {df_step3.shape}")
    print(f"Saída:     {df_original.shape}")
    
    return df_original

def main():
    """Função principal para execução via linha de comando."""
    parser = argparse.ArgumentParser(description="Pipeline de inferência para Smart Ads")
    parser.add_argument("--input", type=str, required=True, 
                       help="Caminho para o arquivo CSV de entrada")
    parser.add_argument("--output", type=str, required=True,
                       help="Caminho para salvar o arquivo CSV de saída")
    parser.add_argument("--probabilities", action="store_true", default=True,
                       help="Retornar probabilidades em vez de apenas classes")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Threshold personalizado para classificação")
    parser.add_argument("--calibrated", action="store_true", default=True,
                       help="Usar modelo calibrado para previsões (padrão: True)")
    
    args = parser.parse_args()
    
    # Verificar se o arquivo de entrada existe
    if not os.path.exists(args.input):
        print(f"ERRO: Arquivo de entrada não encontrado: {args.input}")
        return 1
    
    # Criar diretório de saída se não existir
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Pipeline de Inferência Smart Ads ===")
    print(f"Arquivo de entrada: {args.input}")
    print(f"Arquivo de saída: {args.output}")
    print(f"Modo: {'Probabilidades' if args.probabilities else 'Classes'}")
    print(f"Usando modelo calibrado: {'Sim' if args.calibrated else 'Não'}")
    if args.threshold is not None:
        print(f"Threshold personalizado: {args.threshold}")
    
    # Carregar dados de entrada
    print(f"\nCarregando dados de entrada...")
    try:
        input_data = pd.read_csv(args.input)
        print(f"Dados carregados: {input_data.shape[0]} linhas, {input_data.shape[1]} colunas")
    except Exception as e:
        print(f"ERRO ao carregar dados de entrada: {e}")
        return 1
    
    # Processar dados
    results = process_inference(
        input_data, 
        params=None,  # Carregar automaticamente
        return_proba=args.probabilities,
        threshold=args.threshold,
        use_calibrated=args.calibrated
    )
    
    if results is None:
        print("ERRO durante o processamento. Verifique os logs acima.")
        return 1
    
    # Salvar resultados
    print(f"\nSalvando resultados em: {args.output}")
    try:
        results.to_csv(args.output, index=False)
        print(f"Resultados salvos com sucesso!")
    except Exception as e:
        print(f"ERRO ao salvar resultados: {e}")
        return 1
    
    print("\nPipeline de inferência concluída com sucesso!")
    return 0

if __name__ == "__main__":
    sys.exit(main())