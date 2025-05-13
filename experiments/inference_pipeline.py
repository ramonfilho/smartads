#!/usr/bin/env python
"""
Pipeline de Inferência Completa

Este script integra todos os componentes em uma pipeline única
para garantir a mesma sequência de processamento do treino.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.pipeline import Pipeline

# Configuração de caminhos absolutos
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models/artifacts/gmm_optimized")
PARAMS_DIR = os.path.join(PROJECT_ROOT, "src/preprocessing/preprocessing_params")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "predictions")

# Adicionar diretório do projeto ao path para poder importar os outros scripts
sys.path.append(PROJECT_ROOT)

# Importar os transformadores dos arquivos locais
# Usando importação absoluta em vez de relativa
from EmailNormalizationTransformer import EmailNormalizationTransformer
from CompletePreprocessingTransformer import CompletePreprocessingTransformer
from TextFeatureEngineeringTransformer import TextFeatureEngineeringTransformer
from GMM_InferenceWrapper import GMM_InferenceWrapper

def create_inference_pipeline(models_dir=MODELS_DIR, params_dir=PARAMS_DIR):
    """
    Cria a pipeline de inferência completa.
    
    Args:
        models_dir: Diretório contendo componentes do modelo
        params_dir: Diretório contendo parâmetros salvos
        
    Returns:
        Pipeline configurada e flag indicando se usa modelo calibrado
    """
    # Verificar existência dos diretórios
    if not os.path.exists(models_dir):
        raise ValueError(f"Diretório de modelos não encontrado: {models_dir}")
    if not os.path.exists(params_dir):
        raise ValueError(f"Diretório de parâmetros não encontrado: {params_dir}")
    
    # Definir caminhos para parâmetros
    preprocessing_params_path = os.path.join(params_dir, "all_preprocessing_params.joblib")
    
    # Verificar se os parâmetros existem
    if not os.path.exists(preprocessing_params_path):
        raise ValueError(f"Parâmetros de pré-processamento não encontrados: {preprocessing_params_path}")
    
    # Determinar se existe modelo calibrado
    use_calibrated = False
    calibrated_dir = os.path.join(PROJECT_ROOT, "models/calibrated")
    if os.path.exists(calibrated_dir):
        # Procurar pelo diretório mais recente
        calibrated_dirs = [d for d in os.listdir(calibrated_dir) 
                         if os.path.isdir(os.path.join(calibrated_dir, d)) 
                         and d.startswith("gmm_calibrated_")]
        
        if calibrated_dirs:
            # Ordenar por data de modificação (mais recente primeiro)
            calibrated_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(calibrated_dir, d)), 
                              reverse=True)
            
            calibrated_model_dir = os.path.join(calibrated_dir, calibrated_dirs[0])
            if os.path.exists(os.path.join(calibrated_model_dir, "gmm_calibrated.joblib")):
                models_dir = calibrated_model_dir
                use_calibrated = True
                print(f"Usando modelo GMM calibrado de: {calibrated_model_dir}")
    
    # Criar pipeline
    pipeline = Pipeline([
        ('email_normalization', EmailNormalizationTransformer(email_col='email')),
        ('preprocessing', CompletePreprocessingTransformer(params_path=preprocessing_params_path)),
        ('text_features', TextFeatureEngineeringTransformer(params_path=preprocessing_params_path)),
        ('predictor', GMM_InferenceWrapper(models_dir=models_dir))
    ])
    
    return pipeline, use_calibrated

def validate_input(df):
    """
    Valida o DataFrame de entrada para processamento.
    
    Args:
        df: DataFrame a ser validado
        
    Returns:
        Tuple (válido, mensagem)
    """
    # Verificar se o DataFrame não está vazio
    if df.empty:
        return False, "DataFrame vazio"
    
    # Verificar presença de campo de email
    if 'email' not in df.columns:
        return False, "Campo obrigatório 'email' não encontrado"
    
    # Verificar se há pelo menos uma coluna de texto
    text_columns = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?', 
        'Déjame un mensaje',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
    ]
    
    if not any(col in df.columns for col in text_columns):
        return False, f"Nenhuma coluna de texto encontrada. Esperadas: {text_columns}"
    
    return True, "Dados válidos"

def process_data(input_file, output_file=None):
    """
    Processa um arquivo de dados através da pipeline de inferência.
    
    Args:
        input_file: Caminho para o arquivo CSV com dados para previsão
        output_file: Caminho para salvar resultados (opcional)
        
    Returns:
        DataFrame com resultados
    """
    print(f"\nProcessando arquivo: {input_file}")
    
    # Verificar se o arquivo existe
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")
    
    # Carregar dados
    try:
        df = pd.read_csv(input_file)
        print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    except Exception as e:
        raise ValueError(f"Erro ao carregar arquivo: {e}")
    
    # Validar dados
    is_valid, message = validate_input(df)
    if not is_valid:
        raise ValueError(f"Dados inválidos: {message}")
    
    # Criar pipeline
    print("Criando pipeline de inferência...")
    pipeline, use_calibrated = create_inference_pipeline()
    print(f"Pipeline criada. Usando modelo {'calibrado' if use_calibrated else 'não calibrado'}")
    
    # Aplicar pipeline
    print("\nAplicando pipeline para previsão...")
    start_time = datetime.now()
    
    # Obter probabilidades
    probas = pipeline.predict_proba(df)
    
    # Obter previsões binárias
    predictions = pipeline.predict(df)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    print(f"Processamento concluído em {processing_time:.2f} segundos")
    
    # Criar DataFrame de resultados
    results_df = df.copy()
    results_df['prediction_probability'] = probas[:, 1]
    results_df['is_qualified_lead'] = predictions
    
    # Salvar resultados, se solicitado
    if output_file:
        # Criar diretório de saída se não existir
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results_df.to_csv(output_file, index=False)
        print(f"Resultados salvos em: {output_file}")
    
    # Imprimir estatísticas básicas
    qualified_leads = predictions.sum()
    qualified_pct = (qualified_leads / len(df)) * 100
    
    print(f"\nResultados:")
    print(f"Total de leads processados: {len(df)}")
    print(f"Leads qualificados: {int(qualified_leads)} ({qualified_pct:.2f}%)")
    
    return results_df

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Pipeline de inferência para Smart Ads')
    
    parser.add_argument('--input', required=True,
                        help='Arquivo CSV de entrada com dados para previsão')
    parser.add_argument('--output',
                        help='Arquivo CSV para salvar resultados (opcional)')
    
    args = parser.parse_args()
    
    # Processar dados
    try:
        output_file = args.output
        if not output_file:
            # Criar nome de arquivo de saída baseado no de entrada
            input_basename = os.path.basename(args.input)
            output_basename = f"results_{input_basename}"
            output_file = os.path.join(OUTPUT_DIR, output_basename)
        
        results = process_data(args.input, output_file)
        print("\nProcessamento concluído com sucesso!")
        
    except Exception as e:
        print(f"\nERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Criar diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()