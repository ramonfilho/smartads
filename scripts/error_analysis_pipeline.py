"""
Script unificado para análise de erros.

Este script executa uma análise completa de erros, combinando diferentes
abordagens para identificar padrões em falsos negativos e positivos.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import mlflow

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.evaluation.error_analysis import run_error_analysis

def run_error_analysis_pipeline(
    model_uri=None,
    model_path=None,
    data_path=None,
    train_path=None,
    val_path=None,
    target_col="target",
    threshold=None,
    output_dir="eda_results/error_analysis"
):
    """
    Executa a pipeline completa de análise de erros.
    
    Args:
        model_uri: URI do modelo no MLflow
        model_path: Caminho local para o modelo (alternativa ao URI)
        data_path: Caminho para o dataset único (para análise simplificada)
        train_path: Caminho para o dataset de treino (para análise comparativa)
        val_path: Caminho para o dataset de validação (para análise comparativa)
        target_col: Nome da coluna target
        threshold: Limiar de classificação (se None, será detectado do MLflow ou usado 0.5)
        output_dir: Diretório para salvar resultados
        
    Returns:
        Dicionário com resultados da análise
    """
    print("Iniciando pipeline de análise de erros...")
    
    # Verificar se temos modelo e dados suficientes
    if not (model_uri or model_path):
        raise ValueError("É necessário fornecer model_uri ou model_path")
    
    if data_path:
        # Usar análise de erro único (usando o método principal)
        results = run_error_analysis(
            model_uri=model_uri if model_uri else model_path,
            data_path=data_path,
            target_col=target_col,
            output_dir=output_dir,
            top_n_features=30
        )
    elif train_path and val_path:
        # Carregar o modelo
        if model_uri:
            print(f"Carregando modelo do MLflow: {model_uri}")
            if "random_forest" in model_uri.lower():
                model = mlflow.sklearn.load_model(model_uri)
            elif "lightgbm" in model_uri.lower():
                model = mlflow.lightgbm.load_model(model_uri)
            elif "xgboost" in model_uri.lower():
                model = mlflow.xgboost.load_model(model_uri)
            else:
                model_type = "sklearn"  # assume sklearn como padrão
                model = mlflow.pyfunc.load_model(model_uri)
        else:
            print(f"Carregando modelo local: {model_path}")
            import joblib
            model = joblib.load(model_path)
        
        # Determinar threshold
        if threshold is None and model_uri:
            client = mlflow.tracking.MlflowClient()
            run_id = model_uri.split("/")[1]
            try:
                run = client.get_run(run_id)
                threshold = float(run.data.metrics.get("threshold", 0.5))
                print(f"Usando threshold de {threshold} do MLflow")
            except:
                threshold = 0.15  # default baseado em análises anteriores
                print(f"Threshold não encontrado, usando padrão: {threshold}")
        elif threshold is None:
            threshold = 0.15
            print(f"Usando threshold padrão: {threshold}")
        
        # Carregar dados
        print(f"Carregando dados de {train_path} e {val_path}")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        # Executar análise detalhada
        from src.evaluation.error_analysis import error_analysis_from_validation_data
        results = error_analysis_from_validation_data(
            train_df, val_df, model, 
            best_threshold=threshold,
            target_col=target_col, 
            results_dir=output_dir
        )
    else:
        raise ValueError("É necessário fornecer data_path OU (train_path E val_path)")
    
    print("\nPipeline de análise de erros concluída!")
    print(f"Os resultados estão disponíveis no diretório: {output_dir}")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Executar análise de erros para um modelo treinado')
    parser.add_argument('--model_uri', help='MLflow model URI')
    parser.add_argument('--model_path', help='Caminho local para o modelo')
    parser.add_argument('--data_path', help='Caminho para o dataset único')
    parser.add_argument('--train_path', help='Caminho para o dataset de treino')
    parser.add_argument('--val_path', help='Caminho para o dataset de validação')
    parser.add_argument('--target_col', default='target', help='Nome da coluna target')
    parser.add_argument('--threshold', type=float, help='Limiar de classificação')
    parser.add_argument('--output_dir', default='eda_results/error_analysis', help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    
    print("Executando script de análise de erros...")
    results = run_error_analysis_pipeline(
        model_uri=args.model_uri,
        model_path=args.model_path,
        data_path=args.data_path,
        train_path=args.train_path,
        val_path=args.val_path,
        target_col=args.target_col,
        threshold=args.threshold,
        output_dir=args.output_dir
    )
    print("Script finalizado.")