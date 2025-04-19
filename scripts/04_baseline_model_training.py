#!/usr/bin/env python
"""
Script para treinamento de modelos baseline do projeto Smart Ads.
"""

import os
import sys
import mlflow

# 1. Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Importar módulos do projeto
from src.evaluation import mlflow_utils
from src.evaluation import baseline_model

def run_baseline_training():
    """Pipeline de treinamento de modelos baseline."""
    print("Iniciando treinamento de modelos baseline...")
    
    # 3. Definir caminhos - usando caminhos absolutos para evitar problemas com MLflow
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    train_path = os.path.join(base_dir, "data/feature_selection/train.csv")
    val_path = os.path.join(base_dir, "data/feature_selection/validation.csv")
    mlflow_dir = os.path.join(base_dir, "models/mlflow")
    artifact_dir = os.path.join(base_dir, "models/artifacts")
    output_dir = os.path.join(base_dir, "models")
    
    # 4. Verificar existência dos datasets
    for path in [train_path, val_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset não encontrado: {path}")
            
    # 5. Criar diretórios de saída
    for dir_path in [output_dir, mlflow_dir, artifact_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 6. Configurar MLflow
    experiment_id = mlflow_utils.setup_mlflow_tracking(
        tracking_dir=mlflow_dir,
        experiment_name="smart-ads-baseline",
        clean_previous=False
    )
    
    # 7. Configurar diretório de artefatos
    mlflow_utils.setup_artifact_directory(artifact_dir)
    
    # 8. Treinar modelos baseline
    results = baseline_model.run_baseline_model_training(
        train_path=train_path,
        val_path=val_path,
        experiment_id=experiment_id,
        artifact_dir=artifact_dir,
        generate_learning_curves=False
    )
    
    print(f"\nTreinamento concluído com sucesso!")
    print(f"Os resultados estão disponíveis no MLflow em: {mlflow.get_tracking_uri()}")
    
    return results, experiment_id

if __name__ == "__main__":
    print("Executando script de treinamento de modelos baseline...")
    results, experiment_id = run_baseline_training()
    print("Script finalizado.")