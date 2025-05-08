#!/usr/bin/env python
"""
Script corrigido para treinar modelos baseline e garantir o salvamento correto do modelo RandomForest.
Este script resolve problemas de configuração do MLflow para evitar a falha no salvamento do modelo.
"""

import os
import sys
import mlflow

# Caminho absoluto para o diretório raiz do projeto
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# Importar módulos
from src.evaluation.baseline_model import run_baseline_model_training

# Configuração de caminhos - usando caminhos absolutos
train_path = os.path.join(project_root, "data/03_5_feature_selection_final_treated/train.csv")
val_path = os.path.join(project_root, "data/03_5_feature_selection_final_treated/validation.csv")

# Diretório MLflow para tracking - usando caminho absoluto
mlflow_dir = os.path.join(project_root, "mlflow")
if not os.path.exists(mlflow_dir):
    os.makedirs(mlflow_dir, exist_ok=True)

# Diretório permanente para artefatos
artifact_dir = os.path.join(project_root, "models/mlflow_artifacts")
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir, exist_ok=True)

# Configurar MLflow corretamente
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_dir}"
mlflow.set_tracking_uri(f"file://{mlflow_dir}")
print(f"MLflow tracking URI configurado para: {mlflow.get_tracking_uri()}")

# Verificar experimento
experiment_name = "smart_ads_baseline"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Criado novo experimento: {experiment_name} (ID: {experiment_id})")
else:
    experiment_id = experiment.experiment_id
    print(f"Usando experimento existente: {experiment_name} (ID: {experiment_id})")

# Executar o treinamento
print(f"\nTreinando modelos usando dados de: {train_path}")
print(f"Artefatos serão salvos em: {artifact_dir}")

results = run_baseline_model_training(
    train_path=train_path,
    val_path=val_path,
    experiment_id=experiment_id,
    artifact_dir=artifact_dir,
    generate_learning_curves=False  # Manter como False, igual ao original
)

print("\nTreinamento de modelos baseline concluído.")

# Verificar se o Random Forest foi salvo
rf_model_uri = results.get('random_forest_model_uri')
if rf_model_uri:
    print(f"\nRandomForest model URI: {rf_model_uri}")
    
    # Opcional: Salvar o URI do modelo para referência futura
    model_uri_path = os.path.join(project_root, "models/rf_model_uri.txt")
    with open(model_uri_path, "w") as f:
        f.write(rf_model_uri)
else:
    print("\nAVISO: Não foi possível confirmar o salvamento do modelo RandomForest.")