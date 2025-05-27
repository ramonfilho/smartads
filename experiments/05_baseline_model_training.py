#!/usr/bin/env python
"""
Script para treinar modelos baseline com datasets já filtrados.
"""

import os
import sys
import mlflow

# Caminho absoluto para o diretório raiz do projeto
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# Importar módulos
from src.evaluation.baseline_model import run_baseline_model_training

# Configuração de caminhos
train_path = os.path.join(project_root, "data/new/04_feature_selection/train.csv")
val_path = os.path.join(project_root, "data/new/04_feature_selection/validation.csv")

# Diretório MLflow para tracking
mlflow_dir = os.path.join(project_root, "mlflow")
if not os.path.exists(mlflow_dir):
    os.makedirs(mlflow_dir, exist_ok=True)

# Diretório permanente para artefatos
artifact_dir = os.path.join(project_root, "models/mlflow_artifacts")
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir, exist_ok=True)

# Configurar MLflow
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
print("\nNOTA: Os datasets já contêm apenas as 300 features selecionadas.")

results = run_baseline_model_training(
    train_path=train_path,
    val_path=val_path,
    experiment_id=experiment_id,
    artifact_dir=artifact_dir,
    generate_learning_curves=False
)

print("\nTreinamento de modelos baseline concluído.")

# Verificar se o Random Forest foi salvo
rf_model_uri = results.get('random_forest_model_uri')
if rf_model_uri:
    print(f"\nRandomForest model URI: {rf_model_uri}")
    
    # Salvar o URI do modelo para referência futura
    model_uri_path = os.path.join(project_root, "models/rf_model_uri.txt")
    with open(model_uri_path, "w") as f:
        f.write(rf_model_uri)
else:
    print("\nAVISO: Não foi possível confirmar o salvamento do modelo RandomForest.")

# Resumo final
print("\n" + "="*60)
print("RESUMO FINAL DOS MODELOS")
print("="*60)
for model_name in ['random_forest', 'lightgbm', 'xgboost']:
    if f"{model_name}_f1" in results:
        print(f"\n{model_name.upper()}:")
        print(f"  F1-Score: {results[f'{model_name}_f1']:.4f}")
        print(f"  Precisão: {results[f'{model_name}_precision']:.4f}")
        print(f"  Recall: {results[f'{model_name}_recall']:.4f}")
        print(f"  AUC: {results[f'{model_name}_auc']:.4f}")
        print(f"  PR-AUC: {results[f'{model_name}_pr_auc']:.4f}")
        print(f"  Threshold: {results[f'{model_name}_threshold']:.4f}")
print("="*60)