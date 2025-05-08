import os
import sys
import tempfile
import mlflow
import shutil
import time  # Adicionado para resolver o erro anterior

# Caminho absoluto para o diretório raiz do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar módulos
from src.evaluation.baseline_model import run_baseline_model_training
from src.evaluation.mlflow_utils import setup_mlflow_tracking, get_data_hash

# Configuração de caminhos - DEFINIÇÃO EXPLÍCITA
train_path = "/Users/ramonmoreira/desktop/smart_ads/data/03_5_feature_selection_final_treated/train.csv"
val_path = "/Users/ramonmoreira/desktop/smart_ads/data/03_5_feature_selection_final_treated/validation.csv"

# Definir APENAS o tracking URI, não o artifact root
mlflow_dir = "/Users/ramonmoreira/desktop/smart_ads/mlflow"
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_dir}"

# REMOVER esta linha
# os.environ["MLFLOW_ARTIFACT_ROOT"] = f"file://{mlflow_dir}"

# Configurar MLflow
mlflow.set_tracking_uri(f"file://{mlflow_dir}")
print(f"MLflow tracking URI configurado para: {mlflow.get_tracking_uri()}")

# Diretório temporário para artefatos
temp_dir = tempfile.mkdtemp()
artifact_dir = temp_dir

# Verificar experimento com análise adicional
try:
    experiment = mlflow.get_experiment_by_name("smart_ads_baseline")
    if experiment is not None:
        experiment_id = experiment.experiment_id
        artifact_location = experiment.artifact_location
        print(f"Usando experimento existente: smart_ads_baseline (ID: {experiment_id})")
        print(f"Local dos artefatos: {artifact_location}")
        
        # Se o local dos artefatos estiver apontando para um local não padrão,
        # podemos recriá-lo para usar o padrão do MLflow
        if "/artifacts" in artifact_location and not artifact_location.endswith(f"/{experiment_id}/artifacts"):
            print("Experimento existente tem configuração de artefatos não padrão.")
            print("Excluindo e recriando experimento...")
            mlflow.delete_experiment(experiment_id)
            time.sleep(2)
            # Criar novo experimento sem especificar artifact_location
            experiment_id = mlflow.create_experiment("smart_ads_baseline")
            print(f"Experimento recriado com ID: {experiment_id}")
        else:
            print("Experimento existente possui configuração de artefatos padrão.")
    else:
        # Criar novo experimento sem especificar artifact_location
        experiment_id = mlflow.create_experiment("smart_ads_baseline")
        print(f"Criado novo experimento: smart_ads_baseline (ID: {experiment_id})")
except Exception as e:
    print(f"Erro ao configurar experimento: {e}")
    experiment_id = 0

# Executar o treinamento
try:
    results = run_baseline_model_training(
        train_path=train_path,
        val_path=val_path,
        experiment_id=experiment_id,
        artifact_dir=artifact_dir,
        generate_learning_curves=False
    )
    print("\nTreinamento de modelos baseline concluído.")
finally:
    try:
        shutil.rmtree(temp_dir)
        print(f"Diretório temporário removido: {temp_dir}")
    except Exception as e:
        print(f"Aviso: Não foi possível remover diretório temporário: {e}")