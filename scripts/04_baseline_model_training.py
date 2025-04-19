import os
import sys
import mlflow

# Caminho absoluto para o diretório raiz do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Adicionar o caminho ao sys.path
sys.path.insert(0, project_root)

# Agora tentar importar o módulo
from src.evaluation.baseline_model import run_baseline_model_training
from src.evaluation.mlflow_utils import setup_mlflow_tracking, get_data_hash

# Resto do código...

# 1. Configuração de caminhos
base_dir = os.path.expanduser("~")
train_path = os.path.join(base_dir, "desktop/smart_ads/data/feature_selection/train.csv")
val_path = os.path.join(base_dir, "desktop/smart_ads/data/feature_selection/validation.csv")
mlflow_dir = os.path.join(base_dir, "desktop/smart_ads/models/mlflow")
artifact_dir = os.path.join(base_dir, "desktop/smart_ads/models/artifacts")

# 2. Configurar MLflow
experiment_id = setup_mlflow_tracking(
    tracking_dir=mlflow_dir,
    experiment_name="smart_ads_baseline",
    clean_previous=False
)

# 3. Executar o treinamento do modelo baseline
# A função abaixo irá:
# - Carregar e preparar dados
# - Treinar diferentes modelos (RandomForest, LightGBM, XGBoost)
# - Avaliar e selecionar o melhor modelo
# - Fazer o tracking com MLflow
# - Salvar modelos e métricas
results = run_baseline_model_training(
    train_path=train_path,
    val_path=val_path,
    experiment_id=experiment_id,
    artifact_dir=artifact_dir,
    generate_learning_curves=False  # Desativado para economizar tempo
)

# 4. Exibir resultados finais
print("\nTreinamento de modelos baseline concluído.")