import joblib
import os

# Caminho para o modelo calibrado
model_path = "/Users/ramonmoreira/desktop/smart_ads/inference/params/10_gmm_calibrated.joblib"

# Carregar o modelo
model = joblib.load(model_path)

# Verificar o tipo do modelo
print(f"Tipo do modelo: {type(model).__name__}")

# Verificar atributos
if hasattr(model, 'base_estimator'):
    print(f"Tipo do estimador base: {type(model.base_estimator).__name__}")
    
    # Se o base_estimator for o GMM_Wrapper
    if hasattr(model.base_estimator, 'pipeline'):
        # Verifica os componentes do pipeline
        pipeline = model.base_estimator.pipeline
        for key, value in pipeline.items():
            print(f"Pipeline contém: {key} ({type(value).__name__})")
        
        # Verifica os modelos por cluster
        if 'cluster_models' in pipeline:
            print(f"Número de modelos por cluster: {len(pipeline['cluster_models'])}")