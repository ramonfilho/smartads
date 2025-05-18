#!/usr/bin/env python
"""
Script para inspecionar o modelo GMM usado no script 11.
"""
# Adicionar no início do script compare_models.py (antes de importar joblib):
import numpy as np

class GMM_Wrapper:
    """
    Classe wrapper para o GMM que implementa a API sklearn para calibração.
    Esta definição é necessária apenas para desserialização.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pca_model = pipeline['pca_model']
        self.gmm_model = pipeline['gmm_model']
        self.scaler_model = pipeline['scaler_model']
        self.cluster_models = pipeline['cluster_models']
        self.n_clusters = pipeline.get('n_clusters', 3)
        self.threshold = pipeline.get('threshold', 0.15)
        
        # Adicionar atributos necessários para a API sklearn
        self.classes_ = np.array([0, 1])  # Classes binárias
        self._fitted = True  # Marcar como já ajustado
        self._estimator_type = "classifier"  # Indicar explicitamente que é um classificador

class IdentityCalibratedModel:
    """
    Classe que emula o CalibratedClassifierCV.
    Esta definição é necessária apenas para desserialização.
    """
    def __init__(self, base_estimator, threshold=0.1):
        self.base_estimator = base_estimator
        self.threshold = threshold

import joblib
import os

# Caminho para o modelo calibrado usado pelo script 11
MODEL_PATH = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250508_130725/gmm_calibrated.joblib"

def inspect_model(model, prefix=""):
    """Inspeciona a estrutura de um modelo."""
    print(f"{prefix}Tipo: {type(model)}")
    
    # Verificar se é um modelo de calibração
    if hasattr(model, 'base_estimator'):
        print(f"{prefix}Contém base_estimator: {type(model.base_estimator)}")
        inspect_model(model.base_estimator, prefix + "  ")
    
    # Verificar se é um modelo GMM_Wrapper
    if hasattr(model, 'cluster_models'):
        print(f"{prefix}Número de clusters: {len(model.cluster_models)}")
        print(f"{prefix}IDs dos clusters: {list(model.cluster_models.keys())}")
        
        # Verificar modelo por cluster
        for cluster_id, cluster_info in model.cluster_models.items():
            if isinstance(cluster_info, dict) and 'model' in cluster_info:
                print(f"{prefix}Modelo cluster {cluster_id}: {type(cluster_info['model'])}")
            else:
                print(f"{prefix}Cluster {cluster_id}: {type(cluster_info)}")

def main():
    print(f"Carregando modelo de: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        print("\nINSPEÇÃO DETALHADA DO MODELO DE REFERÊNCIA:")
        print("=" * 50)
        inspect_model(model, "MODELO: ")
        
        # Verificar threshold
        if hasattr(model, 'threshold'):
            print(f"\nThreshold no modelo: {model.threshold}")
        elif hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'threshold'):
            print(f"\nThreshold no estimador base: {model.base_estimator.threshold}")
        
        # Tentar carregar threshold de arquivo separado
        threshold_path = os.path.join(os.path.dirname(MODEL_PATH), "threshold.txt")
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold = float(f.read().strip())
                print(f"Threshold do arquivo: {threshold}")
                
        print("=" * 50)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()