#!/usr/bin/env python
"""
Script para reconstruir o modelo GMM calibrado original, 
preservando o comportamento exato do modelo calibrado pelo script 10.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Adicionar caminho do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# Importar as classes compartilhadas
from src.modeling.gmm_wrapper import GMM_Wrapper
from src.modeling.calibrated_model import IdentityCalibratedModel

# Caminhos para os arquivos
PARAMS_DIR = os.path.join(project_root, "inference", "params")
MODELS_DIR = os.path.join(project_root, "models", "artifacts", "gmm_optimized")

def load_gmm_components():
    """
    Carrega todos os componentes do pipeline GMM.
    """
    print("Carregando componentes do pipeline GMM...")
    
    # Procurar componentes em diferentes diretórios
    possible_dirs = [PARAMS_DIR, MODELS_DIR]
    
    # Componentes a encontrar
    pca_model = None
    scaler_model = None
    gmm_model = None
    cluster_models = {}
    
    # Procurar PCA
    for dir_path in possible_dirs:
        # Procurar PCA
        pca_paths = [
            os.path.join(dir_path, "08_pca_model.joblib"),
            os.path.join(dir_path, "pca_model.joblib")
        ]
        for path in pca_paths:
            if os.path.exists(path):
                print(f"Encontrado PCA em: {path}")
                pca_model = joblib.load(path)
                break
        if pca_model is not None:
            break
    
    # Procurar Scaler
    for dir_path in possible_dirs:
        scaler_paths = [
            os.path.join(dir_path, "08_scaler_model.joblib"),
            os.path.join(dir_path, "scaler_model.joblib")
        ]
        for path in scaler_paths:
            if os.path.exists(path):
                print(f"Encontrado Scaler em: {path}")
                scaler_model = joblib.load(path)
                break
        if scaler_model is not None:
            break
    
    # Procurar GMM
    for dir_path in possible_dirs:
        gmm_paths = [
            os.path.join(dir_path, "gmm_model.joblib")
        ]
        for path in gmm_paths:
            if os.path.exists(path):
                print(f"Encontrado GMM em: {path}")
                gmm_model = joblib.load(path)
                break
        if gmm_model is not None:
            break
    
    # Procurar modelos de cluster
    if gmm_model is not None:
        n_clusters = gmm_model.n_components
        print(f"O modelo GMM tem {n_clusters} clusters")
        
        # Procurar modelos por cluster
        for cluster_id in range(n_clusters):
            for dir_path in possible_dirs:
                cluster_paths = [
                    os.path.join(dir_path, f"cluster_{cluster_id}_model.joblib")
                ]
                for path in cluster_paths:
                    if os.path.exists(path):
                        print(f"Encontrado modelo para cluster {cluster_id} em: {path}")
                        cluster_models[cluster_id] = {
                            'model': joblib.load(path),
                            'threshold': 0.5  # Threshold default
                        }
                        break
    
    # Verificar se encontramos todos os componentes necessários
    if None in [pca_model, scaler_model, gmm_model] or not cluster_models:
        missing = []
        if pca_model is None:
            missing.append("PCA")
        if scaler_model is None:
            missing.append("Scaler")
        if gmm_model is None:
            missing.append("GMM")
        if not cluster_models:
            missing.append("Modelos de cluster")
        
        raise ValueError(f"Componentes ausentes: {', '.join(missing)}")
    
    # Montar pipeline com o threshold original fixo (0.1)
    pipeline = {
        'pca_model': pca_model,
        'scaler_model': scaler_model,
        'gmm_model': gmm_model,
        'cluster_models': cluster_models,
        'threshold': 0.1  # Definindo explicitamente o threshold original
    }
    
    return pipeline

# Classe para emular o comportamento do modelo calibrado original
class IdentityCalibratedModel:
    """
    Classe que emula o CalibratedClassifierCV original com o threshold definido.
    """
    def __init__(self, base_estimator, threshold=0.1):
        self.base_estimator = base_estimator
        self.threshold = threshold
        
    def predict_proba(self, X):
        """
        Retorna as probabilidades do estimador base sem modificação.
        """
        return self.base_estimator.predict_proba(X)
    
    def predict(self, X):
        """
        Aplica o threshold às probabilidades.
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

def reconstruct_model():
    """
    Reconstrói o modelo calibrado original usando a classe GMM_Wrapper importada,
    preservando o threshold original de 0.1.
    """
    # Carregar componentes
    print("Carregando componentes do modelo...")
    try:
        pipeline = load_gmm_components()
    except Exception as e:
        print(f"ERRO ao carregar componentes: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Criar wrapper
    print("Criando wrapper com a classe importada e threshold original de 0.1...")
    gmm_wrapper = GMM_Wrapper(pipeline)
    
    # Garantir que o threshold é o original
    gmm_wrapper.threshold = 0.1
    print(f"Threshold definido como: {gmm_wrapper.threshold}")
    
    # Criar modelo "calibrado"
    print("Criando modelo calibrado emulado...")
    calibrated_model = IdentityCalibratedModel(gmm_wrapper, threshold=0.1)
    
    # Salvar modelo "calibrado"
    print("Salvando modelo calibrado reconstruído...")
    output_path = os.path.join(PARAMS_DIR, "10_gmm_calibrated_fixed.joblib")
    joblib.dump(calibrated_model, output_path)
    print(f"Modelo calibrado emulado salvo em: {output_path}")
    
    # Também salvar o wrapper base
    wrapper_path = os.path.join(PARAMS_DIR, "10_gmm_wrapper_fixed.joblib")
    joblib.dump(gmm_wrapper, wrapper_path)
    print(f"Wrapper base salvo em: {wrapper_path}")
    
    # Salvar threshold
    threshold_path = os.path.join(PARAMS_DIR, "10_threshold_fixed.txt")
    with open(threshold_path, 'w') as f:
        f.write("0.1")
    print(f"Threshold salvo em: {threshold_path}")

if __name__ == "__main__":
    reconstruct_model()