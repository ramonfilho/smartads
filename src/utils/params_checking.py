#!/usr/bin/env python
"""
Script para verificar a disponibilidade de todos os componentes 
necessários para a inferência com o modelo GMM.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Adicionar o diretório raiz do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.append(project_root)

# Importar a classe GMM_Wrapper para garantir que esteja disponível
try:
    from src.modeling.gmm_wrapper import GMM_Wrapper
    print("✅ GMM_Wrapper importado com sucesso.")
except ImportError as e:
    print(f"❌ Erro ao importar GMM_Wrapper: {e}")

# Definir caminhos para verificar
CALIBRATED_MODEL_PATHS = [
    # Modelo calibrado original
    "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250518_152543/gmm_calibrated.joblib",
    # Modelo portável mencionado no erro
    "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_portable/gmm_portable.joblib"
]

THRESHOLD_PATHS = [
    "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250518_152543/threshold.txt",
    "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_portable/threshold.txt"
]

# Componentes individuais
GMM_COMPONENTS_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/artifacts/gmm_optimized"
COMPONENT_PATHS = {
    "PCA Model": os.path.join(GMM_COMPONENTS_DIR, "pca_model.joblib"),
    "GMM Model": os.path.join(GMM_COMPONENTS_DIR, "gmm_model.joblib"),
    "Scaler Model": os.path.join(GMM_COMPONENTS_DIR, "scaler_model.joblib"),
    "GMM Wrapper": os.path.join(GMM_COMPONENTS_DIR, "gmm_wrapper.joblib"),
    "Evaluation Results": os.path.join(GMM_COMPONENTS_DIR, "evaluation_results.json"),
    "Experiment Config": os.path.join(GMM_COMPONENTS_DIR, "experiment_config.json")
}

# Verificar modelos por cluster
def check_cluster_models(base_dir, n_clusters=3):
    results = {}
    for cluster_id in range(n_clusters):
        model_path = os.path.join(base_dir, f"cluster_{cluster_id}_model.joblib")
        results[f"Cluster {cluster_id} Model"] = os.path.exists(model_path)
    return results

# Verificar arquivo
def check_file(file_path, try_load=False):
    if not os.path.exists(file_path):
        return {"exists": False, "size": 0, "loadable": False, "type": None}
    
    size = os.path.getsize(file_path) / (1024 * 1024)  # Em MB
    
    if not try_load:
        return {"exists": True, "size": size, "loadable": None, "type": None}
    
    try:
        loaded = joblib.load(file_path)
        return {
            "exists": True, 
            "size": size, 
            "loadable": True, 
            "type": type(loaded).__name__
        }
    except Exception as e:
        return {
            "exists": True, 
            "size": size, 
            "loadable": False, 
            "error": str(e)
        }

# Função principal
def main():
    print("\n===== VERIFICAÇÃO DE COMPONENTES DO MODELO GMM =====\n")
    
    # 1. Verificar modelos calibrados
    print("1. Verificando modelos GMM calibrados:")
    for model_path in CALIBRATED_MODEL_PATHS:
        result = check_file(model_path, try_load=True)
        status = "✅" if result["exists"] and result.get("loadable", False) else "❌"
        print(f"  {status} {model_path}")
        if result["exists"]:
            print(f"     Tamanho: {result['size']:.2f} MB")
            if result.get("loadable", False):
                print(f"     Tipo: {result['type']}")
            else:
                print(f"     Erro ao carregar: {result.get('error', 'Desconhecido')}")
    
    # 2. Verificar thresholds
    print("\n2. Verificando thresholds:")
    for threshold_path in THRESHOLD_PATHS:
        exists = os.path.exists(threshold_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {threshold_path}")
        if exists:
            try:
                with open(threshold_path, 'r') as f:
                    threshold = float(f.read().strip())
                print(f"     Valor: {threshold}")
            except Exception as e:
                print(f"     Erro ao ler threshold: {e}")
    
    # 3. Verificar componentes individuais
    print("\n3. Verificando componentes individuais:")
    for name, path in COMPONENT_PATHS.items():
        result = check_file(path, try_load=True)
        status = "✅" if result["exists"] and result.get("loadable", False) else "❌"
        print(f"  {status} {name}: {path}")
        if result["exists"]:
            print(f"     Tamanho: {result['size']:.2f} MB")
            if result.get("loadable", False):
                print(f"     Tipo: {result['type']}")
            else:
                print(f"     Erro ao carregar: {result.get('error', 'Desconhecido')}")
    
    # 4. Verificar modelos por cluster
    print("\n4. Verificando modelos por cluster:")
    cluster_results = check_cluster_models(GMM_COMPONENTS_DIR)
    for name, exists in cluster_results.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {name}")
        if exists:
            model_path = os.path.join(GMM_COMPONENTS_DIR, f"cluster_{name.split()[-2]}_model.joblib")
            result = check_file(model_path, try_load=True)
            if result["exists"]:
                print(f"     Tamanho: {result['size']:.2f} MB")
                if result.get("loadable", False):
                    print(f"     Tipo: {result['type']}")
                else:
                    print(f"     Erro ao carregar: {result.get('error', 'Desconhecido')}")
    
    print("\n===== VERIFICAÇÃO CONCLUÍDA =====")

if __name__ == "__main__":
    main()