#!/usr/bin/env python
"""
Script para carregar os componentes do modelo GMM e recalibrar para usar
com a classe GMM_Wrapper importada.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

# Adicionar caminho do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# Importar a classe GMM_Wrapper do módulo compartilhado
from src.modeling.gmm_wrapper import GMM_Wrapper

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
    
    # Procurar PCA e Scaler
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
    
    # Carregar threshold de avaliação
    threshold = 0.5  # Default
    for dir_path in possible_dirs:
        threshold_paths = [
            os.path.join(dir_path, "10_threshold.txt"),
            os.path.join(dir_path, "evaluation_results.json")
        ]
        for path in threshold_paths:
            if os.path.exists(path):
                try:
                    if path.endswith('.txt'):
                        with open(path, 'r') as f:
                            threshold = float(f.read().strip())
                            print(f"Carregado threshold do arquivo {path}: {threshold}")
                    elif path.endswith('.json'):
                        import json
                        with open(path, 'r') as f:
                            eval_results = json.load(f)
                            if 'threshold' in eval_results:
                                threshold = float(eval_results['threshold'])
                                print(f"Carregado threshold do arquivo JSON {path}: {threshold}")
                    break
                except Exception as e:
                    print(f"Erro ao carregar threshold de {path}: {e}")
    
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
    
    # Montar pipeline
    pipeline = {
        'pca_model': pca_model,
        'scaler_model': scaler_model,
        'gmm_model': gmm_model,
        'cluster_models': cluster_models,
        'threshold': threshold
    }
    
    return pipeline

def recalibrate():
    """
    Recalibra o modelo usando a classe GMM_Wrapper importada.
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
    print("Criando wrapper com a classe importada...")
    gmm_wrapper = GMM_Wrapper(pipeline)
    
    # Verificar se temos dados para calibração
    validation_path = os.path.join(project_root, "data", "02_3_processed", "validation.csv")
    if not os.path.exists(validation_path):
        print(f"AVISO: Arquivo de validação não encontrado: {validation_path}")
        # Buscar alternativas
        possible_paths = [
            os.path.join(project_root, "data", "02_3_processed_text", "validation.csv"),
            os.path.join(project_root, "data", "02_3_processed_text_code6", "validation.csv"),
            os.path.join(project_root, "data", "02_fixed_processed", "validation.csv")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                validation_path = path
                print(f"Usando arquivo alternativo: {validation_path}")
                break
        else:
            print("Nenhum arquivo de validação encontrado. Criando modelo sem calibração.")
            # Simplesmente salvar o wrapper
            output_path = os.path.join(PARAMS_DIR, "10_gmm_calibrated_fixed.joblib")
            joblib.dump(gmm_wrapper, output_path)
            print(f"Modelo wrapper salvo em: {output_path}")
            return
    
    # Carregar dados de validação para calibração
    print(f"Carregando dados de validação de: {validation_path}")
    val_df = pd.read_csv(validation_path)
    
    # Separar features e target
    y_val = val_df['target'] if 'target' in val_df.columns else None
    X_val = val_df.drop(columns=['target']) if 'target' in val_df.columns else val_df
    
    if y_val is None:
        print("AVISO: Coluna 'target' não encontrada. Criando modelo sem calibração.")
        # Simplesmente salvar o wrapper
        output_path = os.path.join(PARAMS_DIR, "10_gmm_calibrated_fixed.joblib")
        joblib.dump(gmm_wrapper, output_path)
        print(f"Modelo wrapper salvo em: {output_path}")
        return
    
    # Criar modelo calibrado
    print("Criando modelo calibrado...")
    calibrated_model = CalibratedClassifierCV(
        estimator=gmm_wrapper,
        method='isotonic',  # ou 'sigmoid'
        cv='prefit'  # O modelo já está ajustado
    )
    
    # Ajustar o calibrador
    print("Ajustando o calibrador ao conjunto de validação...")
    try:
        calibrated_model.fit(X_val, y_val)
        
        # Salvar modelo calibrado
        output_path = os.path.join(PARAMS_DIR, "10_gmm_calibrated_fixed.joblib")
        joblib.dump(calibrated_model, output_path)
        print(f"Modelo calibrado salvo em: {output_path}")
        
        # Também salvar o wrapper não calibrado como backup
        wrapper_path = os.path.join(PARAMS_DIR, "10_gmm_wrapper_fixed.joblib")
        joblib.dump(gmm_wrapper, wrapper_path)
        print(f"Wrapper salvo em: {wrapper_path}")
        
        # Salvar threshold
        threshold_path = os.path.join(PARAMS_DIR, "10_threshold_fixed.txt")
        with open(threshold_path, 'w') as f:
            f.write(str(gmm_wrapper.threshold))
        print(f"Threshold salvo em: {threshold_path}")
        
    except Exception as e:
        print(f"ERRO durante calibração: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Em caso de erro, salvar apenas o wrapper
        output_path = os.path.join(PARAMS_DIR, "10_gmm_wrapper_fixed.joblib")
        joblib.dump(gmm_wrapper, output_path)
        print(f"Modelo wrapper (não calibrado) salvo em: {output_path}")

if __name__ == "__main__":
    recalibrate()