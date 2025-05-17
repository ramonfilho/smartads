#!/usr/bin/env python
"""
Script simplificado para investigar o comportamento do cluster 2.
Foca apenas em extrair informações básicas do modelo GMM.
"""
import os
import joblib
import pandas as pd
import numpy as np
import pickle

# Definições de classe necessárias para desserialização
class GMM_Wrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pca_model = pipeline['pca_model']
        self.gmm_model = pipeline['gmm_model']
        self.scaler_model = pipeline['scaler_model']
        self.cluster_models = pipeline['cluster_models']
        self.n_clusters = pipeline.get('n_clusters', 3)
        self.threshold = pipeline.get('threshold', 0.15)
        self.classes_ = np.array([0, 1])
        self._fitted = True
        self._estimator_type = "classifier"

class IdentityCalibratedModel:
    def __init__(self, base_estimator, threshold=0.1):
        self.base_estimator = base_estimator
        self.threshold = threshold

# Caminhos para arquivos
DATA_PATH = "/Users/ramonmoreira/desktop/smart_ads/data/02_3_processed_text_code6/test.csv"
MODEL_PATH = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250508_130725/gmm_calibrated.joblib"
MODEL_PATH_PIPELINE = "/Users/ramonmoreira/desktop/smart_ads/inference/params/10_gmm_calibrated.joblib"

# Função principal para investigação básica
def investigate_model_structure():
    """
    Investiga a estrutura básica dos modelos e imprime informações relevantes.
    """
    print("Investigando estrutura do modelo...")
    
    # Carregar modelo de referência
    print(f"Tentando carregar modelo de referência: {MODEL_PATH}")
    try:
        ref_model = joblib.load(MODEL_PATH)
        print("Modelo de referência carregado com sucesso!")
        
        # Inspecionar estrutura
        print("\nEstrutura do modelo de referência:")
        print(f"Tipo: {type(ref_model)}")
        
        # Tentar acessar calibrated_classifiers_
        if hasattr(ref_model, 'calibrated_classifiers_'):
            print(f"Número de calibradores: {len(ref_model.calibrated_classifiers_)}")
            calibrated_classifier = ref_model.calibrated_classifiers_[0]
            print(f"Tipo do calibrador [0]: {type(calibrated_classifier)}")
            
            # Tentar acessar estimador base
            if hasattr(calibrated_classifier, 'base_estimator'):
                base_estimator = calibrated_classifier.base_estimator
                print(f"Tipo do estimador base: {type(base_estimator)}")
            else:
                print("Calibrador não tem atributo 'base_estimator'")
                
                # Tentar acessar atributos diretamente
                for attr_name in dir(calibrated_classifier):
                    if not attr_name.startswith('_'):
                        try:
                            attr_value = getattr(calibrated_classifier, attr_name)
                            if isinstance(attr_value, (int, float, str, bool, list, dict, tuple)):
                                print(f"  Atributo {attr_name}: {attr_value}")
                        except:
                            pass
        
        # Se não tem calibrated_classifiers_, tentar acessar base_estimator diretamente
        elif hasattr(ref_model, 'base_estimator'):
            base_estimator = ref_model.base_estimator
            print(f"Tipo do estimador base: {type(base_estimator)}")
            
            # Verificar se é GMM_Wrapper
            if hasattr(base_estimator, 'cluster_models'):
                cluster_models = base_estimator.cluster_models
                print(f"Número de modelos de cluster: {len(cluster_models)}")
                print(f"IDs dos clusters: {list(cluster_models.keys())}")
                
                # Verificar especificamente o cluster 2
                if 2 in cluster_models:
                    cluster_2_model = cluster_models[2]
                    print(f"Tipo do modelo do cluster 2: {type(cluster_2_model)}")
                elif '2' in cluster_models:
                    cluster_2_model = cluster_models['2']
                    print(f"Tipo do modelo do cluster 2: {type(cluster_2_model)}")
                else:
                    print("Cluster 2 não encontrado nos modelos")
        else:
            print("Modelo não tem atributos 'calibrated_classifiers_' ou 'base_estimator'")
            
            # Tentar serializar e analisar como string
            print("\nTentando serializar modelo para análise...")
            try:
                serialized = pickle.dumps(ref_model)
                print(f"Modelo serializado com sucesso. Tamanho: {len(serialized)} bytes")
            except Exception as e:
                print(f"Erro ao serializar: {e}")
    
    except Exception as e:
        print(f"Erro ao carregar modelo de referência: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50 + "\n")
    
    # Agora verificar o modelo da pipeline
    print(f"Tentando carregar modelo da pipeline: {MODEL_PATH_PIPELINE}")
    try:
        pipe_model = joblib.load(MODEL_PATH_PIPELINE)
        print("Modelo da pipeline carregado com sucesso!")
        
        # Inspecionar estrutura
        print("\nEstrutura do modelo da pipeline:")
        print(f"Tipo: {type(pipe_model)}")
        
        # Verificar a mesma hierarquia que no modelo de referência
        if hasattr(pipe_model, 'calibrated_classifiers_'):
            print(f"Número de calibradores: {len(pipe_model.calibrated_classifiers_)}")
        elif hasattr(pipe_model, 'base_estimator'):
            base_estimator = pipe_model.base_estimator
            print(f"Tipo do estimador base: {type(base_estimator)}")
            
            # Verificar se é GMM_Wrapper
            if hasattr(base_estimator, 'cluster_models'):
                cluster_models = base_estimator.cluster_models
                print(f"Número de modelos de cluster: {len(cluster_models)}")
                print(f"IDs dos clusters: {list(cluster_models.keys())}")
                
                # Verificar especificamente o cluster 2
                if 2 in cluster_models:
                    cluster_2_model = cluster_models[2]
                    print(f"Tipo do modelo do cluster 2: {type(cluster_2_model)}")
                elif '2' in cluster_models:
                    cluster_2_model = cluster_models['2']
                    print(f"Tipo do modelo do cluster 2: {type(cluster_2_model)}")
                else:
                    print("Cluster 2 não encontrado nos modelos")
        else:
            print("Modelo não tem atributos 'calibrated_classifiers_' ou 'base_estimator'")
    
    except Exception as e:
        print(f"Erro ao carregar modelo da pipeline: {e}")
        import traceback
        traceback.print_exc()

# Execução do script
if __name__ == "__main__":
    investigate_model_structure()