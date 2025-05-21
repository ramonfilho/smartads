#!/usr/bin/env python
"""
Script para reconstruir o modelo GMM calibrado em um formato mais portável.
Este script carrega o modelo atual, cria um wrapper simples, e salva em um novo formato.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

# Adicionar o diretório raiz do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.append(project_root)

# Importar a classe GMM_Wrapper
from src.modeling.gmm_wrapper import GMM_Wrapper

# Caminhos para arquivos
CALIBRATED_MODEL_PATH = os.path.join(project_root, "models/calibrated/gmm_calibrated_20250518_152543/gmm_calibrated.joblib")
THRESHOLD_PATH = os.path.join(project_root, "models/calibrated/gmm_calibrated_20250518_152543/threshold.txt")
NEW_MODEL_DIR = os.path.join(project_root, "models/calibrated/gmm_portable")
NEW_MODEL_PATH = os.path.join(NEW_MODEL_DIR, "gmm_portable.joblib")

# Criar nova classe portável para substituir GMM_Wrapper
class PortableGMMClassifier(BaseEstimator, ClassifierMixin):
    """
    Classe portável para substituir GMM_Wrapper. Implementa a interface sklearn.
    Esta classe serve como um proxy para o modelo calibrado original.
    """
    def __init__(self, calibrated_model=None):
        self.calibrated_model = calibrated_model
        self.classes_ = np.array([0, 1])
    
    def predict_proba(self, X):
        """Retorna probabilidades de classificação."""
        if self.calibrated_model is None:
            raise ValueError("Modelo não foi inicializado.")
        return self.calibrated_model.predict_proba(X)
    
    def predict(self, X, threshold=0.5):
        """Retorna classes preditas."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

def main():
    """Função principal para reconstruir o modelo."""
    print("=== Reconstruindo o Modelo GMM Calibrado ===")
    
    # Criar diretório para o novo modelo
    os.makedirs(NEW_MODEL_DIR, exist_ok=True)
    
    # Carregar o modelo calibrado atual
    print(f"Carregando modelo calibrado de: {CALIBRATED_MODEL_PATH}")
    try:
        calibrated_model = joblib.load(CALIBRATED_MODEL_PATH)
        print("Modelo calibrado carregado com sucesso!")
    except Exception as e:
        print(f"ERRO ao carregar modelo calibrado: {e}")
        return False
    
    # Carregar threshold
    try:
        with open(THRESHOLD_PATH, 'r') as f:
            threshold = float(f.read().strip())
        print(f"Threshold carregado: {threshold}")
    except Exception as e:
        print(f"AVISO: Não foi possível carregar threshold: {e}")
        threshold = 0.5
        print(f"Usando threshold padrão: {threshold}")
    
    # Criar modelo portável
    print("Criando modelo portável...")
    portable_model = PortableGMMClassifier(calibrated_model)
    
    # Salvar modelo portável e threshold
    print(f"Salvando modelo portável em: {NEW_MODEL_PATH}")
    joblib.dump(portable_model, NEW_MODEL_PATH)
    
    # Salvar threshold
    threshold_path = os.path.join(NEW_MODEL_DIR, "threshold.txt")
    with open(threshold_path, 'w') as f:
        f.write(str(threshold))
    
    print("\nModelo portável criado com sucesso!")
    print(f"Novo modelo: {NEW_MODEL_PATH}")
    print(f"Novo threshold: {threshold_path}")
    
    # Verificar se o modelo pode ser carregado novamente
    print("\nVerificando se o modelo pode ser carregado corretamente...")
    try:
        loaded_model = joblib.load(NEW_MODEL_PATH)
        print(f"Modelo carregado com sucesso! Tipo: {type(loaded_model).__name__}")
        
        # Verificar se tem os métodos necessários
        assert hasattr(loaded_model, 'predict_proba'), "Modelo não tem método predict_proba"
        assert hasattr(loaded_model, 'predict'), "Modelo não tem método predict"
        print("Modelo tem todos os métodos necessários!")
        
        print("\nModelo reconstruído com sucesso e pronto para uso!")
        return True
    except Exception as e:
        print(f"ERRO ao verificar modelo portável: {e}")
        return False

# Atualizar o script inference_05_gmm_predict.py para usar o novo modelo
def update_inference_script():
    """Atualiza o script de inferência para usar o novo modelo."""
    inference_file = os.path.join(project_root, "inference_v4/inference_05_gmm_predict.py")
    
    print(f"\nAtualizando script de inferência: {inference_file}")
    
    # Ler conteúdo atual do arquivo
    with open(inference_file, 'r') as f:
        content = f.read()
    
    # Substituir caminhos para usar o novo modelo
    new_content = content.replace(
        'CALIBRATED_MODEL_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250518_152543"',
        'CALIBRATED_MODEL_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_portable"'
    )
    
    new_content = new_content.replace(
        'CALIBRATED_MODEL_PATH = os.path.join(CALIBRATED_MODEL_DIR, "gmm_calibrated.joblib")',
        'CALIBRATED_MODEL_PATH = os.path.join(CALIBRATED_MODEL_DIR, "gmm_portable.joblib")'
    )
    
    # Remover importação da GMM_Wrapper (não é mais necessária)
    new_content = new_content.replace(
        '# Importar a classe GMM_Wrapper para garantir que esteja disponível durante o carregamento\nfrom src.modeling.gmm_wrapper import GMM_Wrapper\n',
        '# A classe GMM_Wrapper não é mais necessária para o modelo portável\n'
    )
    
    # Salvar mudanças
    with open(inference_file, 'w') as f:
        f.write(new_content)
    
    print("Script de inferência atualizado com sucesso!")

if __name__ == "__main__":
    if main():
        update_inference_script()
        print("\nTudo pronto! Agora você pode executar a pipeline de inferência normalmente.")
        print("Use o comando:")
        print("python inference_v4/inference_pipeline.py --input <seu_arquivo_entrada.csv> --output <seu_arquivo_saida.csv>")
    else:
        print("\nFalha ao reconstruir o modelo. Verifique os erros acima.")