#!/usr/bin/env python
"""
Módulo para carregar e aplicar o modelo GMM calibrado para inferência.
Esta versão foca apenas em carregar o modelo calibrado original.
"""

import os
import sys
import io
import pickle
import numpy as np
import pandas as pd
import joblib
import traceback
from datetime import datetime

# Caminho absoluto para o projeto
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"

# Caminhos para os modelos calibrados
MODEL_DIRS = [
    # Primeiro procurar no diretório de parâmetros de inferência
    os.path.join(PROJECT_ROOT, "inference/params"),
    # Depois nos diretórios de modelos calibrados
    os.path.join(PROJECT_ROOT, "models/calibrated/gmm_calibrated_20250508_130725"),
    os.path.join(PROJECT_ROOT, "models/calibrated"),
    # Por último no diretório de artefatos originais
    os.path.join(PROJECT_ROOT, "models/artifacts/gmm_optimized")
]

# Definir as classes necessárias para deserialização
class GMM_Wrapper:
    """
    Classe wrapper para o GMM que implementa a API sklearn para calibração.
    Esta definição é necessária apenas para deserialização.
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
        
    def fit(self, X, y):
        # Como o modelo já está treinado, apenas verificamos as classes
        self.classes_ = np.unique(y)
        self._fitted = True
        return self
        
    def predict_proba(self, X):
        # Implementação necessária para a interface, mas não será usada diretamente
        pass
    
    def predict(self, X):
        # Implementação necessária para a interface, mas não será usada diretamente
        pass


class IdentityCalibratedModel:
    """
    Classe que emula o CalibratedClassifierCV.
    Esta definição é necessária apenas para deserialização.
    """
    def __init__(self, base_estimator, threshold=0.1):
        self.base_estimator = base_estimator
        self.threshold = threshold
        
    def predict_proba(self, X):
        """Retorna as probabilidades do estimador base."""
        pass
    
    def predict(self, X):
        """Aplica o threshold às probabilidades."""
        pass


# Classe especial de unpickler para substituir classes
class ClassSubstitutingUnpickler(pickle.Unpickler):
    """
    Substituidor de classes durante a desserialização.
    Resolve problemas quando uma classe foi serializada em um módulo diferente.
    """
    def find_class(self, module, name):
        # Se a classe for GMM_Wrapper, usar a nossa versão local
        if name == 'GMM_Wrapper':
            return GMM_Wrapper
        # Se a classe for IdentityCalibratedModel, usar a nossa versão local
        elif name == 'IdentityCalibratedModel':
            return IdentityCalibratedModel
        # Para qualquer outra classe, tentar do modo normal
        return super().find_class(module, name)


def find_model_file(filename_patterns, dirs=MODEL_DIRS):
    """
    Procura um arquivo de modelo em múltiplos diretórios.
    
    Args:
        filename_patterns: Lista de possíveis nomes de arquivo
        dirs: Lista de diretórios onde procurar
        
    Returns:
        Caminho completo para o arquivo encontrado ou None
    """
    for dir_path in dirs:
        for pattern in filename_patterns:
            path = os.path.join(dir_path, pattern)
            if os.path.exists(path):
                return path
    return None


def load_joblib_with_class_substitution(filepath):
    """
    Carrega um arquivo joblib substituindo classes problemáticas.
    
    Args:
        filepath: Caminho para o arquivo joblib
        
    Returns:
        Objeto deserializado
    """
    print(f"Carregando arquivo com substituição de classes: {filepath}")
    try:
        # Primeiro tentar carregar normalmente (para casos simples)
        return joblib.load(filepath)
    except (AttributeError, ImportError, ModuleNotFoundError) as e:
        print(f"Erro ao carregar com joblib padrão: {e}")
        print("Tentando carregar com unpickler personalizado...")
        
        # Usar nosso unpickler personalizado
        with open(filepath, 'rb') as f:
            unpickler = ClassSubstitutingUnpickler(f)
            return unpickler.load()


def load_calibrated_model():
    """
    Carrega o modelo GMM calibrado com tratamento robusto para problemas de desserialização.
    
    Returns:
        Tuple (modelo calibrado, threshold)
    """
    print("\nCarregando modelo GMM calibrado...")
    
    # Lista de possíveis arquivos de modelo
    model_paths = [
        find_model_file(["gmm_calibrated.joblib"]),
        find_model_file(["10_gmm_calibrated.joblib"]),
        find_model_file(["10_gmm_calibrated_fixed.joblib"])
    ]
    model_paths = [p for p in model_paths if p]  # Remover None
    
    if not model_paths:
        raise FileNotFoundError("Modelo calibrado não encontrado em nenhum dos diretórios esperados!")
    
    # Tentar carregar modelo com unpickler personalizado
    for model_path in model_paths:
        try:
            print(f"Tentando carregar modelo de: {model_path}")
            calibrated_model = load_joblib_with_class_substitution(model_path)
            
            # Carregar threshold
            threshold_path = find_model_file([
                "threshold.txt", 
                "10_threshold.txt",
                "10_threshold_fixed.txt"
            ])
            
            threshold = 0.1  # Default
            if threshold_path:
                try:
                    with open(threshold_path, 'r') as f:
                        threshold = float(f.read().strip())
                    print(f"Threshold: {threshold}")
                except Exception as e:
                    print(f"AVISO: Usando threshold default {threshold}: {e}")
            elif hasattr(calibrated_model, 'threshold'):
                threshold = calibrated_model.threshold
                print(f"Threshold do modelo: {threshold}")
            
            # Verificar se o modelo tem a interface esperada
            if hasattr(calibrated_model, 'predict_proba') and hasattr(calibrated_model, 'predict'):
                print("Modelo calibrado carregado com sucesso!")
                return calibrated_model, threshold
            else:
                raise ValueError("Objeto carregado não tem a interface esperada")
                
        except Exception as e:
            print(f"Erro ao carregar modelo de {model_path}: {e}")
            print(traceback.format_exc())
    
    # Se todas as tentativas falharem, não temos alternativa
    raise RuntimeError("Falha ao carregar o modelo GMM calibrado. Por favor, verifique a disponibilidade e integridade do modelo.")


def apply_gmm_and_predict(df, params_dir=None):
    """
    Aplica o modelo GMM calibrado para fazer predições.
    
    Args:
        df: DataFrame com features processadas
        params_dir: Diretório com parâmetros (opcional)
        
    Returns:
        DataFrame com predições adicionadas
    """
    print("\n=== Aplicando GMM calibrado e fazendo predição ===")
    print(f"Processando {len(df)} amostras...")
    
    # Copiar o DataFrame para não modificar o original
    result_df = df.copy()
    
    try:
        # Carregar modelo calibrado
        calibrated_model, threshold = load_calibrated_model()
        
        # Fazer predições
        print("Fazendo predições...")
        start_time = datetime.now()
        
        # Calcular probabilidades
        probabilities = calibrated_model.predict_proba(df)[:, 1]
        
        # Aplicar threshold para classes
        predictions = (probabilities >= threshold).astype(int)
        
        # Adicionar ao DataFrame
        result_df['probability'] = probabilities
        result_df['prediction'] = predictions
        
        # Calcular estatísticas
        prediction_counts = dict(zip(*np.unique(predictions, return_counts=True)))
        print(f"  Distribuição de predições: {prediction_counts}")
        
        if 1 in prediction_counts:
            positive_rate = prediction_counts[1] / len(df)
            print(f"  Taxa de positivos: {positive_rate:.4f} ({prediction_counts.get(1, 0)} de {len(df)})")
        
        # Tempo de processamento
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"Predições concluídas em {elapsed_time:.2f} segundos.")
        
    except Exception as e:
        print(f"  ERRO durante predição: {e}")
        print(traceback.format_exc())
        raise RuntimeError(f"Falha ao fazer predições com o modelo calibrado: {str(e)}")
    
    return result_df


if __name__ == "__main__":
    # Teste do módulo
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        try:
            # Testar carregamento do modelo
            model, threshold = load_calibrated_model()
            print(f"Modelo calibrado carregado com sucesso. Threshold: {threshold}")
            
            # Criar DataFrame de teste pequeno
            test_df = pd.DataFrame({
                'feature1': [0.1, 0.2, 0.3],
                'feature2': [1.0, 2.0, 3.0]
            })
            
            # Testar predição
            result = apply_gmm_and_predict(test_df)
            print(f"Predição de teste bem-sucedida. Forma: {result.shape}")
            print(result[['probability', 'prediction']].head())
            
            sys.exit(0)
        except Exception as e:
            print(f"Erro no teste: {e}")
            print(traceback.format_exc())
            sys.exit(1)