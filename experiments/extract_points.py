#!/usr/bin/env python
"""
Script para extrair os pontos de calibração exatos do modelo de referência.
"""
import sys
import os
import joblib
import pickle
import numpy as np

# Caminho para o modelo de referência calibrado
MODEL_PATH = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250508_130725/gmm_calibrated.joblib"

def unpickle_model():
    """Tenta desserializar o modelo usando pickle diretamente."""
    try:
        print(f"Tentando abrir com pickle diretamente: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model_bytes = f.read()
        
        # Modificar a definição da classe no stream de bytes
        model_bytes = model_bytes.replace(
            b'__main__',
            b'extract_calibration'
        )
        
        # Definir a classe necessária para desserialização
        class GMM_Wrapper:
            """Stub para desserialização"""
            pass
        
        # Tentar desserializar
        model = pickle.loads(model_bytes)
        return model
    except Exception as e:
        print(f"Erro ao desserializar com pickle: {e}")
        return None

def print_calibration_details_from_log():
    """Extrai manualmente os detalhes do calibrador a partir do log."""
    print("\nDetalhes da calibração isotônica obtidos do log:")
    points = [
        (0.000000, 0.002611),
        (0.005000, 0.004571),
        (0.010000, 0.006745),
        (0.015000, 0.008924),
        (0.020000, 0.012144)
        # Aqui é onde precisaríamos de mais pontos
    ]
    
    print("Pontos de calibração encontrados:")
    for i, (x, y) in enumerate(points):
        print(f"Ponto {i}: {x:.6f} -> {y:.6f}")
    
    print("\nFormato para código:")
    print("EXACT_CALIBRATION_POINTS = [")
    for x, y in points:
        print(f"    ({x:.6f}, {y:.6f}),")
    print("]")

def create_sample_input_and_check_outputs():
    """Cria várias entradas de teste e compara os resultados entre os modelos."""
    # Valores de probabilidade para testar
    test_probs = np.linspace(0, 1, 20)
    
    print("\nTestando com valores simulados de probabilidade:")
    print("Probabilidade bruta -> Valor calibrado esperado (baseado em interpolação linear)")
    
    # Pontos de calibração conhecidos (do log)
    known_points = [
        (0.000000, 0.002611),
        (0.005000, 0.004571),
        (0.010000, 0.006745),
        (0.015000, 0.008924),
        (0.020000, 0.012144)
    ]
    
    # Interpolar para outros valores
    for prob in test_probs:
        # Encontrar os pontos de referência para interpolação
        if prob <= known_points[0][0]:
            calibrated = known_points[0][1]
        elif prob >= known_points[-1][0]:
            # Extrapolação linear para o último segmento
            last = known_points[-1]
            second_last = known_points[-2]
            slope = (last[1] - second_last[1]) / (last[0] - second_last[0])
            calibrated = last[1] + slope * (prob - last[0])
            calibrated = min(1.0, max(0.0, calibrated))
        else:
            # Interpolação linear entre pontos conhecidos
            for i in range(len(known_points) - 1):
                if known_points[i][0] <= prob <= known_points[i+1][0]:
                    x0, y0 = known_points[i]
                    x1, y1 = known_points[i+1]
                    calibrated = y0 + (y1 - y0) * (prob - x0) / (x1 - x0)
                    break
            else:
                calibrated = prob  # Fallback
        
        print(f"{prob:.4f} -> {calibrated:.6f}")

# Tentar extrair informações
try:
    # Tentar carregar o modelo diretamente (provavelmente falhará)
    try:
        print(f"Tentando carregar modelo com joblib: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print("Modelo carregado com sucesso!")
        
        # Tentar acessar detalhes do calibrador
        if hasattr(model, 'calibrated_classifiers_'):
            calibrators = model.calibrated_classifiers_
            print(f"Número de calibradores: {len(calibrators)}")
            
            for i, calibrator in enumerate(calibrators):
                if hasattr(calibrator, 'calibrators'):
                    print(f"Calibrador {i} tem {len(calibrator.calibrators)} regressores")
                    
                    for j, regressor in enumerate(calibrator.calibrators):
                        print(f"Regressor {j} tipo: {type(regressor)}")
                        
                        if hasattr(regressor, 'X_') and hasattr(regressor, 'y_'):
                            print(f"Pontos de calibração: {len(regressor.X_)}")
                            
                            print("\nPontos exatos:")
                            for k in range(len(regressor.X_)):
                                print(f"({regressor.X_[k]:.6f}, {regressor.y_[k]:.6f}),")
                        else:
                            print("Não foi possível acessar os pontos de calibração")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
    
    # Tentar desserializar com pickle
    model = unpickle_model()
    
    # Se tudo falhar, use os pontos do log
    print_calibration_details_from_log()
    
    # Criar valores simulados
    create_sample_input_and_check_outputs()
    
except Exception as e:
    print(f"Erro durante extração: {e}")