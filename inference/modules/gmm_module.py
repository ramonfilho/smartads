#!/usr/bin/env python
"""
Módulo para aplicar GMM e fazer predições no pipeline de inferência.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Adicionar caminho do projeto ao sys.path se necessário
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar as classes compartilhadas
try:
    from src.modeling.gmm_wrapper import GMM_Wrapper
    from src.modeling.calibrated_model import IdentityCalibratedModel
    print("Classes importadas com sucesso!")
except ImportError as e:
    print(f"AVISO: Erro ao importar classes: {e}")
    # Definir as classes localmente como fallback
    class GMM_Wrapper:
        """Implementação local de GMM_Wrapper..."""
        pass
    
    class IdentityCalibratedModel:
        """Implementação local de IdentityCalibratedModel..."""
        pass

# Diretório para parâmetros
PARAMS_DIR = os.path.join(project_root, "inference/params")

def load_calibrated_model(params_dir=PARAMS_DIR):
    """
    Carrega o modelo GMM calibrado fixado.
    
    Args:
        params_dir: Diretório com os parâmetros
        
    Returns:
        Tuple (modelo calibrado, threshold)
    """
    print("\nCarregando modelo GMM calibrado...")
    
    # Procurar o modelo reconstruído
    model_paths = [
        os.path.join(params_dir, "10_gmm_calibrated_fixed.joblib"),  # Modelo calibrado reconstruído (prioridade)
        os.path.join(params_dir, "10_gmm_calibrated.joblib")         # Modelo calibrado original
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Modelo calibrado encontrado: {model_path}")
            break
    
    if model_path is None:
        # Se não encontrou o modelo calibrado, tente o wrapper
        wrapper_path = os.path.join(params_dir, "10_gmm_wrapper_fixed.joblib")
        if os.path.exists(wrapper_path):
            print(f"Modelo calibrado não encontrado. Usando wrapper base: {wrapper_path}")
            model_path = wrapper_path
            # Carregar wrapper
            try:
                wrapper = joblib.load(wrapper_path)
                # Criar modelo calibrado na hora
                from src.modeling.calibrated_model import IdentityCalibratedModel
                model = IdentityCalibratedModel(wrapper, threshold=0.1)
                print(f"Criado modelo calibrado a partir do wrapper")
                return model, 0.1
            except Exception as e:
                print(f"Erro ao carregar wrapper: {e}")
        
        # Se não encontrou nenhum modelo, erro
        raise FileNotFoundError("Não foi possível encontrar o modelo calibrado em nenhum caminho esperado.")
    
    # Procurar o threshold
    threshold_paths = [
        os.path.join(params_dir, "10_threshold_fixed.txt"),
        os.path.join(params_dir, "10_threshold.txt")
    ]
    
    threshold = 0.1  # Default para o caso de não encontrar o arquivo
    for path in threshold_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    threshold = float(f.read().strip())
                print(f"Threshold carregado de {path}: {threshold}")
                break
            except Exception as e:
                print(f"AVISO: Erro ao ler threshold de {path}: {e}")
    
    # Carregar modelo calibrado
    print(f"  Carregando modelo de: {model_path}")
    try:
        model = joblib.load(model_path)
        print(f"  Modelo carregado com sucesso!")
        print(f"  Threshold: {threshold:.4f}")
        return model, threshold
    except Exception as e:
        print(f"  ERRO ao carregar modelo: {e}")
        import traceback
        print(traceback.format_exc())
        raise

def apply_gmm_and_predict(df, params_dir=PARAMS_DIR):
    """
    Aplica o modelo GMM calibrado para fazer predições.
    
    Args:
        df: DataFrame com features processadas
        params_dir: Diretório com parâmetros do modelo
        
    Returns:
        DataFrame com predições adicionadas
    """
    print("\n=== Aplicando GMM calibrado e fazendo predição ===")
    print(f"Processando {len(df)} amostras...")
    
    # Copiar o DataFrame para não modificar o original
    result_df = df.copy()
    
    try:
        # Carregar modelo calibrado
        calibrated_model, threshold = load_calibrated_model(params_dir)
        
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
        import traceback
        print(traceback.format_exc())
        
        # Em caso de erro, adicionar colunas com valores default
        result_df['prediction'] = 0
        result_df['probability'] = 0.0
    
    return result_df

if __name__ == "__main__":
    # Teste do módulo
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        try:
            # Testar carregamento do modelo
            model, threshold = load_calibrated_model()
            print(f"Modelo calibrado carregado com sucesso. Threshold: {threshold}")
            
            sys.exit(0)
        except Exception as e:
            print(f"Erro no teste: {e}")
            import traceback
            print(traceback.format_exc())
            sys.exit(1)
    
    # Pipeline completo
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Default para teste
        input_file = os.path.join(project_root, "inference/output/predictions_latest.csv")
        output_file = os.path.join(project_root, "inference/output/final_predictions.csv")
    
    # Verificar se o arquivo existe
    if not os.path.exists(input_file):
        print(f"ERRO: Arquivo de entrada não encontrado: {input_file}")
        sys.exit(1)
    
    # Carregar dados
    print(f"Carregando dados de: {input_file}")
    df = pd.read_csv(input_file)
    
    # Aplicar GMM e fazer predições
    result_df = apply_gmm_and_predict(df)
    
    # Salvar resultados se output_file especificado
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result_df.to_csv(output_file, index=False)
        print(f"Resultados salvos em: {output_file}")