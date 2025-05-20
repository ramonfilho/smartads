#!/usr/bin/env python
"""
Script adaptado para inferência com o modelo GMM calibrado.
Função principal apply() recebe um DataFrame e retorna previsões calibradas.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import warnings

# Adicionar o diretório raiz do projeto ao path para importar módulos do projeto
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.append(project_root)

# A classe GMM_Wrapper não é mais necessária para o modelo portável

warnings.filterwarnings('ignore')

# Caminhos para os modelos calibrados e auxiliares
CALIBRATED_MODEL_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_portable"
CALIBRATED_MODEL_PATH = os.path.join(CALIBRATED_MODEL_DIR, "gmm_portable.joblib")
THRESHOLD_PATH = os.path.join(CALIBRATED_MODEL_DIR, "threshold.txt")

# Componentes auxiliares ainda são necessários
GMM_COMPONENTS_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/artifacts/gmm_optimized"

def load_calibrated_model(model_path=None, threshold_path=None):
    """
    Carrega o modelo GMM calibrado e o threshold otimizado.
    
    Args:
        model_path: Caminho para o modelo calibrado (se None, usa o padrão)
        threshold_path: Caminho para o threshold (se None, usa o padrão)
        
    Returns:
        Dicionário com modelo calibrado e threshold
    """
    if model_path is None:
        model_path = CALIBRATED_MODEL_PATH
    
    if threshold_path is None:
        threshold_path = THRESHOLD_PATH
    
    result = {}
    
    # Carregar modelo calibrado
    print(f"Carregando modelo GMM calibrado de: {model_path}")
    try:
        result['model'] = joblib.load(model_path)
        print("Modelo calibrado carregado com sucesso!")
    except Exception as e:
        print(f"ERRO ao carregar modelo calibrado: {e}")
        return None
    
    # Carregar threshold ótimo
    print(f"Carregando threshold de: {threshold_path}")
    try:
        with open(threshold_path, 'r') as f:
            result['threshold'] = float(f.read().strip())
        print(f"Threshold carregado: {result['threshold']}")
    except Exception as e:
        print(f"AVISO: Não foi possível carregar threshold otimizado: {e}")
        print("Usando threshold padrão de 0.5")
        result['threshold'] = 0.5
    
    return result

def apply(df, return_proba=True, threshold=None, model=None):
    """
    Aplica o modelo GMM calibrado para gerar previsões.
    
    Args:
        df: DataFrame processado pelas etapas anteriores
        return_proba: Se True, retorna probabilidades. Se False, retorna classes.
        threshold: Limiar personalizado para classificação (se None, usa o do modelo)
        model: Modelo pré-carregado (se None, será carregado do disco)
        
    Returns:
        Previsões (probabilidades ou classes)
    """
    print(f"Gerando previsões calibradas para {len(df)} instâncias...")
    
    # Carregar modelo calibrado se não fornecido
    if model is None:
        model_info = load_calibrated_model()
        if model_info is None:
            print("ERRO: Falha ao carregar modelo calibrado. Abortando previsões.")
            return None
        
        model = model_info['model']
        default_threshold = model_info['threshold']
    else:
        # Se modelo for fornecido, assumir que é um dicionário com modelo e threshold
        if isinstance(model, dict) and 'model' in model:
            default_threshold = model.get('threshold', 0.5)
            model = model['model']
        else:
            default_threshold = 0.5
    
    # Usar threshold fornecido ou o padrão carregado
    if threshold is None:
        threshold = default_threshold
    
    # Fazer previsões
    try:
        if return_proba:
            # Retornar probabilidades da classe positiva (índice 1)
            predictions = model.predict_proba(df)[:, 1]
            print(f"Probabilidades calibradas geradas com sucesso. Intervalo: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
            # Se threshold fornecido, também retornar classes
            classes = (predictions >= threshold).astype(int)
            return {
                'probability': predictions,
                'class': classes,
                'threshold': threshold
            }
            
        else:
            # Retornar classes usando o threshold especificado
            probas = model.predict_proba(df)[:, 1]
            predictions = (probas >= threshold).astype(int)
            print(f"Classes geradas com threshold: {threshold}")
            
            # Contar distribuição de classes
            class_counts = np.bincount(predictions.astype(int))
            class_pcts = class_counts / len(predictions) * 100
            
            print(f"Distribuição de classes:")
            for i, (count, pct) in enumerate(zip(class_counts, class_pcts)):
                if i < len(class_counts):
                    print(f"  Classe {i}: {count} ({pct:.2f}%)")
            
            return predictions
            
    except Exception as e:
        print(f"ERRO ao gerar previsões: {e}")
        return None

def analyze_predictions(df, predictions, threshold=None):
    """
    Analisa e adiciona as previsões ao DataFrame original.
    
    Args:
        df: DataFrame original
        predictions: Previsões geradas pelo modelo
        threshold: Threshold usado (para referência)
        
    Returns:
        DataFrame com previsões adicionadas
    """
    result_df = df.copy()
    
    # Verificar tipo de previsões
    if isinstance(predictions, dict):
        # Adicionamos probabilidades e classes
        result_df['prediction_probability'] = predictions['probability']
        result_df['prediction_class'] = predictions['class']
        
        # Contar distribuição de classes
        class_counts = np.bincount(predictions['class'].astype(int))
        class_pcts = class_counts / len(predictions['class']) * 100
        
        print(f"\nResultados com threshold {predictions['threshold']}:")
        for i, (count, pct) in enumerate(zip(class_counts, class_pcts)):
            if i < len(class_counts):
                print(f"  Classe {i}: {count} ({pct:.2f}%)")
                
    elif isinstance(predictions, np.ndarray) and len(predictions.shape) == 1:
        # Verificar se são probabilidades ou classes
        if np.all((predictions >= 0) & (predictions <= 1)) and not np.all(np.isin(predictions, [0, 1])):
            # São probabilidades
            result_df['prediction_probability'] = predictions
            
            # Se threshold fornecido, também calcular classes
            if threshold is not None:
                result_df['prediction_class'] = (predictions >= threshold).astype(int)
                
                # Contar distribuição de classes
                class_counts = np.bincount(result_df['prediction_class'].astype(int))
                class_pcts = class_counts / len(result_df['prediction_class']) * 100
                
                print(f"\nResultados com threshold {threshold}:")
                for i, (count, pct) in enumerate(zip(class_counts, class_pcts)):
                    if i < len(class_counts):
                        print(f"  Classe {i}: {count} ({pct:.2f}%)")
        else:
            # São classes
            result_df['prediction_class'] = predictions
            
            # Contar distribuição de classes
            class_counts = np.bincount(predictions.astype(int))
            class_pcts = class_counts / len(predictions) * 100
            
            print(f"\nDistribuição de classes:")
            for i, (count, pct) in enumerate(zip(class_counts, class_pcts)):
                if i < len(class_counts):
                    print(f"  Classe {i}: {count} ({pct:.2f}%)")
    
    return result_df

# Função para uso direto do script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aplicar modelo GMM calibrado em modo de inferência.")
    parser.add_argument("--input", type=str, required=True, 
                       help="Caminho para o arquivo CSV de entrada (já processado)")
    parser.add_argument("--output", type=str, required=True,
                       help="Caminho para salvar o arquivo CSV com previsões")
    parser.add_argument("--probabilities", action="store_true", default=True,
                       help="Retornar probabilidades em vez de apenas classes")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Threshold personalizado para classificação")
    
    args = parser.parse_args()
    
    # Carregar dados
    print(f"Carregando dados de: {args.input}")
    input_data = pd.read_csv(args.input)
    
    # Gerar previsões
    predictions = apply(
        input_data, 
        return_proba=args.probabilities,
        threshold=args.threshold
    )
    
    if predictions is not None:
        # Analisar e adicionar previsões ao DataFrame
        result_df = analyze_predictions(input_data, predictions, args.threshold)
        
        # Salvar resultado
        print(f"Salvando resultado em: {args.output}")
        result_df.to_csv(args.output, index=False)
        
        print("Processamento concluído!")