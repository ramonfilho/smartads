#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para fazer previsões usando os modelos de detecção de anomalias treinados.

Este script carrega o melhor modelo e faz previsões em novos dados.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json

def load_model_and_params(models_dir):
    """
    Carrega modelo e parâmetros necessários para previsão.
    """
    # Verificar se o diretório existe
    if not os.path.exists(models_dir):
        raise ValueError(f"Diretório de modelos não encontrado: {models_dir}")
    
    # Procurar pelo melhor modelo individual
    json_files = [f for f in os.listdir(models_dir) if f.endswith('_params.json')]
    
    if not json_files:
        raise ValueError("Nenhum arquivo de parâmetros encontrado")
    
    # Ordenar para pegar o melhor modelo (assumindo que os nomes de arquivo são consistentes)
    json_file = sorted(json_files)[0]
    
    # Carregar parâmetros do modelo
    params_path = os.path.join(models_dir, json_file)
    model_name = json_file.replace('_params.json', '')
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Carregar modelo
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise ValueError(f"Modelo {model_name} não encontrado")
    
    model = joblib.load(model_path)
    
    # Carregar scaler
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        raise ValueError("Scaler não encontrado")
    
    # Carregar imputer
    imputer_path = os.path.join(models_dir, "feature_imputer.joblib")
    if os.path.exists(imputer_path):
        imputer = joblib.load(imputer_path)
    else:
        print("Aviso: Imputer não encontrado. Será usado preenchimento com 0.")
        imputer = None
    
    return {
        'model': model,
        'model_name': model_name,
        'params': params,
        'scaler': scaler,
        'imputer': imputer
    }

def predict_anomalies(data, model_info, feature_cols):
    """
    Faz previsões usando o modelo carregado.
    
    Args:
        data: DataFrame com os dados para previsão
        model_info: Dicionário com informações do modelo
        feature_cols: Lista de features usadas pelo modelo
    
    Returns:
        DataFrame com as previsões
    """
    # Verificar features disponíveis
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        print(f"Aviso: {len(missing_features)} features estão faltando nos dados: {missing_features[:5]}")
        
    # Selecionar apenas as features relevantes disponíveis
    available_features = [col for col in feature_cols if col in data.columns]
    X = data[available_features].copy()
    
    # Preencher valores NaN (usando imputer ou mediana)
    if model_info['imputer'] is not None:
        X_filled = model_info['imputer'].transform(X)
    else:
        X_filled = X.fillna(0).values
    
    # Normalizar os dados
    X_scaled = model_info['scaler'].transform(X_filled)
    
    # Fazer previsões
    model = model_info['model']
    threshold = model_info['params']['threshold']
    
    # Calcular scores de anomalia
    if hasattr(model, 'score_samples'):
        anomaly_scores = -model.score_samples(X_scaled)
    else:
        anomaly_scores = -model.decision_function(X_scaled)
    
    # Aplicar threshold
    predictions = (anomaly_scores >= threshold).astype(int)
    
    # Criar DataFrame de resultado
    result_df = data.copy()
    result_df['anomaly_score'] = anomaly_scores
    result_df['anomaly_prediction'] = predictions
    
    # Estatísticas básicas
    n_anomalies = predictions.sum()
    total = len(predictions)
    anomaly_rate = n_anomalies / total if total > 0 else 0
    
    print(f"Previsões concluídas: {n_anomalies} anomalias detectadas em {total} amostras ({anomaly_rate:.2%})")
    
    return result_df

def main():
    # Caminhos para os diretórios e arquivos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    features_path = os.path.join(base_dir, "selected_features.csv")
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print(f"Uso: {sys.argv[0]} <caminho_para_dados>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        print(f"Erro: Arquivo de dados não encontrado: {data_path}")
        sys.exit(1)
    
    # Carregar lista de features
    if os.path.exists(features_path):
        feature_cols = pd.read_csv(features_path).iloc[:, 0].tolist()
    else:
        print("Aviso: Lista de features não encontrada. Serão usadas todas as colunas numéricas.")
        feature_cols = None
    
    # Carregar modelo e parâmetros
    try:
        model_info = load_model_and_params(models_dir)
        print(f"Modelo carregado: {model_info['model_name']}")
        print(f"Threshold: {model_info['params']['threshold']:.4f}")
        print(f"Métricas no conjunto de validação: F1={model_info['params']['f1']:.4f}, "
              f"Precision={model_info['params']['precision']:.4f}, Recall={model_info['params']['recall']:.4f}")
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        sys.exit(1)
    
    # Carregar dados
    try:
        data = pd.read_csv(data_path)
        print(f"Dados carregados: {data.shape[0]} amostras, {data.shape[1]} colunas")
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        sys.exit(1)
    
    # Se não temos lista de features, usar todas as colunas numéricas
    if feature_cols is None:
        feature_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        print(f"Usando {len(feature_cols)} colunas numéricas como features")
    else:
        print(f"Usando {len(feature_cols)} features pré-selecionadas")
    
    # Fazer previsões
    result_df = predict_anomalies(data, model_info, feature_cols)
    
    # Salvar resultados
    output_path = data_path.replace('.csv', '_anomalias.csv')
    if output_path == data_path:
        output_path = os.path.join(os.path.dirname(data_path), f"anomalias_{os.path.basename(data_path)}")
    
    result_df.to_csv(output_path, index=False)
    print(f"Resultados salvos em: {output_path}")

if __name__ == "__main__":
    main()
