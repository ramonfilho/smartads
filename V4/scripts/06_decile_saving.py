#!/usr/bin/env python
"""
Script para calcular e salvar os limiares de decis baseados nos dados de treino.
Deve ser executado uma vez após o treino do modelo.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import pickle
import re

# Adicionar o diretório raiz do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

def clean_column_names(df):
    """Limpa nomes de colunas para LightGBM."""
    rename_dict = {}
    for col in df.columns:
        clean_name = re.sub(r'[^\w\s]', '_', str(col))
        clean_name = re.sub(r'\s+', '_', clean_name)
        clean_name = re.sub(r'__+', '_', clean_name)
        clean_name = clean_name.strip('_')
        
        if clean_name != col:
            rename_dict[col] = clean_name
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    return df

def calculate_and_save_thresholds():
    """Calcula e salva os limiares de decis baseados no conjunto de treino."""
    
    print("=== CALCULANDO LIMIARES DE DECIS PARA PRODUÇÃO ===\n")
    
    # 1. Carregar modelo
    model_path = os.path.join(project_root, "models/artifacts/lightgbm_direct_ranking.joblib")
    print(f"Carregando modelo de: {model_path}")
    model = joblib.load(model_path)
    
    # 2. Carregar dados de treino
    train_path = os.path.join(project_root, "data/new/04_feature_selection/train.csv")
    print(f"Carregando dados de treino de: {train_path}")
    train_df = pd.read_csv(train_path)
    
    # Limpar nomes de colunas
    train_df = clean_column_names(train_df)
    
    # Separar features e target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    print(f"Shape dos dados: {X_train.shape}")
    print(f"Taxa de conversão: {y_train.mean():.4f}")
    
    # 3. Predizer probabilidades
    print("\nCalculando probabilidades para todo o conjunto de treino...")
    probabilities = model.predict_proba(X_train)[:, 1]
    
    # 4. Calcular estatísticas
    print(f"\nEstatísticas das probabilidades:")
    print(f"  - Mínima: {probabilities.min():.6f}")
    print(f"  - Máxima: {probabilities.max():.6f}")
    print(f"  - Média: {probabilities.mean():.6f}")
    print(f"  - Mediana: {np.median(probabilities):.6f}")
    print(f"  - Desvio padrão: {probabilities.std():.6f}")
    
    # 5. Calcular limiares de decis
    percentiles = np.arange(10, 100, 10)
    thresholds = np.percentile(probabilities, percentiles)
    
    print(f"\nLIMIARES DE DECIS:")
    print("-" * 50)
    for i, (perc, thresh) in enumerate(zip(percentiles, thresholds)):
        print(f"Decil {i+1} → {i+2}: probabilidade > {thresh:.6f} (percentil {perc})")
    print(f"Decil 10: probabilidade > {thresholds[-1]:.6f}")
    print("-" * 50)
    
    # 6. Validar distribuição
    print("\nVALIDANDO DISTRIBUIÇÃO:")
    deciles = pd.cut(probabilities, 
                     bins=[-np.inf] + list(thresholds) + [np.inf], 
                     labels=range(1, 11))
    
    decile_counts = deciles.value_counts().sort_index()
    print("\nContagem por decil:")
    for decil, count in decile_counts.items():
        pct = count / len(probabilities) * 100
        print(f"  Decil {decil}: {count:,} ({pct:.1f}%)")
    
    # 7. Performance por decil
    print("\nPERFORMANCE POR DECIL:")
    performance_df = pd.DataFrame({
        'probability': probabilities,
        'target': y_train,
        'decile': deciles
    })
    
    decile_stats = performance_df.groupby('decile').agg({
        'target': ['sum', 'count', 'mean'],
        'probability': ['mean', 'min', 'max']
    }).round(6)
    
    decile_stats.columns = ['conversions', 'total', 'conv_rate', 'avg_prob', 'min_prob', 'max_prob']
    
    # Calcular lift
    overall_rate = y_train.mean()
    decile_stats['lift'] = decile_stats['conv_rate'] / overall_rate
    
    print(decile_stats)
    
    # 8. Salvar limiares
    output_dir = os.path.join(project_root, "models/artifacts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar como pickle
    thresholds_path = os.path.join(output_dir, "decile_thresholds.pkl")
    with open(thresholds_path, 'wb') as f:
        pickle.dump(thresholds, f)
    print(f"\n✅ Limiares salvos em: {thresholds_path}")
    
    # Salvar também como JSON para facilitar inspeção
    import json
    thresholds_dict = {
        'thresholds': thresholds.tolist(),
        'percentiles': percentiles.tolist(),
        'statistics': {
            'min_probability': float(probabilities.min()),
            'max_probability': float(probabilities.max()),
            'mean_probability': float(probabilities.mean()),
            'median_probability': float(np.median(probabilities))
        },
        'performance_by_decile': decile_stats.to_dict()
    }
    
    json_path = os.path.join(output_dir, "decile_thresholds.json")
    with open(json_path, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"✅ Limiares também salvos em JSON: {json_path}")
    
    # 9. Criar arquivo de configuração para produção
    config = {
        'model_path': 'models/artifacts/lightgbm_direct_ranking.joblib',
        'thresholds_path': 'models/artifacts/decile_thresholds.pkl',
        'feature_columns': list(X_train.columns),
        'model_version': '1.0',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'training_samples': len(X_train),
        'conversion_rate': float(y_train.mean()),
        'top_decile_lift': float(decile_stats.loc[10, 'lift'])
    }
    
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Configuração do modelo salva em: {config_path}")
    
    return thresholds, decile_stats


if __name__ == "__main__":
    thresholds, performance = calculate_and_save_thresholds()
    
    print("\n\n=== RESUMO PARA PRODUÇÃO ===")
    print(f"\n1. Para usar em produção:")
    print(f"   - Carregar modelo: models/artifacts/lightgbm_direct_ranking.joblib")
    print(f"   - Carregar limiares: models/artifacts/decile_thresholds.pkl")
    print(f"   - Usar a classe DecilePredictor do script production_decile_predictor.py")
    
    print(f"\n2. Performance esperada:")
    print(f"   - Decil 10 (top 10%): Lift de {performance.loc[10, 'lift']:.2f}x")
    print(f"   - Decil 9-10 (top 20%): Lift médio de {performance.loc[[9,10], 'lift'].mean():.2f}x")
    
    print(f"\n3. Exemplo de uso:")
    print(f"   predictor = DecilePredictor()")
    print(f"   result = predictor.predict_single(lead_data)")
    print(f"   print(result['decile'])  # 1-10, onde 10 é melhor")