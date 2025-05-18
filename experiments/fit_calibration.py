#!/usr/bin/env python
"""
Script para comparar probabilidades entre o modelo de referência e a pipeline
e criar uma função de mapeamento que converta as probabilidades da pipeline
nas probabilidades do modelo de referência.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Caminhos para os arquivos
REF_RESULTS = "/Users/ramonmoreira/desktop/smart_ads/reports/calibration_validation_two_models/20250514_084500/gmm_test_results.csv"
PIPELINE_RESULTS = "/Users/ramonmoreira/desktop/smart_ads/inference/output/predictions_20250517_202046.csv"
OUTPUT_PATH = "/Users/ramonmoreira/desktop/smart_ads/inference/params/calibration_mapping.csv"
OUTPUT_PLOT = "/Users/ramonmoreira/desktop/smart_ads/inference/params/calibration_curve.png"

def fit_calibration_function():
    """
    Compara as probabilidades da referência e da pipeline para criar uma função
    de calibração que transforme as probabilidades da pipeline nas da referência.
    """
    print(f"Carregando resultados da referência: {REF_RESULTS}")
    ref_df = pd.read_csv(REF_RESULTS)
    
    print(f"Carregando resultados da pipeline: {PIPELINE_RESULTS}")
    pipeline_df = pd.read_csv(PIPELINE_RESULTS)
    
    # Verificar se ambos têm a coluna 'email' para alinhamento
    if 'email' not in ref_df.columns or 'email' not in pipeline_df.columns:
        print("ERRO: Coluna 'email' não encontrada em um ou ambos os arquivos")
        return
    
    # Remover duplicações para evitar ambiguidades
    if 'email' in ref_df.columns:
        print(f"Removendo duplicações na referência (original: {len(ref_df)} linhas)")
        ref_df = ref_df.drop_duplicates(subset=['email'])
        print(f"Após remoção: {len(ref_df)} linhas")
    
    if 'email' in pipeline_df.columns:
        print(f"Removendo duplicações na pipeline (original: {len(pipeline_df)} linhas)")
        pipeline_df = pipeline_df.drop_duplicates(subset=['email'])
        print(f"Após remoção: {len(pipeline_df)} linhas")
    
    # Mesclar os DataFrames para comparar as probabilidades
    print("Realizando merge de dados...")
    merged_df = pd.merge(
        ref_df, pipeline_df,
        on='email', 
        suffixes=('_ref', '_pipeline')
    )
    
    print(f"Dados alinhados: {len(merged_df)} amostras")
    
    # Extrair probabilidades para comparação
    ref_probs = merged_df['probability_ref'] if 'probability_ref' in merged_df.columns else merged_df['probability']
    pipeline_probs = merged_df['probability_pipeline'] if 'probability_pipeline' in merged_df.columns else merged_df['probability']
    
    print(f"Valores únicos na referência: {len(ref_probs.unique())}")
    print(f"Valores únicos na pipeline: {len(pipeline_probs.unique())}")
    
    # Criar um DataFrame para o mapeamento
    print("Criando mapeamento de calibração...")
    mapping_df = pd.DataFrame({
        'pipeline_prob': pipeline_probs,
        'ref_prob': ref_probs
    })
    
    # Analisar correlação
    correlation = mapping_df['pipeline_prob'].corr(mapping_df['ref_prob'])
    print(f"Correlação entre probabilidades: {correlation:.4f}")
    
    # Gerar um scatter plot para visualização
    plt.figure(figsize=(10, 8))
    plt.scatter(mapping_df['pipeline_prob'], mapping_df['ref_prob'], alpha=0.1, s=5)
    plt.plot([0, 1], [0, 1], 'r--', label='Identidade')
    plt.xlabel('Probabilidade da Pipeline')
    plt.ylabel('Probabilidade da Referência')
    plt.title('Mapeamento das Probabilidades')
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT)
    print(f"Visualização salva em: {OUTPUT_PLOT}")
    
    # Agora vamos criar um modelo de mapeamento suavizado
    # Agrupar por faixas de probabilidade da pipeline e calcular a média da referência
    bins = 100  # Número de faixas
    mapping_df['pipeline_bin'] = pd.cut(mapping_df['pipeline_prob'], bins=bins)
    
    # Calcular a média das probabilidades da referência para cada faixa
    bin_mapping = mapping_df.groupby('pipeline_bin')['ref_prob'].mean().reset_index()
    
    # Extrair o ponto médio de cada faixa
    bin_mapping['pipeline_prob'] = bin_mapping['pipeline_bin'].apply(lambda x: x.mid)
    
    # Ordenar por probabilidade da pipeline
    bin_mapping = bin_mapping.sort_values('pipeline_prob')
    
    # Adicionar pontos extremos se necessário
    if bin_mapping['pipeline_prob'].min() > 0:
        bin_mapping = pd.concat([
            pd.DataFrame({'pipeline_prob': [0], 'ref_prob': [0], 'pipeline_bin': [None]}),
            bin_mapping
        ]).reset_index(drop=True)
    
    if bin_mapping['pipeline_prob'].max() < 1:
        bin_mapping = pd.concat([
            bin_mapping,
            pd.DataFrame({'pipeline_prob': [1], 'ref_prob': [1], 'pipeline_bin': [None]})
        ]).reset_index(drop=True)
    
    # Remover NaN's
    bin_mapping = bin_mapping.dropna(subset=['ref_prob']).reset_index(drop=True)
    
    # Salvar o mapeamento
    bin_mapping[['pipeline_prob', 'ref_prob']].to_csv(OUTPUT_PATH, index=False)
    print(f"Mapeamento de calibração salvo em: {OUTPUT_PATH}")
    
    # Selecionar 20 pontos representativos para o código
    # Podemos selecionar pontos em intervalos regulares
    n_points = 20
    indices = np.linspace(0, len(bin_mapping) - 1, n_points).astype(int)
    
    calibration_points = bin_mapping.iloc[indices][['pipeline_prob', 'ref_prob']].values
    
    print("\nPontos de calibração para o código:")
    print("EXACT_CALIBRATION_POINTS = [")
    
    for pipeline_prob, ref_prob in calibration_points:
        print(f"    ({pipeline_prob:.6f}, {ref_prob:.6f}),")
    
    print("]")
    
    return bin_mapping

if __name__ == "__main__":
    fit_calibration_function()