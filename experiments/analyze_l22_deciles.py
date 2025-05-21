#!/usr/bin/env python
"""
Script para analisar como os limites de decis do conjunto de treinamento
se comportam quando aplicados aos dados do L22.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Caminho para o arquivo de predições do L22
L22_PREDICTIONS_PATH = "/Users/ramonmoreira/desktop/smart_ads/data/L22_test/L22_predictions.csv"

# Limites de decis do conjunto de treinamento
DECILE_BOUNDARIES = [
    0.002510760401721664,  # Limite para decil 1 (valor mínimo)
    0.004405286343612335,  # Limite para decil 2-4 (Limite_Inferior do decil 4)
    0.0067861020629750276, # Limite para decil 5-6 (Limite_Inferior do decil 6) 
    0.008585293019783502,  # Limite para decil 7 (Limite_Inferior do decil 7)
    0.0123334977799704,    # Limite para decil 8 (Limite_Inferior do decil 8)
    0.014276002719238613,  # Limite para decil 9 (Limite_Inferior do decil 9)
    0.022128556375131718,  # Limite para decil 10 (Limite_Inferior do decil 10)
]

def calculate_decile(probability):
    """
    Calcula o decil para uma probabilidade com base nos limites do treinamento.
    """
    # Valores muito baixos (abaixo do mínimo observado no treinamento)
    if probability < DECILE_BOUNDARIES[0]:
        return 1
    
    # Verificar em qual intervalo a probabilidade se encaixa
    for i, boundary in enumerate(DECILE_BOUNDARIES):
        if probability < boundary:
            return i + 1
    
    # Se acima de todos os limites, atribui ao decil mais alto (10)
    return 10

def analyze_l22_deciles():
    """
    Analisa a distribuição dos decis no L22 usando os limites do treinamento.
    """
    # Verificar se o arquivo existe
    if not os.path.exists(L22_PREDICTIONS_PATH):
        print(f"ERRO: Arquivo de predições não encontrado: {L22_PREDICTIONS_PATH}")
        return
    
    # Carregar o arquivo de predições
    print(f"Carregando predições do L22: {L22_PREDICTIONS_PATH}")
    try:
        l22_df = pd.read_csv(L22_PREDICTIONS_PATH)
    except Exception as e:
        print(f"ERRO ao carregar predições: {e}")
        return
    
    # Verificar se temos a coluna de probabilidades
    if 'prediction_probability' not in l22_df.columns:
        # Tentar encontrar uma coluna de probabilidade alternativa
        prob_columns = [col for col in l22_df.columns if 'prob' in col.lower()]
        if prob_columns:
            l22_df['prediction_probability'] = l22_df[prob_columns[0]]
        else:
            print("ERRO: Coluna de probabilidade não encontrada.")
            print(f"Colunas disponíveis: {l22_df.columns.tolist()}")
            return
    
    # Verificar se temos a coluna de target
    target_column = None
    for col in ['target', 'true', 'label', 'class']:
        if col in l22_df.columns:
            target_column = col
            break
    
    if target_column is None:
        print("AVISO: Coluna de target não encontrada. Análise por classe positiva não será realizada.")
    
    # Calcular decil para cada lead
    l22_df['decile_training'] = l22_df['prediction_probability'].apply(calculate_decile)
    
    # Análise 1: Distribuição geral de decis
    decile_counts = l22_df['decile_training'].value_counts().sort_index()
    decile_percentages = (decile_counts / len(l22_df) * 100).round(2)
    
    print("\n=== Distribuição de Decis no L22 ===")
    print(f"Total de leads: {len(l22_df)}")
    
    # Criar tabela de distribuição de decis
    distribution_table = []
    for decile in range(1, 11):
        count = decile_counts.get(decile, 0)
        percentage = decile_percentages.get(decile, 0)
        distribution_table.append({
            'Decil': decile,
            'Contagem': count,
            'Percentual': f"{percentage:.2f}%"
        })
    
    # Exibir tabela de distribuição
    distribution_df = pd.DataFrame(distribution_table)
    print(distribution_df.to_string(index=False))
    
    # Análise 2: Distribuição de probabilidades
    print("\n=== Estatísticas de Probabilidades ===")
    prob_stats = l22_df['prediction_probability'].describe().round(6)
    for stat, value in prob_stats.items():
        print(f"{stat}: {value}")
    
    # Análise 3: Taxa de positivos por decil (se tivermos a coluna de target)
    if target_column:
        print("\n=== Taxa de Positivos por Decil ===")
        
        # Calcular taxas por decil
        positive_rates = []
        for decile in range(1, 11):
            decile_df = l22_df[l22_df['decile_training'] == decile]
            if len(decile_df) > 0:
                positive_count = decile_df[target_column].sum()
                positive_rate = positive_count / len(decile_df) * 100
                positive_rates.append({
                    'Decil': decile,
                    'Total': len(decile_df),
                    'Positivos': positive_count,
                    'Taxa': f"{positive_rate:.2f}%",
                    'Taxa_Num': positive_rate
                })
            else:
                positive_rates.append({
                    'Decil': decile,
                    'Total': 0,
                    'Positivos': 0,
                    'Taxa': "N/A",
                    'Taxa_Num': 0.0
                })
        
        # Exibir tabela de taxas positivas
        rates_df = pd.DataFrame(positive_rates)
        print(rates_df[['Decil', 'Total', 'Positivos', 'Taxa']].to_string(index=False))
        
        # Calcular lift por decil
        global_rate = l22_df[target_column].mean() * 100
        print(f"\nTaxa global de positivos: {global_rate:.2f}%")
        
        print("\n=== Lift por Decil (vs. Taxa Global) ===")
        lifts = []
        for entry in positive_rates:
            if entry['Total'] > 0:
                lift = entry['Taxa_Num'] / global_rate
                lifts.append({
                    'Decil': entry['Decil'],
                    'Lift': f"{lift:.2f}x"
                })
            else:
                lifts.append({
                    'Decil': entry['Decil'],
                    'Lift': "N/A"
                })
        
        # Exibir tabela de lifts
        lifts_df = pd.DataFrame(lifts)
        print(lifts_df.to_string(index=False))
    
    # Análise 4: Métricas de concentração
    if target_column:
        print("\n=== Métricas de Concentração de Positivos ===")
        
        # Ordenar por probabilidade decrescente
        l22_df_sorted = l22_df.sort_values('prediction_probability', ascending=False)
        
        # Calcular métricas de concentração para diferentes percentis
        percentiles = [0.1, 0.2, 0.3, 0.5]
        total_positives = l22_df[target_column].sum()
        
        concentration_metrics = []
        for percentile in percentiles:
            n_leads = int(len(l22_df) * percentile)
            top_leads = l22_df_sorted.head(n_leads)
            n_positives = top_leads[target_column].sum()
            concentration = n_positives / total_positives * 100 if total_positives > 0 else 0
            
            concentration_metrics.append({
                'Percentil': f"Top {percentile*100:.0f}%",
                'Leads': n_leads,
                'Positivos': n_positives,
                'Concentração': f"{concentration:.2f}%"
            })
        
        # Exibir tabela de concentração
        concentration_df = pd.DataFrame(concentration_metrics)
        print(concentration_df.to_string(index=False))
    
    # Salvar relatório em arquivo
    output_path = os.path.join(os.path.dirname(L22_PREDICTIONS_PATH), "l22_decile_analysis.txt")
    print(f"\nSalvando relatório em: {output_path}")
    
    # Redirecionar saída para arquivo
    import sys
    original_stdout = sys.stdout
    with open(output_path, 'w') as f:
        sys.stdout = f
        
        print("=== Análise de Decis do L22 usando Limites do Treinamento ===\n")
        print(f"Arquivo analisado: {L22_PREDICTIONS_PATH}")
        print(f"Total de leads: {len(l22_df)}")
        
        print("\n=== Distribuição de Decis no L22 ===")
        print(distribution_df.to_string(index=False))
        
        print("\n=== Estatísticas de Probabilidades ===")
        for stat, value in prob_stats.items():
            print(f"{stat}: {value}")
        
        if target_column:
            print("\n=== Taxa de Positivos por Decil ===")
            print(rates_df[['Decil', 'Total', 'Positivos', 'Taxa']].to_string(index=False))
            
            print(f"\nTaxa global de positivos: {global_rate:.2f}%")
            
            print("\n=== Lift por Decil (vs. Taxa Global) ===")
            print(lifts_df.to_string(index=False))
            
            print("\n=== Métricas de Concentração de Positivos ===")
            print(concentration_df.to_string(index=False))
    
    # Restaurar saída padrão
    sys.stdout = original_stdout
    
    # Tentar criar visualizações
    try:
        # Gráfico de distribuição de decis
        plt.figure(figsize=(12, 6))
        sns.barplot(x=distribution_df['Decil'], y=distribution_df['Contagem'])
        plt.title('Distribuição de Leads por Decil (Limites do Treinamento)')
        plt.xlabel('Decil')
        plt.ylabel('Contagem de Leads')
        plt.savefig(os.path.join(os.path.dirname(L22_PREDICTIONS_PATH), "l22_decile_distribution.png"))
        
        if target_column:
            # Gráfico de taxa de positivos por decil
            plt.figure(figsize=(12, 6))
            sns.barplot(x=rates_df['Decil'], y=rates_df['Taxa_Num'])
            plt.axhline(y=global_rate, color='r', linestyle='--', label=f'Taxa Global: {global_rate:.2f}%')
            plt.title('Taxa de Positivos por Decil (Limites do Treinamento)')
            plt.xlabel('Decil')
            plt.ylabel('Taxa de Positivos (%)')
            plt.legend()
            plt.savefig(os.path.join(os.path.dirname(L22_PREDICTIONS_PATH), "l22_positive_rates.png"))
        
        print("Visualizações criadas com sucesso!")
    except Exception as e:
        print(f"AVISO: Erro ao criar visualizações: {e}")
    
    print("\nAnálise completa!")
    print(f"Relatório salvo em: {output_path}")

if __name__ == "__main__":
    analyze_l22_deciles()