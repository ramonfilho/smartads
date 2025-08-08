#!/usr/bin/env python
"""
Script para análise de métricas de ranqueamento do modelo Smart Ads.

Este script gera métricas específicas para avaliar a capacidade do modelo
de ranquear leads por probabilidade de conversão, com foco em mostrar
quanto os segmentos superiores concentram compradores.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import argparse
from datetime import datetime

# Configurações de caminhos absolutos - ALTERE CONFORME NECESSÁRIO
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
DATA_DIR = os.path.join(PROJECT_ROOT, "data/L22_test")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports/ranking_metrics")

# Caminhos de arquivos padrão - serão usados se não forem fornecidos via argumentos
DEFAULT_PREDICTIONS_PATH = os.path.join(DATA_DIR, "L22_predictions.csv")
DEFAULT_OUTPUT_DIR = os.path.join(REPORTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))

# Configurações para gráficos mais bonitos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = (12, 7)
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3

def load_data(predictions_path):
    """
    Carrega os dados de predição para análise.
    """
    print(f"Carregando dados de: {predictions_path}")
    
    try:
        # Carregar dados com tipos apropriados
        df = pd.read_csv(predictions_path)
        
        # Verificar colunas necessárias
        required_cols = ['target', 'prediction_probability']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Tentar encontrar colunas alternativas
            if 'target' in missing_cols:
                alternative_targets = ['true', 'actual', 'class', 'label']
                for alt in alternative_targets:
                    if alt in df.columns:
                        df['target'] = df[alt]
                        missing_cols.remove('target')
                        print(f"Usando coluna '{alt}' como target")
                        break
            
            if 'prediction_probability' in missing_cols:
                alternative_probs = ['probability', 'prediction_prob', 'prob', 'score', 'prediction_score', 'probability_1']
                for alt in alternative_probs:
                    if alt in df.columns:
                        df['prediction_probability'] = df[alt]
                        missing_cols.remove('prediction_probability')
                        print(f"Usando coluna '{alt}' como prediction_probability")
                        break
        
        if missing_cols:
            raise ValueError(f"Colunas necessárias não encontradas: {missing_cols}")
        
        print(f"Dados carregados com sucesso: {df.shape[0]} registros")
        print(f"Taxa de conversão global: {df['target'].mean():.4f} ({df['target'].sum()} conversões)")
        
        return df
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def calculate_decile_metrics(df, prob_col='prediction_probability', target_col='target'):
    """
    Calcula métricas por decil de probabilidade.
    """
    # Criar uma cópia para não modificar o original
    data = df.copy()
    
    # Verificar se há valores distintos suficientes para criar 10 bins
    unique_values = data[prob_col].nunique()
    print(f"Valores distintos de probabilidade: {unique_values}")
    
    if unique_values < 10:
        print(f"AVISO: Menos de 10 valores distintos de probabilidade. Usando {unique_values} bins em vez de 10.")
        n_bins = unique_values
    else:
        n_bins = 10
    
    # Calcular decis, com tratamento para valores duplicados
    try:
        # Tentativa 1: qcut normal com tratamento de duplicatas
        data['decile'] = pd.qcut(data[prob_col], n_bins, labels=False, duplicates='drop') + 1
    except ValueError as e:
        print(f"Erro ao usar pd.qcut: {e}")
        print("Usando método alternativo para criar decis...")
        
        # Tentativa 2: Ordenar por probabilidade e dividir em grupos de tamanho igual
        data = data.sort_values(by=prob_col, ascending=True)
        data['rank'] = range(len(data))
        data['decile'] = np.ceil(data['rank'] / len(data) * n_bins)
        data = data.drop('rank', axis=1)
    
    # Taxa de conversão global (baseline)
    global_conversion_rate = data[target_col].mean()
    
    # Calcular métricas por decil
    decile_metrics = data.groupby('decile').agg(
        count=pd.NamedAgg(column=target_col, aggfunc='count'),
        positives=pd.NamedAgg(column=target_col, aggfunc='sum'),
        mean_prob=pd.NamedAgg(column=prob_col, aggfunc='mean')
    ).reset_index()
    
    # Calcular taxa de conversão e lift
    decile_metrics['conversion_rate'] = decile_metrics['positives'] / decile_metrics['count']
    decile_metrics['lift'] = decile_metrics['conversion_rate'] / global_conversion_rate
    
    # Formatar para exibição
    decile_metrics['conversion_rate_pct'] = decile_metrics['conversion_rate'] * 100
    decile_metrics['conversion_share'] = decile_metrics['positives'] / decile_metrics['positives'].sum()
    decile_metrics['conversion_share_pct'] = decile_metrics['conversion_share'] * 100
    
    # Ordenar por decil
    decile_metrics = decile_metrics.sort_values('decile')
    
    return decile_metrics, global_conversion_rate

def calculate_concentration_metrics(df, prob_col='prediction_probability', target_col='target'):
    """
    Calcula métricas de concentração de positivos nos percentis superiores.
    """
    # Criar uma cópia para não modificar o original
    data = df.copy()
    
    # Ordenar por probabilidade (decrescente)
    data = data.sort_values(by=prob_col, ascending=False).reset_index(drop=True)
    
    # Total de positivos
    total_positives = data[target_col].sum()
    
    # Calcular total de registros
    total_records = len(data)
    
    # Calcular concentração para diferentes percentis
    percentiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    concentration_metrics = []
    
    for p in percentiles:
        # Número de registros neste percentil
        n_records = int(total_records * p)
        
        # Contagem de positivos
        if n_records > 0:
            n_positives = data.iloc[:n_records][target_col].sum()
            concentration = n_positives / total_positives
        else:
            n_positives = 0
            concentration = 0
        
        concentration_metrics.append({
            'percentile': p,
            'top_percent': p * 100,
            'n_records': n_records,
            'n_positives': n_positives,
            'concentration': concentration,
            'concentration_pct': concentration * 100
        })
    
    # Converter para DataFrame
    concentration_df = pd.DataFrame(concentration_metrics)
    
    return concentration_df

def plot_decile_metrics(decile_metrics, global_rate, output_dir=None):
    """
    Cria gráficos para visualização das métricas por decil.
    """
    # 1. Taxa de conversão e lift por decil
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Gráfico de taxa de conversão
    bars = ax1.bar(decile_metrics['decile'], decile_metrics['conversion_rate_pct'])
    ax1.axhline(y=global_rate*100, color='r', linestyle='--', 
                label=f'Taxa Global: {global_rate*100:.2f}%')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
        # Adicionar contagem de registros
        ax1.annotate(f"n={decile_metrics.iloc[i]['count']}",
                    xy=(bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, -20),  # 20 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)
    
    ax1.set_title('Taxa de Conversão por Decil de Probabilidade')
    ax1.set_xlabel('Decil (1 = menor prob, 10 = maior prob)')
    ax1.set_ylabel('Taxa de Conversão (%)')
    ax1.legend()
    
    # Gráfico de lift
    bars = ax2.bar(decile_metrics['decile'], decile_metrics['lift'])
    ax2.axhline(y=1, color='r', linestyle='--', label='Baseline (1.0x)')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax2.set_title('Lift por Decil de Probabilidade')
    ax2.set_xlabel('Decil (1 = menor prob, 10 = maior prob)')
    ax2.set_ylabel('Lift (vs. Taxa Média)')
    ax2.legend()
    
    plt.tight_layout()
    
    # Salvar gráfico se diretório fornecido
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'decile_metrics.png'), dpi=300, bbox_inches='tight')
    
    # 2. Distribuição de conversões por decil
    plt.figure(figsize=(12, 7))
    
    # Gráfico de fatia das conversões
    bars = plt.bar(decile_metrics['decile'], decile_metrics['conversion_share_pct'])
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 1:  # Só mostrar percentuais significativos
            plt.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
            # Adicionar contagem de positivos
            plt.annotate(f"n={int(decile_metrics.iloc[i]['positives'])}",
                        xy=(bar.get_x() + bar.get_width() / 2, 0),
                        xytext=(0, -20),  # 20 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    plt.title('Distribuição de Conversões por Decil')
    plt.xlabel('Decil (1 = menor prob, 10 = maior prob)')
    plt.ylabel('% do Total de Conversões')
    
    # Salvar gráfico se diretório fornecido
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'conversion_distribution.png'), dpi=300, bbox_inches='tight')
    
    return

def plot_cumulative_gains(concentration_df, output_dir=None):
    """
    Cria gráficos para visualização das métricas de concentração.
    """
    # 1. Gráfico de concentração cumulativa
    plt.figure(figsize=(12, 7))
    
    # Gráfico de concentração
    plt.plot(concentration_df['top_percent'], concentration_df['concentration_pct'], 
             'o-', linewidth=3, markersize=10)
    
    # Linha de referência (modelo aleatório)
    plt.plot([0, 100], [0, 100], 'r--', label='Modelo Aleatório')
    
    # Adicionar anotações para pontos específicos
    for i, row in concentration_df.iterrows():
        if row['top_percent'] in [10, 20, 30, 50]:
            plt.annotate(f"{row['concentration_pct']:.1f}%",
                        xy=(row['top_percent'], row['concentration_pct']),
                        xytext=(5, 0),  # 5 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center',
                        fontweight='bold')
    
    plt.title('Curva de Ganho Cumulativo')
    plt.xlabel('% de Leads (Ordenados por Probabilidade)')
    plt.ylabel('% de Conversões Capturadas')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Adicionar texto explicativo
    text_x = 60
    text_y = 30
    
    for i, row in concentration_df.iterrows():
        if row['top_percent'] in [10, 20, 30]:
            plt.text(text_x, text_y - i*5, 
                     f"Os {row['top_percent']:.0f}% dos leads melhor ranqueados contêm {row['concentration_pct']:.1f}% de todos os compradores",
                     fontsize=11, va='center')
    
    # Salvar gráfico se diretório fornecido
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'cumulative_gains.png'), dpi=300, bbox_inches='tight')
    
    return

def generate_report(df, output_dir=None):
    """
    Gera relatório completo com todas as métricas e visualizações.
    """
    if df is None:
        print("Sem dados para análise.")
        return
    
    # Criar diretório de saída se não existir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Gerando Relatório de Métricas de Ranqueamento ===\n")
    
    # 1. Calcular métricas por decil
    decile_metrics, global_rate = calculate_decile_metrics(df)
    
    # 2. Calcular métricas de concentração
    concentration_metrics = calculate_concentration_metrics(df)
    
    # 3. Gerar visualizações
    plot_decile_metrics(decile_metrics, global_rate, output_dir)
    plot_cumulative_gains(concentration_metrics, output_dir)
    
    # 4. Métricas de performance global de ranqueamento
    auc_score = roc_auc_score(df['target'], df['prediction_probability'])
    avg_precision = average_precision_score(df['target'], df['prediction_probability'])
    
    # 5. Imprimir resumo de métricas principais
    print("\n=== Métricas de Concentração ===")
    print(f"Taxa de conversão global: {global_rate*100:.2f}%")
    print("\nConcentração de conversões:")
    
    for i, row in concentration_metrics.iterrows():
        if row['top_percent'] in [10, 20, 30, 50]:
            print(f"• Os {row['top_percent']:.0f}% de leads melhor ranqueados contêm {row['concentration_pct']:.1f}% de todos os compradores")
    
    print("\n=== Métricas por Decil ===")
    print("Lift por decil (quantas vezes acima da média):")
    
    for i, row in decile_metrics.iterrows():
        if row['decile'] in [8, 9, 10] or row['decile'] >= decile_metrics['decile'].max() - 2:
            print(f"• Decil {int(row['decile'])}: {row['lift']:.2f}x a taxa média ({row['conversion_rate_pct']:.2f}% vs {global_rate*100:.2f}%)")
    
    # 6. Salvar métricas em CSV
    if output_dir:
        decile_metrics.to_csv(os.path.join(output_dir, 'decile_metrics.csv'), index=False)
        concentration_metrics.to_csv(os.path.join(output_dir, 'concentration_metrics.csv'), index=False)
        
        # Salvar métricas resumidas para apresentação
        summary = {
            'Métrica': ['AUC', 'Average Precision', 'Taxa de Conversão Global']
        }
        
        values = [
            f"{auc_score:.4f}",
            f"{avg_precision:.4f}",
            f"{global_rate*100:.2f}%"
        ]
        
        # Adicionar métricas de concentração
        for top_pct in [10, 20, 30]:
            if top_pct in concentration_metrics['top_percent'].values:
                summary['Métrica'].append(f'Concentração (Top {top_pct}%)')
                conc_val = concentration_metrics[concentration_metrics['top_percent']==top_pct]['concentration_pct'].values[0]
                values.append(f"{conc_val:.1f}%")
        
        # Adicionar lift dos decis superiores
        max_decile = decile_metrics['decile'].max()
        for d in range(max_decile, max(max_decile-3, 0), -1):
            if d in decile_metrics['decile'].values:
                summary['Métrica'].append(f'Lift Decil {d}')
                lift_val = decile_metrics[decile_metrics['decile']==d]['lift'].values[0]
                values.append(f"{lift_val:.2f}x")
        
        summary['Valor'] = values
        pd.DataFrame(summary).to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
    
    print("\n=== Relatório de Ranqueamento Concluído ===")
    if output_dir:
        print(f"Resultados salvos em: {output_dir}")
    else:
        print("Nenhum diretório de saída especificado. Resultados não foram salvos.")

def main():
    """Função principal para execução do script."""
    parser = argparse.ArgumentParser(description="Análise de métricas de ranqueamento para o modelo Smart Ads")
    parser.add_argument("--input", type=str, default=DEFAULT_PREDICTIONS_PATH, 
                       help=f"Caminho para o arquivo CSV com predições e valores reais (padrão: {DEFAULT_PREDICTIONS_PATH})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f"Diretório para salvar resultados (padrão: {DEFAULT_OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    # Verificar se o arquivo de entrada existe
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"ERRO: Arquivo de entrada não encontrado: {input_path}")
        return 1
    
    # Definir diretório de saída
    output_dir = args.output
    
    # Carregar dados
    df = load_data(input_path)
    
    # Gerar relatório
    generate_report(df, output_dir)
    
    return 0

if __name__ == "__main__":
    main()