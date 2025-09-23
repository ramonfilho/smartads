"""
Script para gerar gráficos de performance do modelo RF Temporal.
Baseado nos metadados do modelo para visualizar métricas de desempenho.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Configurar estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_model_metadata(model_name="v1_devclub_rf_temporal"):
    """Carrega metadados do modelo."""
    metadata_file = f"arquivos_modelo/model_metadata_{model_name}.json"

    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_decil_performance_chart(metadata):
    """Cria gráfico de performance por decil."""

    # Extrair dados dos decis
    decil_data = []
    for i in range(1, 11):
        decil_info = metadata['decil_analysis'][f'decil_{i}']
        decil_data.append({
            'decil': i,
            'total_leads': decil_info['total_leads'],
            'conversions': decil_info['conversions'],
            'conversion_rate': decil_info['conversion_rate'] * 100,  # Converter para %
            'pct_total_conversions': decil_info['pct_total_conversions'],
            'lift': decil_info['lift']
        })

    df_decis = pd.DataFrame(decil_data)

    # Criar subplot com 3 gráficos
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 Performance do Modelo RF Temporal - Análise por Decis', fontsize=16, fontweight='bold')

    # 1. Taxa de Conversão por Decil
    bars1 = ax1.bar(df_decis['decil'], df_decis['conversion_rate'],
                    color='steelblue', alpha=0.7, edgecolor='navy')
    ax1.set_title('Taxa de Conversão por Decil', fontweight='bold')
    ax1.set_xlabel('Decil (1=Pior, 10=Melhor)')
    ax1.set_ylabel('Taxa de Conversão (%)')
    ax1.grid(True, alpha=0.3)

    # Adicionar valores nas barras
    for bar, rate in zip(bars1, df_decis['conversion_rate']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')

    # Linha da taxa base
    baseline_rate = metadata['performance_metrics']['baseline_conversion_rate'] * 100
    ax1.axhline(y=baseline_rate, color='red', linestyle='--',
                label=f'Taxa Base: {baseline_rate:.2f}%')
    ax1.legend()

    # 2. Lift por Decil
    bars2 = ax2.bar(df_decis['decil'], df_decis['lift'],
                    color='green', alpha=0.7, edgecolor='darkgreen')
    ax2.set_title('Lift por Decil', fontweight='bold')
    ax2.set_xlabel('Decil (1=Pior, 10=Melhor)')
    ax2.set_ylabel('Lift (vs. Taxa Base)')
    ax2.grid(True, alpha=0.3)

    # Adicionar valores nas barras
    for bar, lift in zip(bars2, df_decis['lift']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{lift:.2f}x', ha='center', va='bottom', fontweight='bold')

    # Linha de referência (lift = 1.0)
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Lift = 1.0 (Taxa Base)')
    ax2.legend()

    # 3. Concentração de Conversões
    bars3 = ax3.bar(df_decis['decil'], df_decis['pct_total_conversions'],
                    color='orange', alpha=0.7, edgecolor='darkorange')
    ax3.set_title('% do Total de Conversões por Decil', fontweight='bold')
    ax3.set_xlabel('Decil (1=Pior, 10=Melhor)')
    ax3.set_ylabel('% do Total de Conversões')
    ax3.grid(True, alpha=0.3)

    # Adicionar valores nas barras
    for bar, pct in zip(bars3, df_decis['pct_total_conversions']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Linha de referência (10% = distribuição uniforme)
    ax3.axhline(y=10.0, color='red', linestyle='--', label='10% (Uniforme)')
    ax3.legend()

    # 4. Conversões Acumuladas (Top N decis)
    df_decis_desc = df_decis.sort_values('decil', ascending=False)
    conversions_cumsum = df_decis_desc['pct_total_conversions'].cumsum()

    ax4.plot(range(1, 11), conversions_cumsum, marker='o', linewidth=2,
             markersize=8, color='purple')
    ax4.set_title('Concentração Cumulativa de Conversões', fontweight='bold')
    ax4.set_xlabel('Top N Decis')
    ax4.set_ylabel('% Acumulado de Conversões')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(1, 11))

    # Adicionar valores nos pontos
    for i, pct in enumerate(conversions_cumsum, 1):
        ax4.text(i, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Destacar métricas importantes
    top3 = metadata['performance_metrics']['top3_decil_concentration']
    top5 = metadata['performance_metrics']['top5_decil_concentration']
    ax4.axhline(y=top3, color='red', linestyle=':', alpha=0.7, label=f'Top 3: {top3}%')
    ax4.axhline(y=top5, color='blue', linestyle=':', alpha=0.7, label=f'Top 5: {top5}%')
    ax4.legend()

    plt.tight_layout()
    return fig

def create_model_summary_chart(metadata):
    """Cria gráfico resumo das métricas do modelo."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📈 Resumo de Performance - RF Temporal', fontsize=16, fontweight='bold')

    # 1. Métricas Principais
    metrics = metadata['performance_metrics']
    metric_names = ['AUC', 'Top 3 Decis\n(%)', 'Top 5 Decis\n(%)', 'Lift Máximo', 'Monotonia\n(%)']
    metric_values = [
        metrics['auc'],
        metrics['top3_decil_concentration'],
        metrics['top5_decil_concentration'],
        metrics['lift_maximum'],
        metrics['monotonia_percentage']
    ]

    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
    bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Métricas Principais do Modelo', fontweight='bold')
    ax1.set_ylabel('Valor da Métrica')

    # Adicionar valores nas barras
    for bar, value in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # 2. Distribuição dos Dados de Treino
    training_data = metadata['training_data']
    sizes = [training_data['training_records'], training_data['test_records']]
    labels = ['Treino', 'Teste']
    colors = ['lightblue', 'lightcoral']

    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90)
    ax2.set_title(f'Divisão dos Dados\n(Total: {training_data["total_records"]:,} registros)',
                  fontweight='bold')

    # 3. Taxa de Conversão por Split
    target_dist = training_data['target_distribution']
    splits = ['Treino', 'Teste']
    rates = [target_dist['training_positive_rate'] * 100,
             target_dist['test_positive_rate'] * 100]

    bars3 = ax3.bar(splits, rates, color=['lightblue', 'lightcoral'],
                    alpha=0.8, edgecolor='black')
    ax3.set_title('Taxa de Conversão por Split', fontweight='bold')
    ax3.set_ylabel('Taxa de Conversão (%)')

    for bar, rate in zip(bars3, rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')

    # 4. Informações do Modelo
    model_info = metadata['model_info']
    hyperparams = metadata['hyperparameters']

    info_text = f"""
    Modelo: {model_info['model_type']}
    Split: {model_info['split_type']}
    Features: {training_data['features_count']}

    Hiperparâmetros:
    • Estimadores: {hyperparams['n_estimators']}
    • Max Depth: {hyperparams['max_depth']}
    • Class Weight: {hyperparams['class_weight']}
    • Random State: {hyperparams['random_state']}

    Treinado em: {model_info['trained_at'][:10]}
    """

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Informações do Modelo', fontweight='bold')

    plt.tight_layout()
    return fig

def main():
    """Função principal para gerar todos os gráficos."""

    print("🎨 Gerando gráficos de performance do modelo RF Temporal...")

    try:
        # Carregar metadados
        metadata = load_model_metadata("v1_devclub_rf_temporal")
        print("✓ Metadados carregados")

        # Criar gráficos
        print("📊 Criando gráfico de performance por decis...")
        fig1 = create_decil_performance_chart(metadata)
        fig1.savefig('model_performance_decils.png', dpi=300, bbox_inches='tight')
        print("✓ Salvo: model_performance_decils.png")

        print("📈 Criando gráfico resumo do modelo...")
        fig2 = create_model_summary_chart(metadata)
        fig2.savefig('model_summary.png', dpi=300, bbox_inches='tight')
        print("✓ Salvo: model_summary.png")

        # Mostrar gráficos
        plt.show()

        print("\n🎉 Gráficos gerados com sucesso!")
        print("📂 Arquivos salvos: model_performance_decils.png, model_summary.png")

    except Exception as e:
        print(f"❌ Erro ao gerar gráficos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()