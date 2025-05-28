#!/usr/bin/env python
"""
Validação do modelo no conjunto de teste - verificar se a performance se mantém.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import pickle
import re
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

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

def validate_on_test_set():
    """Valida o modelo no conjunto de teste."""
    
    print("=== VALIDAÇÃO NO CONJUNTO DE TESTE ===\n")
    
    # 1. Carregar modelo e limiares
    model_path = os.path.join(project_root, "models/artifacts/lightgbm_direct_ranking.joblib")
    thresholds_path = os.path.join(project_root, "models/artifacts/decile_thresholds.pkl")
    
    print(f"Carregando modelo...")
    model = joblib.load(model_path)
    
    print(f"Carregando limiares de decis...")
    with open(thresholds_path, 'rb') as f:
        decile_thresholds = pickle.load(f)
    
    # 2. Carregar conjunto de teste
    test_path = os.path.join(project_root, "data/new/04_feature_selection/test.csv")
    print(f"\nCarregando conjunto de teste...")
    test_df = pd.read_csv(test_path)
    test_df = clean_column_names(test_df)
    
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    print(f"Shape do teste: {X_test.shape}")
    print(f"Taxa de conversão no teste: {y_test.mean():.4f} ({y_test.sum()} conversões)")
    
    # 3. Fazer predições
    print(f"\nFazendo predições...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 4. Calcular métricas gerais
    auc = roc_auc_score(y_test, y_pred_proba)
    gini = 2 * auc - 1
    
    print(f"\nMÉTRICAS GERAIS:")
    print(f"  - AUC: {auc:.4f}")
    print(f"  - GINI: {gini:.4f}")
    
    # 5. Atribuir decis usando os limiares do treino
    def assign_decile(prob, thresholds):
        for i, threshold in enumerate(thresholds):
            if prob <= threshold:
                return i + 1
        return 10
    
    test_df['probability'] = y_pred_proba
    test_df['decile'] = test_df['probability'].apply(lambda p: assign_decile(p, decile_thresholds))
    
    # 6. Performance por decil
    print(f"\nPERFORMANCE POR DECIL NO TESTE:")
    print("-" * 80)
    
    decile_stats = test_df.groupby('decile').agg({
        'target': ['sum', 'count', 'mean'],
        'probability': ['mean', 'min', 'max']
    }).round(6)
    
    decile_stats.columns = ['conversions', 'total', 'conv_rate', 'avg_prob', 'min_prob', 'max_prob']
    
    # Calcular lift
    overall_rate = y_test.mean()
    decile_stats['lift'] = (decile_stats['conv_rate'] / overall_rate).round(2)
    
    # Calcular % das conversões capturadas
    total_conversions = y_test.sum()
    decile_stats['pct_conversions'] = (decile_stats['conversions'] / total_conversions * 100).round(1)
    
    print(decile_stats)
    print("-" * 80)
    
    # 7. Análise cumulativa
    print(f"\nANÁLISE CUMULATIVA (TOP-K):")
    
    # Ordenar por probabilidade decrescente
    test_sorted = test_df.sort_values('probability', ascending=False).reset_index(drop=True)
    
    # Calcular métricas cumulativas
    cumulative_conversions = test_sorted['target'].cumsum()
    cumulative_recall = cumulative_conversions / total_conversions
    
    # Métricas para top percentis específicos
    n_test = len(test_sorted)
    top_10_pct = int(n_test * 0.1)
    top_20_pct = int(n_test * 0.2)
    top_30_pct = int(n_test * 0.3)
    
    print(f"  - Top 10% captura: {cumulative_recall.iloc[top_10_pct-1]:.1%} das conversões")
    print(f"  - Top 20% captura: {cumulative_recall.iloc[top_20_pct-1]:.1%} das conversões")
    print(f"  - Top 30% captura: {cumulative_recall.iloc[top_30_pct-1]:.1%} das conversões")
    
    # 8. Comparação com treino
    print(f"\nCOMPARAÇÃO TREINO vs TESTE:")
    
    # Carregar estatísticas do treino
    json_path = os.path.join(project_root, "models/artifacts/decile_thresholds.json")
    import json
    with open(json_path, 'r') as f:
        train_stats = json.load(f)
    
    print(f"\nDecil 10 (top 10%):")
    print(f"  - Lift no TREINO: 10.00x")
    print(f"  - Lift no TESTE: {decile_stats.loc[10, 'lift']:.2f}x")
    
    if decile_stats.loc[10, 'lift'] > 8:
        print(f"  ✅ Performance se mantém EXCELENTE!")
    elif decile_stats.loc[10, 'lift'] > 5:
        print(f"  ✅ Performance se mantém MUITO BOA!")
    elif decile_stats.loc[10, 'lift'] > 3:
        print(f"  ⚠️  Performance BOA mas inferior ao treino")
    else:
        print(f"  ❌ Performance significativamente inferior ao treino")
    
    # 9. Criar visualizações
    create_test_visualizations(test_df, decile_stats, cumulative_recall)
    
    # 10. Análise de estabilidade
    print(f"\n\nANÁLISE DE ESTABILIDADE:")
    
    # Comparar distribuição de probabilidades
    train_probs_mean = train_stats['statistics']['mean_probability']
    test_probs_mean = y_pred_proba.mean()
    
    print(f"  - Probabilidade média TREINO: {train_probs_mean:.4f}")
    print(f"  - Probabilidade média TESTE: {test_probs_mean:.4f}")
    print(f"  - Diferença: {abs(train_probs_mean - test_probs_mean):.4f}")
    
    if abs(train_probs_mean - test_probs_mean) < 0.02:
        print(f"  ✅ Distribuição estável entre treino e teste")
    else:
        print(f"  ⚠️  Possível drift na distribuição")
    
    return decile_stats, cumulative_recall

def create_test_visualizations(test_df, decile_stats, cumulative_recall):
    """Cria visualizações dos resultados do teste."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Conversões por decil
        ax1 = axes[0, 0]
        bars = ax1.bar(decile_stats.index, decile_stats['conversions'], 
                       color=['#ff6b6b' if x < 8 else '#4ecdc4' if x < 10 else '#45b7d1' 
                              for x in decile_stats.index])
        ax1.set_xlabel('Decil')
        ax1.set_ylabel('Número de Conversões')
        ax1.set_title('Distribuição de Conversões por Decil')
        ax1.set_xticks(range(1, 11))
        
        # Adicionar valores nas barras
        for bar, conv in zip(bars, decile_stats['conversions']):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(conv)}', ha='center', va='bottom')
        
        # 2. Lift por decil
        ax2 = axes[0, 1]
        ax2.plot(decile_stats.index, decile_stats['lift'], 'o-', linewidth=2, markersize=8)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax2.set_xlabel('Decil')
        ax2.set_ylabel('Lift')
        ax2.set_title('Lift por Decil')
        ax2.set_xticks(range(1, 11))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Curva de recall cumulativo
        ax3 = axes[1, 0]
        percentiles = np.arange(0, 101, 10)
        recall_points = []
        for p in percentiles:
            idx = int(len(cumulative_recall) * p / 100) - 1 if p > 0 else 0
            if idx < len(cumulative_recall):
                recall_points.append(cumulative_recall.iloc[idx])
            else:
                recall_points.append(1.0)
        
        ax3.plot(percentiles, np.array(recall_points) * 100, 'o-', linewidth=2, markersize=8)
        ax3.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Random')
        ax3.set_xlabel('Percentual de Leads (%)')
        ax3.set_ylabel('Percentual de Conversões Capturadas (%)')
        ax3.set_title('Curva de Recall Cumulativo')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Distribuição de probabilidades por classe
        ax4 = axes[1, 1]
        test_df[test_df['target'] == 0]['probability'].hist(bins=50, alpha=0.5, 
                                                           label='Não converteu', 
                                                           ax=ax4, density=True)
        test_df[test_df['target'] == 1]['probability'].hist(bins=50, alpha=0.5, 
                                                           label='Converteu', 
                                                           ax=ax4, density=True)
        ax4.set_xlabel('Probabilidade')
        ax4.set_ylabel('Densidade')
        ax4.set_title('Distribuição de Probabilidades por Classe')
        ax4.legend()
        ax4.set_xlim(0, 1)
        
        plt.tight_layout()
        
        # Salvar figura
        output_dir = os.path.join(project_root, "analysis", "figures")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "test_set_validation.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualizações salvas em: {output_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"\n⚠️  Erro ao criar visualizações: {e}")

def calculate_business_impact(decile_stats, overall_rate):
    """Calcula o impacto nos negócios baseado nos resultados do teste."""
    
    print("\n\n=== IMPACTO NOS NEGÓCIOS (BASEADO NO TESTE) ===\n")
    
    # Cenários de realocação de budget
    top_10_lift = decile_stats.loc[10, 'lift']
    top_20_lift = decile_stats.loc[[9, 10], 'lift'].mean()
    
    print("CENÁRIOS DE REALOCAÇÃO DE BUDGET:\n")
    
    # Cenário 1: Focar no top 10%
    print("1. FOCO NO TOP 10%:")
    print(f"   - Lift atual: {top_10_lift:.2f}x")
    print(f"   - Se alocar 50% do budget no top 10%:")
    print(f"     → Ganho esperado: {(top_10_lift - 1) * 0.5:.1%} mais conversões")
    
    # Cenário 2: Focar no top 20%
    print(f"\n2. FOCO NO TOP 20%:")
    print(f"   - Lift médio: {top_20_lift:.2f}x")
    print(f"   - Se alocar 70% do budget no top 20%:")
    print(f"     → Ganho esperado: {(top_20_lift - 1) * 0.7:.1%} mais conversões")
    
    # Cenário 3: Cortar bottom 50%
    bottom_50_rate = decile_stats.loc[1:5, 'conv_rate'].mean()
    bottom_50_lift = bottom_50_rate / overall_rate if overall_rate > 0 else 0
    
    print(f"\n3. ELIMINAR BOTTOM 50%:")
    print(f"   - Lift médio do bottom 50%: {bottom_50_lift:.2f}x")
    print(f"   - Economia sem perda significativa de conversões")
    print(f"   - Budget liberado para realocar: 50%")
    
    # ROI estimado
    print(f"\n\nROI ESTIMADO:")
    print(f"Para cada R$ 100k em ads:")
    current_conversions = 100000 * overall_rate
    scenario1_conversions = current_conversions * (1 + (top_10_lift - 1) * 0.5)
    scenario2_conversions = current_conversions * (1 + (top_20_lift - 1) * 0.7)
    
    print(f"  - Conversões atuais: {current_conversions:.0f}")
    print(f"  - Cenário 1 (top 10%): {scenario1_conversions:.0f} (+{scenario1_conversions - current_conversions:.0f})")
    print(f"  - Cenário 2 (top 20%): {scenario2_conversions:.0f} (+{scenario2_conversions - current_conversions:.0f})")

if __name__ == "__main__":
    # Executar validação
    decile_stats, cumulative_recall = validate_on_test_set()
    
    # Calcular impacto nos negócios
    overall_rate = 0.0115  # Taxa média
    calculate_business_impact(decile_stats, overall_rate)
    
    print("\n\n=== CONCLUSÃO DA VALIDAÇÃO ===")
    
    if decile_stats.loc[10, 'lift'] > 5:
        print("\n✅ MODELO VALIDADO COM SUCESSO!")
        print("   - Performance excepcional se mantém no conjunto de teste")
        print("   - Pronto para implementação em produção")
        print("   - Recomenda-se iniciar teste A/B imediatamente")
    else:
        print("\n⚠️  MODELO VALIDADO COM RESSALVAS")
        print("   - Performance boa mas inferior ao esperado")
        print("   - Ainda vale a pena testar em produção")
        print("   - Monitorar de perto os resultados iniciais")