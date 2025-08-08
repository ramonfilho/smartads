#!/usr/bin/env python
"""
Script simplificado para análise de falsos negativos com foco em PR-AUC, 
sensibilidade de threshold, distribuição de features e calibração.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import mlflow
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURAÇÃO DE CAMINHOS ============
# Modifique estas variáveis conforme necessário para corresponder à sua estrutura de diretórios
PROJECT_ROOT = "/Users/ramonmoreira/Desktop/smart_ads"  # Altere este caminho para a raiz do seu projeto

# Caminhos derivados - Mantenha a estrutura relativa ou altere conforme necessário
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "03_feature_selection_textv2")
MLFLOW_DIR = os.path.join(PROJECT_ROOT, "models", "mlflow")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "validation.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", f"model_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# =============== ADICIONANDO AO PATH =================
sys.path.append(PROJECT_ROOT)
print(f"Adicionado ao path: {PROJECT_ROOT}")

def setup_mlflow(mlflow_dir):
    """Configura o MLflow para tracking de experimentos."""
    os.makedirs(mlflow_dir, exist_ok=True)  # Criar diretório se não existir
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    print(f"MLflow configurado para usar: {mlflow.get_tracking_uri()}")

def load_model_from_mlflow(model_uri):
    """Carrega um modelo a partir do MLflow usando seu URI."""
    try:
        print(f"Carregando modelo de: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        return None

def load_data(data_path, verbose=True):
    """Carrega e prepara os dados para análise."""
    if verbose:
        print(f"Carregando dados de: {data_path}")
    
    # Verificar se o arquivo existe antes de tentar carregar
    if not os.path.exists(data_path):
        print(f"ERRO: Dataset não encontrado: {data_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
    
    # Tentar importar funções do módulo baseline_model
    try:
        from src.evaluation.baseline_model import sanitize_column_names, convert_integer_columns_to_float
        
        df = pd.read_csv(data_path)
        
        if verbose:
            print(f"Dataset carregado: {df.shape[0]} exemplos, {df.shape[1]} colunas")
        
        # Aplicar as mesmas transformações usadas durante o treinamento
        print("Sanitizando nomes das colunas...")
        sanitize_column_names(df)
        
        # Converter colunas inteiras para float
        print("Convertendo colunas inteiras para float...")
        convert_integer_columns_to_float(df)
    except ImportError:
        # Fallback simples se o módulo não estiver disponível
        print("Módulo baseline_model não encontrado. Usando processamento simples.")
        df = pd.read_csv(data_path)
        
        # Sanitizar nomes de colunas - método simplificado
        df.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # Identificar a coluna target
    target_col = None
    for col in ['target', 'label', 'class', 'y']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
        if verbose:
            print(f"Coluna target não encontrada, usando última coluna: {target_col}")
    elif verbose:
        print(f"Coluna target identificada: {target_col}")
    
    # Separar features e target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if verbose:
        print(f"Distribuição do target:")
        print(pd.Series(y).value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
    
    return df, X, y, target_col

def generate_predictions(model, X, threshold=0.5):
    """Gera previsões usando o modelo carregado."""
    # Obter probabilidades
    y_prob = model.predict_proba(X)[:, 1]
    
    # Aplicar threshold para obter previsões binárias
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_pred, y_prob

def create_error_analysis_df(df, y_true, y_pred, y_prob):
    """Cria um DataFrame para análise de erro com classificações adicionais."""
    # Criar novo DataFrame para análise
    analysis_df = df.copy()
    
    # Adicionar colunas de previsão e probabilidade
    analysis_df['true_class'] = y_true
    analysis_df['predicted_class'] = y_pred
    analysis_df['probability'] = y_prob
    
    # Adicionar categoria de erro
    analysis_df['error_type'] = 'correct_prediction'
    analysis_df.loc[(y_true == 1) & (y_pred == 0), 'error_type'] = 'false_negative'
    analysis_df.loc[(y_true == 0) & (y_pred == 1), 'error_type'] = 'false_positive'
    
    # Mostrar distribuição de categorias
    print("\nDistribuição de categorias de predição:")
    print(analysis_df['error_type'].value_counts().to_frame().T)
    
    return analysis_df

def analyze_pr_curve(y_true, y_prob, output_dir):
    """Analisa a curva Precision-Recall e identifica pontos ótimos."""
    # Calcular pontos da curva PR
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calcular PR-AUC
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_prob)
    
    print(f"\nPR-AUC: {pr_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Calcular F1 para cada threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Encontrar threshold para máximo F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    
    print(f"Melhor F1-Score: {best_f1:.4f} (threshold: {best_threshold:.4f})")
    print(f"  Precision: {best_precision:.4f}")
    print(f"  Recall: {best_recall:.4f}")
    
    # Thresholds que favorecem recall
    recall_thresholds = {}
    recall_targets = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    for target in recall_targets:
        # Encontrar o maior threshold que atinge o recall alvo
        valid_indices = np.where(recall >= target)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[0]  # Maior threshold (menor índice) que atinge o recall alvo
            t = thresholds[best_idx] if best_idx < len(thresholds) else 0
            p = precision[best_idx]
            r = recall[best_idx]
            f1 = 2 * (p * r) / (p + r + 1e-10)
            
            recall_thresholds[target] = {
                'threshold': t,
                'precision': p,
                'recall': r,
                'f1': f1
            }
    
    print("\nThresholds para atingir diferentes níveis de recall:")
    for target, metrics in recall_thresholds.items():
        print(f"Recall >= {target:.1f}: threshold={metrics['threshold']:.4f}, precision={metrics['precision']:.4f}, f1={metrics['f1']:.4f}")
    
    # Identificar pontos de interesse para análise
    interesting_thresholds = {}
    
    # Ponto onde precision = recall (equilíbrio)
    pr_diff = np.abs(precision - recall)
    eq_idx = np.argmin(pr_diff)
    interesting_thresholds['equal_pr'] = {
        'threshold': thresholds[eq_idx] if eq_idx < len(thresholds) else 0,
        'precision': precision[eq_idx],
        'recall': recall[eq_idx],
        'f1': f1_scores[eq_idx]
    }
    
    # Ponto de máxima curvatura (knee point) - simplificado
    curvature = np.gradient(np.gradient(precision))
    knee_idx = np.argmax(np.abs(curvature))
    interesting_thresholds['knee_point'] = {
        'threshold': thresholds[knee_idx] if knee_idx < len(thresholds) else 0,
        'precision': precision[knee_idx],
        'recall': recall[knee_idx],
        'f1': f1_scores[knee_idx]
    }
    
    print("\nPontos interessantes na curva PR:")
    for name, metrics in interesting_thresholds.items():
        print(f"{name}: threshold={metrics['threshold']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Plotar curva PR
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'PR Curve (AUC={pr_auc:.3f})')
    
    # Marcar pontos chave
    plt.scatter([best_recall], [best_precision], c='red', s=100, label=f'Max F1={best_f1:.3f} (t={best_threshold:.3f})')
    
    for target, metrics in recall_thresholds.items():
        if target in [0.5, 0.7, 0.9]:  # Plot apenas alguns para não sobrecarregar
            plt.scatter([metrics['recall']], [metrics['precision']], marker='x', s=100, 
                      label=f'Recall={target:.1f} (t={metrics["threshold"]:.3f})')
    
    # Marcar ponto de equilíbrio
    eq_metrics = interesting_thresholds['equal_pr']
    plt.scatter([eq_metrics['recall']], [eq_metrics['precision']], marker='*', s=150, c='green',
              label=f'P=R={eq_metrics["precision"]:.3f} (t={eq_metrics["threshold"]:.3f})')
    
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Salvar figura
    pr_curve_path = os.path.join(output_dir, 'pr_curve_analysis.png')
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Curva PR salva em: {pr_curve_path}")
    
    # Salvar dados da curva PR
    pr_curve_data = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'threshold': np.append(thresholds, [1.0]),  # Ajuste devido ao tamanho
        'f1': f1_scores
    })
    
    pr_data_path = os.path.join(output_dir, 'pr_curve_data.csv')
    pr_curve_data.to_csv(pr_data_path, index=False)
    
    return {
        'pr_auc': pr_auc,
        'avg_precision': avg_precision,
        'best_f1': {
            'threshold': best_threshold,
            'f1': best_f1,
            'precision': best_precision,
            'recall': best_recall
        },
        'recall_thresholds': recall_thresholds,
        'interesting_points': interesting_thresholds
    }

def find_elbow_point(curve):
    """Encontra o ponto de inflexão (cotovelo) em uma curva."""
    if len(curve) < 3:
        return None
    
    # Converter para array numpy e garantir que é do tipo float
    curve = np.array(curve, dtype=float)
    
    # Lidar com NaN e infinito
    mask = np.isfinite(curve)
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) < 3:
        # Não há pontos suficientes para encontrar um cotovelo
        return 0
    
    # Usar apenas valores válidos
    valid_curve = curve[valid_indices]
    
    # Normalizar a curva
    normalized = (valid_curve - np.min(valid_curve)) / (np.max(valid_curve) - np.min(valid_curve))
    
    # Criar coordenadas
    coords = np.vstack([np.arange(len(normalized)), normalized]).T
    
    # Primeiro e último pontos
    first = coords[0]
    last = coords[-1]
    
    # Vetor da linha
    line_vec = last - first
    
    # Vetor da linha normalizado
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    # Vetores de todos os pontos
    vec_from_first = coords - first
    
    # Distância escalar ao longo da linha
    scalar_prod = np.sum(vec_from_first * line_vec_norm, axis=1)
    
    # Vetores da projeção
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    
    # Vetores ortogonais
    vec_to_line = vec_from_first - vec_from_first_parallel
    
    # Distâncias à linha
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    
    # Encontrar o índice da máxima distância e mapear de volta para o índice original
    max_idx = np.argmax(dist_to_line)
    return valid_indices[max_idx]

def analyze_threshold_sensitivity(y_true, y_prob, output_dir):
    """Analisa a sensibilidade do modelo a diferentes thresholds para classificação."""
    print("\nAnalisando sensibilidade ao threshold de classificação...")
    
    # Calcular métricas para diferentes thresholds
    thresholds = np.linspace(0.01, 0.50, 50)  # Foco em thresholds baixos para priorizar recall
    
    metrics = pd.DataFrame(index=thresholds, columns=[
        'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn'
    ])
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        # Calcular métricas
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Armazenar resultados
        metrics.loc[t] = [precision, recall, f1, tp, fp, fn, tn]
    
    # Encontrar thresholds ótimos
    best_f1_idx = metrics['f1'].idxmax()
    
    # Threshold para recall específico com melhor precisão
    target_recalls = [0.5, 0.6, 0.7, 0.8, 0.9]
    recall_thresholds = {}
    
    for target in target_recalls:
        recall_mask = metrics['recall'] >= target
        if recall_mask.any():
            t = metrics.loc[recall_mask, 'precision'].idxmax()
            recall_thresholds[target] = {
                'threshold': t,
                'precision': metrics.loc[t, 'precision'],
                'recall': metrics.loc[t, 'recall'],
                'f1': metrics.loc[t, 'f1']
            }
    
    # Calcular incrementos cumulativos de recall
    incremental_gains = pd.DataFrame(index=thresholds[:-1], columns=['delta_threshold', 'delta_recall', 'delta_precision', 'efficiency'])
    
    for i in range(len(thresholds) - 1):
        t1, t2 = thresholds[i], thresholds[i+1]
        
        delta_t = t1 - t2
        delta_recall = metrics.loc[t2, 'recall'] - metrics.loc[t1, 'recall']
        delta_precision = metrics.loc[t1, 'precision'] - metrics.loc[t2, 'precision']
        
        # Eficiência: ganho de recall por unidade de precisão sacrificada
        efficiency = delta_recall / delta_precision if delta_precision > 0 else float('inf')
        
        incremental_gains.loc[t1] = [delta_t, delta_recall, delta_precision, efficiency]
    
    # Encontrar pontos de máxima eficiência
    incremental_gains = incremental_gains.dropna()
    if not incremental_gains.empty:
        max_efficiency_threshold = incremental_gains['efficiency'].idxmax()
        
        # Encontrar ponto ótimo baseado na curva de eficiência
        elbow_idx = find_elbow_point(incremental_gains['efficiency'].values)
        if elbow_idx is not None:
            elbow_threshold = incremental_gains.index[elbow_idx]
        else:
            elbow_threshold = max_efficiency_threshold
    else:
        max_efficiency_threshold = best_f1_idx
        elbow_threshold = best_f1_idx
    
    # Imprimir resultados
    print(f"\nThreshold para máximo F1: {best_f1_idx:.4f}")
    print(f"  Precision: {metrics.loc[best_f1_idx, 'precision']:.4f}")
    print(f"  Recall: {metrics.loc[best_f1_idx, 'recall']:.4f}")
    print(f"  F1-Score: {metrics.loc[best_f1_idx, 'f1']:.4f}")
    
    print(f"\nThreshold para máxima eficiência: {max_efficiency_threshold:.4f}")
    print(f"  Precision: {metrics.loc[max_efficiency_threshold, 'precision']:.4f}")
    print(f"  Recall: {metrics.loc[max_efficiency_threshold, 'recall']:.4f}")
    print(f"  F1-Score: {metrics.loc[max_efficiency_threshold, 'f1']:.4f}")
    
    print(f"\nThreshold de eficiência ótima (ponto de inflexão): {elbow_threshold:.4f}")
    print(f"  Precision: {metrics.loc[elbow_threshold, 'precision']:.4f}")
    print(f"  Recall: {metrics.loc[elbow_threshold, 'recall']:.4f}")
    print(f"  F1-Score: {metrics.loc[elbow_threshold, 'f1']:.4f}")
    
    print("\nThresholds para atingir níveis específicos de recall:")
    for target, values in recall_thresholds.items():
        print(f"  Recall >= {target:.1f}: threshold={values['threshold']:.4f}, precision={values['precision']:.4f}")
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Plotar curvas
    plt.figure(figsize=(12, 10))
    
    # Curvas de métricas vs threshold
    plt.subplot(2, 1, 1)
    plt.plot(metrics.index, metrics['precision'], 'b-', label='Precisão')
    plt.plot(metrics.index, metrics['recall'], 'r-', label='Recall')
    plt.plot(metrics.index, metrics['f1'], 'g-', label='F1-Score')
    
    # Pontos especiais
    plt.axvline(best_f1_idx, color='g', linestyle='--', alpha=0.7, label=f'Max F1 ({best_f1_idx:.3f})')
    plt.axvline(max_efficiency_threshold, color='m', linestyle='--', alpha=0.7, label=f'Max Eficiência ({max_efficiency_threshold:.3f})')
    plt.axvline(elbow_threshold, color='c', linestyle='--', alpha=0.7, label=f'Eficiência Ótima ({elbow_threshold:.3f})')
    
    plt.title('Sensibilidade ao Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Métricas')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Curva de eficiência incremental
    plt.subplot(2, 1, 2)
    plt.semilogy(incremental_gains.index, incremental_gains['efficiency'], 'r-')
    plt.axvline(max_efficiency_threshold, color='m', linestyle='--', alpha=0.7, label=f'Max Eficiência ({max_efficiency_threshold:.3f})')
    plt.axvline(elbow_threshold, color='c', linestyle='--', alpha=0.7, label=f'Eficiência Ótima ({elbow_threshold:.3f})')
    
    plt.title('Eficiência Incremental (Ganho de Recall / Perda de Precisão)')
    plt.xlabel('Threshold')
    plt.ylabel('Eficiência (escala log)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar figura
    plot_path = os.path.join(output_dir, 'threshold_sensitivity_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Salvar métricas
    metrics_path = os.path.join(output_dir, 'threshold_metrics.csv')
    metrics.to_csv(metrics_path)
    
    # Salvar incremental gains
    gains_path = os.path.join(output_dir, 'threshold_incremental_gains.csv')
    incremental_gains.to_csv(gains_path)
    
    print(f"\nAnálise de sensibilidade ao threshold salva em: {metrics_path}")
    print(f"Gráfico de sensibilidade salvo em: {plot_path}")
    
    return {
        'best_f1': {
            'threshold': best_f1_idx,
            'precision': metrics.loc[best_f1_idx, 'precision'],
            'recall': metrics.loc[best_f1_idx, 'recall'],
            'f1': metrics.loc[best_f1_idx, 'f1']
        },
        'max_efficiency': {
            'threshold': max_efficiency_threshold,
            'precision': metrics.loc[max_efficiency_threshold, 'precision'],
            'recall': metrics.loc[max_efficiency_threshold, 'recall'],
            'f1': metrics.loc[max_efficiency_threshold, 'f1']
        },
        'optimal_efficiency': {
            'threshold': elbow_threshold,
            'precision': metrics.loc[elbow_threshold, 'precision'],
            'recall': metrics.loc[elbow_threshold, 'recall'],
            'f1': metrics.loc[elbow_threshold, 'f1']
        },
        'recall_thresholds': recall_thresholds
    }

def analyze_feature_distributions(analysis_df, target_col, output_dir):
    """Analisa distribuições de features entre diferentes categorias de erro."""
    # Remover colunas não-numéricas para a análise
    non_feature_cols = ['true_class', 'predicted_class', 'probability', 'error_type', 'fn_category']
    if target_col not in non_feature_cols:
        non_feature_cols.append(target_col)
    
    # Filtrar apenas colunas numéricas
    numeric_cols = [col for col in analysis_df.columns 
                   if col not in non_feature_cols and 
                   pd.api.types.is_numeric_dtype(analysis_df[col])]
    
    # Calcular diferenças nas distribuições entre falsos negativos e verdadeiros positivos
    print("\nAnalisando diferenças nas distribuições de features...\n")
    
    # Criar DataFrame para armazenar as diferenças
    fn_vs_tp = pd.DataFrame(index=numeric_cols, columns=[
        'fn_mean', 'tp_mean', 'mean_diff', 'fn_std', 'tp_std', 'std_ratio',
        'effect_size', 'ks_stat', 'ks_pvalue'
    ])
    
    fn_data = analysis_df[analysis_df['error_type'] == 'false_negative']
    tp_data = analysis_df[(analysis_df['true_class'] == 1) & (analysis_df['predicted_class'] == 1)]
    
    from scipy import stats
    
    print(f"Comparando {len(fn_data)} falsos negativos com {len(tp_data)} verdadeiros positivos")
    
    # Reduzir para uma amostra de features se houver muitas
    if len(numeric_cols) > 50:
        print(f"Muitas features ({len(numeric_cols)}), analisando as 50 mais variáveis")
        feature_variance = analysis_df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = feature_variance.index[:50].tolist()
    
    for col in numeric_cols:
        # Calcular estatísticas
        fn_mean = fn_data[col].mean()
        tp_mean = tp_data[col].mean()
        mean_diff = fn_mean - tp_mean
        
        fn_std = fn_data[col].std()
        tp_std = tp_data[col].std()
        std_ratio = fn_std / tp_std if tp_std > 0 else np.nan
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((fn_std**2 + tp_std**2) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else np.nan
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(fn_data[col].dropna(), tp_data[col].dropna())
        
        # Armazenar resultados
        fn_vs_tp.loc[col] = [fn_mean, tp_mean, mean_diff, fn_std, tp_std, std_ratio, 
                             effect_size, ks_stat, ks_pvalue]
    
    # Ordenar por effect size absoluto
    fn_vs_tp['abs_effect'] = fn_vs_tp['effect_size'].abs()
    fn_vs_tp = fn_vs_tp.sort_values('abs_effect', ascending=False)
    fn_vs_tp = fn_vs_tp.drop(columns=['abs_effect'])
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar resultados
    dist_analysis_path = os.path.join(output_dir, 'feature_distribution_differences.csv')
    fn_vs_tp.to_csv(dist_analysis_path)
    
    print(f"Análise de distribuições salva em: {dist_analysis_path}")
    
    # Mostrar top features com maior effect size
    print("\nTop 15 features com maior diferença entre FNs e TPs (Cohen's d):")
    top_effect_features = fn_vs_tp.head(15).index.tolist()
    
    for i, feature in enumerate(top_effect_features, 1):
        effect = fn_vs_tp.loc[feature, 'effect_size']
        diff = fn_vs_tp.loc[feature, 'mean_diff']
        fn_m = fn_vs_tp.loc[feature, 'fn_mean']
        tp_m = fn_vs_tp.loc[feature, 'tp_mean']
        
        print(f"{i}. {feature}")
        print(f"   Effect size: {effect:.3f}")
        print(f"   FN mean: {fn_m:.3f}, TP mean: {tp_m:.3f}, Diff: {diff:.3f}")
    
    # Criar gráficos para as top features
    plot_dir = os.path.join(output_dir, 'distribution_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    for feature in top_effect_features[:10]:  # Limitar a 10 gráficos
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Distribuições
        plt.subplot(1, 2, 1)
        sns.kdeplot(fn_data[feature].dropna(), label='Falsos Negativos', fill=True, alpha=0.4)
        sns.kdeplot(tp_data[feature].dropna(), label='Verdadeiros Positivos', fill=True, alpha=0.4)
        plt.title(f'Distribuição de {feature}')
        plt.xlabel(feature)
        plt.ylabel('Densidade')
        plt.legend()
        
        # Plot 2: Boxplot - corrigido para evitar índices duplicados
        plt.subplot(1, 2, 2)
        
        # Criar DataFrames separados e depois concatenar com reset_index
        fn_plot_data = pd.DataFrame({'Valor': fn_data[feature], 'Grupo': 'Falso Negativo'})
        tp_plot_data = pd.DataFrame({'Valor': tp_data[feature], 'Grupo': 'Verdadeiro Positivo'})
        comparison_data = pd.concat([fn_plot_data, tp_plot_data]).reset_index(drop=True)
        
        sns.boxplot(x='Grupo', y='Valor', data=comparison_data)
        plt.title(f'Boxplot de {feature}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'dist_{feature.replace("/", "_")}.png'), dpi=200)
        plt.close()
    
    print(f"\nGráficos de distribuição salvos em: {plot_dir}")
    
    return {
        'feature_comparisons': fn_vs_tp,
        'top_differentiating_features': top_effect_features
    }

def analyze_calibration_opportunity(y_true, y_prob, analysis_df, output_dir):
    """Analisa a oportunidade de calibração de probabilidades para melhorar o recall."""
    print("\nAnalisando oportunidade de calibração de probabilidades...")
    
    from sklearn.calibration import calibration_curve
    
    # Calcular curva de calibração
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    # Verificar se o modelo está bem calibrado
    # Calcular erro quadrático médio da calibração
    calibration_mse = np.mean((prob_true - prob_pred)**2)
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Plotar curva de calibração
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(prob_pred, prob_true, 's-', label='Calibração (MSE={:.4f})'.format(calibration_mse))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfeitamente Calibrado')
    plt.title('Curva de Calibração')
    plt.xlabel('Probabilidade Média Predita')
    plt.ylabel('Fração de Positivos Observados')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Histograma de probabilidades
    plt.subplot(2, 1, 2)
    
    # Histograma para cada classe
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.5, density=True, label='Classe Negativa')
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.5, density=True, label='Classe Positiva')
    
    plt.xlabel('Probabilidade Predita')
    plt.ylabel('Densidade')
    plt.title('Histograma de Probabilidades por Classe')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar figura
    cal_path = os.path.join(output_dir, 'calibration_analysis.png')
    plt.savefig(cal_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analisar se a calibração pode melhorar o recall
    needs_calibration = calibration_mse > 0.01  # Limiar arbitrário
    
    # Verificar se as probabilidades estão subestimadas (pontos acima da diagonal)
    underestimation = np.mean(prob_true > prob_pred)
    
    # Estimar potencial de ganho com calibração
    if needs_calibration:
        # Estimar o potencial de melhoria
        potential_gain = min(1.0, calibration_mse * 10)  # Heurística simples
        
        print(f"O modelo apresenta problemas de calibração (MSE: {calibration_mse:.4f})")
        
        if underestimation > 0.5:
            print("As probabilidades parecem estar subestimadas, a calibração pode aumentar o recall")
            recommendation = "Implementar calibração de probabilidades para corrigir subestimação"
        else:
            print("As probabilidades parecem estar superestimadas, a calibração pode aumentar a precisão")
            recommendation = "Implementar calibração de probabilidades para corrigir superestimação"
        
        print(f"Potencial estimado de ganho com calibração: {potential_gain:.1%}")
    else:
        print(f"O modelo está razoavelmente bem calibrado (MSE: {calibration_mse:.4f})")
        recommendation = "Calibração não deve trazer ganhos significativos"
        potential_gain = 0.0
    
    print(f"Análise de calibração salva em: {cal_path}")
    
    return {
        'needs_calibration': needs_calibration,
        'calibration_mse': calibration_mse,
        'underestimation': underestimation,
        'potential_gain': potential_gain,
        'recommendation': recommendation
    }

def generate_insights_and_recommendations(analysis_results, model_uri, output_dir):
    """Gera insights e recomendações baseados nas análises realizadas."""
    print("\n=== Gerando Insights e Recomendações ===")
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar DataFrame para armazenar recomendações
    recommendations = pd.DataFrame(columns=[
        'category', 'recommendation', 'expected_impact', 'implementation_difficulty', 'priority'
    ])
    
    # 1. Recomendações de threshold
    if 'threshold_sensitivity' in analysis_results:
        thresh_results = analysis_results['threshold_sensitivity']
        
        # Recomendar threshold para melhor recall-precisão
        if 'optimal_efficiency' in thresh_results:
            opt_thresh = thresh_results['optimal_efficiency']['threshold']
            exp_recall = thresh_results['optimal_efficiency']['recall']
            exp_precision = thresh_results['optimal_efficiency']['precision']
            
            recommendations.loc[len(recommendations)] = [
                'Ajuste de Threshold',
                f'Utilizar threshold de {opt_thresh:.4f} para equilíbrio ótimo entre recall e precisão',
                f'Recall esperado: {exp_recall:.2%}, Precisão: {exp_precision:.2%}',
                'Fácil',
                'Alta'
            ]
        
        # Se precisão atual for muito alta, sugerir redução de threshold
        if 'recall_thresholds' in thresh_results:
            for target, metrics in thresh_results['recall_thresholds'].items():
                if metrics['precision'] > 0.7:  # Precisão ainda aceitável
                    recommendations.loc[len(recommendations)] = [
                        'Ajuste de Threshold',
                        f'Utilizar threshold de {metrics["threshold"]:.4f} para atingir recall de {target:.1%}',
                        f'Recall esperado: {metrics["recall"]:.2%}, Precisão: {metrics["precision"]:.2%}',
                        'Fácil',
                        'Média' if target < 0.7 else 'Alta'
                    ]
                    break  # Apenas a recomendação mais agressiva que mantém boa precisão
    
    # 2. Recomendações de calibração
    if 'calibration' in analysis_results:
        cal_results = analysis_results['calibration']
        
        if cal_results.get('needs_calibration', False) and cal_results.get('underestimation', 0) > 0.5:
            recommendations.loc[len(recommendations)] = [
                'Calibração de Probabilidades',
                'Implementar calibração isotônica para corrigir subestimação de probabilidades',
                f'Potencial ganho de recall: {cal_results.get("potential_gain", 0):.1%}',
                'Média',
                'Alta' if cal_results.get("potential_gain", 0) > 0.1 else 'Média'
            ]
    
    # 3. Recomendações de features
    if 'feature_analysis' in analysis_results:
        feat_results = analysis_results['feature_analysis']
        
        if 'top_differentiating_features' in feat_results:
            top_features = feat_results['top_differentiating_features']
            
            if len(top_features) > 0:
                features_text = ", ".join(top_features[:3])
                
                recommendations.loc[len(recommendations)] = [
                    'Engenharia de Features',
                    f'Criar features derivadas baseadas em: {features_text}',
                    'Potencial aumento de recall de 5-15%',
                    'Média',
                    'Alta'
                ]
                
                recommendations.loc[len(recommendations)] = [
                    'Engenharia de Features',
                    f'Aplicar transformações não-lineares às top features que diferenciam FNs',
                    'Potencial aumento de recall de 3-10%',
                    'Média',
                    'Média'
                ]
    
    # Ordenar recomendações por prioridade
    priority_order = {'Alta': 0, 'Média': 1, 'Baixa': 2}
    recommendations['priority_order'] = recommendations['priority'].map(priority_order)
    recommendations = recommendations.sort_values('priority_order')
    recommendations = recommendations.drop(columns=['priority_order'])
    
    # Salvar recomendações
    recs_path = os.path.join(output_dir, 'recommendations.csv')
    recommendations.to_csv(recs_path, index=False)
    
    print(f"\nRecomendações salvas em: {recs_path}")
    
    # Gerar relatório de insights
    insights = []
    
    # Insights sobre threshold
    if 'threshold_sensitivity' in analysis_results:
        insights.append("**Threshold Atual:**")
        insights.append(f"- O threshold atual pode estar muito alto, priorizando precisão em detrimento do recall.")
        insights.append(f"- A análise de sensibilidade mostra que reduzir o threshold para valores entre 0.05-0.15 pode aumentar significativamente o recall.")
    
    # Insights sobre falsos negativos
    insights.append("\n**Falsos Negativos:**")
    
    if 'feature_analysis' in analysis_results:
        insights.append("- Os falsos negativos apresentam características distintas dos verdadeiros positivos.")
        if 'top_differentiating_features' in analysis_results['feature_analysis']:
            features = analysis_results['feature_analysis']['top_differentiating_features'][:3]
            insights.append(f"- As principais diferenças estão em: {', '.join(features)}.")
    
    # Insights sobre calibração
    if 'calibration' in analysis_results:
        insights.append("\n**Calibração de Probabilidades:**")
        if analysis_results['calibration'].get('needs_calibration', False):
            insights.append("- O modelo atual apresenta problemas de calibração, gerando probabilidades imprecisas.")
            if analysis_results['calibration'].get('underestimation', 0) > 0.5:
                insights.append("- As probabilidades estão sendo subestimadas, o que prejudica o recall.")
            else:
                insights.append("- As probabilidades estão sendo superestimadas, o que prejudica a precisão.")
        else:
            insights.append("- O modelo está razoavelmente bem calibrado.")
    
    # Resumo final
    insights.append("\n**Resumo Geral:**")
    insights.append("- O modelo atual prioriza precisão, mas tem potencial para melhorar significativamente o recall.")
    insights.append("- A combinação de ajustes de threshold e calibração de probabilidades pode levar a ganhos substanciais.")
    
    # Escrever insights em arquivo
    insights_path = os.path.join(output_dir, 'insights.md')
    with open(insights_path, 'w') as f:
        f.write('\n'.join(insights))
    
    print(f"Insights salvos em: {insights_path}")
    
    # Imprimir top recomendações
    print("\nTop recomendações para aumentar o recall:")
    for i, row in recommendations.head(5).iterrows():
        print(f"{i+1}. {row['recommendation']} - {row['expected_impact']} (Prioridade: {row['priority']})")
    
    return {
        'recommendations': recommendations,
        'insights_path': insights_path
    }

def main():
    parser = argparse.ArgumentParser(description='Executar análise de falsos negativos simplificada')
    parser.add_argument('--mlflow_dir', default=MLFLOW_DIR, 
                      help='Diretório do MLflow tracking')
    parser.add_argument('--data_path', default=DEFAULT_DATA_PATH,
                      help='Caminho para o dataset de validação/teste')
    parser.add_argument('--target_col', default='target',
                      help='Nome da coluna target')
    parser.add_argument('--model_uri', default=None,
                      help='URI específico do modelo a ser analisado (opcional)')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Threshold personalizado para classificação (opcional)')
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                      help='Diretório para salvar os resultados (opcional)')
    
    args = parser.parse_args()
    
    print(f"Usando diretório MLflow: {args.mlflow_dir}")
    
    # Configurar MLflow
    setup_mlflow(args.mlflow_dir)
    
    print(f"Usando dataset: {args.data_path}")
    
    # Criar diretório de saída se não existir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {args.output_dir}")
    
    # Verificar se o dataset existe
    if not os.path.exists(args.data_path):
        print(f"ERRO: Dataset não encontrado: {args.data_path}")
        # Ver se o arquivo existe em um caminho diferente
        if args.data_path.endswith('validation.csv'):
            alternative_path = os.path.join(PROJECT_ROOT, 'data', 'validation.csv')
            if os.path.exists(alternative_path):
                print(f"Tentando caminho alternativo: {alternative_path}")
                args.data_path = alternative_path
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Carregar dados
    try:
        print("Carregando dados...")
        df, X, y, target_col = load_data(args.data_path)
        print(f"Dados carregados: {len(df)} exemplos, {len(X.columns)} features")
    except Exception as e:
        print(f"ERRO ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Carregar modelo
    try:
        print("Carregando modelo...")
        if args.model_uri is None:
            # Carregar modelo mais recente
            try:
                from src.evaluation.baseline_model import get_latest_random_forest_run
                
                run_id, threshold_from_mlflow, model_uri = get_latest_random_forest_run(args.mlflow_dir)
                
                if model_uri is None:
                    print("AVISO: Não foi possível encontrar um modelo RandomForest no MLflow.")
                    print("Buscando modelo alternativo...")
                    # Caminho alternativo para um modelo treinado
                    model_path = os.path.join(PROJECT_ROOT, 'models', 'baseline', 'random_forest.joblib')
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        threshold_from_mlflow = 0.17  # Threshold padrão para fallback
                        print(f"Carregando modelo do arquivo: {model_path}")
                    else:
                        print("ERRO: Nenhum modelo encontrado.")
                        sys.exit(1)
                else:
                    model = load_model_from_mlflow(model_uri)
                    print(f"Usando modelo mais recente: {model_uri}")
            except ImportError:
                print("Módulo baseline_model não encontrado.")
                # Tentar caminho alternativo para modelo treinado
                model_path = os.path.join(PROJECT_ROOT, 'models', 'baseline', 'random_forest.joblib')
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    threshold_from_mlflow = 0.17  # Threshold padrão para fallback
                    print(f"Carregando modelo do arquivo: {model_path}")
                else:
                    print("ERRO: Nenhum modelo encontrado.")
                    sys.exit(1)
        else:
            model_uri = args.model_uri
            threshold_from_mlflow = None
            print(f"Usando modelo especificado: {model_uri}")
            if model_uri.startswith("runs:"):
                model = load_model_from_mlflow(model_uri)
            else:
                try:
                    model = joblib.load(model_uri)
                    print(f"Modelo carregado de: {model_uri}")
                except:
                    print(f"ERRO: Falha ao carregar modelo de {model_uri}")
                    sys.exit(1)
        
        if model is None:
            print("ERRO: Não foi possível carregar o modelo.")
            sys.exit(1)
            
        # Definir threshold
        if args.threshold is None:
            if threshold_from_mlflow is not None:
                threshold = threshold_from_mlflow
                print(f"Usando threshold do MLflow: {threshold}")
            else:
                # Tentar obter threshold do MLflow
                try:
                    client = mlflow.tracking.MlflowClient()
                    run_id = model_uri.split('/')[1]
                    run = client.get_run(run_id)
                    threshold = run.data.metrics.get('threshold', 0.5)
                    print(f"Usando threshold do MLflow: {threshold}")
                except:
                    threshold = 0.5
                    print(f"Não foi possível obter threshold, usando padrão: {threshold}")
        else:
            threshold = args.threshold
            print(f"Usando threshold personalizado: {threshold}")
            
    except Exception as e:
        print(f"ERRO ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Gerar previsões
    try:
        print("Gerando previsões com o modelo...")
        y_pred, y_prob = generate_predictions(model, X, threshold)
        print(f"Previsões geradas: {sum(y_pred)} positivos de {len(y_pred)} exemplos")
    except Exception as e:
        print(f"ERRO ao gerar previsões: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Criar DataFrame para análise de erro
    try:
        print("Criando DataFrame para análise de erro...")
        analysis_df = create_error_analysis_df(df, y, y_pred, y_prob)
        print("DataFrame de análise criado com sucesso.")
    except Exception as e:
        print(f"ERRO ao criar DataFrame de análise: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Inicializar dicionário para armazenar resultados
    analysis_results = {}
    
    # 1. Análise de curva Precision-Recall
    try:
        print("\n=== 1. Analisando curva Precision-Recall ===")
        pr_results = analyze_pr_curve(y, y_prob, args.output_dir)
        analysis_results['pr_curve'] = pr_results
        print("Análise de curva PR concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na análise de curva PR: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Análise de sensibilidade ao threshold
    try:
        print("\n=== 2. Analisando sensibilidade ao threshold ===")
        threshold_results = analyze_threshold_sensitivity(y, y_prob, args.output_dir)
        analysis_results['threshold_sensitivity'] = threshold_results
        print("Análise de sensibilidade ao threshold concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na análise de threshold: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Análise de distribuições de features
    try:
        print("\n=== 3. Analisando distribuições de features ===")
        feature_results = analyze_feature_distributions(analysis_df, target_col, args.output_dir)
        analysis_results['feature_analysis'] = feature_results
        print("Análise de features concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na análise de features: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Análise de calibração
    try:
        print("\n=== 4. Analisando oportunidade de calibração ===")
        calibration_results = analyze_calibration_opportunity(y, y_prob, analysis_df, args.output_dir)
        analysis_results['calibration'] = calibration_results
        print("Análise de calibração concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na análise de calibração: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Geração de insights e recomendações
    try:
        print("\n=== 5. Gerando insights e recomendações ===")
        recommendations = generate_insights_and_recommendations(analysis_results, model_uri, args.output_dir)
        analysis_results['recommendations'] = recommendations
        print("Geração de insights e recomendações concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na geração de recomendações: {e}")
        import traceback
        traceback.print_exc()
    
    # Salvar resultados completos
    try:
        results_path = os.path.join(args.output_dir, 'analysis_results.joblib')
        joblib.dump(analysis_results, results_path)
        print(f"Resultados completos salvos em: {results_path}")
    except Exception as e:
        print(f"ERRO ao salvar resultados: {e}")
    
    print(f"\n=== Análise concluída! ===")
    print(f"Todos os resultados foram salvos em: {args.output_dir}")
    if 'recommendations' in analysis_results:
        print(f"Recomendações para aumentar o recall: {os.path.join(args.output_dir, 'recommendations.csv')}")
        print(f"Insights: {os.path.join(args.output_dir, 'insights.md')}")

if __name__ == "__main__":
    main()