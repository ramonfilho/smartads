#!/usr/bin/env python
"""
Script para análise avançada de falsos negativos com foco em melhorar o recall.
Este script realiza uma análise detalhada dos falsos negativos para encontrar
padrões e características que possam ser utilizados para melhorar o recall
mantendo uma alta precisão.
"""

import os
import sys
try:
    import argparse
    import pandas as pd
    import numpy as np
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import mlflow
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    print(f"ERRO ao importar bibliotecas: {e}")
    sys.exit(1)

# Adicionar diretório raiz ao path
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    print(f"Diretório raiz adicionado ao path: {project_root}")
except Exception as e:
    print(f"ERRO ao configurar path: {e}")
    sys.exit(1)

def setup_mlflow(mlflow_dir):
    """Configura o MLflow para tracking de experimentos."""
    if os.path.exists(mlflow_dir):
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        print(f"MLflow configurado para usar: {mlflow.get_tracking_uri()}")
    else:
        print(f"AVISO: Diretório MLflow não encontrado: {mlflow_dir}")
        print("MLflow será configurado para usar o diretório local.")
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")

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
    
    # Importar funções do módulo baseline_model
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
    
    # Adicionar gradações dentro dos falsos negativos
    fn_indices = analysis_df[analysis_df['error_type'] == 'false_negative'].index
    fn_probs = analysis_df.loc[fn_indices, 'probability']
    
    # Dividir FNs em 3 categorias: longe, médio e perto do threshold
    if len(fn_probs) > 0:
        # Usar quantis para divisão
        quantiles = fn_probs.quantile([0.33, 0.67])
        
        analysis_df['fn_category'] = np.nan
        analysis_df.loc[fn_indices, 'fn_category'] = 'marginal_fn'  # Padrão
        
        # Falsos negativos longe do threshold (baixa probabilidade)
        far_indices = fn_indices[fn_probs < quantiles[0.33]]
        analysis_df.loc[far_indices, 'fn_category'] = 'distant_fn'
        
        # Falsos negativos próximos ao threshold
        close_indices = fn_indices[fn_probs > quantiles[0.67]]
        analysis_df.loc[close_indices, 'fn_category'] = 'near_threshold_fn'
    
    # Mostrar distribuição de categorias
    print("\nDistribuição de categorias de predição:")
    print(analysis_df['error_type'].value_counts().to_frame().T)
    
    if 'fn_category' in analysis_df.columns:
        print("\nDistribuição de categorias de falsos negativos:")
        fn_counts = analysis_df['fn_category'].value_counts()
        print(fn_counts.to_frame().T)
    
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
    
    # Análise por categoria de FN
    if 'fn_category' in analysis_df.columns and len(fn_data) > 0:
        print("\nAnalisando diferenças entre categorias de falsos negativos...")
        
        fn_categories = analysis_df['fn_category'].dropna().unique()
        
        if len(fn_categories) > 1:
            # Análise de variância entre categorias de FN
            fn_category_analysis = pd.DataFrame(index=numeric_cols)
            
            for feature in numeric_cols:
                # Preparar dados para ANOVA
                feature_data = []
                for category in fn_categories:
                    values = analysis_df[analysis_df['fn_category'] == category][feature].dropna()
                    feature_data.append(values)
                
                # Verificar se há dados suficientes
                if all(len(d) > 1 for d in feature_data):
                    try:
                        f_val, p_val = stats.f_oneway(*feature_data)
                        fn_category_analysis.loc[feature, 'f_value'] = f_val
                        fn_category_analysis.loc[feature, 'p_value'] = p_val
                        
                        # Médias por categoria
                        for category in fn_categories:
                            cat_mean = analysis_df[analysis_df['fn_category'] == category][feature].mean()
                            fn_category_analysis.loc[feature, f'mean_{category}'] = cat_mean
                    except:
                        pass
            
            # Ordenar por significância
            if 'p_value' in fn_category_analysis.columns:
                fn_category_analysis = fn_category_analysis.sort_values('p_value')
                
                # Salvar resultados
                fn_cat_path = os.path.join(output_dir, 'fn_category_analysis.csv')
                fn_category_analysis.to_csv(fn_cat_path)
                
                print(f"Análise de categorias de FN salva em: {fn_cat_path}")
                
                # Mostrar top features que diferenciam categorias de FN
                print("\nTop 10 features que diferenciam categorias de falsos negativos:")
                for i, feature in enumerate(fn_category_analysis.head(10).index, 1):
                    p_val = fn_category_analysis.loc[feature, 'p_value']
                    
                    print(f"{i}. {feature} (p-value: {p_val:.5f})")
                    # Mostrar médias por categoria
                    for category in fn_categories:
                        col_name = f'mean_{category}'
                        if col_name in fn_category_analysis.columns:
                            mean_val = fn_category_analysis.loc[feature, col_name]
                            print(f"   {category}: {mean_val:.3f}")
    
    return {
        'feature_comparisons': fn_vs_tp,
        'top_differentiating_features': top_effect_features
    }

def perform_clustering_analysis(analysis_df, target_col, output_dir):
    """Realiza análise de clustering nos falsos negativos e falsos positivos."""
    print("\nIniciando análise de clustering...")
    
    # Selecionar apenas falsos negativos
    fn_data = analysis_df[analysis_df['error_type'] == 'false_negative'].copy()
    
    if len(fn_data) < 10:
        print("Número insuficiente de falsos negativos para clustering")
        return None
    
    # Remover colunas não-numéricas para o clustering
    non_feature_cols = ['true_class', 'predicted_class', 'probability', 'error_type', 'fn_category']
    if target_col not in non_feature_cols:
        non_feature_cols.append(target_col)
    
    # Colunas numéricas para clustering
    numeric_cols = [col for col in fn_data.columns 
                   if col not in non_feature_cols and 
                   pd.api.types.is_numeric_dtype(fn_data[col])]
    
    fn_data_filled = fn_data[numeric_cols].fillna(fn_data[numeric_cols].mean())

    # Aplicar escalamento
    scaler = StandardScaler()
    fn_scaled = pd.DataFrame(
        scaler.fit_transform(fn_data_filled),
        columns=numeric_cols,
        index=fn_data.index
    )
    
    # Aplicar PCA para redução de dimensionalidade (se necessário)
    if len(numeric_cols) > 30:
        print(f"Aplicando PCA para reduzir dimensionalidade de {len(numeric_cols)} para 30 componentes")
        pca = PCA(n_components=30)
        fn_pca = pd.DataFrame(
            pca.fit_transform(fn_scaled),
            columns=[f'PC{i+1}' for i in range(30)],
            index=fn_data.index
        )
        
        # Salvar variância explicada
        explained_var = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'Explained_Variance': pca.explained_variance_ratio_,
            'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        
        explained_var_path = os.path.join(output_dir, 'pca_explained_variance.csv')
        explained_var.to_csv(explained_var_path, index=False)
        
        print(f"Variância explicada: {explained_var['Cumulative_Variance'].iloc[29]:.2%} com 30 componentes")
        
        # Usando os componentes principais para clustering
        data_for_clustering = fn_pca
    else:
        # Usar dados escalados diretamente
        data_for_clustering = fn_scaled
    
    # K-Means Clustering
    n_clusters_to_try = min(5, len(fn_data) // 10) if len(fn_data) >= 50 else min(3, len(fn_data) // 5)
    n_clusters_to_try = max(2, n_clusters_to_try)  # No mínimo 2 clusters
    
    print(f"Executando K-Means com {n_clusters_to_try} clusters...")
    kmeans = KMeans(n_clusters=n_clusters_to_try, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_for_clustering)
    
    # Adicionar labels ao DataFrame original
    fn_data['cluster'] = cluster_labels
    
    # Analisar características de cada cluster
    cluster_stats = pd.DataFrame(index=range(n_clusters_to_try))
    
    print("\nCaracterísticas dos clusters de falsos negativos:")
    for cluster_id in range(n_clusters_to_try):
        cluster_members = fn_data[fn_data['cluster'] == cluster_id]
        cluster_size = len(cluster_members)
        cluster_prob_mean = cluster_members['probability'].mean()
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Tamanho: {cluster_size} ({cluster_size/len(fn_data):.1%} dos FNs)")
        print(f"  Probabilidade média: {cluster_prob_mean:.4f}")
        
        # Salvar estatísticas do cluster
        cluster_stats.loc[cluster_id, 'size'] = cluster_size
        cluster_stats.loc[cluster_id, 'proportion'] = cluster_size/len(fn_data)
        cluster_stats.loc[cluster_id, 'prob_mean'] = cluster_prob_mean
        
        # Encontrar as features mais distintivas para cada cluster
        if len(numeric_cols) > 0:
            cluster_means = fn_data.groupby('cluster')[numeric_cols].mean()
            global_means = fn_data[numeric_cols].mean()
            
            # Calcular diferenças normalizadas
            diff = (cluster_means.loc[cluster_id] - global_means) / global_means.replace(0, 1)
            top_features = diff.abs().sort_values(ascending=False).head(5)
            
            print("  Features mais distintivas:")
            for feature, value in top_features.items():
                cluster_value = cluster_means.loc[cluster_id, feature]
                print(f"    {feature}: {cluster_value:.3f} (diff: {value:+.2%})")
                
                # Salvar no DataFrame de estatísticas
                cluster_stats.loc[cluster_id, f'top_feature_{feature}'] = value
    
    # Salvar estatísticas de cluster
    cluster_stats_path = os.path.join(output_dir, 'fn_cluster_statistics.csv')
    cluster_stats.to_csv(cluster_stats_path)
    
    # Salvar dados com clusters
    fn_cluster_path = os.path.join(output_dir, 'fn_cluster_data.csv')
    fn_data.to_csv(fn_cluster_path)
    
    print(f"\nEstatísticas de clusters salvas em: {cluster_stats_path}")
    print(f"Dados com atribuição de clusters salvos em: {fn_cluster_path}")
    
    # Visualização 2D com t-SNE
    print("\nGerando visualização t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fn_data)//5))
    fn_tsne = tsne.fit_transform(data_for_clustering)
    
    # Criar DataFrame para visualização
    tsne_df = pd.DataFrame({
        'x': fn_tsne[:, 0],
        'y': fn_tsne[:, 1],
        'cluster': cluster_labels,
        'probability': fn_data['probability'].values
    })
    
    # Plotar
    plt.figure(figsize=(12, 10))
    
    # Plot clusters
    plt.subplot(2, 1, 1)
    sns.scatterplot(data=tsne_df, x='x', y='y', hue='cluster', palette='viridis', s=100, alpha=0.7)
    plt.title('Clusters de Falsos Negativos (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Plot probabilidades
    plt.subplot(2, 1, 2)
    scatter = plt.scatter(tsne_df['x'], tsne_df['y'], c=tsne_df['probability'], 
                        cmap='RdYlGn', s=100, alpha=0.7)
    plt.colorbar(scatter, label='Probabilidade')
    plt.title('Probabilidades de Falsos Negativos (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    tsne_path = os.path.join(output_dir, 'fn_tsne_visualization.png')
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualização t-SNE salva em: {tsne_path}")
    
    return {
        'n_clusters': n_clusters_to_try,
        'cluster_stats': cluster_stats,
        'fn_data_with_clusters': fn_data
    }

def train_specialized_model(X, y, analysis_df, output_dir):
    """Treina um modelo especializado para recuperar falsos negativos."""
    print("\nTreinando modelo especializado para detectar falsos negativos...")
    
    # Preparar dataset específico
    # Incluímos verdadeiros positivos e falsos negativos, ambos com target=1
    fn_mask = (analysis_df['error_type'] == 'false_negative')
    tp_mask = (analysis_df['true_class'] == 1) & (analysis_df['predicted_class'] == 1)
    
    # Também incluímos uma amostra aleatória de verdadeiros negativos (target=0)
    tn_mask = (analysis_df['true_class'] == 0) & (analysis_df['predicted_class'] == 0)
    
    # Obter índices
    fn_indices = analysis_df[fn_mask].index
    tp_indices = analysis_df[tp_mask].index
    tn_indices = analysis_df[tn_mask].index
    
    if len(fn_indices) < 10:
        print("Número insuficiente de falsos negativos para treinar modelo especializado")
        return None
    
    # Balancear o conjunto de dados
    n_pos = len(fn_indices) + len(tp_indices)
    n_neg = min(max(n_pos, 2*len(fn_indices)), len(tn_indices))
    
    # Selecionar amostra de verdadeiros negativos
    np.random.seed(42)
    tn_indices_sample = np.random.choice(tn_indices, size=n_neg, replace=False)
    
    # Combinar índices
    train_indices = np.concatenate([fn_indices, tp_indices, tn_indices_sample])
    
    # Criar dataset de treinamento
    X_train = X.loc[train_indices]
    y_train = np.zeros(len(train_indices))
    
    # Marcar verdadeiros positivos e falsos negativos como classe positiva
    y_train[:len(fn_indices) + len(tp_indices)] = 1
    
    print(f"Dataset para modelo especializado: {len(X_train)} exemplos")
    print(f"  - Positivos: {np.sum(y_train == 1)} ({np.mean(y_train == 1):.1%})")
    print(f"  - Negativos: {np.sum(y_train == 0)} ({np.mean(y_train == 0):.1%})")
    
    # Treinar modelo RandomForest especializado
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Executar validação cruzada
    k_folds = min(5, len(fn_indices))
    
    if k_folds >= 2:
        print(f"\nExecutando validação cruzada com {k_folds} folds...")
        
        # Índices para positivos (FN + TP)
        pos_indices = np.arange(len(fn_indices) + len(tp_indices))
        
        # Índices para negativos (TN)
        neg_indices = np.arange(len(fn_indices) + len(tp_indices), len(train_indices))
        
        # Preparar arrays para armazenar resultados
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        
        # Validação cruzada estratificada apenas nos positivos
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_weights = []
        fold_results = []
        
        # Criar y_strat apenas para os positivos para estratificar os folds
        # 1 para FN, 0 para TP
        y_strat = np.zeros(len(pos_indices))
        y_strat[:len(fn_indices)] = 1
        
        for fold, (train_pos_idx, val_pos_idx) in enumerate(skf.split(pos_indices, y_strat)):
            print(f"\nFold {fold+1}/{k_folds}")
            
            # Índices de treino e validação para positivos
            train_pos = pos_indices[train_pos_idx]
            val_pos = pos_indices[val_pos_idx]
            
            # Incluir todos os negativos no treino, nenhum na validação
            # (queremos avaliar apenas a capacidade de recuperar FNs)
            train_idx = np.concatenate([train_pos, neg_indices])
            val_idx = val_pos
            
            # Datasets para este fold
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train[val_idx]
            
            # Treinar modelo
            model.fit(X_train_fold, y_train_fold)
            
            # Avaliar no conjunto de validação de positivos
            y_prob_fold = model.predict_proba(X_val_fold)[:, 1]
            
            # Otimizar threshold para este fold
            thresholds = np.linspace(0.1, 0.9, 9)
            best_threshold = 0.5
            best_f1 = 0
            
            for t in thresholds:
                y_pred_t = (y_prob_fold >= t).astype(int)
                try:
                    from sklearn.metrics import f1_score
                    f1 = f1_score(y_val_fold, y_pred_t)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = t
                except:
                    pass
            
            # Usar melhor threshold
            y_pred_fold = (y_prob_fold >= best_threshold).astype(int)
            
            # Calcular recall apenas para os FNs originais
            val_fn_mask = np.zeros(len(val_idx), dtype=bool)
            val_fn_mask[:len(val_pos_idx)] = True  # Os primeiros são FNs
            
            n_fn = np.sum(val_fn_mask)
            n_recovered = np.sum(y_pred_fold[val_fn_mask] == 1)
            recovery_rate = n_recovered / n_fn if n_fn > 0 else 0
            
            print(f"  Threshold: {best_threshold:.2f}")
            print(f"  Recuperação de FNs: {n_recovered}/{n_fn} ({recovery_rate:.1%})")
            
            # Pesos das features
            feature_weights = pd.Series(model.feature_importances_, index=X_train.columns)
            
            # Guardar métricas
            fold_results.append({
                'threshold': best_threshold,
                'recovery_rate': recovery_rate,
                'fn_true': n_fn,
                'fn_recovered': n_recovered
            })
            
            # Guardar pesos
            fold_weights.append(feature_weights)
        
        # Combinar resultados de todos os folds
        print("\nResumindo resultados da validação cruzada:")
        
        avg_recovery = np.mean([r['recovery_rate'] for r in fold_results])
        avg_threshold = np.mean([r['threshold'] for r in fold_results])
        
        print(f"Recuperação média de FNs: {avg_recovery:.1%}")
        print(f"Threshold médio: {avg_threshold:.2f}")
        
        # Calcular média dos pesos das features
        avg_weights = pd.concat(fold_weights, axis=1).mean(axis=1)
        avg_weights = avg_weights.sort_values(ascending=False)
        
        # Salvar pesos médios
        weights_path = os.path.join(output_dir, 'fn_specialist_feature_weights.csv')
        avg_weights.to_csv(weights_path)
        
        print(f"\nPesos de features do modelo especializado salvos em: {weights_path}")
        
        # Mostrar top features
        print("\nTop 15 features mais importantes para recuperar falsos negativos:")
        for i, (feature, weight) in enumerate(avg_weights.head(15).items(), 1):
            print(f"{i}. {feature}: {weight:.5f}")
    
    # Treinar modelo final com todos os dados
    print("\nTreinando modelo especialista final com todos os dados...")
    model.fit(X_train, y_train)
    
    # Avaliar recuperação potencial em todos os falsos negativos
    fn_X = X.loc[fn_indices]
    fn_recovery_probs = model.predict_proba(fn_X)[:, 1]
    
    # Analisar distribuição das probabilidades
    plt.figure(figsize=(10, 6))
    plt.hist(fn_recovery_probs, bins=20, alpha=0.7)
    plt.axvline(avg_threshold, color='red', linestyle='--', label=f'Threshold médio: {avg_threshold:.2f}')
    plt.title('Distribuição de Probabilidades para Falsos Negativos - Modelo Especialista')
    plt.xlabel('Probabilidade')
    plt.ylabel('Contagem')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    hist_path = os.path.join(output_dir, 'fn_specialist_probability_hist.png')
    plt.savefig(hist_path, dpi=300)
    plt.close()
    
    # Salvar modelo especialista
    model_path = os.path.join(output_dir, 'fn_specialist_model.joblib')
    joblib.dump(model, model_path)
    
    print(f"Modelo especialista salvo em: {model_path}")
    print(f"Histograma de probabilidades salvo em: {hist_path}")
    
    # Estimar potencial ganho de recall
    recovered = np.sum(fn_recovery_probs >= avg_threshold)
    potential_recovery_rate = recovered / len(fn_indices)
    
    print(f"\nPotencial de recuperação de FNs: {recovered}/{len(fn_indices)} ({potential_recovery_rate:.1%})")
    
    return {
        'model': model,
        'avg_threshold': avg_threshold,
        'potential_recovery_rate': potential_recovery_rate,
        'feature_weights': avg_weights,
        'model_path': model_path
    }

def analyze_class_balance_strategies(df, X, y, analysis_df, output_dir):
    """Avalia diferentes estratégias de balanceamento de classes."""
    print("\nAnalisando estratégias de balanceamento de classes...")
    
    # Verificar desbalanceamento atual
    class_counts = np.bincount(y)
    class_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
    
    print(f"Distribuição atual: {class_counts[0]} negativos, {class_counts[1]} positivos")
    print(f"Razão negativo:positivo = {class_ratio:.1f}:1")
    
    # Preparar DataFrame para resultados
    balance_results = pd.DataFrame(columns=[
        'strategy', 'negative_weight', 'positive_weight', 'threshold',
        'precision', 'recall', 'f1', 'auc', 'pr_auc'
    ])
    
    # Definir estratégias a testar
    strategies = [
        # Nome, peso negativo, peso positivo
        ('atual', 1, 1),
        ('proporcional', 1, class_ratio),
        ('quadrático', 1, class_ratio**2),
        ('raiz_quadrada', 1, np.sqrt(class_ratio)),
        ('extremo', 1, 2*class_ratio)
    ]
    
    # Testar cada estratégia com validação cruzada
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for strategy_name, neg_weight, pos_weight in strategies:
        print(f"\nAvaliando estratégia: {strategy_name}")
        print(f"  Pesos: negativo={neg_weight}, positivo={pos_weight}")
        
        # Arrays para armazenar resultados
        precisions = []
        recalls = []
        f1s = []
        aucs = []
        pr_aucs = []
        thresholds = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Criar class_weight dict
            class_weight = {0: neg_weight, 1: pos_weight}
            
            # Treinar modelo
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predições
            y_prob = model.predict_proba(X_val)[:, 1]
            
            # Calcular AUC e PR-AUC
            auc = roc_auc_score(y_val, y_prob)
            pr_auc = average_precision_score(y_val, y_prob)
            
            # Encontrar melhor threshold para F1
            precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_val, y_prob)
            f1_curve = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
            
            # Ajustar tamanhos para threshold_curve (um elemento a menos)
            threshold_curve = np.append(threshold_curve, 1.0)
            
            best_idx = np.argmax(f1_curve)
            best_threshold = threshold_curve[best_idx]
            best_precision = precision_curve[best_idx]
            best_recall = recall_curve[best_idx]
            best_f1 = f1_curve[best_idx]
            
            # Guardar resultados
            precisions.append(best_precision)
            recalls.append(best_recall)
            f1s.append(best_f1)
            aucs.append(auc)
            pr_aucs.append(pr_auc)
            thresholds.append(best_threshold)
        
        # Médias
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1s)
        avg_auc = np.mean(aucs)
        avg_pr_auc = np.mean(pr_aucs)
        avg_threshold = np.mean(thresholds)
        
        print(f"  Resultados (média de {n_splits} folds):")
        print(f"    Precision: {avg_precision:.4f}")
        print(f"    Recall: {avg_recall:.4f}")
        print(f"    F1-Score: {avg_f1:.4f}")
        print(f"    AUC: {avg_auc:.4f}")
        print(f"    PR-AUC: {avg_pr_auc:.4f}")
        print(f"    Threshold: {avg_threshold:.4f}")
        
        # Adicionar ao DataFrame de resultados
        balance_results.loc[len(balance_results)] = [
            strategy_name, neg_weight, pos_weight, avg_threshold,
            avg_precision, avg_recall, avg_f1, avg_auc, avg_pr_auc
        ]
    
    # Ordenar por recall
    balance_results = balance_results.sort_values('recall', ascending=False)
    
    # Salvar resultados
    results_path = os.path.join(output_dir, 'class_balance_strategy_results.csv')
    balance_results.to_csv(results_path, index=False)
    
    print(f"\nResultados das estratégias de balanceamento salvos em: {results_path}")
    print("\nMelhores estratégias por recall:")
    print(balance_results[['strategy', 'recall', 'precision', 'f1']].head())
    
    # Visualizar relação Precisão-Recall para diferentes estratégias
    plt.figure(figsize=(10, 6))
    
    for i, row in balance_results.iterrows():
        plt.plot(row['recall'], row['precision'], 'o', markersize=10, 
               label=f"{row['strategy']} (F1={row['f1']:.3f})")
    
    plt.title('Relação Precision-Recall para Diferentes Estratégias de Balanceamento')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(output_dir, 'balance_strategy_pr_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Gráfico de comparação salvo em: {plot_path}")
    
    return {
        'results': balance_results,
        'best_recall_strategy': balance_results.iloc[0]['strategy'],
        'best_f1_strategy': balance_results.sort_values('f1', ascending=False).iloc[0]['strategy']
    }

def analyze_ensemble_opportunities(analysis_df, target_col, output_dir):
    """Analisa oportunidades para estratégias de ensemble."""
    print("\nAnalisando oportunidades para estratégias de ensemble...")
    
    # Extrair falsos negativos
    fn_data = analysis_df[analysis_df['error_type'] == 'false_negative'].copy()
    
    if len(fn_data) < 10:
        print("Número insuficiente de falsos negativos para análise de ensemble")
        return None
    
    # Analisar distribuição de probabilidades
    plt.figure(figsize=(10, 6))
    sns.histplot(fn_data['probability'], bins=20, kde=True)
    plt.axvline(0.5, color='red', linestyle='--', label='Threshold padrão (0.5)')
    plt.title('Distribuição de Probabilidades dos Falsos Negativos')
    plt.xlabel('Probabilidade')
    plt.ylabel('Contagem')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    prob_dist_path = os.path.join(output_dir, 'fn_probability_distribution.png')
    plt.savefig(prob_dist_path, dpi=300)
    plt.close()
    
    # Estatísticas da distribuição
    prob_stats = fn_data['probability'].describe()
    print("\nEstatísticas da distribuição de probabilidades dos falsos negativos:")
    print(prob_stats)
    
    # Verificar se há agrupamentos naturais nas probabilidades
    from sklearn.cluster import KMeans
    
    # Preparar dados para clustering
    X_prob = fn_data['probability'].values.reshape(-1, 1)
    
    # Determinar número ótimo de clusters
    inertias = []
    k_range = range(1, min(5, len(fn_data) // 10 + 1))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_prob)
        inertias.append(kmeans.inertia_)
    
    # Plotar "elbow method"
    if len(k_range) > 2:
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertias, 'o-')
        plt.title('Método do Cotovelo para Determinar Número Ótimo de Clusters')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inércia')
        plt.grid(True, alpha=0.3)
        
        elbow_path = os.path.join(output_dir, 'probability_clusters_elbow.png')
        plt.savefig(elbow_path, dpi=300)
        plt.close()
    
    # Determinar número ótimo de clusters
    if len(k_range) > 2:
        # Calcular diferenças de inércia
        inertia_diffs = np.diff(inertias)
        
        # Normalizar
        norm_diffs = inertia_diffs / inertias[:-1]
        
        # Encontrar o "cotovelo"
        elbow_idx = np.argmax(norm_diffs) + 1
        optimal_k = k_range[elbow_idx]
    else:
        optimal_k = min(2, len(k_range))
    
    # Aplicar clustering com k ótimo
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    fn_data['prob_cluster'] = kmeans.fit_predict(X_prob)
    
    # Encontrar centros dos clusters
    cluster_centers = kmeans.cluster_centers_.flatten()
    
    # Ordenar clusters por probabilidade
    cluster_order = np.argsort(cluster_centers)
    cluster_mapping = {old: new for new, old in enumerate(cluster_order)}
    fn_data['prob_cluster'] = fn_data['prob_cluster'].map(cluster_mapping)
    
    # Atualizar centros
    cluster_centers = sorted(cluster_centers)
    
    # Estatísticas por cluster
    print(f"\nDistribuição em {optimal_k} clusters de probabilidade:")
    
    cluster_stats = []
    
    for cluster_id in range(optimal_k):
        cluster_probs = fn_data[fn_data['prob_cluster'] == cluster_id]['probability']
        
        print(f"Cluster {cluster_id}:")
        print(f"  Centro: {cluster_centers[cluster_id]:.4f}")
        print(f"  Tamanho: {len(cluster_probs)} ({len(cluster_probs)/len(fn_data):.1%} dos FNs)")
        print(f"  Faixa: {cluster_probs.min():.4f} - {cluster_probs.max():.4f}")
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'center': cluster_centers[cluster_id],
            'size': len(cluster_probs),
            'proportion': len(cluster_probs)/len(fn_data),
            'min_prob': cluster_probs.min(),
            'max_prob': cluster_probs.max()
        })
    
    # Visualizar clusters
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))
    
    for cluster_id in range(optimal_k):
        cluster_data = fn_data[fn_data['prob_cluster'] == cluster_id]
        
        sns.kdeplot(
            cluster_data['probability'], 
            fill=True, 
            color=colors[cluster_id],
            label=f'Cluster {cluster_id} (centro: {cluster_centers[cluster_id]:.3f})'
        )
    
    plt.title('Clusters de Probabilidade dos Falsos Negativos')
    plt.xlabel('Probabilidade')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    cluster_path = os.path.join(output_dir, 'fn_probability_clusters.png')
    plt.savefig(cluster_path, dpi=300)
    plt.close()
    
    # Salvar estatísticas de clusters
    cluster_stats_df = pd.DataFrame(cluster_stats)
    cluster_stats_path = os.path.join(output_dir, 'probability_cluster_stats.csv')
    cluster_stats_df.to_csv(cluster_stats_path, index=False)
    
    print(f"\nEstatísticas de clusters de probabilidade salvas em: {cluster_stats_path}")
    
    # Analisar características por cluster de probabilidade
    # Remover colunas não-numéricas para a análise
    non_feature_cols = ['true_class', 'predicted_class', 'probability', 'error_type', 
                      'fn_category', 'cluster', 'prob_cluster']
    
    if target_col not in non_feature_cols:
        non_feature_cols.append(target_col)
    
    # Filtrar apenas colunas numéricas
    numeric_cols = [col for col in fn_data.columns 
                   if col not in non_feature_cols and 
                   pd.api.types.is_numeric_dtype(fn_data[col])]
    
    # Limitar número de features para análise
    if len(numeric_cols) > 30:
        feature_variance = fn_data[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = feature_variance.index[:30].tolist()
    
    # Comparar características entre clusters
    cluster_features = pd.DataFrame(index=numeric_cols)
    
    for cluster_id in range(optimal_k):
        cluster_data = fn_data[fn_data['prob_cluster'] == cluster_id]
        
        # Média para este cluster
        cluster_features[f'cluster_{cluster_id}_mean'] = cluster_data[numeric_cols].mean()
        
        # Desvio padrão
        cluster_features[f'cluster_{cluster_id}_std'] = cluster_data[numeric_cols].std()
    
    # Média global
    cluster_features['global_mean'] = fn_data[numeric_cols].mean()
    
    # Calcular diferenças normalizadas
    for cluster_id in range(optimal_k):
        diff = (cluster_features[f'cluster_{cluster_id}_mean'] - cluster_features['global_mean'])
        norm_diff = diff / cluster_features['global_mean'].replace(0, 1)
        cluster_features[f'cluster_{cluster_id}_diff'] = norm_diff
    
    # Salvar análise de features por cluster
    features_path = os.path.join(output_dir, 'probability_cluster_features.csv')
    cluster_features.to_csv(features_path)
    
    print(f"Análise de features por cluster de probabilidade salva em: {features_path}")
    
    # Identificar top features distintivas por cluster
    print("\nTop features distintivas por cluster de probabilidade:")
    
    for cluster_id in range(optimal_k):
        diff_col = f'cluster_{cluster_id}_diff'
        top_features = cluster_features[diff_col].abs().sort_values(ascending=False).head(5)
        
        print(f"\nCluster {cluster_id} (centro: {cluster_centers[cluster_id]:.3f}):")
        for feature, diff in top_features.items():
            feature_mean = cluster_features.loc[feature, f'cluster_{cluster_id}_mean']
            global_mean = cluster_features.loc[feature, 'global_mean']
            
            print(f"  {feature}: {feature_mean:.3f} (diff: {diff:+.1%} da média global)")
    
    # Avaliar potencial para diferentes estratégias de ensemble
    ensemble_recommendations = []
    
    # Verificar se há um cluster próximo ao threshold
    near_threshold_clusters = [
        i for i, center in enumerate(cluster_centers) 
        if center > 0.1 and center < 0.4  # Próximo, mas abaixo do threshold
    ]
    
    if near_threshold_clusters:
        ensemble_recommendations.append({
            'strategy': 'Combinar com modelo especialista para clusters próximos ao threshold',
            'target_clusters': near_threshold_clusters,
            'potential_recovery': sum(cluster_stats[i]['proportion'] for i in near_threshold_clusters),
            'rationale': 'Clusters próximos ao threshold podem ser recuperados com um modelo especializado'
        })
    
    # Verificar se há clusters muito distantes
    far_clusters = [
        i for i, center in enumerate(cluster_centers) 
        if center < 0.1  # Muito abaixo do threshold
    ]
    
    if far_clusters:
        ensemble_recommendations.append({
            'strategy': 'Modelo específico para características distintivas de clusters distantes',
            'target_clusters': far_clusters,
            'potential_recovery': sum(cluster_stats[i]['proportion'] for i in far_clusters),
            'rationale': 'Clusters distantes do threshold têm características distintas que podem exigir abordagens especializadas'
        })
    
    # Verificar diversidade entre clusters
    if optimal_k >= 2:
        ensemble_recommendations.append({
            'strategy': 'Ensemble de múltiplos modelos específicos por cluster',
            'target_clusters': list(range(optimal_k)),
            'potential_recovery': 0.6,  # Estimativa conservadora
            'rationale': 'A diversidade entre clusters sugere que modelos especializados podem capturar diferentes padrões'
        })
    
    # Verificar potencial para calibração de probabilidade
    if prob_stats['mean'] < 0.3 and prob_stats['std'] < 0.15:
        ensemble_recommendations.append({
            'strategy': 'Calibração de probabilidades',
            'target_clusters': list(range(optimal_k)),
            'potential_recovery': 0.5,  # Estimativa conservadora
            'rationale': 'Probabilidades concentradas sugerem que calibração pode ajustar valores para melhor refletir a verdadeira probabilidade'
        })
    
    # Imprimir recomendações de ensemble
    print("\nRecomendações para estratégias de ensemble:")
    for i, rec in enumerate(ensemble_recommendations, 1):
        print(f"{i}. {rec['strategy']}")
        print(f"   Alvos: Clusters {rec['target_clusters']}")
        print(f"   Potencial de recuperação: {rec['potential_recovery']:.1%}")
        print(f"   Justificativa: {rec['rationale']}")
    
    # Salvar recomendações
    recs_df = pd.DataFrame(ensemble_recommendations)
    recs_path = os.path.join(output_dir, 'ensemble_strategy_recommendations.csv')
    recs_df.to_csv(recs_path, index=False)
    
    print(f"\nRecomendações de estratégias de ensemble salvas em: {recs_path}")
    
    return {
        'optimal_clusters': optimal_k,
        'cluster_stats': cluster_stats_df,
        'ensemble_recommendations': ensemble_recommendations,
        'probability_distribution': prob_stats
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

def analyze_resampling_methods(X, y, output_dir):
    """Analisa diferentes métodos de reamostragem para lidar com desbalanceamento de classes."""
    print("\nAnalisando métodos de reamostragem para balanceamento de classes...")
    
    # Importar técnicas de reamostragem
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
        from imblearn.under_sampling import RandomUnderSampler, NearMiss
        from imblearn.combine import SMOTETomek, SMOTEENN
        
        resamplers_available = True
    except ImportError:
        print("Biblioteca imbalanced-learn não encontrada. Instalando via pip...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "imbalanced-learn"])
            from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
            from imblearn.under_sampling import RandomUnderSampler, NearMiss
            from imblearn.combine import SMOTETomek, SMOTEENN
            
            resamplers_available = True
            print("Biblioteca instalada com sucesso!")
        except:
            print("Falha ao instalar imbalanced-learn. Análise de reamostragem limitada.")
            resamplers_available = False
    
    # Usar validação cruzada para avaliar métodos
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
    
    # Definir resampling methods a serem testados
    resampling_methods = []
    
    # Sempre adicionar o baseline (sem reamostragem)
    resampling_methods.append(("Sem reamostragem", None))
    
    if resamplers_available:
        resampling_methods.extend([
            ("Random Oversampling", RandomOverSampler(random_state=42)),
            ("SMOTE", SMOTE(random_state=42)),
            ("BorderlineSMOTE", BorderlineSMOTE(random_state=42)),
            ("ADASYN", ADASYN(random_state=42)),
            ("Random Undersampling", RandomUnderSampler(random_state=42)),
            ("NearMiss", NearMiss(version=3)),
            ("SMOTE+Tomek", SMOTETomek(random_state=42)),
            ("SMOTE+ENN", SMOTEENN(random_state=42))
        ])
    
    # Configurar validação cruzada
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # DataFrame para resultados
    results = pd.DataFrame(columns=[
        'method', 'precision', 'recall', 'f1', 'pr_auc', 
        'train_pos_ratio', 'train_time', 'inference_time'
    ])
    
    # Executar validação cruzada para cada método
    for name, resampler in resampling_methods:
        print(f"\nAvaliando método: {name}")
        
        # Arrays para métricas
        precisions = []
        recalls = []
        f1s = []
        pr_aucs = []
        train_pos_ratios = []
        train_times = []
        inference_times = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold+1}/{n_splits}")
            
            # Split
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Aplicar resampling se especificado
            if resampler is not None:
                start_time = datetime.now()
                try:
                    X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
                    resampling_time = (datetime.now() - start_time).total_seconds()
                    
                    # Calcular proporção de positivos
                    pos_ratio = np.mean(y_train_res == 1)
                    
                    print(f"    Reamostragem: {X_train.shape} -> {X_train_res.shape} ({pos_ratio:.1%} positivos)")
                    print(f"    Tempo de reamostragem: {resampling_time:.2f}s")
                except Exception as e:
                    print(f"    Erro na reamostragem: {str(e)}")
                    continue
            else:
                X_train_res, y_train_res = X_train, y_train
                resampling_time = 0
                pos_ratio = np.mean(y_train_res == 1)
            
            # Treinar modelo
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            start_time = datetime.now()
            model.fit(X_train_res, y_train_res)
            train_time = (datetime.now() - start_time).total_seconds() + resampling_time
            
            # Predição
            start_time = datetime.now()
            y_prob = model.predict_proba(X_val)[:, 1]
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Encontrar melhor threshold para F1
            precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
            f1s_curve = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            best_idx = np.argmax(f1s_curve)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            
            y_pred = (y_prob >= best_threshold).astype(int)
            
            # Calcular métricas
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            pr_auc = average_precision_score(y_val, y_prob)
            
            print(f"    Threshold: {best_threshold:.4f}")
            print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Armazenar resultados
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            pr_aucs.append(pr_auc)
            train_pos_ratios.append(pos_ratio)
            train_times.append(train_time)
            inference_times.append(inference_time)
        
        # Calcular médias
        if len(precisions) > 0:
            results.loc[len(results)] = [
                name,
                np.mean(precisions),
                np.mean(recalls),
                np.mean(f1s),
                np.mean(pr_aucs),
                np.mean(train_pos_ratios),
                np.mean(train_times),
                np.mean(inference_times)
            ]
    
    # Ordenar por recall
    results = results.sort_values('recall', ascending=False)
    
    # Salvar resultados
    results_path = os.path.join(output_dir, 'resampling_methods_comparison.csv')
    results.to_csv(results_path, index=False)
    
    # Visualizar comparação
    plt.figure(figsize=(12, 8))
    
    # Plot de barras para recall e precision
    x = np.arange(len(results))
    width = 0.35
    
    plt.subplot(2, 1, 1)
    plt.bar(x - width/2, results['recall'], width, label='Recall')
    plt.bar(x + width/2, results['precision'], width, label='Precision')
    
    plt.ylabel('Valor')
    plt.title('Comparação de Métodos de Reamostragem')
    plt.xticks(x, results['method'], rotation=45, ha='right')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Plot de F1 e PR-AUC
    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, results['f1'], width, label='F1-Score')
    plt.bar(x + width/2, results['pr_auc'], width, label='PR-AUC')
    
    plt.ylabel('Valor')
    plt.xlabel('Método de Reamostragem')
    plt.xticks(x, results['method'], rotation=45, ha='right')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar figura
    plot_path = os.path.join(output_dir, 'resampling_methods_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparação de métodos de reamostragem salva em: {results_path}")
    print(f"Gráfico comparativo salvo em: {plot_path}")
    
    # Mostrar top métodos
    print("\nTop métodos de reamostragem por recall:")
    top_recall = results.head(3)
    for i, row in top_recall.iterrows():
        print(f"  {row['method']}: Recall={row['recall']:.4f}, Precision={row['precision']:.4f}, F1={row['f1']:.4f}")
    
    return {
        'results': results,
        'best_recall_method': results.iloc[0]['method'] if not results.empty else None,
        'best_f1_method': results.sort_values('f1', ascending=False).iloc[0]['method'] if not results.empty else None
    }

def generate_insights_and_recommendations(analysis_results, model_uri, output_dir):
    """Gera insights e recomendações baseados em todas as análises realizadas."""
    print("\n=== Gerando Insights e Recomendações ===")
    
    # Criar DataFrame para armazenar recomendações
    recommendations = pd.DataFrame(columns=[
        'category', 'recommendation', 'expected_impact', 'implementation_difficulty', 'priority'
    ])
    
    # Adicionar recomendações com base nos resultados
    
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
    
    # 4. Recomendações de clustering
    if 'clustering' in analysis_results:
        cluster_results = analysis_results['clustering']
        
        if cluster_results and 'n_clusters' in cluster_results and cluster_results['n_clusters'] > 1:
            recommendations.loc[len(recommendations)] = [
                'Modelo Especializado',
                f'Treinar modelos específicos para cada um dos {cluster_results["n_clusters"]} clusters de falsos negativos',
                'Potencial recuperação de 15-30% dos falsos negativos',
                'Alta',
                'Média'
            ]
    
    # 5. Recomendações de ensemble
    if 'ensemble' in analysis_results and 'ensemble_recommendations' in analysis_results.get('ensemble', {}):
        ensemble_recs = analysis_results['ensemble']['ensemble_recommendations']
        
        for i, rec in enumerate(ensemble_recs):
            if rec['potential_recovery'] > 0.3:
                recommendations.loc[len(recommendations)] = [
                    'Estratégia de Ensemble',
                    rec['strategy'],
                    f'Potencial recuperação de {rec["potential_recovery"]:.1%} dos falsos negativos',
                    'Alta',
                    'Alta' if rec["potential_recovery"] > 0.5 else 'Média'
                ]
    
    # 6. Recomendações de resampling
    if 'resampling' in analysis_results:
        resampling_results = analysis_results['resampling']
        
        if 'best_recall_method' in resampling_results and resampling_results['best_recall_method']:
            best_method = resampling_results['best_recall_method']
            
            if best_method != 'Sem reamostragem':
                # Encontrar resultados para esse método
                method_results = resampling_results['results']
                method_row = method_results[method_results['method'] == best_method].iloc[0]
                
                recommendations.loc[len(recommendations)] = [
                    'Balanceamento de Classes',
                    f'Utilizar {best_method} para balancear os dados de treinamento',
                    f'Recall esperado: {method_row["recall"]:.2%}, Precisão: {method_row["precision"]:.2%}',
                    'Fácil',
                    'Alta'
                ]
    
    # 7. Recomendação para modelo especialista em FNs
    if 'specialist_model' in analysis_results:
        specialist_results = analysis_results['specialist_model']
        
        if 'potential_recovery_rate' in specialist_results:
            recovery_rate = specialist_results['potential_recovery_rate']
            
            if recovery_rate > 0.3:
                recommendations.loc[len(recommendations)] = [
                    'Modelo Especializado',
                    'Implementar modelo especialista para recuperar falsos negativos em cascata',
                    f'Potencial recuperação de {recovery_rate:.1%} dos falsos negativos',
                    'Média',
                    'Alta' if recovery_rate > 0.5 else 'Média'
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
    
    if 'false_negatives' in analysis_results:
        fn_count = analysis_results.get('false_negatives', {}).get('count', 0)
        insights.append(f"- O modelo atual está deixando de identificar {fn_count} leads positivos.")
    
    if 'feature_analysis' in analysis_results:
        insights.append("- Os falsos negativos apresentam características distintas dos verdadeiros positivos.")
        if 'top_differentiating_features' in analysis_results['feature_analysis']:
            features = analysis_results['feature_analysis']['top_differentiating_features'][:3]
            insights.append(f"- As principais diferenças estão em: {', '.join(features)}.")
    
    if 'clustering' in analysis_results and analysis_results['clustering']:
        insights.append("\n**Agrupamentos de Falsos Negativos:**")
        insights.append(f"- Foram identificados {analysis_results['clustering'].get('n_clusters', 0)} perfis distintos de falsos negativos.")
        insights.append("- Cada grupo pode exigir uma estratégia específica para ser corretamente classificado.")
    
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
    
    # Insights sobre resampling
    if 'resampling' in analysis_results:
        insights.append("\n**Balanceamento de Classes:**")
        best_method = analysis_results['resampling'].get('best_recall_method', '')
        if best_method and best_method != 'Sem reamostragem':
            insights.append(f"- O método {best_method} mostrou os melhores resultados para aumentar o recall.")
            insights.append("- Técnicas de balanceamento podem ajudar a reduzir o viés do modelo para a classe majoritária.")
        else:
            insights.append("- As técnicas de balanceamento testadas não trouxeram ganhos significativos.")
    
    # Resumo final
    insights.append("\n**Resumo Geral:**")
    insights.append("- O modelo atual prioriza precisão, mas tem potencial para melhorar significativamente o recall.")
    insights.append("- A combinação de ajustes de threshold, tratamento de falsos negativos e possivelmente ensemble pode levar a ganhos substanciais.")
    
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
    parser = argparse.ArgumentParser(description='Executar análise avançada de falsos negativos com foco em melhorar o recall')
    parser.add_argument('--mlflow_dir', default=None, 
                      help='Diretório do MLflow tracking')
    parser.add_argument('--data_path', default=None,
                      help='Caminho para o dataset de validação/teste')
    parser.add_argument('--target_col', default='target',
                      help='Nome da coluna target')
    parser.add_argument('--model_uri', default=None,
                      help='URI específico do modelo a ser analisado (opcional)')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Threshold personalizado para classificação (opcional)')
    parser.add_argument('--output_dir', default=None,
                      help='Diretório para salvar os resultados (opcional)')
    
    args = parser.parse_args()
    
    # Definir valor padrão para mlflow_dir se não fornecido
    if args.mlflow_dir is None:
        default_mlflow_dir = os.path.join(os.path.expanduser("~"), "desktop", "smart_ads", "models", "mlflow")
        if os.path.exists(default_mlflow_dir):
            args.mlflow_dir = default_mlflow_dir
        else:
            args.mlflow_dir = os.path.join(project_root, "models", "mlflow")
    
    print(f"Usando diretório MLflow: {args.mlflow_dir}")
    
    # Configurar MLflow
    setup_mlflow(args.mlflow_dir)
    
    # Definir caminho padrão para os dados se não fornecido
    if args.data_path is None:
        default_data_path = os.path.join(os.path.expanduser("~"), "desktop", "smart_ads", "data", "03_feature_selection_textv2", "validation.csv")
        if os.path.exists(default_data_path):
            args.data_path = default_data_path
        else:
            args.data_path = os.path.join(project_root, "data", "03_feature_selection_textv2", "validation.csv")
    
    print(f"Usando dataset: {args.data_path}")
    
    # Definir diretório de saída
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(project_root, "reports", f"false_negative_analysis_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {args.output_dir}")
    
    # Verificar se o dataset existe
    if not os.path.exists(args.data_path):
        print(f"ERRO: Dataset não encontrado: {args.data_path}")
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
            from src.evaluation.baseline_model import get_latest_random_forest_run
            
            run_id, threshold_from_mlflow, model_uri = get_latest_random_forest_run(args.mlflow_dir)
            
            if model_uri is None:
                print("ERRO: Não foi possível encontrar um modelo RandomForest no MLflow.")
                sys.exit(1)
                
            print(f"Usando modelo mais recente: {model_uri}")
        else:
            model_uri = args.model_uri
            threshold_from_mlflow = None
            print(f"Usando modelo especificado: {model_uri}")
        
        model = load_model_from_mlflow(model_uri)
        
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
            
    except ImportError as e:
        print(f"ERRO de importação ao carregar modelo: {e}")
        print("Verifique se os módulos necessários estão disponíveis.")
        sys.exit(1)
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
    
    # 4. Análise de clustering
    try:
        print("\n=== 4. Realizando análise de clustering ===")
        cluster_results = perform_clustering_analysis(analysis_df, target_col, args.output_dir)
        analysis_results['clustering'] = cluster_results
        print("Análise de clustering concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na análise de clustering: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Treinamento de modelo especializado
    try:
        print("\n=== 5. Avaliando potencial de modelo especializado ===")
        specialist_results = train_specialized_model(X, y, analysis_df, args.output_dir)
        analysis_results['specialist_model'] = specialist_results
        print("Avaliação de modelo especializado concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na avaliação de modelo especializado: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Análise de balanceamento de classes
    try:
        print("\n=== 6. Analisando estratégias de balanceamento de classes ===")
        balance_results = analyze_class_balance_strategies(df, X, y, analysis_df, args.output_dir)
        analysis_results['class_balance'] = balance_results
        print("Análise de balanceamento de classes concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na análise de balanceamento: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. Análise de oportunidades de ensemble
    try:
        print("\n=== 7. Analisando oportunidades de ensemble ===")
        ensemble_results = analyze_ensemble_opportunities(analysis_df, target_col, args.output_dir)
        analysis_results['ensemble'] = ensemble_results
        print("Análise de oportunidades de ensemble concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na análise de ensemble: {e}")
        import traceback
        traceback.print_exc()
    
    # 8. Análise de calibração
    try:
        print("\n=== 8. Analisando oportunidade de calibração ===")
        calibration_results = analyze_calibration_opportunity(y, y_prob, analysis_df, args.output_dir)
        analysis_results['calibration'] = calibration_results
        print("Análise de calibração concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na análise de calibração: {e}")
        import traceback
        traceback.print_exc()
    
    # 9. Análise de métodos de reamostragem
    try:
        print("\n=== 9. Analisando métodos de reamostragem ===")
        resampling_results = analyze_resampling_methods(X, y, args.output_dir)
        analysis_results['resampling'] = resampling_results
        print("Análise de métodos de reamostragem concluída com sucesso.")
    except Exception as e:
        print(f"ERRO na análise de reamostragem: {e}")
        import traceback
        traceback.print_exc()
    
    # 10. Geração de insights e recomendações
    try:
        print("\n=== 10. Gerando insights e recomendações ===")
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