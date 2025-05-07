import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import json
import mlflow
import joblib
from pathlib import Path
import contextlib
import warnings
import tempfile

# Adicionar o caminho do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar funções existentes
from src.evaluation.mlflow_utils import find_optimal_threshold

# Constantes de configuração centralizadas
CONFIG = {
    'base_dir': os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'data_dir': os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data/02_3_processed_text_code6"),
    'mlflow_dir': os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "models/mlflow"),
    'experiment_tuning': "smart_ads_cluster_tuning",
    'experiment_best': "smart_ads_cluster_best",
    'runs_per_experiment': 3,  # Limitar número de runs por experimento
}

def configure_mlflow():
    """
    Configura o MLflow para usar o diretório existente.
    """
    # Caminho para o diretório MLflow existente
    mlflow_dir = CONFIG['mlflow_dir']
    
    # Verificar se o diretório existe, se não existir, criar
    os.makedirs(mlflow_dir, exist_ok=True)
    
    # Configurar MLflow para usar este diretório
    mlflow_tracking_uri = f"file://{mlflow_dir}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLflow configurado para usar: {mlflow_tracking_uri}")
    
    return mlflow_tracking_uri

def load_processed_datasets(data_dir):
    """
    Carrega os datasets já processados pelos scripts 02 e 06.
    
    Args:
        data_dir: Diretório com os dados processados
        
    Returns:
        Dicionário com datasets carregados
    """
    print("Carregando datasets pré-processados...")
    
    # Caminhos dos datasets
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    
    # Verificar existência dos arquivos
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Datasets não encontrados em {data_dir}")
    
    # Carregar datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"Dataset de treino: {train_df.shape}")
    print(f"Dataset de validação: {val_df.shape}")
    
    # Separar features e target
    target_col = 'target'
    
    if target_col not in train_df.columns or target_col not in val_df.columns:
        raise ValueError(f"Coluna target '{target_col}' não encontrada nos datasets")
    
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    
    X_train = train_df.drop(columns=[target_col])
    X_val = val_df.drop(columns=[target_col])
    
    # Garantir que ambos têm as mesmas colunas
    common_cols = list(set(X_train.columns).intersection(set(X_val.columns)))
    
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    
    print(f"Features comuns: {len(common_cols)}")
    print(f"Taxa de conversão (treino): {y_train.mean():.4f}")
    print(f"Taxa de conversão (validação): {y_val.mean():.4f}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'feature_names': common_cols
    }

def identify_numeric_features(X_train, X_val):
    """
    Identifica features estritamente numéricas para uso no clustering.
    
    Args:
        X_train: DataFrame de treino
        X_val: DataFrame de validação
        
    Returns:
        Tupla (X_train_numeric, X_val_numeric, numeric_cols)
    """
    print("Identificando features numéricas para clustering...")
    
    # Inicializar lista de colunas numéricas
    numeric_cols = []
    
    # Verificar cada coluna
    for col in X_train.columns:
        try:
            # Converter para float para verificar se todos os valores são numéricos
            X_train[col].astype(float)
            X_val[col].astype(float)
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    # Extrair apenas as colunas numéricas
    X_train_numeric = X_train[numeric_cols].copy().astype(float)
    X_val_numeric = X_val[numeric_cols].copy().astype(float)
    
    # Substituir NaN por 0
    X_train_numeric.fillna(0, inplace=True)
    X_val_numeric.fillna(0, inplace=True)
    
    print(f"Total de features numéricas: {len(numeric_cols)}")
    
    return X_train_numeric, X_val_numeric, numeric_cols

def apply_dimension_reduction(X_train_numeric, X_val_numeric, variance_threshold=0.8, max_components=100, 
                             random_state=42):
    """
    Aplica redução de dimensionalidade com PCA, determinando automaticamente o número ideal de componentes.
    
    Args:
        X_train_numeric: Features numéricas de treino
        X_val_numeric: Features numéricas de validação
        variance_threshold: Variância explicada cumulativa mínima (default: 0.8)
        max_components: Número máximo de componentes a considerar
        random_state: Semente aleatória para reprodutibilidade
        
    Returns:
        Tupla (X_train_pca, X_val_pca, pca_model, n_components, variance_explained)
    """
    print(f"Aplicando PCA para redução de dimensionalidade...")
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_val_scaled = scaler.transform(X_val_numeric)
    
    # Determinar número máximo de componentes possíveis
    max_possible = min(X_train_scaled.shape[0], X_train_scaled.shape[1]) - 1
    max_components = min(max_components, max_possible)
    
    # Primeiro criar um PCA para analisar a variância explicada
    pca_analysis = PCA(n_components=max_components, random_state=random_state)
    pca_analysis.fit(X_train_scaled)
    
    # Calcular variância explicada cumulativa
    cumulative_variance = np.cumsum(pca_analysis.explained_variance_ratio_)
    
    # Encontrar número ideal de componentes baseado no threshold de variância
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Garantir um mínimo de componentes
    n_components = max(n_components, 10)
    
    variance_explained = cumulative_variance[n_components-1]
    
    print(f"Número ideal de componentes: {n_components} (variância explicada: {variance_explained:.4f})")
    
    # Criar PCA com o número ideal de componentes
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    print(f"Dimensão final após PCA: {X_train_pca.shape}")
    
    return X_train_pca, X_val_pca, pca, n_components, variance_explained

def cluster_data(X_train_pca, X_val_pca, n_clusters=5, random_state=42):
    """
    Aplica K-means para clusterizar os dados no espaço PCA.
    
    Args:
        X_train_pca: Features PCA de treino
        X_val_pca: Features PCA de validação
        n_clusters: Número de clusters
        random_state: Semente aleatória para reprodutibilidade
        
    Returns:
        Tupla (cluster_labels_train, cluster_labels_val, kmeans_model)
    """
    print(f"Aplicando K-means com {n_clusters} clusters...")
    
    # Combinar dados para clustering conjunto
    X_combined = np.vstack([X_train_pca, X_val_pca])
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_combined)
    
    # Separar os clusters por conjunto
    cluster_labels_combined = kmeans.labels_
    cluster_labels_train = cluster_labels_combined[:len(X_train_pca)]
    cluster_labels_val = cluster_labels_combined[len(X_train_pca):]
    
    return cluster_labels_train, cluster_labels_val, kmeans

def analyze_clusters(cluster_labels_train, y_train, cluster_labels_val=None, y_val=None):
    """
    Analisa os clusters e sua relação com a variável target.
    
    Args:
        cluster_labels_train: Labels de cluster para o conjunto de treino
        y_train: Target do conjunto de treino
        cluster_labels_val: Labels de cluster para o conjunto de validação (opcional)
        y_val: Target do conjunto de validação (opcional)
        
    Returns:
        ID do cluster com menor taxa de conversão
    """
    # Adicionar labels de cluster e target a um DataFrame
    train_clusters_df = pd.DataFrame({
        'cluster': cluster_labels_train,
        'target': y_train.values
    })
    
    # Análise dos clusters no conjunto de treino
    print("\nAnálise dos clusters no conjunto de treino:")
    cluster_stats = train_clusters_df.groupby('cluster').agg({
        'target': ['count', 'sum', 'mean']
    })
    cluster_stats.columns = ['samples', 'conversions', 'conversion_rate']
    print(cluster_stats)
    
    # Se dados de validação forem fornecidos, fazer análise similar
    if cluster_labels_val is not None and y_val is not None:
        val_clusters_df = pd.DataFrame({
            'cluster': cluster_labels_val,
            'target': y_val.values
        })
        
        print("\nAnálise dos clusters no conjunto de validação:")
        val_cluster_stats = val_clusters_df.groupby('cluster').agg({
            'target': ['count', 'sum', 'mean']
        })
        val_cluster_stats.columns = ['samples', 'conversions', 'conversion_rate']
        print(val_cluster_stats)
    
    # Retornar o cluster com menor taxa de conversão
    min_conv_cluster_id = cluster_stats['conversion_rate'].idxmin()
    print(f"\nCluster com menor taxa de conversão: {min_conv_cluster_id} ({cluster_stats.loc[min_conv_cluster_id, 'conversion_rate']:.4f})")
    
    return min_conv_cluster_id

def analyze_low_conversion_cluster(X_train, y_train, cluster_labels_train, cluster_id):
    """
    Analisa detalhadamente um cluster com baixa taxa de conversão - versão simplificada.
    
    Args:
        X_train: Features completas de treino
        y_train: Target de treino
        cluster_labels_train: Labels de cluster
        cluster_id: ID do cluster a analisar
    """
    # Esta função é chamada apenas para o modelo final, não para os experimentos de hiperparâmetros
    cluster_mask = (cluster_labels_train == cluster_id)
    X_cluster = X_train[cluster_mask]
    y_cluster = y_train[cluster_mask]
    
    # Selecionar dados de outros clusters para comparação
    other_mask = ~cluster_mask
    X_other = X_train[other_mask]
    y_other = y_train[other_mask]
    
    # Estatísticas básicas
    cluster_size = len(X_cluster)
    conversion_rate = y_cluster.mean()
    other_conversion_rate = y_other.mean()
    
    print(f"\nAnalisando detalhadamente o Cluster {cluster_id}...")
    print(f"  Tamanho: {cluster_size} amostras ({cluster_size/len(X_train):.2%} do total)")
    print(f"  Taxa de conversão: {conversion_rate:.6f}")
    print(f"  Taxa nos outros clusters: {other_conversion_rate:.6f}")
    print(f"  Razão de conversão (cluster/outros): {conversion_rate/other_conversion_rate if other_conversion_rate > 0 else 0:.4f}")

def train_cluster_models(X_train, y_train, cluster_labels_train, n_clusters, max_depth=None, n_estimators=100, random_state=42):
    """
    Treina modelos específicos para cada cluster sem criar runs adicionais.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Estrutura para armazenar modelos
    cluster_models = []
    
    # Filtrar features numéricas
    numeric_cols = []
    for col in X_train.columns:
        try:
            X_train[col].astype(float)
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    X_train_numeric = X_train[numeric_cols].copy().astype(float)
    print(f"  Usando {len(numeric_cols)} features numéricas para treinamento")
    
    # Identificar clusters únicos
    unique_clusters = np.unique(cluster_labels_train)
    
    for cluster_id in unique_clusters:
        print(f"\n{'='*50}")
        print(f"Treinando modelo para cluster {cluster_id}...")
        
        # Selecionar dados do cluster atual
        cluster_mask = (cluster_labels_train == cluster_id)
        X_train_cluster = X_train_numeric[cluster_mask]
        y_train_cluster = y_train[cluster_mask]
        
        # Verificar se há dados suficientes
        print(f"  Amostras de treino no cluster: {len(X_train_cluster)}")
        print(f"  Taxa de conversão (treino): {y_train_cluster.mean():.4f}")
        
        # Pular clusters com poucos dados
        min_samples = 100
        if len(X_train_cluster) < min_samples:
            print(f"  Cluster {cluster_id} tem poucos dados (<{min_samples}), pulando treinamento.")
            continue
        
        # Calcular peso da classe minoritária para balanceamento
        conversion_rate = y_train_cluster.mean()
        if conversion_rate > 0 and conversion_rate < 0.5:
            # Scale negativo para positivo
            scale_pos_weight = (1 - conversion_rate) / conversion_rate
            
            # Se taxa for muito baixa, limitar o peso
            if scale_pos_weight > 100:
                scale_pos_weight = 100
                
            class_weight = {0: 1, 1: float(scale_pos_weight)}
            print(f"  Aplicando class_weight: {class_weight}")
        else:
            class_weight = None
        
        # Criar e treinar modelo
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )
        
        try:
            model.fit(X_train_cluster, y_train_cluster)
            
            # Armazenar informações
            model_info = {
                "cluster_id": int(cluster_id),
                "model": model,
                "threshold": 0.5,  # Será otimizado depois
                "features": numeric_cols,
                "n_samples": len(X_train_cluster),
                "conversion_rate": float(conversion_rate)
            }
            
            # Adicionar à lista
            cluster_models.append(model_info)
            
        except Exception as e:
            print(f"  Erro ao treinar modelo para cluster {cluster_id}: {e}")
    
    return cluster_models

def evaluate_ensemble_with_global_threshold(X_val, y_val, cluster_labels_val, cluster_models):
    """
    Avalia o ensemble com um threshold global.
    """
    # Criar dicionário para mapear cluster -> modelo
    cluster_model_map = {model["cluster_id"]: model for model in cluster_models}
    
    # Inicializar arrays para previsões
    y_pred_proba = np.zeros_like(y_val, dtype=float)
    
    # Para cada cluster que tem modelo, fazer previsões
    for cluster_id, model_info in cluster_model_map.items():
        # Selecionar amostras deste cluster
        cluster_mask = (cluster_labels_val == cluster_id)
        
        if not any(cluster_mask):
            continue
            
        # Obter features para este cluster - usar apenas features que o modelo conhece
        model = model_info["model"]
        features = model_info["features"]  # Lista de features usadas no treinamento
        
        # Extrair apenas as features que foram usadas no treinamento
        X_cluster = X_val[features][cluster_mask].astype(float)
        
        if len(X_cluster) > 0:
            # Prever probabilidades
            proba = model.predict_proba(X_cluster)[:, 1]
            
            # Armazenar probabilidades
            y_pred_proba[cluster_mask] = proba
    
    # Encontrar threshold ótimo
    threshold_results = find_optimal_threshold(y_val, y_pred_proba)
    best_threshold = threshold_results['best_threshold']
    
    # Aplicar threshold ótimo
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Calcular métricas globais
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    cm = confusion_matrix(y_val, y_pred)
    
    # Imprimir resultados
    print("\nResultados do Ensemble com Threshold Global:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"  Threshold: {best_threshold:.4f}")
    print("\nMatrix de Confusão:")
    print(cm)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": best_threshold,
        "confusion_matrix": cm.tolist()  # Convertido para que seja serializável em JSON
    }

def evaluate_ensemble_with_cluster_thresholds(X_val, y_val, cluster_labels_val, cluster_models):
    """
    Avalia o ensemble com thresholds específicos por cluster.
    """
    # Criar dicionário para mapear cluster -> modelo
    cluster_model_map = {model["cluster_id"]: model for model in cluster_models}
    
    # Inicializar arrays para previsões
    y_pred_proba = np.zeros_like(y_val, dtype=float)
    y_pred = np.zeros_like(y_val, dtype=int)
    
    # Calcular e armazenar threshold ótimo por cluster
    cluster_thresholds = {}
    
    # Primeiro pass: calcular probabilidades para cada cluster
    for cluster_id, model_info in cluster_model_map.items():
        # Selecionar amostras deste cluster
        cluster_mask = (cluster_labels_val == cluster_id)
        
        if not any(cluster_mask):
            continue
            
        # Obter dados para este cluster - usar apenas features que o modelo conhece
        model = model_info["model"]
        features = model_info["features"]  # Lista de features usadas no treinamento
        
        # Extrair apenas as features que foram usadas no treinamento
        X_cluster = X_val[features][cluster_mask].astype(float)
        y_cluster = y_val[cluster_mask]
        
        if len(X_cluster) > 0:
            # Prever probabilidades
            proba = model.predict_proba(X_cluster)[:, 1]
            
            # Armazenar probabilidades
            y_pred_proba[cluster_mask] = proba
            
            # Calcular threshold ótimo para este cluster
            if len(np.unique(y_cluster)) > 1 and y_cluster.sum() > 0:
                threshold_results = find_optimal_threshold(y_cluster, proba)
                best_threshold = threshold_results['best_threshold']
            else:
                # Usar threshold padrão para clusters sem conversões
                best_threshold = 0.5
            
            # Armazenar threshold
            cluster_thresholds[cluster_id] = best_threshold
            model_info["threshold"] = best_threshold
            
            # Aplicar threshold
            y_pred[cluster_mask] = (proba >= best_threshold).astype(int)
    
    # Calcular métricas globais
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    cm = confusion_matrix(y_val, y_pred)
    
    # Calcular métricas por cluster
    cluster_metrics = {}
    
    for cluster_id, threshold in cluster_thresholds.items():
        cluster_mask = (cluster_labels_val == cluster_id)
        
        if sum(cluster_mask) < 20:  # Pular clusters muito pequenos
            continue
            
        y_true_cluster = y_val[cluster_mask]
        y_pred_cluster = y_pred[cluster_mask]
        
        if len(np.unique(y_true_cluster)) > 1:
            precision_cluster = precision_score(y_true_cluster, y_pred_cluster)
            recall_cluster = recall_score(y_true_cluster, y_pred_cluster)
            f1_cluster = f1_score(y_true_cluster, y_pred_cluster)
        else:
            # Lidar com clusters que têm apenas uma classe
            only_class = y_true_cluster.iloc[0]
            all_same_pred = all(y_pred_cluster == only_class)
            
            if only_class == 1 and all_same_pred:
                precision_cluster, recall_cluster, f1_cluster = 1.0, 1.0, 1.0
            elif only_class == 0 and all_same_pred:
                precision_cluster, recall_cluster, f1_cluster = 1.0, 1.0, 1.0
            else:
                precision_cluster, recall_cluster, f1_cluster = 0.0, 0.0, 0.0
        
        # Armazenar métricas
        cluster_metrics[cluster_id] = {
            'threshold': threshold,
            'precision': precision_cluster,
            'recall': recall_cluster,
            'f1': f1_cluster,
            'samples': int(sum(cluster_mask)),
            'positives': int(y_true_cluster.sum()),
            'conversion_rate': float(y_true_cluster.mean())
        }
    
    # Imprimir resultados
    print("\nResultados do Ensemble com Thresholds por Cluster:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    
    print("\nThresholds e métricas por cluster:")
    for cluster_id, metrics in cluster_metrics.items():
        print(f"  Cluster {cluster_id}:")
        print(f"    Threshold: {metrics['threshold']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Amostras: {metrics['samples']}")
        print(f"    Conversões: {metrics['positives']}")
        print(f"    Taxa de conversão: {metrics['conversion_rate']:.4f}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "thresholds": cluster_thresholds,
        "cluster_metrics": cluster_metrics,
        "confusion_matrix": cm.tolist()  # Convertido para que seja serializável em JSON
    }

def find_best_model_configuration(results_data):
    """
    Encontra a melhor configuração de modelo com base nos resultados
    de validação cruzada, priorizando precisão alta com recall aceitável.
    
    Args:
        results_data: Lista de dicionários com resultados de avaliação
        
    Returns:
        Dicionário com a melhor configuração encontrada
    """
    # Verificar se há resultados
    if not results_data:
        print("AVISO: Nenhum resultado de experimento fornecido para análise.")
        return None
    
    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results_data)
    
    # Verificar se há baseline informado (precisão: 0.94, recall: 0.27)
    baseline_precision = 0.94
    baseline_recall = 0.27
    baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall)
    
    print(f"\nBaseline de referência: Precision={baseline_precision:.4f}, Recall={baseline_recall:.4f}, F1={baseline_f1:.4f}")
    
    # Primeiro, verificar se algum modelo supera o baseline em F1
    better_than_baseline = results_df[results_df['global_f1'] > baseline_f1]
    
    if len(better_than_baseline) > 0:
        # Se houver modelos melhores que o baseline, escolher o melhor entre eles
        best_results = better_than_baseline.sort_values('global_f1', ascending=False)
        best_config = best_results.iloc[0].to_dict()
        print("Encontrado modelo superior ao baseline!")
    else:
        # Se nenhum modelo superar o baseline, considerar outras métricas
        # Filtrar resultados com precisão mínima de 0.9 (próximo ao baseline)
        high_precision_df = results_df[results_df['global_precision'] >= 0.9]
        
        if len(high_precision_df) > 0:
            # Entre os modelos de alta precisão, escolher o de maior recall
            best_results = high_precision_df.sort_values('global_recall', ascending=False)
            best_config = best_results.iloc[0].to_dict()
            print("Nenhum modelo supera o baseline em F1. Escolhendo modelo com maior recall e precisão >= 0.9")
        else:
            # Se não houver resultados com alta precisão, usar F1 como critério
            best_results = results_df.sort_values('global_f1', ascending=False)
            best_config = best_results.iloc[0].to_dict()
            print("AVISO: Nenhum modelo com precisão >= 0.9 encontrado. Escolhendo modelo com melhor F1.")
    
    # Verificar se o melhor modelo ainda é inferior ao baseline
    if best_config['global_f1'] < baseline_f1:
        print(f"AVISO: O melhor modelo (F1={best_config['global_f1']:.4f}) é inferior ao baseline (F1={baseline_f1:.4f})")
        print("Considere usar o modelo baseline em vez do modelo treinado.")
    
    # Imprimir resumo
    print("\nMelhor configuração encontrada:")
    print(f"  Clusters: {best_config['n_clusters']}")
    print(f"  Max Depth: {best_config['max_depth']}")
    print(f"  N Estimators: {best_config['n_estimators']}")
    print(f"  Precision: {best_config['global_precision']:.4f}")
    print(f"  Recall: {best_config['global_recall']:.4f}")
    print(f"  F1: {best_config['global_f1']:.4f}")
    
    return best_config

def run_cluster_experiment_with_params(n_clusters_list=[3, 5], max_depth_list=[None], 
                                     n_estimators_list=[200]):
    """
    Executa experimentos completos variando número de clusters e hiperparâmetros.
    
    Args:
        n_clusters_list: Lista de números de clusters para testar
        max_depth_list: Lista de valores de max_depth para testar
        n_estimators_list: Lista de valores de n_estimators para testar
        
    Returns:
        DataFrame com resultados dos experimentos
    """
    data_dir = CONFIG['data_dir']
    
    # Carregar datasets
    datasets = load_processed_datasets(data_dir)
    X_train = datasets['X_train']
    y_train = datasets['y_train']
    X_val = datasets['X_val']
    y_val = datasets['y_val']
    feature_names = datasets['feature_names']
    
    # Identificar features numéricas para o PCA e clustering
    X_train_numeric, X_val_numeric, numeric_cols = identify_numeric_features(X_train, X_val)
    
    # Aplicar PCA
    X_train_pca, X_val_pca, pca_model, n_components, variance_explained = apply_dimension_reduction(
        X_train_numeric, X_val_numeric, variance_threshold=0.8)
    
    # Armazenar resultados
    results_data = []
    
    # Reduzir o número de experimentos mantendo apenas as combinações mais promissoras
    param_combinations = []
    for n_clusters in n_clusters_list:
        for max_depth in max_depth_list:
            for n_estimators in n_estimators_list:
                param_combinations.append((n_clusters, max_depth, n_estimators))
    
    # Limitar o número de combinações se for muito grande
    max_combinations = CONFIG['runs_per_experiment']
    if len(param_combinations) > max_combinations:
        param_combinations = param_combinations[:max_combinations]
        warnings.warn(f"Limitando para {max_combinations} combinações de parâmetros para reduzir número de runs.")
    
    # Criar ou obter experimento
    experiment_name = CONFIG['experiment_tuning']
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Criar diretório temporário para os artefatos
    with tempfile.TemporaryDirectory() as temp_dir:
        # Usar um único run para todos os testes
        with mlflow.start_run(experiment_id=experiment_id, run_name="cluster_parameter_search") as parent_run:
            run_id = parent_run.info.run_id
            
            # Registrar parâmetros gerais
            mlflow.log_params({
                "n_clusters_tested": str(n_clusters_list),
                "max_depth_tested": str(max_depth_list),
                "n_estimators_tested": str(n_estimators_list),
                "total_combinations": len(param_combinations)
            })
            
            # Salvar modelos PCA e informações diretamente como artefatos do MLflow
            config_file = os.path.join(temp_dir, "best_model_config.json")
            with open(config_file, "w") as f:
                json.dump({
                    "pca_n_components": n_components,
                    "pca_variance_explained": float(variance_explained)
                }, f)
            mlflow.log_artifact(config_file)
            
            # Salvar modelo PCA
            pca_file = os.path.join(temp_dir, "pca_model.joblib")
            joblib.dump(pca_model, pca_file)
            mlflow.log_artifact(pca_file)
            
            for n_clusters, max_depth, n_estimators in param_combinations:
                config_name = f"clusters_{n_clusters}_depth_{str(max_depth)}_trees_{n_estimators}"
                print(f"\n{'='*50}")
                print(f"Testando configuração: {config_name}")
                
                # Aplicar clustering
                cluster_labels_train, cluster_labels_val, kmeans_model = cluster_data(
                    X_train_pca, X_val_pca, n_clusters=n_clusters)
                
                # Analisar clusters
                min_conv_cluster_id = analyze_clusters(
                    cluster_labels_train, y_train, cluster_labels_val, y_val)
                    
                # Registrar parâmetros desta configuração
                mlflow.log_params({
                    f"{config_name}_n_clusters": n_clusters,
                    f"{config_name}_max_depth": str(max_depth),
                    f"{config_name}_n_estimators": n_estimators
                })
                
                # Treinar modelos sem criar runs aninhados
                cluster_models = train_cluster_models(
                    X_train, y_train, cluster_labels_train, n_clusters, 
                    max_depth=max_depth, n_estimators=n_estimators
                )
                
                # Salvar modelo kmeans
                kmeans_file = os.path.join(temp_dir, f"kmeans_model_{n_clusters}.joblib")
                joblib.dump(kmeans_model, kmeans_file)
                mlflow.log_artifact(kmeans_file)
                
                # Avaliar com threshold global
                global_results = evaluate_ensemble_with_global_threshold(
                    X_val, y_val, cluster_labels_val, cluster_models
                )
                
                # Avaliar com thresholds por cluster
                cluster_results = evaluate_ensemble_with_cluster_thresholds(
                    X_val, y_val, cluster_labels_val, cluster_models
                )
                
                # Armazenar resultados
                config_results = {
                    'n_clusters': n_clusters,
                    'max_depth': str(max_depth),  # None -> "None"
                    'n_estimators': n_estimators,
                    'global_precision': global_results['precision'],
                    'global_recall': global_results['recall'],
                    'global_f1': global_results['f1'],
                    'global_threshold': global_results['threshold'],
                    'cluster_precision': cluster_results['precision'],
                    'cluster_recall': cluster_results['recall'],
                    'cluster_f1': cluster_results['f1']
                }
                
                results_data.append(config_results)
                
                # Registrar métricas no run principal
                mlflow.log_metrics({
                    f"{config_name}_global_precision": global_results['precision'],
                    f"{config_name}_global_recall": global_results['recall'],
                    f"{config_name}_global_f1": global_results['f1'],
                    f"{config_name}_cluster_precision": cluster_results['precision'],
                    f"{config_name}_cluster_recall": cluster_results['recall'],
                    f"{config_name}_cluster_f1": cluster_results['f1']
                })
                
                # Salvar modelos individualmente
                for model_info in cluster_models:
                    cluster_id = model_info["cluster_id"]
                    model_filename = os.path.join(temp_dir, f"cluster_{n_clusters}_id_{cluster_id}_model.joblib")
                    joblib.dump(model_info["model"], model_filename)
                    mlflow.log_artifact(model_filename)
                    
                    # Salvar informações do modelo
                    info_filename = os.path.join(temp_dir, f"cluster_{n_clusters}_id_{cluster_id}_info.json")
                    with open(info_filename, "w") as f:
                        json.dump({
                            "threshold": float(model_info["threshold"]),
                            "n_samples": int(model_info["n_samples"]),
                            "conversion_rate": float(model_info["conversion_rate"])
                        }, f)
                    mlflow.log_artifact(info_filename)
        
        # Criar DataFrame com resultados
        results_df = pd.DataFrame(results_data)
        
        # Encontrar melhor configuração
        best_config = find_best_model_configuration(results_data)
        
        # Salvar configuração como artefato MLflow
        if best_config:
            best_config_file = os.path.join(temp_dir, "best_config.json")
            with open(best_config_file, "w") as f:
                json.dump(best_config, f, indent=2)
            
            # Garantir que um novo run MLflow não está ativo antes de registrar artefatos
            active_run = mlflow.active_run()
            if active_run:
                # Se ainda estiver ativo, encerre-o
                mlflow.end_run()
            
            # Iniciar um novo run apenas para registrar a configuração
            with mlflow.start_run(experiment_id=experiment_id, run_name="best_config"):
                mlflow.log_artifact(best_config_file)
                
                # Também registrar como parâmetros
                mlflow.log_params({
                    "best_n_clusters": best_config['n_clusters'],
                    "best_max_depth": best_config['max_depth'],
                    "best_n_estimators": best_config['n_estimators'],
                    "best_precision": best_config['global_precision'],
                    "best_recall": best_config['global_recall'],
                    "best_f1": best_config['global_f1']
                })
    
    return results_df, best_config, run_id

def run_best_cluster_configuration(best_config=None, tuning_run_id=None):
    """
    Executa o experimento com a melhor configuração encontrada.
    
    Args:
        best_config: Dicionário com a melhor configuração (opcional)
        tuning_run_id: ID do run de tuning (para buscar config se best_config=None)
        
    Returns:
        Resultados da melhor configuração
    """
    data_dir = CONFIG['data_dir']
    
    # Se não foi fornecida uma configuração, buscar do MLflow
    if best_config is None and tuning_run_id is not None:
        try:
            # Tentar buscar do MLflow
            client = mlflow.tracking.MlflowClient()
            artifacts_dir = client.download_artifacts(tuning_run_id, "best_config.json")
            with open(artifacts_dir, "r") as f:
                best_config = json.load(f)
            print(f"Carregada configuração do MLflow run {tuning_run_id}")
        except Exception as e:
            print(f"Erro ao carregar configuração do MLflow: {e}")
            # Usar configuração padrão
            best_config = {
                'n_clusters': 5,  
                'max_depth': "None",  # Nota: será convertido para None
                'n_estimators': 200,
                'global_precision': 0.94,
                'global_recall': 0.27
            }
            print("Usando configuração padrão definida no código.")
    elif best_config is None:
        # Usar configuração padrão
        best_config = {
            'n_clusters': 5,  
            'max_depth': "None",  # Nota: será convertido para None
            'n_estimators': 200,
            'global_precision': 0.94,
            'global_recall': 0.27
        }
        print("Usando configuração padrão definida no código.")
    
    # Converter de string para None se necessário
    if best_config['max_depth'] == "None":
        best_config['max_depth'] = None
    
    n_clusters = best_config['n_clusters']
    max_depth = best_config['max_depth']
    n_estimators = best_config['n_estimators']
    
    print(f"Executando experimento com melhor configuração:")
    print(f"  Clusters: {n_clusters}")
    print(f"  Max Depth: {max_depth}")
    print(f"  N Estimators: {n_estimators}")
    
    # Carregar datasets
    datasets = load_processed_datasets(data_dir)
    X_train = datasets['X_train']
    y_train = datasets['y_train']
    X_val = datasets['X_val']
    y_val = datasets['y_val']
    
    # Identificar features numéricas para o PCA e clustering
    X_train_numeric, X_val_numeric, numeric_cols = identify_numeric_features(X_train, X_val)
    
    # Aplicar PCA
    X_train_pca, X_val_pca, pca_model, n_components, variance_explained = apply_dimension_reduction(
        X_train_numeric, X_val_numeric, variance_threshold=0.8)
    
    # Aplicar clustering
    cluster_labels_train, cluster_labels_val, kmeans_model = cluster_data(
        X_train_pca, X_val_pca, n_clusters=n_clusters)
    
    # Analisar clusters (resumido)
    min_conv_cluster_id = analyze_clusters(
        cluster_labels_train, y_train, cluster_labels_val, y_val)
    
    # Para o modelo final, fazemos análise detalhada do cluster de menor conversão
    analyze_low_conversion_cluster(
        X_train, y_train, cluster_labels_train, min_conv_cluster_id)
    
    # Criar ou obter experimento
    experiment_name = CONFIG['experiment_best']
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Criar diretório temporário para os artefatos
    with tempfile.TemporaryDirectory() as temp_dir:
        # Garantimos que nenhum run MLflow está ativo antes de iniciar um novo
        active_run = mlflow.active_run()
        if active_run:
            print(f"Encerrando run MLflow ativo: {active_run.info.run_id}")
            mlflow.end_run()
            
        # Usar run principal para todo o processamento final
        with mlflow.start_run(experiment_id=experiment_id, run_name="best_cluster_model") as run:
            run_id = run.info.run_id
            
            # Registrar parâmetros
            mlflow.log_params({
                "n_clusters": n_clusters,
                "max_depth": str(max_depth),
                "n_estimators": n_estimators,
                "min_conversion_cluster_id": min_conv_cluster_id
            })
            
            # Treinar modelos
            cluster_models = train_cluster_models(
                X_train, y_train, cluster_labels_train, n_clusters,
                max_depth=max_depth, n_estimators=n_estimators
            )
            
            # Avaliar com threshold global
            global_results = evaluate_ensemble_with_global_threshold(
                X_val, y_val, cluster_labels_val, cluster_models
            )
            
            # Avaliar com thresholds por cluster
            cluster_results = evaluate_ensemble_with_cluster_thresholds(
                X_val, y_val, cluster_labels_val, cluster_models
            )
            
            # Comparar resultados e escolher o melhor (global vs. cluster)
            if global_results['f1'] >= cluster_results['f1']:
                best_approach = "global_threshold"
                best_metrics = global_results
            else:
                best_approach = "cluster_thresholds"
                best_metrics = cluster_results
                
            # Comparar com baseline
            baseline_precision = 0.94
            baseline_recall = 0.27
            baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall)
            
            print("\nComparação com o Baseline:")
            print(f"  Baseline: Precision={baseline_precision:.4f}, Recall={baseline_recall:.4f}, F1={baseline_f1:.4f}")
            print(f"  Global Threshold: Precision={global_results['precision']:.4f}, Recall={global_results['recall']:.4f}, F1={global_results['f1']:.4f}")
            print(f"  Cluster Thresholds: Precision={cluster_results['precision']:.4f}, Recall={cluster_results['recall']:.4f}, F1={cluster_results['f1']:.4f}")
            
            # Verificar se nosso melhor resultado é inferior ao baseline
            if best_metrics['f1'] < baseline_f1:
                print("\nATENÇÃO: O melhor modelo encontrado tem desempenho inferior ao baseline!")
                print("Considerando usar o modelo baseline em vez do modelo treinado.")
                
                # Definir uma flag para usar o baseline em vez do modelo treinado
                use_baseline = True
            else:
                use_baseline = False
                
            # Registrar qual abordagem foi melhor
            mlflow.log_param("best_approach", best_approach)
            mlflow.log_param("use_baseline", use_baseline)
            mlflow.log_metrics({
                "best_precision": best_metrics['precision'],
                "best_recall": best_metrics['recall'],
                "best_f1": best_metrics['f1'],
                "baseline_f1": baseline_f1,
                "improvement_over_baseline": best_metrics['f1'] - baseline_f1
            })
            
            # Salvar modelos como artefatos do MLflow
            pca_file = os.path.join(temp_dir, "pca_model.joblib")
            kmeans_file = os.path.join(temp_dir, "kmeans_model.joblib")
            joblib.dump(pca_model, pca_file)
            joblib.dump(kmeans_model, kmeans_file)
            mlflow.log_artifact(pca_file)
            mlflow.log_artifact(kmeans_file)
            
            # Salvar informações dos clusters
            cluster_info = {
                "approach": best_approach,
                "n_clusters": int(n_clusters),
                "global_threshold": float(global_results['threshold']),
                "cluster_thresholds": {str(k): float(v) for k, v in cluster_results.get('thresholds', {}).items()} if cluster_results else {},
                "use_baseline": use_baseline,
                "baseline": {
                    "precision": baseline_precision,
                    "recall": baseline_recall,
                    "f1": baseline_f1
                }
            }
            
            cluster_info_file = os.path.join(temp_dir, "cluster_info.json")
            with open(cluster_info_file, "w") as f:
                json.dump(cluster_info, f, indent=2)
            mlflow.log_artifact(cluster_info_file)
            
            # Se devemos usar os modelos treinados, salve-os
            if not use_baseline:
                # Salvar modelo para cada cluster
                for model_info in cluster_models:
                    cluster_id = model_info["cluster_id"]
                    
                    # Salvar modelo
                    model_filename = os.path.join(temp_dir, f"cluster_{cluster_id}_model.joblib")
                    joblib.dump(model_info["model"], model_filename)
                    mlflow.log_artifact(model_filename)
                    
                    # Salvar threshold e metadados
                    info_filename = os.path.join(temp_dir, f"cluster_{cluster_id}_info.json")
                    with open(info_filename, "w") as f:
                        json.dump({
                            "threshold": float(model_info["threshold"]),
                            "n_samples": int(model_info["n_samples"]),
                            "conversion_rate": float(model_info["conversion_rate"])
                        }, f)
                    mlflow.log_artifact(info_filename)
            
            # Também salvar a performance final
            performance_summary = {
                "baseline": {
                    "precision": baseline_precision,
                    "recall": baseline_recall,
                    "f1": baseline_f1
                },
                "global_threshold": {
                    "precision": float(global_results["precision"]),
                    "recall": float(global_results["recall"]),
                    "f1": float(global_results["f1"])
                },
                "cluster_thresholds": {
                    "precision": float(cluster_results["precision"]),
                    "recall": float(cluster_results["recall"]),
                    "f1": float(cluster_results["f1"])
                },
                "best_approach": best_approach,
                "use_baseline": use_baseline,
                "metrics": {
                    "precision": float(best_metrics["precision"]),
                    "recall": float(best_metrics["recall"]),
                    "f1": float(best_metrics["f1"])
                }
            }
        
            performance_file = os.path.join(temp_dir, "performance_summary.json")
            with open(performance_file, "w") as f:
                json.dump(performance_summary, f, indent=2)
            mlflow.log_artifact(performance_file)
    
    return {
        "global_results": global_results,
        "cluster_results": cluster_results,
        "kmeans_model": kmeans_model,
        "pca_model": pca_model,
        "cluster_models": cluster_models,
        "best_approach": best_approach,
        "best_metrics": best_metrics,
        "use_baseline": use_baseline,
        "run_id": run_id
    }

def run_full_cluster_experiment():
    """
    Função principal que executa o experimento completo.
    
    Returns:
        Resultados completos do experimento
    """
    print("Iniciando experimento completo de modelos baseados em clusters...")
    
    # Configurar MLflow para usar o diretório correto
    mlflow_uri = configure_mlflow()
    
    # Limpar arquivos anteriores do MLflow
    active_run = mlflow.active_run()
    if active_run:
        print(f"Encerrando run MLflow ativo: {active_run.info.run_id}")
        mlflow.end_run()
    
    # Executar experimento com combinações reduzidas de parâmetros
    results_df, best_config, tuning_run_id = run_cluster_experiment_with_params(
        n_clusters_list=[3, 5],  # Reduzido de [3, 5, 7]
        max_depth_list=[None],   # Reduzido de [None, 15]
        n_estimators_list=[200], # Reduzido de [100, 200]
    )
    
    # Garantir que todos os runs ativos sejam encerrados antes de prosseguir
    active_run = mlflow.active_run()
    if active_run:
        print(f"Encerrando run MLflow ativo antes de continuar: {active_run.info.run_id}")
        mlflow.end_run()
    
    # Executar melhor configuração
    print("\nExecutando experimento com a melhor configuração encontrada...")
    best_result = run_best_cluster_configuration(best_config, tuning_run_id)
    
    print("\nExperimento completo concluído!")
    print(f"Logs MLflow em: {CONFIG['mlflow_dir']}")
    
    # Verificar se o melhor modelo é inferior ao baseline
    baseline_f1 = 2 * (0.94 * 0.27) / (0.94 + 0.27)
    best_f1 = best_result["best_metrics"]["f1"]
    
    if best_f1 < baseline_f1:
        print("\nAVISO: O melhor modelo encontrado tem performance inferior ao baseline!")
        print("Usando o modelo baseline para a produção conforme configurado.")
    
    return {
        "results_df": results_df,
        "best_config": best_config,
        "best_result": best_result,
        "tuning_run_id": tuning_run_id,
        "best_run_id": best_result["run_id"]
    }

if __name__ == "__main__":
    # Limpar runs MLflow ativos ao iniciar (em caso de falhas anteriores)
    active_run = mlflow.active_run()
    if active_run:
        print(f"Encerrando run MLflow ativo: {active_run.info.run_id}")
        mlflow.end_run()
    
    # Executar experimento
    results = run_full_cluster_experiment()
    
    # Mensagem final
    print("\nExperimento de modelos baseados em cluster concluído com sucesso!")