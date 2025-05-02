import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Adicionar o caminho do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar funções existentes
from src.evaluation.mlflow_utils import (
    setup_mlflow_tracking, 
    find_optimal_threshold, 
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_prob_histogram
)

def load_and_align_datasets(train_path, val_path):
    """
    Carrega os datasets e garante que tenham as mesmas colunas,
    identificando e preparando apenas colunas estritamente numéricas.
    
    Args:
        train_path: Caminho para o dataset de treino
        val_path: Caminho para o dataset de validação
        
    Returns:
        Tupla (X_train_numeric, y_train, X_val_numeric, y_val, numeric_cols)
    """
    print("Carregando datasets...")
    
    # Carregar dados
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Identificar coluna target
    target_col = 'target'
    
    # Extrair target
    y_train = train_df[target_col].copy()
    y_val = val_df[target_col].copy()
    
    # Remover coluna target dos conjuntos de features
    X_train_full = train_df.drop(columns=[target_col])
    X_val_full = val_df.drop(columns=[target_col])
    
    # Encontrar as colunas em comum entre os dois conjuntos
    common_columns = list(set(X_train_full.columns).intersection(set(X_val_full.columns)))
    print(f"Total de features originais no treino: {X_train_full.shape[1]}")
    print(f"Total de features originais na validação: {X_val_full.shape[1]}")
    print(f"Total de features em comum: {len(common_columns)}")
    
    # Identificar colunas estritamente numéricas
    numeric_cols = []
    for col in common_columns:
        # Verificar se as colunas contêm apenas valores numéricos
        # ou NaN (que podem ser tratados)
        is_numeric_train = pd.to_numeric(X_train_full[col], errors='coerce').notna().all()
        is_numeric_val = pd.to_numeric(X_val_full[col], errors='coerce').notna().all()
        
        if is_numeric_train and is_numeric_val:
            numeric_cols.append(col)
    
    print(f"Total de features estritamente numéricas: {len(numeric_cols)}")
    
    # Criar DataFrames apenas com colunas numéricas
    X_train_numeric = X_train_full[numeric_cols].copy()
    X_val_numeric = X_val_full[numeric_cols].copy()
    
    # Converter para float
    X_train_numeric = X_train_numeric.astype(float)
    X_val_numeric = X_val_numeric.astype(float)
    
    # Substituir valores NaN por 0
    X_train_numeric.fillna(0, inplace=True)
    X_val_numeric.fillna(0, inplace=True)
    
    # Log das colunas não-numéricas para referência
    non_numeric = set(common_columns) - set(numeric_cols)
    print(f"Total de colunas não-numéricas (excluídas do treinamento): {len(non_numeric)}")
    if len(non_numeric) > 0:
        print("Exemplos de colunas não-numéricas (excluídas):")
        for col in list(non_numeric)[:5]:
            print(f"  - {col}: {X_train_full[col].iloc[0]}")
    
    print(f"Conjunto de treino final (apenas numérico): {X_train_numeric.shape}")
    print(f"Conjunto de validação final (apenas numérico): {X_val_numeric.shape}")
    print(f"Taxa de conversão (treino): {y_train.mean():.4f}")
    print(f"Taxa de conversão (validação): {y_val.mean():.4f}")
    
    return X_train_numeric, y_train, X_val_numeric, y_val, numeric_cols

def apply_dimension_reduction(X_train, X_val, n_components=50, random_state=42):
    """
    Aplica redução de dimensionalidade com PCA.
    
    Args:
        X_train: Features de treino (já numéricas)
        X_val: Features de validação (já numéricas)
        n_components: Número de componentes PCA
        random_state: Semente aleatória para reprodutibilidade
        
    Returns:
        Tupla (X_train_pca, X_val_pca, pca_model)
    """
    print(f"Aplicando PCA para reduzir para {n_components} dimensões...")
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Ajustar número de componentes ao número de features disponíveis
    n_components = min(n_components, min(X_train_scaled.shape[0], X_train_scaled.shape[1]) - 1)
    print(f"Usando {n_components} componentes PCA...")
    
    # Aplicar PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    # Calcular variância explicada
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Variância explicada com {n_components} componentes: {explained_variance:.4f}")
    
    return X_train_pca, X_val_pca, pca

def cluster_data(X_train_pca, X_val_pca, n_clusters=5, random_state=42):
    """
    Aplica K-means para clusterizar os dados no espaço PCA.
    
    Args:
        X_train_pca: Features de treino após PCA
        X_val_pca: Features de validação após PCA
        n_clusters: Número de clusters
        random_state: Semente aleatória para reprodutibilidade
        
    Returns:
        Tupla (cluster_labels_train, cluster_labels_val, kmeans_model)
    """
    print(f"Aplicando K-means com {n_clusters} clusters...")
    
    # Combinar os dados para clustering conjunto
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
    
    # Plotar distribuição dos clusters
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cluster', data=train_clusters_df)
    plt.title('Distribuição dos Clusters no Conjunto de Treino')
    plt.savefig("cluster_distribution.png")
    plt.close()
    
    # Plotar taxa de conversão por cluster
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_stats.index, y=cluster_stats['conversion_rate'])
    plt.title('Taxa de Conversão por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Taxa de Conversão')
    plt.savefig("cluster_conversion_rates.png")
    plt.close()
    
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
        
        # Comparar distribuição dos clusters entre treino e validação
        plt.figure(figsize=(12, 6))
        
        # Calcular proporções para normalizar
        train_props = train_clusters_df['cluster'].value_counts(normalize=True).sort_index()
        val_props = val_clusters_df['cluster'].value_counts(normalize=True).sort_index()
        
        # Criar DataFrame para plot
        compare_df = pd.DataFrame({
            'Treino': train_props,
            'Validação': val_props
        })
        
        compare_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Distribuição Proporcional dos Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Proporção')
        plt.savefig("cluster_distribution_comparison.png")
        plt.close()
        
        # Comparar taxa de conversão por cluster
        plt.figure(figsize=(12, 6))
        
        # Criar DataFrame para plot
        conversion_compare = pd.DataFrame({
            'Treino': cluster_stats['conversion_rate'],
            'Validação': val_cluster_stats['conversion_rate']
        })
        
        conversion_compare.plot(kind='bar', figsize=(12, 6))
        plt.title('Taxa de Conversão por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Taxa de Conversão')
        plt.savefig("cluster_conversion_comparison.png")
        plt.close()

def train_cluster_models(X_train, y_train, cluster_labels_train, n_clusters, experiment_id, artifact_dir, random_state=42):
    """
    Treina modelos específicos para cada cluster.
    
    Args:
        X_train: Features de treino (já numéricas)
        y_train: Target de treino
        cluster_labels_train: Labels de cluster para o conjunto de treino
        n_clusters: Número de clusters
        experiment_id: ID do experimento MLflow
        artifact_dir: Diretório para artefatos
        random_state: Semente aleatória para reprodutibilidade
        
    Returns:
        Lista de dicionários com informações dos modelos treinados
    """
    cluster_models = []
    
    # Para cada cluster
    for cluster_id in range(n_clusters):
        print(f"\n{'='*50}")
        print(f"Treinando modelo para cluster {cluster_id}...")
        
        # Selecionar dados do cluster atual
        cluster_mask = (cluster_labels_train == cluster_id)
        X_train_cluster = X_train[cluster_mask]
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
                
            class_weight = {0: 1, 1: float(scale_pos_weight)}  # Convertendo para float explicitamente
            print(f"  Aplicando class_weight: {class_weight}")
        else:
            class_weight = None
        
        # Criar e treinar modelo
        with mlflow.start_run(experiment_id=experiment_id, 
                              run_name=f"cluster_{cluster_id}_model") as run:
            # Registar meta dados
            mlflow.set_tags({
                "model_type": "random_forest",
                "experiment_type": "cluster_based",
                "cluster_id": cluster_id,
                "cluster_size_train": len(X_train_cluster),
                "conversion_rate_train": float(conversion_rate)
            })
            
            # Criar modelo
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1
            )
            
            # Registrar parâmetros
            mlflow.log_params(model.get_params())
            
            # Treinar modelo
            model.fit(X_train_cluster, y_train_cluster)
            
            # Registrar modelo
            mlflow.sklearn.log_model(model, f"cluster_{cluster_id}_model")
            
            # Adicionar à lista de modelos
            cluster_models.append({
                "cluster_id": cluster_id,
                "model": model,
                "threshold": 0.5,  # Valor inicial, será otimizado depois
                "run_id": run.info.run_id
            })
    
    return cluster_models

def evaluate_models(X_val, y_val, cluster_labels_val, cluster_models, artifact_dir, experiment_id):
    """
    Avalia os modelos específicos por cluster e o ensemble.
    
    Args:
        X_val: Features de validação (já numéricas)
        y_val: Target de validação
        cluster_labels_val: Labels de cluster para o conjunto de validação
        cluster_models: Lista de modelos por cluster
        artifact_dir: Diretório para artefatos
        experiment_id: ID do experimento MLflow
        
    Returns:
        Dicionário com métricas de avaliação
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
            
        # Obter features para este cluster
        X_cluster = X_val[cluster_mask]
        
        if len(X_cluster) > 0:
            model = model_info["model"]
            
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
    
    # Calcular métricas por cluster
    cluster_metrics = {}
    for cluster_id in np.unique(cluster_labels_val):
        cluster_mask = (cluster_labels_val == cluster_id)
        
        # Pular clusters muito pequenos
        if sum(cluster_mask) < 20:
            continue
            
        y_true_cluster = y_val[cluster_mask]
        y_pred_cluster = y_pred[cluster_mask]
        
        # Se o cluster tem amostras de ambas as classes
        if len(np.unique(y_true_cluster)) > 1:
            metrics = {
                "precision": precision_score(y_true_cluster, y_pred_cluster),
                "recall": recall_score(y_true_cluster, y_pred_cluster),
                "f1": f1_score(y_true_cluster, y_pred_cluster),
                "conversion_rate": float(y_true_cluster.mean()),
                "samples": int(sum(cluster_mask))
            }
        else:
            # Lidar com clusters que têm apenas uma classe
            only_class = y_true_cluster.iloc[0]
            all_same_pred = all(y_pred_cluster == only_class)
            
            if only_class == 1 and all_same_pred:
                precision_val, recall_val, f1_val = 1.0, 1.0, 1.0
            elif only_class == 0 and all_same_pred:
                precision_val, recall_val, f1_val = 1.0, 1.0, 1.0
            else:
                precision_val, recall_val, f1_val = 0.0, 0.0, 0.0
                
            metrics = {
                "precision": precision_val,
                "recall": recall_val,
                "f1": f1_val,
                "conversion_rate": float(y_true_cluster.mean()),
                "samples": int(sum(cluster_mask))
            }
            
        cluster_metrics[cluster_id] = metrics
    
    # Registrar resultados no MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name="cluster_ensemble_evaluation"):
        # Registrar métricas do ensemble
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "threshold": best_threshold
        })
        
        # Criar visualizações
        # Matriz de confusão
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            y_val, y_pred, 
            f"Matriz de Confusão - Ensemble de Clusters (threshold={best_threshold:.2f})", 
            cm_path
        )
        mlflow.log_artifact(cm_path)
        
        # Histograma de probabilidades
        hist_path = os.path.join(artifact_dir, "probability_histogram.png")
        plot_prob_histogram(
            y_val, y_pred_proba, best_threshold,
            "Distribuição de Probabilidades - Ensemble de Clusters", 
            hist_path
        )
        mlflow.log_artifact(hist_path)
        
        # Curva Precision-Recall
        pr_path = os.path.join(artifact_dir, "precision_recall_curve.png")
        plot_precision_recall_curve(
            y_val, y_pred_proba,
            "Curva Precision-Recall - Ensemble de Clusters", 
            pr_path
        )
        mlflow.log_artifact(pr_path)
        
        # Gráfico de desempenho por cluster
        plt.figure(figsize=(10, 6))
        
        # Preparar dados para o gráfico
        cluster_ids = []
        precisions = []
        recalls = []
        f1s = []
        
        for cluster_id, metrics in sorted(cluster_metrics.items()):
            cluster_ids.append(f"Cluster {cluster_id}")
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1s.append(metrics["f1"])
        
        if len(cluster_ids) > 0:  # Verificar se há dados para plotar
            x = np.arange(len(cluster_ids))
            width = 0.2
            
            plt.bar(x - width, precisions, width, label='Precision')
            plt.bar(x, recalls, width, label='Recall')
            plt.bar(x + width, f1s, width, label='F1')
            
            plt.axhline(y=precision, color='r', linestyle='--', label=f'Ensemble Precision: {precision:.2f}')
            plt.axhline(y=recall, color='g', linestyle='--', label=f'Ensemble Recall: {recall:.2f}')
            plt.axhline(y=f1, color='b', linestyle='--', label=f'Ensemble F1: {f1:.2f}')
            
            plt.xlabel('Cluster')
            plt.ylabel('Métrica')
            plt.title('Performance por Cluster vs. Ensemble')
            plt.xticks(x, cluster_ids, rotation=45)
            plt.legend()
            plt.tight_layout()
            performance_path = os.path.join(artifact_dir, "cluster_performance.png")
            plt.savefig(performance_path)
            plt.close()
            mlflow.log_artifact(performance_path)
    
    # Imprimir resultados
    print("\nResultados do Ensemble:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"  Threshold: {best_threshold:.4f}")
    print("\nMatrix de Confusão:")
    print(cm)
    
    print("\nDesempenho por Cluster:")
    for cluster_id, metrics in sorted(cluster_metrics.items()):
        print(f"  Cluster {cluster_id}:")
        print(f"    Amostras: {metrics['samples']}")
        print(f"    Taxa de Conversão: {metrics['conversion_rate']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
    
    # Comparação com baseline
    baseline_precision = 0.94
    baseline_recall = 0.27
    baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall)
    
    print("\nComparação com o Baseline:")
    print(f"  Baseline: Precision={baseline_precision:.2f}, Recall={baseline_recall:.2f}, F1={baseline_f1:.2f}")
    print(f"  Cluster Ensemble: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
    print(f"  Diferença: Precision={precision-baseline_precision:.2f}, Recall={recall-baseline_recall:.2f}, F1={f1-baseline_f1:.2f}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": best_threshold,
        "confusion_matrix": cm,
        "cluster_metrics": cluster_metrics
    }

def run_cluster_experiment():
    """
    Executa o experimento completo com modelos baseados em cluster.
    """
    # Configurações
    base_dir = os.path.expanduser("~")
    data_dir = os.path.join(base_dir, "desktop/smart_ads/data/02_3_processed_text_code6")
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    mlflow_dir = os.path.join(base_dir, "desktop/smart_ads/models/mlflow")
    artifact_dir = os.path.join(base_dir, "desktop/smart_ads/models/artifacts/cluster_experiment")
    
    # Criar diretório de artefatos
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Configurar MLflow
    experiment_id = setup_mlflow_tracking(
        tracking_dir=mlflow_dir,
        experiment_name="smart_ads_cluster_models",
        clean_previous=False
    )
    
    # Carregar e alinhar datasets, identificando colunas numéricas
    X_train, y_train, X_val, y_val, numeric_cols = load_and_align_datasets(train_path, val_path)
    
    # Reduzir dimensionalidade com PCA
    n_components = 50  # Ajuste conforme necessário
    X_train_pca, X_val_pca, pca_model = apply_dimension_reduction(
        X_train, X_val,
        n_components=n_components, 
        random_state=42
    )
    
    # Número fixo de clusters para experimento inicial
    n_clusters = 5
    
    # Clusterizar os dados usando PCA
    cluster_labels_train, cluster_labels_val, kmeans_model = cluster_data(
        X_train_pca, X_val_pca,
        n_clusters=n_clusters,
        random_state=42
    )
    
    # Analisar clusters
    analyze_clusters(cluster_labels_train, y_train, cluster_labels_val, y_val)
    
    # Treinar modelos específicos para cada cluster
    cluster_models = train_cluster_models(
        X_train, y_train,  
        cluster_labels_train,
        n_clusters,
        experiment_id,
        artifact_dir,
        random_state=42
    )
    
    # Avaliar modelos
    results = evaluate_models(
        X_val, y_val,  
        cluster_labels_val,
        cluster_models,
        artifact_dir,
        experiment_id
    )
    
    return results

if __name__ == "__main__":
    # Executar experimento
    results = run_cluster_experiment()
    
    # Mensagem final
    print("\nExperimento com modelos baseados em cluster concluído!")
    print("Verifique os artefatos e visualizações para análise detalhada.")