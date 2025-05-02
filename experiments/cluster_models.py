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
from datetime import datetime
import json

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

def apply_dimension_reduction(X_train_numeric, X_val_numeric, variance_threshold=0.8, max_components=100, random_state=42):
    """
    Aplica redução de dimensionalidade com PCA, determinando automaticamente o número ideal de componentes.
    
    Args:
        X_train_numeric: Features numéricas de treino
        X_val_numeric: Features numéricas de validação
        variance_threshold: Variância explicada cumulativa mínima (default: 0.8)
        max_components: Número máximo de componentes a considerar
        random_state: Semente aleatória para reprodutibilidade
        
    Returns:
        Tupla (X_train_pca, X_val_pca, pca_model, n_components)
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
    
    print(f"Número ideal de componentes: {n_components} (variância explicada: {cumulative_variance[n_components-1]:.4f})")
    
    # Visualizar scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.axhline(y=variance_threshold, color='r', linestyle='--', label=f'Threshold: {variance_threshold}')
    plt.axvline(x=n_components, color='g', linestyle='--', label=f'Componentes: {n_components}')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Cumulativa')
    plt.title('Scree Plot - Variância Explicada por Componentes PCA')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("pca_variance_explained.png")
    plt.close()
    
    # Criar PCA com o número ideal de componentes
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    print(f"Dimensão final após PCA: {X_train_pca.shape}")
    
    return X_train_pca, X_val_pca, pca, n_components

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
    
    # Retornar o cluster com menor taxa de conversão
    min_conv_cluster_id = cluster_stats['conversion_rate'].idxmin()
    print(f"\nCluster com menor taxa de conversão: {min_conv_cluster_id} ({cluster_stats.loc[min_conv_cluster_id, 'conversion_rate']:.4f})")
    
    return min_conv_cluster_id

def analyze_low_conversion_cluster(X_train, y_train, cluster_labels_train, cluster_id, artifact_dir):
    """
    Analisa detalhadamente um cluster com baixa taxa de conversão.
    
    Args:
        X_train: Features completas de treino
        y_train: Target de treino
        cluster_labels_train: Labels de cluster
        cluster_id: ID do cluster a analisar
        artifact_dir: Diretório para salvar artefatos
    """
    print(f"\nAnalisando detalhadamente o Cluster {cluster_id}...")
    
    # Garantir que o diretório existe
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Selecionar dados do cluster
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
    
    print(f"  Tamanho: {cluster_size} amostras ({cluster_size/len(X_train):.2%} do total)")
    print(f"  Taxa de conversão: {conversion_rate:.6f}")
    print(f"  Taxa nos outros clusters: {other_conversion_rate:.6f}")
    print(f"  Razão de conversão (cluster/outros): {conversion_rate/other_conversion_rate if other_conversion_rate > 0 else 0:.4f}")
    
    # Inicializar variável para evitar UnboundLocalError
    top_diff_features = pd.Series()
    
    # Encontrar features que diferenciam este cluster
    try:
        # Filtragem para garantir apenas features numéricas na análise de diferenças
        numeric_columns = []
        for col in X_train.columns:
            try:
                # Verificar se a coluna pode ser convertida para float
                X_cluster[col].astype(float)
                X_other[col].astype(float)
                numeric_columns.append(col)
            except (ValueError, TypeError):
                continue
        
        # Usar apenas colunas numéricas para cálculo de médias e diferenças
        X_cluster_numeric = X_cluster[numeric_columns]
        X_other_numeric = X_other[numeric_columns]
        
        # 1. Comparar médias
        cluster_means = X_cluster_numeric.mean()
        other_means = X_other_numeric.mean()
        
        # Calcular diferença absoluta entre médias
        mean_diff = (cluster_means - other_means).abs()
        top_diff_features = mean_diff.sort_values(ascending=False).head(20)
        
        print("\nFeatures com maior diferença de média:")
        for feature, diff_val in top_diff_features.items():
            print(f"  {feature}: {cluster_means[feature]:.4f} vs {other_means[feature]:.4f} (diff: {diff_val:.4f})")
        
        # Salvar como artefato
        diff_df = pd.DataFrame({
            'feature': top_diff_features.index,
            'cluster_mean': cluster_means[top_diff_features.index],
            'other_mean': other_means[top_diff_features.index],
            'abs_diff': top_diff_features.values
        })
        
        diff_path = os.path.join(artifact_dir, f"cluster_{cluster_id}_top_diff_features.csv")
        diff_df.to_csv(diff_path, index=False)
        
        # 2. Visualizar distribuições das features mais diferentes
        n_cols = 3
        n_rows = 2
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_diff_features.index[:n_cols * n_rows]):
            plt.subplot(n_rows, n_cols, i+1)
            
            sns.histplot(X_cluster[feature], color='blue', alpha=0.5, label=f'Cluster {cluster_id}')
            sns.histplot(X_other[feature], color='red', alpha=0.5, label='Outros clusters')
            
            plt.title(feature)
            plt.legend()
        
        plt.tight_layout()
        hist_path = os.path.join(artifact_dir, f"cluster_{cluster_id}_feature_histograms.png")
        plt.savefig(hist_path)
        plt.close()
        
        # 3. Correlação com o target dentro do cluster
        # Apenas se houver conversões positivas
        if y_cluster.sum() > 0:
            X_with_target = X_cluster_numeric.copy()
            X_with_target['target'] = y_cluster.values
            
            corr = X_with_target.corr()['target'].sort_values()
            
            # Top correlações positivas e negativas
            top_pos_corr = corr.tail(10)
            top_neg_corr = corr.head(10)
            
            print("\nFeatures mais correlacionadas com conversão dentro do cluster:")
            for feature, corr_val in top_pos_corr.items():
                if feature != 'target':  # Evitar mostrar a autocorrelação do target
                    print(f"  {feature}: {corr_val:.4f}")
            
            print("\nFeatures mais negativamente correlacionadas com conversão:")
            for feature, corr_val in top_neg_corr.items():
                print(f"  {feature}: {corr_val:.4f}")
            
            # Visualizar as correlações
            plt.figure(figsize=(12, 8))
            
            # Combinar top positivas e negativas
            top_features = pd.concat([top_neg_corr, top_pos_corr])
            top_features = top_features[top_features.index != 'target']  # Remover autocorrelação
            
            sns.barplot(x=top_features.values, y=top_features.index)
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title(f'Correlação com Target no Cluster {cluster_id}')
            plt.tight_layout()
            
            corr_path = os.path.join(artifact_dir, f"cluster_{cluster_id}_target_correlation.png")
            plt.savefig(corr_path)
            plt.close()
    
    except Exception as e:
        print(f"Erro na análise de features para o cluster {cluster_id}: {e}")
        # Garantir que top_diff_features tenha um valor válido mesmo após exceção
        if top_diff_features.empty:
            top_diff_features = pd.Series(['None available'], index=['error'])
    
    # Salvar relatório de perfil do cluster
    profile = {
        'cluster_id': int(cluster_id),
        'size': int(cluster_size),
        'proportion': float(cluster_size / len(X_train)),
        'conversion_rate': float(conversion_rate),
        'other_conversion_rate': float(other_conversion_rate),
        'conversion_ratio': float(conversion_rate/other_conversion_rate if other_conversion_rate > 0 else 0),
        'timestamp': datetime.now().isoformat(),
        'top_differentiating_features': top_diff_features.index.tolist()
    }
    
    profile_path = os.path.join(artifact_dir, f"cluster_{cluster_id}_profile.json")
    
    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)
    
    print(f"Perfil do cluster salvo em: {profile_path}")

def train_cluster_models(X_train, y_train, cluster_labels_train, n_clusters, 
                         experiment_id, artifact_dir, max_depth=None, n_estimators=100, random_state=42):
    """
    Treina modelos específicos para cada cluster.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        cluster_labels_train: Labels de cluster
        n_clusters: Número de clusters
        experiment_id: ID do experimento MLflow
        artifact_dir: Diretório para artefatos
        max_depth: Profundidade máxima das árvores
        n_estimators: Número de estimadores
        random_state: Semente aleatória
        
    Returns:
        Lista de modelos treinados
    """
    cluster_models = []
    
    # Primeiro, filtrar apenas colunas numéricas para garantir compatibilidade
    numeric_cols = []
    for col in X_train.columns:
        try:
            # Verificar se todos os valores são numéricos
            X_train[col].astype(float)
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    # Usar apenas as colunas numéricas
    X_train_numeric = X_train[numeric_cols].copy().astype(float)
    print(f"  Usando {len(numeric_cols)} features numéricas para treinamento")
    
    # Para cada cluster
    for cluster_id in range(n_clusters):
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
            
            # Registrar parâmetros
            mlflow.log_params({
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "class_weight": str(class_weight),
                "random_state": random_state,
                "num_features": len(numeric_cols)
            })
            
            # Criar modelo
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
            
            # Treinar modelo
            model.fit(X_train_cluster, y_train_cluster)
            
            # Analisar feature importance
            analyze_feature_importance(
                model, X_train_cluster.columns,
                cluster_id, artifact_dir, experiment_id
            )
            
            # Registrar modelo no MLflow
            mlflow.sklearn.log_model(model, f"cluster_{cluster_id}_model")
            
            # Adicionar à lista de modelos
            cluster_models.append({
                "cluster_id": cluster_id,
                "model": model,
                "threshold": 0.5,  # Valor inicial, será otimizado depois
                "run_id": run.info.run_id,
                "features": numeric_cols  # Armazenar quais features o modelo usa
            })
    
    return cluster_models

def analyze_feature_importance(model, feature_names, cluster_id, artifact_dir, experiment_id, top_n=20):
    """
    Analisa a importância das features para um modelo específico de cluster.
    
    Args:
        model: Modelo treinado
        feature_names: Nomes das features
        cluster_id: ID do cluster
        artifact_dir: Diretório para artefatos
        experiment_id: ID do experimento MLflow
        top_n: Número de top features para exibir
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Modelo para cluster {cluster_id} não suporta feature importances")
        return
    
    # Obter importância das features
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Criar DataFrame com as top features
    top_features = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices[:top_n]],
        'Importance': importances[indices[:top_n]]
    })
    
    # Salvar em CSV
    importance_path = os.path.join(artifact_dir, f"cluster_{cluster_id}_feature_importance.csv")
    top_features.to_csv(importance_path, index=False)
    
    # Visualizar importância das features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Features para Cluster {cluster_id}')
    plt.tight_layout()
    
    # Salvar visualização
    viz_path = os.path.join(artifact_dir, f"cluster_{cluster_id}_feature_importance.png")
    plt.savefig(viz_path)
    plt.close()
    
    # Registrar no MLflow
    mlflow.log_artifact(importance_path)
    mlflow.log_artifact(viz_path)
    
    # Imprimir top features
    print(f"\nTop {top_n} features para Cluster {cluster_id}:")
    for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Registrar importância das features no MLflow - com nomes válidos
    for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
        # Usar apenas um índice numérico para evitar problemas com caracteres especiais
        mlflow.log_metric(f"importance_feature_{i+1}_cluster_{cluster_id}", importance)
        
        # Também registrar o nome da feature como um parâmetro
        # Normalizar o nome da feature para evitar caracteres problemáticos
        safe_feature_name = f"feature_{i+1}_name"
        mlflow.log_param(safe_feature_name, str(feature)[:250])  # Limitar tamanho

def evaluate_ensemble_with_global_threshold(X_val, y_val, cluster_labels_val, cluster_models, 
                                          artifact_dir, experiment_id):
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
    
    # Salvar matriz de confusão
    cm_path = os.path.join(artifact_dir, "global_threshold_confusion_matrix.png")
    plot_confusion_matrix(
        y_val, y_pred, 
        f"Matriz de Confusão - Threshold Global ({best_threshold:.2f})", 
        cm_path
    )
    
    # Salvar histograma de probabilidades
    hist_path = os.path.join(artifact_dir, "global_threshold_probabilities.png")
    plot_prob_histogram(
        y_val, y_pred_proba, best_threshold,
        "Distribuição de Probabilidades - Threshold Global", 
        hist_path
    )
    
    # Salvar curva precision-recall
    pr_path = os.path.join(artifact_dir, "global_threshold_pr_curve.png")
    plot_precision_recall_curve(
        y_val, y_pred_proba,
        "Curva Precision-Recall - Threshold Global", 
        pr_path
    )
    
    # Registrar no MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name="ensemble_global_threshold"):
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "threshold": best_threshold
        })
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(hist_path)
        mlflow.log_artifact(pr_path)
    
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
        "confusion_matrix": cm
    }

def evaluate_ensemble_with_cluster_thresholds(X_val, y_val, cluster_labels_val, cluster_models, 
                                            artifact_dir, experiment_id):
    """
    Avalia o ensemble com thresholds específicos por cluster.
    
    Args:
        X_val: Features de validação
        y_val: Target de validação
        cluster_labels_val: Labels de cluster para validação
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
    
    # Salvar matriz de confusão
    cm_path = os.path.join(artifact_dir, "cluster_thresholds_confusion_matrix.png")
    plot_confusion_matrix(
        y_val, y_pred, 
        "Matriz de Confusão - Thresholds por Cluster", 
        cm_path
    )
    
    # Salvar métricas por cluster
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
    
    # Registrar no MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name="ensemble_cluster_thresholds"):
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })
        
        # Registrar thresholds por cluster
        for cluster_id, threshold in cluster_thresholds.items():
            mlflow.log_metric(f"threshold_cluster_{cluster_id}", threshold)
        
        # Registrar métricas por cluster
        for cluster_id, metrics in cluster_metrics.items():
            mlflow.log_metric(f"precision_cluster_{cluster_id}", metrics['precision'])
            mlflow.log_metric(f"recall_cluster_{cluster_id}", metrics['recall'])
            mlflow.log_metric(f"f1_cluster_{cluster_id}", metrics['f1'])
        
        # Registrar artefatos
        mlflow.log_artifact(cm_path)
        
        # Registrar dict com métricas detalhadas
        mlflow.log_dict(cluster_metrics, "cluster_metrics.json")
    
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
        "confusion_matrix": cm
    }

def run_cluster_experiment_with_params(n_clusters_list=[3, 5, 7], max_depth_list=[None, 15], 
                                     n_estimators_list=[100, 200]):
    """
    Executa experimentos completos variando número de clusters e hiperparâmetros.
    
    Args:
        n_clusters_list: Lista de números de clusters para testar
        max_depth_list: Lista de valores de max_depth para testar
        n_estimators_list: Lista de valores de n_estimators para testar
        
    Returns:
        DataFrame com resultados dos experimentos
    """
    # Configurações
    base_dir = os.path.expanduser("~")
    data_dir = os.path.join(base_dir, "desktop/smart_ads/data/02_3_processed_text_code6")
    mlflow_dir = os.path.join(base_dir, "desktop/smart_ads/models/mlflow")
    artifact_dir = os.path.join(base_dir, "desktop/smart_ads/models/artifacts/cluster_experiment")
    
    # Criar diretório de artefatos
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Configurar MLflow
    experiment_id = setup_mlflow_tracking(
        tracking_dir=mlflow_dir,
        experiment_name="smart_ads_cluster_tuning",
        clean_previous=False
    )
    
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
    X_train_pca, X_val_pca, pca_model, n_components = apply_dimension_reduction(
    X_train_numeric, X_val_numeric, variance_threshold=0.8)
    
    # Armazenar resultados
    results_data = []
    
    # Para cada combinação de parâmetros
    for n_clusters in n_clusters_list:
        print(f"\n{'='*50}")
        print(f"Testando com {n_clusters} clusters...")
        
        # Aplicar clustering
        cluster_labels_train, cluster_labels_val, kmeans_model = cluster_data(
            X_train_pca, X_val_pca, n_clusters=n_clusters)
        
        # Analisar clusters
        min_conv_cluster_id = analyze_clusters(
            cluster_labels_train, y_train, cluster_labels_val, y_val)
        
        # Analisar cluster com menor conversão
        analyze_low_conversion_cluster(
            X_train, y_train, cluster_labels_train, min_conv_cluster_id, artifact_dir)
        
        # Para cada combinação de hiperparâmetros
        for max_depth in max_depth_list:
            for n_estimators in n_estimators_list:
                config_name = f"clusters_{n_clusters}_depth_{max_depth}_trees_{n_estimators}"
                print(f"\nTestando configuração: {config_name}")
                
                # Treinar modelos
                cluster_models = train_cluster_models(
                    X_train, y_train, cluster_labels_train, n_clusters,
                    experiment_id, artifact_dir, 
                    max_depth=max_depth, n_estimators=n_estimators
                )
                
                # Avaliar com threshold global
                global_results = evaluate_ensemble_with_global_threshold(
                    X_val, y_val, cluster_labels_val, cluster_models, 
                    artifact_dir, experiment_id
                )
                
                # Avaliar com thresholds por cluster
                cluster_results = evaluate_ensemble_with_cluster_thresholds(
                    X_val, y_val, cluster_labels_val, cluster_models, 
                    artifact_dir, experiment_id
                )
                
                # Armazenar resultados
                results_data.append({
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
                })
    
    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results_data)
    
    # Salvar resultados
    results_path = os.path.join(artifact_dir, "cluster_experiment_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Imprimir melhores resultados
    print("\nMelhores resultados (threshold global):")
    if len(results_df) > 0:
        best_global = results_df.loc[results_df['global_f1'].idxmax()]
        print(best_global)
        
        print("\nMelhores resultados (thresholds por cluster):")
        best_cluster = results_df.loc[results_df['cluster_f1'].idxmax()]
        print(best_cluster)
    
    return results_df

def run_best_cluster_configuration():
    """
    Executa o experimento com a melhor configuração encontrada.
    
    Returns:
        Resultados da melhor configuração
    """
    # Configurações
    base_dir = os.path.expanduser("~")
    data_dir = os.path.join(base_dir, "desktop/smart_ads/data/02_3_processed_text_code6")
    mlflow_dir = os.path.join(base_dir, "desktop/smart_ads/models/mlflow")
    artifact_dir = os.path.join(base_dir, "desktop/smart_ads/models/artifacts/cluster_best")
    
    # Criar diretório de artefatos
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Configurar MLflow
    experiment_id = setup_mlflow_tracking(
        tracking_dir=mlflow_dir,
        experiment_name="smart_ads_cluster_best",
        clean_previous=False
    )
    
    # Parâmetros da melhor configuração
    # Nota: Estes valores seriam determinados após rodar run_cluster_experiment_with_params
    n_clusters = 5  # Exemplo, ajuste conforme resultados
    max_depth = None
    n_estimators = 200
    
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
    X_train_pca, X_val_pca, pca_model, n_components = apply_dimension_reduction(
        X_train_numeric, X_val_numeric, variance_threshold=0.8)
    
    # Agora podemos usar n_components
    print(f"  Componentes PCA: {n_components}")
    
    # Aplicar clustering
    cluster_labels_train, cluster_labels_val, kmeans_model = cluster_data(
        X_train_pca, X_val_pca, n_clusters=n_clusters)
    
    # Analisar clusters
    min_conv_cluster_id = analyze_clusters(
        cluster_labels_train, y_train, cluster_labels_val, y_val)
    
    # Analisar cluster com menor conversão
    analyze_low_conversion_cluster(
        X_train, y_train, cluster_labels_train, min_conv_cluster_id, artifact_dir)
    
    # Treinar modelos
    cluster_models = train_cluster_models(
        X_train, y_train, cluster_labels_train, n_clusters,
        experiment_id, artifact_dir, 
        max_depth=max_depth, n_estimators=n_estimators
    )
    
    # Avaliar com threshold global
    global_results = evaluate_ensemble_with_global_threshold(
        X_val, y_val, cluster_labels_val, cluster_models, 
        artifact_dir, experiment_id
    )
    
    # Avaliar com thresholds por cluster
    cluster_results = evaluate_ensemble_with_cluster_thresholds(
        X_val, y_val, cluster_labels_val, cluster_models, 
        artifact_dir, experiment_id
    )
    
    # Comparar com baseline
    baseline_precision = 0.94
    baseline_recall = 0.27
    baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall)
    
    print("\nComparação com o Baseline:")
    print(f"  Baseline: Precision={baseline_precision:.2f}, Recall={baseline_recall:.2f}, F1={baseline_f1:.2f}")
    print(f"  Global Threshold: Precision={global_results['precision']:.2f}, Recall={global_results['recall']:.2f}, F1={global_results['f1']:.2f}")
    print(f"  Cluster Thresholds: Precision={cluster_results['precision']:.2f}, Recall={cluster_results['recall']:.2f}, F1={cluster_results['f1']:.2f}")
    
    # Registrar modelos para uso futuro
    with mlflow.start_run(experiment_id=experiment_id, run_name="final_cluster_models"):
        # Registrar parâmetros gerais
        mlflow.log_params({
            "n_clusters": n_clusters,
            "max_depth": str(max_depth),
            "n_estimators": n_estimators
        })
        
        # Registrar resultados
        mlflow.log_metrics({
            "global_precision": global_results['precision'],
            "global_recall": global_results['recall'],
            "global_f1": global_results['f1'],
            "cluster_precision": cluster_results['precision'],
            "cluster_recall": cluster_results['recall'],
            "cluster_f1": cluster_results['f1']
        })
        
        # Salvar cluster labels
        cluster_labels_df = pd.DataFrame({
            "index": range(len(cluster_labels_train)),
            "cluster": cluster_labels_train
        })
        labels_path = os.path.join(artifact_dir, "cluster_labels.csv")
        cluster_labels_df.to_csv(labels_path, index=False)
        mlflow.log_artifact(labels_path)
        
        # Salvar PCA e KMeans
        import joblib
        pca_path = os.path.join(artifact_dir, "pca_model.joblib")
        kmeans_path = os.path.join(artifact_dir, "kmeans_model.joblib")
        joblib.dump(pca_model, pca_path)
        joblib.dump(kmeans_model, kmeans_path)
        mlflow.log_artifact(pca_path)
        mlflow.log_artifact(kmeans_path)
        
        # Registrar modelos
        mlflow.sklearn.log_model(kmeans_model, "kmeans_model")
    
    return {
        "global_results": global_results,
        "cluster_results": cluster_results,
        "kmeans_model": kmeans_model,
        "pca_model": pca_model,
        "cluster_models": cluster_models
    }

def run_full_cluster_experiment():
    """
    Função principal que executa o experimento completo.
    
    Returns:
        DataFrame com resultados dos experimentos
    """
    print("Iniciando experimento completo de modelos baseados em clusters...")
    
    # Executar experimento com várias combinações de parâmetros
    results_df = run_cluster_experiment_with_params(
        n_clusters_list=[3, 5, 7],
        max_depth_list=[None, 15],
        n_estimators_list=[100, 200]
    )
    
    # Executar melhor configuração
    print("\nExecutando experimento com a melhor configuração encontrada...")
    best_result = run_best_cluster_configuration()
    
    print("\nExperimento completo concluído!")
    return results_df

if __name__ == "__main__":
    # Executar experimento
    results_df = run_full_cluster_experiment()
    
    # Mensagem final
    print("\nExperimento de modelos baseados em cluster concluído com sucesso!")
    print("Verifique os artefatos e visualizações para análise detalhada.")