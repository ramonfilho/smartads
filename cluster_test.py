#!/usr/bin/env python3
"""
Script simplificado para treinar modelos de cluster.
Foca apenas nas funcionalidades essenciais e garante
o correto funcionamento do MLflow.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import json
import mlflow
import shutil
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Adicionar o caminho do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, project_root)

# Função para encontrar o threshold ótimo
def find_optimal_threshold(y_true, y_pred_proba, metric='f1', thresholds=None):
    """Encontra o threshold ótimo para uma métrica específica."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return {'best_threshold': best_threshold, 'best_score': best_score}

def setup_mlflow():
    """Configura o MLflow corretamente."""
    # Configura e limpa o diretório do MLflow
    mlflow_dir = os.path.join(project_root, "models", "mlflow")
    os.makedirs(mlflow_dir, exist_ok=True)
    
    # Remove a pasta mlruns do diretório principal
    if os.path.exists(os.path.join(project_root, "mlruns")):
        shutil.rmtree(os.path.join(project_root, "mlruns"))
    
    # Configurar MLflow para usar o diretório correto
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_dir}"
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    print(f"MLflow configurado para usar: {mlflow.get_tracking_uri()}")
    
    # Certifique-se de que nenhum run está ativo
    if mlflow.active_run():
        mlflow.end_run()
    
    return mlflow_dir

def load_data():
    """Carrega os dados do projeto."""
    # Caminho CORRETO para os datasets
    data_dir = os.path.join(project_root, "data", "03_4_feature_selection_final")
    
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
    
    return X_train, y_train, X_val, y_val

def prepare_numeric_features(X_train, X_val):
    """Identifica e prepara features numéricas."""
    print("Identificando features numéricas...")
    
    numeric_cols = []
    for col in X_train.columns:
        try:
            X_train[col].astype(float)
            X_val[col].astype(float)
            numeric_cols.append(col)
        except:
            continue
    
    X_train_numeric = X_train[numeric_cols].copy().astype(float)
    X_val_numeric = X_val[numeric_cols].copy().astype(float)
    
    # Substituir NaN por 0
    X_train_numeric.fillna(0, inplace=True)
    X_val_numeric.fillna(0, inplace=True)
    
    print(f"Total de features numéricas: {len(numeric_cols)}")
    
    return X_train_numeric, X_val_numeric, numeric_cols

def apply_pca(X_train_numeric, X_val_numeric):
    """Aplica PCA para redução de dimensionalidade."""
    print("Aplicando PCA...")
    
    # Normalizar dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_val_scaled = scaler.transform(X_val_numeric)
    
    # Aplicar PCA com número fixo de componentes
    n_components = 10
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    variance_explained = sum(pca.explained_variance_ratio_)
    print(f"Dimensão final após PCA: {X_train_pca.shape} (variância explicada: {variance_explained:.4f})")
    
    return X_train_pca, X_val_pca, pca, scaler

def cluster_data(X_train_pca, X_val_pca, n_clusters=3):
    """Aplica K-means para clustering."""
    print(f"Aplicando K-means com {n_clusters} clusters...")
    
    # Combinar dados para clustering conjunto
    X_combined = np.vstack([X_train_pca, X_val_pca])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_combined)
    
    # Separar os clusters
    labels_combined = kmeans.labels_
    train_labels = labels_combined[:len(X_train_pca)]
    val_labels = labels_combined[len(X_train_pca):]
    
    return train_labels, val_labels, kmeans

def analyze_clusters(train_labels, y_train, val_labels, y_val):
    """Analisa os clusters e sua relação com a target."""
    # Análise do treino
    train_clusters_df = pd.DataFrame({
        'cluster': train_labels,
        'target': y_train.values
    })
    
    print("\nAnálise dos clusters no conjunto de treino:")
    train_stats = train_clusters_df.groupby('cluster').agg({
        'target': ['count', 'sum', 'mean']
    })
    train_stats.columns = ['samples', 'conversions', 'conversion_rate']
    print(train_stats)
    
    # Análise da validação
    val_clusters_df = pd.DataFrame({
        'cluster': val_labels,
        'target': y_val.values
    })
    
    print("\nAnálise dos clusters no conjunto de validação:")
    val_stats = val_clusters_df.groupby('cluster').agg({
        'target': ['count', 'sum', 'mean']
    })
    val_stats.columns = ['samples', 'conversions', 'conversion_rate']
    print(val_stats)
    
    # Cluster com menor taxa de conversão
    min_cluster = train_stats['conversion_rate'].idxmin()
    print(f"\nCluster com menor taxa de conversão: {min_cluster} ({train_stats.loc[min_cluster, 'conversion_rate']:.4f})")
    
    return min_cluster, train_stats, val_stats

def train_cluster_models(X_train, y_train, train_labels, n_clusters, X_train_numeric):
    """Treina modelos específicos para cada cluster."""
    # Armazenar modelos e informações
    cluster_models = []
    
    # Features numéricas
    numeric_cols = X_train_numeric.columns.tolist()
    print(f"Usando {len(numeric_cols)} features numéricas para treinamento")
    
    # Treinar um modelo para cada cluster
    for cluster_id in range(n_clusters):
        print(f"\n{'='*50}")
        print(f"Treinando modelo para cluster {cluster_id}...")
        
        # Selecionar dados do cluster
        cluster_mask = (train_labels == cluster_id)
        X_cluster = X_train_numeric[cluster_mask]
        y_cluster = y_train[cluster_mask]
        
        # Stats
        print(f"  Amostras de treino no cluster: {len(X_cluster)}")
        print(f"  Taxa de conversão (treino): {y_cluster.mean():.4f}")
        
        # Pular clusters pequenos
        if len(X_cluster) < 100:
            print(f"  Cluster {cluster_id} tem poucos dados (<100), pulando.")
            continue
        
        # Calcular balanceamento de classe
        conversion_rate = y_cluster.mean()
        if conversion_rate > 0 and conversion_rate < 0.5:
            scale_pos_weight = min((1 - conversion_rate) / conversion_rate, 100)
            class_weight = {0: 1, 1: float(scale_pos_weight)}
            print(f"  Aplicando class_weight: {class_weight}")
        else:
            class_weight = None
        
        # Treinar modelo
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_cluster, y_cluster)
        
        # Armazenar informações
        cluster_models.append({
            "cluster_id": cluster_id,
            "model": model,
            "threshold": 0.5,  # Será otimizado depois
            "features": numeric_cols,
            "n_samples": len(X_cluster),
            "conversion_rate": float(conversion_rate)
        })
    
    return cluster_models

def evaluate_ensemble(X_val, y_val, val_labels, cluster_models, X_val_numeric):
    """Avalia o ensemble de modelos."""
    # Mapear cluster para modelo
    cluster_model_map = {model["cluster_id"]: model for model in cluster_models}
    
    # Arrays para previsões
    y_pred_proba = np.zeros_like(y_val, dtype=float)
    
    # Fazer previsões para cada cluster
    for cluster_id, model_info in cluster_model_map.items():
        cluster_mask = (val_labels == cluster_id)
        
        if not any(cluster_mask):
            continue
        
        model = model_info["model"]
        X_cluster = X_val_numeric[cluster_mask]
        
        if len(X_cluster) > 0:
            proba = model.predict_proba(X_cluster)[:, 1]
            y_pred_proba[cluster_mask] = proba
    
    # Encontrar threshold ótimo
    threshold_results = find_optimal_threshold(y_val, y_pred_proba)
    best_threshold = threshold_results['best_threshold']
    
    # Aplicar threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Calcular métricas
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    
    # Imprimir resultados
    print("\nResultados do Ensemble:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Threshold: {best_threshold:.4f}")
    print("\nMatrix de Confusão:")
    print(cm)
    
    # Atualizar thresholds dos modelos
    for model_info in cluster_models:
        model_info["threshold"] = best_threshold
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": best_threshold,
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred,
        "probabilities": y_pred_proba
    }

def run_experiment():
    """Função principal para executar o experimento completo."""
    # Configurar MLflow corretamente
    mlflow_dir = setup_mlflow()
    
    # Criar experimento
    experiment_name = "smart_ads_cluster_ensemble"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Criar diretório temporário para artefatos
    with tempfile.TemporaryDirectory() as temp_dir:
        # Carregar e preparar dados
        X_train, y_train, X_val, y_val = load_data()
        X_train_numeric, X_val_numeric, numeric_cols = prepare_numeric_features(X_train, X_val)
        X_train_pca, X_val_pca, pca, scaler = apply_pca(X_train_numeric, X_val_numeric)
        
        # Aplicar clustering
        n_clusters = 3  # Fixo
        train_labels, val_labels, kmeans = cluster_data(X_train_pca, X_val_pca, n_clusters)
        
        # Analisar clusters
        min_cluster, train_stats, val_stats = analyze_clusters(train_labels, y_train, val_labels, y_val)
        
        # Iniciar run MLflow
        with mlflow.start_run(experiment_id=experiment_id, run_name="cluster_ensemble_model") as run:
            # Registrar parâmetros
            mlflow.log_params({
                "n_clusters": n_clusters,
                "n_estimators": 200,
                "max_depth": "None",
                "pca_components": 10
            })
            
            # Treinar modelos
            cluster_models = train_cluster_models(X_train, y_train, train_labels, n_clusters, X_train_numeric)
            
            # Avaliar ensemble
            results = evaluate_ensemble(X_val, y_val, val_labels, cluster_models, X_val_numeric)
            
            # Verificar com baseline
            baseline_precision = 0.94
            baseline_recall = 0.27
            baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall)
            
            print("\nComparação com o Baseline:")
            print(f"  Baseline: Precision={baseline_precision:.4f}, Recall={baseline_recall:.4f}, F1={baseline_f1:.4f}")
            print(f"  Ensemble: Precision={results['precision']:.4f}, Recall={results['recall']:.4f}, F1={results['f1']:.4f}")
            
            # Melhor que baseline?
            better_than_baseline = results['f1'] > baseline_f1
            
            # Registrar métricas
            mlflow.log_metrics({
                "precision": results['precision'],
                "recall": results['recall'],
                "f1": results['f1'],
                "threshold": results['threshold'],
                "baseline_f1": baseline_f1,
                "improvement": results['f1'] - baseline_f1
            })
            
            # Salvar modelo de cada cluster
            for model_info in cluster_models:
                cluster_id = model_info["cluster_id"]
                
                # Salvar modelo
                model_path = os.path.join(temp_dir, f"cluster_{cluster_id}_model.joblib")
                joblib.dump(model_info["model"], model_path)
                mlflow.log_artifact(model_path)
                
                # Salvar metadados
                info_path = os.path.join(temp_dir, f"cluster_{cluster_id}_info.json")
                with open(info_path, 'w') as f:
                    json.dump({
                        "threshold": float(model_info["threshold"]),
                        "n_samples": int(model_info["n_samples"]),
                        "conversion_rate": float(model_info["conversion_rate"])
                    }, f, indent=2)
                mlflow.log_artifact(info_path)
            
            # Salvar outros modelos
            pca_path = os.path.join(temp_dir, "pca_model.joblib")
            kmeans_path = os.path.join(temp_dir, "kmeans_model.joblib")
            scaler_path = os.path.join(temp_dir, "scaler_model.joblib")
            
            joblib.dump(pca, pca_path)
            joblib.dump(kmeans, kmeans_path)
            joblib.dump(scaler, scaler_path)
            
            mlflow.log_artifact(pca_path)
            mlflow.log_artifact(kmeans_path)
            mlflow.log_artifact(scaler_path)
            
            # Salvar info geral
            summary_path = os.path.join(temp_dir, "cluster_summary.json")
            with open(summary_path, 'w') as f:
                json.dump({
                    "baseline": {
                        "precision": baseline_precision,
                        "recall": baseline_recall,
                        "f1": baseline_f1
                    },
                    "ensemble": {
                        "precision": float(results["precision"]),
                        "recall": float(results["recall"]),
                        "f1": float(results["f1"]),
                        "threshold": float(results["threshold"])
                    },
                    "better_than_baseline": better_than_baseline,
                    "n_clusters": n_clusters,
                    "min_conversion_cluster": int(min_cluster)
                }, f, indent=2)
            mlflow.log_artifact(summary_path)
            
            # Registrar modelo completo
            mlflow.sklearn.log_model(kmeans, "kmeans_model")
            
            print("\nExperimento completo e resultados registrados com sucesso!")
            print(f"Run ID: {run.info.run_id}")
            print(f"Artefatos salvos em: {mlflow.get_artifact_uri()}")
    
    return {
        "run_id": run.info.run_id,
        "results": results,
        "better_than_baseline": better_than_baseline
    }

if __name__ == "__main__":
    print("Iniciando experimento simplificado de modelos baseados em clusters...")
    
    # Encerrar qualquer run ativo
    if mlflow.active_run():
        mlflow.end_run()
    
    # Executar experimento
    results = run_experiment()
    
    print("\nExperimento de modelos baseados em cluster concluído com sucesso!")