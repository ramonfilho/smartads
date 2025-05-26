import os
import sys
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import json
import mlflow
import joblib
import contextlib
from datetime import datetime

# Adicionar caminho do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar a classe GMM_Wrapper do módulo compartilhado
from src.modeling.gmm_wrapper import GMM_Wrapper

# Importar funções de avaliação existentes
from src.utils.mlflow_utils import setup_mlflow_tracking, find_optimal_threshold

# Configuração centralizada
CONFIG = {
    'base_dir': os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'data_dir': os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data/04_feature_engineering_2"),  # Atualizado para usar o novo dataset
    'mlflow_dir': os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "models/mlflow"),  # Alterado para models/mlflow
    'artifact_dir': os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "models/artifacts"),
    'experiment_name': "smart_ads_gmm_optimized_full",  # Nome atualizado para refletir o conjunto de dados completo
    'model_params': {
        'max_depth': None,
        'n_estimators': 200,
        'random_state': 42
    },
    # Parâmetros ótimos do GMM já identificados
    'gmm_params': {
        'n_components': 3,
        'covariance_type': 'spherical'
    }
}

# Context manager para garantir que runs do MLflow sejam encerrados
@contextlib.contextmanager
def safe_mlflow_run(experiment_id=None, run_name=None, nested=False):
    """Context manager para garantir que runs do MLflow sejam encerrados corretamente."""
    active_run = mlflow.active_run()
    if active_run and not nested:
        yield active_run
    else:
        run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=nested)
        try:
            yield run
        finally:
            mlflow.end_run()

def create_directory_structure():
    """Cria a estrutura de diretórios para o experimento."""
    # Diretórios base
    os.makedirs(CONFIG['mlflow_dir'], exist_ok=True)
    os.makedirs(CONFIG['artifact_dir'], exist_ok=True)
    
    # Diretório para GMM
    gmm_dir = os.path.join(CONFIG['artifact_dir'], 'gmm_optimized')
    os.makedirs(gmm_dir, exist_ok=True)
    
    return {
        'gmm_dir': gmm_dir,
    }

def load_data():
    """Carrega e prepara o dataset."""
    print("Carregando datasets...")
    
    # Caminhos dos arquivos
    train_path = os.path.join(CONFIG['data_dir'], "train.csv")
    val_path = os.path.join(CONFIG['data_dir'], "validation.csv")
    
    # Verificar existência
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Datasets não encontrados em {CONFIG['data_dir']}")
    
    # Carregar dados
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
    
    # Garantir colunas iguais
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

def prepare_numeric_features(X_train, X_val):
    """Identifica e prepara features numéricas para clustering."""
    print("Identificando features numéricas...")
    
    # Identificar colunas numéricas
    numeric_cols = []
    for col in X_train.columns:
        try:
            X_train[col].astype(float)
            X_val[col].astype(float)
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    # Extrair e processar colunas numéricas
    X_train_numeric = X_train[numeric_cols].copy().astype(float)
    X_val_numeric = X_val[numeric_cols].copy().astype(float)
    
    # Substituir valores ausentes
    X_train_numeric.fillna(0, inplace=True)
    X_val_numeric.fillna(0, inplace=True)
    
    print(f"Total de features numéricas: {len(numeric_cols)}")
    
    return X_train_numeric, X_val_numeric, numeric_cols

def apply_pca(X_train_numeric, X_val_numeric, variance_threshold=0.8, max_components=100, random_state=42):
    """Aplica PCA para redução de dimensionalidade."""
    print("Aplicando PCA...")
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_val_scaled = scaler.transform(X_val_numeric)
    
    # Determinar componentes
    max_possible = min(X_train_scaled.shape[0], X_train_scaled.shape[1]) - 1
    max_components = min(max_components, max_possible)
    
    # Análise de variância
    pca_analysis = PCA(n_components=max_components, random_state=random_state)
    pca_analysis.fit(X_train_scaled)
    
    # Calcular variância explicada
    cumulative_variance = np.cumsum(pca_analysis.explained_variance_ratio_)
    
    # Encontrar número ideal de componentes
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    n_components = max(n_components, 10)  # Garantir mínimo de 10 componentes
    
    print(f"Número ideal de componentes: {n_components} (variância explicada: {cumulative_variance[n_components-1]:.4f})")
    
    # Aplicar PCA final
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    print(f"Dimensão final após PCA: {X_train_pca.shape}")
    
    return X_train_pca, X_val_pca, pca, scaler

def analyze_clusters(cluster_labels, y, prefix="treino"):
    """Analisa clusters quanto à taxa de conversão e distribuição."""
    # Verificar se os arrays têm o mesmo tamanho
    if len(cluster_labels) != len(y):
        print(f"AVISO: Tamanhos diferentes - cluster_labels: {len(cluster_labels)}, y: {len(y)}")
        # Ajustar para o menor tamanho
        min_size = min(len(cluster_labels), len(y))
        cluster_labels = cluster_labels[:min_size]
        if isinstance(y, pd.Series):
            y = y.iloc[:min_size]
        else:
            y = y[:min_size]
        print(f"  Ajustados para tamanho comum: {min_size}")
    
    # Adicionar labels de cluster e target a um DataFrame
    clusters_df = pd.DataFrame({
        'cluster': cluster_labels,
        'target': y.values if isinstance(y, pd.Series) else y
    })
    
    # Análise dos clusters
    print(f"\nAnálise dos clusters no conjunto de {prefix}:")
    cluster_stats = clusters_df.groupby('cluster').agg({
        'target': ['count', 'sum', 'mean']
    })
    cluster_stats.columns = ['samples', 'conversions', 'conversion_rate']
    print(cluster_stats)
    
    # Identificar cluster com menor taxa de conversão
    if not cluster_stats.empty:
        try:
            min_conv_cluster_id = cluster_stats['conversion_rate'].idxmin()
            min_conv_rate = cluster_stats.loc[min_conv_cluster_id, 'conversion_rate']
            print(f"\nCluster com menor taxa de conversão: {min_conv_cluster_id} ({min_conv_rate:.4f})")
        except:
            min_conv_cluster_id = None
            print("\nNão foi possível identificar o cluster com menor taxa de conversão.")
    else:
        min_conv_cluster_id = None
        print("\nNão foi possível identificar clusters com estatísticas válidas.")
    
    return cluster_stats, min_conv_cluster_id

def train_cluster_models(X_train, y_train, cluster_labels_train, 
                         experiment_id, gmm_dir, max_depth=None, n_estimators=100, random_state=42):
    """
    Treina modelos específicos para cada cluster.
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
    
    # Treinar modelos para cada cluster
    with safe_mlflow_run(experiment_id=experiment_id, 
                         run_name="gmm_cluster_models") as parent_run:
        
        mlflow.log_params({
            "algorithm": "GaussianMixture",
            "n_components": CONFIG['gmm_params']['n_components'],
            "covariance_type": CONFIG['gmm_params']['covariance_type'],
            "max_depth": str(max_depth),
            "n_estimators": n_estimators,
            "random_state": random_state
        })
        
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
                
                # Salvar modelo localmente
                model_path = os.path.join(gmm_dir, f"cluster_{cluster_id}_model.joblib")
                joblib.dump(model, model_path)
                
                # Adicionar à lista
                cluster_models.append(model_info)
                
                # Registrar metadados
                mlflow.log_params({
                    f"cluster_{cluster_id}_samples": len(X_train_cluster),
                    f"cluster_{cluster_id}_conversion_rate": float(conversion_rate),
                    f"cluster_{cluster_id}_class_weight": str(class_weight)
                })
                
                # Registrar modelo no MLflow
                mlflow.sklearn.log_model(model, f"gmm_cluster_{cluster_id}_model")
                
            except Exception as e:
                print(f"  Erro ao treinar modelo para cluster {cluster_id}: {e}")
    
    return cluster_models

def evaluate_ensemble(X_val, y_val, cluster_labels_val, cluster_models, 
                     experiment_id, gmm_dir):
    """
    Avalia o ensemble de modelos usando threshold global.
    """
    # Criar dicionário cluster -> modelo
    cluster_model_map = {model["cluster_id"]: model for model in cluster_models}
    
    # Arrays para previsões
    y_pred_proba = np.zeros_like(y_val, dtype=float)
    
    # Fazer previsões por cluster
    for cluster_id, model_info in cluster_model_map.items():
        # Selecionar amostras
        cluster_mask = (cluster_labels_val == cluster_id)
        
        if not any(cluster_mask):
            continue
            
        # Obter modelo e features
        model = model_info["model"]
        features = model_info["features"]
        
        # Extrair features relevantes
        X_cluster = X_val[features][cluster_mask].astype(float)
        
        if len(X_cluster) > 0:
            # Prever probabilidades
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
    
    # Registrar métricas
    with safe_mlflow_run(experiment_id=experiment_id, run_name="gmm_evaluation"):
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": best_threshold
        })
        
        # Salvar matriz de confusão no MLflow como JSON
        mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")
    
    # Imprimir resultados
    print(f"\nResultados do Ensemble com GMM:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Threshold: {best_threshold:.4f}")
    print("\nMatrix de Confusão:")
    print(cm)
    
    # Salvar resultados localmente
    results = {
        "algorithm": "GaussianMixture",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(best_threshold),
        "confusion_matrix": cm.tolist()
    }
    
    with open(os.path.join(gmm_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def run_gmm_clustering():
    """Função principal para executar GMM."""
    print("Iniciando clustering com GMM otimizado...")
    
    # Criar estrutura de diretórios
    dirs = create_directory_structure()
    gmm_dir = dirs['gmm_dir']
    
    # Limpar runs MLflow ativos
    active_run = mlflow.active_run()
    if active_run:
        print(f"Encerrando run MLflow ativo: {active_run.info.run_id}")
        mlflow.end_run()
    
    # Configurar MLflow
    experiment_id = setup_mlflow_tracking(
        tracking_dir=CONFIG['mlflow_dir'],
        experiment_name=CONFIG['experiment_name'],
        clean_previous=False
    )
    
    # Carregar dados
    data = load_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Preparar features numéricas
    X_train_numeric, X_val_numeric, numeric_cols = prepare_numeric_features(X_train, X_val)
    
    # Aplicar PCA
    X_train_pca, X_val_pca, pca_model, scaler = apply_pca(X_train_numeric, X_val_numeric)
    
    # Criar GMM com parâmetros otimizados
    print(f"\nCriando modelo GMM com parâmetros otimizados:")
    print(f"  n_components: {CONFIG['gmm_params']['n_components']}")
    print(f"  covariance_type: {CONFIG['gmm_params']['covariance_type']}")
    
    gmm = GaussianMixture(
        n_components=CONFIG['gmm_params']['n_components'],
        covariance_type=CONFIG['gmm_params']['covariance_type'],
        random_state=CONFIG['model_params']['random_state'],
        max_iter=200,
        n_init=5
    )
    
    # Treinar GMM
    print("\nTreinando modelo GMM...")
    start_time = time.time()
    gmm.fit(X_train_pca)
    training_time = time.time() - start_time
    print(f"  Tempo de treinamento: {training_time:.2f} segundos")
    
    # Obter clusters
    print("\nAtribuindo clusters aos dados...")
    train_labels = gmm.predict(X_train_pca)
    val_labels = gmm.predict(X_val_pca)
    
    # Analisar clusters
    train_stats, _ = analyze_clusters(train_labels, y_train, prefix="treino")
    val_stats, _ = analyze_clusters(val_labels, y_val, prefix="validação")
    
    # Treinar modelos por cluster
    print("\nTreinando modelos específicos por cluster...")
    cluster_models = train_cluster_models(
        X_train, y_train, train_labels, experiment_id, gmm_dir,
        max_depth=CONFIG['model_params']['max_depth'],
        n_estimators=CONFIG['model_params']['n_estimators']
    )
    
    # Avaliar ensemble
    print("\nAvaliando ensemble de modelos...")
    if cluster_models:
        results = evaluate_ensemble(X_val, y_val, val_labels, cluster_models, experiment_id, gmm_dir)
        
        # Criar pipeline para o GMM_Wrapper
        pipeline = {
            'pca_model': pca_model,
            'gmm_model': gmm,
            'scaler_model': scaler,
            'cluster_models': {model["cluster_id"]: {"model": model["model"], "threshold": model["threshold"]} 
                              for model in cluster_models},
            'n_clusters': CONFIG['gmm_params']['n_components'],
            'threshold': results['threshold']
        }
        
        # Criar instância do GMM_Wrapper
        gmm_wrapper = GMM_Wrapper(pipeline)
        
        # Salvar o wrapper usando joblib
        wrapper_path = os.path.join(gmm_dir, 'gmm_wrapper.joblib')
        joblib.dump(gmm_wrapper, wrapper_path)
        print(f"GMM_Wrapper salvo em: {wrapper_path}")
        
        # Salvar componentes individuais para compatibilidade
        joblib.dump(gmm, os.path.join(gmm_dir, 'gmm_model.joblib'))
        joblib.dump(pca_model, os.path.join(gmm_dir, 'pca_model.joblib'))
        joblib.dump(scaler, os.path.join(gmm_dir, 'scaler_model.joblib'))
        
        # Salvar estatísticas dos clusters
        train_stats.to_json(os.path.join(gmm_dir, 'train_cluster_stats.json'))
        val_stats.to_json(os.path.join(gmm_dir, 'val_cluster_stats.json'))
    else:
        print("Nenhum modelo de cluster criado. Não é possível avaliar o ensemble.")
        results = None
    
    # Baseline para comparação
    baseline_precision = 0.94
    baseline_recall = 0.27
    baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall)
    baseline_f1 = round(baseline_f1, 4)
    
    # Comparar com baseline
    if results:
        print("\nComparação com o Baseline:")
        print(f"  Baseline: Precision={baseline_precision:.4f}, Recall={baseline_recall:.4f}, F1={baseline_f1:.4f}")
        print(f"  GMM:      Precision={results['precision']:.4f}, Recall={results['recall']:.4f}, F1={results['f1']:.4f}")
        
        if results['f1'] > baseline_f1:
            print("  GMM tem desempenho superior ao baseline!")
        else:
            print("  GMM tem desempenho inferior ao baseline.")
    
    # Salvar informações de configuração
    with open(os.path.join(gmm_dir, 'experiment_config.json'), 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'n_samples_train': len(X_train),
            'n_samples_val': len(X_val),
            'n_features': len(numeric_cols),
            'n_pca_components': X_train_pca.shape[1],
            'conversion_rate_train': float(y_train.mean()),
            'conversion_rate_val': float(y_val.mean()),
            'gmm_params': CONFIG['gmm_params'],
            'model_params': CONFIG['model_params'],
            'training_time': training_time,
            'baseline': {
                'precision': baseline_precision,
                'recall': baseline_recall,
                'f1': baseline_f1
            }
        }, f, indent=2)
    
    print("\nExperimento GMM concluído com sucesso!")
    print(f"Resultados salvos em: {gmm_dir}")
    print(f"Logs MLflow em: {CONFIG['mlflow_dir']}")
    
    return results

if __name__ == "__main__":
    # Limpar runs MLflow ativos
    active_run = mlflow.active_run()
    if active_run:
        print(f"Encerrando run MLflow ativo: {active_run.info.run_id}")
        mlflow.end_run()
    
    # Executar GMM
    results = run_gmm_clustering()