import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

# Configurar caminhos
BASE_DIR = "/Users/ramonmoreira/desktop/smart_ads/"
DATA_DIR = "/Users/ramonmoreira/desktop/smart_ads/data/02_3_processed_text_code6/"
MODELS_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/artifacts/k_means/"
VALIDATION_PATH = os.path.join(DATA_DIR, "validation.csv")

# Função para carregar o pipeline completo
def load_cluster_pipeline():
    """
    Carrega todos os componentes do pipeline de clustering.
    """
    print("Carregando componentes do pipeline...")
    
    # Carregar modelos principais
    pca_model = joblib.load(os.path.join(MODELS_DIR, "pca_model.joblib"))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    
    # Carregar informações do cluster
    with open(os.path.join(MODELS_DIR, "cluster_info.json"), 'r') as f:
        cluster_info = json.load(f)
    
    approach = cluster_info['approach']
    use_baseline = cluster_info.get('use_baseline', False)
    n_clusters = cluster_info['n_clusters']
    
    print(f"Abordagem: {approach}")
    print(f"Usar baseline: {use_baseline}")
    print(f"Número de clusters: {n_clusters}")
    
    # Carregar modelos específicos de cada cluster
    cluster_models = {}
    
    if not use_baseline:
        for cluster_id in range(n_clusters):
            model_path = os.path.join(MODELS_DIR, f"cluster_{cluster_id}_model.joblib")
            info_path = os.path.join(MODELS_DIR, f"cluster_{cluster_id}_info.json")
            
            try:
                model = joblib.load(model_path)
                
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                
                # Definir threshold de acordo com a abordagem
                if approach == 'cluster_thresholds':
                    threshold = model_info['threshold']
                else:
                    threshold = cluster_info.get('global_threshold', 0.5)
                
                cluster_models[cluster_id] = {
                    'model': model,
                    'threshold': threshold,
                    'features': [], # Será preenchido depois
                    'info': model_info
                }
                
                print(f"  Carregado modelo para cluster {cluster_id}, threshold: {threshold:.4f}")
            except FileNotFoundError as e:
                print(f"  AVISO: {e}")
                print(f"  Modelo para cluster {cluster_id} não encontrado. Este cluster será ignorado.")
    else:
        print("Pipeline configurado para usar o modelo baseline em vez dos modelos de cluster.")
    
    return {
        'pca_model': pca_model,
        'kmeans_model': kmeans_model,
        'cluster_info': cluster_info,
        'cluster_models': cluster_models,
        'approach': approach,
        'use_baseline': use_baseline
    }

# Função para identificar features numéricas (mesmas do treino)
def identify_numeric_features(df):
    """
    Identifica features numéricas no DataFrame.
    """
    numeric_cols = []
    for col in df.columns:
        try:
            df[col].astype(float)
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    return numeric_cols

# Função para preparar dados para os modelos
def prepare_data(df, target_col='target'):
    """
    Prepara os dados para uso com os modelos.
    """
    # Separar features e target
    y = df[target_col] if target_col in df.columns else None
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    
    # Identificar features numéricas
    numeric_cols = identify_numeric_features(X)
    X_numeric = X[numeric_cols].copy().astype(float)
    
    # Substituir NaN por 0
    X_numeric.fillna(0, inplace=True)
    
    return X, y, X_numeric, numeric_cols

# Função para aplicar PCA e clustering
def transform_data(X_numeric, pipeline):
    """
    Aplica normalização, PCA e clustering aos dados.
    """
    from sklearn.preprocessing import StandardScaler
    
    # Verificar o número de features esperado pelo modelo PCA
    pca_model = pipeline['pca_model']
    expected_n_features = pca_model.n_features_in_
    actual_n_features = X_numeric.shape[1]
    
    print(f"Número de features no conjunto de validação: {actual_n_features}")
    print(f"Número de features esperado pelo PCA: {expected_n_features}")
    
    # Se houver discrepância no número de features
    if actual_n_features != expected_n_features:
        print("AVISO: Discrepância no número de features!")
        
        # Se temos features extras, precisamos removê-las
        if actual_n_features > expected_n_features:
            print("Removendo features extras...")
            
            # Tentar obter os nomes das features que o PCA conhece
            if hasattr(pca_model, 'feature_names_in_'):
                # Usar apenas as features que o PCA conhece
                known_features = pca_model.feature_names_in_
                features_to_use = [f for f in known_features if f in X_numeric.columns]
                
                # Se ainda faltam features, estamos em problemas
                if len(features_to_use) < expected_n_features:
                    print(f"ERRO: Não foi possível identificar todas as features esperadas pelo PCA")
                    # Usar as primeiras expected_n_features colunas como fallback
                    X_numeric = X_numeric.iloc[:, :expected_n_features]
                else:
                    # Usar apenas as features conhecidas
                    X_numeric = X_numeric[features_to_use]
            else:
                # Se não temos os nomes das features, usar as primeiras expected_n_features colunas
                print("PCA não tem nomes de features, removendo as últimas colunas...")
                X_numeric = X_numeric.iloc[:, :expected_n_features]
        
        # Se faltam features, precisamos adicionar colunas de zeros
        elif actual_n_features < expected_n_features:
            print("Adicionando colunas de zeros para features ausentes...")
            missing_count = expected_n_features - actual_n_features
            # Adicionar colunas extras com zeros
            for i in range(missing_count):
                X_numeric[f'missing_feature_{i}'] = 0.0
    
    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Verificar dimensões novamente
    print(f"Dimensões após ajuste: {X_scaled.shape}")
    
    # Aplicar PCA
    X_pca = pipeline['pca_model'].transform(X_scaled)
    
    # Aplicar clustering
    cluster_labels = pipeline['kmeans_model'].predict(X_pca)
    
    return X_pca, cluster_labels

# Função para fazer previsões usando a abordagem de thresholds por cluster
def predict_with_cluster_thresholds(X, numeric_cols, cluster_labels, pipeline):
    """
    Faz previsões usando modelos específicos para cada cluster.
    """
    cluster_models = pipeline['cluster_models']
    n_samples = len(X)
    
    # Arrays para previsões
    y_pred_proba = np.zeros(n_samples, dtype=float)
    y_pred = np.zeros(n_samples, dtype=int)
    
    # Para cada cluster, fazer previsões
    for cluster_id, model_info in cluster_models.items():
        # Selecionar amostras deste cluster
        cluster_mask = (cluster_labels == cluster_id)
        
        if not any(cluster_mask):
            continue
        
        # Obter modelo e threshold
        model = model_info['model']
        threshold = model_info['threshold']
        
        # Detectar quais features o modelo espera
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            model_info['features'] = expected_features
        else:
            expected_features = [col for col in X.columns if col in numeric_cols]
            model_info['features'] = expected_features
        
        # Selecionar apenas as features necessárias
        features_to_use = [f for f in expected_features if f in X.columns]
        
        # Verificar se faltam features
        if len(features_to_use) < len(expected_features):
            missing = set(expected_features) - set(features_to_use)
            print(f"  AVISO: Faltam {len(missing)} features para o modelo do cluster {cluster_id}")
            print(f"  Exemplos de features ausentes: {list(missing)[:3]}")
            
            # Criar colunas ausentes com zeros
            for col in missing:
                X[col] = 0.0
                features_to_use.append(col)
        
        # Extrair dados do cluster com as features corretas
        X_cluster = X.loc[cluster_mask, features_to_use].astype(float)
        
        if len(X_cluster) > 0:
            # Fazer previsões
            proba = model.predict_proba(X_cluster)[:, 1]
            
            # Armazenar resultados
            y_pred_proba[cluster_mask] = proba
            y_pred[cluster_mask] = (proba >= threshold).astype(int)
    
    return y_pred, y_pred_proba

# Função para avaliar o modelo
def evaluate_model(y_true, y_pred, y_pred_proba):
    """
    Avalia o modelo e retorna métricas.
    """
    if y_true is None:
        print("AVISO: Sem dados de target, não é possível avaliar.")
        return None
    
    # Calcular métricas
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calcular métricas baseadas em probabilidades
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'positive_rate': y_pred.mean(),
        'true_positive_rate': y_true.mean()
    }

# Função para analisar distribuições por cluster
def analyze_clusters(cluster_labels, y_true, y_pred):
    """
    Analisa a distribuição e desempenho por cluster.
    """
    df_analysis = pd.DataFrame({
        'cluster': cluster_labels,
        'true': y_true,
        'pred': y_pred
    })
    
    # Estatísticas por cluster
    cluster_stats = []
    
    for cluster_id in np.unique(cluster_labels):
        cluster_df = df_analysis[df_analysis['cluster'] == cluster_id]
        
        # Calcular estatísticas
        samples = len(cluster_df)
        conversion_rate = cluster_df['true'].mean()
        prediction_rate = cluster_df['pred'].mean()
        
        # Calcular métricas se houver dados suficientes
        if samples > 0 and cluster_df['true'].sum() > 0:
            prec = precision_score(cluster_df['true'], cluster_df['pred'])
            rec = recall_score(cluster_df['true'], cluster_df['pred'])
            f1 = f1_score(cluster_df['true'], cluster_df['pred'])
        else:
            prec, rec, f1 = 0, 0, 0
        
        # Construir matriz de confusão
        tp = ((cluster_df['true'] == 1) & (cluster_df['pred'] == 1)).sum()
        fp = ((cluster_df['true'] == 0) & (cluster_df['pred'] == 1)).sum()
        tn = ((cluster_df['true'] == 0) & (cluster_df['pred'] == 0)).sum()
        fn = ((cluster_df['true'] == 1) & (cluster_df['pred'] == 0)).sum()
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'samples': samples,
            'samples_pct': samples / len(df_analysis) * 100,
            'conversion_rate': conversion_rate,
            'prediction_rate': prediction_rate,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        })
    
    return pd.DataFrame(cluster_stats)

# Função principal
def main():
    # Carregar o pipeline
    pipeline = load_cluster_pipeline()
    
    # Carregar dados de validação
    print(f"\nCarregando dados de validação de: {VALIDATION_PATH}")
    val_df = pd.read_csv(VALIDATION_PATH)
    print(f"Dados carregados: {val_df.shape}")
    
    # Preparar dados
    X_val, y_val, X_val_numeric, numeric_cols = prepare_data(val_df)
    print(f"Features numéricas: {len(numeric_cols)}")
    
    # Aplicar transformações
    print("\nAplicando PCA e clustering...")
    X_val_pca, cluster_labels = transform_data(X_val_numeric, pipeline)
    print(f"Dimensões após PCA: {X_val_pca.shape}")
    
    # Distribuição de clusters
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    print("\nDistribuição de clusters:")
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        print(f"  Cluster {cluster_id}: {count} amostras ({count/len(X_val)*100:.1f}%)")
    
    # Fazer previsões
    print("\nRealizando previsões...")
    y_pred, y_pred_proba = predict_with_cluster_thresholds(X_val, numeric_cols, cluster_labels, pipeline)
    
    # Avaliar modelo
    print("\nAvaliando modelo...")
    metrics = evaluate_model(y_val, y_pred, y_pred_proba)
    
    if metrics:
        print("\n=== Métricas Globais ===")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Taxa de previsões positivas: {metrics['positive_rate']:.4f}")
        print(f"Taxa real de positivos: {metrics['true_positive_rate']:.4f}")
        
        print("\nMatriz de Confusão:")
        cm = metrics['confusion_matrix']
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Analisar clusters
        print("\nAnalisando desempenho por cluster...")
        cluster_analysis = analyze_clusters(cluster_labels, y_val, y_pred)
        print("\n=== Análise por Cluster ===")
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(cluster_analysis.sort_values('cluster_id'))
        
        # Salvar resultados
        output_dir = os.path.join(BASE_DIR, "reports", "cluster_validation")
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar métricas globais
        with open(os.path.join(output_dir, "global_metrics.json"), 'w') as f:
            # Criar uma cópia do dicionário para não modificar o original
            metrics_json = {}
            for k, v in metrics.items():
                if k == 'confusion_matrix':
                    # Converter matriz de confusão para lista de listas
                    metrics_json[k] = v.tolist() if isinstance(v, np.ndarray) else v
                elif isinstance(v, np.ndarray):
                    # Converter outros arrays para float ou lista dependendo do tamanho
                    if v.size == 1:
                        metrics_json[k] = float(v)
                    else:
                        metrics_json[k] = v.tolist()
                else:
                    # Valores não-array
                    metrics_json[k] = v
            
            json.dump(metrics_json, f, indent=2)
        
        # Salvar análise por cluster
        cluster_analysis.to_csv(os.path.join(output_dir, "cluster_analysis.csv"), index=False)
        
        # Criar DataFrame de previsões
        results_df = val_df.copy()
        results_df['cluster'] = cluster_labels
        results_df['prediction'] = y_pred
        results_df['probability'] = y_pred_proba
        results_df['correct'] = (results_df['target'] == results_df['prediction'])
        
        # Salvar resultados detalhados
        results_df.to_csv(os.path.join(output_dir, "validation_predictions.csv"), index=False)
        
        print(f"\nResultados salvos em: {output_dir}")

if __name__ == "__main__":
    main()