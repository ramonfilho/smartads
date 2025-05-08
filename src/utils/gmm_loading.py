import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

# Configurar caminhos absolutos corretos
BASE_DIR = "/Users/ramonmoreira/desktop/smart_ads"
DATA_DIR = "/Users/ramonmoreira/desktop/smart_ads/data/02_3_processed_text_code6"
MODELS_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/artifacts/gmm_optimized"
VALIDATION_PATH = os.path.join(DATA_DIR, "validation.csv")

# Função para carregar o pipeline completo
def load_gmm_pipeline():
    """
    Carrega todos os componentes do pipeline de GMM.
    """
    print("Carregando componentes do pipeline GMM...")
    
    # Carregar modelos principais
    pca_model = joblib.load(os.path.join(MODELS_DIR, "pca_model.joblib"))
    gmm_model = joblib.load(os.path.join(MODELS_DIR, "gmm_model.joblib"))
    scaler_model = joblib.load(os.path.join(MODELS_DIR, "scaler_model.joblib"))
    
    # Carregar informações de avaliação e configuração
    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), 'r') as f:
        eval_results = json.load(f)
    
    with open(os.path.join(MODELS_DIR, "experiment_config.json"), 'r') as f:
        config = json.load(f)
    
    # Extrair informações relevantes
    n_clusters = config.get('gmm_params', {}).get('n_components', 3)
    threshold = eval_results.get('threshold', 0.15)
    
    print(f"Modelo GMM com {n_clusters} componentes")
    print(f"Threshold global: {threshold:.4f}")
    
    # Carregar estatísticas de clusters, se existirem
    train_cluster_stats = None
    val_cluster_stats = None
    try:
        train_cluster_stats = pd.read_json(os.path.join(MODELS_DIR, "train_cluster_stats.json"))
        val_cluster_stats = pd.read_json(os.path.join(MODELS_DIR, "val_cluster_stats.json"))
    except:
        print("Informações estatísticas dos clusters não encontradas")
    
    # Carregar modelos específicos de cada cluster
    cluster_models = {}
    
    for cluster_id in range(n_clusters):
        model_path = os.path.join(MODELS_DIR, f"cluster_{cluster_id}_model.joblib")
        
        try:
            model = joblib.load(model_path)
            
            # Definir threshold para este modelo
            cluster_models[cluster_id] = {
                'model': model,
                'threshold': threshold,
                'features': [],  # Será preenchido depois
                'info': {'threshold': threshold}
            }
            
            print(f"  Carregado modelo para cluster {cluster_id}, threshold: {threshold:.4f}")
        except FileNotFoundError as e:
            print(f"  AVISO: {e}")
            print(f"  Modelo para cluster {cluster_id} não encontrado. Este cluster será ignorado.")
    
    return {
        'pca_model': pca_model,
        'gmm_model': gmm_model,
        'scaler_model': scaler_model,
        'threshold': threshold,
        'config': config,
        'cluster_models': cluster_models,
        'n_clusters': n_clusters,
        'train_stats': train_cluster_stats,
        'val_stats': val_cluster_stats
    }

# Função para identificar features numéricas
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

# Função para aplicar PCA e GMM clustering
def transform_data_gmm(X_numeric, pipeline):
    """
    Aplica normalização usando o scaler salvo, PCA e GMM clustering aos dados.
    """
    # Usar o scaler pré-treinado em vez de criar um novo
    scaler = pipeline['scaler_model']
    
    # Verificar e ajustar nomes de features - passo crucial para garantir compatibilidade
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = scaler.feature_names_in_
        
        # Identificar features em X_numeric que não estão no scaler
        unseen_features = [col for col in X_numeric.columns if col not in scaler_features]
        if unseen_features:
            print(f"Removendo {len(unseen_features)} features não vistas durante treinamento:")
            print(f"  Exemplos: {unseen_features[:3]}")
            X_numeric = X_numeric.drop(columns=unseen_features)
        
        # Identificar features que faltam em X_numeric mas estão no scaler
        missing_features = [col for col in scaler_features if col not in X_numeric.columns]
        if missing_features:
            print(f"Adicionando {len(missing_features)} features ausentes vistas durante treinamento:")
            print(f"  Exemplos: {missing_features[:3]}")
            for col in missing_features:
                X_numeric[col] = 0.0
        
        # Garantir a ordem correta das colunas
        X_numeric = X_numeric[scaler_features]
    
    # Verificar o número de features
    expected_n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
    actual_n_features = X_numeric.shape[1]
    
    print(f"Número de features no conjunto de validação após ajustes: {actual_n_features}")
    if expected_n_features:
        print(f"Número de features esperado pelo Scaler: {expected_n_features}")
    
    # Aplicar scaler pré-treinado
    X_scaled = scaler.transform(X_numeric)
    
    # Verificar dimensões
    print(f"Dimensões após scaling: {X_scaled.shape}")
    
    # Aplicar PCA
    pca_model = pipeline['pca_model']
    X_pca = pca_model.transform(X_scaled)
    
    # Aplicar GMM para obter cluster labels
    gmm_model = pipeline['gmm_model']
    cluster_labels = gmm_model.predict(X_pca)
    
    # Também obter probabilidades de pertencer a cada cluster
    cluster_probs = gmm_model.predict_proba(X_pca)
    
    print(f"Dimensões após PCA: {X_pca.shape}")
    
    return X_pca, cluster_labels, cluster_probs

# Função para fazer previsões usando modelos específicos por cluster
def predict_with_gmm_cluster_models(X, numeric_cols, cluster_labels, pipeline):
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
        # Converter cluster_id para inteiro se for string
        cluster_id_int = int(cluster_id) if isinstance(cluster_id, str) else cluster_id
        
        # Selecionar amostras deste cluster
        cluster_mask = (cluster_labels == cluster_id_int)
        
        if not any(cluster_mask):
            print(f"  AVISO: Nenhuma amostra atribuída ao cluster {cluster_id_int}")
            continue
        
        # Obter modelo e threshold
        model = model_info['model']
        threshold = model_info['threshold']
        
        # Detectar quais features o modelo espera
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            model_info['features'] = expected_features
            
            # Verificar e ajustar nomes de features para este modelo específico
            unseen_features = [col for col in X.columns if col not in expected_features]
            missing_features = [col for col in expected_features if col not in X.columns]
            
            # Criar um DataFrame temporário com as features corretas para este modelo
            X_temp = X.copy()
            
            # Remover features não vistas durante treinamento
            if unseen_features:
                X_temp = X_temp.drop(columns=[col for col in unseen_features if col in X_temp.columns])
            
            # Adicionar features ausentes com zeros
            for col in missing_features:
                X_temp[col] = 0.0
            
            # Garantir a ordem correta
            X_cluster = X_temp.loc[cluster_mask, expected_features].astype(float)
        else:
            # Para modelos que não armazenam nomes de features, assumir features numéricas
            X_cluster = X.loc[cluster_mask, numeric_cols].astype(float)
        
        if len(X_cluster) > 0:
            # Fazer previsões
            try:
                proba = model.predict_proba(X_cluster)[:, 1]
                
                # Armazenar resultados
                y_pred_proba[cluster_mask] = proba
                y_pred[cluster_mask] = (proba >= threshold).astype(int)
            except Exception as e:
                print(f"  ERRO ao fazer previsões para o cluster {cluster_id_int}: {e}")
                print(f"  Causado por: {type(e).__name__}")
                
                # Em caso de erro, atribuir probabilidade 0 para este cluster
                y_pred_proba[cluster_mask] = 0.0
                y_pred[cluster_mask] = 0
    
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
            try:
                prec = precision_score(cluster_df['true'], cluster_df['pred'])
                rec = recall_score(cluster_df['true'], cluster_df['pred'])
                f1 = f1_score(cluster_df['true'], cluster_df['pred'])
            except:
                prec, rec, f1 = 0, 0, 0
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
    # Carregar o pipeline GMM
    pipeline = load_gmm_pipeline()
    
    # Carregar dados de validação
    print(f"\nCarregando dados de validação de: {VALIDATION_PATH}")
    val_df = pd.read_csv(VALIDATION_PATH)
    print(f"Dados carregados: {val_df.shape}")
    
    # Preparar dados
    X_val, y_val, X_val_numeric, numeric_cols = prepare_data(val_df)
    print(f"Features numéricas: {len(numeric_cols)}")
    
    # Aplicar transformações
    print("\nAplicando normalização, PCA e GMM clustering...")
    X_val_pca, cluster_labels, cluster_probs = transform_data_gmm(X_val_numeric, pipeline)
    
    # Distribuição de clusters
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    print("\nDistribuição de clusters:")
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        cluster_mask = (cluster_labels == cluster_id)
        conversion_rate = y_val[cluster_mask].mean() if len(y_val[cluster_mask]) > 0 else 0
        print(f"  Cluster {cluster_id}: {count} amostras ({count/len(X_val)*100:.1f}%), "
              f"Taxa de conversão: {conversion_rate:.4f}")
    
    # Fazer previsões
    print("\nRealizando previsões com modelos específicos por cluster...")
    y_pred, y_pred_proba = predict_with_gmm_cluster_models(X_val, numeric_cols, cluster_labels, pipeline)
    
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
        
        # Comparação com resultados registrados durante treinamento
        if pipeline.get('config') and 'baseline' in pipeline['config']:
            baseline = pipeline['config']['baseline']
            print("\n=== Comparação com Baseline ===")
            print(f"  Baseline: Precision={baseline.get('precision', 0):.4f}, "
                  f"Recall={baseline.get('recall', 0):.4f}, "
                  f"F1={baseline.get('f1', 0):.4f}")
            print(f"  GMM Valid: Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, "
                  f"F1={metrics['f1']:.4f}")
        
        # Salvar resultados
        output_dir = os.path.join(BASE_DIR, "reports", "gmm_validation")
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
        
        # Adicionar probabilidades de pertencer a cada cluster
        for i in range(pipeline['n_clusters']):
            results_df[f'cluster_prob_{i}'] = cluster_probs[:, i]
        
        # Salvar resultados detalhados
        results_df.to_csv(os.path.join(output_dir, "validation_predictions.csv"), index=False)
        
        print(f"\nResultados salvos em: {output_dir}")

if __name__ == "__main__":
    main()