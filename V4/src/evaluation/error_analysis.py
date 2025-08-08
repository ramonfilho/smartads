"""
Error analysis module for model evaluation.

This module provides functions to analyze model errors, identify
patterns in false positives and false negatives, and generate
visualizations to guide feature engineering efforts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import re
from sklearn.metrics import (confusion_matrix, precision_recall_curve, f1_score,
                            precision_score, recall_score, roc_auc_score, 
                            average_precision_score)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_model_for_analysis(model_uri):
    """
    Load a trained model from MLflow for error analysis.
    
    Args:
        model_uri: MLflow model URI
        
    Returns:
        Loaded model
    """
    if "random_forest" in model_uri:
        model = mlflow.sklearn.load_model(model_uri)
        model_type = "random_forest"
    elif "lightgbm" in model_uri:
        model = mlflow.lightgbm.load_model(model_uri)
        model_type = "lightgbm"
    elif "xgboost" in model_uri:
        model = mlflow.xgboost.load_model(model_uri)
        model_type = "xgboost"
    else:
        raise ValueError(f"Unsupported model type in URI: {model_uri}")
    
    return model, model_type

def get_prediction_errors(model, X, y, threshold=0.5):
    """
    Get prediction errors (false positives and false negatives).
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        threshold: Classification threshold
        
    Returns:
        DataFrame with predictions and error types
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Combine into a DataFrame
    results_df = pd.DataFrame({
        'actual': y,
        'predicted': y_pred,
        'probability': y_pred_proba
    })
    
    # Identify error types
    results_df['error_type'] = 'correct'
    results_df.loc[(results_df['actual'] == 1) & (results_df['predicted'] == 0), 'error_type'] = 'false_negative'
    results_df.loc[(results_df['actual'] == 0) & (results_df['predicted'] == 1), 'error_type'] = 'false_positive'
    
    # Add original index for reference
    results_df['original_index'] = X.index
    
    return results_df, y_pred_proba, y_pred

def get_latest_model_id(experiment_name, model_type="random_forest"):
    """
    Obtém o ID do modelo mais recente para um determinado tipo no MLflow.
    
    Args:
        experiment_name: Nome do experimento no MLflow
        model_type: Tipo de modelo (random_forest, lightgbm, xgboost)
        
    Returns:
        run_id do modelo mais recente, ou None se não encontrado
    """
    import mlflow
    from mlflow.entities import ViewType
    import datetime
    
    # Configurar URI do tracking
    mlflow_dir = os.path.join(os.path.expanduser("~"), "desktop/smart_ads/models/mlflow")
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    # Obter ID do experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experimento '{experiment_name}' não encontrado.")
        return None
    
    experiment_id = experiment.experiment_id
    
    # Buscar runs do experimento
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.model_type = '{model_type}'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["start_time DESC"]  # Ordenar pelo mais recente
    )
    
    # Verificar se encontrou algum modelo
    if len(runs) == 0:
        print(f"Nenhum modelo do tipo '{model_type}' encontrado no experimento '{experiment_name}'.")
        return None
    
    # Pegar o ID do modelo mais recente
    latest_run_id = runs.iloc[0]['run_id']
    start_time = runs.iloc[0]['start_time']
    
    # Converter timestamp para data/hora legível
    dt = datetime.datetime.fromtimestamp(start_time/1000.0)
    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Modelo mais recente ({model_type}) encontrado: {latest_run_id}")
    print(f"Data de criação: {formatted_time}")
    
    return latest_run_id

def analyze_errors_by_feature(df, feature_results, continuous_features=None, categorical_features=None):
    """
    Analyze errors by feature to identify patterns.
    
    Args:
        df: Original DataFrame with features
        feature_results: DataFrame with prediction results and error types
        continuous_features: List of continuous features to analyze
        categorical_features: List of categorical features to analyze
        
    Returns:
        Dictionary with analysis results
    """
    if continuous_features is None:
        continuous_features = df.select_dtypes(include=['float', 'int']).columns.tolist()
    
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Combine the dataframes
    analysis_df = pd.merge(
        df, 
        feature_results[['original_index', 'error_type', 'probability']], 
        left_index=True, 
        right_on='original_index'
    )
    
    results = {
        'continuous_features': {},
        'categorical_features': {}
    }
    
    # Analyze continuous features
    for feature in continuous_features:
        if feature not in df.columns:
            continue
        
        feature_stats = {}
        
        # Calculate statistics by error type
        for error_type in ['correct', 'false_positive', 'false_negative']:
            subset = analysis_df[analysis_df['error_type'] == error_type]
            
            if not subset.empty:
                feature_stats[error_type] = {
                    'mean': subset[feature].mean(),
                    'median': subset[feature].median(),
                    'std': subset[feature].std(),
                    'min': subset[feature].min(),
                    'max': subset[feature].max(),
                    'count': subset[feature].count()
                }
        
        results['continuous_features'][feature] = feature_stats
    
    # Analyze categorical features
    for feature in categorical_features:
        if feature not in df.columns:
            continue
        
        feature_stats = {}
        
        # Calculate distribution by error type
        for error_type in ['correct', 'false_positive', 'false_negative']:
            subset = analysis_df[analysis_df['error_type'] == error_type]
            
            if not subset.empty:
                value_counts = subset[feature].value_counts(normalize=True)
                feature_stats[error_type] = value_counts.to_dict()
                
        results['categorical_features'][feature] = feature_stats
    
    return results, analysis_df

def cluster_errors(analysis_df, features, n_clusters=3):
    """
    Cluster errors to find patterns.
    
    Args:
        analysis_df: DataFrame with features and error types
        features: List of features to use for clustering
        n_clusters: Number of clusters
        
    Returns:
        DataFrame with cluster assignments
    """
    # Filter features that exist in the dataframe
    valid_features = [f for f in features if f in analysis_df.columns]
    
    # Handle missing values
    X_cluster = analysis_df[valid_features].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Apply clustering separately to different error types
    for error_type in ['false_positive', 'false_negative']:
        mask = analysis_df['error_type'] == error_type
        
        if mask.sum() > n_clusters:  # Ensure we have enough samples
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled[mask])
            
            # Add cluster labels to dataframe
            cluster_col = f'cluster_{error_type}'
            analysis_df.loc[mask, cluster_col] = cluster_labels
            
            # Calculate feature importance for each cluster
            cluster_centers = kmeans.cluster_centers_
            
            for i in range(n_clusters):
                cluster_mask = (analysis_df['error_type'] == error_type) & (analysis_df[cluster_col] == i)
                if cluster_mask.sum() > 0:
                    print(f"\nCluster {i} para {error_type} ({cluster_mask.sum()} amostras):")
                    # Get top features by absolute deviation from overall mean
                    overall_means = X_cluster.loc[mask].mean()
                    cluster_means = X_cluster.loc[cluster_mask].mean()
                    deviations = (cluster_means - overall_means).abs().sort_values(ascending=False)
                    
                    # Print top distinctive features
                    top_n = min(10, len(deviations))
                    for j, (feat, dev) in enumerate(deviations.iloc[:top_n].items()):
                        overall = overall_means[feat]
                        cluster = cluster_means[feat]
                        change = ((cluster - overall) / overall * 100) if overall != 0 else float('inf')
                        direction = "maior" if cluster > overall else "menor"
                        print(f"  {j+1}. {feat}: {cluster:.4f} ({direction} que média geral de {overall:.4f}, {abs(change):.1f}% diferença)")
    
    return analysis_df

def plot_error_distributions(analysis_df, features, output_dir="eda_results/error_analysis"):
    """
    Plot distributions of features by error type.
    
    Args:
        analysis_df: DataFrame with features and error types
        features: List of features to plot
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter features that exist in the dataframe
    valid_features = [f for f in features if f in analysis_df.columns]
    
    for feature in valid_features:
        try:
            plt.figure(figsize=(12, 6))
            
            # Check if feature is numeric
            if pd.api.types.is_numeric_dtype(analysis_df[feature]):
                # Create KDE plot for numeric features
                sns.kdeplot(
                    data=analysis_df, x=feature, hue="error_type",
                    fill=True, common_norm=False, alpha=0.5,
                    linewidth=2
                )
                plt.title(f'Distribuição de {feature} por tipo de erro')
                plt.xlabel(feature)
                plt.ylabel('Densidade')
            else:
                # Create count plot for categorical features
                ax = sns.countplot(
                    data=analysis_df, x=feature, hue="error_type"
                )
                plt.title(f'Contagem de {feature} por tipo de erro')
                plt.xlabel(feature)
                plt.ylabel('Contagem')
                plt.xticks(rotation=45)
                
                # Adjust layout for categorical features
                plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f'error_dist_{feature}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Erro ao plotar feature {feature}: {e}")
    
    # Plot probability distributions by error type
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=analysis_df, x="probability", hue="error_type",
        fill=True, common_norm=False, alpha=0.5,
        linewidth=2
    )
    plt.title('Distribuição de probabilidades por tipo de erro')
    plt.xlabel('Probabilidade prevista')
    plt.ylabel('Densidade')
    plt.savefig(os.path.join(output_dir, 'error_probability_dist.png'))
    plt.close()

def analyze_feature_interactions(analysis_df, features, output_dir="eda_results/error_analysis"):
    """
    Analyze and plot feature interactions for errors.
    
    Args:
        analysis_df: DataFrame with features and error types
        features: List of features to analyze
        output_dir: Directory to save plots
        
    Returns:
        DataFrame with interaction importance scores
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select only numeric features for correlations
    numeric_features = [f for f in features if f in analysis_df.columns and pd.api.types.is_numeric_dtype(analysis_df[f])]
    
    interaction_scores = []
    
    # Analyze correlations separately for each error type
    for error_type in ['correct', 'false_positive', 'false_negative']:
        subset = analysis_df[analysis_df['error_type'] == error_type]
        
        if len(subset) > 5:  # Ensure we have enough samples
            # Calculate correlation matrix
            corr_matrix = subset[numeric_features].corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(
                corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=False
            )
            plt.title(f'Correlação entre features para {error_type}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'correlation_{error_type}.png'))
            plt.close()
            
            # Find strongest correlations
            for i in range(len(numeric_features)):
                for j in range(i+1, len(numeric_features)):
                    feat1 = numeric_features[i]
                    feat2 = numeric_features[j]
                    corr = corr_matrix.iloc[i, j]
                    
                    if abs(corr) > 0.3:  # Only consider moderate to strong correlations
                        interaction_scores.append({
                            'error_type': error_type,
                            'feature1': feat1,
                            'feature2': feat2,
                            'correlation': corr,
                            'abs_correlation': abs(corr)
                        })
    
    # Convert to DataFrame and sort
    if interaction_scores:
        interactions_df = pd.DataFrame(interaction_scores)
        interactions_df = interactions_df.sort_values('abs_correlation', ascending=False)
        
        # Save to CSV
        interactions_df.to_csv(os.path.join(output_dir, 'feature_interactions.csv'), index=False)
        
        # Plot top interactions
        top_interactions = interactions_df.head(15)
        for _, row in top_interactions.iterrows():
            try:
                feat1 = row['feature1']
                feat2 = row['feature2']
                error_type = row['error_type']
                
                plt.figure(figsize=(10, 8))
                subset = analysis_df[analysis_df['error_type'] == error_type]
                
                plt.scatter(subset[feat1], subset[feat2], alpha=0.6)
                plt.title(f'Interação entre {feat1} e {feat2} para {error_type}\nCorrelação: {row["correlation"]:.2f}')
                plt.xlabel(feat1)
                plt.ylabel(feat2)
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(output_dir, f'interaction_{error_type}_{feat1}_{feat2}.png'))
                plt.close()
            except Exception as e:
                print(f"Erro ao plotar interação {feat1} x {feat2}: {e}")
        
        return interactions_df
    
    return pd.DataFrame()

def generate_error_analysis_summary(analysis_results, analysis_df, output_dir="eda_results/error_analysis"):
    """
    Generate a text summary of error analysis findings.
    
    Args:
        analysis_results: Dictionary with analysis results
        analysis_df: DataFrame with features and error types
        output_dir: Directory to save summary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Count error types
    error_counts = analysis_df['error_type'].value_counts()
    
    with open(os.path.join(output_dir, 'error_analysis_summary.md'), 'w') as f:
        f.write("# Análise de Erros do Modelo\n\n")
        
        # Overall statistics
        f.write("## Estatísticas Gerais\n\n")
        f.write(f"- Total de previsões analisadas: {len(analysis_df)}\n")
        f.write(f"- Previsões corretas: {error_counts.get('correct', 0)} ({error_counts.get('correct', 0)/len(analysis_df)*100:.2f}%)\n")
        f.write(f"- Falsos negativos: {error_counts.get('false_negative', 0)} ({error_counts.get('false_negative', 0)/len(analysis_df)*100:.2f}%)\n")
        f.write(f"- Falsos positivos: {error_counts.get('false_positive', 0)} ({error_counts.get('false_positive', 0)/len(analysis_df)*100:.2f}%)\n\n")
        
        # Continuous features
        f.write("## Análise de Features Contínuas\n\n")
        for feature, stats in analysis_results['continuous_features'].items():
            f.write(f"### {feature}\n\n")
            f.write("| Estatística | Corretos | Falsos Positivos | Falsos Negativos |\n")
            f.write("| --- | --- | --- | --- |\n")
            
            metrics = ['mean', 'median', 'std', 'min', 'max']
            metric_names = {'mean': 'Média', 'median': 'Mediana', 'std': 'Desvio Padrão', 'min': 'Mínimo', 'max': 'Máximo'}
            
            for metric in metrics:
                values = []
                for error_type in ['correct', 'false_positive', 'false_negative']:
                    if error_type in stats and metric in stats[error_type]:
                        values.append(f"{stats[error_type][metric]:.4f}")
                    else:
                        values.append("N/A")
                
                f.write(f"| {metric_names[metric]} | {values[0]} | {values[1]} | {values[2]} |\n")
            f.write("\n")
        
        # Top categorical features
        f.write("## Principais Diferenças em Features Categóricas\n\n")
        categorical_insights = []
        
        for feature, stats in analysis_results['categorical_features'].items():
            if 'correct' in stats and 'false_negative' in stats:
                for category, correct_pct in stats['correct'].items():
                    if category in stats['false_negative']:
                        fn_pct = stats['false_negative'][category]
                        diff = fn_pct - correct_pct
                        
                        if abs(diff) > 0.1:  # Only report substantial differences
                            categorical_insights.append({
                                'feature': feature,
                                'category': category,
                                'correct_pct': correct_pct * 100,
                                'fn_pct': fn_pct * 100,
                                'diff': diff * 100,
                                'type': 'false_negative'
                            })
            
            if 'correct' in stats and 'false_positive' in stats:
                for category, correct_pct in stats['correct'].items():
                    if category in stats['false_positive']:
                        fp_pct = stats['false_positive'][category]
                        diff = fp_pct - correct_pct
                        
                        if abs(diff) > 0.1:  # Only report substantial differences
                            categorical_insights.append({
                                'feature': feature,
                                'category': category,
                                'correct_pct': correct_pct * 100,
                                'fp_pct': fp_pct * 100,
                                'diff': diff * 100,
                                'type': 'false_positive'
                            })
        
        # Sort by absolute difference
        categorical_insights = sorted(categorical_insights, key=lambda x: abs(x['diff']), reverse=True)
        
        # Write top insights
        for insight in categorical_insights[:15]:  # Top 15
            error_type = "falsos negativos" if insight['type'] == 'false_negative' else "falsos positivos"
            direction = "mais comum" if insight['diff'] > 0 else "menos comum"
            f.write(f"- **{insight['feature']} = {insight['category']}** é {direction} em {error_type} ")
            
            if insight['type'] == 'false_negative':
                f.write(f"({insight['fn_pct']:.1f}% vs {insight['correct_pct']:.1f}% em corretos, {abs(insight['diff']):.1f}% de diferença)\n")
            else:
                f.write(f"({insight['fp_pct']:.1f}% vs {insight['correct_pct']:.1f}% em corretos, {abs(insight['diff']):.1f}% de diferença)\n")
        
        f.write("\n## Recomendações para Feature Engineering\n\n")
        
        # Add recommendations based on findings
        if categorical_insights:
            f.write("### Features Categóricas\n\n")
            f.write("1. Considere criar features de interação para as seguintes combinações:\n")
            for insight in categorical_insights[:5]:
                f.write(f"   - `{insight['feature']}_{insight['category']}` (indicador binário)\n")
        
        # Additional sections
        f.write("\n### Potenciais Novas Features\n\n")
        f.write("1. **Features de interação**: Multiplique ou combine features que mostraram correlações diferentes por tipo de erro\n")
        f.write("2. **Features compostas**: Crie razões ou diferenças entre features numéricas relevantes\n")
        f.write("3. **Transformações não-lineares**: Aplique funções logarítmicas ou polinomiais para capturar relações não-lineares\n")
        f.write("4. **Agregações temporais**: Se houver dados temporais, considere médias móveis ou tendências\n")
        f.write("5. **Engenharia específica para clusters de erros**: Desenvolva features específicas para os padrões identificados nos clusters de erros\n")

def analyze_high_value_false_negatives(val_df, y_val, y_pred_val, y_pred_prob_val, target_col, results_dir="error_analysis"):
    """
    Analisa falsos negativos de alto valor (conversões não detectadas).
    
    Args:
        val_df: DataFrame de validação
        y_val: Valores reais do target
        y_pred_val: Valores previstos
        y_pred_prob_val: Probabilidades previstas
        target_col: Nome da coluna target
        results_dir: Diretório para salvar resultados
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Análise de Falsos Negativos (conversões não detectadas)
    false_negatives = (y_val == 1) & (y_pred_val == 0)
    fn_indices = np.where(false_negatives)[0]
    fn_data = val_df.iloc[fn_indices].copy()
    fn_data['predicted_prob'] = y_pred_prob_val[fn_indices]

    # Ordenar por probabilidade (do maior para o menor - casos limítrofes)
    fn_data = fn_data.sort_values('predicted_prob', ascending=False)

    # Salvar para análise detalhada
    fn_data.to_csv(f"{results_dir}/false_negatives.csv", index=False)
    print(f"\nNúmero de falsos negativos: {len(fn_data)}")
    print(f"Lista detalhada salva em: {results_dir}/false_negatives.csv")
    
    return fn_data

def analyze_feature_distributions_by_error_type(val_df, y_val, y_pred_val, y_pred_prob_val, target_col, results_dir="error_analysis"):
    """
    Analisa a distribuição de features por tipo de erro.
    
    Args:
        val_df: DataFrame de validação
        y_val: Valores reais do target
        y_pred_val: Valores previstos
        y_pred_prob_val: Probabilidades previstas
        target_col: Nome da coluna target
        results_dir: Diretório para salvar resultados
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Identificar diferentes tipos de predição
    false_negatives = (y_val == 1) & (y_pred_val == 0)
    false_positives = (y_val == 0) & (y_pred_val == 1)
    true_positives = (y_val == 1) & (y_pred_val == 1)
    true_negatives = (y_val == 0) & (y_pred_val == 0)
    
    # Obter índices
    fn_indices = np.where(false_negatives)[0]
    fp_indices = np.where(false_positives)[0]
    tp_indices = np.where(true_positives)[0]
    
    # Criar dataframes para cada tipo
    fn_data = val_df.iloc[fn_indices].copy()
    fp_data = val_df.iloc[fp_indices].copy()
    tp_data = val_df.iloc[tp_indices].copy()
    
    fn_data['predicted_prob'] = y_pred_prob_val[fn_indices]
    fp_data['predicted_prob'] = y_pred_prob_val[fp_indices]
    
    # Salvar falsos positivos
    fp_data = fp_data.sort_values('predicted_prob', ascending=False)
    fp_data.to_csv(f"{results_dir}/false_positives.csv", index=False)
    print(f"\nNúmero de falsos positivos: {len(fp_data)}")
    
    # Tentar identificar colunas numéricas e categóricas
    numeric_cols = []
    categorical_cols = []
    
    for col in val_df.columns:
        if col not in [target_col, 'predicted_prob']:
            try:
                if pd.api.types.is_numeric_dtype(val_df[col]) and val_df[col].nunique() > 10:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except:
                categorical_cols.append(col)
    
    # Limitar o número de colunas para análise
    numeric_cols = numeric_cols[:20]
    categorical_cols = categorical_cols[:20]
    
    # Análise de características numéricas
    if numeric_cols and len(fn_data) > 0:
        print("\nCaracterísticas numéricas distintivas em falsos negativos:")
        numeric_analysis = []
        
        for col in numeric_cols:
            try:
                # Calcular estatísticas
                fn_mean = fn_data[col].mean()
                pop_mean = val_df[col].mean()
                
                # Calcular diferença percentual
                diff_pct = ((fn_mean - pop_mean) / pop_mean * 100) if pop_mean != 0 else 0
                
                numeric_analysis.append({
                    'Feature': col,
                    'FN_Mean': fn_mean,
                    'Population_Mean': pop_mean,
                    'Diff_Pct': diff_pct
                })
            except Exception as e:
                pass  # Ignorar erros silenciosamente
        
        # Ordenar por diferença percentual
        numeric_df = pd.DataFrame(numeric_analysis)
        if not numeric_df.empty:
            numeric_df = numeric_df.sort_values('Diff_Pct', ascending=False)
            
            # Mostrar top 5 características numéricas distintivas
            for i, row in numeric_df.head(5).iterrows():
                direction = "maior" if row['Diff_Pct'] > 0 else "menor"
                print(f"  - {row['Feature']}: {abs(row['Diff_Pct']):.2f}% {direction} que a média geral")
            
            # Salvar análise completa
            numeric_df.to_csv(f"{results_dir}/fn_numeric_features.csv", index=False)
    
    # Análise de características categóricas
    if categorical_cols and len(fn_data) > 0:
        print("\nCaracterísticas categóricas distintivas em falsos negativos:")
        categorical_analysis = []
        
        for col in categorical_cols:
            try:
                # Calcular frequências
                pop_freq = val_df[col].value_counts(normalize=True).to_dict()
                fn_freq = fn_data[col].value_counts(normalize=True).to_dict()
                
                # Identificar valores com maior diferença
                for val, freq in fn_freq.items():
                    if val in pop_freq:
                        diff = freq - pop_freq[val]
                        ratio = freq / pop_freq[val] if pop_freq[val] > 0 else float('inf')
                        
                        if abs(diff) > 0.05:  # Apenas diferenças significativas
                            categorical_analysis.append({
                                'Feature': col,
                                'Value': val,
                                'FN_Freq': freq,
                                'Pop_Freq': pop_freq[val],
                                'Difference': diff,
                                'Ratio': ratio
                            })
            except Exception as e:
                pass  # Ignorar erros silenciosamente
        
        # Ordenar por maior diferença
        cat_df = pd.DataFrame(categorical_analysis)
        if not cat_df.empty:
            cat_df = cat_df.sort_values('Difference', ascending=False)
            
            # Mostrar top 5 características categóricas distintivas
            for i, row in cat_df.head(5).iterrows():
                print(f"  - {row['Feature']} = {row['Value']}: {row['FN_Freq']:.2%} (vs {row['Pop_Freq']:.2%} na população), {row['Ratio']:.2f}x mais comum")
            
            # Salvar análise completa
            cat_df.to_csv(f"{results_dir}/fn_categorical_features.csv", index=False)
    
    return {
        'fn_data': fn_data,
        'fp_data': fp_data,
        'tp_data': tp_data,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }

def analyze_segments(val_df, y_val, y_pred_val, results_dir="error_analysis"):
    """
    Analisa o desempenho do modelo por segmentos.
    
    Args:
        val_df: DataFrame de validação
        y_val: Valores reais do target
        y_pred_val: Valores previstos
        results_dir: Diretório para salvar resultados
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Identificar colunas de segmentação potencialmente interessantes
    segment_cols = []
    for pattern in ['launch', 'lançament', 'country', 'pais', 'age', 'idade']:
        cols = [col for col in val_df.columns if pattern.lower() in col.lower()]
        if cols:
            segment_cols.extend(cols[:1])  # Adicionar apenas a primeira coluna encontrada para cada padrão
    
    # Limitar o número total de segmentos
    segment_cols = segment_cols[:5]
    
    segment_analysis = {}
    
    if segment_cols:
        print("\nAnálise de falsos negativos por segmentos:")
        
        for col in segment_cols:
            if col in val_df.columns:
                try:
                    # Calcular taxa de falsos negativos por segmento
                    segment_stats = []
                    
                    # Limitar a 10 valores mais frequentes para evitar análise excessiva
                    top_values = val_df[col].value_counts().nlargest(10).index
                    
                    for value in top_values:
                        # Filtrar dados para este segmento
                        segment_mask = val_df[col] == value
                        segment_y_true = y_val[segment_mask]
                        segment_y_pred = y_pred_val[segment_mask]
                        
                        # Se houver dados suficientes
                        if sum(segment_mask) >= 20 and sum(segment_y_true) > 0:
                            # Calcular taxa de falsos negativos
                            segment_fn = ((segment_y_true == 1) & (segment_y_pred == 0)).sum()
                            segment_fn_rate = segment_fn / sum(segment_y_true)
                            
                            segment_stats.append({
                                'Segmento': value,
                                'Tamanho': sum(segment_mask),
                                'Conversões': sum(segment_y_true),
                                'Falsos_Negativos': segment_fn,
                                'Taxa_FN': segment_fn_rate
                            })
                    
                    # Ordenar por taxa de falsos negativos
                    segment_df = pd.DataFrame(segment_stats)
                    if not segment_df.empty:
                        segment_df = segment_df.sort_values('Taxa_FN', ascending=False)
                        segment_df.to_csv(f"{results_dir}/segment_analysis_{col}.csv", index=False)
                        
                        print(f"\nSegmentos com maior taxa de falsos negativos para {col}:")
                        for i, row in segment_df.head(3).iterrows():
                            print(f"  - {col}={row['Segmento']}: {row['Taxa_FN']:.2%} das conversões não detectadas ({row['Falsos_Negativos']} de {row['Conversões']})")
                        
                        segment_analysis[col] = segment_df
                except Exception as e:
                    pass  # Ignorar erros silenciosamente
    
    return segment_analysis

def compare_detected_vs_missed(tp_data, fn_data, features, model=None, results_dir="error_analysis"):
    """
    Compara o perfil de conversões detectadas vs não detectadas.
    
    Args:
        tp_data: DataFrame com verdadeiros positivos
        fn_data: DataFrame com falsos negativos
        features: Lista de features para comparar
        model: Modelo treinado (opcional, para obter importância)
        results_dir: Diretório para salvar resultados
    """
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n--- Perfil Comparativo: Conversões Detectadas vs Não Detectadas ---")
    
    # Selecionar features importantes para comparação
    top_features = []
    
    if model is not None and hasattr(model, 'feature_importances_'):
        try:
            feature_imp = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            })
            top_features = feature_imp.sort_values('Importance', ascending=False).head(10)['Feature'].tolist()
        except:
            # Se não conseguir obter importância, usar todas as features
            top_features = features[:10]
    else:
        top_features = features[:10]
    
    # Comparar médias entre TPs e FNs
    comparison = []
    for col in top_features:
        if col in tp_data.columns and col in fn_data.columns:
            try:
                tp_mean = tp_data[col].mean()
                fn_mean = fn_data[col].mean()
                diff = tp_mean - fn_mean
                diff_pct = ((tp_mean - fn_mean) / fn_mean * 100) if fn_mean != 0 else 0
                
                comparison.append({
                    'Feature': col,
                    'Detected_Mean': tp_mean,
                    'Missed_Mean': fn_mean,
                    'Difference': diff,
                    'Diff_Pct': diff_pct
                })
            except Exception as e:
                pass  # Ignorar erros silenciosamente
    
    # Mostrar diferenças mais significativas
    comp_df = pd.DataFrame(comparison)
    if not comp_df.empty:
        comp_df = comp_df.sort_values('Difference', ascending=False)
        comp_df.to_csv(f"{results_dir}/detected_vs_missed_comparison.csv", index=False)
        
        print("Principais diferenças entre conversões detectadas e não detectadas:")
        for i, row in comp_df.head(5).iterrows():
            diff_dir = "maior" if row['Difference'] > 0 else "menor"
            print(f"  - {row['Feature']}: {row['Detected_Mean']:.4f} nas detectadas vs {row['Missed_Mean']:.4f} nas não detectadas ({abs(row['Diff_Pct']):.1f}% {diff_dir})")
    
    return comp_df

def error_analysis_from_validation_data(
    train_df, val_df, model, best_threshold=0.15,
    target_col='target', results_dir="error_analysis"):
    """
    Executa uma análise completa de erros utilizando dados de validação.
    
    Args:
        train_df: DataFrame de treinamento
        val_df: DataFrame de validação
        model: Modelo treinado
        best_threshold: Threshold de classificação
        target_col: Nome da coluna target
        results_dir: Diretório para salvar resultados
        
    Returns:
        Dicionário com resultados da análise
    """
    print("Executando análise de erros detalhada...")
    
    # Criar diretório para resultados
    os.makedirs(results_dir, exist_ok=True)
    
    # Identificar colunas comuns entre os conjuntos
    common_cols = set(train_df.columns).intersection(set(val_df.columns))
    print(f"Encontradas {len(common_cols)} colunas em comum entre os conjuntos")
    
    # Usar apenas colunas em comum
    train_df = train_df[list(common_cols)]
    val_df = val_df[list(common_cols)]
    
    # Preparar features e target
    feature_cols = [col for col in val_df.columns if col != target_col]
    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].copy()
    
    # Garantir conversão para float
    for col in X_val.columns:
        if pd.api.types.is_integer_dtype(X_val[col].dtype):
            X_val[col] = X_val[col].astype(float)
    
    # Gerar previsões
    print("Gerando previsões para análise de erros...")
    y_pred_prob_val = model.predict_proba(X_val)[:, 1]
    y_pred_val = (y_pred_prob_val >= best_threshold).astype(int)
    
    # Calcular e visualizar matriz de confusão
    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não Converteu', 'Converteu'],
                yticklabels=['Não Converteu', 'Converteu'])
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/confusion_matrix.png")
    plt.close()
    
    # Calcular métricas
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\nMétricas de desempenho:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Taxa de Falsos Positivos: {fpr:.4f}")
    print(f"Falsos Negativos: {fn} de {fn+tp} conversões reais")
    print(f"Falsos Positivos: {fp} de {fp+tn} não-conversões reais")
    
    # Analisar distribuição de probabilidades
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_prob_val[y_val == 0], bins=50, alpha=0.5, color='blue', label='Não Converteu')
    plt.hist(y_pred_prob_val[y_val == 1], bins=50, alpha=0.5, color='red', label='Converteu')
    plt.axvline(x=best_threshold, color='green', linestyle='--', label=f'Threshold: {best_threshold:.4f}')
    plt.title('Distribuição de Probabilidades Previstas')
    plt.xlabel('Probabilidade')
    plt.ylabel('Frequência')
    plt.legend()
    plt.savefig(f"{results_dir}/probability_distribution.png")
    plt.close()
    
    # Análise detalhada de falsos negativos
    fn_data = analyze_high_value_false_negatives(val_df, y_val, y_pred_val, y_pred_prob_val, target_col, results_dir)
    
    # Análise de distribuições por tipo de erro
    error_distributions = analyze_feature_distributions_by_error_type(
        val_df, y_val, y_pred_val, y_pred_prob_val, target_col, results_dir
    )
    
    # Análise por segmentos
    segment_analysis = analyze_segments(val_df, y_val, y_pred_val, results_dir)
    
    # Comparar perfil de conversões detectadas vs. não detectadas
    comparison = compare_detected_vs_missed(
        error_distributions['tp_data'], 
        error_distributions['fn_data'], 
        feature_cols, 
        model, 
        results_dir
    )
    
    # Importância das features
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        feature_imp = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        })
        feature_imp = feature_imp.sort_values('Importance', ascending=False).head(30)
        sns.barplot(x='Importance', y='Feature', data=feature_imp)
        plt.title('Top 30 Features (Importância do Modelo)')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/feature_importance.png")
        plt.close()
        feature_imp.to_csv(f"{results_dir}/feature_importance.csv", index=False)
    
    # Resumo final
    print("\n=== RESUMO DA ANÁLISE DE ERROS ===")
    print(f"Total de registros analisados: {len(y_val)}")
    print(f"Conversões reais: {y_val.sum()} ({y_val.mean():.2%})")
    print(f"Conversões previstas: {y_pred_val.sum()} ({y_pred_val.mean():.2%})")
    print(f"Falsos negativos: {fn} ({fn/(fn+tp):.2%} das conversões reais)")
    print(f"Falsos positivos: {fp} ({fp/(fp+tn):.2%} das não-conversões reais)")
    print(f"Precisão: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Todos os resultados foram salvos em: {results_dir}/")
    
    return {
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'fn': fn,
            'fp': fp,
            'tp': tp,
            'tn': tn
        },
        'error_distributions': error_distributions,
        'segment_analysis': segment_analysis,
        'comparison': comparison,
        'y_pred_val': y_pred_val,
        'y_pred_prob_val': y_pred_prob_val
    }

def run_error_analysis(
    model_uri,
    data_path,
    target_col="target",
    output_dir="eda_results/error_analysis",
    top_n_features=30
):
    """
    Run the complete error analysis pipeline.
    
    Args:
        model_uri: MLflow model URI
        data_path: Path to the dataset to analyze
        target_col: Name of the target column
        output_dir: Directory to save output files
        top_n_features: Number of top features to analyze
        
    Returns:
        Dictionary with analysis results
    """
    print(f"Iniciando análise de erros para o modelo: {model_uri}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, model_type = load_model_for_analysis(model_uri)
    print(f"Modelo {model_type} carregado com sucesso")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Get model threshold from MLflow if available
    client = mlflow.tracking.MlflowClient()
    run_id = model_uri.split("/")[1]
    try:
        run = client.get_run(run_id)
        threshold = float(run.data.metrics.get("threshold", 0.5))
        print(f"Usando threshold de {threshold} do MLflow")
    except:
        threshold = 0.5
        print(f"Threshold não encontrado, usando padrão: {threshold}")
    
    # Get prediction errors
    results_df, y_pred_proba, y_pred = get_prediction_errors(model, X, y, threshold)
    print(f"Análise de erros: {results_df['error_type'].value_counts().to_dict()}")
    
    # Save results DataFrame
    results_df.to_csv(os.path.join(output_dir, "prediction_results.csv"), index=False)
    
    # Identify important features
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        top_features = importance['feature'].head(top_n_features).tolist()
    else:
        top_features = X.columns.tolist()[:top_n_features]
    
    # Analyze errors by feature
    analysis_results, analysis_df = analyze_errors_by_feature(
        df, results_df, 
        continuous_features=top_features, 
        categorical_features=[col for col in df.columns if col in top_features and df[col].dtype == 'object']
    )
    
    # Save analysis DataFrame
    analysis_df.to_csv(os.path.join(output_dir, "error_analysis_data.csv"), index=False)
    
    # Cluster errors
    analysis_df = cluster_errors(analysis_df, top_features)
    analysis_df.to_csv(os.path.join(output_dir, "error_clusters.csv"), index=False)
    
    # Plot error distributions
    print("Gerando visualizações de distribuições de erros...")
    plot_error_distributions(analysis_df, top_features, output_dir)
    
    # Analyze feature interactions
    print("Analisando interações entre features...")
    interactions_df = analyze_feature_interactions(analysis_df, top_features, output_dir)
    
    # Generate summary
    print("Gerando resumo da análise de erros...")
    generate_error_analysis_summary(analysis_results, analysis_df, output_dir)
    
    # Perform detailed analysis using your original code
    print("Executando análise detalhada com o código original...")
    # This will reuse the loaded model and dataset with a different approach
    detailed_results = error_analysis_from_validation_data(
        df.copy(), df.copy(), model, 
        best_threshold=threshold,
        target_col=target_col, 
        results_dir=os.path.join(output_dir, "detailed")
    )
    
    print(f"Análise de erros concluída. Resultados salvos em {output_dir}")
    
    return {
        'analysis_results': analysis_results,
        'analysis_df': analysis_df,
        'results_df': results_df,
        'interactions_df': interactions_df,
        'top_features': top_features,
        'detailed_results': detailed_results
    }