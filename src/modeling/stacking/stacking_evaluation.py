# Avaliação e análise dos modelos
"""
Módulo para avaliação de modelos de stacking.

Este módulo contém funções para avaliar e comparar o desempenho de
modelos de stacking, incluindo análise de erros e geração de visualizações.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, f1_score,
    precision_score, recall_score, roc_auc_score, 
    average_precision_score, classification_report
)
import mlflow

def evaluate_model(model, X, y, threshold=None, prefix="model"):
    """
    Avalia um modelo com métricas padrão.
    
    Args:
        model: Modelo a ser avaliado (deve implementar predict_proba)
        X: Features para predição
        y: Target real
        threshold: Threshold para classificação binária (None para usar default do modelo)
        prefix: Prefixo para nomes de métricas
        
    Returns:
        Dicionário com métricas
    """
    # Obter probabilidades
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Determinar threshold
    if threshold is None and hasattr(model, 'threshold'):
        threshold = model.threshold
    elif threshold is None:
        threshold = 0.5
    
    # Converter para classes
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calcular métricas
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    pr_auc = average_precision_score(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    # Calcular positivos previstos
    positive_count = y_pred.sum()
    positive_pct = positive_count / len(y_pred) * 100
    
    # Relatório de classificação
    report = classification_report(y, y_pred, output_dict=True)
    
    metrics = {
        f'{prefix}_precision': precision,
        f'{prefix}_recall': recall,
        f'{prefix}_f1': f1,
        f'{prefix}_pr_auc': pr_auc,
        f'{prefix}_roc_auc': roc_auc,
        f'{prefix}_threshold': threshold,
        f'{prefix}_positive_count': positive_count,
        f'{prefix}_positive_pct': positive_pct,
        f'{prefix}_predictions': y_pred,
        f'{prefix}_probabilities': y_pred_proba,
        f'{prefix}_report': report
    }
    
    return metrics

def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Encontra o threshold ótimo para maximizar uma métrica específica.
    
    Args:
        y_true: Target real
        y_pred_proba: Probabilidades previstas
        metric: Métrica para otimizar ('f1', 'precision', 'recall')
        
    Returns:
        Threshold ótimo
    """
    best_value = 0
    best_threshold = 0.5
    
    # Testar diferentes thresholds
    for threshold in np.arange(0.01, 0.5, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            value = f1_score(y_true, y_pred)
        elif metric == 'precision':
            value = precision_score(y_true, y_pred)
        elif metric == 'recall':
            value = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Métrica não suportada: {metric}")
        
        if value > best_value:
            best_value = value
            best_threshold = threshold
    
    return best_threshold, best_value

def plot_confusion_matrix(y_true, y_pred, title, output_dir, filename):
    """
    Plota e salva matriz de confusão.
    
    Args:
        y_true: Target real
        y_pred: Previsões
        title: Título do gráfico
        output_dir: Diretório para salvar o gráfico
        filename: Nome do arquivo
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotar
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
              xticklabels=['Não Converteu', 'Converteu'],
              yticklabels=['Não Converteu', 'Converteu'])
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title(title)
    plt.tight_layout()
    
    # Salvar
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_probability_distribution(y_true, y_pred_proba, threshold, title, output_dir, filename):
    """
    Plota e salva distribuição de probabilidades por classe.
    
    Args:
        y_true: Target real
        y_pred_proba: Probabilidades previstas
        threshold: Threshold para classificação
        title: Título do gráfico
        output_dir: Diretório para salvar o gráfico
        filename: Nome do arquivo
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Plotar
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_proba[y_true == 0], bins=50, alpha=0.5, 
                color='blue', label='Não Converteu')
    sns.histplot(y_pred_proba[y_true == 1], bins=50, alpha=0.5, 
                color='red', label='Converteu')
    plt.axvline(x=threshold, color='green', linestyle='--', 
               label=f'Threshold: {threshold:.2f}')
    plt.title(title)
    plt.xlabel('Probabilidade prevista')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, title, output_dir, filename):
    """
    Plota e salva curva precision-recall.
    
    Args:
        y_true: Target real
        y_pred_proba: Probabilidades previstas
        title: Título do gráfico
        output_dir: Diretório para salvar o gráfico
        filename: Nome do arquivo
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Calcular precision e recall para diferentes thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calcular F1 para cada threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Encontrar threshold ótimo para F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    # Plotar
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.scatter([recall[best_idx]], [precision[best_idx]], 
               color='red', marker='o', 
               label=f'Melhor F1: {best_f1:.3f} (threshold={best_threshold:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_threshold_analysis(y_true, y_pred_proba, title, output_dir, filename):
    """
    Plota e salva análise de threshold.
    
    Args:
        y_true: Target real
        y_pred_proba: Probabilidades previstas
        title: Título do gráfico
        output_dir: Diretório para salvar o gráfico
        filename: Nome do arquivo
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Testar diferentes thresholds
    thresholds = np.arange(0.01, 0.5, 0.01)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
    
    # Encontrar threshold ótimo para F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    # Plotar
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.axvline(x=best_threshold, color='red', linestyle='--', 
               label=f'Melhor threshold: {best_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Valor')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    return {
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'best_threshold': best_threshold,
        'best_f1': f1_scores[best_idx]
    }

def compare_models(models_dict, X, y, output_dir):
    """
    Compara diferentes modelos e gera visualizações.
    
    Args:
        models_dict: Dicionário com modelos a serem comparados
                    {'model_name': model_instance, ...}
        X: Features para predição
        y: Target real
        output_dir: Diretório para salvar resultados
        
    Returns:
        DataFrame com resultados comparativos
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    predictions = {}
    
    # Para cada modelo
    for model_name, model in models_dict.items():
        print(f"Avaliando modelo: {model_name}")
        
        # Obter probabilidades
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Encontrar threshold ótimo
        threshold, best_f1 = find_optimal_threshold(y, y_pred_proba)
        
        # Obter previsões
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calcular métricas
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        pr_auc = average_precision_score(y, y_pred_proba)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        # Calcular positivos previstos
        positive_count = y_pred.sum()
        positive_pct = positive_count / len(y_pred) * 100
        
        # Guardar resultados
        results.append({
            'model': model_name,
            'precision': precision,
            'recall': recall,
            'f1': best_f1,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'threshold': threshold,
            'positive_count': positive_count,
            'positive_pct': positive_pct
        })
        
        predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Gerar visualizações
        plot_confusion_matrix(
            y, y_pred, 
            f"Matriz de Confusão - {model_name}",
            output_dir, f"confusion_matrix_{model_name}.png"
        )
        
        plot_probability_distribution(
            y, y_pred_proba, threshold,
            f"Distribuição de Probabilidades - {model_name}",
            output_dir, f"prob_dist_{model_name}.png"
        )
        
        plot_precision_recall_curve(
            y, y_pred_proba,
            f"Curva Precision-Recall - {model_name}",
            output_dir, f"pr_curve_{model_name}.png"
        )
        
        plot_threshold_analysis(
            y, y_pred_proba,
            f"Análise de Threshold - {model_name}",
            output_dir, f"threshold_analysis_{model_name}.png"
        )
    
    # Converter para DataFrame
    results_df = pd.DataFrame(results)
    
    # Salvar resultados
    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Plotar comparação
    plt.figure(figsize=(12, 6))
    metrics = ['precision', 'recall', 'f1', 'pr_auc']
    results_plot = results_df.melt(
        id_vars=['model'], 
        value_vars=metrics,
        var_name='metric', 
        value_name='value'
    )
    
    sns.barplot(data=results_plot, x='model', y='value', hue='metric')
    plt.title('Comparação de Métricas entre Modelos')
    plt.ylabel('Valor')
    plt.xlabel('Modelo')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()
    
    return results_df, predictions

def analyze_disagreements(models_dict, X, y, output_dir):
    """
    Analisa desacordos entre modelos para entender comportamentos.
    
    Args:
        models_dict: Dicionário com modelos a serem comparados
        X: Features para predição
        y: Target real
        output_dir: Diretório para salvar resultados
        
    Returns:
        DataFrame com análise de desacordos
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Obter previsões para cada modelo
    model_predictions = {}
    for model_name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Usar threshold ótimo
        threshold, _ = find_optimal_threshold(y, y_pred_proba)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        model_predictions[model_name] = {
            'pred': y_pred,
            'proba': y_pred_proba,
            'threshold': threshold
        }
    
    # Criar DataFrame com todas as previsões
    predictions_df = pd.DataFrame({
        'true': y
    })
    
    for model_name, preds in model_predictions.items():
        predictions_df[f'{model_name}_pred'] = preds['pred']
        predictions_df[f'{model_name}_proba'] = preds['proba']
    
    # Identificar exemplos onde modelos discordam
    model_names = list(models_dict.keys())
    disagreements = {}
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            
            # Exemplos onde modelos discordam
            disagree_mask = predictions_df[f'{model1}_pred'] != predictions_df[f'{model2}_pred']
            disagree_df = predictions_df[disagree_mask].copy()
            
            # Adicionar coluna de acertos
            disagree_df[f'{model1}_correct'] = disagree_df[f'{model1}_pred'] == disagree_df['true']
            disagree_df[f'{model2}_correct'] = disagree_df[f'{model2}_pred'] == disagree_df['true']
            
            # Estatísticas de acertos
            model1_correct = disagree_df[f'{model1}_correct'].mean()
            model2_correct = disagree_df[f'{model2}_correct'].mean()
            
            disagreements[f'{model1}_vs_{model2}'] = {
                'count': len(disagree_df),
                'pct_of_total': len(disagree_df) / len(predictions_df) * 100,
                f'{model1}_correct_pct': model1_correct * 100,
                f'{model2}_correct_pct': model2_correct * 100,
                'dataframe': disagree_df
            }
    
    # Criar resumo de desacordos
    disagreement_summary = []
    
    for comparison, data in disagreements.items():
        disagreement_summary.append({
            'comparison': comparison,
            'count': data['count'],
            'pct_of_total': data['pct_of_total'],
            **{k: v for k, v in data.items() if k.endswith('_pct') and k != 'pct_of_total'}
        })
    
    disagreement_summary_df = pd.DataFrame(disagreement_summary)
    
    # Salvar resultados
    disagreement_summary_df.to_csv(os.path.join(output_dir, "disagreement_summary.csv"), index=False)
    predictions_df.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False)
    
    for comparison, data in disagreements.items():
        data['dataframe'].to_csv(os.path.join(output_dir, f"disagreement_{comparison}.csv"), index=False)
    
    # Plotar resumo de desacordos
    plt.figure(figsize=(12, 6))
    disagreement_plot = disagreement_summary_df.copy()
    disagreement_plot = disagreement_plot.sort_values('count', ascending=False)
    
    sns.barplot(data=disagreement_plot, x='comparison', y='count')
    plt.title('Número de Desacordos entre Modelos')
    plt.ylabel('Contagem')
    plt.xlabel('Comparação')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "disagreement_counts.png"))
    plt.close()
    
    return disagreement_summary_df, disagreements

def analyze_specialist_contributions(ensemble_model, X, y, output_dir):
    """
    Analisa contribuições dos modelos especialistas no ensemble.
    
    Args:
        ensemble_model: Modelo de ensemble (deve ter acesso aos especialistas)
        X: Features para predição
        y: Target real
        output_dir: Diretório para salvar resultados
        
    Returns:
        DataFrame com análise de contribuições
    """
    # Verificar se o modelo possui os especialistas necessários
    if not hasattr(ensemble_model, 'specialist_models') and not hasattr(ensemble_model, 'trained_specialists'):
        raise ValueError("O modelo ensemble não possui atributo 'specialist_models' ou 'trained_specialists'")
    
    # Obter lista de especialistas
    specialists = getattr(ensemble_model, 'trained_specialists', 
                        getattr(ensemble_model, 'specialist_models', []))
    
    if not specialists:
        raise ValueError("Nenhum modelo especialista encontrado no ensemble")
    
    # Obter previsões para cada especialista
    specialist_predictions = {}
    for i, specialist in enumerate(specialists):
        name = getattr(specialist, 'name', f"specialist_{i}")
        feature_type = getattr(specialist, 'feature_type', "unknown")
        
        # Obter features específicas para este especialista
        X_specialist = X.get(feature_type, X)
        
        # Obter previsões
        try:
            y_pred_proba = specialist.predict_proba(X_specialist)[:, 1]
            threshold, _ = find_optimal_threshold(y, y_pred_proba)
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            specialist_predictions[name] = {
                'pred': y_pred,
                'proba': y_pred_proba,
                'threshold': threshold,
                'feature_type': feature_type
            }
        except Exception as e:
            print(f"Erro ao obter previsões para {name}: {e}")
    
    # Obter previsões do ensemble
    try:
        y_ensemble_proba = ensemble_model.predict_proba(X)[:, 1]
        threshold, _ = find_optimal_threshold(y, y_ensemble_proba)
        y_ensemble_pred = (y_ensemble_proba >= threshold).astype(int)
    except Exception as e:
        print(f"Erro ao obter previsões do ensemble: {e}")
        raise
    
    # Criar DataFrame com todas as previsões
    predictions_df = pd.DataFrame({
        'true': y,
        'ensemble_pred': y_ensemble_pred,
        'ensemble_proba': y_ensemble_proba
    })
    
    for name, preds in specialist_predictions.items():
        predictions_df[f'{name}_pred'] = preds['pred']
        predictions_df[f'{name}_proba'] = preds['proba']
    
    # Calcular métricas para cada modelo
    metrics = []
    
    # Ensemble
    ensemble_metrics = {
        'model': 'ensemble',
        'precision': precision_score(y, y_ensemble_pred),
        'recall': recall_score(y, y_ensemble_pred),
        'f1': f1_score(y, y_ensemble_pred),
        'pr_auc': average_precision_score(y, y_ensemble_proba),
        'threshold': threshold
    }
    metrics.append(ensemble_metrics)
    
    # Especialistas
    for name, preds in specialist_predictions.items():
        specialist_metrics = {
            'model': name,
            'feature_type': preds['feature_type'],
            'precision': precision_score(y, preds['pred']),
            'recall': recall_score(y, preds['pred']),
            'f1': f1_score(y, preds['pred']),
            'pr_auc': average_precision_score(y, preds['proba']),
            'threshold': preds['threshold']
        }
        metrics.append(specialist_metrics)
    
    metrics_df = pd.DataFrame(metrics)
    
    # Calcular correlações entre previsões
    proba_columns = [col for col in predictions_df.columns if col.endswith('_proba')]
    corr_matrix = predictions_df[proba_columns].corr()
    
    # Analisar onde o ensemble acerta e os especialistas erram
    predictions_df['ensemble_correct'] = predictions_df['ensemble_pred'] == predictions_df['true']
    
    for name in specialist_predictions.keys():
        predictions_df[f'{name}_correct'] = predictions_df[f'{name}_pred'] == predictions_df['true']
    
    # Identificar casos interessantes
    ensemble_correct_all_wrong = predictions_df['ensemble_correct']
    all_correct_ensemble_wrong = ~predictions_df['ensemble_correct']
    
    for name in specialist_predictions.keys():
        ensemble_correct_all_wrong &= ~predictions_df[f'{name}_correct']
        all_correct_ensemble_wrong &= predictions_df[f'{name}_correct']
    
    interesting_cases = {
        'ensemble_correct_all_wrong': predictions_df[ensemble_correct_all_wrong],
        'all_correct_ensemble_wrong': predictions_df[all_correct_ensemble_wrong]
    }
    
    # Salvar resultados
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_df.to_csv(os.path.join(output_dir, "specialist_metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(output_dir, "specialist_predictions.csv"), index=False)
    corr_matrix.to_csv(os.path.join(output_dir, "prediction_correlations.csv"))
    
    for case_name, case_df in interesting_cases.items():
        case_df.to_csv(os.path.join(output_dir, f"{case_name}.csv"), index=False)
    
    # Plotar comparação de métricas
    plt.figure(figsize=(12, 6))
    metrics_plot = metrics_df.melt(
        id_vars=['model'], 
        value_vars=['precision', 'recall', 'f1', 'pr_auc'],
        var_name='metric', 
        value_name='value'
    )
    
    sns.barplot(data=metrics_plot, x='model', y='value', hue='metric')
    plt.title('Comparação de Métricas entre Especialistas e Ensemble')
    plt.ylabel('Valor')
    plt.xlabel('Modelo')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "specialist_comparison.png"))
    plt.close()
    
    # Plotar mapa de calor de correlações
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', center=0)
    plt.title('Correlação entre Previsões dos Modelos')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_correlations.png"))
    plt.close()
    
    return {
        'metrics': metrics_df,
        'correlations': corr_matrix,
        'predictions': predictions_df,
        'interesting_cases': interesting_cases
    }

def evaluate_and_log_stacking_ensemble(ensemble, X, y, run_id=None, output_dir=None):
    """
    Avalia um ensemble de stacking e registra resultados no MLflow.
    
    Args:
        ensemble: Modelo de ensemble
        X: Features para predição
        y: Target real
        run_id: ID do run MLflow (opcional)
        output_dir: Diretório para salvar resultados
        
    Returns:
        Dicionário com resultados da avaliação
    """
    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp()
    
    # Avaliar modelo
    y_pred_proba = ensemble.predict_proba(X)[:, 1]
    
    # Encontrar threshold ótimo
    threshold_results = plot_threshold_analysis(
        y, y_pred_proba, "Análise de Threshold - Ensemble",
        output_dir, "threshold_analysis.png"
    )
    
    best_threshold = threshold_results['best_threshold']
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Calcular métricas
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    pr_auc = average_precision_score(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    # Gerar visualizações
    plot_confusion_matrix(
        y, y_pred, "Matriz de Confusão - Ensemble",
        output_dir, "confusion_matrix.png"
    )
    
    plot_probability_distribution(
        y, y_pred_proba, best_threshold, "Distribuição de Probabilidades - Ensemble",
        output_dir, "probability_distribution.png"
    )
    
    plot_precision_recall_curve(
        y, y_pred_proba, "Curva Precision-Recall - Ensemble",
        output_dir, "pr_curve.png"
    )
    
    # Analisar contribuições dos especialistas
    if hasattr(ensemble, 'specialist_models') or hasattr(ensemble, 'trained_specialists'):
        try:
            specialist_results = analyze_specialist_contributions(
                ensemble, X, y, os.path.join(output_dir, "specialists")
            )
        except Exception as e:
            print(f"Erro ao analisar contribuições dos especialistas: {e}")
            specialist_results = None
    else:
        specialist_results = None
    
     # Registrar no MLflow
    if run_id is not None:
        with mlflow.start_run(run_id=run_id, nested=True):  # Adicionar nested=True aqui
            # Registrar métricas
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("threshold", best_threshold)
            mlflow.log_metric("positive_count", y_pred.sum())
            mlflow.log_metric("positive_pct", y_pred.mean() * 100)
            
            # Registrar artefatos
            mlflow.log_artifacts(output_dir)
    
    # Preparar resultados
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'threshold': best_threshold,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'threshold_results': threshold_results,
        'output_dir': output_dir
    }
    
    if specialist_results:
        results['specialist_results'] = specialist_results
    
    return results