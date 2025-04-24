#!/usr/bin/env python
"""
Script para testar diferentes técnicas de balanceamento e cost-sensitive learning
para melhorar o recall do modelo mantendo precisão aceitável.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import mlflow

# 1. Configuração de caminhos e estrutura do projeto
PROJ_ROOT = os.path.expanduser("/Users/ramonmoreira/desktop/smart_ads")
INPUT_DIR = os.path.join(PROJ_ROOT, "data/03_feature_selection_text_code6")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "data/08_balanced_data")
REPORT_DIR = os.path.join(PROJ_ROOT, "reports/balancing_experiments")
MODEL_DIR = os.path.join(PROJ_ROOT, "models")
MLFLOW_DIR = os.path.join(MODEL_DIR, "mlflow")

# Nomes dos arquivos
TRAIN_FILE = "train.csv"
VALID_FILE = "validation.csv"
TEST_FILE = "test.csv"

# Criar diretórios necessários
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Configuração de logging
log_file = os.path.join(REPORT_DIR, "balancing_experiments.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 2. Carregamento de dados
def load_datasets(input_dir):
    """Carrega os datasets de treino, validação e teste"""
    logger.info(f"Carregando datasets de {input_dir}")
    
    try:
        train_df = pd.read_csv(os.path.join(input_dir, TRAIN_FILE))
        valid_df = pd.read_csv(os.path.join(input_dir, VALID_FILE))
        test_df = pd.read_csv(os.path.join(input_dir, TEST_FILE))
        
        logger.info(f"Dados carregados - treino: {train_df.shape}, validação: {valid_df.shape}, teste: {test_df.shape}")
        
        # Identificar coluna target
        target_col = 'target' if 'target' in train_df.columns else None
        if target_col is None:
            # Procurar outras possíveis colunas target
            target_candidates = [col for col in train_df.columns if 'target' in col.lower() or 'label' in col.lower()]
            if target_candidates:
                target_col = target_candidates[0]
            else:
                raise ValueError("Não foi possível identificar a coluna target")
        
        logger.info(f"Coluna target identificada: {target_col}")
        logger.info(f"Distribuição do target no treino: {train_df[target_col].value_counts(normalize=True)}")
        
        return train_df, valid_df, test_df, target_col
    
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        raise

# 3. Carregar modelo do MLflow
def get_latest_random_forest_run(mlflow_dir):
    """Obtém o run_id do modelo RandomForest mais recente."""
    # Configurar MLflow
    if os.path.exists(mlflow_dir):
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    else:
        logger.warning(f"AVISO: Diretório MLflow não encontrado: {mlflow_dir}")
        return None, None, None
    
    # Inicializar cliente MLflow
    client = mlflow.tracking.MlflowClient()
    
    # Procurar todos os experimentos
    experiments = client.search_experiments()
    
    for experiment in experiments:
        logger.info(f"Verificando experimento: {experiment.name} (ID: {experiment.experiment_id})")
        
        # Buscar runs específicos do RandomForest ordenados pelo mais recente
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_type = 'random_forest'",
            order_by=["attribute.start_time DESC"]
        )
        
        if not runs:
            # Se não achou pela tag, procurar pelo nome do artefato
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"]
            )
        
        for run in runs:
            run_id = run.info.run_id
            logger.info(f"  Encontrado run: {run_id}")
            
            # Verificar artefatos
            artifacts = client.list_artifacts(run_id)
            rf_artifact = None
            
            for artifact in artifacts:
                if artifact.is_dir and artifact.path == 'random_forest':
                    rf_artifact = artifact
                    break
            
            if rf_artifact:
                # Extrair o threshold das métricas
                threshold = run.data.metrics.get('threshold', 0.17)  # Fallback para 0.17 se não encontrar
                model_uri = f"runs:/{run_id}/random_forest"
                
                logger.info(f"  Usando modelo RandomForest de {run.info.start_time}")
                logger.info(f"  Run ID: {run_id}")
                logger.info(f"  Model URI: {model_uri}")
                logger.info(f"  Threshold: {threshold}")
                
                # Mostrar métricas registradas no MLflow
                precision = run.data.metrics.get('precision', None)
                recall = run.data.metrics.get('recall', None)
                f1 = run.data.metrics.get('f1', None)
                
                if precision and recall and f1:
                    logger.info(f"  Métricas do MLflow: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                
                return run_id, threshold, model_uri
    
    logger.warning("Nenhum modelo RandomForest encontrado em MLflow.")
    return None, None, None

def load_model_from_mlflow(model_uri):
    """Carrega um modelo a partir do MLflow usando seu URI."""
    try:
        logger.info(f"Carregando modelo de: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        return None

# 4. Implementação de técnicas de balanceamento
def apply_balancing_technique(X_train, y_train, technique='smote', random_state=42):
    """Aplica uma técnica de balanceamento específica aos dados de treino"""
    logger.info(f"Aplicando técnica de balanceamento: {technique}")
    
    balancing_techniques = {
        'random_over': RandomOverSampler(random_state=random_state),
        'smote': SMOTE(random_state=random_state),
        'adasyn': ADASYN(random_state=random_state),
        'random_under': RandomUnderSampler(random_state=random_state),
        'tomek': TomekLinks(),
        'smote_tomek': SMOTETomek(random_state=random_state),
        'smote_enn': SMOTEENN(random_state=random_state)
    }
    
    if technique not in balancing_techniques:
        logger.error(f"Técnica de balanceamento '{technique}' não implementada")
        return X_train, y_train
    
    try:
        # Registrar distribuição original
        original_dist = np.bincount(y_train)
        logger.info(f"Distribuição original: {original_dist}")
        
        # Aplicar técnica de balanceamento
        start_time = time.time()
        X_resampled, y_resampled = balancing_techniques[technique].fit_resample(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        # Registrar nova distribuição
        new_dist = np.bincount(y_resampled)
        logger.info(f"Nova distribuição após {technique}: {new_dist}")
        logger.info(f"Proporção de classes após balanceamento: {new_dist[1]/new_dist[0]:.4f}")
        logger.info(f"Tempo para balanceamento: {elapsed_time:.2f} segundos")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        logger.error(f"Erro ao aplicar técnica de balanceamento {technique}: {str(e)}")
        return X_train, y_train

# 5. Implementação de ajuste de threshold
def find_optimal_threshold(y_true, y_prob, metric='f1', beta=1.0):
    """Encontra o threshold ótimo com base na métrica especificada"""
    logger.info(f"Buscando threshold ótimo baseado em {metric}")
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Adicionar threshold=1.0 (omitido pela função precision_recall_curve)
    thresholds = np.append(thresholds, 1.0)
    
    if metric == 'f1':
        # Calcular F1 para cada threshold
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_score = f1_scores[best_idx]
        logger.info(f"Melhor threshold para F1: {best_threshold:.4f} (F1={best_score:.4f})")
    
    elif metric == 'f_beta':
        # Calcular F-beta para cada threshold
        f_beta_scores = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-8)
        best_idx = np.argmax(f_beta_scores)
        best_threshold = thresholds[best_idx]
        best_score = f_beta_scores[best_idx]
        logger.info(f"Melhor threshold para F-beta (beta={beta}): {best_threshold:.4f} (Score={best_score:.4f})")
    
    else:
        # Usar F1 como padrão
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_score = f1_scores[best_idx]
        logger.info(f"Melhor threshold: {best_threshold:.4f} (F1={best_score:.4f})")
    
    # Criar plot da curva precision-recall e threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], 'b--', label='Precisão')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    
    if metric == 'f1':
        plt.plot(thresholds, f1_scores[:-1], 'r-.', label='F1 Score')
    elif metric == 'f_beta':
        plt.plot(thresholds, f_beta_scores[:-1], 'r-.', label=f'F-beta (beta={beta})')
    
    plt.axvline(x=best_threshold, color='k', linestyle=':',
               label=f'Threshold ótimo: {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Efeito do Threshold na Performance - Métrica: {metric}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar plot
    threshold_plot_path = os.path.join(REPORT_DIR, f"threshold_optimization_{metric}.png")
    plt.savefig(threshold_plot_path)
    plt.close()
    
    return best_threshold, best_score

# 6. Implementação de class weights
def calculate_class_weights(y, strategy='balanced'):
    """Calcula os pesos para cada classe com base na estratégia especificada"""
    logger.info(f"Calculando pesos de classe com estratégia: {strategy}")
    
    class_counts = np.bincount(y)
    n_samples = len(y)
    n_classes = len(class_counts)
    
    if strategy == 'balanced':
        # Peso inversamente proporcional à frequência da classe
        weights = n_samples / (n_classes * class_counts)
        logger.info(f"Pesos 'balanced': {weights}")
        return {i: w for i, w in enumerate(weights)}
    
    elif strategy == 'custom':
        # Pesos personalizados enfatizando mais a classe minoritária
        neg_weight = 1.0
        pos_weight = class_counts[0] / class_counts[1] * 2  # 2x o peso balanceado
        weights = {0: neg_weight, 1: pos_weight}
        logger.info(f"Pesos 'custom': {weights}")
        return weights
    
    elif isinstance(strategy, dict):
        # Usar pesos fornecidos diretamente
        logger.info(f"Usando pesos personalizados: {strategy}")
        return strategy
    
    else:
        logger.warning(f"Estratégia de peso '{strategy}' não reconhecida. Usando 'balanced'.")
        weights = n_samples / (n_classes * class_counts)
        return {i: w for i, w in enumerate(weights)}

# 7. Avaliação de modelo
def evaluate_model(model, X, y, threshold=0.5, prefix=""):
    """Avalia o modelo com métricas detalhadas"""
    logger.info(f"Avaliando modelo {'('+prefix+')' if prefix else ''}")
    
    # Fazer previsões
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculando métricas
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    
    # Extrair métricas principais
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    
    logger.info(f"Threshold usado: {threshold:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Matriz de confusão:\n{cm}")
    
    # Visualizar matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Converteu', 'Converteu'],
                yticklabels=['Não Converteu', 'Converteu'])
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão {prefix}')
    
    # Salvar plot
    cm_plot_path = os.path.join(REPORT_DIR, f"confusion_matrix_{prefix or 'model'}.png")
    plt.savefig(cm_plot_path)
    plt.close()
    
    # Distribuição de probabilidades
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y == 0], bins=50, alpha=0.5, color='blue', label='Não Converteu')
    plt.hist(y_prob[y == 1], bins=50, alpha=0.5, color='red', label='Converteu')
    plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title(f'Distribuição de Probabilidades {prefix}')
    plt.xlabel('Probabilidade Prevista')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar plot de probabilidades
    prob_plot_path = os.path.join(REPORT_DIR, f"prob_dist_{prefix or 'model'}.png")
    plt.savefig(prob_plot_path)
    plt.close()
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_prob': y_prob,
        'y_pred': y_pred
    }

# 8. Execução de experimentos
def run_experiments(train_df, valid_df, model, target_col, experiments=None):
    """Executa uma série de experimentos e avalia os resultados"""
    logger.info("=== Iniciando experimentos de balanceamento ===")
    
    # Separar features e target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_valid = valid_df.drop(columns=[target_col])
    y_valid = valid_df[target_col]
    
    # Definir experimentos padrão se nenhum for especificado
    if experiments is None:
        experiments = [
            # Baseline (sem balanceamento, threshold padrão)
            {'name': 'baseline', 'balancing': None, 'threshold': 0.5, 'class_weight': None},
            
            # Apenas threshold otimizado (sem balanceamento)
            {'name': 'optimal_threshold', 'balancing': None, 'threshold': 'optimal', 'class_weight': None},
            
            # Apenas class weights
            {'name': 'class_weights', 'balancing': None, 'threshold': 0.5, 'class_weight': 'balanced'},
            {'name': 'custom_weights', 'balancing': None, 'threshold': 0.5, 'class_weight': 'custom'},
            
            # Class weights + threshold otimizado
            {'name': 'weights_threshold', 'balancing': None, 'threshold': 'optimal', 'class_weight': 'balanced'},
            
            # Balanceamento com threshold padrão
            {'name': 'smote', 'balancing': 'smote', 'threshold': 0.5, 'class_weight': None},
            {'name': 'smote_tomek', 'balancing': 'smote_tomek', 'threshold': 0.5, 'class_weight': None},
            {'name': 'smote_enn', 'balancing': 'smote_enn', 'threshold': 0.5, 'class_weight': None},
            
            # Balanceamento + threshold otimizado
            {'name': 'smote_optimal', 'balancing': 'smote', 'threshold': 'optimal', 'class_weight': None},
            {'name': 'smote_tomek_optimal', 'balancing': 'smote_tomek', 'threshold': 'optimal', 'class_weight': None},
            
            # Combinações completas
            {'name': 'smote_weights_optimal', 'balancing': 'smote', 'threshold': 'optimal', 'class_weight': 'balanced'},
        ]
    
    # Resultados para cada experimento
    results = {}
    
    # Avaliar baseline primeiro (modelo original)
    baseline_results = evaluate_model(model, X_valid, y_valid, threshold=0.5, prefix="baseline")
    results['baseline'] = baseline_results
    
    # Encontrar threshold ótimo usando o modelo original
    optimal_threshold, _ = find_optimal_threshold(y_valid, baseline_results['y_prob'])
    
    # Executar cada experimento
    for exp in experiments:
        exp_name = exp['name']
        balancing = exp['balancing']
        threshold_strategy = exp['threshold']
        class_weight_strategy = exp['class_weight']
        
        if exp_name == 'baseline':
            # Já avaliado, pular
            continue
        
        logger.info(f"\n=== Experimento: {exp_name} ===")
        logger.info(f"Balanceamento: {balancing}")
        logger.info(f"Threshold: {threshold_strategy}")
        logger.info(f"Class Weight: {class_weight_strategy}")
        
        # Aplicar balanceamento se especificado
        if balancing:
            X_resampled, y_resampled = apply_balancing_technique(X_train, y_train, technique=balancing)
        else:
            X_resampled, y_resampled = X_train, y_train
        
        # Calcular class weights se especificado
        if class_weight_strategy:
            class_weights = calculate_class_weights(y_resampled, strategy=class_weight_strategy)
        else:
            class_weights = None
        
        # Treinar novo modelo com os dados balanceados/pesos
        exp_model = RandomForestClassifier(
            n_estimators=100,  # Usar mesmo número que o modelo original
            random_state=42,
            class_weight=class_weights,
            n_jobs=-1
        )
        
        logger.info(f"Treinando modelo para experimento {exp_name}...")
        exp_model.fit(X_resampled, y_resampled)
        
        # Determinar threshold para avaliação
        if threshold_strategy == 'optimal':
            # Fazer previsões no conjunto de validação
            y_valid_prob = exp_model.predict_proba(X_valid)[:, 1]
            
            # Encontrar threshold ótimo
            threshold, _ = find_optimal_threshold(y_valid, y_valid_prob)
        else:
            threshold = float(threshold_strategy)
        
        # Avaliar modelo
        exp_results = evaluate_model(exp_model, X_valid, y_valid, threshold=threshold, prefix=exp_name)
        results[exp_name] = exp_results
        
        # Se os resultados forem satisfatórios, salvar o modelo e os dados
        if exp_results['recall'] >= 0.40 and exp_results['precision'] >= 0.85:
            logger.info(f"*** Experimento {exp_name} atende aos critérios de desempenho! ***")
            
            # Salvar modelo
            model_path = os.path.join(MODEL_DIR, "enhanced_model.joblib")
            joblib.dump(exp_model, model_path)
            logger.info(f"Modelo salvo em {model_path}")
            
            # Preparar metadados do modelo
            model_info = {
                'name': exp_name,
                'balancing_technique': balancing,
                'threshold': threshold,
                'class_weights': class_weights,
                'precision': exp_results['precision'],
                'recall': exp_results['recall'],
                'f1': exp_results['f1'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Salvar metadados
            model_info_path = os.path.join(MODEL_DIR, "enhanced_model_info.json")
            import json
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            
            # Salvar dados balanceados para treino futuro
            if balancing:
                balanced_df = pd.DataFrame(X_resampled, columns=X_train.columns)
                balanced_df[target_col] = y_resampled
                balanced_output_path = os.path.join(OUTPUT_DIR, f"train_balanced_{exp_name}.csv")
                balanced_df.to_csv(balanced_output_path, index=False)
                logger.info(f"Dados de treino balanceados salvos em {balanced_output_path}")
                
                # Copiar dados de validação e teste
                valid_df.to_csv(os.path.join(OUTPUT_DIR, "validation.csv"), index=False)
                if os.path.exists(os.path.join(INPUT_DIR, TEST_FILE)):
                    test_df = pd.read_csv(os.path.join(INPUT_DIR, TEST_FILE))
                    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
        else:
            logger.info(f"Experimento {exp_name} não atende aos critérios de desempenho")
    
    # Gerar relatório comparativo
    generate_comparison_report(results)
    
    return results

# 9. Geração de relatório comparativo
def generate_comparison_report(results):
    """Gera um relatório comparativo dos experimentos"""
    logger.info("Gerando relatório comparativo dos experimentos")
    
    # Extrair métricas principais para cada experimento
    summary = []
    for exp_name, exp_results in results.items():
        summary.append({
            'Experimento': exp_name,
            'Threshold': exp_results['threshold'],
            'Precision': exp_results['precision'],
            'Recall': exp_results['recall'],
            'F1 Score': exp_results['f1']
        })
    
    # Converter para DataFrame e ordenar por F1
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('F1 Score', ascending=False)
    
    # Salvar tabela comparativa
    summary_path = os.path.join(REPORT_DIR, "experimentos_comparacao.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Gerar visualização comparativa
    plt.figure(figsize=(12, 8))
    
    # Ordenar experimentos por F1 para visualização
    sorted_indices = summary_df.index
    exp_names = summary_df['Experimento']
    
    # Criar barras lado a lado para cada métrica
    x = np.arange(len(exp_names))
    width = 0.25
    
    plt.bar(x - width, summary_df['Precision'], width, label='Precision')
    plt.bar(x, summary_df['Recall'], width, label='Recall')
    plt.bar(x + width, summary_df['F1 Score'], width, label='F1 Score')
    
    plt.xlabel('Experimento')
    plt.ylabel('Score')
    plt.title('Comparação de Experimentos')
    plt.xticks(x, exp_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar visualização
    plot_path = os.path.join(REPORT_DIR, "experimentos_comparacao.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Relatório comparativo salvo em {summary_path}")
    logger.info(f"Visualização comparativa salva em {plot_path}")
    
    # Exibir resultados ordenados
    logger.info("\n=== Resultados dos Experimentos (ordenados por F1) ===")
    for i, row in summary_df.iterrows():
        logger.info(f"{row['Experimento']}: Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1 Score']:.4f}")

# 10. Função principal
def main():
    """Função principal que orquestra todo o processo"""
    logger.info("=== Iniciando experimentos de balanceamento e cost-sensitive learning ===")
    
    try:
        # Carregar dados
        train_df, valid_df, test_df, target_col = load_datasets(INPUT_DIR)
        
        # Carregar modelo
        run_id, default_threshold, model_uri = get_latest_random_forest_run(MLFLOW_DIR)
        
        if model_uri is None:
            logger.error("Não foi possível encontrar o modelo Random Forest. Abortando.")
            return
        
        model = load_model_from_mlflow(model_uri)
        
        if model is None:
            logger.error("Falha ao carregar modelo. Abortando.")
            return
        
        # Registrar threshold padrão
        logger.info(f"Threshold padrão do modelo: {default_threshold}")
        
        # Executar experimentos
        results = run_experiments(train_df, valid_df, model, target_col)
        
        # Verificar se algum experimento foi bem-sucedido (recall >= 40%, precision >= 85%)
        success = any(r['recall'] >= 0.40 and r['precision'] >= 0.85 for r in results.values())
        
        if success:
            logger.info("=== Sucesso! Pelo menos um experimento atendeu aos critérios de desempenho ===")
        else:
            logger.info("=== Nenhum experimento atendeu aos critérios de desempenho ===")
        
        # Log final
        logger.info("Experimentos concluídos. Consulte o relatório comparativo para detalhes.")
        
    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()