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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, fbeta_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import mlflow

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar a função de sanitização original
from src.evaluation.baseline_model import sanitize_column_names, get_latest_random_forest_run

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

# 4. Preparar dados para o modelo usando a mesma função de sanitização
def prepare_data_for_model(model, df, target_col="target"):
    """
    Prepara os dados para serem compatíveis com o modelo usando a mesma
    função de sanitização que foi usada no treinamento original.
    
    Args:
        model: Modelo treinado
        df: DataFrame a preparar
        target_col: Nome da coluna target
        
    Returns:
        X, y preparados para o modelo
    """
    logger.info("Preparando dados para compatibilidade com o modelo...")
    
    # Criar cópia para evitar modificar o original
    df_copy = df.copy()
    
    # 1. Aplicar a mesma sanitização que foi usada durante o treinamento
    logger.info("Aplicando sanitização aos nomes das colunas...")
    column_mapping = sanitize_column_names(df_copy)
    
    # 2. Separar features e target
    X = df_copy.drop(columns=[target_col]) if target_col in df_copy.columns else df_copy.copy()
    y = df_copy[target_col] if target_col in df_copy.columns else None
    
    # 3. Verificar se o modelo tem atributo feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        expected_features = set(model.feature_names_in_)
        actual_features = set(X.columns)
        
        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features
        
        if missing_features:
            logger.info(f"Adicionando {len(missing_features)} features faltantes")
            if len(missing_features) <= 5:
                logger.info(f"  Features faltantes: {missing_features}")
            else:
                logger.info(f"  Exemplos de features faltantes: {list(missing_features)[:5]}...")
            
            # Criar DataFrame vazio com as colunas faltantes
            missing_cols = list(missing_features)
            missing_dict = {col: [0] * len(X) for col in missing_cols}
            missing_df = pd.DataFrame(missing_dict)
            
            # Concatenar com o DataFrame original
            X = pd.concat([X, missing_df], axis=1)
        
        if extra_features:
            logger.info(f"Removendo {len(extra_features)} features extras que o modelo não espera")
            if len(extra_features) <= 5:
                logger.info(f"  Features extras: {extra_features}")
            else:
                logger.info(f"  Exemplos de features extras: {list(extra_features)[:5]}...")
            
            # Remover features extras
            X = X.drop(columns=list(extra_features))
        
        # Garantir a ordem exata das colunas como esperado pelo modelo
        X = X[model.feature_names_in_]
        logger.info(f"Dados preparados com {len(model.feature_names_in_)} features na ordem correta")
    
    # 4. Converter inteiros para float como no script original
    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col].dtype):
            X.loc[:, col] = X[col].astype(float)
    
    return X, y

# 5. Pré-processamento para imputação de valores faltantes
def impute_missing_values(X):
    """
    Imputa valores faltantes no DataFrame para compatibilidade com SMOTE.
    
    Args:
        X: DataFrame com features
        
    Returns:
        DataFrame com valores faltantes imputados
    """
    logger.info("Imputando valores faltantes para compatibilidade com SMOTE...")
    
    # Verificar se há valores faltantes
    missing_count = X.isna().sum().sum()
    if missing_count == 0:
        logger.info("Não há valores faltantes para imputar.")
        return X
    
    logger.info(f"Total de valores faltantes: {missing_count}")
    
    # Usar SimpleImputer para substituir valores NaN com a média
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Verificar se ainda há valores faltantes após imputação
    remaining_missing = X_imputed.isna().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"Ainda restam {remaining_missing} valores faltantes após imputação!")
        # Substituir qualquer NaN remanescente com 0
        X_imputed.fillna(0, inplace=True)
    
    logger.info("Imputação de valores faltantes concluída.")
    return X_imputed

# 6. Implementação de técnicas de balanceamento
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
        
        # Verificar e imputar valores faltantes antes do balanceamento
        X_train_imputed = impute_missing_values(X_train)
        
        # Aplicar técnica de balanceamento
        start_time = time.time()
        X_resampled, y_resampled = balancing_techniques[technique].fit_resample(X_train_imputed, y_train)
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

# 7. Implementação de ajuste de threshold
def find_optimal_threshold(y_true, y_prob, metric='f1', beta=1.0):
    """Encontra o threshold ótimo com base na métrica especificada"""
    logger.info(f"Buscando threshold ótimo baseado em {metric} (beta={beta if metric=='f_beta' else 1.0})")
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Garantir que tenhamos arrays de tamanhos compatíveis para o plot
    # precision e recall têm um elemento a mais que thresholds
    plot_precision = precision[:-1]
    plot_recall = recall[:-1]
    
    # Verificar e garantir que os tamanhos sejam compatíveis
    if len(thresholds) != len(plot_precision):
        logger.warning(f"Incompatibilidade de tamanhos: thresholds={len(thresholds)}, precision={len(plot_precision)}")
        # Ajustar tamanhos se necessário
        min_len = min(len(thresholds), len(plot_precision))
        thresholds = thresholds[:min_len]
        plot_precision = plot_precision[:min_len]
        plot_recall = plot_recall[:min_len]
    
    if metric == 'f1':
        # Calcular F1 para cada threshold
        f1_scores = (2 * plot_precision * plot_recall) / (plot_precision + plot_recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = f1_scores[best_idx]
        logger.info(f"Melhor threshold para F1: {best_threshold:.4f} (F1={best_score:.4f})")
    
    elif metric == 'f_beta':
        # Calcular F-beta para cada threshold
        f_beta_scores = ((1 + beta**2) * plot_precision * plot_recall) / (beta**2 * plot_precision + plot_recall + 1e-8)
        best_idx = np.argmax(f_beta_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = f_beta_scores[best_idx]
        logger.info(f"Melhor threshold para F-beta (beta={beta}): {best_threshold:.4f} (Score={best_score:.4f})")
    
    elif metric == 'recall_precision_constraint':
        # Encontrar o threshold que maximiza recall com precision >= 0.85
        valid_idx = plot_precision >= 0.85
        
        if np.any(valid_idx):
            # Filtrar apenas os thresholds que satisfazem a constraint
            valid_recalls = plot_recall[valid_idx]
            valid_thresholds = thresholds[valid_idx]
            
            # Encontrar o índice de maximum recall
            best_idx = np.argmax(valid_recalls)
            best_threshold = valid_thresholds[best_idx]
            best_precision = plot_precision[valid_idx][best_idx]
            best_recall = valid_recalls[best_idx]
            
            logger.info(f"Melhor threshold para recall com precision >= 0.85: {best_threshold:.4f}")
            logger.info(f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
        else:
            # Se nenhum threshold atende a constraint
            logger.warning("Nenhum threshold satisfaz precision >= 0.85")
            # Usar o threshold com maior precisão
            best_idx = np.argmax(plot_precision)
            best_threshold = thresholds[best_idx]
            logger.info(f"Usando threshold com maior precisão: {best_threshold:.4f}")
            
        best_score = 0  # Não relevante nesse caso
    
    else:
        # Usar F1 como padrão
        f1_scores = (2 * plot_precision * plot_recall) / (plot_precision + plot_recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_score = f1_scores[best_idx]
        logger.info(f"Melhor threshold: {best_threshold:.4f} (F1={best_score:.4f})")
    
    # Criar plot da curva precision-recall e threshold
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, plot_precision, 'b--', label='Precisão')
        plt.plot(thresholds, plot_recall, 'g-', label='Recall')
        
        if metric == 'f1':
            plt.plot(thresholds, f1_scores, 'r-.', label='F1 Score')
        elif metric == 'f_beta':
            plt.plot(thresholds, f_beta_scores, 'r-.', label=f'F-beta (beta={beta})')
        
        plt.axvline(x=best_threshold, color='k', linestyle=':',
                   label=f'Threshold ótimo: {best_threshold:.3f}')
        
        # Adicionar linha para threshold de precision = 0.85
        plt.axhline(y=0.85, color='m', linestyle='--', label='Precision = 0.85')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Efeito do Threshold na Performance - Métrica: {metric}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Salvar plot
        threshold_plot_path = os.path.join(REPORT_DIR, f"threshold_optimization_{metric}.png")
        plt.savefig(threshold_plot_path)
        plt.close()
    except Exception as e:
        logger.warning(f"Erro ao criar gráfico de threshold: {str(e)}")
    
    return best_threshold, best_score

# 8. Implementação de class weights
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
    
    elif strategy == 'extreme':
        # Pesos extremos para priorizar muito a classe minoritária
        neg_weight = 1.0
        pos_weight = class_counts[0] / class_counts[1] * 5  # 5x o peso balanceado
        weights = {0: neg_weight, 1: pos_weight}
        logger.info(f"Pesos 'extreme': {weights}")
        return weights
    
    elif isinstance(strategy, dict):
        # Usar pesos fornecidos diretamente
        logger.info(f"Usando pesos personalizados: {strategy}")
        return strategy
    
    else:
        logger.warning(f"Estratégia de peso '{strategy}' não reconhecida. Usando 'balanced'.")
        weights = n_samples / (n_classes * class_counts)
        return {i: w for i, w in enumerate(weights)}

# 9. Avaliação de modelo
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
    precision = report['1']['precision'] if '1' in report else 0
    recall = report['1']['recall'] if '1' in report else 0
    f1 = report['1']['f1-score'] if '1' in report else 0
    
    # Calcular F2-score (dá mais peso ao recall)
    f2 = fbeta_score(y, y_pred, beta=2.0)
    
    logger.info(f"Threshold usado: {threshold:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"F2 Score: {f2:.4f}")
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
        'f2': f2,
        'confusion_matrix': cm,
        'y_prob': y_prob,
        'y_pred': y_pred
    }

# 10. Execução de experimentos
def run_experiments(train_df, valid_df, model, target_col, experiments=None):
    """Executa uma série de experimentos e avalia os resultados"""
    logger.info("=== Iniciando experimentos de balanceamento ===")
    
    # Primeiro preparar os dados para serem compatíveis com o modelo
    X_train, y_train = prepare_data_for_model(model, train_df, target_col)
    X_valid, y_valid = prepare_data_for_model(model, valid_df, target_col)
    
    # Definir experimentos padrão se nenhum for especificado
    if experiments is None:
        experiments = [
            # Baseline (sem balanceamento, usando threshold do MLflow)
            {'name': 'baseline_mlflow', 'balancing': None, 'threshold': 0.12, 'class_weight': None},
            
            # Threshold para maximizar recall com precision >= 0.85
            {'name': 'recall_precision_constraint', 'balancing': None, 'threshold': 'recall_precision_constraint', 'class_weight': None},
            
            # Pesos extremos para classe minoritária
            {'name': 'extreme_weights', 'balancing': None, 'threshold': 0.5, 'class_weight': 'extreme'},
            {'name': 'extreme_weights_optimal', 'balancing': None, 'threshold': 'optimal', 'class_weight': 'extreme'},
            
            # F-beta com diferentes betas
            {'name': 'f2_score', 'balancing': None, 'threshold': 'f_beta', 'threshold_params': {'beta': 2.0}, 'class_weight': 'balanced'},
            {'name': 'f3_score', 'balancing': None, 'threshold': 'f_beta', 'threshold_params': {'beta': 3.0}, 'class_weight': 'balanced'},
            
            # Usar F2 com pesos extremos
            {'name': 'f2_extreme_weights', 'balancing': None, 'threshold': 'f_beta', 'threshold_params': {'beta': 2.0}, 'class_weight': 'extreme'},
            
            # Random Forest com mais estimadores
            {'name': 'more_estimators', 'balancing': None, 'threshold': 'optimal', 'class_weight': 'balanced', 'n_estimators': 200},
            
            # Combinação de técnicas
            {'name': 'combined_approach', 'balancing': 'random_over', 'threshold': 'f_beta', 'threshold_params': {'beta': 2.0}, 'class_weight': 'balanced'}
        ]
    
    # Resultados para cada experimento
    results = {}
    
    # Avaliar baseline primeiro (modelo original com threshold do MLflow)
    baseline_results = evaluate_model(model, X_valid, y_valid, threshold=0.12, prefix="baseline_mlflow")
    results['baseline_mlflow'] = baseline_results
    
    # Executar cada experimento
    for exp in experiments:
        exp_name = exp['name']
        balancing = exp.get('balancing')
        threshold_strategy = exp.get('threshold')
        threshold_params = exp.get('threshold_params', {})
        class_weight_strategy = exp.get('class_weight')
        n_estimators = exp.get('n_estimators', 100)
        
        if exp_name == 'baseline_mlflow':
            # Já avaliado, pular
            continue
        
        logger.info(f"\n=== Experimento: {exp_name} ===")
        logger.info(f"Balanceamento: {balancing}")
        logger.info(f"Threshold: {threshold_strategy}")
        logger.info(f"Class Weight: {class_weight_strategy}")
        logger.info(f"N Estimators: {n_estimators}")
        
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
            n_estimators=n_estimators,
            random_state=42,
            class_weight=class_weights,
            n_jobs=-1,
            max_depth=15  # Aumentar profundidade para capturar padrões mais complexos
        )
        
        logger.info(f"Treinando modelo para experimento {exp_name}...")
        exp_model.fit(X_resampled, y_resampled)
        
        # Determinar threshold para avaliação
        if threshold_strategy == 'optimal':
            # Fazer previsões no conjunto de validação
            y_valid_prob = exp_model.predict_proba(X_valid)[:, 1]
            
            # Encontrar threshold ótimo
            threshold, _ = find_optimal_threshold(y_valid, y_valid_prob)
        elif threshold_strategy == 'f_beta':
            # Fazer previsões no conjunto de validação
            y_valid_prob = exp_model.predict_proba(X_valid)[:, 1]
            
            # Encontrar threshold ótimo para F-beta
            beta = threshold_params.get('beta', 2.0)
            threshold, _ = find_optimal_threshold(y_valid, y_valid_prob, metric='f_beta', beta=beta)
        elif threshold_strategy == 'recall_precision_constraint':
            # Fazer previsões no conjunto de validação
            y_valid_prob = exp_model.predict_proba(X_valid)[:, 1]
            
            # Encontrar threshold que maximiza recall com precision >= 0.85
            threshold, _ = find_optimal_threshold(y_valid, y_valid_prob, metric='recall_precision_constraint')
        else:
            threshold = float(threshold_strategy)
        
        # Avaliar modelo
        exp_results = evaluate_model(exp_model, X_valid, y_valid, threshold=threshold, prefix=exp_name)
        results[exp_name] = exp_results
        
        # Se os resultados forem satisfatórios, salvar o modelo e os dados
        if exp_results['recall'] >= 0.40 and exp_results['precision'] >= 0.85:
            logger.info(f"*** Experimento {exp_name} atende aos critérios de desempenho! ***")
            
            # Salvar modelo
            model_path = os.path.join(MODEL_DIR, f"enhanced_model_{exp_name}.joblib")
            joblib.dump(exp_model, model_path)
            logger.info(f"Modelo salvo em {model_path}")
            
            # Preparar metadados do modelo
            model_info = {
                'name': exp_name,
                'balancing_technique': balancing,
                'threshold': threshold,
                'class_weights': class_weights,
                'n_estimators': n_estimators,
                'precision': exp_results['precision'],
                'recall': exp_results['recall'],
                'f1': exp_results['f1'],
                'f2': exp_results['f2'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Salvar metadados
            model_info_path = os.path.join(MODEL_DIR, f"enhanced_model_{exp_name}_info.json")
            import json
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=4, default=str)  # default=str para serializar objetos não padrão
            
            # Salvar dados balanceados para treino futuro
            if balancing:
                # Reconstruir DataFrame com as features originais
                balanced_df = pd.DataFrame(X_resampled, columns=model.feature_names_in_)
                balanced_df[target_col] = y_resampled
                balanced_output_path = os.path.join(OUTPUT_DIR, f"train_balanced_{exp_name}.csv")
                balanced_df.to_csv(balanced_output_path, index=False)
                logger.info(f"Dados de treino balanceados salvos em {balanced_output_path}")
                
                # Copiar dados de validação e teste
                valid_df_prepared = pd.DataFrame(X_valid, columns=model.feature_names_in_)
                valid_df_prepared[target_col] = y_valid
                valid_df_prepared.to_csv(os.path.join(OUTPUT_DIR, "validation.csv"), index=False)
                
                if os.path.exists(os.path.join(INPUT_DIR, TEST_FILE)):
                    test_df = pd.read_csv(os.path.join(INPUT_DIR, TEST_FILE))
                    X_test, y_test = prepare_data_for_model(model, test_df, target_col)
                    test_df_prepared = pd.DataFrame(X_test, columns=model.feature_names_in_)
                    test_df_prepared[target_col] = y_test
                    test_df_prepared.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
        else:
            logger.info(f"Experimento {exp_name} não atende aos critérios de desempenho")
    
    # Gerar relatório comparativo
    generate_comparison_report(results)
    
    return results

# 11. Geração de relatório comparativo
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
            'F1 Score': exp_results['f1'],
            'F2 Score': exp_results['f2']
        })
    
    # Converter para DataFrame e ordenar por F1
    summary_df = pd.DataFrame(summary)
    
    # Criar duas versões ordenadas - uma por F1 e outra por F2
    summary_df_f1 = summary_df.sort_values('F1 Score', ascending=False)
    summary_df_f2 = summary_df.sort_values('F2 Score', ascending=False)
    
    # Salvar tabela comparativa
    summary_path = os.path.join(REPORT_DIR, "experimentos_comparacao.csv")
    summary_df_f1.to_csv(summary_path, index=False)
    
    # Gerar visualização comparativa para F1
    plt.figure(figsize=(12, 8))
    x = np.arange(len(summary_df_f1))
    width = 0.2
    
    plt.bar(x - width*1.5, summary_df_f1['Precision'], width, label='Precision')
    plt.bar(x - width/2, summary_df_f1['Recall'], width, label='Recall')
    plt.bar(x + width/2, summary_df_f1['F1 Score'], width, label='F1 Score')
    plt.bar(x + width*1.5, summary_df_f1['F2 Score'], width, label='F2 Score')
    
    plt.xlabel('Experimento')
    plt.ylabel('Score')
    plt.title('Comparação de Experimentos (ordenado por F1)')
    plt.xticks(x, summary_df_f1['Experimento'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar visualização
    plot_path = os.path.join(REPORT_DIR, "experimentos_comparacao_f1.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Gerar visualização comparativa para F2
    plt.figure(figsize=(12, 8))
    x = np.arange(len(summary_df_f2))
    
    plt.bar(x - width*1.5, summary_df_f2['Precision'], width, label='Precision')
    plt.bar(x - width/2, summary_df_f2['Recall'], width, label='Recall')
    plt.bar(x + width/2, summary_df_f2['F1 Score'], width, label='F1 Score')
    plt.bar(x + width*1.5, summary_df_f2['F2 Score'], width, label='F2 Score')
    
    plt.xlabel('Experimento')
    plt.ylabel('Score')
    plt.title('Comparação de Experimentos (ordenado por F2)')
    plt.xticks(x, summary_df_f2['Experimento'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar visualização
    plot_path_f2 = os.path.join(REPORT_DIR, "experimentos_comparacao_f2.png")
    plt.savefig(plot_path_f2)
    plt.close()
    
    logger.info(f"Relatório comparativo salvo em {summary_path}")
    logger.info(f"Visualizações comparativas salvas em {plot_path} e {plot_path_f2}")
    
    # Exibir resultados ordenados por F1
    logger.info("\n=== Resultados dos Experimentos (ordenados por F1) ===")
    for i, row in summary_df_f1.iterrows():
        logger.info(f"{row['Experimento']}: Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1 Score']:.4f}, F2={row['F2 Score']:.4f}")
    
    # Exibir resultados ordenados por F2 (enfatiza recall)
    logger.info("\n=== Resultados dos Experimentos (ordenados por F2) ===")
    for i, row in summary_df_f2.iterrows():
        logger.info(f"{row['Experimento']}: Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1 Score']:.4f}, F2={row['F2 Score']:.4f}")

# 12. Função principal
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
            
            # Identificar o melhor experimento por F1 e F2
            best_f1_exp = max(results.items(), key=lambda x: x[1]['f1'])
            best_f2_exp = max(results.items(), key=lambda x: x[1]['f2'])
            
            logger.info(f"Melhor experimento por F1: {best_f1_exp[0]}")
            logger.info(f"Precision: {best_f1_exp[1]['precision']:.4f}, Recall: {best_f1_exp[1]['recall']:.4f}, F1: {best_f1_exp[1]['f1']:.4f}")
            
            logger.info(f"Melhor experimento por F2: {best_f2_exp[0]}")
            logger.info(f"Precision: {best_f2_exp[1]['precision']:.4f}, Recall: {best_f2_exp[1]['recall']:.4f}, F2: {best_f2_exp[1]['f2']:.4f}")
        else:
            logger.info("=== Nenhum experimento atendeu aos critérios de desempenho ===")
            
            # Identificar o experimento que chegou mais perto
            closest_exp = max(results.items(), key=lambda x: (x[1]['precision'] >= 0.85) * x[1]['recall'])
            logger.info(f"Experimento mais próximo: {closest_exp[0]}")
            logger.info(f"Precision: {closest_exp[1]['precision']:.4f}, Recall: {closest_exp[1]['recall']:.4f}, F1: {closest_exp[1]['f1']:.4f}")
        
        # Log final
        logger.info("Experimentos concluídos. Consulte o relatório comparativo para detalhes.")
        
    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()