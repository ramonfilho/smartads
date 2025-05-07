#!/usr/bin/env python
"""
Script melhorado para calibração de probabilidades e ajuste de threshold.
Esta versão corrige problemas de compatibilidade entre o modelo treinado no MLflow 
e os dados de validação.

Foco em:
1. Carregar corretamente o modelo salvo pelo MLflow
2. Implementar calibração de probabilidades
3. Otimizar thresholds para equilibrar precisão e recall
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
import mlflow
import glob
import warnings
import re
import traceback

warnings.filterwarnings('ignore')

# Adicionar o diretório raiz do projeto ao path para importações corretas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# ======================================
# CONFIGURAÇÃO DE CAMINHOS
# ======================================
# Diretórios principais
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "03_3_feature_selection_text_code6")
MLFLOW_DIR = os.path.join(PROJECT_ROOT, "mlflow")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CANDIDATES_DIR = os.path.join(MODELS_DIR, "candidates")
RF_MODELS_DIR = os.path.join(CANDIDATES_DIR, "random_forest")
KMEANS_MODELS_DIR = os.path.join(CANDIDATES_DIR, "k_means")
GMM_MODELS_DIR = os.path.join(CANDIDATES_DIR, "gmm")
ARTIFACTS_DIR = os.path.join(MODELS_DIR, "artifacts")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Arquivos de dados
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
VALIDATION_DATA_PATH = os.path.join(DATA_DIR, "validation.csv")

# Output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(MODELS_DIR, f"calibrated_{timestamp}")

print(f"Configuração de caminhos:")
print(f"- Diretório raiz: {PROJECT_ROOT}")
print(f"- Diretório de dados: {DATA_DIR}")
print(f"- Diretório MLflow: {MLFLOW_DIR}")
print(f"- Diretório de modelos RF: {RF_MODELS_DIR}")
print(f"- Diretório de saída: {OUTPUT_DIR}")

# ======================================
# FUNÇÕES DE UTILIDADE
# ======================================

def setup_mlflow(mlflow_dir=MLFLOW_DIR, experiment_name="smart_ads_calibrated"):
    """Configura o MLflow para tracking de experimentos usando a API correta."""
    print(f"Configurando MLflow em: {mlflow_dir}")
    
    # Garantir que o diretório exista
    os.makedirs(mlflow_dir, exist_ok=True)
    
    # Configurar MLflow
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Usar a API correta do MLflow para gerenciar experimentos
    try:
        # Tentar obter o experimento pelo nome
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
            print(f"Usando experimento existente: {experiment_name} (ID: {experiment_id})")
        else:
            # Criar novo experimento se não existir
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Criado novo experimento: {experiment_name} (ID: {experiment_id})")
        
        return experiment_id
    except Exception as e:
        print(f"Erro ao configurar experimento MLflow: {e}")
        # Criar um experimento com nome baseado em timestamp como fallback
        fallback_name = f"{experiment_name}_{timestamp}"
        try:
            experiment_id = mlflow.create_experiment(fallback_name)
            print(f"Criado experimento fallback: {fallback_name} (ID: {experiment_id})")
            return experiment_id
        except Exception as e2:
            print(f"Erro ao criar experimento fallback: {e2}")
            return 0

def find_random_forest_model(models_dir=RF_MODELS_DIR, mlflow_dir=MLFLOW_DIR):
    """
    Busca o modelo Random Forest usando diferentes estratégias.
    Modificado para trabalhar com modelos salvos pelo MLflow.
    """
    print(f"Buscando modelo Random Forest...")
    
    # Estratégia 1: Verificar o diretório específico de modelos RF
    if os.path.exists(models_dir):
        print(f"Verificando diretório de modelos RF: {models_dir}")
        
        # Verificar se existe arquivo model.pkl (formato MLflow)
        model_pkl_path = os.path.join(models_dir, "model.pkl")
        if os.path.exists(model_pkl_path):
            print(f"Encontrado arquivo model.pkl (formato MLflow)")
            
            # Buscar threshold (geralmente armazenado separadamente)
            threshold_files = glob.glob(os.path.join(models_dir, "*threshold*.joblib"))
            if threshold_files:
                try:
                    threshold_file = sorted(threshold_files, key=os.path.getmtime, reverse=True)[0]
                    thresholds = joblib.load(threshold_file)
                    if isinstance(thresholds, dict) and 'rf_original' in thresholds:
                        threshold = thresholds['rf_original']
                    else:
                        threshold = 0.5
                except Exception:
                    threshold = 0.5
            else:
                threshold = 0.5
                
            print(f"Usando threshold: {threshold}")
            return model_pkl_path, threshold
        
        # Verificar arquivos joblib se não encontrou model.pkl
        rf_files = glob.glob(os.path.join(models_dir, "*.joblib"))
        model_files = [f for f in rf_files if all(x not in os.path.basename(f).lower() 
                                               for x in ["threshold", "config", "thresholds"])]
        
        if model_files:
            print(f"Encontrados {len(model_files)} arquivos de modelo RF:")
            for f in model_files:
                print(f"  - {os.path.basename(f)}")
            
            # Ordenar por data (mais recente primeiro)
            model_files.sort(key=os.path.getmtime, reverse=True)
            rf_model_path = model_files[0]
            print(f"Usando modelo mais recente: {os.path.basename(rf_model_path)}")
            
            # Verificar threshold
            threshold_files = glob.glob(os.path.join(models_dir, "*threshold*.joblib"))
            if threshold_files:
                try:
                    threshold_file = sorted(threshold_files, key=os.path.getmtime, reverse=True)[0]
                    thresholds = joblib.load(threshold_file)
                    if isinstance(thresholds, dict) and 'rf_original' in thresholds:
                        threshold = thresholds['rf_original']
                    else:
                        threshold = 0.5
                except Exception:
                    threshold = 0.5
            else:
                threshold = 0.5
                
            print(f"Usando threshold: {threshold}")
            return rf_model_path, threshold
    
    # Estratégia 2: Buscar no MLflow usando a API correta
    print(f"Buscando modelo no MLflow: {mlflow_dir}")
    
    # Definir tracking URI
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    try:
        # Usar a API do MLflow para buscar modelos
        client = mlflow.tracking.MlflowClient()
        
        # Listar experimentos
        experiments = client.search_experiments()
        
        for experiment in experiments:
            print(f"Verificando experimento: {experiment.name} (ID: {experiment.experiment_id})")
            
            # Buscar runs com modelos RF
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.model_type = 'random_forest'",
                order_by=["attribute.start_time DESC"]
            )
            
            if not runs:
                # Se não encontrar pela tag, buscar todas as runs
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["attribute.start_time DESC"]
                )
            
            for run in runs:
                try:
                    # Verificar se a run tem modelo de RF
                    artifacts = client.list_artifacts(run.info.run_id)
                    rf_artifact = None
                    
                    for artifact in artifacts:
                        if (artifact.is_dir and artifact.path == 'random_forest') or \
                           (not artifact.is_dir and artifact.path == 'model.pkl'):
                            rf_artifact = artifact
                            break
                    
                    if rf_artifact:
                        # Construir URI do modelo para carregamento
                        if rf_artifact.is_dir:
                            model_uri = f"runs:/{run.info.run_id}/random_forest"
                        else:
                            model_uri = f"runs:/{run.info.run_id}/model.pkl"
                        
                        # Obter threshold
                        threshold = run.data.metrics.get('threshold', 0.5)
                        
                        print(f"Encontrado modelo no MLflow (Run ID: {run.info.run_id})")
                        print(f"URI do modelo: {model_uri}")
                        print(f"Threshold: {threshold}")
                        
                        return model_uri, threshold
                except Exception as e:
                    print(f"Erro ao verificar run {run.info.run_id}: {e}")
                    continue
    except Exception as e:
        print(f"Erro ao buscar no MLflow: {e}")
    
    # Estratégia 3: Busca ampla no diretório de modelos
    print("Realizando busca ampla por modelos...")
    model_candidates = []
    
    # Procurar arquivos model.pkl
    model_candidates.extend(glob.glob(os.path.join(MODELS_DIR, "**", "model.pkl"), recursive=True))
    
    # Procurar arquivos MLmodel (parte da estrutura do MLflow)
    mlmodel_files = glob.glob(os.path.join(MODELS_DIR, "**", "MLmodel"), recursive=True)
    for mlmodel_file in mlmodel_files:
        model_dir = os.path.dirname(mlmodel_file)
        model_pkl = os.path.join(model_dir, "model.pkl")
        if os.path.exists(model_pkl):
            model_candidates.append(model_pkl)
    
    # Procurar arquivos .joblib relacionados ao RandomForest
    model_candidates.extend(glob.glob(os.path.join(MODELS_DIR, "**", "*random_forest*.joblib"), recursive=True))
    model_candidates.extend(glob.glob(os.path.join(MODELS_DIR, "**", "rf_*.joblib"), recursive=True))
    
    if model_candidates:
        print(f"Encontrados {len(model_candidates)} possíveis modelos:")
        for i, model_path in enumerate(model_candidates[:10]):
            rel_path = os.path.relpath(model_path, PROJECT_ROOT)
            print(f"  {i+1}. {rel_path}")
        
        if len(model_candidates) > 10:
            print(f"  ... e mais {len(model_candidates) - 10} modelos encontrados")
        
        # Ordenar por data (mais recente primeiro)
        model_candidates.sort(key=os.path.getmtime, reverse=True)
        model_path = model_candidates[0]
        rel_path = os.path.relpath(model_path, PROJECT_ROOT)
        print(f"Selecionando modelo mais recente: {rel_path}")
        
        return model_path, 0.5
    
    print("ERRO: Nenhum modelo Random Forest encontrado!")
    return None, None

def load_model_from_mlflow_or_file(model_path):
    """
    Carrega um modelo a partir do caminho especificado, 
    tratando tanto URIs do MLflow quanto arquivos locais.
    """
    print(f"Tentando carregar modelo de: {model_path}")
    
    try:
        if isinstance(model_path, str) and model_path.startswith("runs:/"):
            # É uma URI do MLflow - usar a API do MLflow para carregar
            model = mlflow.sklearn.load_model(model_path)
            print("Modelo carregado com sucesso via MLflow!")
        elif os.path.basename(model_path) == "model.pkl" and os.path.exists(os.path.join(os.path.dirname(model_path), "MLmodel")):
            # É um modelo MLflow salvo localmente
            model_dir = os.path.dirname(model_path)
            model = mlflow.sklearn.load_model(model_dir)
            print("Modelo carregado com sucesso do diretório MLflow local!")
        else:
            # Tentar carregar com joblib
            model = joblib.load(model_path)
            print("Modelo carregado com sucesso via joblib!")
        
        # Verificar se parece um modelo válido
        if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
            # Extrair informações sobre as features
            if hasattr(model, 'feature_names_in_'):
                print(f"Modelo usa {len(model.feature_names_in_)} features")
            else:
                print("Modelo não tem informação explícita sobre features")
            
            return model
        else:
            print("ERRO: O objeto carregado não parece ser um modelo válido (sem predict/predict_proba)")
            return None
            
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        traceback.print_exc()
        return None

def get_feature_names(model):
    """
    Extrai nomes de features do modelo de forma robusta.
    Tenta diferentes atributos comuns em modelos scikit-learn.
    """
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    elif hasattr(model, 'feature_names'):
        return list(model.feature_names)
    elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        # Tentar obter do primeiro estimador em modelos ensemble
        first_estimator = model.estimators_[0]
        if hasattr(first_estimator, 'feature_names_in_'):
            return list(first_estimator.feature_names_in_)
    
    # Se for um pipeline, tentar obter do último passo
    if hasattr(model, 'steps') and len(model.steps) > 0:
        last_step_name, last_step = model.steps[-1]
        if hasattr(last_step, 'feature_names_in_'):
            return list(last_step.feature_names_in_)
    
    # Se chegou aqui, não conseguimos obter os nomes das features
    print("AVISO: Não foi possível extrair nomes de features do modelo")
    return None

def load_and_preprocess_data(data_path, feature_names=None, target_col='target'):
    """
    Carrega e pré-processa os dados para garantir compatibilidade com o modelo.
    
    Args:
        data_path: Caminho para o arquivo de dados
        feature_names: Lista de nomes de features usados pelo modelo
        target_col: Nome da coluna target
        
    Returns:
        Tuple (DataFrame original, X preprocessado, y)
    """
    print(f"Carregando dados de: {data_path}")
    
    # Verificar se o arquivo existe
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
    
    try:
        # Carregar dados
        df = pd.read_csv(data_path)
        print(f"Dados carregados: {df.shape[0]} exemplos, {df.shape[1]} colunas")
        
        # Identificar coluna target
        if target_col not in df.columns:
            # Tentar encontrar coluna target
            target_candidates = ['target', 'label', 'class', 'converted', 'conversion']
            found_target = next((col for col in target_candidates if col in df.columns), None)
            
            if found_target:
                target_col = found_target
                print(f"Usando '{target_col}' como coluna target")
            else:
                print("Coluna target não encontrada. Usando última coluna.")
                target_col = df.columns[-1]
        
        # Separar target
        y = df[target_col]
        
        # Se não temos nomes de features do modelo, usar todas exceto target
        if feature_names is None:
            X = df.drop(columns=[target_col])
            print(f"Usando todas as {X.shape[1]} colunas exceto target")
            return df, X, y
        
        # Verificar quais features do modelo existem nos dados
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            print(f"ATENÇÃO: {len(missing_features)} features do modelo não estão presentes nos dados")
            if len(missing_features) <= 10:
                print(f"Features faltantes: {missing_features}")
            else:
                print(f"Primeiras 10 features faltantes: {missing_features[:10]}...")
            
            # Adicionar colunas faltantes preenchidas com zeros
            for feat in missing_features:
                df[feat] = 0
            print(f"Adicionadas {len(missing_features)} features faltantes preenchidas com zeros")
        
        # Criar X apenas com as features do modelo
        X = df[feature_names].copy()
        print(f"Dados preparados com exatamente as {len(feature_names)} features do modelo original")
        
        return df, X, y
    
    except Exception as e:
        print(f"Erro ao carregar/processar dados: {e}")
        traceback.print_exc()
        raise

def train_calibrated_model(X, y, base_model, cv=5, method='isotonic'):
    """
    Treina um modelo calibrado para corrigir probabilidades.
    
    Args:
        X: Features de treinamento
        y: Target de treinamento
        base_model: Modelo base a ser calibrado
        cv: Número de folds para validação cruzada
        method: Método de calibração ('isotonic' ou 'sigmoid')
        
    Returns:
        Modelo calibrado
    """
    print(f"\n=== Treinando modelo calibrado usando método '{method}' ===")
    print(f"Usando {cv} folds para validação cruzada")
    
    try:
        # Verificar se temos exemplos positivos suficientes para CV
        positive_count = y.sum()
        if positive_count < cv:
            print(f"AVISO: Apenas {positive_count} exemplos positivos para {cv} folds")
            print(f"Reduzindo número de folds para {max(2, positive_count//2)}")
            cv = max(2, positive_count//2)
        
        # Criar modelo calibrado
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv=cv,
            n_jobs=-1,
            ensemble=True  # Usar ensemble para maior robustez
        )
        
        # Treinar modelo
        calibrated_model.fit(X, y)
        
        print("Modelo calibrado treinado com sucesso")
        return calibrated_model
    
    except Exception as e:
        print(f"Erro ao treinar modelo calibrado: {e}")
        traceback.print_exc()
        return None

def optimize_threshold(y_true, y_probs, metric='f1', verbose=True):
    """
    Encontra o threshold ótimo para maximizar a métrica escolhida.
    
    Args:
        y_true: Valores reais
        y_probs: Probabilidades preditas
        metric: Métrica a otimizar ('f1', 'precision', 'recall')
        verbose: Se deve imprimir informações
        
    Returns:
        Dicionário com threshold ótimo e métricas
    """
    if verbose:
        print(f"\n=== Otimizando threshold para maximizar {metric} ===")
    
    # Calcular curva precision-recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Calcular F1 para cada threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Encontrar threshold ótimo baseado na métrica escolhida
    if metric == 'f1':
        best_idx = np.argmax(f1_scores)
    elif metric == 'precision':
        # Para precision, queremos o maior threshold que atinja pelo menos 0.5 de recall
        valid_recall_idx = np.where(recall >= 0.5)[0]
        if len(valid_recall_idx) > 0:
            best_idx = valid_recall_idx[np.argmax(precision[valid_recall_idx])]
        else:
            best_idx = np.argmax(precision)
    elif metric == 'recall':
        # Para recall, queremos o menor threshold que atinja pelo menos 0.5 de precision
        valid_precision_idx = np.where(precision >= 0.5)[0]
        if len(valid_precision_idx) > 0:
            best_idx = valid_precision_idx[np.argmax(recall[valid_precision_idx])]
        else:
            best_idx = np.argmax(recall)
    else:
        # Padrão: otimizar F1
        best_idx = np.argmax(f1_scores)
    
    # Para evitar index out of bounds se best_idx == len(thresholds)
    if best_idx >= len(thresholds):
        best_threshold = 0.5 if best_idx > 0 else 0.0
    else:
        best_threshold = thresholds[best_idx]
    
    # Se o best_idx está no último elemento, usar um threshold de 0.0
    if best_idx == 0 and len(thresholds) > 0:
        best_threshold = 0.0
    
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    
    if verbose:
        print(f"Threshold ótimo: {best_threshold:.4f}")
        print(f"  - Precision: {best_precision:.4f}")
        print(f"  - Recall: {best_recall:.4f}")
        print(f"  - F1-score: {best_f1:.4f}")
    
    # Encontrar threshold para alto recall (ao menos 0.7)
    high_recall_indices = np.where(recall >= 0.7)[0]
    if len(high_recall_indices) > 0:
        hr_idx = high_recall_indices[0]  # Primeiro threshold que atinge recall 0.7
        hr_threshold = thresholds[hr_idx] if hr_idx < len(thresholds) else 0.01
        hr_precision = precision[hr_idx]
        hr_recall = recall[hr_idx]
        hr_f1 = 2 * (hr_precision * hr_recall) / (hr_precision + hr_recall + 1e-10)
        
        if verbose:
            print(f"\nThreshold para alto recall (>= 70%): {hr_threshold:.4f}")
            print(f"  - Precision: {hr_precision:.4f}")
            print(f"  - Recall: {hr_recall:.4f}")
            print(f"  - F1-score: {hr_f1:.4f}")
    else:
        hr_threshold = None
        hr_precision = None
        hr_recall = None
        hr_f1 = None
    
    return {
        'threshold': best_threshold,
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1,
        'high_recall_threshold': hr_threshold,
        'high_recall_precision': hr_precision,
        'high_recall_recall': hr_recall, 
        'high_recall_f1': hr_f1,
        'all_thresholds': thresholds,
        'all_precision': precision,
        'all_recall': recall,
        'all_f1': f1_scores
    }

def plot_calibration_curve(y_true, y_prob_uncalibrated, y_prob_calibrated, filename=None):
    """
    Plota a curva de calibração que mostra a relação entre probabilidade 
    prevista e frequência observada.
    
    Args:
        y_true: Valores verdadeiros (0/1)
        y_prob_uncalibrated: Probabilidades do modelo não calibrado
        y_prob_calibrated: Probabilidades do modelo calibrado
        filename: Caminho para salvar o gráfico
    """
    from sklearn.calibration import calibration_curve
    
    plt.figure(figsize=(10, 8))
    
    # Curva de calibração perfeita (diagonal)
    plt.plot([0, 1], [0, 1], 'k--', label='Calibração perfeita')
    
    # Modelo não calibrado
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, y_prob_uncalibrated, n_bins=10)
    plt.plot(prob_pred_uncal, prob_true_uncal, 's-', label='Modelo não calibrado')
    
    # Modelo calibrado
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=10)
    plt.plot(prob_pred_cal, prob_true_cal, 'o-', label='Modelo calibrado')
    
    plt.xlabel('Probabilidade prevista')
    plt.ylabel('Fração de positivos (frequência observada)')
    plt.title('Curva de Calibração')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    if filename:
        plt.savefig(filename)
        print(f"Curva de calibração salva em: {filename}")
    
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, thresholds_result, title, filename=None):
    """
    Plota a curva precision-recall com os thresholds otimizados.
    
    Args:
        y_true: Valores verdadeiros
        y_prob: Probabilidades preditas
        thresholds_result: Resultado da otimização de thresholds
        title: Título do gráfico
        filename: Caminho para salvar o gráfico
    """
    plt.figure(figsize=(10, 8))
    
    # Curva precision-recall
    plt.plot(thresholds_result['all_recall'], thresholds_result['all_precision'], 'b-', label='Precision-Recall')
    
    # Marcar threshold ótimo
    plt.scatter(
        [thresholds_result['recall']], 
        [thresholds_result['precision']], 
        c='red', marker='o', s=100,
        label=f"Threshold ótimo F1: {thresholds_result['threshold']:.3f}"
    )
    
    # Marcar threshold alto recall (se disponível)
    if thresholds_result['high_recall_threshold'] is not None:
        plt.scatter(
            [thresholds_result['high_recall_recall']], 
            [thresholds_result['high_recall_precision']], 
            c='green', marker='s', s=100,
            label=f"Threshold alto recall: {thresholds_result['high_recall_threshold']:.3f}"
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    if filename:
        plt.savefig(filename)
        print(f"Curva precision-recall salva em: {filename}")
    
    plt.close()

def evaluate_model(y_true, y_prob, threshold, model_name="Modelo"):
    """
    Avalia o modelo usando o threshold especificado.
    
    Args:
        y_true: Valores verdadeiros
        y_prob: Probabilidades preditas
        threshold: Threshold a aplicar
        model_name: Nome do modelo para exibição
        
    Returns:
        Dicionário com métricas
    """
    print(f"\n=== Avaliando {model_name} (threshold={threshold:.4f}) ===")
    
    # Aplicar threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calcular métricas
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Contar predições
    n_positives = np.sum(y_pred)
    pos_rate = n_positives / len(y_pred) if len(y_pred) > 0 else 0
    
    # Falsos positivos e negativos
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    print(f"Métricas para {model_name}:")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-score: {f1:.4f}")
    print(f"  - Taxa de positivos: {pos_rate:.2%} ({n_positives} de {len(y_pred)})")
    print(f"  - Falsos positivos: {fp}")
    print(f"  - Falsos negativos: {fn}")
    
    results = {
        'model_name': model_name,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'positives': int(n_positives),
        'positives_rate': float(pos_rate),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    return results

def save_models(models_dict, thresholds_dict, output_dir):
    """
    Salva os modelos treinados e seus thresholds.
    Suporta tanto formato MLflow quanto joblib.
    
    Args:
        models_dict: Dicionário com modelos para salvar
        thresholds_dict: Dicionário com thresholds para salvar
        output_dir: Diretório de saída
        
    Returns:
        Diretório onde os modelos foram salvos
    """
    print(f"\n=== Salvando modelos em {output_dir} ===")
    
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar cada modelo
    for name, model in models_dict.items():
        # Salvar como joblib para facilitar o uso posterior
        model_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"  - Modelo {name} salvo como joblib em: {os.path.basename(model_path)}")
        
        # Salvar também no formato MLflow para consistência
        mlflow_dir = os.path.join(output_dir, name)
        os.makedirs(mlflow_dir, exist_ok=True)
        try:
            mlflow.sklearn.save_model(model, mlflow_dir)
            print(f"  - Modelo {name} salvo no formato MLflow em: {name}/")
        except Exception as e:
            print(f"  - Aviso: Não foi possível salvar no formato MLflow: {e}")
    
    # Salvar thresholds
    thresholds_path = os.path.join(output_dir, "thresholds.joblib")
    joblib.dump(thresholds_dict, thresholds_path)
    print(f"  - Thresholds salvos em: {os.path.basename(thresholds_path)}")
    
    # Salvar configuração
    config = {
        'models': list(models_dict.keys()),
        'thresholds': thresholds_dict,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_path = os.path.join(output_dir, "config.joblib")
    joblib.dump(config, config_path)
    print(f"  - Configuração salva em: {os.path.basename(config_path)}")
    
    # Criar um README com instruções de uso
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# Modelos Calibrados com Threshold Otimizado\n\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Modelos disponíveis\n\n")
        for name in models_dict.keys():
            f.write(f"- **{name}**: `{name}.joblib` ou diretório MLflow `{name}/`\n")
        f.write("\n## Thresholds\n\n")
        f.write("Arquivo: `thresholds.joblib`\n\n")
        for name, threshold in thresholds_dict.items():
            f.write(f"- **{name}**: {threshold:.4f}\n")
        f.write("\n## Como usar\n\n")
        f.write("```python\n")
        f.write("# Opção 1: Carregar com joblib\n")
        f.write("import joblib\n\n")
        f.write("# Carregar modelo\n")
        f.write("model = joblib.load('nome_do_modelo.joblib')\n\n")
        f.write("# Carregar thresholds\n")
        f.write("thresholds = joblib.load('thresholds.joblib')\n")
        f.write("threshold = thresholds['nome_do_modelo']\n\n")
        f.write("# Fazer predições\n")
        f.write("probs = model.predict_proba(X)[:, 1]\n")
        f.write("predictions = (probs >= threshold).astype(int)\n")
        f.write("\n# Opção 2: Carregar com MLflow\n")
        f.write("import mlflow.sklearn\n\n")
        f.write("# Carregar modelo\n")
        f.write("model = mlflow.sklearn.load_model('diretório_do_modelo')\n\n")
        f.write("# Fazer predições\n") 
        f.write("probs = model.predict_proba(X)[:, 1]\n")
        f.write("predictions = (probs >= threshold).astype(int)\n")
        f.write("```\n")
    
    print(f"  - README com instruções salvo em: README.md")
    
    return output_dir

# ======================================
# FUNÇÃO PRINCIPAL
# ======================================

def main():
    parser = argparse.ArgumentParser(description='Implementa calibração e ajuste de threshold robusta')
    parser.add_argument('--data_dir', default=DATA_DIR,
                        help='Diretório com os arquivos de dados')
    parser.add_argument('--models_dir', default=RF_MODELS_DIR,
                        help='Diretório para buscar modelos')
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                        help='Diretório para salvar os modelos calibrados')
    parser.add_argument('--mlflow_dir', default=MLFLOW_DIR,
                        help='Diretório do MLflow tracking')
    parser.add_argument('--calibration_method', default='isotonic', choices=['isotonic', 'sigmoid'],
                        help='Método de calibração (isotonic ou sigmoid)')
    parser.add_argument('--cv', type=int, default=5,
                        help='Número de folds para validação cruzada na calibração')
    
    args = parser.parse_args()
    
    # Definir caminhos de arquivos baseados nos diretórios fornecidos
    train_path = os.path.join(args.data_dir, "train.csv")
    validation_path = os.path.join(args.data_dir, "validation.csv")
    
    # Verificar se os caminhos existem
    for path, desc in [
        (train_path, "treinamento"), 
        (validation_path, "validação")
    ]:
        if not os.path.exists(path):
            print(f"AVISO: Caminho de {desc} não encontrado: {path}")
            if desc in ["treinamento", "validação"]:
                print(f"ERRO: O arquivo de {desc} é necessário para continuar.")
                sys.exit(1)
    
    try:
        # 1. Configurar MLflow
        experiment_id = setup_mlflow(args.mlflow_dir, "smart_ads_calibrated")
        
        # 2. Iniciar run do MLflow
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            print(f"MLflow run iniciado: {run_id}")
            
            # Registrar parâmetros
            mlflow.log_params({
                'train_data': train_path,
                'validation_data': validation_path,
                'models_dir': args.models_dir,
                'calibration_method': args.calibration_method,
                'cv_folds': args.cv
            })
            
            # 3. Buscar e carregar o modelo RandomForest
            print("\n=== Etapa 1: Buscando modelo Random Forest ===")
            rf_model_path, rf_threshold = find_random_forest_model(args.models_dir, args.mlflow_dir)
            
            if not rf_model_path:
                print("ERRO: Não foi possível encontrar um modelo Random Forest.")
                return
            
            # 4. Carregar o modelo
            print("\n=== Etapa 2: Carregando modelo ===")
            rf_model = load_model_from_mlflow_or_file(rf_model_path)
            
            if not rf_model:
                print("ERRO: Falha ao carregar o modelo Random Forest.")
                return
            
            # 5. Extrair nomes de features do modelo
            print("\n=== Etapa 3: Extraindo informações de features do modelo ===")
            feature_names = get_feature_names(rf_model)
            
            if not feature_names:
                print("AVISO: Não foi possível extrair nomes de features do modelo.")
                print("Utilizaremos todas as features exceto o target nos dados.")
            else:
                print(f"Extraídas {len(feature_names)} features do modelo")
                
                # Mostrar algumas features para depuração
                if len(feature_names) > 10:
                    print(f"Exemplos de features: {feature_names[:10]}...")
                else:
                    print(f"Features: {feature_names}")
            
            # 6. Carregar dados com as features corretas
            print("\n=== Etapa 4: Carregando e pré-processando dados ===")
            _, X_train, y_train = load_and_preprocess_data(train_path, feature_names)
            _, X_val, y_val = load_and_preprocess_data(validation_path, feature_names)
            
            # 7. Avaliar modelo original
            print("\n=== Etapa 5: Avaliando modelo original ===")
            original_probs = rf_model.predict_proba(X_val)[:, 1]
            original_threshold_results = optimize_threshold(y_val, original_probs)
            
            # Se o threshold carregado for muito diferente do ótimo, avisar
            if abs(rf_threshold - original_threshold_results['threshold']) > 0.1:
                print(f"AVISO: Threshold carregado ({rf_threshold:.4f}) é diferente do ótimo encontrado ({original_threshold_results['threshold']:.4f})")
                print("Usando threshold carregado para manter consistência com o modelo original")
            
            # Avaliar com o threshold original
            original_results = evaluate_model(
                y_val, original_probs, rf_threshold, 
                model_name="Random Forest Original"
            )
            
            # Registrar métricas no MLflow
            mlflow.log_metrics({
                'original_precision': original_results['precision'],
                'original_recall': original_results['recall'],
                'original_f1': original_results['f1'],
                'original_threshold': rf_threshold
            })
            
            # 8. Treinar modelo calibrado
            print("\n=== Etapa 6: Treinando modelo calibrado ===")
            calibrated_model = train_calibrated_model(
                X_train, y_train, rf_model, 
                cv=args.cv, 
                method=args.calibration_method
            )
            
            if calibrated_model is None:
                print("ERRO: Falha ao treinar modelo calibrado.")
                return
            
            # 9. Avaliar modelo calibrado
            print("\n=== Etapa 7: Avaliando modelo calibrado ===")
            calibrated_probs = calibrated_model.predict_proba(X_val)[:, 1]
            
            # Avaliar com o threshold original
            calibrated_results_original_threshold = evaluate_model(
                y_val, calibrated_probs, rf_threshold,
                model_name="Random Forest Calibrado (threshold original)"
            )
            
            # Otimizar threshold para o modelo calibrado
            calibrated_threshold_results = optimize_threshold(y_val, calibrated_probs)
            calibrated_optimal_threshold = calibrated_threshold_results['threshold']
            
            # Avaliar com o threshold otimizado
            calibrated_results_optimal_threshold = evaluate_model(
                y_val, calibrated_probs, calibrated_optimal_threshold,
                model_name="Random Forest Calibrado (threshold otimizado)"
            )
            
            # Registrar métricas no MLflow
            mlflow.log_metrics({
                'calibrated_original_threshold_precision': calibrated_results_original_threshold['precision'],
                'calibrated_original_threshold_recall': calibrated_results_original_threshold['recall'],
                'calibrated_original_threshold_f1': calibrated_results_original_threshold['f1'],
                'calibrated_optimal_threshold': calibrated_optimal_threshold,
                'calibrated_optimal_threshold_precision': calibrated_results_optimal_threshold['precision'],
                'calibrated_optimal_threshold_recall': calibrated_results_optimal_threshold['recall'],
                'calibrated_optimal_threshold_f1': calibrated_results_optimal_threshold['f1']
            })
            
            # 10. Gerar visualizações
            print("\n=== Etapa 8: Gerando visualizações ===")
            
            # Criar diretório de plots
            os.makedirs(PLOTS_DIR, exist_ok=True)
            
            # Curva de calibração
            calibration_plot_path = os.path.join(PLOTS_DIR, f"calibration_curve_{run_id}.png")
            plot_calibration_curve(
                y_val, original_probs, calibrated_probs, 
                filename=calibration_plot_path
            )
            mlflow.log_artifact(calibration_plot_path)
            
            # Curva precision-recall modelo original
            pr_plot_original_path = os.path.join(PLOTS_DIR, f"pr_curve_original_{run_id}.png")
            plot_precision_recall_curve(
                y_val, original_probs, original_threshold_results,
                title="Curva Precision-Recall (Modelo Original)",
                filename=pr_plot_original_path
            )
            mlflow.log_artifact(pr_plot_original_path)
            
            # Curva precision-recall modelo calibrado
            pr_plot_calibrated_path = os.path.join(PLOTS_DIR, f"pr_curve_calibrated_{run_id}.png")
            plot_precision_recall_curve(
                y_val, calibrated_probs, calibrated_threshold_results,
                title="Curva Precision-Recall (Modelo Calibrado)",
                filename=pr_plot_calibrated_path
            )
            mlflow.log_artifact(pr_plot_calibrated_path)
            
            # 11. Salvar modelos
            print("\n=== Etapa 9: Salvando modelos ===")
            
            # Preparar modelos para salvar
            models_to_save = {
                'rf_original': rf_model,
                'rf_calibrated': calibrated_model
            }
            
            # Preparar thresholds para salvar
            thresholds_to_save = {
                'rf_original': rf_threshold,
                'rf_calibrated_original': rf_threshold,
                'rf_calibrated_optimal': calibrated_optimal_threshold,
                'rf_calibrated_high_recall': calibrated_threshold_results['high_recall_threshold']
            }
            
            # Salvar modelos e thresholds
            output_dir = save_models(
                models_to_save, thresholds_to_save, 
                args.output_dir
            )
            
            # Registrar modelos no MLflow
            input_example = X_train.iloc[:5].copy() if len(X_train) >= 5 else X_train.copy()
            
            # Registrar modelo original
            mlflow.sklearn.log_model(
                rf_model, 
                "rf_original_model", 
                input_example=input_example
            )
            
            # Registrar modelo calibrado
            mlflow.sklearn.log_model(
                calibrated_model, 
                "rf_calibrated_model", 
                input_example=input_example
            )
            
            # 12. Resumo dos resultados
            print("\n=== Etapa 10: Resumo dos resultados ===")
            
            # Criar DataFrame de resumo
            results_data = [
                original_results,
                calibrated_results_original_threshold,
                calibrated_results_optimal_threshold
            ]
            
            results_df = pd.DataFrame(results_data)
            
            # Mostrar resumo
            print("\nComparação de performance:")
            display_cols = ['model_name', 'threshold', 'precision', 'recall', 'f1', 'positives_rate']
            print(results_df[display_cols])
            
            # Salvar resumo
            results_path = os.path.join(args.output_dir, "results_summary.csv")
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path)
            
            print("\n=== Calibração e ajuste de threshold concluídos com sucesso! ===")
            print(f"Modelos salvos em: {output_dir}")
            print(f"ID do MLflow run: {run_id}")
            
            return output_dir, run_id
    
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()