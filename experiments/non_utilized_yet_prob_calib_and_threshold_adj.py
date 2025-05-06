#!/usr/bin/env python
"""
Script para implementação de calibração de probabilidades e ajuste de threshold.
Este script implementa as fases 1 e 2 do plano de melhoria para o modelo Smart Ads.

Fase 1: Modelo Calibrado
   - Implementar calibração de probabilidades para o modelo baseline

Fase 2: Ajuste de Threshold
   - Otimizar o threshold para equilibrar precisão e recall
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
import re  # Importado para função de sanitização
warnings.filterwarnings('ignore')

# ======================================
# CONFIGURAÇÃO DE CAMINHOS
# ======================================
# Diretórios principais
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "03_3_feature_selection_text_code6")
MLFLOW_DIR = os.path.join(PROJECT_ROOT, "mlflow")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARTIFACTS_DIR = os.path.join(MODELS_DIR, "artifacts")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Arquivos de dados
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
VALIDATION_DATA_PATH = os.path.join(DATA_DIR, "validation.csv")

# Output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(MODELS_DIR, f"calibrated_{timestamp}")

# Adicionar diretório raiz ao path
sys.path.append(PROJECT_ROOT)
print(f"Diretório raiz adicionado ao path: {PROJECT_ROOT}")
print(f"Diretório de dados: {DATA_DIR}")
print(f"Diretório MLflow: {MLFLOW_DIR}")
print(f"Diretório de artefatos: {ARTIFACTS_DIR}")

# ======================================
# FUNÇÕES IMPORTADAS DE OUTROS MÓDULOS
# ======================================

def sanitize_column_names(df):
    """
    Sanitize column names to avoid issues with special characters.
    
    Importado de src/evaluation/feature_importance.py (usado no script 03)
    
    Args:
        df: DataFrame with columns to sanitize
        
    Returns:
        Dictionary mapping original column names to sanitized names
    """
    print("Sanitizando nomes das features para evitar problemas com caracteres especiais...")
    rename_dict = {}
    
    for col in df.columns:
        # Substituir caracteres especiais e espaços por underscores
        new_col = re.sub(r'[^0-9a-zA-Z_]', '_', col)
        # Garantir que não comece com número
        if new_col[0].isdigit():
            new_col = 'f_' + new_col
        # Verificar se já existe esse novo nome
        i = 1
        temp_col = new_col
        while temp_col in rename_dict.values():
            temp_col = f"{new_col}_{i}"
            i += 1
        new_col = temp_col
        
        # Só adicionar ao dicionário se o nome mudou
        if col != new_col:
            rename_dict[col] = new_col
    
    return rename_dict


def get_latest_random_forest_run(mlflow_dir):
    """
    Obtém o run_id do modelo RandomForest mais recente de forma mais robusta.
    """
    print(f"Procurando o melhor modelo Random Forest em: {mlflow_dir}")
    
    # Configurar MLflow
    if os.path.exists(mlflow_dir):
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    else:
        print(f"AVISO: Diretório MLflow não encontrado: {mlflow_dir}")
        return None, None, None
    
    try:
        # Busca alternativa: procurar modelos diretamente por arquivo
        model_files = glob.glob(os.path.join(mlflow_dir, "**", "*random_forest*.joblib"), recursive=True)
        model_files += glob.glob(os.path.join(mlflow_dir, "**", "random_forest", "model.pkl"), recursive=True)
        
        if model_files:
            print(f"Encontrados {len(model_files)} arquivos de modelo Random Forest")
            # Ordenar por data de modificação (mais recente primeiro)
            model_files.sort(key=os.path.getmtime, reverse=True)
            
            run_id = "modelo_encontrado_manualmente"
            threshold = 0.12  # Valor padrão baseado em execuções anteriores
            model_path = model_files[0]
            
            print(f"Usando arquivo de modelo mais recente: {model_path}")
            print(f"Threshold padrão: {threshold}")
            
            return run_id, threshold, model_path
            
        # Se nenhum arquivo foi encontrado, buscar em todo o projeto
        print("Procurando em todo o projeto...")
        project_files = glob.glob(os.path.join("/Users/ramonmoreira/desktop/smart_ads", "**", "*random_forest*.joblib"), recursive=True)
        project_files += glob.glob(os.path.join("/Users/ramonmoreira/desktop/smart_ads", "**", "random_forest", "model.pkl"), recursive=True)
        
        if project_files:
            print(f"Encontrados {len(project_files)} arquivos de modelo Random Forest no projeto")
            # Ordenar por data de modificação (mais recente primeiro)
            project_files.sort(key=os.path.getmtime, reverse=True)
            
            run_id = "modelo_encontrado_projeto"
            threshold = 0.12  # Valor padrão baseado em execuções anteriores
            model_path = project_files[0]
            
            print(f"Usando arquivo de modelo mais recente: {model_path}")
            print(f"Threshold padrão: {threshold}")
            
            return run_id, threshold, model_path
            
        print("Nenhum modelo RandomForest encontrado.")
        return None, None, None
        
    except Exception as e:
        print(f"Erro ao buscar modelos: {e}")
        return None, None, None


def load_model_from_mlflow(model_uri):
    """
    Carrega um modelo a partir do MLflow usando seu URI.
    
    Importado de scripts/05_error_analysis_pipeline.py
    
    Args:
        model_uri: URI do modelo no formato 'runs:/<run_id>/<artifact_path>'
        
    Returns:
        Modelo carregado ou None se falhar
    """
    try:
        print(f"Carregando modelo de: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        return None

# ======================================
# FUNÇÕES DE UTILIDADE
# ======================================

def setup_mlflow(mlflow_dir=MLFLOW_DIR, experiment_name="smart_ads_calibrated"):
    """Configura o MLflow para tracking de experimentos."""
    print(f"Configurando MLflow em: {mlflow_dir}")
    
    # Configurar MLflow
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Tentar diretamente criar o experimento sem verificar existência
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experimento '{experiment_name}' criado (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            # Se já existe, precisamos encontrá-lo manualmente
            print(f"Experimento '{experiment_name}' já existe")
            
            # Abordagem alternativa: listar diretórios de experimento manualmente
            experiment_dirs = [d for d in os.listdir(mlflow_dir) 
                               if os.path.isdir(os.path.join(mlflow_dir, d)) and d.isdigit()]
            
            if experiment_dirs:
                # Use o primeiro experimento disponível
                experiment_id = experiment_dirs[0]
                print(f"Usando experimento existente (ID: {experiment_id})")
            else:
                print("Nenhum experimento válido encontrado. Usando ID padrão.")
                experiment_id = "0"
        else:
            # Outro erro, usar ID padrão
            print(f"Erro ao acessar experimento: {e}")
            print("Usando experimento padrão (ID: 0)")
            experiment_id = "0"
    
    return experiment_id

def find_best_random_forest_model(mlflow_dir=MLFLOW_DIR):
    """
    Encontra o modelo Random Forest com melhor desempenho no MLflow.
    Modificado para usar a função importada get_latest_random_forest_run
    """
    print(f"Procurando o melhor modelo Random Forest em: {mlflow_dir}")
    
    # Usar a função importada de 05_error_analysis_pipeline.py
    run_id, threshold, model_uri = get_latest_random_forest_run(mlflow_dir)
    
    if model_uri:
        return model_uri, threshold
    
    # Se a função não encontrou nada, tentar busca alternativa como fallback
    try:
        # Opção alternativa: se não conseguirmos encontrar um modelo no MLflow,
        # vamos procurar diretamente nos arquivos do diretório mlflow
        model_files = glob.glob(os.path.join(mlflow_dir, "**", "*random_forest*.joblib"), recursive=True)
        if model_files:
            print(f"Encontrados {len(model_files)} modelos Random Forest nos arquivos:")
            for file in model_files:
                print(f"  - {file}")
            
            # Ordenar por data de modificação (mais recente primeiro)
            model_files.sort(key=os.path.getmtime, reverse=True)
            best_model_path = model_files[0]
            print(f"Selecionado modelo Random Forest mais recente: {best_model_path}")
            return best_model_path, 0.5  # Usamos um threshold padrão
        
        # Se ainda não encontrou nada, não há modelo disponível
        print("Nenhum modelo encontrado em nenhuma busca.")
        return None, 0.5
    
    except Exception as e:
        print(f"Erro ao buscar melhor modelo Random Forest: {str(e)}")
        # A busca falhou completamente
        return None, 0.5

def find_best_cluster_model(artifacts_dir=ARTIFACTS_DIR, model_type="kmeans"):
    """Encontra o melhor modelo de cluster (K-means ou GMM) nos artefatos."""
    print(f"Procurando o melhor modelo {model_type} em: {artifacts_dir}")
    
    # Definir diretórios específicos para cada tipo de modelo
    specific_dirs = {
        "kmeans": "/Users/ramonmoreira/desktop/smart_ads/models/candidates/k_means",
        "gmm": "/Users/ramonmoreira/desktop/smart_ads/models/candidates/gmm"
    }
    
    try:
        # Primeiro tenta buscar nos diretórios específicos
        if model_type.lower() in specific_dirs:
            specific_dir = specific_dirs[model_type.lower()]
            print(f"Verificando diretório específico: {specific_dir}")
            
            if os.path.exists(specific_dir):
                # Definir padrões de busca específicos para cada tipo
                if model_type.lower() == 'kmeans':
                    # Priorizar o modelo principal, não os modelos de cluster individual
                    patterns = [
                        os.path.join(specific_dir, "kmeans_model.joblib"),
                        os.path.join(specific_dir, "best_kmeans.joblib"),
                        os.path.join(specific_dir, "*kmeans*.joblib")
                    ]
                elif model_type.lower() == 'gmm':
                    # Priorizar o modelo principal, não o scaler ou modelos de cluster individual
                    patterns = [
                        os.path.join(specific_dir, "gmm_model.joblib"), 
                        os.path.join(specific_dir, "*gmm*.joblib")
                    ]
                
                # Tentar cada padrão na ordem
                for pattern in patterns:
                    if '*' in pattern:
                        # Padrão com wildcard
                        model_files = glob.glob(pattern)
                    else:
                        # Arquivo específico
                        model_files = [pattern] if os.path.exists(pattern) else []
                    
                    if model_files:
                        print(f"Modelos {model_type} encontrados usando padrão {pattern}:")
                        for file in model_files:
                            print(f"  - {file}")
                        
                        # Filtrar arquivos indesejados (como scaler_model.joblib para GMM)
                        if model_type.lower() == 'gmm':
                            filtered_files = [f for f in model_files if 'scaler_model.joblib' not in f]
                            if filtered_files:
                                model_files = filtered_files
                                print(f"Filtrando arquivos para evitar carregar o scaler em vez do modelo")
                        
                        if model_files:
                            # Ordenar por data de modificação (mais recente primeiro)
                            model_files.sort(key=os.path.getmtime, reverse=True)
                            
                            # Retornar o arquivo mais recente
                            best_model_path = model_files[0]
                            print(f"Selecionado modelo {model_type} mais recente: {best_model_path}")
                            return best_model_path
                
                print(f"Nenhum modelo apropriado encontrado com os padrões definidos")
            else:
                print(f"Diretório específico não encontrado: {specific_dir}")
        
        # Tenta busca padrão em artifacts_dir
        print(f"Tentando encontrar o modelo em: {artifacts_dir}")
        if os.path.exists(artifacts_dir):
            # Padrão de busca para cada tipo de modelo
            if model_type.lower() == 'kmeans':
                pattern = os.path.join(artifacts_dir, "**", "*kmeans*.joblib")
            elif model_type.lower() == 'gmm':
                pattern = os.path.join(artifacts_dir, "**", "*gmm*.joblib")
            else:
                print(f"Tipo de modelo não suportado: {model_type}")
                return None
            
            # Buscar todos os arquivos que correspondem ao padrão
            model_files = glob.glob(pattern, recursive=True)
            
            if model_files:
                print(f"Modelos {model_type} encontrados em artifacts_dir:")
                for file in model_files:
                    print(f"  - {file}")
                
                # Filtrar arquivos indesejados
                if model_type.lower() == 'gmm':
                    filtered_files = [f for f in model_files if 'scaler_model.joblib' not in f]
                    if filtered_files:
                        model_files = filtered_files
                        print(f"Filtrando arquivos para evitar carregar o scaler em vez do modelo")
                
                # Ordenar por data de modificação (mais recente primeiro)
                model_files.sort(key=os.path.getmtime, reverse=True)
                
                # Retornar o arquivo mais recente
                best_model_path = model_files[0]
                print(f"Selecionado modelo {model_type} mais recente: {best_model_path}")
                return best_model_path
        
        # Busca ampla em todo o sistema de arquivos do projeto
        print(f"Realizando busca ampla por modelos {model_type} em todo o projeto...")
        project_root = "/Users/ramonmoreira/desktop/smart_ads"
        
        if model_type.lower() == 'kmeans':
            pattern = os.path.join(project_root, "**", "*kmeans*.joblib")
        elif model_type.lower() == 'gmm':
            pattern = os.path.join(project_root, "**", "*gmm*.joblib")
        
        # Buscar todos os arquivos que correspondem ao padrão
        model_files = glob.glob(pattern, recursive=True)
        
        if model_files:
            print(f"Modelos {model_type} encontrados na busca ampla:")
            for file in model_files:
                print(f"  - {file}")
            
            # Filtrar arquivos indesejados
            if model_type.lower() == 'gmm':
                filtered_files = [f for f in model_files if 'scaler_model.joblib' not in f]
                if filtered_files:
                    model_files = filtered_files
                    print(f"Filtrando arquivos para evitar carregar o scaler em vez do modelo")
            
            # Ordenar por data de modificação (mais recente primeiro)
            model_files.sort(key=os.path.getmtime, reverse=True)
            
            # Retornar o arquivo mais recente
            best_model_path = model_files[0]
            print(f"Selecionado modelo {model_type} mais recente: {best_model_path}")
            return best_model_path
        
        print(f"Nenhum modelo {model_type} encontrado em nenhum diretório.")
        return None
    
    except Exception as e:
        print(f"Erro ao buscar modelo {model_type}: {str(e)}")
        return None

def load_model(model_path):
    """
    Carrega um modelo treinado a partir do caminho especificado.
    Modificado para usar a função importada load_model_from_mlflow
    """
    try:
        print(f"Carregando modelo de: {model_path}")
        if isinstance(model_path, str) and model_path.startswith("runs:"):
            # Carregar modelo do MLflow usando a função importada
            model = load_model_from_mlflow(model_path)
        else:
            # Carregar modelo de arquivo
            model = joblib.load(model_path)
        
        if model is not None:
            print("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        return None

def load_data(data_path, verbose=True, maintain_feature_names=None, fill_missing=True):
    """
    Carrega dados de treinamento/validação.
    
    Args:
        data_path: Caminho para o arquivo de dados
        verbose: Se deve imprimir informações
        maintain_feature_names: Lista de nomes de features para manter (sem sanitizar)
        fill_missing: Se deve preencher features faltantes com zeros
        
    Returns:
        DataFrame original, features, target, nome da coluna target
    """
    # Verificar se o arquivo existe no caminho fornecido
    if not os.path.exists(data_path):
        print(f"ERRO: Arquivo não encontrado: {data_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
    
    try:
        # Função simplificada para carregar dados
        df = pd.read_csv(data_path)
        
        # Identificar coluna target antes da sanitização
        target_candidates = ['target', 'label', 'class', 'y', 'converted']
        target_col = next((col for col in target_candidates if col in df.columns), None)
        
        if target_col is None:
            print("Coluna target não encontrada. Usando última coluna.")
            target_col = df.columns[-1]
            
        # Se foi fornecida uma lista de nomes de features para manter, usar essa
        if maintain_feature_names is not None:
            # Verificar se todas as colunas necessárias existem no DataFrame
            missing_cols = [col for col in maintain_feature_names if col not in df.columns]
            if missing_cols and verbose:
                print(f"INFO: {len(missing_cols)} colunas do modelo não existem no DataFrame original")
                
                # Tentar normalizar os nomes das colunas para lidar com acentos
                print("Tentando normalizar nomes de colunas para corrigir problemas de acentuação...")
                normalized_df_cols = {}
                normalized_model_cols = {}
                
                import unicodedata
                
                # Normalizar nomes de colunas do DataFrame
                for col in df.columns:
                    norm_col = unicodedata.normalize('NFKD', col).encode('ASCII', 'ignore').decode('ASCII')
                    normalized_df_cols[norm_col] = col
                
                # Normalizar nomes de colunas do modelo
                for col in missing_cols:
                    norm_col = unicodedata.normalize('NFKD', col).encode('ASCII', 'ignore').decode('ASCII')
                    normalized_model_cols[norm_col] = col
                
                # Verificar correspondências
                fixed_columns = {}
                for norm_model_col, model_col in normalized_model_cols.items():
                    if norm_model_col in normalized_df_cols:
                        df_col = normalized_df_cols[norm_model_col]
                        fixed_columns[df_col] = model_col
                
                if fixed_columns:
                    print(f"Encontradas {len(fixed_columns)} correspondências por normalização")
                    df = df.rename(columns=fixed_columns)
            
            # Se precisar preencher colunas faltantes com zeros
            if fill_missing:
                for col in maintain_feature_names:
                    if col not in df.columns and col != target_col:
                        df[col] = 0
                print(f"Adicionadas todas as {len(missing_cols)} features faltantes com zeros")
                
            # Garantir que todas as colunas necessárias existam após este ponto
            # Criar uma lista de colunas que existem no DataFrame
            cols_to_use = [col for col in maintain_feature_names if col in df.columns]
            
            # Adicionar a coluna target se não estiver na lista
            if target_col not in cols_to_use:
                cols_to_use.append(target_col)
                
            # Reorganizar o DataFrame
            df = df[cols_to_use]
            print(f"Usando {len(cols_to_use)} colunas no total (incluindo target)")
            
        else:
            # Sanitizar nomes das colunas se não for fornecida uma lista
            column_mapping = sanitize_column_names(df)
            if column_mapping:
                df = df.rename(columns=column_mapping)
                print(f"Renomeadas {len(column_mapping)} colunas")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
    
        if verbose:
            print(f"Dados carregados - {X.shape[0]} exemplos, {X.shape[1]} features")
            print(f"Taxa de conversão: {y.mean():.4f}")
        
        return df, X, y, target_col
    
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        raise

# ======================================
# CALIBRAÇÃO DE PROBABILIDADES
# ======================================

def train_calibrated_model(X, y, base_model, cv=3):
    """Treina um modelo calibrado para corrigir probabilidades."""
    print("\n=== Treinando modelo com calibração de probabilidades ===")
    
    # Criar modelo calibrado
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,  # Usar 'estimator' em vez de 'base_estimator'
        method='isotonic',     # Isotonic regression para calibração
        cv=cv
    )
    
    # Treinar modelo
    calibrated_model.fit(X, y)
    
    print("Modelo calibrado treinado com sucesso")
    return calibrated_model

# ======================================
# OTIMIZAÇÃO DE THRESHOLD
# ======================================

def optimize_threshold(model, X, y, mlflow_run_id=None):
    """Encontra o threshold ótimo para balancear precisão e recall."""
    print("\n=== Otimizando threshold para balancear precisão e recall ===")
    
    # Obter probabilidades
    try:
        y_probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"Erro ao gerar probabilidades: {e}")
        raise
    
    # Calcular curva precision-recall
    try:
        precision, recall, thresholds = precision_recall_curve(y, y_probs)
    except Exception as e:
        print(f"Erro ao calcular curva precision-recall: {e}")
        raise
    
    # Calcular F1 para cada threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Encontrar threshold com melhor F1-score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    
    print(f"Threshold ótimo (melhor F1): {best_threshold:.4f}")
    print(f"  - Precision: {best_precision:.4f}")
    print(f"  - Recall: {best_recall:.4f}")
    print(f"  - F1-score: {best_f1:.4f}")
    
    # Encontrar threshold para alto recall (ao menos 0.7) com precisão aceitável
    high_recall_indices = np.where(recall >= 0.7)[0]
    if len(high_recall_indices) > 0:
        hr_idx = high_recall_indices[0]  # Primeiro threshold que atinge recall 0.7
        hr_threshold = thresholds[hr_idx] if hr_idx < len(thresholds) else 0.01
        hr_precision = precision[hr_idx]
        hr_recall = recall[hr_idx]
        hr_f1 = 2 * (hr_precision * hr_recall) / (hr_precision + hr_recall + 1e-10)
        
        print(f"\nThreshold para alto recall (>= 70%): {hr_threshold:.4f}")
        print(f"  - Precision: {hr_precision:.4f}")
        print(f"  - Recall: {hr_recall:.4f}")
        print(f"  - F1-score: {hr_f1:.4f}")
    
    # Plotar curva precision-recall
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', label='Precision-Recall curve')
    plt.scatter([best_recall], [best_precision], c='red', marker='o', 
               label=f'Melhor F1: {best_f1:.3f} (t={best_threshold:.3f})')
    
    if len(high_recall_indices) > 0:
        plt.scatter([hr_recall], [hr_precision], c='green', marker='s',
                   label=f'Alto Recall: {hr_recall:.3f} (t={hr_threshold:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    # Salvar figura - diretamente no diretório de artifacts do MLflow se fornecido
    if mlflow_run_id:
        mlflow_artifacts_dir = os.path.join(MLFLOW_DIR, mlflow_run_id, "artifacts")
        os.makedirs(mlflow_artifacts_dir, exist_ok=True)
        plt_path = os.path.join(mlflow_artifacts_dir, f'pr_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    else:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt_path = os.path.join(PLOTS_DIR, f'pr_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    plt.savefig(plt_path)
    print(f"Curva PR salva em: {plt_path}")
    
    # Registrar também no MLflow diretamente se run_id fornecido
    if mlflow_run_id:
        try:
            active_run = mlflow.active_run()
            if active_run and active_run.info.run_id == mlflow_run_id:
                mlflow.log_artifact(plt_path)
                print(f"Curva PR registrada no MLflow run: {mlflow_run_id}")
        except Exception as e:
            print(f"Aviso: Erro ao registrar curva PR no MLflow: {e}")
    
    return best_threshold, best_f1, best_precision, best_recall

# ======================================
# AVALIAÇÃO DO MODELO
# ======================================

def evaluate_model(model, X, y, threshold=0.5, model_name="Model"):
    """Avalia o modelo usando o threshold especificado."""
    print(f"\n=== Avaliando {model_name} (threshold={threshold:.4f}) ===")
    
    # Gerar probabilidades
    y_probs = model.predict_proba(X)[:, 1]
    
    # Aplicar threshold
    y_preds = (y_probs >= threshold).astype(int)
    
    # Calcular métricas
    precision = precision_score(y, y_preds)
    recall = recall_score(y, y_preds)
    f1 = f1_score(y, y_preds)
    
    # Contar predições
    n_positives = np.sum(y_preds)
    pos_rate = n_positives / len(y_preds)
    
    # Falsos positivos e negativos
    fp = np.sum((y == 0) & (y_preds == 1))
    fn = np.sum((y == 1) & (y_preds == 0))
    
    print(f"Métricas para {model_name}:")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-score: {f1:.4f}")
    print(f"  - Taxa de positivos: {pos_rate:.2%} ({n_positives} de {len(y_preds)})")
    print(f"  - Falsos positivos: {fp}")
    print(f"  - Falsos negativos: {fn}")
    
    return {
        'model_name': model_name,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'positives': n_positives,
        'false_positives': fp,
        'false_negatives': fn
    }

# ======================================
# SALVAR MODELOS
# ======================================

def save_models(models_dict, thresholds_dict, output_dir=OUTPUT_DIR, mlflow_run_id=None):
    """
    Salva os modelos treinados e seus thresholds.
    
    Args:
        models_dict: Dicionário com modelos para salvar
        thresholds_dict: Dicionário com thresholds para salvar
        output_dir: Diretório de saída padrão
        mlflow_run_id: ID da run do MLflow para salvar diretamente nos artifacts
        
    Returns:
        Diretório onde os modelos foram salvos
    """
    # Determinar diretório de saída baseado no mlflow_run_id
    if mlflow_run_id:
        # Usar o diretório de artifacts do MLflow
        mlflow_artifacts_dir = os.path.join(MLFLOW_DIR, mlflow_run_id, "artifacts")
        final_output_dir = os.path.join(mlflow_artifacts_dir, "models")
        print(f"\n=== Salvando modelos em {final_output_dir} (MLflow run {mlflow_run_id}) ===")
    else:
        # Usar o diretório padrão
        final_output_dir = output_dir
        print(f"\n=== Salvando modelos em {final_output_dir} ===")
    
    # Criar diretório se não existir
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Salvar cada modelo
    for name, model in models_dict.items():
        model_path = os.path.join(final_output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"  - Modelo {name} salvo em: {model_path}")
    
    # Salvar thresholds
    thresholds_path = os.path.join(final_output_dir, "thresholds.joblib")
    joblib.dump(thresholds_dict, thresholds_path)
    print(f"  - Thresholds salvos em: {thresholds_path}")
    
    # Salvar configuração
    config = {
        'models': list(models_dict.keys()),
        'thresholds': thresholds_dict,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_path = os.path.join(final_output_dir, "config.joblib")
    joblib.dump(config, config_path)
    
    # Se estiver usando MLflow, registrar diretamente no MLflow
    if mlflow_run_id:
        try:
            active_run = mlflow.active_run()
            if active_run and active_run.info.run_id == mlflow_run_id:
                # Registrar todos os arquivos salvos como artifacts
                for filename in os.listdir(final_output_dir):
                    file_path = os.path.join(final_output_dir, filename)
                    if os.path.isfile(file_path):
                        mlflow.log_artifact(file_path)
                print(f"Todos os modelos e configurações registrados no MLflow run: {mlflow_run_id}")
        except Exception as e:
            print(f"Aviso: Erro ao registrar modelos no MLflow: {e}")
    
    return final_output_dir

# ======================================
# FUNÇÃO PRINCIPAL
# ======================================

def main():
    parser = argparse.ArgumentParser(description='Implementa calibração e ajuste de threshold')
    parser.add_argument('--mlflow_dir', default=MLFLOW_DIR, 
                      help='Diretório do MLflow tracking')
    parser.add_argument('--data_dir', default=DATA_DIR,
                      help='Diretório com os arquivos de dados')
    parser.add_argument('--artifacts_dir', default=ARTIFACTS_DIR,
                      help='Diretório para buscar modelos de cluster')
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                      help='Diretório para salvar os modelos')
    
    args = parser.parse_args()
    
    # Definir caminhos de arquivos baseados nos diretórios fornecidos
    train_path = os.path.join(args.data_dir, "train.csv")
    validation_path = os.path.join(args.data_dir, "validation.csv")
    
    # Verificar se os caminhos existem
    for path, desc in [
        (train_path, "treinamento"), 
        (validation_path, "validação"), 
        (args.mlflow_dir, "MLflow")
    ]:
        if not os.path.exists(path):
            print(f"AVISO: Caminho de {desc} não encontrado: {path}")
            if desc in ["treinamento", "validação"]:
                print(f"ERRO: O arquivo de {desc} é necessário para continuar.")
                return
    
    # Configurar MLflow
    experiment_id = setup_mlflow(args.mlflow_dir, "smart_ads_calibrated")
    
    # Iniciar run do MLflow
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"MLflow run iniciado: {run_id}")
        
        # Registrar parâmetros
        mlflow.log_params({
            'train_data': train_path,
            'validation_data': validation_path,
            'mlflow_dir': args.mlflow_dir,
            'artifacts_dir': args.artifacts_dir,
            'output_dir': args.output_dir
        })
        
        # 1. Buscar e carregar o melhor modelo RandomForest
        print("\n=== Etapa 1: Buscando e carregando o melhor modelo RandomForest ===")
        
        rf_model_uri, rf_threshold = find_best_random_forest_model(args.mlflow_dir)
        if not rf_model_uri:
            print("ERRO: Não foi possível encontrar um modelo Random Forest válido.")
            return
            
        print("Carregando modelo para obter nomes de features...")
        rf_model = load_model(rf_model_uri)
        if not rf_model:
            print("ERRO: Falha ao carregar o modelo Random Forest.")
            return
            
        # Extrair nomes de features do modelo
        if hasattr(rf_model, 'feature_names_in_'):
            print(f"Modelo possui {len(rf_model.feature_names_in_)} features de entrada")
            model_features = list(rf_model.feature_names_in_)
        else:
            print("AVISO: Modelo não possui atributo 'feature_names_in_'")
            model_features = None
        
        # 2. Carregar dados usando os nomes de features do modelo
        print("\n=== Etapa 2: Carregando dados com nomes de features consistentes ===")
        
        # Carregar dados de treinamento e validação usando os nomes de features do modelo
        df_train, X_train, y_train, target_col = load_data(train_path, maintain_feature_names=model_features, fill_missing=True)
        df_val, X_val, y_val, _ = load_data(validation_path, maintain_feature_names=model_features, fill_missing=True)
        
        # Verificar se as features do modelo existem nos dados
        if model_features:
            # Garantir que a ordem das colunas seja a mesma do modelo
            X_train = X_train[model_features]
            X_val = X_val[model_features]
            print("Reordenadas colunas para corresponder ao modelo")
        
        # 3. Buscar e carregar os modelos de cluster
        print("\n=== Etapa 3: Buscando e carregando modelos de cluster ===")
        
        # 3.1 K-means - buscar pelo arquivo kmeans_model.joblib diretamente
        kmeans_model_path = find_best_cluster_model(args.artifacts_dir, "kmeans")
        if kmeans_model_path:
            kmeans_model = load_model(kmeans_model_path)
            if kmeans_model:
                print(f"Modelo K-means carregado com sucesso")
            else:
                print("AVISO: Falha ao carregar modelo K-means")
                kmeans_model = None
        else:
            print("AVISO: Não foi possível encontrar um modelo K-means válido.")
            kmeans_model = None
        
        # 3.2 GMM - buscar pelo arquivo gmm_model.joblib diretamente
        gmm_model_path = find_best_cluster_model(args.artifacts_dir, "gmm")
        if gmm_model_path:
            gmm_model = load_model(gmm_model_path)
            # Verificar se é um StandardScaler em vez de um modelo
            if hasattr(gmm_model, 'predict_proba'):
                print(f"Modelo GMM carregado com sucesso")
            else:
                print("AVISO: O arquivo carregado não parece ser um modelo GMM válido (sem predict_proba).")
                gmm_model = None
        else:
            print("AVISO: Não foi possível encontrar um modelo GMM válido.")
            gmm_model = None
        
        # 4. Avaliar modelos originais
        print("\n=== Etapa 4: Avaliando modelos originais ===")
        models_results = {}
        
        # 4.1 Avaliar Random Forest
        rf_results = evaluate_model(
            rf_model, X_val, y_val, 
            threshold=rf_threshold,
            model_name="Random Forest Original"
        )
        models_results["rf_original"] = rf_results
        
        # Registrar métricas no MLflow
        mlflow.log_metrics({
            'rf_original_precision': rf_results['precision'],
            'rf_original_recall': rf_results['recall'],
            'rf_original_f1': rf_results['f1']
        })
        
        # 5. Treinar modelos calibrados
        print("\n=== Etapa 5: Treinando modelos calibrados ===")
        calibrated_models = {}
        
        # 5.1 Calibrar Random Forest
        calibrated_rf = train_calibrated_model(X_train, y_train, rf_model)
        calibrated_models["rf_calibrated"] = calibrated_rf
        
        # 5.2 Calibrar K-means se disponível
        if kmeans_model and hasattr(kmeans_model, 'predict_proba'):
            try:
                calibrated_kmeans = train_calibrated_model(X_train, y_train, kmeans_model)
                calibrated_models["kmeans_calibrated"] = calibrated_kmeans
            except Exception as e:
                print(f"Erro ao calibrar K-means: {str(e)}")
        
        # 5.3 Calibrar GMM se disponível
        if gmm_model and hasattr(gmm_model, 'predict_proba'):
            try:
                calibrated_gmm = train_calibrated_model(X_train, y_train, gmm_model)
                calibrated_models["gmm_calibrated"] = calibrated_gmm
            except Exception as e:
                print(f"Erro ao calibrar GMM: {str(e)}")
        
        # 6. Avaliar modelos calibrados com threshold padrão
        print("\n=== Etapa 6: Avaliando modelos calibrados com threshold padrão ===")
        
        # 6.1 Avaliar Random Forest calibrado
        rf_cal_results = evaluate_model(
            calibrated_rf, X_val, y_val,
            threshold=rf_threshold,
            model_name="Random Forest Calibrado (threshold original)"
        )
        models_results["rf_calibrated_default"] = rf_cal_results
        
        # Registrar métricas no MLflow
        mlflow.log_metrics({
            'rf_calibrated_default_precision': rf_cal_results['precision'],
            'rf_calibrated_default_recall': rf_cal_results['recall'],
            'rf_calibrated_default_f1': rf_cal_results['f1']
        })
        
        # 6.2 Avaliar K-means calibrado se disponível
        if "kmeans_calibrated" in calibrated_models:
            kmeans_cal_results = evaluate_model(
                calibrated_models["kmeans_calibrated"], X_val, y_val,
                threshold=0.5,  # Threshold padrão para K-means
                model_name="K-means Calibrado (threshold padrão)"
            )
            models_results["kmeans_calibrated_default"] = kmeans_cal_results
            
            # Registrar métricas no MLflow
            mlflow.log_metrics({
                'kmeans_calibrated_default_precision': kmeans_cal_results['precision'],
                'kmeans_calibrated_default_recall': kmeans_cal_results['recall'],
                'kmeans_calibrated_default_f1': kmeans_cal_results['f1']
            })
        
        # 6.3 Avaliar GMM calibrado se disponível
        if "gmm_calibrated" in calibrated_models:
            gmm_cal_results = evaluate_model(
                calibrated_models["gmm_calibrated"], X_val, y_val,
                threshold=0.5,  # Threshold padrão para GMM
                model_name="GMM Calibrado (threshold padrão)"
            )
            models_results["gmm_calibrated_default"] = gmm_cal_results
            
            # Registrar métricas no MLflow
            mlflow.log_metrics({
                'gmm_calibrated_default_precision': gmm_cal_results['precision'],
                'gmm_calibrated_default_recall': gmm_cal_results['recall'],
                'gmm_calibrated_default_f1': gmm_cal_results['f1']
            })
        
        # 7. Otimizar thresholds para os modelos calibrados
        print("\n=== Etapa 7: Otimizando thresholds para modelos calibrados ===")
        optimized_thresholds = {}
        
        # 7.1 Otimizar threshold para Random Forest calibrado
        rf_best_threshold, rf_best_f1, rf_best_precision, rf_best_recall = optimize_threshold(
            calibrated_rf, X_val, y_val, mlflow_run_id=run_id
        )
        optimized_thresholds["rf"] = rf_best_threshold
        
        # Registrar threshold otimizado no MLflow
        mlflow.log_metrics({
            'rf_optimized_threshold': rf_best_threshold,
            'rf_optimized_precision': rf_best_precision,
            'rf_optimized_recall': rf_best_recall,
            'rf_optimized_f1': rf_best_f1
        })
        
        # 7.2 Otimizar threshold para K-means calibrado se disponível
        if "kmeans_calibrated" in calibrated_models:
            kmeans_best_threshold, kmeans_best_f1, kmeans_best_precision, kmeans_best_recall = optimize_threshold(
                calibrated_models["kmeans_calibrated"], X_val, y_val, mlflow_run_id=run_id
            )
            optimized_thresholds["kmeans"] = kmeans_best_threshold
            
            # Registrar threshold otimizado no MLflow
            mlflow.log_metrics({
                'kmeans_optimized_threshold': kmeans_best_threshold,
                'kmeans_optimized_precision': kmeans_best_precision,
                'kmeans_optimized_recall': kmeans_best_recall,
                'kmeans_optimized_f1': kmeans_best_f1
            })
        
        # 7.3 Otimizar threshold para GMM calibrado se disponível
        if "gmm_calibrated" in calibrated_models:
            gmm_best_threshold, gmm_best_f1, gmm_best_precision, gmm_best_recall = optimize_threshold(
                calibrated_models["gmm_calibrated"], X_val, y_val, mlflow_run_id=run_id
            )
            optimized_thresholds["gmm"] = gmm_best_threshold
            
            # Registrar threshold otimizado no MLflow
            mlflow.log_metrics({
                'gmm_optimized_threshold': gmm_best_threshold,
                'gmm_optimized_precision': gmm_best_precision,
                'gmm_optimized_recall': gmm_best_recall,
                'gmm_optimized_f1': gmm_best_f1
            })
        
        # 8. Avaliar modelos calibrados com thresholds otimizados
        print("\n=== Etapa 8: Avaliando modelos calibrados com thresholds otimizados ===")
        
        # 8.1 Avaliar Random Forest calibrado com threshold otimizado
        rf_cal_opt_results = evaluate_model(
            calibrated_rf, X_val, y_val,
            threshold=rf_best_threshold,
            model_name="Random Forest Calibrado (threshold otimizado)"
        )
        models_results["rf_calibrated_optimized"] = rf_cal_opt_results
        
        # Registrar métricas otimizadas no MLflow
        mlflow.log_metrics({
            'rf_calibrated_optimized_precision': rf_cal_opt_results['precision'],
            'rf_calibrated_optimized_recall': rf_cal_opt_results['recall'],
            'rf_calibrated_optimized_f1': rf_cal_opt_results['f1']
        })
        
        # 8.2 Avaliar K-means calibrado com threshold otimizado se disponível
        if "kmeans_calibrated" in calibrated_models and "kmeans" in optimized_thresholds:
            kmeans_cal_opt_results = evaluate_model(
                calibrated_models["kmeans_calibrated"], X_val, y_val,
                threshold=optimized_thresholds["kmeans"],
                model_name="K-means Calibrado (threshold otimizado)"
            )
            models_results["kmeans_calibrated_optimized"] = kmeans_cal_opt_results
            
            # Registrar métricas otimizadas no MLflow
            mlflow.log_metrics({
                'kmeans_calibrated_optimized_precision': kmeans_cal_opt_results['precision'],
                'kmeans_calibrated_optimized_recall': kmeans_cal_opt_results['recall'],
                'kmeans_calibrated_optimized_f1': kmeans_cal_opt_results['f1']
            })
        
        # 8.3 Avaliar GMM calibrado com threshold otimizado se disponível
        if "gmm_calibrated" in calibrated_models and "gmm" in optimized_thresholds:
            gmm_cal_opt_results = evaluate_model(
                calibrated_models["gmm_calibrated"], X_val, y_val,
                threshold=optimized_thresholds["gmm"],
                model_name="GMM Calibrado (threshold otimizado)"
            )
            models_results["gmm_calibrated_optimized"] = gmm_cal_opt_results
            
            # Registrar métricas otimizadas no MLflow
            mlflow.log_metrics({
                'gmm_calibrated_optimized_precision': gmm_cal_opt_results['precision'],
                'gmm_calibrated_optimized_recall': gmm_cal_opt_results['recall'],
                'gmm_calibrated_optimized_f1': gmm_cal_opt_results['f1']
            })
        
        # 9. Salvar modelos e thresholds
        print("\n=== Etapa 9: Salvando modelos ===")
        
        # Preparar dicionários para salvar
        models_to_save = {
            'rf_original': rf_model,
            'rf_calibrated': calibrated_rf
        }
        
        # Adicionar modelos K-means se disponíveis
        if kmeans_model and "kmeans_calibrated" in calibrated_models:
            models_to_save['kmeans_original'] = kmeans_model
            models_to_save['kmeans_calibrated'] = calibrated_models["kmeans_calibrated"]
        
        # Adicionar modelos GMM se disponíveis
        if gmm_model and "gmm_calibrated" in calibrated_models:
            models_to_save['gmm_original'] = gmm_model
            models_to_save['gmm_calibrated'] = calibrated_models["gmm_calibrated"]
        
        # Preparar dicionário de thresholds
        thresholds_to_save = {
            'rf_original': rf_threshold,
            'rf_calibrated_default': rf_threshold,
            'rf_calibrated_optimized': rf_best_threshold
        }
        
        # Adicionar thresholds K-means se disponíveis
        if "kmeans" in optimized_thresholds:
            thresholds_to_save['kmeans_calibrated_default'] = 0.5
            thresholds_to_save['kmeans_calibrated_optimized'] = optimized_thresholds["kmeans"]
        
        # Adicionar thresholds GMM se disponíveis
        if "gmm" in optimized_thresholds:
            thresholds_to_save['gmm_calibrated_default'] = 0.5
            thresholds_to_save['gmm_calibrated_optimized'] = optimized_thresholds["gmm"]
        
        # Salvar nos diretórios do MLflow e no dir de output
        output_dir = save_models(models_to_save, thresholds_to_save, args.output_dir, mlflow_run_id=run_id)
        
        # Registrar modelos no MLflow com input_example
        input_example = X_train.iloc[:5].copy() if len(X_train) > 5 else X_train.copy()
        
        mlflow.sklearn.log_model(rf_model, "rf_original_model", input_example=input_example)
        mlflow.sklearn.log_model(calibrated_rf, "rf_calibrated_model", input_example=input_example)
        
        if kmeans_model and "kmeans_calibrated" in calibrated_models:
            mlflow.sklearn.log_model(kmeans_model, "kmeans_original_model", input_example=input_example)
            mlflow.sklearn.log_model(calibrated_models["kmeans_calibrated"], "kmeans_calibrated_model", input_example=input_example)
        
        if gmm_model and "gmm_calibrated" in calibrated_models:
            mlflow.sklearn.log_model(gmm_model, "gmm_original_model", input_example=input_example)
            mlflow.sklearn.log_model(calibrated_models["gmm_calibrated"], "gmm_calibrated_model", input_example=input_example)
        
        # 10. Resumo dos resultados
        print("\n=== Resumo dos Resultados ===")
        results_df = pd.DataFrame(list(models_results.values()))
        print(results_df[['model_name', 'threshold', 'precision', 'recall', 'f1']])
        
        # Salvar resumo em CSV
        results_path = os.path.join(args.output_dir, "results_summary.csv")
        results_df.to_csv(results_path, index=False)
        
        # Registrar o resumo também no MLflow
        mlflow_artifacts_dir = os.path.join(MLFLOW_DIR, run_id, "artifacts")
        os.makedirs(mlflow_artifacts_dir, exist_ok=True)
        mlflow_results_path = os.path.join(mlflow_artifacts_dir, "results_summary.csv")
        results_df.to_csv(mlflow_results_path, index=False)
        mlflow.log_artifact(mlflow_results_path)
        
        print("\n=== Treinamento e avaliação concluídos com sucesso! ===")
        print(f"Modelos salvos em: {output_dir}")
        print(f"Artefatos do MLflow em: {mlflow_artifacts_dir}")
        print(f"MLflow run ID: {run_id}")
        
        return output_dir, run_id
        
    # Carregar dados de treinamento e validação usando os nomes de features do modelo
    df_train, X_train, y_train, target_col = load_data(train_path, maintain_feature_names=model_features, fill_missing=True)
    df_val, X_val, y_val, _ = load_data(validation_path, maintain_feature_names=model_features, fill_missing=True)
    
    # Verificar se as features do modelo existem nos dados
    if model_features:
        # Garantir que a ordem das colunas seja a mesma do modelo
        X_train = X_train[model_features]
        X_val = X_val[model_features]
        print("Reordenadas colunas para corresponder ao modelo")
    
    # 3. Buscar e carregar os modelos de cluster
    print("\n=== Etapa 3: Buscando e carregando modelos de cluster ===")
    
    # 3.1 K-means - buscar pelo arquivo kmeans_model.joblib diretamente
    kmeans_model_path = find_best_cluster_model(args.artifacts_dir, "kmeans")
    if kmeans_model_path:
        kmeans_model = load_model(kmeans_model_path)
        if kmeans_model:
            print(f"Modelo K-means carregado com sucesso")
        else:
            print("AVISO: Falha ao carregar modelo K-means")
            kmeans_model = None
    else:
        print("AVISO: Não foi possível encontrar um modelo K-means válido.")
        kmeans_model = None
    
    # 3.2 GMM - buscar pelo arquivo gmm_model.joblib diretamente
    gmm_model_path = find_best_cluster_model(args.artifacts_dir, "gmm")
    if gmm_model_path:
        gmm_model = load_model(gmm_model_path)
        # Verificar se é um StandardScaler em vez de um modelo
        if hasattr(gmm_model, 'predict_proba'):
            print(f"Modelo GMM carregado com sucesso")
        else:
            print("AVISO: O arquivo carregado não parece ser um modelo GMM válido (sem predict_proba).")
            gmm_model = None
    else:
        print("AVISO: Não foi possível encontrar um modelo GMM válido.")
        gmm_model = None
    
    # 4. Avaliar modelos originais
    print("\n=== Etapa 4: Avaliando modelos originais ===")
    models_results = {}
    
    # 4.1 Avaliar Random Forest
    rf_results = evaluate_model(
        rf_model, X_val, y_val, 
        threshold=rf_threshold,
        model_name="Random Forest Original"
    )
    models_results["rf_original"] = rf_results
    
    # Registrar métricas no MLflow
    mlflow.log_metrics({
        'rf_original_precision': rf_results['precision'],
        'rf_original_recall': rf_results['recall'],
        'rf_original_f1': rf_results['f1']
    })
    
    # 5. Treinar modelos calibrados
    print("\n=== Etapa 5: Treinando modelos calibrados ===")
    calibrated_models = {}
    
    # 5.1 Calibrar Random Forest
    calibrated_rf = train_calibrated_model(X_train, y_train, rf_model)
    calibrated_models["rf_calibrated"] = calibrated_rf
    
    # 5.2 Calibrar K-means se disponível
    if kmeans_model and hasattr(kmeans_model, 'predict_proba'):
        try:
            calibrated_kmeans = train_calibrated_model(X_train, y_train, kmeans_model)
            calibrated_models["kmeans_calibrated"] = calibrated_kmeans
        except Exception as e:
            print(f"Erro ao calibrar K-means: {str(e)}")
    
    # 5.3 Calibrar GMM se disponível
    if gmm_model and hasattr(gmm_model, 'predict_proba'):
        try:
            calibrated_gmm = train_calibrated_model(X_train, y_train, gmm_model)
            calibrated_models["gmm_calibrated"] = calibrated_gmm
        except Exception as e:
            print(f"Erro ao calibrar GMM: {str(e)}")
    
    # 6. Avaliar modelos calibrados com threshold padrão
    print("\n=== Etapa 6: Avaliando modelos calibrados com threshold padrão ===")
    
    # 6.1 Avaliar Random Forest calibrado
    rf_cal_results = evaluate_model(
        calibrated_rf, X_val, y_val,
        threshold=rf_threshold,
        model_name="Random Forest Calibrado (threshold original)"
    )
    models_results["rf_calibrated_default"] = rf_cal_results
    
    # Registrar métricas no MLflow
    mlflow.log_metrics({
        'rf_calibrated_default_precision': rf_cal_results['precision'],
        'rf_calibrated_default_recall': rf_cal_results['recall'],
        'rf_calibrated_default_f1': rf_cal_results['f1']
    })
    
    # 6.2 Avaliar K-means calibrado se disponível
    if "kmeans_calibrated" in calibrated_models:
        kmeans_cal_results = evaluate_model(
            calibrated_models["kmeans_calibrated"], X_val, y_val,
            threshold=0.5,  # Threshold padrão para K-means
            model_name="K-means Calibrado (threshold padrão)"
        )
        models_results["kmeans_calibrated_default"] = kmeans_cal_results
        
        # Registrar métricas no MLflow
        mlflow.log_metrics({
            'kmeans_calibrated_default_precision': kmeans_cal_results['precision'],
            'kmeans_calibrated_default_recall': kmeans_cal_results['recall'],
            'kmeans_calibrated_default_f1': kmeans_cal_results['f1']
        })
    
    # 6.3 Avaliar GMM calibrado se disponível
    if "gmm_calibrated" in calibrated_models:
        gmm_cal_results = evaluate_model(
            calibrated_models["gmm_calibrated"], X_val, y_val,
            threshold=0.5,  # Threshold padrão para GMM
            model_name="GMM Calibrado (threshold padrão)"
        )
        models_results["gmm_calibrated_default"] = gmm_cal_results
        
        # Registrar métricas no MLflow
        mlflow.log_metrics({
            'gmm_calibrated_default_precision': gmm_cal_results['precision'],
            'gmm_calibrated_default_recall': gmm_cal_results['recall'],
            'gmm_calibrated_default_f1': gmm_cal_results['f1']
        })
    
    # 7. Otimizar thresholds para os modelos calibrados
    print("\n=== Etapa 7: Otimizando thresholds para modelos calibrados ===")
    optimized_thresholds = {}
    
    # 7.1 Otimizar threshold para Random Forest calibrado
    rf_best_threshold, rf_best_f1, rf_best_precision, rf_best_recall = optimize_threshold(
        calibrated_rf, X_val, y_val, mlflow_run_id=run_id
    )
    optimized_thresholds["rf"] = rf_best_threshold
    
    # Registrar threshold otimizado no MLflow
    mlflow.log_metrics({
        'rf_optimized_threshold': rf_best_threshold,
        'rf_optimized_precision': rf_best_precision,
        'rf_optimized_recall': rf_best_recall,
        'rf_optimized_f1': rf_best_f1
    })
    
    # 7.2 Otimizar threshold para K-means calibrado se disponível
    if "kmeans_calibrated" in calibrated_models:
        kmeans_best_threshold, kmeans_best_f1, kmeans_best_precision, kmeans_best_recall = optimize_threshold(
            calibrated_models["kmeans_calibrated"], X_val, y_val, mlflow_run_id=run_id
        )
        optimized_thresholds["kmeans"] = kmeans_best_threshold
        
        # Registrar threshold otimizado no MLflow
        mlflow.log_metrics({
            'kmeans_optimized_threshold': kmeans_best_threshold,
            'kmeans_optimized_precision': kmeans_best_precision,
            'kmeans_optimized_recall': kmeans_best_recall,
            'kmeans_optimized_f1': kmeans_best_f1
        })
    
    # 7.3 Otimizar threshold para GMM calibrado se disponível
    if "gmm_calibrated" in calibrated_models:
        gmm_best_threshold, gmm_best_f1, gmm_best_precision, gmm_best_recall = optimize_threshold(
            calibrated_models["gmm_calibrated"], X_val, y_val, mlflow_run_id=run_id
        )
        optimized_thresholds["gmm"] = gmm_best_threshold
        
        # Registrar threshold otimizado no MLflow
        mlflow.log_metrics({
            'gmm_optimized_threshold': gmm_best_threshold,
            'gmm_optimized_precision': gmm_best_precision,
            'gmm_optimized_recall': gmm_best_recall,
            'gmm_optimized_f1': gmm_best_f1
        })
    
    # 8. Avaliar modelos calibrados com thresholds otimizados
    print("\n=== Etapa 8: Avaliando modelos calibrados com thresholds otimizados ===")
    
    # 8.1 Avaliar Random Forest calibrado com threshold otimizado
    rf_cal_opt_results = evaluate_model(
        calibrated_rf, X_val, y_val,
        threshold=rf_best_threshold,
        model_name="Random Forest Calibrado (threshold otimizado)"
    )
    models_results["rf_calibrated_optimized"] = rf_cal_opt_results
    
    # Registrar métricas otimizadas no MLflow
    mlflow.log_metrics({
        'rf_calibrated_optimized_precision': rf_cal_opt_results['precision'],
        'rf_calibrated_optimized_recall': rf_cal_opt_results['recall'],
        'rf_calibrated_optimized_f1': rf_cal_opt_results['f1']
    })
    
    # 8.2 Avaliar K-means calibrado com threshold otimizado se disponível
    if "kmeans_calibrated" in calibrated_models and "kmeans" in optimized_thresholds:
        kmeans_cal_opt_results = evaluate_model(
            calibrated_models["kmeans_calibrated"], X_val, y_val,
            threshold=optimized_thresholds["kmeans"],
            model_name="K-means Calibrado (threshold otimizado)"
        )
        models_results["kmeans_calibrated_optimized"] = kmeans_cal_opt_results
        
        # Registrar métricas otimizadas no MLflow
        mlflow.log_metrics({
            'kmeans_calibrated_optimized_precision': kmeans_cal_opt_results['precision'],
            'kmeans_calibrated_optimized_recall': kmeans_cal_opt_results['recall'],
            'kmeans_calibrated_optimized_f1': kmeans_cal_opt_results['f1']
        })
    
    # 8.3 Avaliar GMM calibrado com threshold otimizado se disponível
    if "gmm_calibrated" in calibrated_models and "gmm" in optimized_thresholds:
        gmm_cal_opt_results = evaluate_model(
            calibrated_models["gmm_calibrated"], X_val, y_val,
            threshold=optimized_thresholds["gmm"],
            model_name="GMM Calibrado (threshold otimizado)"
        )
        models_results["gmm_calibrated_optimized"] = gmm_cal_opt_results
        
        # Registrar métricas otimizadas no MLflow
        mlflow.log_metrics({
            'gmm_calibrated_optimized_precision': gmm_cal_opt_results['precision'],
            'gmm_calibrated_optimized_recall': gmm_cal_opt_results['recall'],
            'gmm_calibrated_optimized_f1': gmm_cal_opt_results['f1']
        })
    
    # 9. Salvar modelos e thresholds
    print("\n=== Etapa 9: Salvando modelos ===")
    
    # Preparar dicionários para salvar
    models_to_save = {
        'rf_original': rf_model,
        'rf_calibrated': calibrated_rf
    }
    
    # Adicionar modelos K-means se disponíveis
    if kmeans_model and "kmeans_calibrated" in calibrated_models:
        models_to_save['kmeans_original'] = kmeans_model
        models_to_save['kmeans_calibrated'] = calibrated_models["kmeans_calibrated"]
    
    # Adicionar modelos GMM se disponíveis
    if gmm_model and "gmm_calibrated" in calibrated_models:
        models_to_save['gmm_original'] = gmm_model
        models_to_save['gmm_calibrated'] = calibrated_models["gmm_calibrated"]
    
    # Preparar dicionário de thresholds
    thresholds_to_save = {
        'rf_original': rf_threshold,
        'rf_calibrated_default': rf_threshold,
        'rf_calibrated_optimized': rf_best_threshold
    }
    
    # Adicionar thresholds K-means se disponíveis
    if "kmeans" in optimized_thresholds:
        thresholds_to_save['kmeans_calibrated_default'] = 0.5
        thresholds_to_save['kmeans_calibrated_optimized'] = optimized_thresholds["kmeans"]
    
    # Adicionar thresholds GMM se disponíveis
    if "gmm" in optimized_thresholds:
        thresholds_to_save['gmm_calibrated_default'] = 0.5
        thresholds_to_save['gmm_calibrated_optimized'] = optimized_thresholds["gmm"]
    
    # Salvar nos diretórios do MLflow e no dir de output
    output_dir = save_models(models_to_save, thresholds_to_save, args.output_dir, mlflow_run_id=run_id)
    
    # Registrar modelos no MLflow com input_example
    input_example = X_train.iloc[:5].copy() if len(X_train) > 5 else X_train.copy()
    
    mlflow.sklearn.log_model(rf_model, "rf_original_model", input_example=input_example)
    mlflow.sklearn.log_model(calibrated_rf, "rf_calibrated_model", input_example=input_example)
    
    if kmeans_model and "kmeans_calibrated" in calibrated_models:
        mlflow.sklearn.log_model(kmeans_model, "kmeans_original_model", input_example=input_example)
        mlflow.sklearn.log_model(calibrated_models["kmeans_calibrated"], "kmeans_calibrated_model", input_example=input_example)
    
    if gmm_model and "gmm_calibrated" in calibrated_models:
        mlflow.sklearn.log_model(gmm_model, "gmm_original_model", input_example=input_example)
        mlflow.sklearn.log_model(calibrated_models["gmm_calibrated"], "gmm_calibrated_model", input_example=input_example)
    
    # 10. Resumo dos resultados
    print("\n=== Resumo dos Resultados ===")
    results_df = pd.DataFrame(list(models_results.values()))
    print(results_df[['model_name', 'threshold', 'precision', 'recall', 'f1']])
    
    # Salvar resumo em CSV
    results_path = os.path.join(args.output_dir, "results_summary.csv")
    results_df.to_csv(results_path, index=False)
    
    # Registrar o resumo também no MLflow
    mlflow_artifacts_dir = os.path.join(MLFLOW_DIR, run_id, "artifacts")
    os.makedirs(mlflow_artifacts_dir, exist_ok=True)
    mlflow_results_path = os.path.join(mlflow_artifacts_dir, "results_summary.csv")
    results_df.to_csv(mlflow_results_path, index=False)
    mlflow.log_artifact(mlflow_results_path)
    
    print("\n=== Treinamento e avaliação concluídos com sucesso! ===")
    print(f"Modelos salvos em: {output_dir}")
    print(f"Artefatos do MLflow em: {mlflow_artifacts_dir}")
    print(f"MLflow run ID: {run_id}")
    
    return output_dir, run_id
    
    # Carregar dados de treinamento e validação usando os nomes de features do modelo
    df_train, X_train, y_train, target_col = load_data(train_path, maintain_feature_names=model_features)
    df_val, X_val, y_val, _ = load_data(validation_path, maintain_feature_names=model_features)
    
    # Verificar se as features do modelo existem nos dados
    if model_features:
        missing_in_train = [f for f in model_features if f not in X_train.columns]
        missing_in_val = [f for f in model_features if f not in X_val.columns]
        
        if missing_in_train or missing_in_val:
            print(f"AVISO: Alguns nomes de features do modelo estão faltando nos dados:")
            if missing_in_train:
                print(f"  - Treino: {len(missing_in_train)} features faltando")
            if missing_in_val:
                print(f"  - Validação: {len(missing_in_val)} features faltando")
            
            # Adicionar colunas faltantes preenchidas com zeros
            for col in missing_in_train:
                X_train[col] = 0
            for col in missing_in_val:
                X_val[col] = 0
            
            print("Adicionadas colunas faltantes preenchidas com zeros")
        
        # Garantir que a ordem das colunas seja a mesma do modelo
        X_train = X_train[model_features]
        X_val = X_val[model_features]
        print("Reordenadas colunas para corresponder ao modelo")
    
    # 3. Buscar e carregar os modelos de cluster
    print("\n=== Etapa 3: Buscando e carregando modelos de cluster ===")
    
    # 3.1 K-means
    kmeans_model_path = find_best_cluster_model(args.artifacts_dir, "kmeans")
    if kmeans_model_path:
        kmeans_model = load_model(kmeans_model_path)
        print(f"Modelo K-means carregado com sucesso")
    else:
        print("AVISO: Não foi possível encontrar um modelo K-means válido.")
        kmeans_model = None
    
    # 3.2 GMM
    gmm_model_path = find_best_cluster_model(args.artifacts_dir, "gmm")
    if gmm_model_path:
        gmm_model = load_model(gmm_model_path)
        print(f"Modelo GMM carregado com sucesso")
    else:
        print("AVISO: Não foi possível encontrar um modelo GMM válido.")
        gmm_model = None
    
    # 4. Avaliar modelos originais
    print("\n=== Etapa 4: Avaliando modelos originais ===")
    models_results = {}
    
    # 4.1 Avaliar Random Forest
    rf_results = evaluate_model(
        rf_model, X_val, y_val, 
        threshold=rf_threshold,
        model_name="Random Forest Original"
    )
    models_results["rf_original"] = rf_results
    
    # Registrar métricas no MLflow
    mlflow.log_metrics({
        'rf_original_precision': rf_results['precision'],
        'rf_original_recall': rf_results['recall'],
        'rf_original_f1': rf_results['f1']
    })
    
    # 5. Treinar modelos calibrados
    print("\n=== Etapa 5: Treinando modelos calibrados ===")
    calibrated_models = {}
    
    # 5.1 Calibrar Random Forest
    calibrated_rf = train_calibrated_model(X_train, y_train, rf_model)
    calibrated_models["rf_calibrated"] = calibrated_rf
    
    # 5.2 Calibrar K-means se disponível
    if kmeans_model:
        try:
            calibrated_kmeans = train_calibrated_model(X_train, y_train, kmeans_model)
            calibrated_models["kmeans_calibrated"] = calibrated_kmeans
        except Exception as e:
            print(f"Erro ao calibrar K-means: {str(e)}")
    
    # 5.3 Calibrar GMM se disponível
    if gmm_model:
        try:
            calibrated_gmm = train_calibrated_model(X_train, y_train, gmm_model)
            calibrated_models["gmm_calibrated"] = calibrated_gmm
        except Exception as e:
            print(f"Erro ao calibrar GMM: {str(e)}")
    
    # 6. Avaliar modelos calibrados com threshold padrão
    print("\n=== Etapa 6: Avaliando modelos calibrados com threshold padrão ===")
    
    # 6.1 Avaliar Random Forest calibrado
    rf_cal_results = evaluate_model(
        calibrated_rf, X_val, y_val,
        threshold=rf_threshold,
        model_name="Random Forest Calibrado (threshold original)"
    )
    models_results["rf_calibrated_default"] = rf_cal_results
    
    # Registrar métricas no MLflow
    mlflow.log_metrics({
        'rf_calibrated_default_precision': rf_cal_results['precision'],
        'rf_calibrated_default_recall': rf_cal_results['recall'],
        'rf_calibrated_default_f1': rf_cal_results['f1']
    })
    
    # 6.2 Avaliar K-means calibrado se disponível
    if "kmeans_calibrated" in calibrated_models:
        kmeans_cal_results = evaluate_model(
            calibrated_models["kmeans_calibrated"], X_val, y_val,
            threshold=0.5,  # Threshold padrão para K-means
            model_name="K-means Calibrado (threshold padrão)"
        )
        models_results["kmeans_calibrated_default"] = kmeans_cal_results
        
        # Registrar métricas no MLflow
        mlflow.log_metrics({
            'kmeans_calibrated_default_precision': kmeans_cal_results['precision'],
            'kmeans_calibrated_default_recall': kmeans_cal_results['recall'],
            'kmeans_calibrated_default_f1': kmeans_cal_results['f1']
        })
    
    # 6.3 Avaliar GMM calibrado se disponível
    if "gmm_calibrated" in calibrated_models:
        gmm_cal_results = evaluate_model(
            calibrated_models["gmm_calibrated"], X_val, y_val,
            threshold=0.5,  # Threshold padrão para GMM
            model_name="GMM Calibrado (threshold padrão)"
        )
        models_results["gmm_calibrated_default"] = gmm_cal_results
        
        # Registrar métricas no MLflow
        mlflow.log_metrics({
            'gmm_calibrated_default_precision': gmm_cal_results['precision'],
            'gmm_calibrated_default_recall': gmm_cal_results['recall'],
            'gmm_calibrated_default_f1': gmm_cal_results['f1']
        })
    
    # 7. Otimizar thresholds para os modelos calibrados
    print("\n=== Etapa 7: Otimizando thresholds para modelos calibrados ===")
    optimized_thresholds = {}
    
    # 7.1 Otimizar threshold para Random Forest calibrado
    rf_best_threshold, rf_best_f1, rf_best_precision, rf_best_recall = optimize_threshold(
        calibrated_rf, X_val, y_val
    )
    optimized_thresholds["rf"] = rf_best_threshold
    
    # Registrar threshold otimizado no MLflow
    mlflow.log_metrics({
        'rf_optimized_threshold': rf_best_threshold,
        'rf_optimized_precision': rf_best_precision,
        'rf_optimized_recall': rf_best_recall,
        'rf_optimized_f1': rf_best_f1
    })
    
    # 7.2 Otimizar threshold para K-means calibrado se disponível
    if "kmeans_calibrated" in calibrated_models:
        kmeans_best_threshold, kmeans_best_f1, kmeans_best_precision, kmeans_best_recall = optimize_threshold(
            calibrated_models["kmeans_calibrated"], X_val, y_val
        )
        optimized_thresholds["kmeans"] = kmeans_best_threshold
        
        # Registrar threshold otimizado no MLflow
        mlflow.log_metrics({
            'kmeans_optimized_threshold': kmeans_best_threshold,
            'kmeans_optimized_precision': kmeans_best_precision,
            'kmeans_optimized_recall': kmeans_best_recall,
            'kmeans_optimized_f1': kmeans_best_f1
        })
    
    # 7.3 Otimizar threshold para GMM calibrado se disponível
    if "gmm_calibrated" in calibrated_models:
        gmm_best_threshold, gmm_best_f1, gmm_best_precision, gmm_best_recall = optimize_threshold(
            calibrated_models["gmm_calibrated"], X_val, y_val
        )
        optimized_thresholds["gmm"] = gmm_best_threshold
        
        # Registrar threshold otimizado no MLflow
        mlflow.log_metrics({
            'gmm_optimized_threshold': gmm_best_threshold,
            'gmm_optimized_precision': gmm_best_precision,
            'gmm_optimized_recall': gmm_best_recall,
            'gmm_optimized_f1': gmm_best_f1
        })
    
    # 8. Avaliar modelos calibrados com thresholds otimizados
    print("\n=== Etapa 8: Avaliando modelos calibrados com thresholds otimizados ===")
    
    # 8.1 Avaliar Random Forest calibrado com threshold otimizado
    rf_cal_opt_results = evaluate_model(
        calibrated_rf, X_val, y_val,
        threshold=rf_best_threshold,
        model_name="Random Forest Calibrado (threshold otimizado)"
    )
    models_results["rf_calibrated_optimized"] = rf_cal_opt_results
    
    # Registrar métricas otimizadas no MLflow
    mlflow.log_metrics({
        'rf_calibrated_optimized_precision': rf_cal_opt_results['precision'],
        'rf_calibrated_optimized_recall': rf_cal_opt_results['recall'],
        'rf_calibrated_optimized_f1': rf_cal_opt_results['f1']
    })
    
    # 8.2 Avaliar K-means calibrado com threshold otimizado se disponível
    if "kmeans_calibrated" in calibrated_models and "kmeans" in optimized_thresholds:
        kmeans_cal_opt_results = evaluate_model(
            calibrated_models["kmeans_calibrated"], X_val, y_val,
            threshold=optimized_thresholds["kmeans"],
            model_name="K-means Calibrado (threshold otimizado)"
        )
        models_results["kmeans_calibrated_optimized"] = kmeans_cal_opt_results
        
        # Registrar métricas otimizadas no MLflow
        mlflow.log_metrics({
            'kmeans_calibrated_optimized_precision': kmeans_cal_opt_results['precision'],
            'kmeans_calibrated_optimized_recall': kmeans_cal_opt_results['recall'],
            'kmeans_calibrated_optimized_f1': kmeans_cal_opt_results['f1']
        })
    
    # 8.3 Avaliar GMM calibrado com threshold otimizado se disponível
    if "gmm_calibrated" in calibrated_models and "gmm" in optimized_thresholds:
        gmm_cal_opt_results = evaluate_model(
            calibrated_models["gmm_calibrated"], X_val, y_val,
            threshold=optimized_thresholds["gmm"],
            model_name="GMM Calibrado (threshold otimizado)"
        )
        models_results["gmm_calibrated_optimized"] = gmm_cal_opt_results
        
        # Registrar métricas otimizadas no MLflow
        mlflow.log_metrics({
            'gmm_calibrated_optimized_precision': gmm_cal_opt_results['precision'],
            'gmm_calibrated_optimized_recall': gmm_cal_opt_results['recall'],
            'gmm_calibrated_optimized_f1': gmm_cal_opt_results['f1']
        })
    
    # 9. Salvar modelos e thresholds
    print("\n=== Etapa 9: Salvando modelos ===")
    
    # Preparar dicionários para salvar
    models_to_save = {
        'rf_original': rf_model,
        'rf_calibrated': calibrated_rf
    }
    
    # Adicionar modelos K-means se disponíveis
    if kmeans_model and "kmeans_calibrated" in calibrated_models:
        models_to_save['kmeans_original'] = kmeans_model
        models_to_save['kmeans_calibrated'] = calibrated_models["kmeans_calibrated"]
    
    # Adicionar modelos GMM se disponíveis
    if gmm_model and "gmm_calibrated" in calibrated_models:
        models_to_save['gmm_original'] = gmm_model
        models_to_save['gmm_calibrated'] = calibrated_models["gmm_calibrated"]
    
    # Preparar dicionário de thresholds
    thresholds_to_save = {
        'rf_original': rf_threshold,
        'rf_calibrated_default': rf_threshold,
        'rf_calibrated_optimized': rf_best_threshold
    }
    
    # Adicionar thresholds K-means se disponíveis
    if "kmeans" in optimized_thresholds:
        thresholds_to_save['kmeans_calibrated_default'] = 0.5
        thresholds_to_save['kmeans_calibrated_optimized'] = optimized_thresholds["kmeans"]
    
    # Adicionar thresholds GMM se disponíveis
    if "gmm" in optimized_thresholds:
        thresholds_to_save['gmm_calibrated_default'] = 0.5
        thresholds_to_save['gmm_calibrated_optimized'] = optimized_thresholds["gmm"]
    
    output_dir = save_models(models_to_save, thresholds_to_save, args.output_dir)
    
    # Registrar modelos no MLflow
    mlflow.sklearn.log_model(rf_model, "rf_original_model")
    mlflow.sklearn.log_model(calibrated_rf, "rf_calibrated_model")
    
    if kmeans_model and "kmeans_calibrated" in calibrated_models:
        mlflow.sklearn.log_model(kmeans_model, "kmeans_original_model")
        mlflow.sklearn.log_model(calibrated_models["kmeans_calibrated"], "kmeans_calibrated_model")
    
    if gmm_model and "gmm_calibrated" in calibrated_models:
        mlflow.sklearn.log_model(gmm_model, "gmm_original_model")
        mlflow.sklearn.log_model(calibrated_models["gmm_calibrated"], "gmm_calibrated_model")
    
    # 10. Resumo dos resultados
    print("\n=== Resumo dos Resultados ===")
    results_df = pd.DataFrame(list(models_results.values()))
    print(results_df[['model_name', 'threshold', 'precision', 'recall', 'f1']])
    
    # Salvar resumo em CSV
    results_path = os.path.join(args.output_dir, "results_summary.csv")
    results_df.to_csv(results_path, index=False)
    
    print("\n=== Treinamento e avaliação concluídos com sucesso! ===")
    print(f"Modelos salvos em: {output_dir}")
    print(f"MLflow run ID: {run_id}")
    
    return output_dir, run_id

# ======================================
# EXECUTAR O SCRIPT
# ======================================

if __name__ == "__main__":
    main()