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
warnings.filterwarnings('ignore')

# Adicionar diretório raiz ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.append(project_root)
print(f"Diretório raiz adicionado ao path: {project_root}")

# 1. Funções de Utilidade
def setup_mlflow(mlflow_dir, experiment_name="smart_ads_calibrated"):
    """Configura o MLflow para tracking de experimentos."""
    # Importar função de setup do MLflow do módulo existente
    try:
        from src.evaluation.mlflow_utils import setup_mlflow_tracking
        experiment_id = setup_mlflow_tracking(
            tracking_dir=mlflow_dir,
            experiment_name=experiment_name,
            clean_previous=False
        )
    except ImportError:
        print("Módulo mlflow_utils não encontrado, usando setup simplificado.")
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        # Verificar se o experimento já existe
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            print(f"Experimento '{experiment_name}' já existe (ID: {experiment.experiment_id})")
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Experimento '{experiment_name}' criado (ID: {experiment_id})")
    
    return experiment_id

def find_best_random_forest_model(mlflow_dir):
    """Encontra o modelo Random Forest com melhor desempenho no MLflow."""
    print(f"Procurando o melhor modelo Random Forest em: {mlflow_dir}")
    
    try:
        # Configurar MLflow
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        
        # Buscar todos os experimentos
        experiments = mlflow.search_experiments()
        
        best_f1 = -1
        best_model_uri = None
        best_threshold = 0.5
        
        # Iterar sobre todos os experimentos
        for exp in experiments:
            print(f"Verificando experimento: {exp.name} (ID: {exp.experiment_id})")
            
            # Buscar runs do experimento
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            
            # Filtrar runs com modelos Random Forest
            rf_runs = runs[runs['tags.mlflow.runName'].str.contains('random_forest', case=False, na=False)]
            
            if rf_runs.empty:
                print(f"  Nenhum modelo RandomForest encontrado no experimento {exp.name}")
                continue
            
            # Iterar sobre as runs de Random Forest
            for _, run in rf_runs.iterrows():
                run_id = run.run_id
                print(f"  Encontrado run: {run_id}")
                
                # Verificar se tem métrica F1
                if 'metrics.f1' in run and not np.isnan(run['metrics.f1']):
                    f1 = run['metrics.f1']
                    
                    # Verificar se é melhor que o atual
                    if f1 > best_f1:
                        best_f1 = f1
                        
                        # Verificar se existe threshold
                        if 'metrics.threshold' in run and not np.isnan(run['metrics.threshold']):
                            best_threshold = run['metrics.threshold']
                        
                        # Construir URI do modelo
                        best_model_uri = f"runs:/{run_id}/model"
                        print(f"  Novo melhor modelo: {best_model_uri} (F1: {best_f1:.4f}, Threshold: {best_threshold:.4f})")
        
        if best_model_uri:
            print(f"Melhor modelo Random Forest encontrado: {best_model_uri} (F1: {best_f1:.4f})")
            return best_model_uri, best_threshold
        else:
            print("Nenhum modelo Random Forest encontrado com métricas válidas.")
            return None, 0.5
    
    except Exception as e:
        print(f"Erro ao buscar melhor modelo Random Forest: {str(e)}")
        return None, 0.5

def find_best_cluster_model(artifacts_dir, model_type):
    """Encontra o melhor modelo de cluster (K-means ou GMM) nos artefatos."""
    print(f"Procurando o melhor modelo {model_type} em: {artifacts_dir}")
    
    try:
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
        
        if not model_files:
            print(f"Nenhum modelo {model_type} encontrado.")
            return None
        
        print(f"Modelos {model_type} encontrados:")
        for file in model_files:
            print(f"  - {file}")
        
        # Ordenar por data de modificação (mais recente primeiro)
        model_files.sort(key=os.path.getmtime, reverse=True)
        
        # Retornar o arquivo mais recente
        best_model_path = model_files[0]
        print(f"Selecionado modelo {model_type} mais recente: {best_model_path}")
        
        return best_model_path
    
    except Exception as e:
        print(f"Erro ao buscar modelo {model_type}: {str(e)}")
        return None

def load_model(model_path):
    """Carrega um modelo treinado a partir do caminho especificado."""
    try:
        print(f"Carregando modelo de: {model_path}")
        if isinstance(model_path, str) and model_path.startswith("runs:"):
            # Carregar modelo do MLflow
            import mlflow.sklearn
            model = mlflow.sklearn.load_model(model_path)
        else:
            # Carregar modelo de arquivo
            model = joblib.load(model_path)
        print("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        return None

def load_data(data_path, verbose=True):
    """Carrega dados de treinamento/validação."""
    # Verificar se o arquivo existe no caminho fornecido
    if not os.path.exists(data_path):
        print(f"ERRO: Arquivo não encontrado: {data_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
    
    try:
        # Função simplificada para carregar dados
        df = pd.read_csv(data_path)
        
        # Identificar coluna target
        target_candidates = ['target', 'label', 'class', 'y', 'converted']
        target_col = next((col for col in target_candidates if col in df.columns), None)
        
        if target_col is None:
            print("Coluna target não encontrada. Usando última coluna.")
            target_col = df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
    
        if verbose:
            print(f"Dados carregados - {X.shape[0]} exemplos, {X.shape[1]} features")
            print(f"Taxa de conversão: {y.mean():.4f}")
        
        return df, X, y, target_col
    
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        raise

# 2. Calibração de Probabilidades
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

# 3. Otimização de Threshold
def optimize_threshold(model, X, y):
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
    
    # Salvar figura
    os.makedirs('plots', exist_ok=True)
    plt_path = os.path.join('plots', f'pr_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plt_path)
    print(f"Curva PR salva em: {plt_path}")
    
    return best_threshold, best_f1, best_precision, best_recall

# 4. Avaliação do Modelo
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

# 5. Salvar Modelos
def save_models(models_dict, thresholds_dict, output_dir):
    """Salva os modelos treinados e seus thresholds."""
    print(f"\n=== Salvando modelos em {output_dir} ===")
    
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar cada modelo
    for name, model in models_dict.items():
        model_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"  - Modelo {name} salvo em: {model_path}")
    
    # Salvar thresholds
    thresholds_path = os.path.join(output_dir, "thresholds.joblib")
    joblib.dump(thresholds_dict, thresholds_path)
    print(f"  - Thresholds salvos em: {thresholds_path}")
    
    # Salvar configuração
    config = {
        'models': list(models_dict.keys()),
        'thresholds': thresholds_dict,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_path = os.path.join(output_dir, "config.joblib")
    joblib.dump(config, config_path)
    
    return output_dir

# 6. Função Principal
def main():
    parser = argparse.ArgumentParser(description='Implementa calibração e ajuste de threshold')
    parser.add_argument('--mlflow_dir', default=None, 
                      help='Diretório do MLflow tracking')
    parser.add_argument('--data_path', default=None,
                      help='Caminho para o dataset de treinamento')
    parser.add_argument('--validation_path', default=None,
                      help='Caminho para o dataset de validação')
    parser.add_argument('--artifacts_dir', default=None,
                      help='Diretório para buscar modelos de cluster')
    parser.add_argument('--output_dir', default=None,
                      help='Diretório para salvar os modelos')
    
    args = parser.parse_args()
    
    # Definir valores padrão se não fornecidos
    if args.mlflow_dir is None:
        args.mlflow_dir = os.path.join(project_root, "mlflow")
    
    if args.data_path is None:
        args.data_path = os.path.join(project_root, "data", "03_3_feature_selection_text_code6", "train.csv")
    
    if args.validation_path is None:
        args.validation_path = os.path.join(project_root, "data", "03_3_feature_selection_text_code6", "validation.csv")
    
    if args.artifacts_dir is None:
        args.artifacts_dir = os.path.join(project_root, "models", "artifacts")
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(project_root, "models", f"calibrated_{timestamp}")
    
    # Verificar se os caminhos existem
    if not os.path.exists(args.data_path):
        print(f"ERRO: Arquivo de treinamento não encontrado: {args.data_path}")
        return
    
    if not os.path.exists(args.validation_path):
        print(f"ERRO: Arquivo de validação não encontrado: {args.validation_path}")
        return
    
    # Configurar MLflow
    experiment_id = setup_mlflow(args.mlflow_dir, "smart_ads_calibrated")
    
    # Iniciar run do MLflow
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"MLflow run iniciado: {run_id}")
        
        # Registrar parâmetros
        mlflow.log_params({
            'data_path': args.data_path,
            'validation_path': args.validation_path,
            'artifacts_dir': args.artifacts_dir
        })
        
        # 1. Carregar dados
        print("\n=== Etapa 1: Carregando dados ===")
        df_train, X_train, y_train, target_col = load_data(args.data_path)
        df_val, X_val, y_val, _ = load_data(args.validation_path)
        
        # 2. Buscar e carregar os melhores modelos
        print("\n=== Etapa 2: Buscando e carregando os melhores modelos ===")
        
        # 2.1 Random Forest
        rf_model_uri, rf_threshold = find_best_random_forest_model(args.mlflow_dir)
        if rf_model_uri:
            rf_model = load_model(rf_model_uri)
        else:
            print("ERRO: Não foi possível encontrar um modelo Random Forest válido.")
            return
        
        # 2.2 K-means
        kmeans_model_path = find_best_cluster_model(args.artifacts_dir, "kmeans")
        if kmeans_model_path:
            kmeans_model = load_model(kmeans_model_path)
        else:
            print("AVISO: Não foi possível encontrar um modelo K-means válido.")
            kmeans_model = None
        
        # 2.3 GMM
        gmm_model_path = find_best_cluster_model(args.artifacts_dir, "gmm")
        if gmm_model_path:
            gmm_model = load_model(gmm_model_path)
        else:
            print("AVISO: Não foi possível encontrar um modelo GMM válido.")
            gmm_model = None
        
        # 3. Avaliar modelos originais
        print("\n=== Etapa 3: Avaliando modelos originais ===")
        models_results = {}
        
        # 3.1 Avaliar Random Forest
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
        
        # 4. Treinar modelos calibrados
        print("\n=== Etapa 4: Treinando modelos calibrados ===")
        calibrated_models = {}
        
        # 4.1 Calibrar Random Forest
        calibrated_rf = train_calibrated_model(X_train, y_train, rf_model)
        calibrated_models["rf_calibrated"] = calibrated_rf
        
        # 4.2 Calibrar K-means se disponível
        if kmeans_model:
            try:
                calibrated_kmeans = train_calibrated_model(X_train, y_train, kmeans_model)
                calibrated_models["kmeans_calibrated"] = calibrated_kmeans
            except Exception as e:
                print(f"Erro ao calibrar K-means: {str(e)}")
        
        # 4.3 Calibrar GMM se disponível
        if gmm_model:
            try:
                calibrated_gmm = train_calibrated_model(X_train, y_train, gmm_model)
                calibrated_models["gmm_calibrated"] = calibrated_gmm
            except Exception as e:
                print(f"Erro ao calibrar GMM: {str(e)}")
        
        # 5. Avaliar modelos calibrados com threshold padrão
        print("\n=== Etapa 5: Avaliando modelos calibrados com threshold padrão ===")
        
        # 5.1 Avaliar Random Forest calibrado
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
        
        # 5.2 Avaliar K-means calibrado se disponível
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
        
        # 5.3 Avaliar GMM calibrado se disponível
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
        
        # 6. Otimizar thresholds para os modelos calibrados
        print("\n=== Etapa 6: Otimizando thresholds para modelos calibrados ===")
        optimized_thresholds = {}
        
        # 6.1 Otimizar threshold para Random Forest calibrado
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
        
        # 6.2 Otimizar threshold para K-means calibrado se disponível
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
        
        # 6.3 Otimizar threshold para GMM calibrado se disponível
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
        
        # 7. Avaliar modelos calibrados com thresholds otimizados
        print("\n=== Etapa 7: Avaliando modelos calibrados com thresholds otimizados ===")
        
        # 7.1 Avaliar Random Forest calibrado com threshold otimizado
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
        
        # 7.2 Avaliar K-means calibrado com threshold otimizado se disponível
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
        
        # 7.3 Avaliar GMM calibrado com threshold otimizado se disponível
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
        
        # 8. Salvar modelos e thresholds
        print("\n=== Etapa 8: Salvando modelos ===")
        
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
        
        # 9. Resumo dos resultados
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

if __name__ == "__main__":
    main()