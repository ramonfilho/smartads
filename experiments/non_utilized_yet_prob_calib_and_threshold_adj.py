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
import warnings
warnings.filterwarnings('ignore')

# Adicionar diretório raiz ao path
project_root = os.path.abspath(os.path.dirname(__file__))
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

def load_model(model_path):
    """Carrega um modelo treinado a partir do caminho especificado."""
    try:
        print(f"Carregando modelo de: {model_path}")
        if model_path.startswith("runs:"):
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
    # Tentar usar a função existente do projeto
    try:
        from src.evaluation.baseline_model import prepare_data_for_training
        
        # Verificar se o arquivo existe no caminho fornecido
        if not os.path.exists(data_path):
            # Tentar caminhos alternativos
            alt_path = os.path.join(project_root, "data", "03_feature_selection_textv2", 
                                       os.path.basename(data_path))
            if os.path.exists(alt_path):
                print(f"Usando caminho alternativo: {alt_path}")
                data_path = alt_path
            else:
                raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
        
        # Usar a função existente
        data_dict = prepare_data_for_training(data_path)
        
        df = data_dict['train_df']
        X = data_dict['X_train']
        y = data_dict['y_train']
        target_col = data_dict['target_col']
        
    except (ImportError, KeyError) as e:
        print(f"Erro ao usar função existente: {e}")
        print("Usando função de carregamento simplificada.")
        
        # Função simplificada para carregar dados
        df = pd.read_csv(data_path)
        
        # Identificar coluna target
        target_candidates = ['target', 'label', 'class', 'y']
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
    parser.add_argument('--baseline_model_path', default=None,
                      help='Caminho para o modelo baseline')
    parser.add_argument('--output_dir', default=None,
                      help='Diretório para salvar os modelos')
    
    args = parser.parse_args()
    
    # Definir valores padrão se não fornecidos
    if args.mlflow_dir is None:
        args.mlflow_dir = os.path.join(project_root, "models", "mlflow")
    
    if args.data_path is None:
        args.data_path = os.path.join(project_root, "data", "03_3_feature_selection_text_code6", "train.csv")
    
    if args.validation_path is None:
        args.validation_path = os.path.join(project_root, "data", "03_3_feature_selection_text_code6", "validation.csv")
    
    if args.baseline_model_path is None:
        # Buscar modelo mais recente no MLflow
        try:
            from src.evaluation.baseline_model import get_latest_random_forest_run
            run_id, threshold, model_uri = get_latest_random_forest_run(args.mlflow_dir)
            if model_uri:
                args.baseline_model_path = model_uri
                print(f"Usando modelo mais recente do MLflow: {model_uri}")
            else:
                args.baseline_model_path = os.path.join(project_root, "models", "baseline", "random_forest.joblib")
                print(f"Usando modelo padrão: {args.baseline_model_path}")
        except ImportError:
            args.baseline_model_path = os.path.join(project_root, "models", "baseline", "random_forest.joblib")
            print(f"Usando modelo padrão: {args.baseline_model_path}")
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(project_root, "models", f"calibrated_{timestamp}")
    
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
            'baseline_model_path': args.baseline_model_path
        })
        
        # 1. Carregar dados
        print("\n=== Etapa 1: Carregando dados ===")
        df_train, X_train, y_train, target_col = load_data(args.data_path)
        df_val, X_val, y_val, _ = load_data(args.validation_path)
        
        # 2. Carregar modelo baseline
        print("\n=== Etapa 2: Carregando modelo baseline ===")
        baseline_model = load_model(args.baseline_model_path)
        
        if baseline_model is None:
            print("ERRO: Não foi possível carregar o modelo baseline.")
            return
        
        # 3. Avaliar modelo baseline
        baseline_threshold = 0.12  # Threshold padrão do modelo baseline
        
        # Avaliar no conjunto de validação
        baseline_results = evaluate_model(
            baseline_model, X_val, y_val, 
            threshold=baseline_threshold,
            model_name="Baseline"
        )
        
        # Registrar métricas baseline no MLflow
        mlflow.log_metrics({
            'baseline_precision': baseline_results['precision'],
            'baseline_recall': baseline_results['recall'],
            'baseline_f1': baseline_results['f1']
        })
        
        # 4. Treinar modelo calibrado (Fase 1)
        print("\n=== Etapa 4: Treinando modelo calibrado (Fase 1) ===")
        calibrated_model = train_calibrated_model(X_train, y_train, baseline_model)
        
        # 5. Avaliar modelo calibrado com threshold padrão
        calibrated_results_default = evaluate_model(
            calibrated_model, X_val, y_val,
            threshold=baseline_threshold,
            model_name="Calibrado (threshold padrão)"
        )
        
        # Registrar métricas do modelo calibrado no MLflow
        mlflow.log_metrics({
            'calibrated_default_precision': calibrated_results_default['precision'],
            'calibrated_default_recall': calibrated_results_default['recall'],
            'calibrated_default_f1': calibrated_results_default['f1']
        })
        
        # 6. Otimizar threshold para o modelo calibrado (Fase 2)
        print("\n=== Etapa 6: Otimizando threshold para modelo calibrado (Fase 2) ===")
        best_threshold, best_f1, best_precision, best_recall = optimize_threshold(
            calibrated_model, X_val, y_val
        )
        
        # Registrar threshold otimizado no MLflow
        mlflow.log_metrics({
            'optimized_threshold': best_threshold,
            'optimized_precision': best_precision,
            'optimized_recall': best_recall,
            'optimized_f1': best_f1
        })
        
        # 7. Avaliar modelo calibrado com threshold otimizado
        calibrated_results_optimized = evaluate_model(
            calibrated_model, X_val, y_val,
            threshold=best_threshold,
            model_name="Calibrado (threshold otimizado)"
        )
        
        # Registrar métricas otimizadas no MLflow
        mlflow.log_metrics({
            'calibrated_optimized_precision': calibrated_results_optimized['precision'],
            'calibrated_optimized_recall': calibrated_results_optimized['recall'],
            'calibrated_optimized_f1': calibrated_results_optimized['f1']
        })
        
        # 8. Salvar modelos e thresholds
        print("\n=== Etapa 8: Salvando modelos ===")
        models_dict = {
            'baseline': baseline_model,
            'calibrated': calibrated_model
        }
        
        thresholds_dict = {
            'baseline': baseline_threshold,
            'calibrated_default': baseline_threshold,
            'calibrated_optimized': best_threshold
        }
        
        output_dir = save_models(models_dict, thresholds_dict, args.output_dir)
        
        # Registrar modelos no MLflow
        mlflow.sklearn.log_model(baseline_model, "baseline_model")
        mlflow.sklearn.log_model(calibrated_model, "calibrated_model")
        
        # 9. Resumo dos resultados
        print("\n=== Resumo dos Resultados ===")
        results = [baseline_results, calibrated_results_default, calibrated_results_optimized]
        
        results_df = pd.DataFrame(results)
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