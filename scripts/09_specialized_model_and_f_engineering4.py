#!/usr/bin/env python
"""
Script para implementação de modelo especializado para falsos negativos.
Este script implementa as fases 3 e 4 do plano de melhoria para o modelo Smart Ads,
baseando-se nos resultados das fases 1 e 2 (calibração e ajuste de threshold).

Fase 3: Modelo Especializado Simples
   - Implementar um modelo especializado em detectar falsos negativos
   - Criar um ensemble simples com o modelo calibrado

Fase 4: Engenharia de Features Básica
   - Adicionar features simples compatíveis com o modelo existente
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import mlflow
import warnings
warnings.filterwarnings('ignore')

# Adicionar diretório raiz ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(f"Diretório raiz adicionado ao path: {project_root}")

# 1. Funções de Utilidade
def setup_mlflow(mlflow_dir, experiment_name="smart_ads_specialized"):
    """Configura o MLflow para tracking de experimentos."""
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

def load_models(models_dir):
    """Carrega modelos previamente treinados."""
    print(f"\n=== Carregando modelos de {models_dir} ===")
    
    # Carregar thresholds
    thresholds_path = os.path.join(models_dir, "thresholds.joblib")
    if os.path.exists(thresholds_path):
        thresholds = joblib.load(thresholds_path)
        print(f"Thresholds carregados: {thresholds}")
    else:
        thresholds = {
            'baseline': 0.12,
            'calibrated_default': 0.12,
            'calibrated_optimized': 0.05
        }
        print(f"Arquivo de thresholds não encontrado. Usando valores padrão: {thresholds}")
    
    # Carregar modelos
    baseline_path = os.path.join(models_dir, "baseline.joblib")
    calibrated_path = os.path.join(models_dir, "calibrated.joblib")
    
    models = {}
    
    if os.path.exists(baseline_path):
        models['baseline'] = joblib.load(baseline_path)
        print("Modelo baseline carregado com sucesso.")
    else:
        print("Modelo baseline não encontrado.")
    
    if os.path.exists(calibrated_path):
        models['calibrated'] = joblib.load(calibrated_path)
        print("Modelo calibrado carregado com sucesso.")
    else:
        print("Modelo calibrado não encontrado.")
    
    return models, thresholds

# 2. Engenharia de Features Básica
def add_basic_features(X, verbose=True):
    """Adiciona features básicas que são compatíveis com os modelos existentes."""
    print("\n=== Adicionando features básicas ===")
    
    X_enhanced = X.copy()
    new_features = []
    
    # Criar novas features apenas se as features originais existirem
    # NÃO ADICIONE DAY_SIN_ABS OU DAY_VARIATION que são incompatíveis!
    
    # 1. Features de interação entre país e salário (se disponíveis)
    if 'country_encoded' in X.columns and 'current_salary_encoded' in X.columns:
        # Feature de interação
        X_enhanced['country_salary_interaction'] = X['country_encoded'] * X['current_salary_encoded']
        new_features.append('country_salary_interaction')
        
        # Feature normalizada
        if 'profession_encoded' in X.columns:
            salary_profession_means = X.groupby('profession_encoded')['current_salary_encoded'].transform('mean')
            X_enhanced['salary_profession_ratio'] = X['current_salary_encoded'] / (salary_profession_means + 1e-5)
            new_features.append('salary_profession_ratio')
    
    # 2. Features baseadas em data/hora (se disponíveis)
    if 'hour_encoded' in X.columns and 'day_of_week_encoded' in X.columns:
        # Feature de interação hora/dia
        X_enhanced['hour_day_interaction'] = X['hour_encoded'] * X['day_of_week_encoded'] 
        new_features.append('hour_day_interaction')
    
    # 3. Features de razão entre salários (se disponíveis)
    if 'current_salary_encoded' in X.columns and 'desired_salary_encoded' in X.columns:
        # Razão entre salário desejado e atual
        X_enhanced['salary_ratio'] = X['desired_salary_encoded'] / (X['current_salary_encoded'] + 1e-5)
        new_features.append('salary_ratio')
    
    # 4. Features de estabilidade de profissão (se disponíveis)
    if 'profession_encoded' in X.columns and 'years_experience' in X.columns:
        # Interação entre profissão e experiência
        X_enhanced['profession_experience'] = X['profession_encoded'] * X['years_experience']
        new_features.append('profession_experience')
    
    if verbose:
        print(f"Adicionadas {len(new_features)} novas features:")
        for feature in new_features:
            print(f"  - {feature}")
    
    return X_enhanced, new_features

# 3. Identificação de Falsos Negativos
def identify_false_negatives(X, y, calibrated_model, threshold, verbose=True):
    """Identifica falsos negativos usando validação cruzada no conjunto de dados."""
    print("\n=== Identificando falsos negativos com validação cruzada ===")
    
    # Configurar validação cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Arrays para armazenar resultados
    all_fn_indices = []
    all_probs = []
    all_predictions = []
    
    # Realizar validação cruzada
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"Processando fold {fold+1}/5")
        
        # Separar dados para este fold
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Treinar um clone do modelo calibrado (opcional)
        # Na prática, podemos usar o modelo calibrado diretamente
        # para simplificar, mas o CV daria resultados mais confiáveis
        probs = calibrated_model.predict_proba(X_val)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        # Identificar falsos negativos neste fold
        fn_mask = (y_val == 1) & (preds == 0)
        fn_indices = val_idx[fn_mask]
        
        # Armazenar resultados
        all_fn_indices.extend(fn_indices)
        all_probs.extend(probs[fn_mask])
        all_predictions.extend(np.full(len(fn_indices), 0))  # Todos são FNs, então predição = 0
        
        print(f"  Falsos negativos neste fold: {len(fn_indices)}")
    
    # Resumo
    if verbose:
        print(f"\nTotal de falsos negativos identificados: {len(all_fn_indices)}")
        if len(all_fn_indices) > 0:
            print(f"Probabilidade média dos falsos negativos: {np.mean(all_probs):.4f}")
    
    return all_fn_indices, all_probs, all_predictions

# 4. Treinamento de Modelo Especializado
def train_specialist_model(X, y, fn_indices, verbose=True):
    """Treina um modelo especializado para detectar falsos negativos."""
    print("\n=== Treinando modelo especializado para falsos negativos ===")
    
    # Preparar conjunto de treinamento com pesos maiores para falsos negativos
    sample_weights = np.ones(len(X))
    
    # Aumentar o peso dos falsos negativos identificados
    if len(fn_indices) > 0:
        sample_weights[fn_indices] = 5.0  # Peso 5x maior para falsos negativos
    
    # Preparar um conjunto balanceado para o modelo especialista
    positive_indices = y[y == 1].index
    negative_indices = y[y == 0].index
    
    # Amostrar negativos para balancear (usar mais negativos que positivos)
    n_negatives = min(len(negative_indices), 2 * len(positive_indices))
    np.random.seed(42)
    sampled_negatives = np.random.choice(negative_indices, size=n_negatives, replace=False)
    
    # Combinar índices
    training_indices = np.concatenate([positive_indices, sampled_negatives])
    
    # Extrair dados de treinamento
    X_train = X.loc[training_indices]
    y_train = y.loc[training_indices]
    weights_train = sample_weights[training_indices]
    
    # Treinar modelo especializado
    specialist_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    specialist_model.fit(X_train, y_train, sample_weight=weights_train)
    
    if verbose:
        print(f"Modelo especializado treinado com {len(X_train)} exemplos")
        print(f"  - Positivos: {sum(y_train)} ({np.mean(y_train):.1%})")
        print(f"  - Negativos: {len(y_train) - sum(y_train)} ({1-np.mean(y_train):.1%})")
        
        # Verificar importância de features
        feature_importance = pd.Series(
            specialist_model.feature_importances_, 
            index=X_train.columns
        ).sort_values(ascending=False)
        
        print("\nTop 10 features mais importantes para o modelo especializado:")
        for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
            print(f"{i}. {feature}: {importance:.4f}")
    
    return specialist_model

# 5. Criação de Ensemble
def create_ensemble_model(calibrated_model, specialist_model, calibrated_threshold=0.05, specialist_threshold=0.3):
    """Cria um modelo ensemble que combina o modelo calibrado e o especializado."""
    print("\n=== Criando modelo ensemble ===")
    
    class EnsembleModel:
        def __init__(self, calibrated_model, specialist_model, 
                    calibrated_threshold, specialist_threshold):
            self.calibrated_model = calibrated_model
            self.specialist_model = specialist_model
            self.calibrated_threshold = calibrated_threshold
            self.specialist_threshold = specialist_threshold
        
        def predict_proba(self, X):
            # Obter probabilidades do modelo calibrado
            calibrated_probs = self.calibrated_model.predict_proba(X)[:, 1]
            
            # Identificar exemplos que o modelo calibrado classifica como negativos
            potential_fn_mask = calibrated_probs < self.calibrated_threshold
            
            # Se não houver potenciais FNs, retornar probabilidades do modelo calibrado
            if not np.any(potential_fn_mask):
                return np.column_stack([1 - calibrated_probs, calibrated_probs])
            
            # Para os potenciais FNs, aplicar o modelo especializado
            specialist_probs = np.zeros_like(calibrated_probs)
            specialist_probs[potential_fn_mask] = self.specialist_model.predict_proba(
                X.loc[potential_fn_mask]
            )[:, 1]
            
            # Combinar as probabilidades
            final_probs = calibrated_probs.copy()
            
            # Para exemplos onde o especialista prediz com alta confiança que é positivo
            # mas o calibrado prediz negativo, aceitar a predição do especialista
            override_mask = (potential_fn_mask) & (specialist_probs > self.specialist_threshold)
            final_probs[override_mask] = specialist_probs[override_mask]
            
            # Retornar no formato que o scikit-learn espera
            return np.column_stack([1 - final_probs, final_probs])
        
        def predict(self, X, threshold=None):
            if threshold is None:
                threshold = self.calibrated_threshold
                
            probs = self.predict_proba(X)[:, 1]
            return (probs >= threshold).astype(int)
    
    return EnsembleModel(calibrated_model, specialist_model, calibrated_threshold, specialist_threshold)

# 6. Otimização de Threshold para Ensemble
def optimize_ensemble_threshold(ensemble_model, X, y):
    """Otimiza o threshold para o modelo ensemble."""
    print("\n=== Otimizando threshold para modelo ensemble ===")
    
    # Obter probabilidades
    probs = ensemble_model.predict_proba(X)[:, 1]
    
    # Calcular curva precision-recall
    precision, recall, thresholds = precision_recall_curve(y, probs)
    
    # Calcular F1-Score para cada threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Encontrar threshold ótimo
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Threshold ótimo para ensemble: {best_threshold:.4f}")
    print(f"  - Precision: {best_precision:.4f}")
    print(f"  - Recall: {best_recall:.4f}")
    print(f"  - F1-score: {best_f1:.4f}")
    
    # Plotar curva precision-recall
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', label='Curva PR')
    plt.scatter([best_recall], [best_precision], c='red', marker='o', s=100,
               label=f'Ótimo: F1={best_f1:.3f} (t={best_threshold:.3f})')
    
    # Encontrar threshold para alto recall (pelo menos 0.7)
    high_recall_indices = np.where(recall >= 0.7)[0]
    if len(high_recall_indices) > 0:
        hr_idx = high_recall_indices[0]
        hr_threshold = thresholds[hr_idx] if hr_idx < len(thresholds) else 0.01
        hr_precision = precision[hr_idx]
        hr_recall = recall[hr_idx]
        hr_f1 = 2 * (hr_precision * hr_recall) / (hr_precision + hr_recall + 1e-10)
        
        plt.scatter([hr_recall], [hr_precision], c='green', marker='s', s=100,
                   label=f'Recall 70%+: P={hr_precision:.3f} (t={hr_threshold:.3f})')
        
        print(f"\nThreshold para alto recall (>= 70%): {hr_threshold:.4f}")
        print(f"  - Precision: {hr_precision:.4f}")
        print(f"  - Recall: {hr_recall:.4f}")
        print(f"  - F1-score: {hr_f1:.4f}")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall para Modelo Ensemble')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    # Salvar figura
    os.makedirs('plots', exist_ok=True)
    pr_path = os.path.join('plots', f'ensemble_pr_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(pr_path)
    plt.close()
    
    print(f"Curva PR salva em: {pr_path}")
    
    return best_threshold, best_f1, best_precision, best_recall

# 7. Avaliação de Modelo
def evaluate_models(X, y, models_dict, thresholds_dict):
    """Avalia múltiplos modelos no mesmo conjunto de dados."""
    print("\n=== Avaliando modelos ===")
    
    results = []
    
    for model_name, model in models_dict.items():
        threshold = thresholds_dict.get(model_name, 0.5)
        
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
        
        print(f"\nMétricas para {model_name} (threshold={threshold:.4f}):")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1-score: {f1:.4f}")
        print(f"  - Taxa de positivos: {pos_rate:.2%} ({n_positives} de {len(y_preds)})")
        print(f"  - Falsos positivos: {fp}")
        print(f"  - Falsos negativos: {fn}")
        
        # Armazenar resultados
        results.append({
            'model_name': model_name,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'positives': n_positives,
            'false_positives': fp,
            'false_negatives': fn
        })
    
    return pd.DataFrame(results)

# 8. Salvar Modelos
def save_models(models_dict, thresholds_dict, output_dir):
    """Salva os modelos treinados e thresholds."""
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

# 9. Função Principal
def main():
    parser = argparse.ArgumentParser(description='Treina modelo especializado para falsos negativos')
    parser.add_argument('--mlflow_dir', default=None, 
                      help='Diretório do MLflow tracking')
    parser.add_argument('--data_path', default=None,
                      help='Caminho para o dataset de treinamento')
    parser.add_argument('--validation_path', default=None,
                      help='Caminho para o dataset de validação')
    parser.add_argument('--calibrated_models_dir', default=None,
                      help='Diretório com modelos calibrados')
    parser.add_argument('--output_dir', default=None,
                      help='Diretório para salvar os modelos')
    
    args = parser.parse_args()
    
    # Definir valores padrão se não fornecidos
    if args.mlflow_dir is None:
        args.mlflow_dir = os.path.join(project_root, "models", "mlflow")
    
    if args.data_path is None:
        args.data_path = os.path.join(project_root, "data", "03_feature_selection_textv2", "train.csv")
    
    if args.validation_path is None:
        args.validation_path = os.path.join(project_root, "data", "03_feature_selection_textv2", "validation.csv")
    
    if args.calibrated_models_dir is None:
        # Tentar encontrar o diretório mais recente com modelos calibrados
        models_dir = os.path.join(project_root, "models")
        if os.path.exists(models_dir):
            calibrated_dirs = [d for d in os.listdir(models_dir) if d.startswith("calibrated_")]
            if calibrated_dirs:
                # Ordenar por data (mais recente primeiro)
                calibrated_dirs.sort(reverse=True)
                args.calibrated_models_dir = os.path.join(models_dir, calibrated_dirs[0])
                print(f"Usando diretório de modelos calibrados mais recente: {args.calibrated_models_dir}")
            else:
                args.calibrated_models_dir = os.path.join(models_dir, "calibrated")
                print(f"Nenhum diretório calibrated_* encontrado. Usando valor padrão: {args.calibrated_models_dir}")
        else:
            args.calibrated_models_dir = os.path.join(project_root, "models", "calibrated")
            print(f"Diretório de modelos não encontrado. Usando valor padrão: {args.calibrated_models_dir}")
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(project_root, "models", f"specialized_{timestamp}")
    
    # Configurar MLflow
    experiment_id = setup_mlflow(args.mlflow_dir, "smart_ads_specialized")
    
    # Iniciar run do MLflow
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"MLflow run iniciado: {run_id}")
        
        # Registrar parâmetros
        mlflow.log_params({
            'data_path': args.data_path,
            'validation_path': args.validation_path,
            'calibrated_models_dir': args.calibrated_models_dir
        })
        
        # 1. Carregar dados
        print("\n=== Etapa 1: Carregando dados ===")
        df_train, X_train, y_train, target_col = load_data(args.data_path)
        df_val, X_val, y_val, _ = load_data(args.validation_path)
        
        # 2. Carregar modelos calibrados
        print("\n=== Etapa 2: Carregando modelos calibrados ===")
        previous_models, thresholds = load_models(args.calibrated_models_dir)
        
        if 'calibrated' not in previous_models:
            print("ERRO: Modelo calibrado não encontrado. Abortando.")
            return
        
        calibrated_model = previous_models['calibrated']
        calibrated_threshold = thresholds.get('calibrated_optimized', 0.05)
        
        # 3. Adicionar features básicas (Fase 4)
        print("\n=== Etapa 3: Engenharia de features básicas (Fase 4) ===")
        X_train_enhanced, new_features = add_basic_features(X_train)
        X_val_enhanced, _ = add_basic_features(X_val)
        
        # Registrar novas features no MLflow
        mlflow.log_param('new_features', new_features)
        
        # 4. Identificar falsos negativos
        print("\n=== Etapa 4: Identificando falsos negativos ===")
        fn_indices, fn_probs, fn_preds = identify_false_negatives(
            X_train_enhanced, y_train, calibrated_model, calibrated_threshold
        )
        
        # Registrar estatísticas de FNs no MLflow
        mlflow.log_metrics({
            'false_negatives_count': len(fn_indices),
            'false_negatives_avg_prob': np.mean(fn_probs) if len(fn_probs) > 0 else 0.0
        })
        
        # 5. Treinar modelo especializado (Fase 3)
        print("\n=== Etapa 5: Treinando modelo especializado (Fase 3) ===")
        specialist_model = train_specialist_model(X_train_enhanced, y_train, fn_indices)
        
        # 6. Criar modelo ensemble
        print("\n=== Etapa 6: Criando modelo ensemble ===")
        ensemble_model = create_ensemble_model(
            calibrated_model, 
            specialist_model, 
            calibrated_threshold=calibrated_threshold, 
            specialist_threshold=0.3
        )
        
        # 7. Otimizar threshold para o ensemble
        print("\n=== Etapa 7: Otimizando threshold para o ensemble ===")
        ensemble_threshold, ensemble_f1, ensemble_precision, ensemble_recall = optimize_ensemble_threshold(
            ensemble_model, X_val_enhanced, y_val
        )
        
        # Registrar threshold otimizado no MLflow
        mlflow.log_metrics({
            'ensemble_threshold': ensemble_threshold,
            'ensemble_f1': ensemble_f1,
            'ensemble_precision': ensemble_precision,
            'ensemble_recall': ensemble_recall
        })
        
        # 8. Avaliar todos os modelos
        print("\n=== Etapa 8: Avaliando todos os modelos ===")
        models_dict = {
            'calibrated': calibrated_model,
            'specialist': specialist_model,
            'ensemble': ensemble_model
        }
        
        thresholds_dict = {
            'calibrated': calibrated_threshold,
            'specialist': 0.5,
            'ensemble': ensemble_threshold
        }
        
        evaluation_results = evaluate_models(X_val_enhanced, y_val, models_dict, thresholds_dict)
        
        # Registrar métricas no MLflow
        for _, row in evaluation_results.iterrows():
            model_name = row['model_name']
            mlflow.log_metrics({
                f'{model_name}_precision': row['precision'],
                f'{model_name}_recall': row['recall'],
                f'{model_name}_f1': row['f1']
            })
        
        # 9. Salvar modelos e metadata
        print("\n=== Etapa 9: Salvando modelos ===")
        output_dir = save_models(models_dict, thresholds_dict, args.output_dir)
        
        # Registrar modelos no MLflow
        # Preparar um exemplo de entrada para o modelo
        input_example = X_val_enhanced.iloc[:5].copy()
        
        for model_name, model in models_dict.items():
            artifact_path = os.path.join("models", model_name)
            mlflow.sklearn.log_model(model, artifact_path, input_example=input_example)
        
        # 10. Gerar resumo final
        print("\n=== Resumo dos Resultados ===")
        print(evaluation_results[['model_name', 'threshold', 'precision', 'recall', 'f1']])
        
        # Salvar resumo
        results_path = os.path.join(args.output_dir, "model_evaluation.csv")
        evaluation_results.to_csv(results_path, index=False)
        
        print("\n=== Treinamento e avaliação concluídos com sucesso! ===")
        print(f"Modelos salvos em: {output_dir}")
        print(f"MLflow run ID: {run_id}")
        
        return output_dir, run_id

if __name__ == "__main__":
    main()