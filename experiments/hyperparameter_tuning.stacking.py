#!/usr/bin/env python
"""
Script para otimização de hiperparâmetros de modelos de stacking.

Este script implementa uma estratégia em duas etapas para otimização:
1. Primeiro otimiza os parâmetros dos modelos especialistas
2. Depois otimiza os parâmetros do meta-modelo
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Importar módulos do projeto
from src.evaluation.mlflow_utils import setup_mlflow_tracking, get_data_hash
from src.evaluation.baseline_model import sanitize_column_names
from src.modeling.stacking.data_splitter import (
    get_feature_categories, prepare_data_for_specialists,
    print_feature_group_stats, validate_feature_groups
)
from src.modeling.stacking.specialist_models import (
    SpecialistModel, StackingEnsemble, prepare_specialist_data
)
from src.modeling.stacking.meta_learner import MetaLearner
from src.modeling.stacking.stacking_evaluation import evaluate_and_log_stacking_ensemble

def parse_arguments():
    """
    Processa argumentos da linha de comando.
    
    Returns:
        Namespace com argumentos processados
    """
    parser = argparse.ArgumentParser(description="Otimização de hiperparâmetros para stacking")
    
    # Caminhos de dados
    parser.add_argument('--train_path', required=True, 
                      help='Caminho para o dataset de treino')
    parser.add_argument('--val_path', required=True, 
                      help='Caminho para o dataset de validação')
    
    # Configurações gerais
    parser.add_argument('--target_col', default='target', 
                      help='Nome da coluna target')
    parser.add_argument('--output_dir', default=None, 
                      help='Diretório para salvar resultados')
    parser.add_argument('--mlflow_dir', default=None, 
                      help='Diretório para tracking do MLflow')
    parser.add_argument('--experiment_name', default='smart_ads_stacking_hyperopt', 
                      help='Nome do experimento no MLflow')
    
    # Configurações específicas
    parser.add_argument('--categorize_method', default='combined', 
                      choices=['name', 'content', 'combined'],
                      help='Método para categorizar features')
    parser.add_argument('--n_folds', type=int, default=5, 
                      help='Número de folds para cross-validation')
    parser.add_argument('--max_evals', type=int, default=20, 
                      help='Número máximo de avaliações por modelo')
    parser.add_argument('--feature_groups_path', default=None, 
                      help='Caminho para arquivo com grupos de features pré-definidos (opcional)')
    
    return parser.parse_args()

def load_feature_groups(filepath):
    """
    Carrega grupos de features de um arquivo CSV.
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Dicionário com grupos de features
    """
    import csv
    from collections import defaultdict
    
    feature_groups = defaultdict(list)
    
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Pular cabeçalho
        
        for row in reader:
            group, feature = row
            feature_groups[group].append(feature)
    
    return dict(feature_groups)

def save_feature_groups(feature_groups, filepath):
    """
    Salva grupos de features em um arquivo CSV.
    
    Args:
        feature_groups: Dicionário com grupos de features
        filepath: Caminho para salvar o arquivo
    """
    import csv
    import os
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['group', 'feature'])
        
        for group, features in feature_groups.items():
            for feature in features:
                writer.writerow([group, feature])

def get_hyperparameter_space(model_type, feature_type=None):
    """
    Define o espaço de hiperparâmetros para um tipo de modelo.
    
    Args:
        model_type: Tipo de modelo ('lightgbm', 'xgboost', 'random_forest')
        feature_type: Tipo de feature (opcional, para especialização)
        
    Returns:
        Dicionário com espaço de hiperparâmetros para hyperopt
    """
    if model_type == "lightgbm":
        # Base para LightGBM
        space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'num_leaves': hp.quniform('num_leaves', 20, 60, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
            'bagging_freq': hp.quniform('bagging_freq', 1, 10, 1),
            'min_child_samples': hp.quniform('min_child_samples', 5, 50, 1),
            'scale_pos_weight': hp.quniform('scale_pos_weight', 30, 100, 5)
        }
        
        # Ajustes específicos por tipo de feature
        if feature_type == "text":
            # Para features textuais, maior regularização
            space.update({
                'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(10.0)),
                'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(10.0)),
            })
        elif feature_type == "temporal":
            # Para features temporais, ajustes específicos
            space.update({
                'max_depth': hp.quniform('max_depth', 5, 12, 1),
            })
    
    elif model_type == "xgboost":
        space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'gamma': hp.uniform('gamma', 0, 1),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': hp.quniform('scale_pos_weight', 30, 100, 5)
        }
    
    elif model_type == "random_forest":
        space = {
            'n_estimators': hp.quniform('n_estimators', 100, 500, 10),
            'max_depth': hp.quniform('max_depth', 5, 30, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
        }
    
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    return space

def optimize_specialist(X_train, y_train, X_val, y_val, model_type, feature_type, max_evals=20):
    """
    Otimiza os hiperparâmetros de um modelo especialista.
    
    Args:
        X_train: Features de treinamento
        y_train: Target de treinamento
        X_val: Features de validação
        y_val: Target de validação
        model_type: Tipo de modelo ('lightgbm', 'xgboost', 'random_forest')
        feature_type: Tipo de feature
        max_evals: Número máximo de avaliações
        
    Returns:
        Dicionário com melhores hiperparâmetros e modelo treinado
    """
    print(f"Otimizando hiperparâmetros para modelo {model_type} de features {feature_type}...")
    
    # Definir espaço de hiperparâmetros
    space = get_hyperparameter_space(model_type, feature_type)
    
    # Definir função objetivo
    def objective(params):
        # Converter tipos numéricos para inteiros onde necessário
        for param in ['num_leaves', 'bagging_freq', 'min_child_samples', 
                     'max_depth', 'min_child_weight', 'n_estimators',
                     'min_samples_split', 'min_samples_leaf']:
            if param in params:
                params[param] = int(params[param])
        
        if 'scale_pos_weight' in params:
            params['scale_pos_weight'] = int(params['scale_pos_weight'])
        
        # Criar e treinar modelo
        model = SpecialistModel(
            model_type=model_type,
            feature_type=feature_type,
            params=params,
            name=f"specialist_{feature_type}"
        )
        
        try:
            model.fit(X_train, y_train)
            
            # Avaliar no conjunto de validação
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Encontrar threshold ótimo para F1
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in np.arange(0.01, 0.5, 0.01):
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            # Calcular métricas finais
            from sklearn.metrics import precision_score, recall_score, average_precision_score
            
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            pr_auc = average_precision_score(y_val, y_pred_proba)
            
            print(f"  F1: {best_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Threshold: {best_threshold:.4f}")
            
            # Retornar resultado
            return {
                'loss': -best_f1,  # Negativo porque queremos maximizar
                'status': STATUS_OK,
                'model': model,
                'threshold': best_threshold,
                'f1': best_f1,
                'precision': precision,
                'recall': recall,
                'pr_auc': pr_auc,
                'params': params
            }
            
        except Exception as e:
            print(f"  Erro ao treinar modelo: {e}")
            return {
                'loss': 0,
                'status': STATUS_OK,
                'model': None,
                'threshold': 0.5,
                'f1': 0,
                'precision': 0,
                'recall': 0,
                'pr_auc': 0,
                'params': params
            }
    
    # Executar otimização
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    # Obter melhor resultado
    best_idx = np.argmin([t['result']['loss'] for t in trials.trials])
    best_trial = trials.trials[best_idx]
    best_result = best_trial['result']
    
    print(f"Melhores hiperparâmetros para {feature_type}:")
    for param, value in best_result['params'].items():
        print(f"  {param}: {value}")
    
    print(f"Métricas: F1={best_result['f1']:.4f}, PR-AUC={best_result['pr_auc']:.4f}")
    
    return best_result

def optimize_meta_model(meta_features_train, y_train, meta_features_val, y_val, model_type='lightgbm', max_evals=20):
    """
    Otimiza os hiperparâmetros do meta-modelo.
    
    Args:
        meta_features_train: Meta-features de treinamento
        y_train: Target de treinamento
        meta_features_val: Meta-features de validação
        y_val: Target de validação
        model_type: Tipo de modelo ('lightgbm', 'xgboost', 'random_forest')
        max_evals: Número máximo de avaliações
        
    Returns:
        Dicionário com melhores hiperparâmetros e modelo treinado
    """
    print(f"Otimizando hiperparâmetros para meta-modelo {model_type}...")
    
    # Definir espaço de hiperparâmetros
    space = get_hyperparameter_space(model_type)
    
    # Definir função objetivo
    def objective(params):
        # Converter tipos numéricos para inteiros onde necessário
        for param in ['num_leaves', 'bagging_freq', 'min_child_samples', 
                     'max_depth', 'min_child_weight', 'n_estimators',
                     'min_samples_split', 'min_samples_leaf']:
            if param in params:
                params[param] = int(params[param])
        
        if 'scale_pos_weight' in params:
            params['scale_pos_weight'] = int(params['scale_pos_weight'])
        
        # Criar e treinar modelo
        meta_model = MetaLearner(
            model_type=model_type,
            params=params,
            name="meta_learner"
        )
        
        try:
            meta_model.fit(meta_features_train, y_train)
            
            # Avaliar no conjunto de validação
            y_pred_proba = meta_model.predict_proba(meta_features_val)[:, 1]
            
            # Encontrar threshold ótimo para F1
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in np.arange(0.01, 0.5, 0.01):
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            # Calcular métricas finais
            from sklearn.metrics import precision_score, recall_score, average_precision_score
            
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            pr_auc = average_precision_score(y_val, y_pred_proba)
            
            print(f"  F1: {best_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Threshold: {best_threshold:.4f}")
            
            # Retornar resultado
            return {
                'loss': -best_f1,  # Negativo porque queremos maximizar
                'status': STATUS_OK,
                'model': meta_model,
                'threshold': best_threshold,
                'f1': best_f1,
                'precision': precision,
                'recall': recall,
                'pr_auc': pr_auc,
                'params': params
            }
            
        except Exception as e:
            print(f"  Erro ao treinar modelo: {e}")
            return {
                'loss': 0,
                'status': STATUS_OK,
                'model': None,
                'threshold': 0.5,
                'f1': 0,
                'precision': 0,
                'recall': 0,
                'pr_auc': 0,
                'params': params
            }
    
    # Executar otimização
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    # Obter melhor resultado
    best_idx = np.argmin([t['result']['loss'] for t in trials.trials])
    best_trial = trials.trials[best_idx]
    best_result = best_trial['result']
    
    print(f"Melhores hiperparâmetros para meta-modelo:")
    for param, value in best_result['params'].items():
        print(f"  {param}: {value}")
    
    print(f"Métricas: F1={best_result['f1']:.4f}, PR-AUC={best_result['pr_auc']:.4f}")
    
    return best_result

def generate_meta_features(specialists, X, y, feature_groups, cv=5):
    """
    Gera meta-features usando cross-validation.
    
    Args:
        specialists: Lista de modelos especialistas
        X: DataFrame com features originais
        y: Series com target
        feature_groups: Dicionário com grupos de features
        cv: Número de folds para cross-validation
        
    Returns:
        DataFrame com meta-features
    """
    print(f"Gerando meta-features com {cv}-fold cross-validation...")
    
    # Inicializar arrays para armazenar meta-features
    n_samples = len(y)
    n_specialists = len(specialists)
    meta_features = np.zeros((n_samples, n_specialists))
    
    # Preparar cross-validation
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Para cada fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"  Processando fold {fold+1}/{cv}")
        
        # Para cada especialista
        for i, specialist in enumerate(specialists):
            # Obter tipo de feature
            feature_type = specialist['feature_type']
            
            # Selecionar features específicas
            features = feature_groups[feature_type]
            X_fold = X[features]
            
            # Preparar dados de treino e validação
            X_train_fold = X_fold.iloc[train_idx]
            X_val_fold = X_fold.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            
            # Criar modelo com melhores hiperparâmetros
            model = SpecialistModel(
                model_type=specialist['model_type'],
                feature_type=feature_type,
                params=specialist['params'],
                name=f"specialist_{feature_type}"
            )
            
            # Treinar modelo
            model.fit(X_train_fold, y_train_fold)
            
            # Gerar previsões
            val_preds = model.predict_proba(X_val_fold)[:, 1]
            
            # Armazenar previsões
            meta_features[val_idx, i] = val_preds
    
    # Converter para DataFrame
    meta_df = pd.DataFrame(meta_features, index=X.index)
    
    # Renomear colunas
    meta_df.columns = [f"specialist_{s['feature_type']}" for s in specialists]
    
    return meta_df

def main():
    # Processar argumentos
    args = parse_arguments()
    
    # Configurar diretório de saída
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(project_root, "models", f"stacking_hyperopt_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {args.output_dir}")
    
    # Configurar MLflow
    if args.mlflow_dir is None:
        args.mlflow_dir = os.path.join(args.output_dir, "mlflow")
    
    experiment_id = setup_mlflow_tracking(
        tracking_dir=args.mlflow_dir,
        experiment_name=args.experiment_name,
        clean_previous=False
    )
    
    # Iniciar run MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name="stacking_hyperopt") as run:
        run_id = run.info.run_id
        
        # Carregar dados
        print(f"\nCarregando dados de {args.train_path} e {args.val_path}...")
        train_df = pd.read_csv(args.train_path)
        val_df = pd.read_csv(args.val_path)
        
        # Sanitizar nomes das colunas
        sanitize_column_names(train_df)
        sanitize_column_names(val_df)
        
        # Calcular hashes para tracking
        train_hash = get_data_hash(train_df)
        val_hash = get_data_hash(val_df)
        
        # Registrar tags básicas
        mlflow.set_tags({
            "model_type": "stacking_hyperopt",
            "train_data_hash": train_hash,
            "val_data_hash": val_hash,
            "train_size": len(train_df),
            "val_size": len(val_df)
        })
        
        # Carregar ou criar grupos de features
        if args.feature_groups_path and os.path.exists(args.feature_groups_path):
            print(f"Carregando grupos de features de {args.feature_groups_path}...")
            feature_groups = load_feature_groups(args.feature_groups_path)
        else:
            print(f"Categorizando features usando método: {args.categorize_method}...")
            feature_groups = get_feature_categories(
                train_df, 
                target_col=args.target_col, 
                method=args.categorize_method
            )
            
            # Validar grupos
            feature_groups = validate_feature_groups(feature_groups, train_df)
            
            # Salvar grupos de features
            feature_groups_path = os.path.join(args.output_dir, "feature_groups.csv")
            save_feature_groups(feature_groups, feature_groups_path)
        
        # Mostrar estatísticas dos grupos
        print_feature_group_stats(feature_groups, train_df)
        
        # Registrar informações sobre as features
        for group, features in feature_groups.items():
            mlflow.log_param(f"feature_count_{group}", len(features))
        
        # Preparar dados para treino
        X_train = train_df.drop(columns=[args.target_col])
        y_train = train_df[args.target_col]
        
        X_val = val_df.drop(columns=[args.target_col])
        y_val = val_df[args.target_col]
        
        # Otimizar especialistas para cada grupo de features
        optimized_specialists = []
        
        for group, features in feature_groups.items():
            if features:  # Verificar se há features neste grupo
                print(f"\n=== Otimizando especialista para {group} ===")
                
                # Selecionar features específicas
                X_train_group = X_train[features]
                X_val_group = X_val[features]
                
                # Escolher tipo de modelo adequado
                if group == "text":
                    model_type = "lightgbm"
                elif group == "temporal":
                    model_type = "lightgbm"
                else:
                    model_type = "lightgbm"
                
                # Executar otimização
                best_result = optimize_specialist(
                    X_train_group, y_train,
                    X_val_group, y_val,
                    model_type=model_type,
                    feature_type=group,
                    max_evals=args.max_evals
                )
                
                # Registrar parâmetros e métricas no MLflow
                for param, value in best_result['params'].items():
                    mlflow.log_param(f"{group}_{param}", value)
                
                mlflow.log_metric(f"{group}_f1", best_result['f1'])
                mlflow.log_metric(f"{group}_precision", best_result['precision'])
                mlflow.log_metric(f"{group}_recall", best_result['recall'])
                mlflow.log_metric(f"{group}_pr_auc", best_result['pr_auc'])
                mlflow.log_metric(f"{group}_threshold", best_result['threshold'])
                
                # Adicionar aos resultados
                specialist_info = {
                    'feature_type': group,
                    'model_type': model_type,
                    'params': best_result['params'],
                    'threshold': best_result['threshold'],
                    'f1': best_result['f1'],
                    'model': best_result['model']
                }
                
                optimized_specialists.append(specialist_info)
        
        # Gerar meta-features para treinar o meta-modelo
        print("\n=== Gerando meta-features para treinar o meta-modelo ===")
        
        meta_features_train = generate_meta_features(
            optimized_specialists, X_train, y_train, feature_groups, cv=args.n_folds
        )
        
        # Treinar especialistas com todos os dados para gerar meta-features de validação
        print("\n=== Treinando especialistas para gerar meta-features de validação ===")
        
        trained_specialists = []
        meta_features_val = pd.DataFrame(index=X_val.index)
        
        for specialist in optimized_specialists:
            # Selecionar features específicas
            features = feature_groups[specialist['feature_type']]
            X_train_group = X_train[features]
            X_val_group = X_val[features]
            
            # Treinar modelo
            model = specialist['model']
            if model is None:
                model = SpecialistModel(
                    model_type=specialist['model_type'],
                    feature_type=specialist['feature_type'],
                    params=specialist['params'],
                    name=f"specialist_{specialist['feature_type']}"
                )
                model.fit(X_train_group, y_train)
            
            # Gerar previsões para validação
            val_preds = model.predict_proba(X_val_group)[:, 1]
            
            # Adicionar ao DataFrame de meta-features
            meta_features_val[f"specialist_{specialist['feature_type']}"] = val_preds
            
            # Adicionar aos especialistas treinados
            trained_specialists.append(model)
        
        # Otimizar meta-modelo
        print("\n=== Otimizando meta-modelo ===")
        
        meta_result = optimize_meta_model(
            meta_features_train, y_train,
            meta_features_val, y_val,
            model_type="lightgbm",
            max_evals=args.max_evals
        )
        
        # Registrar parâmetros e métricas no MLflow
        for param, value in meta_result['params'].items():
            mlflow.log_param(f"meta_{param}", value)
        
        mlflow.log_metric("meta_f1", meta_result['f1'])
        mlflow.log_metric("meta_precision", meta_result['precision'])
        mlflow.log_metric("meta_recall", meta_result['recall'])
        mlflow.log_metric("meta_pr_auc", meta_result['pr_auc'])
        mlflow.log_metric("meta_threshold", meta_result['threshold'])
        
        # Construir ensemble final
        print("\n=== Construindo ensemble final ===")
        
        # Criar especialistas com parâmetros otimizados
        final_specialists = []
        for specialist in optimized_specialists:
            final_model = SpecialistModel(
                model_type=specialist['model_type'],
                feature_type=specialist['feature_type'],
                params=specialist['params'],
                name=f"specialist_{specialist['feature_type']}"
            )
            final_specialists.append(final_model)
        
        # Criar meta-modelo
        meta_model = MetaLearner(
            model_type="lightgbm",
            params=meta_result['params'],
            name="meta_learner"
        )
        
        # Criar ensemble
        ensemble = StackingEnsemble(
            specialist_models=final_specialists,
            meta_model=meta_model,
            cv=args.n_folds,
            random_state=42,
            threshold=meta_result['threshold']
        )
        
        # Preparar dados para ensemble
        X_train_dict, y_train_dict = prepare_data_for_specialists(
            train_df, feature_groups, args.target_col
        )
        
        X_val_dict, y_val_dict = prepare_data_for_specialists(
            val_df, feature_groups, args.target_col
        )
        
        # Treinar ensemble
        print("Treinando ensemble final...")
        ensemble.fit(X_train_dict, y_train)
        
        # Avaliar no conjunto de validação
        print("Avaliando ensemble no conjunto de validação...")
        val_results_dir = os.path.join(args.output_dir, "validation_results")
        
        val_results = evaluate_and_log_stacking_ensemble(
            ensemble, X_val_dict, y_val, 
            run_id=run_id, 
            output_dir=val_results_dir
        )
        
        print(f"\nResultados de validação:")
        print(f"F1: {val_results['f1']:.4f}")
        print(f"Precision: {val_results['precision']:.4f}")
        print(f"Recall: {val_results['recall']:.4f}")
        print(f"PR-AUC: {val_results['pr_auc']:.4f}")
        print(f"Threshold: {val_results['threshold']:.4f}")
        
        # Salvar modelo final
        print("\nSalvando modelo final...")
        model_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Salvar ensemble
        ensemble_path = os.path.join(model_dir, "stacking_ensemble.joblib")
        joblib.dump(ensemble, ensemble_path)
        
        # Salvar feature groups
        feature_groups_path = os.path.join(model_dir, "feature_groups.csv")
        save_feature_groups(feature_groups, feature_groups_path)
        
        # Registrar artefatos
        mlflow.log_artifact(model_dir)
        
        print(f"\nOtimização de hiperparâmetros concluída com sucesso!")
        print(f"Modelo final salvo em: {model_dir}")
        print(f"Run ID do MLflow: {run_id}")
        print(f"Todos os resultados foram salvos em: {args.output_dir}")

if __name__ == "__main__":
    main()