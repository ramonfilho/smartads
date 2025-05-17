#!/usr/bin/env python
"""
Script para executar a pipeline de stacking completa de modelos especialistas.

Este script treina e avalia um sistema de stacking com modelos especialistas
para dados demográficos, temporais e textuais, combinados por um meta-modelo.
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
import joblib

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Importar módulos do projeto
from src.evaluation.mlflow_utils import setup_mlflow_tracking, get_data_hash
from src.evaluation.baseline_model import sanitize_column_names
from src.models1.stacking.data_splitter import (
    get_feature_categories, prepare_data_for_specialists, 
    print_feature_group_stats, validate_feature_groups,
    adjust_feature_groups_manually, save_feature_groups
)
from src.models1.stacking.specialist_models import (
    SpecialistModel, StackingEnsemble, prepare_specialist_data
)
from src.models1.stacking.meta_learner import (
    MetaLearner, WeightedAverageMetaLearner
)
from src.models1.stacking.stacking_evaluation import (
    evaluate_and_log_stacking_ensemble, compare_models,
    analyze_disagreements, analyze_specialist_contributions
)

def parse_arguments():
    """
    Processa argumentos da linha de comando.
    
    Returns:
        Namespace com argumentos processados
    """
    parser = argparse.ArgumentParser(description="Pipeline de Stacking para Smart Ads")
    
    # Caminhos de dados
    parser.add_argument('--train_path', required=True, 
                      help='Caminho para o dataset de treino')
    parser.add_argument('--val_path', required=True, 
                      help='Caminho para o dataset de validação')
    parser.add_argument('--test_path', default=None, 
                      help='Caminho para o dataset de teste (opcional)')
    
    # Configurações gerais
    parser.add_argument('--target_col', default='target', 
                      help='Nome da coluna target')
    parser.add_argument('--output_dir', default=None, 
                      help='Diretório base para salvar resultados')
    parser.add_argument('--mlflow_dir', default=None, 
                      help='Diretório para o tracking do MLflow')
    parser.add_argument('--experiment_name', default='smart_ads_stacking', 
                      help='Nome do experimento no MLflow')
    
    # Configurações específicas
    parser.add_argument('--categorize_method', default='combined', 
                      choices=['name', 'content', 'combined'],
                      help='Método para categorizar features')
    parser.add_argument('--n_folds', type=int, default=5, 
                      help='Número de folds para cross-validation')
    parser.add_argument('--meta_model_type', default='lightgbm', 
                      choices=['lightgbm', 'xgboost', 'random_forest', 'weighted_average'],
                      help='Tipo de meta-modelo')
    parser.add_argument('--manual_adjustments', default=None, 
                      help='Caminho para arquivo JSON com ajustes manuais de categorias')
    
    return parser.parse_args()

def main():
    """
    Função principal que executa toda a pipeline.
    """
    # Processar argumentos
    args = parse_arguments()
    
    # Configurar diretório de saída
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(project_root, "models", f"stacking_{timestamp}")
    
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
    
    # Carregar dados
    print(f"Carregando dados de {args.train_path} e {args.val_path}...")
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    
    if args.test_path and os.path.exists(args.test_path):
        test_df = pd.read_csv(args.test_path)
        has_test = True
    else:
        test_df = None
        has_test = False
    
    # Sanitizar nomes das colunas
    sanitize_column_names(train_df)
    sanitize_column_names(val_df)
    if has_test:
        sanitize_column_names(test_df)
    
    # Calcular hashes para tracking
    train_hash = get_data_hash(train_df)
    val_hash = get_data_hash(val_df)
    
    # Iniciar run MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name="stacking_pipeline") as run:
        run_id = run.info.run_id
        
        # Registrar tags e parâmetros básicos
        mlflow.set_tags({
            "model_type": "stacking",
            "experiment_type": "stacking_specialists",
            "meta_model": args.meta_model_type,
            "train_data_hash": train_hash,
            "val_data_hash": val_hash,
            "cv_folds": args.n_folds,
            "categorize_method": args.categorize_method
        })
        
        # Categorizar features
        print(f"\nCategorizando features usando método: {args.categorize_method}...")
        feature_groups = get_feature_categories(
            train_df, 
            target_col=args.target_col, 
            method=args.categorize_method
        )
        
        # Validar e ajustar grupos de features
        feature_groups = validate_feature_groups(feature_groups, train_df)
        
        # Carregar ajustes manuais se fornecidos
        if args.manual_adjustments:
            if os.path.exists(args.manual_adjustments):
                import json
                with open(args.manual_adjustments, 'r') as f:
                    manual_adjustments = json.load(f)
                
                print("Aplicando ajustes manuais nas categorias de features...")
                feature_groups = adjust_feature_groups_manually(feature_groups, manual_adjustments)
            else:
                print(f"AVISO: Arquivo de ajustes manuais não encontrado: {args.manual_adjustments}")
        
        # Mostrar estatísticas dos grupos
        print_feature_group_stats(feature_groups, train_df)
        
        # Salvar grupos de features
        feature_groups_path = os.path.join(args.output_dir, "feature_groups.csv")
        save_feature_groups(feature_groups, feature_groups_path)
        mlflow.log_artifact(feature_groups_path)
        
        # Registrar informações sobre as features
        for group, features in feature_groups.items():
            mlflow.set_tag(f"feature_count_{group}", len(features))
            mlflow.log_param(f"feature_count_{group}", len(features))
        
        # Preparar dados para modelos especialistas
        print("\nPreparando dados para modelos especialistas...")
        X_train, y_train = prepare_specialist_data(
            train_df, feature_groups, args.target_col
        )
        
        X_val, y_val = prepare_specialist_data(
            val_df, feature_groups, args.target_col
        )
        
        if has_test:
            X_test, y_test = prepare_specialist_data(
                test_df, feature_groups, args.target_col
            )
        
        # Criar e treinar modelos especialistas
        print("\nCriando modelos especialistas...")
        specialist_models = []
        
        for group, features in feature_groups.items():
            if features:  # Verificar se há features neste grupo
                print(f"Criando modelo especialista para {group} ({len(features)} features)")
                
                # Escolher tipo de modelo adequado para cada grupo
                if group == "text":
                    model_type = "lightgbm"  # LightGBM é bom para features textuais
                elif group == "temporal":
                    model_type = "lightgbm"  # LightGBM também funciona bem para temporais
                else:
                    model_type = "lightgbm"  # Padrão para outras features
                
                specialist = SpecialistModel(
                    model_type=model_type,
                    feature_type=group,
                    name=f"specialist_{group}"
                )
                
                specialist_models.append(specialist)
        
        # Configurar ensemble de stacking
        if args.meta_model_type == "weighted_average":
            meta_model = WeightedAverageMetaLearner()
        else:
            meta_model = MetaLearner(model_type=args.meta_model_type)
        
        ensemble = StackingEnsemble(
            specialist_models=specialist_models,
            meta_model=meta_model,
            cv=args.n_folds,
            random_state=42
        )
        
        # Treinar ensemble
        print("\nTreinando ensemble de stacking...")
        ensemble.fit(X_train, y_train)
        
        # Avaliar no conjunto de validação
        print("\nAvaliando ensemble no conjunto de validação...")
        val_results_dir = os.path.join(args.output_dir, "validation_results")
        
        val_results = evaluate_and_log_stacking_ensemble(
            ensemble, X_val, y_val, 
            run_id=run_id, 
            output_dir=val_results_dir
        )
        
        print(f"\nResultados de validação:")
        print(f"F1: {val_results['f1']:.4f}")
        print(f"Precision: {val_results['precision']:.4f}")
        print(f"Recall: {val_results['recall']:.4f}")
        print(f"PR-AUC: {val_results['pr_auc']:.4f}")
        print(f"Threshold: {val_results['threshold']:.4f}")
        
        # Avaliar no conjunto de teste, se disponível
        if has_test and y_test is not None:
            print("\nAvaliando ensemble no conjunto de teste...")
            test_results_dir = os.path.join(args.output_dir, "test_results")
            
            test_results = evaluate_and_log_stacking_ensemble(
                ensemble, X_test, y_test, 
                run_id=run_id, 
                output_dir=test_results_dir
            )
            
            print(f"\nResultados de teste:")
            print(f"F1: {test_results['f1']:.4f}")
            print(f"Precision: {test_results['precision']:.4f}")
            print(f"Recall: {test_results['recall']:.4f}")
            print(f"PR-AUC: {test_results['pr_auc']:.4f}")
        
        # Avaliar e comparar modelos especialistas individualmente
        print("\nComparando desempenho dos modelos especialistas...")
        
        # Criar dicionário de modelos para comparação
        models_to_compare = {}
        
        # Adicionar especialistas
        for model in ensemble.trained_specialists:
            models_to_compare[model.name] = model
        
        # Adicionar ensemble
        models_to_compare['ensemble'] = ensemble
        
        # Comparar modelos
        comparison_dir = os.path.join(args.output_dir, "model_comparison")
        
        comparison_results, predictions = compare_models(
            models_to_compare, X_val, y_val, comparison_dir
        )
        
        mlflow.log_artifact(comparison_dir)
        
        # Analisar desacordos entre modelos
        print("\nAnalisando desacordos entre modelos...")
        disagreement_dir = os.path.join(args.output_dir, "disagreements")
        
        disagreement_results, disagreements = analyze_disagreements(
            models_to_compare, X_val, y_val, disagreement_dir
        )
        
        mlflow.log_artifact(disagreement_dir)
        
        # Exportar modelo final
        print("\nExportando modelo final...")
        model_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Salvar ensemble
        ensemble_path = os.path.join(model_dir, "stacking_ensemble.joblib")
        joblib.dump(ensemble, ensemble_path)
        
        # Salvar configuração
        config_path = os.path.join(model_dir, "config.joblib")
        config = {
            'feature_groups': feature_groups,
            'meta_model_type': args.meta_model_type,
            'threshold': val_results['threshold'],
            'cv_folds': args.n_folds
        }
        joblib.dump(config, config_path)
        
        # Registrar artefatos no MLflow
        mlflow.log_artifact(model_dir)
        
        print(f"\nPipeline concluída com sucesso.")
        print(f"Modelo final salvo em: {model_dir}")
        print(f"Run ID do MLflow: {run_id}")
        print(f"Todos os resultados foram salvos em: {args.output_dir}")
        
        return {
            'ensemble': ensemble,
            'feature_groups': feature_groups,
            'val_results': val_results,
            'comparison_results': comparison_results,
            'disagreement_results': disagreement_results,
            'run_id': run_id,
            'output_dir': args.output_dir
        }

if __name__ == "__main__":
    main()