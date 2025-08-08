#!/usr/bin/env python
"""
Script direto para treinar GMM com a melhor configuração encontrada.
Usa n_components=3, covariance_type=spherical (Top Decile Lift: 3.30x)
"""

import os
import sys
import mlflow

PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from experiments.gmm_ranking_trainer import GMMRankingTrainer

# Configuração com os MELHORES parâmetros encontrados
CONFIG = {
    'base_dir': PROJECT_ROOT,
    'data_dir': os.path.join(PROJECT_ROOT, "data/new/03_feature_engineering"),
    'mlflow_dir': os.path.join(PROJECT_ROOT, "models/mlflow"),
    'artifact_dir': os.path.join(PROJECT_ROOT, "models/artifacts"),
    'experiment_name': "smart_ads_gmm_best_config",
    'random_state': 42,
    'pca_variance_thresholds': [0.90],  # Fixar em 90%
    'param_search': {
        'enable_search': True,
        'n_components_range': [3],  # APENAS testar 3
        'covariance_types': ['spherical']  # APENAS testar spherical
    },
    'gmm_params_default': {
        'n_components': 3,
        'covariance_type': 'spherical'
    }
}

def main():
    """Executa treinamento com configuração otimizada."""
    print("="*80)
    print("GMM TRAINING - CONFIGURAÇÃO OTIMIZADA")
    print("="*80)
    print("Usando configuração que obteve Top Decile Lift: 3.30x")
    print("n_components=3, covariance_type=spherical")
    print("="*80)
    
    # Garantir que não há run MLflow ativo
    if mlflow.active_run():
        mlflow.end_run()
    
    # Criar trainer e executar
    trainer = GMMRankingTrainer(CONFIG)
    results = trainer.run()
    
    # Imprimir resultados
    if results:
        print("\n" + "="*80)
        print("RESULTADOS FINAIS")
        print("="*80)
        print(f"GINI: {results['metrics']['gini']:.4f}")
        print(f"Top Decile Lift: {results['metrics']['top_decile_lift']:.2f}x")
        print(f"Top 20% Recall: {results['metrics']['top_20pct_recall']:.2%}")
        print("="*80)

if __name__ == "__main__":
    main()