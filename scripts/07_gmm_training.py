#!/usr/bin/env python
"""
Script principal para treinar GMM otimizado para ranking.
Usa o módulo GMMRankingTrainer para evitar problemas de serialização.
"""

import os
import sys
import mlflow

# Caminho absoluto do projeto
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Importar o trainer
from src.modeling.gmm_ranking_trainer import GMMRankingTrainer

# Configuração
CONFIG = {
    'base_dir': PROJECT_ROOT,
    'data_dir': os.path.join(PROJECT_ROOT, "data/new/03_feature_engineering_1"),  # Dataset completo
    'mlflow_dir': os.path.join(PROJECT_ROOT, "models/mlflow"),
    'artifact_dir': os.path.join(PROJECT_ROOT, "models/artifacts"),
    'experiment_name': "smart_ads_gmm_ranking",
    'random_state': 42,
    'param_search': {
        'enable_search': True,
        'n_components_range': [2, 3, 4, 5, 6, 7, 8],  # Expandido
        'covariance_types': ['spherical', 'tied', 'diag', 'full']
    },
    'gmm_params_default': {
        'n_components': 3,
        'covariance_type': 'spherical'
    }
}

def main():
    """Função principal."""
    print("="*80)
    print("GMM TRAINING - OTIMIZADO PARA RANKING")
    print("="*80)
    print(f"Dataset: {CONFIG['data_dir']}")
    print(f"MLflow: {CONFIG['mlflow_dir']}")
    print(f"Artefatos: {CONFIG['artifact_dir']}")
    print("="*80)
    
    # Garantir que não há run MLflow ativo
    if mlflow.active_run():
        mlflow.end_run()
    
    # Criar e executar trainer
    trainer = GMMRankingTrainer(CONFIG)
    results = trainer.run()
    
    # Imprimir resumo final
    if results:
        print("\n" + "="*80)
        print("RESUMO FINAL")
        print("="*80)
        print(f"Sucesso: {all(results['success_criteria'].values())}")
        print(f"GINI: {results['metrics']['gini']:.4f}")
        print(f"Top Decile Lift: {results['metrics']['top_decile_lift']:.2f}x")
        print(f"Top 20% Recall: {results['metrics']['top_20pct_recall']:.2%}")
        print("="*80)

if __name__ == "__main__":
    main()