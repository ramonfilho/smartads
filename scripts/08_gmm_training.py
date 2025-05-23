#!/usr/bin/env python
"""
Script de execução para treinamento do modelo GMM.
Importa e executa o módulo de treinamento para evitar problemas de serialização.
"""
import os
import sys
import mlflow

# Adicionar caminho do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar o módulo de treinamento
from src.modeling.gmm_trainer import GMMTrainer


def main():
    """Função principal que configura e executa o treinamento."""
    
    # Configuração centralizada
    config = {
        'base_dir': project_root,
        'data_dir': os.path.join(project_root, "data/V5/04_feature_engineering_2"),
        'mlflow_dir': os.path.join(project_root, "models/mlflow"),
        'artifact_dir': os.path.join(project_root, "models/artifacts"),
        'experiment_name': "smart_ads_gmm_optimized_full_v2",
        'model_params': {
            'max_depth': None,
            'n_estimators': 200,
            'random_state': 42
        },
        # Parâmetros originais (como fallback)
        'gmm_params_original': {
            'n_components': 3,
            'covariance_type': 'spherical'
        },
        # Opções para busca de parâmetros
        'param_search': {
            'n_components_range': [2, 3, 4, 5],  # Removido 6 (muitos componentes)
            'covariance_types': ['spherical', 'diag', 'full'],  # Removido 'tied' (resultados ruins)
            'enable_search': True  # Flag para habilitar/desabilitar busca
        },
        # Parâmetros do PCA
        'pca_params': {
            'variance_threshold': 0.95,  # Aumentado de 0.8 para capturar mais variância
            'max_components': 200,       # Aumentado de 100 para permitir mais componentes
            'min_components': 10         # Mínimo de componentes
        }
    }
    
    # Limpar runs MLflow ativos antes de começar
    active_run = mlflow.active_run()
    if active_run:
        print(f"Encerrando run MLflow ativo: {active_run.info.run_id}")
        mlflow.end_run()
    
    # Criar instância do trainer
    trainer = GMMTrainer(config)
    
    # Executar treinamento
    # use_param_search=True para buscar parâmetros ótimos
    # use_param_search=False para usar parâmetros originais
    results = trainer.run(use_param_search=True)
    
    return results


if __name__ == "__main__":
    # Executar treinamento
    results = main()
    
    # Mostrar resultados finais
    if results:
        print("\n" + "="*80)
        print("RESULTADOS FINAIS DO TREINAMENTO")
        print("="*80)
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"Threshold: {results['threshold']:.4f}")
    else:
        print("\n❌ Treinamento falhou ou não produziu resultados.")