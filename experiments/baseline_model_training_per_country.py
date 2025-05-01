#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para treinar modelos baseline para múltiplas subpastas de dados.

Este script processa todas as subpastas em "03_4_feature_selection_text_per_country",
treina modelos baseline para cada conjunto de dados e registra as métricas
em um arquivo CSV para análise comparativa.
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# Caminho absoluto para o diretório raiz do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Adicionar o caminho ao sys.path
sys.path.insert(0, project_root)

# Importar os módulos necessários
from src.evaluation.baseline_model import run_baseline_model_training, prepare_data_for_training
from src.evaluation.mlflow_utils import (
    setup_mlflow_tracking, get_data_hash, 
    plot_confusion_matrix, plot_prob_histogram, 
    plot_precision_recall_curve, find_optimal_threshold
)

def find_subfolders(base_path):
    """
    Encontra todas as subpastas no caminho base.
    
    Args:
        base_path: Caminho base para procurar subpastas
        
    Returns:
        Lista de subpastas encontradas
    """
    return [f.path for f in os.scandir(base_path) if f.is_dir()]

def ensure_directory_exists(directory_path):
    """
    Garante que um diretório exista, criando-o se necessário.
    
    Args:
        directory_path: Caminho do diretório
    """
    os.makedirs(directory_path, exist_ok=True)
    
def train_models_for_folder(folder_path, output_metrics, artifact_base_dir):
    """
    Treina modelos baseline para uma pasta específica.
    
    Args:
        folder_path: Caminho da pasta contendo os conjuntos de dados
        output_metrics: Lista para armazenar métricas de saída
        artifact_base_dir: Diretório base para artefatos
        
    Returns:
        Dicionário de resultados
    """
    folder_name = os.path.basename(folder_path)
    print(f"\n{'='*80}")
    print(f"Processando pasta: {folder_name}")
    print(f"{'='*80}")
    
    # Definir caminhos para os arquivos de treino e validação
    train_path = os.path.join(folder_path, "train.csv")
    val_path = os.path.join(folder_path, "validation.csv")
    
    # Verificar se os arquivos existem
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"ERRO: Arquivos de treino ou validação não encontrados em {folder_path}")
        return None
    
    # Criar diretório específico para artefatos desta pasta
    artifact_dir = os.path.join(artifact_base_dir, folder_name)
    ensure_directory_exists(artifact_dir)
    
    # Executar o treinamento sem registrar no MLflow
    try:
        # Preparar os dados
        print("Preparando dados...")
        data = prepare_data_for_training(train_path, val_path)
        
        # Verificar a taxa de conversão
        train_conversion_rate = data['y_train'].mean()
        val_conversion_rate = data['y_val'].mean()
        
        print(f"Taxa de conversão - treino: {train_conversion_rate:.4f}, validação: {val_conversion_rate:.4f}")
        
        # Executar o treinamento
        print("Treinando modelos...")
        start_time = time.time()
        results = run_baseline_model_training(
            train_path=train_path,
            val_path=val_path,
            experiment_id=None,  # Não usar MLflow
            artifact_dir=artifact_dir,
            generate_learning_curves=False
        )
        total_time = time.time() - start_time
        
        # Adicionar resultados ao dataframe
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Adicionar métricas do RandomForest
        output_metrics.append({
            "timestamp": timestamp,
            "folder": folder_name,
            "model": "random_forest",
            "f1": results.get("random_forest_f1", 0),
            "precision": results.get("random_forest_precision", 0),
            "recall": results.get("random_forest_recall", 0),
            "auc": results.get("random_forest_auc", 0),
            "pr_auc": results.get("random_forest_pr_auc", 0),
            "threshold": results.get("random_forest_threshold", 0),
            "train_time": results.get("random_forest_train_time", 0),
            "train_conversion_rate": train_conversion_rate,
            "val_conversion_rate": val_conversion_rate,
            "total_training_time": total_time
        })
        
        # Adicionar métricas do LightGBM (se disponível)
        if "lightgbm_f1" in results:
            output_metrics.append({
                "timestamp": timestamp,
                "folder": folder_name,
                "model": "lightgbm",
                "f1": results.get("lightgbm_f1", 0),
                "precision": results.get("lightgbm_precision", 0),
                "recall": results.get("lightgbm_recall", 0),
                "auc": results.get("lightgbm_auc", 0),
                "pr_auc": results.get("lightgbm_pr_auc", 0),
                "threshold": results.get("lightgbm_threshold", 0),
                "train_time": results.get("lightgbm_train_time", 0),
                "train_conversion_rate": train_conversion_rate,
                "val_conversion_rate": val_conversion_rate,
                "total_training_time": total_time
            })
        
        # Adicionar métricas do XGBoost (se disponível)
        if "xgboost_f1" in results:
            output_metrics.append({
                "timestamp": timestamp,
                "folder": folder_name,
                "model": "xgboost",
                "f1": results.get("xgboost_f1", 0),
                "precision": results.get("xgboost_precision", 0),
                "recall": results.get("xgboost_recall", 0),
                "auc": results.get("xgboost_auc", 0),
                "pr_auc": results.get("xgboost_pr_auc", 0),
                "threshold": results.get("xgboost_threshold", 0),
                "train_time": results.get("xgboost_train_time", 0),
                "train_conversion_rate": train_conversion_rate,
                "val_conversion_rate": val_conversion_rate,
                "total_training_time": total_time
            })
        
        print(f"Processamento da pasta {folder_name} concluído em {total_time:.2f} segundos.")
        return results
    
    except Exception as e:
        print(f"ERRO ao processar a pasta {folder_name}: {str(e)}")
        # Adicionar erro ao dataframe
        output_metrics.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "folder": folder_name,
            "model": "ERROR",
            "f1": 0,
            "precision": 0,
            "recall": 0,
            "auc": 0,
            "pr_auc": 0,
            "threshold": 0,
            "train_time": 0,
            "train_conversion_rate": 0,
            "val_conversion_rate": 0,
            "total_training_time": 0,
            "error": str(e)
        })
        return None

def main():
    # 1. Configuração de caminhos
    base_dir = os.path.expanduser("~")
    data_dir = os.path.join(base_dir, "desktop/smart_ads/data/03_4_feature_selection_text_per_country")
    artifact_base_dir = os.path.join(base_dir, "desktop/smart_ads/models/artifacts/country_comparison")
    reports_dir = os.path.join(base_dir, "desktop/smart_ads/reports")
    
    # Garantir que os diretórios existam
    ensure_directory_exists(artifact_base_dir)
    ensure_directory_exists(reports_dir)
    
    # 2. Encontrar todas as subpastas
    print(f"Buscando subpastas em: {data_dir}")
    subfolders = find_subfolders(data_dir)
    
    if not subfolders:
        print(f"ERRO: Nenhuma subpasta encontrada em {data_dir}")
        sys.exit(1)
    
    print(f"Encontradas {len(subfolders)} subpastas:")
    for folder in subfolders:
        print(f"  - {os.path.basename(folder)}")
    
    # 3. Preparar para armazenar métricas
    output_metrics = []
    
    # 4. Processar cada subpasta
    for folder in subfolders:
        train_models_for_folder(folder, output_metrics, artifact_base_dir)
    
    # 5. Criar e salvar DataFrame de métricas
    if output_metrics:
        metrics_df = pd.DataFrame(output_metrics)
        
        # Formatar timestamp para nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(reports_dir, f"baseline_country_comparison_{timestamp}.csv")
        
        # Salvar métricas
        metrics_df.to_csv(report_path, index=False)
        print(f"\nRelatório de métricas salvo em: {report_path}")
        
        # Exibir resumo das métricas
        print("\nResumo das métricas (média de F1-score por modelo):")
        model_summary = metrics_df.groupby("model")["f1"].mean().sort_values(ascending=False)
        print(model_summary)
        
        print("\nResumo das métricas (top 3 pastas por F1-score):")
        top_folders = metrics_df.sort_values("f1", ascending=False).head(3)[["folder", "model", "f1"]]
        print(top_folders)
    else:
        print("AVISO: Nenhuma métrica foi coletada.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"\nTempo total de execução: {total_time/60:.2f} minutos")