"""
ETAPA 1: Carregar Modelo e Dados de Teste

Objetivo:
- Carregar o modelo treinado (RandomForest)
- Carregar os dados de teste (temporal split)
- Verificar que tudo está correto antes de prosseguir

AGUARDAR APROVAÇÃO DO USUÁRIO antes de prosseguir para análise de erros.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

def load_model_artifacts():
    """Carrega modelo, features e metadata."""
    model_dir = Path(__file__).parent.parent / "arquivos_modelo"
    model_name = "v1_devclub_rf_temporal_single"

    print("📦 Carregando artefatos do modelo...")

    # Carregar modelo
    model_path = model_dir / f"modelo_lead_scoring_{model_name}.pkl"
    model = joblib.load(model_path)
    print(f"✅ Modelo carregado: {type(model).__name__}")

    # Carregar features
    features_path = model_dir / f"features_ordenadas_{model_name}-3.json"
    with open(features_path, 'r') as f:
        features_data = json.load(f)
    feature_names = features_data['feature_names']
    print(f"✅ Features carregadas: {len(feature_names)} features")

    # Carregar metadata
    metadata_path = model_dir / f"model_metadata_{model_name}-3.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✅ Metadata carregado")

    return model, feature_names, metadata

def load_test_data():
    """
    Carrega dados de teste do split temporal.

    NOTA: Preciso identificar onde estão os dados de teste salvos.
    Opções:
    1. Reprocessar pipeline completo para obter split
    2. Carregar dados salvos (se existirem)
    """
    print("\n🔍 Procurando dados de teste...")

    # Verificar se há dados salvos
    data_dir = Path(__file__).parent.parent / "data" / "processed"

    if not data_dir.exists():
        print("⚠️  Diretório 'data/processed' não encontrado")
        print("Precisamos rodar o pipeline de treino para gerar os dados de teste")
        return None

    # Procurar arquivos de teste
    test_files = list(data_dir.glob("*test*.csv")) + list(data_dir.glob("*test*.pkl"))

    if not test_files:
        print("⚠️  Nenhum arquivo de teste encontrado")
        print("Precisamos rodar o pipeline de treino para gerar os dados de teste")
        return None

    print(f"✅ Encontrados {len(test_files)} arquivos de teste:")
    for f in test_files:
        print(f"  - {f.name}")

    return test_files

def main():
    """Executa etapa 1: Carregamento."""

    print("\n" + "=" * 80)
    print("ANÁLISE DE ERRO - ETAPA 1: CARREGAMENTO")
    print("=" * 80)

    # Carregar modelo
    model, feature_names, metadata = load_model_artifacts()

    print("\n📊 INFORMAÇÕES DO MODELO:")
    print(f"  - Tipo: {metadata['model_info']['model_type']}")
    print(f"  - Split: {metadata['model_info']['split_type']}")
    print(f"  - Treinado em: {metadata['model_info']['trained_at']}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - AUC: {metadata['performance_metrics']['auc']:.4f}")
    print(f"  - Lift Máximo: {metadata['performance_metrics']['lift_maximum']:.2f}")

    print("\n📊 DADOS DE TREINO/TESTE:")
    print(f"  - Total de registros: {metadata['training_data']['total_records']:,}")
    print(f"  - Treino: {metadata['training_data']['training_records']:,}")
    print(f"  - Teste: {metadata['training_data']['test_records']:,}")
    print(f"  - Taxa de conversão (teste): {metadata['training_data']['target_distribution']['test_positive_rate']:.4f}")
    print(f"  - Conversões (teste): {metadata['training_data']['target_distribution']['test_positive_count']}")

    print("\n📅 SPLIT TEMPORAL:")
    temporal = metadata['training_data']['temporal_split']
    print(f"  - Período: {temporal['period_start']} a {temporal['period_end']}")
    print(f"  - Data de corte: {temporal['cut_date']}")
    print(f"  - Dias de treino: {temporal['training_days']}")
    print(f"  - Total de dias: {temporal['total_days']}")

    # Tentar carregar dados de teste
    test_files = load_test_data()

    print("\n" + "=" * 80)
    print("PRÓXIMOS PASSOS:")
    print("=" * 80)

    if test_files is None:
        print("⚠️  AÇÃO NECESSÁRIA:")
        print("   Precisamos dos dados de teste para fazer a análise de erro.")
        print("   Opções:")
        print("   1. Rodar pipeline de treino e salvar dados de teste")
        print("   2. Modificar train_pipeline.py para salvar splits")
        print("\n❓ O QUE VOCÊ PREFERE FAZER?")
    else:
        print("✅ Modelo e dados carregados com sucesso!")
        print("   Pronto para prosseguir para ETAPA 2: Análise de Erros")
        print("\n❓ POSSO PROSSEGUIR PARA A PRÓXIMA ETAPA?")

if __name__ == "__main__":
    main()
