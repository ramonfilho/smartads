"""
ETAPA 2: Carregar Dados de Teste

Carrega LF_19-24_pipeline_v2_predictions_FIXED.xlsx e analisa estrutura.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import json
from pathlib import Path

def main():
    print("\n" + "=" * 80)
    print("ETAPA 2: CARREGAMENTO DOS DADOS DE TESTE")
    print("=" * 80)

    # Carregar arquivo
    test_file = Path(__file__).parent.parent / "files" / "LF_19-24_pipeline_v2_predictions_FIXED.xlsx"
    print(f"\n📂 Carregando: {test_file.name}")

    df = pd.read_excel(test_file)
    print(f"✅ Carregado: {len(df):,} registros x {len(df.columns)} colunas")

    # Verificar colunas importantes
    key_cols = ['lead_score', 'decil', 'target', 'Data', 'E-mail']
    print(f"\n🔑 Colunas Importantes:")
    for col in key_cols:
        if col in df.columns:
            print(f"  ✅ {col}")
        else:
            print(f"  ❌ {col}")

    # Stats básicos
    if 'target' in df.columns:
        print(f"\n📊 Target:")
        print(f"  Conversões: {df['target'].sum()} ({df['target'].mean()*100:.2f}%)")

    if 'lead_score' in df.columns:
        print(f"\n📊 Lead Score:")
        print(f"  Min: {df['lead_score'].min():.4f}")
        print(f"  Max: {df['lead_score'].max():.4f}")
        print(f"  Mean: {df['lead_score'].mean():.4f}")

    if 'decil' in df.columns:
        print(f"\n📊 Decis:")
        print(df['decil'].value_counts().sort_index())

    # Validar com metadata
    metadata_path = Path(__file__).parent.parent / "arquivos_modelo" / "model_metadata_v1_devclub_rf_temporal_single-3.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    expected = metadata['training_data']['test_records']
    print(f"\n🔍 Validação: Esperado {expected:,} vs Obtido {len(df):,}")

    if len(df) == expected:
        print("  ✅ MATCH!")
    else:
        print(f"  ⚠️  Diferença: {len(df) - expected:+,}")

    # Salvar
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    df.to_pickle(output_dir / "test_data_raw.pkl")
    print(f"\n💾 Salvo em: {output_dir / 'test_data_raw.pkl'}")

    print("\n✅ Pronto para próxima etapa!")

if __name__ == "__main__":
    main()
