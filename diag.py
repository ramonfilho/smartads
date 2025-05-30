#!/usr/bin/env python
"""Investigação mais profunda das features de topic"""

import pandas as pd
import os

base_path = "/Users/ramonmoreira/desktop/smart_ads"

print("=== INVESTIGAÇÃO DETALHADA DAS FEATURES DE TOPIC ===\n")

# 1. Verificar o arquivo de importância combinada
print("1. Verificando arquivo de importância combinada...")
combined_path = os.path.join(base_path, "reports/feature_importance_results/feature_importance_combined.csv")
if os.path.exists(combined_path):
    combined_df = pd.read_csv(combined_path)
    topic_in_combined = combined_df[combined_df['Feature'].str.contains('topic_|dominant_topic', na=False)]
    print(f"   Features de topic em 'feature_importance_combined.csv': {len(topic_in_combined)}")
    
    if len(topic_in_combined) > 0:
        print("\n   Top 5 features de topic por importância:")
        for idx, row in topic_in_combined.sort_values('Mean_Importance', ascending=False).head().iterrows():
            print(f"   - {row['Feature']}: {row['Mean_Importance']:.6f}")
else:
    print("   Arquivo não encontrado")

# 2. Verificar datasets em diferentes estágios
print("\n2. Rastreando features de topic através dos estágios:")

stages = [
    ("02_processed", "data/new/02_processed/train.csv"),
    ("03_feature_engineering", "data/new/03_feature_engineering_1/train.csv"),
    ("04_feature_selection", "data/new/04_feature_selection/train.csv")
]

for stage_name, path in stages:
    full_path = os.path.join(base_path, path)
    if os.path.exists(full_path):
        df = pd.read_csv(full_path, nrows=1)
        topic_cols = [col for col in df.columns if 'topic_' in col or 'dominant_topic' in col]
        print(f"   {stage_name}: {len(topic_cols)} features de topic")
    else:
        print(f"   {stage_name}: arquivo não encontrado")

# 3. Verificar ordem de processamento
print("\n3. Análise da ordem de processamento:")
print("   Se as features de topic aparecem apenas no estágio 03 ou depois,")
print("   elas foram criadas APÓS a análise de importância do estágio 02.")

# 4. Carregar lista completa de features recomendadas
print("\n4. Features de topic recomendadas (detalhes):")
with open(os.path.join(base_path, "reports/feature_importance_results/recommended_features.txt"), 'r') as f:
    all_recommended = [line.strip() for line in f.readlines()]

topic_features = sorted([f for f in all_recommended if 'topic_' in f or 'dominant_topic' in f])
print(f"   Total: {len(topic_features)}")
print("\n   Lista completa:")
for feat in topic_features:
    print(f"   - {feat}")

print("\n" + "="*50)
print("\nCONCLUSÃO PRELIMINAR:")
print("As features de topic foram adicionadas ao pipeline mas não passaram")
print("pela análise de feature importance. Elas foram incluídas diretamente")
print("no conjunto final de features recomendadas.")