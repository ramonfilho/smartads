#!/usr/bin/env python
"""
Re-avaliar modelo GMM já treinado com métricas corretas.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Carregar o modelo salvo
model_path = "/Users/ramonmoreira/desktop/smart_ads/models/artifacts/gmm_ranking_optimized/gmm_ranking_wrapper.joblib"
gmm_wrapper = joblib.load(model_path)

# Carregar dados de validação
val_df = pd.read_csv("/Users/ramonmoreira/desktop/smart_ads/data/new/03_feature_engineering/validation.csv")

# Separar X e y
y_val = val_df['target']
X_val = val_df.drop('target', axis=1)

# Fazer predições
y_pred_proba = gmm_wrapper.predict_proba(X_val)[:, 1]

# Avaliar
from src.modeling.gmm_ranking_trainer import GMMRankingTrainer
trainer = GMMRankingTrainer({})  # Config vazia, só precisamos dos métodos
metrics, decile_stats = trainer.evaluate_ranking_metrics(y_val, y_pred_proba)

print(f"GINI: {metrics['gini']:.4f}")
print(f"Top Decile Lift: {metrics['top_decile_lift']:.2f}x")
print(f"Top 20% Recall: {metrics['top_20pct_recall']:.2%}")