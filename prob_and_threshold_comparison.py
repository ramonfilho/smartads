#!/usr/bin/env python
"""
Script diagnóstico para comparar probabilidades e thresholds entre modelo de referência e pipeline.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import precision_recall_curve

# Caminhos para os arquivos
PIPELINE_PREDICTIONS_PATH = "/Users/ramonmoreira/desktop/smart_ads/inference/output/predictions_20250517_150650.csv"
REFERENCE_PREDICTIONS_PATH = "/Users/ramonmoreira/desktop/smart_ads/reports/calibration_validation_two_models/20250517_152839/gmm_test_results.csv"
MODEL_PATH_REF = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250508_130725/gmm_calibrated.joblib"
THRESHOLD_PATH_REF = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250508_130725/threshold.txt"

# Carregar predições
pipeline_df = pd.read_csv(PIPELINE_PREDICTIONS_PATH)
reference_df = pd.read_csv(REFERENCE_PREDICTIONS_PATH)

# Verificar threshold de referência
try:
    with open(THRESHOLD_PATH_REF, 'r') as f:
        threshold_ref = float(f.read().strip())
    print(f"Threshold de referência: {threshold_ref}")
except:
    threshold_ref = 0.1
    print(f"Threshold padrão usado: {threshold_ref}")

# Criar diretório para resultados
output_dir = "/Users/ramonmoreira/desktop/smart_ads/inference/output/diagnostics"
os.makedirs(output_dir, exist_ok=True)

# Estatísticas gerais das probabilidades
print("\n=== Estatísticas das Probabilidades ===")
for name, df in [("Pipeline", pipeline_df), ("Referência", reference_df)]:
    desc = df['probability'].describe()
    print(f"\n{name}:")
    print(f"  Min: {desc['min']:.6f}")
    print(f"  Max: {desc['max']:.6f}")
    print(f"  Mean: {desc['mean']:.6f}")
    print(f"  Std: {desc['std']:.6f}")
    print(f"  25%: {desc['25%']:.6f}")
    print(f"  50%: {desc['50%']:.6f}")
    print(f"  75%: {desc['75%']:.6f}")
    print(f"  Positivos (>= {threshold_ref}): {(df['probability'] >= threshold_ref).mean():.6f}")

# Histograma comparativo
plt.figure(figsize=(12, 6))
bins = np.linspace(0, max(pipeline_df['probability'].max(), reference_df['probability'].max()), 100)

plt.hist(pipeline_df['probability'], bins=bins, alpha=0.5, label='Pipeline')
plt.hist(reference_df['probability'], bins=bins, alpha=0.5, label='Referência')
plt.axvline(x=threshold_ref, color='r', linestyle='--', label=f'Threshold ({threshold_ref})')

plt.xlabel('Probabilidade')
plt.ylabel('Frequência')
plt.title('Comparação de Probabilidades entre Pipeline e Referência')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, "probabilidades_histograma.png"), dpi=300)

# Histograma para probabilidades baixas (região do threshold)
plt.figure(figsize=(12, 6))
threshold_range = min(0.3, max(pipeline_df['probability'].max(), reference_df['probability'].max()))
bins = np.linspace(0, threshold_range, 100)

plt.hist(pipeline_df['probability'], bins=bins, alpha=0.5, label='Pipeline')
plt.hist(reference_df['probability'], bins=bins, alpha=0.5, label='Referência')
plt.axvline(x=threshold_ref, color='r', linestyle='--', label=f'Threshold ({threshold_ref})')

plt.xlabel('Probabilidade')
plt.ylabel('Frequência')
plt.title('Zoom nas Probabilidades Baixas')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, "probabilidades_zoom.png"), dpi=300)

# Curvas Precision-Recall para encontrar o threshold ótimo
plt.figure(figsize=(10, 8))

# Dados de referência
precision_ref, recall_ref, thresholds_ref = precision_recall_curve(
    reference_df['true'], reference_df['probability']
)
f1_scores_ref = 2 * (precision_ref * recall_ref) / (precision_ref + recall_ref + 1e-9)
best_idx_ref = np.argmax(f1_scores_ref)
best_threshold_ref = thresholds_ref[best_idx_ref] if best_idx_ref < len(thresholds_ref) else threshold_ref

# Dados da pipeline
if 'true' in pipeline_df.columns:
    precision_pipe, recall_pipe, thresholds_pipe = precision_recall_curve(
        pipeline_df['true'], pipeline_df['probability']
    )
    f1_scores_pipe = 2 * (precision_pipe * recall_pipe) / (precision_pipe + recall_pipe + 1e-9)
    best_idx_pipe = np.argmax(f1_scores_pipe)
    best_threshold_pipe = thresholds_pipe[best_idx_pipe] if best_idx_pipe < len(thresholds_pipe) else threshold_ref
    
    # Plotar curvas
    plt.plot(recall_ref, precision_ref, label='Referência')
    plt.plot(recall_pipe, precision_pipe, label='Pipeline')
    
    # Marcar thresholds ótimos
    plt.scatter(recall_ref[best_idx_ref], precision_ref[best_idx_ref], color='blue', s=100, marker='o',
               label=f'Ref (threshold={best_threshold_ref:.4f}, F1={f1_scores_ref[best_idx_ref]:.4f})')
    
    plt.scatter(recall_pipe[best_idx_pipe], precision_pipe[best_idx_pipe], color='orange', s=100, marker='o',
               label=f'Pipeline (threshold={best_threshold_pipe:.4f}, F1={f1_scores_pipe[best_idx_pipe]:.4f})')
    
    plt.title('Curvas Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=300)
    
    print(f"\nThresholds ótimos calculados:")
    print(f"  Referência: {best_threshold_ref:.6f} (F1={f1_scores_ref[best_idx_ref]:.4f})")
    print(f"  Pipeline: {best_threshold_pipe:.6f} (F1={f1_scores_pipe[best_idx_pipe]:.4f})")
else:
    print("\nNão é possível calcular as curvas precision-recall para a pipeline (coluna 'true' ausente)")

print(f"\nResultados diagnósticos salvos em: {output_dir}")