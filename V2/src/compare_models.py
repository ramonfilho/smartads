#!/usr/bin/env python
"""
Compara predições do modelo do pipeline vs modelo do notebook.
"""

import pandas as pd
import joblib
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# Caminhos
test_file = "V2/arquivos_modelo/20251105_101902/test_set_predictions.csv"
modelo_pipeline = "V2/arquivos_modelo/20251105_101902/modelo_lead_scoring_v1_devclub_rf_temporal_single.pkl"
features_pipeline = "V2/arquivos_modelo/20251105_101902/features_ordenadas_v1_devclub_rf_temporal_single.json"
modelo_notebook = "V2/arquivos_modelo/modelo_lead_scoring_v1_devclub_rf_temporal_single.pkl"
features_notebook = "V2/arquivos_modelo/features_ordenadas_v1_devclub_rf_temporal_single-3.json"

print("=" * 80)
print("COMPARAÇÃO: MODELO PIPELINE vs MODELO NOTEBOOK")
print("=" * 80)

# 1. Carregar dados de teste
print("\n1. Carregando dados de teste...")
df_test = pd.read_csv(test_file)
print(f"   ✓ {len(df_test):,} registros carregados")
print(f"   ✓ Probabilidade do pipeline já presente no arquivo")

# 2. Preparar features
print("\n2. Preparando features...")
probabilidade_pipeline = df_test['probabilidade'].values
target_real = df_test['target_real'].values
X = df_test.drop(columns=['target_real', 'probabilidade'])
print(f"   ✓ {len(X.columns)} features")

# 3. Carregar e testar modelo do pipeline
print("\n3. Testando modelo do PIPELINE...")
with open(features_pipeline, 'r') as f:
    features_esperadas_pipeline = json.load(f)['feature_names']

# Verificar se features batem
X_pipeline = X[features_esperadas_pipeline]
modelo_p = joblib.load(modelo_pipeline)
prob_pipeline_rerun = modelo_p.predict_proba(X_pipeline)[:, 1]

# Verificar se as probabilidades batem (devem ser idênticas)
diff_pipeline = np.abs(probabilidade_pipeline - prob_pipeline_rerun).max()
print(f"   ✓ Modelo carregado")
print(f"   ✓ Diferença máxima ao re-rodar: {diff_pipeline:.10f}")
if diff_pipeline < 1e-10:
    print("   ✓ Predições IDÊNTICAS (modelo consistente)")
else:
    print(f"   ⚠️  Diferença detectada!")

# 4. Carregar e testar modelo do notebook
print("\n4. Testando modelo do NOTEBOOK...")
with open(features_notebook, 'r') as f:
    features_esperadas_notebook = json.load(f)['feature_names']

# Verificar se features batem
if list(features_esperadas_pipeline) != list(features_esperadas_notebook):
    print("   ⚠️  FEATURES DIFERENTES!")
    print(f"   Pipeline: {len(features_esperadas_pipeline)} features")
    print(f"   Notebook: {len(features_esperadas_notebook)} features")
else:
    print(f"   ✓ Features idênticas ({len(features_esperadas_notebook)} features)")

X_notebook = X[features_esperadas_notebook]
modelo_n = joblib.load(modelo_notebook)
prob_notebook = modelo_n.predict_proba(X_notebook)[:, 1]
print(f"   ✓ Modelo carregado e predições feitas")

# 5. Comparar predições
print("\n5. COMPARANDO PREDIÇÕES")
print("=" * 80)

# Estatísticas descritivas
print(f"\nEstatísticas Descritivas:")
print(f"{'Métrica':<30} {'Pipeline':>15} {'Notebook':>15} {'Diferença':>15}")
print("-" * 80)
print(f"{'Média':<30} {prob_pipeline_rerun.mean():>15.6f} {prob_notebook.mean():>15.6f} {abs(prob_pipeline_rerun.mean() - prob_notebook.mean()):>15.6f}")
print(f"{'Mediana':<30} {np.median(prob_pipeline_rerun):>15.6f} {np.median(prob_notebook):>15.6f} {abs(np.median(prob_pipeline_rerun) - np.median(prob_notebook)):>15.6f}")
print(f"{'Desvio Padrão':<30} {prob_pipeline_rerun.std():>15.6f} {prob_notebook.std():>15.6f} {abs(prob_pipeline_rerun.std() - prob_notebook.std()):>15.6f}")
print(f"{'Mínimo':<30} {prob_pipeline_rerun.min():>15.6f} {prob_notebook.min():>15.6f} {abs(prob_pipeline_rerun.min() - prob_notebook.min()):>15.6f}")
print(f"{'Máximo':<30} {prob_pipeline_rerun.max():>15.6f} {prob_notebook.max():>15.6f} {abs(prob_pipeline_rerun.max() - prob_notebook.max()):>15.6f}")

# Diferenças absolutas
diffs = np.abs(prob_pipeline_rerun - prob_notebook)
print(f"\nDiferenças Absolutas:")
print(f"  Média: {diffs.mean():.6f}")
print(f"  Mediana: {np.median(diffs):.6f}")
print(f"  Máxima: {diffs.max():.6f}")
print(f"  Mínima: {diffs.min():.6f}")

# Correlações
pearson_corr, _ = pearsonr(prob_pipeline_rerun, prob_notebook)
spearman_corr, _ = spearmanr(prob_pipeline_rerun, prob_notebook)
print(f"\nCorrelações:")
print(f"  Pearson: {pearson_corr:.6f}")
print(f"  Spearman: {spearman_corr:.6f}")

# Métricas de erro
mae = mean_absolute_error(prob_pipeline_rerun, prob_notebook)
rmse = np.sqrt(mean_squared_error(prob_pipeline_rerun, prob_notebook))
print(f"\nMétricas de Erro:")
print(f"  MAE: {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")

# Predições idênticas?
identicos = (diffs < 1e-10).sum()
print(f"\nPredições Idênticas (diff < 1e-10): {identicos:,}/{len(diffs):,} ({identicos/len(diffs)*100:.2f}%)")

# 6. Salvar comparação
print("\n6. Salvando comparação...")
df_comparacao = pd.DataFrame({
    'target_real': target_real,
    'prob_pipeline': prob_pipeline_rerun,
    'prob_notebook': prob_notebook,
    'diff_absoluta': diffs,
    'diff_relativa': diffs / (prob_pipeline_rerun + 1e-10)  # evitar divisão por zero
})

output_file = "V2/arquivos_modelo/20251105_101902/model_comparison.csv"
df_comparacao.to_csv(output_file, index=False)
print(f"   ✓ Comparação salva em: {output_file}")

print("\n" + "=" * 80)
print("CONCLUSÃO")
print("=" * 80)

if diffs.max() < 1e-10:
    print("✅ MODELOS SÃO IDÊNTICOS! Pipeline reproduziu o notebook perfeitamente.")
elif pearson_corr > 0.99:
    print(f"✅ MODELOS MUITO SIMILARES! Correlação = {pearson_corr:.6f}")
    print(f"   Diferença média: {diffs.mean():.6f}")
elif pearson_corr > 0.95:
    print(f"⚠️  MODELOS SIMILARES mas com diferenças. Correlação = {pearson_corr:.6f}")
    print(f"   Diferença média: {diffs.mean():.6f}")
else:
    print(f"❌ MODELOS DIFERENTES! Correlação = {pearson_corr:.6f}")
    print(f"   Diferença média: {diffs.mean():.6f}")

print("=" * 80)
