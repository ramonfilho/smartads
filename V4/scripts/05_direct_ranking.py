#!/usr/bin/env python
"""
Modelo de ranking direto com LightGBM - mais simples e rápido.
CORRIGIDO: Limpa nomes de features para compatibilidade.
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import sys
import os

PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

def clean_column_names(df):
    """Limpa nomes de colunas para LightGBM."""
    rename_dict = {}
    for col in df.columns:
        # Remover/substituir caracteres problemáticos
        clean_name = re.sub(r'[^\w\s]', '_', str(col))  # Remove caracteres especiais
        clean_name = re.sub(r'\s+', '_', clean_name)    # Substitui espaços
        clean_name = re.sub(r'__+', '_', clean_name)    # Remove underscores múltiplos
        clean_name = clean_name.strip('_')               # Remove underscores nas extremidades
        
        if clean_name != col:
            rename_dict[col] = clean_name
    
    if rename_dict:
        print(f"Limpando {len(rename_dict)} nomes de colunas...")
        df = df.rename(columns=rename_dict)
    
    return df

# Carregar dados
print("Carregando dados...")
train_df = pd.read_csv("/Users/ramonmoreira/desktop/smart_ads/data/unified_v1/04_feature_selection/train.csv")
val_df = pd.read_csv("/Users/ramonmoreira/desktop/smart_ads/data/unified_v1/04_feature_selection/validation.csv")

print(f"Train shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")

# Limpar nomes de colunas
train_df = clean_column_names(train_df)
val_df = clean_column_names(val_df)

# Separar features e target
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_val = val_df.drop('target', axis=1)
y_val = val_df['target']

print(f"\nTaxa de conversão - Train: {y_train.mean():.4f}")
print(f"Taxa de conversão - Val: {y_val.mean():.4f}")

# Modelo direto otimizado para ranking
print("\nTreinando LightGBM direto...")
model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    is_unbalance=True,
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

# Treinar
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)],
          eval_metric='auc',
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

# Avaliar
print("\nFazendo predições...")
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Calcular métricas de ranking
def evaluate_ranking(y_true, y_pred_proba):
    """Avalia métricas de ranking."""
    # Criar DataFrame para análise
    df = pd.DataFrame({
        'y_true': y_true,
        'y_proba': y_pred_proba
    })
    
    # Análise por decil
    df['decile'] = pd.qcut(df['y_proba'].rank(method='first'), 
                           q=10, labels=range(1, 11))
    
    decile_stats = df.groupby('decile').agg({
        'y_true': ['sum', 'count', 'mean']
    })
    decile_stats.columns = ['conversions', 'total', 'rate']
    
    # Calcular lift
    overall_rate = df['y_true'].mean()
    decile_stats['lift'] = decile_stats['rate'] / overall_rate
    
    # KS Statistic
    df_sorted = df.sort_values('y_proba', ascending=False)
    cum_pos = np.cumsum(df_sorted['y_true']) / df['y_true'].sum()
    cum_neg = np.cumsum(~df_sorted['y_true']) / (~df['y_true']).sum()
    ks_statistic = np.max(np.abs(cum_pos - cum_neg))
    
    # GINI
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc - 1
    
    # Top-K metrics
    n_samples = len(df)
    top_10_pct = int(n_samples * 0.1)
    top_20_pct = int(n_samples * 0.2)
    
    df_sorted_desc = df.sort_values('y_proba', ascending=False)
    
    metrics = {
        'gini': gini,
        'auc': auc,
        'ks_statistic': ks_statistic,
        'top_decile_lift': decile_stats.loc[10, 'lift'],
        'top_2deciles_lift': decile_stats.loc[[9, 10], 'lift'].mean(),
        'top_10pct_recall': df_sorted_desc.head(top_10_pct)['y_true'].sum() / df['y_true'].sum(),
        'top_20pct_recall': df_sorted_desc.head(top_20_pct)['y_true'].sum() / df['y_true'].sum(),
    }
    
    return metrics, decile_stats

# Avaliar modelo
metrics, decile_stats = evaluate_ranking(y_val, y_pred_proba)

print("\n" + "="*60)
print("LIGHTGBM DIRETO - RESULTADOS")
print("="*60)
print(f"GINI: {metrics['gini']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")
print(f"KS Statistic: {metrics['ks_statistic']:.4f}")
print(f"Top Decile Lift: {metrics['top_decile_lift']:.2f}x")
print(f"Top 2 Deciles Lift: {metrics['top_2deciles_lift']:.2f}x")
print(f"Top 10% Recall: {metrics['top_10pct_recall']:.2%}")
print(f"Top 20% Recall: {metrics['top_20pct_recall']:.2%}")

print("\nANÁLISE POR DECIL:")
print(decile_stats.round(4))

# Verificar critérios de sucesso
print("\nCRITÉRIOS DE SUCESSO:")
print(f"Top Decile Lift > 3x: {'✅ PASSOU' if metrics['top_decile_lift'] > 3.0 else '❌ FALHOU'}")
print(f"Top 20% Recall > 50%: {'✅ PASSOU' if metrics['top_20pct_recall'] > 0.5 else '❌ FALHOU'}")

# Comparar com GMM
print("\n" + "="*60)
print("COMPARAÇÃO COM GMM")
print("="*60)
print("Métrica          | LightGBM | GMM Best")
print("-----------------|----------|----------")
print(f"GINI             | {metrics['gini']:.4f}   | 0.3865")
print(f"Top Decile Lift  | {metrics['top_decile_lift']:.2f}x    | 2.85x")
print(f"Top 20% Recall   | {metrics['top_20pct_recall']:.1%}    | 42.4%")
print("="*60)

# Salvar modelo se for melhor
if metrics['top_decile_lift'] > 2.85:
    import joblib
    model_path = os.path.join(PROJECT_ROOT, "models/artifacts/unified_v1/lightgbm_direct_ranking.joblib")
    joblib.dump(model, model_path)
    print(f"\n✅ Modelo salvo em: {model_path}")