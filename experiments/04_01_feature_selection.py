#ESTE SCRIPT TRATA OS VALORES AUSENTES NAS FEATURES PRODUZIDAS PELA TERCEIRA RODADA DE FEATURE ENGINEERING, QUE HAVIAM FICADO SEM ESSE TRATAMENTO, ANTES DE FAZER A SELEÇÃO DE FEATURES. 
#ELE PRODUZ UM DATASET COM 188 FEATURES.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import re
import shap
import warnings
warnings.filterwarnings('ignore')
from src.preprocessing.data_cleaning import handle_missing_values

# Definir caminhos de entrada e saída
input_dir = 'data/02_3_processed_text_code6'
output_dir = 'data/03_5_feature_selection_final_treated'

# Criar diretórios de saída
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Pasta '{output_dir}' criada para salvar os datasets processados")
else:
    print(f"Pasta '{output_dir}' já existe")

# Criar pasta para resultados de análise
analysis_dir = 'eda_results/feature_importance_results'
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)
    print(f"Pasta '{analysis_dir}' criada para salvar resultados de análise")
else:
    print(f"Pasta '{analysis_dir}' já existe")

# 1 - Carregar dataset de treino
print("Carregando datasets...")
try:
    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    print(f"Dataset de treino carregado: {train_df.shape[0]} linhas, {train_df.shape[1]} colunas")
    
    val_df = pd.read_csv(os.path.join(input_dir, "validation.csv"))
    print(f"Dataset de validação carregado: {val_df.shape[0]} linhas, {val_df.shape[1]} colunas")
    
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
    print(f"Dataset de teste carregado: {test_df.shape[0]} linhas, {test_df.shape[1]} colunas")
except Exception as e:
    print(f"Erro ao carregar datasets: {e}")
    exit(1)

# Após carregar os datasets, mas antes de fazer a seleção de features
print("\n--- Verificando e tratando valores ausentes ---")
train_df, missing_params = handle_missing_values(train_df, fit=True)
val_df, _ = handle_missing_values(val_df, fit=False, params=missing_params)
test_df, _ = handle_missing_values(test_df, fit=False, params=missing_params)

# Registrar estatísticas após o tratamento
print(f"Valores ausentes restantes após tratamento:")
print(f"  Train: {train_df.isna().sum().sum()} valores em {(train_df.isna().sum() > 0).sum()} colunas")
print(f"  Validation: {val_df.isna().sum().sum()} valores em {(val_df.isna().sum() > 0).sum()} colunas")
print(f"  Test: {test_df.isna().sum().sum()} valores em {(test_df.isna().sum() > 0).sum()} colunas")

# Vamos usar o dataset de treino para a análise de features
df = train_df

# 2 - Identificar coluna de lançamento (usando especificamente 'lançamento')
print("\nIdentificando coluna de lançamento...")
launch_col = 'lançamento'  # Nome específico conforme indicado

if launch_col in df.columns:
    print(f"Coluna de lançamento encontrada: '{launch_col}'")
    n_launches = df[launch_col].nunique()
    print(f"Número de lançamentos: {n_launches}")
    print(f"Lançamentos identificados: {sorted(df[launch_col].unique())}")
    print(f"Distribuição de lançamentos:\n{df[launch_col].value_counts(normalize=True)*100}")
else:
    print(f"Coluna '{launch_col}' não encontrada. Verificando alternativas...")
    # Procurar colunas alternativas
    alt_launch_cols = [col for col in df.columns if 'lanc' in col.lower() or 'launch' in col.lower()]
    if alt_launch_cols:
        launch_col = alt_launch_cols[0]
        print(f"Usando coluna alternativa: '{launch_col}'")
        print(f"Número de lançamentos: {df[launch_col].nunique()}")
        print(f"Distribuição de lançamentos:\n{df[launch_col].value_counts(normalize=True)*100}")
    else:
        launch_col = None
        print("Nenhuma coluna de lançamento identificada.")

# 3 - Preparar dados para modelagem
print("\nPreparando dados para análise de importância...")
# Verificar coluna target
if 'target' not in df.columns:
    print("Coluna 'target' não encontrada. Verificando alternativas...")
    target_cols = [col for col in df.columns if col.lower() in ['target', 'comprou', 'converted', 'conversion']]
    if target_cols:
        target_col = target_cols[0]
        print(f"Usando '{target_col}' como target.")
    else:
        raise ValueError("Não foi possível encontrar uma coluna target.")
else:
    target_col = 'target'

# Selecionar colunas numéricas para análise
numeric_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

# Remover colunas com mais de 90% de valores ausentes
missing_pct = df[numeric_cols].isna().mean()
high_missing_cols = missing_pct[missing_pct > 0.9].index.tolist()
if high_missing_cols:
    print(f"Removendo {len(high_missing_cols)} colunas com mais de 90% de valores ausentes")
    numeric_cols = [col for col in numeric_cols if col not in high_missing_cols]

# Remover colunas com variância zero
try:
    selector = VarianceThreshold(threshold=0)
    selector.fit(df[numeric_cols].fillna(0))
    zero_var_cols = [numeric_cols[i] for i, var in enumerate(selector.variances_) if var == 0]
    if zero_var_cols:
        print(f"Removendo {len(zero_var_cols)} colunas com variância zero")
        numeric_cols = [col for col in numeric_cols if col not in zero_var_cols]
except Exception as e:
    print(f"Erro ao verificar variância: {e}")

# Verificar se há features textuais no dataset
text_derived_cols = [col for col in numeric_cols if any(text_indicator in col for text_indicator in 
                                                     ['_tfidf_', '_sentiment', '_word_count', '_length', '_motiv_', '_has_question'])]
print(f"Features derivadas de texto identificadas: {len(text_derived_cols)}")
if text_derived_cols:
    print("Exemplos de features textuais:")
    for col in text_derived_cols[:5]:  # Mostrar alguns exemplos
        print(f"  - {col}")
    if len(text_derived_cols) > 5:
        print(f"  - ... e mais {len(text_derived_cols) - 5} features textuais")

# IMPORTANTE: Sanitizar nomes de colunas para evitar erro de caracteres especiais JSON
# Isso resolve o erro "Do not support special JSON characters in feature name"
print("\nSanitizando nomes das features para evitar problemas com caracteres especiais...")
rename_dict = {}
for col in numeric_cols:
    # Substituir caracteres especiais e espaços por underscores
    new_col = re.sub(r'[^0-9a-zA-Z_]', '_', col)
    # Garantir que não comece com número
    if new_col[0].isdigit():
        new_col = 'f_' + new_col
    # Verificar se já existe esse novo nome
    i = 1
    temp_col = new_col
    while temp_col in rename_dict.values():
        temp_col = f"{new_col}_{i}"
        i += 1
    new_col = temp_col
    
    # Só adicionar ao dicionário se o nome mudou
    if col != new_col:
        rename_dict[col] = new_col

# Aplicar renomeação
if rename_dict:
    print(f"Renomeando {len(rename_dict)} colunas para evitar erros com caracteres especiais")
    df = df.rename(columns=rename_dict)
    
    # Atualizar lista de colunas numéricas
    numeric_cols = [rename_dict.get(col, col) for col in numeric_cols]
    
    # Atualizar lista de colunas textuais
    text_derived_cols = [rename_dict.get(col, col) for col in text_derived_cols]

X = df[numeric_cols].fillna(0)
y = df[target_col]

print(f"Usando {len(numeric_cols)} features numéricas para análise")
print(f"Distribuição do target: {y.value_counts(normalize=True) * 100}")

# 4 - Análise de Multicolinearidade
print("\n--- Análise de Multicolinearidade ---")
# Identificar pares de features com correlação alta
corr_matrix = X.corr()
high_corr_pairs = []

# Limiar de correlação (ajustável)
corr_threshold = 0.8

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > corr_threshold:
            high_corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

# Ordenar por correlação absoluta
high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)

print(f"Encontrados {len(high_corr_pairs)} pares de features com correlação > {corr_threshold}:")
for i, pair in enumerate(high_corr_pairs[:10]):  # Mostrar os 10 primeiros
    print(f"{i+1}. {pair['feature1']} & {pair['feature2']}: {pair['correlation']:.4f}")

if len(high_corr_pairs) > 10:
    print(f"... e mais {len(high_corr_pairs) - 10} pares.")

# 5 - Verificação Específica: country_freq vs country_encoded
print("\n--- Análise de Redundância: country_freq vs country_encoded ---")
# Procurar versões sanitizadas dos nomes
country_freq_col = next((col for col in X.columns if 'country_freq' in col), None)
country_encoded_col = next((col for col in X.columns if 'country_encoded' in col), None)

if country_freq_col and country_encoded_col:
    # Calcular correlação
    corr, p_value = pearsonr(X[country_freq_col].fillna(0), X[country_encoded_col].fillna(0))
    print(f"Correlação entre {country_freq_col} e {country_encoded_col}: {corr:.4f} (p-value: {p_value:.4f})")
    
    # Avaliar valor preditivo relativo para o target
    corr_target_freq = pearsonr(X[country_freq_col].fillna(0), y)[0]
    corr_target_encoded = pearsonr(X[country_encoded_col].fillna(0), y)[0]
    
    print(f"Correlação com target:")
    print(f"- {country_freq_col}: {corr_target_freq:.4f}")
    print(f"- {country_encoded_col}: {corr_target_encoded:.4f}")
    
    recommendation = f"{country_freq_col if abs(corr_target_freq) > abs(corr_target_encoded) else country_encoded_col} parece ter maior valor preditivo."
    print(f"Recomendação: {recommendation}")
else:
    print("Colunas country_freq e/ou country_encoded não encontradas.")

# 6 - Separar dados para treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Treinamento: {X_train.shape[0]} amostras, Validação: {X_val.shape[0]} amostras")
print(f"Proporção da classe positiva no treino: {y_train.mean()*100:.2f}%")
print(f"Proporção da classe positiva na validação: {y_val.mean()*100:.2f}%")

# 7 - Definir funções de avaliação para dados desbalanceados
def evaluate_model(model, X, y, feature_names):
    """Avalia o modelo usando métricas adequadas para dados desbalanceados"""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:  # Para modelos como XGBoost com DMatrix
        y_proba = model.predict(X)
    
    # Calcular métricas
    # AUC - avalia ranking independente do threshold
    auc = roc_auc_score(y, y_proba)
    
    # Average Precision - média ponderada de precisões em diferentes thresholds
    ap = average_precision_score(y, y_proba)
    
    # Encontrar melhor F1-score ajustando threshold
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
    f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = 0 if len(thresholds) == 0 else thresholds[min(best_threshold_idx, len(thresholds)-1)]
    best_f1 = f1_scores[best_threshold_idx]
    
    print(f"Desempenho do modelo:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  Melhor F1-Score: {best_f1:.4f} (threshold: {best_threshold:.4f})")
    
    return {
        'auc': auc,
        'ap': ap,
        'f1': best_f1,
        'threshold': best_threshold
    }

# 8 - Análise de importância com múltiplos modelos
print("\n--- Iniciando análise de importância de features ---")

# 8.1 - RandomForest com validação cruzada para dados desbalanceados
print("\nAnalisando com RandomForest e validação cruzada para dados desbalanceados...")
try:
    # Usar validação cruzada estratificada para lidar com o desbalanceamento
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    rf_importances = np.zeros(len(numeric_cols))
    rf_metrics = {'auc': [], 'ap': [], 'f1': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_folds}")
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Calcular class_weight
        n_samples = len(y_fold_train)
        n_pos = y_fold_train.sum()
        n_neg = n_samples - n_pos
        weight_pos = (n_samples / (2 * n_pos)) if n_pos > 0 else 1.0
        weight_neg = (n_samples / (2 * n_neg)) if n_neg > 0 else 1.0
        
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            class_weight={0: weight_neg, 1: weight_pos},
            max_depth=10,
            random_state=42 + fold,
            n_jobs=-1
        )
        
        rf_model.fit(X_fold_train, y_fold_train)
        
        # Avaliar modelo
        metrics = evaluate_model(rf_model, X_fold_val, y_fold_val, numeric_cols)
        for key in rf_metrics:
            rf_metrics[key].append(metrics[key])
        
        # Acumular importâncias
        rf_importances += rf_model.feature_importances_
    
    # Calcular média das importâncias
    rf_importances /= n_folds
    
    # Criar dataframe de importância
    rf_importance = pd.DataFrame({
        'Feature': numeric_cols,
        'Importance_RF': rf_importances
    }).sort_values(by='Importance_RF', ascending=False)
    
    print("\nMétricas médias da validação cruzada (RandomForest):")
    for key, values in rf_metrics.items():
        print(f"  {key.upper()}: {np.mean(values):.4f} (±{np.std(values):.4f})")
    
    print("\nTop 15 features (RandomForest):")
    print(rf_importance.head(15))
except Exception as e:
    print(f"Erro ao executar RandomForest: {e}")
    # Criar dataframe vazio em caso de erro
    rf_importance = pd.DataFrame({
        'Feature': numeric_cols,
        'Importance_RF': [0] * len(numeric_cols)
    })

# 8.2 - LightGBM (com cuidado para evitar erros)
print("\nAnalisando com LightGBM...")
try:
    # Validação cruzada com LightGBM para dados desbalanceados
    lgb_importances = np.zeros(len(numeric_cols))
    lgb_metrics = {'auc': [], 'ap': [], 'f1': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_folds}")
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Calcular scale_pos_weight
        pos_scale = (y_fold_train == 0).sum() / max(1, (y_fold_train == 1).sum())
        
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'seed': 42 + fold,
            'learning_rate': 0.05,
            'scale_pos_weight': pos_scale,
            'n_jobs': -1
        }
        
        # Treinando modelo
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)]
        
        lgb_model = lgb.train(
            params, 
            train_data, 
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        
        # Avaliar modelo
        metrics = evaluate_model(lgb_model, X_fold_val, y_fold_val, numeric_cols)
        for key in lgb_metrics:
            lgb_metrics[key].append(metrics[key])
        
        # Acumular importâncias
        fold_importance = lgb_model.feature_importance(importance_type='gain')
        lgb_importances += fold_importance
    
    # Calcular média das importâncias
    lgb_importances /= n_folds
    
    # Criar dataframe de importância
    lgb_importance = pd.DataFrame({
        'Feature': numeric_cols,
        'Importance_LGB': lgb_importances
    }).sort_values(by='Importance_LGB', ascending=False)
    
    print("\nMétricas médias da validação cruzada (LightGBM):")
    for key, values in lgb_metrics.items():
        print(f"  {key.upper()}: {np.mean(values):.4f} (±{np.std(values):.4f})")

    print("\nTop 15 features (LightGBM):")
    print(lgb_importance.head(15))
except Exception as e:
    print(f"Erro ao executar LightGBM: {e}")
    print("Criando dataframe de importância vazio para LightGBM")
    # Criar dataframe vazio em caso de erro
    lgb_importance = pd.DataFrame({
        'Feature': numeric_cols,
        'Importance_LGB': [0] * len(numeric_cols)
    })

# 8.3 - XGBoost
print("\nAnalisando com XGBoost...")
try:
    # Validação cruzada com XGBoost para dados desbalanceados
    xgb_importances = {}  # Dicionário para acumular importâncias
    xgb_metrics = {'auc': [], 'ap': [], 'f1': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_folds}")
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Calcular scale_pos_weight
        pos_scale = (y_fold_train == 0).sum() / max(1, (y_fold_train == 1).sum())
        
        # Preparar dados
        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train, feature_names=numeric_cols)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val, feature_names=numeric_cols)
        
        # Configuração para dados desbalanceados
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': pos_scale,
            'learning_rate': 0.05,
            'seed': 42 + fold,
            'tree_method': 'hist'
        }
        
        # Treinando o modelo
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Avaliar modelo
        dval_pred = xgb.DMatrix(X_fold_val, feature_names=numeric_cols)
        y_pred = xgb_model.predict(dval_pred)
        auc = roc_auc_score(y_fold_val, y_pred)
        ap = average_precision_score(y_fold_val, y_pred)
        
        # Encontrar melhor F1-score ajustando threshold
        precisions, recalls, thresholds = precision_recall_curve(y_fold_val, y_pred)
        f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = 0 if len(thresholds) == 0 else thresholds[min(best_threshold_idx, len(thresholds)-1)]
        best_f1 = f1_scores[best_threshold_idx]
        
        xgb_metrics['auc'].append(auc)
        xgb_metrics['ap'].append(ap)
        xgb_metrics['f1'].append(best_f1)
        
        # Acumular importâncias
        importance_dict = xgb_model.get_score(importance_type='gain')
        for feat, score in importance_dict.items():
            if feat in xgb_importances:
                xgb_importances[feat] += score
            else:
                xgb_importances[feat] = score
    
    # Calcular média das importâncias
    for feat in xgb_importances:
        xgb_importances[feat] /= n_folds
    
    # Criar dataframe de importância
    xgb_features = []
    xgb_scores = []
    
    for feat, score in xgb_importances.items():
        xgb_features.append(feat)
        xgb_scores.append(score)
    
    xgb_importance = pd.DataFrame({
        'Feature': xgb_features,
        'Importance_XGB': xgb_scores
    }).sort_values(by='Importance_XGB', ascending=False)
    
    # Adicionar features ausentes
    missing_features = set(numeric_cols) - set(xgb_importance['Feature'])
    for feat in missing_features:
        xgb_importance = pd.concat([xgb_importance, 
                                   pd.DataFrame({'Feature': [feat], 'Importance_XGB': [0]})],
                                  ignore_index=True)
    
    print("\nMétricas médias da validação cruzada (XGBoost):")
    for key, values in xgb_metrics.items():
        print(f"  {key.upper()}: {np.mean(values):.4f} (±{np.std(values):.4f})")
    
    print("\nTop 15 features (XGBoost):")
    print(xgb_importance.head(15))
    
except Exception as e:
    print(f"Erro ao executar XGBoost: {e}")
    print("Criando dataframe de importância vazio para XGBoost")
    # Criar dataframe vazio em caso de erro
    xgb_importance = pd.DataFrame({
        'Feature': numeric_cols,
        'Importance_XGB': [0] * len(numeric_cols)
    })

# 9 - Combinando importâncias de múltiplos modelos
print("\nCombinando resultados de diferentes métodos...")

# Normalizar importâncias para comparabilidade
for df_imp, col in [(rf_importance, 'Importance_RF'), 
                    (lgb_importance, 'Importance_LGB'), 
                    (xgb_importance, 'Importance_XGB')]:
    if df_imp[col].sum() > 0:  # Evitar divisão por zero
        df_imp[col] = df_imp[col] / df_imp[col].sum() * 100

# Mesclar resultados
try:
    combined = pd.merge(rf_importance, lgb_importance, on='Feature', how='outer')
    combined = pd.merge(combined, xgb_importance, on='Feature', how='outer')
    combined = combined.fillna(0)
except Exception as e:
    print(f"Erro ao combinar resultados: {e}")
    # Alternativa: usar apenas o modelo que funcionou
    if rf_importance['Importance_RF'].sum() > 0:
        combined = rf_importance.copy()
        if 'Importance_LGB' not in combined.columns:
            combined['Importance_LGB'] = 0
        if 'Importance_XGB' not in combined.columns:
            combined['Importance_XGB'] = 0
    elif lgb_importance['Importance_LGB'].sum() > 0:
        combined = lgb_importance.copy()
        if 'Importance_RF' not in combined.columns:
            combined['Importance_RF'] = 0
        if 'Importance_XGB' not in combined.columns:
            combined['Importance_XGB'] = 0
    else:
        combined = xgb_importance.copy()
        if 'Importance_RF' not in combined.columns:
            combined['Importance_RF'] = 0
        if 'Importance_LGB' not in combined.columns:
            combined['Importance_LGB'] = 0

# Calcular média e desvio padrão das importâncias
combined['Mean_Importance'] = combined[['Importance_RF', 'Importance_LGB', 'Importance_XGB']].mean(axis=1)
combined['Std_Importance'] = combined[['Importance_RF', 'Importance_LGB', 'Importance_XGB']].std(axis=1)
combined['CV'] = combined['Std_Importance'] / combined['Mean_Importance'].replace(0, 1e-10)

# Ordenar por importância média
final_importance = combined.sort_values(by='Mean_Importance', ascending=False)

print("\nImportância combinada (top 20 features):")
print(final_importance[['Feature', 'Mean_Importance', 'Std_Importance', 'CV']].head(20))

# Salvar importância das features
final_importance.to_csv(os.path.join(analysis_dir, 'feature_importance_combined.csv'), index=False)
print(f"\nImportância das features salva em {os.path.join(analysis_dir, 'feature_importance_combined.csv')}")

# 10 - Análise de Robustez entre Lançamentos
if launch_col and df[launch_col].nunique() >= 2:
    print("\n--- Análise de Robustez entre Lançamentos ---")
    
    # Verificar se as colunas de lançamento foram renomeadas
    if launch_col in rename_dict:
        launch_col = rename_dict[launch_col]
    
    launch_imp_results = {}
    
    # Selecionar os principais lançamentos para análise (top 6)
    main_launches = df[launch_col].value_counts().nlargest(6).index.tolist()
    
    for launch in main_launches:
        launch_mask = df[launch_col] == launch
        if launch_mask.sum() < 100:  # Pular lançamentos muito pequenos
            print(f"Pulando lançamento {launch} (menos de 100 amostras)")
            continue
            
        print(f"\nAnalisando lançamento: {launch} ({launch_mask.sum()} amostras)")
        
        # Separar dados deste lançamento
        X_launch = X[launch_mask]
        y_launch = y[launch_mask]
        
        # Verificar se há amostras positivas suficientes
        n_pos = y_launch.sum()
        if n_pos < 5:
            print(f"  Pulando: apenas {n_pos} amostras positivas")
            continue
            
        # Criar split proporcional a quantidade de dados
        test_size = min(0.2, 100/len(y_launch))
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_launch, y_launch, 
                                                  test_size=test_size, 
                                                  random_state=42,
                                                  stratify=y_launch)
        
        # Tentar RandomForest (mais robusto a erros)
        try:
            # Calcular class_weight
            n_samples = len(y_tr)
            n_pos = y_tr.sum()
            n_neg = n_samples - n_pos
            weight_pos = (n_samples / (2 * n_pos)) if n_pos > 0 else 1.0
            weight_neg = (n_samples / (2 * n_neg)) if n_neg > 0 else 1.0
            
            # Treinar modelo apenas para este lançamento
            rf_model_launch = RandomForestClassifier(
                n_estimators=50, 
                class_weight={0: weight_neg, 1: weight_pos},
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            rf_model_launch.fit(X_tr, y_tr)
            
            # Obter importância
            launch_imp = pd.DataFrame({
                'Feature': numeric_cols,
                f'Imp_{launch}': rf_model_launch.feature_importances_
            })
            
            # Normalizar importância
            if launch_imp[f'Imp_{launch}'].sum() > 0:
                launch_imp[f'Imp_{launch}'] = launch_imp[f'Imp_{launch}'] / launch_imp[f'Imp_{launch}'].sum() * 100
            
            # Guardar resultados
            launch_imp_results[launch] = launch_imp
        except Exception as e:
            print(f"  Erro ao analisar lançamento {launch}: {e}")
    
    # Combinar resultados de diferentes lançamentos
    if launch_imp_results:
        combined_launch_imp = launch_imp_results[list(launch_imp_results.keys())[0]].copy()
        
        for launch, imp_df in list(launch_imp_results.items())[1:]:
            combined_launch_imp = pd.merge(combined_launch_imp, imp_df, on='Feature', how='outer')
        
        combined_launch_imp = combined_launch_imp.fillna(0)
        
        # Calcular média e desvio padrão entre lançamentos
        imp_cols = [col for col in combined_launch_imp.columns if col.startswith('Imp_')]
        combined_launch_imp['Mean_Launch_Imp'] = combined_launch_imp[imp_cols].mean(axis=1)
        combined_launch_imp['Std_Launch_Imp'] = combined_launch_imp[imp_cols].std(axis=1)
        combined_launch_imp['CV_Launch'] = combined_launch_imp['Std_Launch_Imp'] / combined_launch_imp['Mean_Launch_Imp'].replace(0, 1e-10)
        
        # Ordenar por importância média
        launch_importance = combined_launch_imp.sort_values(by='Mean_Launch_Imp', ascending=False)
        
        print("\nImportância média entre lançamentos (top 15 features):")
        print(launch_importance[['Feature', 'Mean_Launch_Imp', 'Std_Launch_Imp', 'CV_Launch']].head(15))
        
        # Identificar features com alta variabilidade entre lançamentos
        unstable_features = launch_importance[
            (launch_importance['CV_Launch'] > 1.2) & 
            (launch_importance['Mean_Launch_Imp'] > 0.5)
        ].sort_values(by='CV_Launch', ascending=False)
        
        print("\nFeatures com alta variabilidade entre lançamentos:")
        print(unstable_features[['Feature', 'Mean_Launch_Imp', 'CV_Launch']].head(10))
        
        # Merge com importância geral para comparação
        launch_vs_global = pd.merge(
            launch_importance[['Feature', 'Mean_Launch_Imp', 'CV_Launch']],
            final_importance[['Feature', 'Mean_Importance']],
            on='Feature', how='inner'
        )
        
        # Identificar features consistentemente importantes
        consistent_features = launch_vs_global[
            (launch_vs_global['Mean_Launch_Imp'] > launch_vs_global['Mean_Launch_Imp'].median()) &
            (launch_vs_global['Mean_Importance'] > launch_vs_global['Mean_Importance'].median()) &
            (launch_vs_global['CV_Launch'] < 1.0)
        ].sort_values(by='Mean_Importance', ascending=False)
        
        print("\nFeatures consistentemente importantes entre lançamentos:")
        print(consistent_features[['Feature', 'Mean_Importance', 'Mean_Launch_Imp', 'CV_Launch']].head(15))
        
        # Salvar análise de robustez
        launch_vs_global.to_csv(os.path.join(analysis_dir, 'feature_robustness_analysis.csv'), index=False)
        print(f"\nAnálise de robustez entre lançamentos salva em {os.path.join(analysis_dir, 'feature_robustness_analysis.csv')}")
else:
    print("\nAnálise de robustez entre lançamentos não realizada (coluna de lançamento não identificada ou insuficiente)")

# 11 - Identificar features potencialmente irrelevantes
print("\nIdentificando features potencialmente irrelevantes...")

# Critérios para considerar uma feature como potencialmente irrelevante:
# 1. Baixa importância média (< 0.1% da importância total média)
threshold_importance = 0.1
irrelevant_by_importance = final_importance[final_importance['Mean_Importance'] < threshold_importance]

# 2. Alta variabilidade entre modelos (coef. variação > 1.5)
threshold_cv = 1.5
irrelevant_by_variance = final_importance[(final_importance['CV'] > threshold_cv) & 
                                         (final_importance['Mean_Importance'] < final_importance['Mean_Importance'].median())]

# 3. Features altamente correlacionadas com outras mais importantes
irrelevant_by_correlation = []
for pair in high_corr_pairs:
    f1, f2 = pair['feature1'], pair['feature2']
    f1_imp = final_importance[final_importance['Feature'] == f1]['Mean_Importance'].values[0] if f1 in final_importance['Feature'].values else 0
    f2_imp = final_importance[final_importance['Feature'] == f2]['Mean_Importance'].values[0] if f2 in final_importance['Feature'].values else 0
    
    # A feature menos importante é considerada irrelevante
    if f1_imp < f2_imp:
        irrelevant_by_correlation.append({
            'Feature': f1, 
            'Correlation_With': f2, 
            'Correlation': pair['correlation'],
            'Mean_Importance': f1_imp,
            'Better_Feature_Importance': f2_imp
        })
    else:
        irrelevant_by_correlation.append({
            'Feature': f2, 
            'Correlation_With': f1, 
            'Correlation': pair['correlation'],
            'Mean_Importance': f2_imp,
            'Better_Feature_Importance': f1_imp
        })

# Converter para DataFrame
irrelevant_by_correlation_df = pd.DataFrame(irrelevant_by_correlation)

# Combinando critérios
potentially_irrelevant = pd.concat([
    irrelevant_by_importance[['Feature', 'Mean_Importance', 'CV']],
    irrelevant_by_variance[['Feature', 'Mean_Importance', 'CV']]
]).drop_duplicates().sort_values(by='Mean_Importance')

print(f"\nFeatures potencialmente irrelevantes ({len(potentially_irrelevant)}):")
print(potentially_irrelevant[['Feature', 'Mean_Importance', 'CV']].head(20))

if len(potentially_irrelevant) > 20:
    print(f"... e mais {len(potentially_irrelevant) - 20} features.")

# 12 - Análise específica de features textuais
if text_derived_cols:
    print("\n--- Análise Específica de Features Textuais ---")
    
    # Extrair apenas features textuais do dataframe de importância
    text_importance = final_importance[final_importance['Feature'].isin(text_derived_cols)].copy()
    
    # Agrupar por tipo de feature textual
    text_feature_types = {
        'Comprimento Texto': [col for col in text_derived_cols if '_length' in col or '_word_count' in col],
        'Sentimento': [col for col in text_derived_cols if '_sentiment' in col],
        'Motivação': [col for col in text_derived_cols if '_motiv_' in col],
        'Características': [col for col in text_derived_cols if '_has_' in col],
        'TF-IDF': [col for col in text_derived_cols if '_tfidf_' in col]
    }
    
    print("\nImportância das features textuais por categoria:")
    for category, cols in text_feature_types.items():
        if cols:
            category_importance = text_importance[text_importance['Feature'].isin(cols)]
            avg_importance = category_importance['Mean_Importance'].mean() if not category_importance.empty else 0
            print(f"{category}: {len(cols)} features, importância média: {avg_importance:.2f}")
            
            # Top 3 features nesta categoria
            top_features = category_importance.head(3)
            if not top_features.empty:
                print("  Top features nesta categoria:")
                for i, row in top_features.iterrows():
                    print(f"    - {row['Feature']}: {row['Mean_Importance']:.2f}")
    
    # Top 10 features textuais gerais
    print("\nTop 10 features textuais:")
    print(text_importance[['Feature', 'Mean_Importance']].head(10))
    
    # Proporção de importância das features textuais
    text_importance_sum = text_importance['Mean_Importance'].sum()
    total_importance_sum = final_importance['Mean_Importance'].sum()
    text_proportion = (text_importance_sum / total_importance_sum) * 100 if total_importance_sum > 0 else 0
    
    print(f"\nContribuição total das features textuais: {text_proportion:.2f}% da importância total")
    
    # Salvar análise de features textuais
    text_importance.to_csv(os.path.join(analysis_dir, 'text_features_importance.csv'), index=False)
    print(f"Análise de features textuais salva em {os.path.join(analysis_dir, 'text_features_importance.csv')}")

# 13 - Recomendações finais e criação da lista de features não recomendadas com justificativas
print("\n--- Preparando Recomendações Finais e Documentação ---")

# Definir um limiar de importância
importance_threshold = final_importance['Mean_Importance'].sum() * 0.001  # 0.1% da importância total

# Filtrar features relevantes e não redundantes
relevant_features = final_importance[final_importance['Mean_Importance'] > importance_threshold]['Feature'].tolist()

# Remover uma feature de cada par altamente correlacionado (manter a mais importante)
features_to_remove_corr = []
if high_corr_pairs:
    for pair in high_corr_pairs:
        f1, f2 = pair['feature1'], pair['feature2']
        if f1 in relevant_features and f2 in relevant_features:
            f1_imp = final_importance[final_importance['Feature'] == f1]['Mean_Importance'].values[0] 
            f2_imp = final_importance[final_importance['Feature'] == f2]['Mean_Importance'].values[0]
            # Remover a feature menos importante
            if f1_imp < f2_imp and f1 in relevant_features:
                relevant_features.remove(f1)
                features_to_remove_corr.append((f1, f2, pair['correlation'], f1_imp, f2_imp))
            elif f2 in relevant_features:
                relevant_features.remove(f2)
                features_to_remove_corr.append((f2, f1, pair['correlation'], f2_imp, f1_imp))

# Criar conjunto de features recomendadas
# Converter para dicionário para facilitar a conversão de volta aos nomes originais
reverse_rename_dict = {v: k for k, v in rename_dict.items()}

# Recuperar os nomes originais das features relevantes
original_relevant_features = [reverse_rename_dict.get(feature, feature) for feature in relevant_features]

# Salvar lista de features recomendadas
with open(os.path.join(analysis_dir, 'recommended_features.txt'), 'w') as f:
    for feature in original_relevant_features:
        f.write(f"{feature}\n")

print(f"\nLista de {len(original_relevant_features)} features recomendadas salva em {os.path.join(analysis_dir, 'recommended_features.txt')}")

# Criar lista e documentação de features não recomendadas
unrecommended_features = set(numeric_cols) - set(relevant_features)
unrecommended_features_original = [reverse_rename_dict.get(feature, feature) for feature in unrecommended_features]

# Preparar justificativas para cada feature não recomendada
unrecommended_with_reasons = []

for feature in unrecommended_features:
    original_feature = reverse_rename_dict.get(feature, feature)
    reasons = []
    
    # Verificar se tem baixa importância
    if feature in irrelevant_by_importance['Feature'].values:
        imp = final_importance[final_importance['Feature'] == feature]['Mean_Importance'].values[0]
        reasons.append(f"Baixa importância preditiva ({imp:.4f})")
    
    # Verificar se tem alta variabilidade entre modelos
    if feature in irrelevant_by_variance['Feature'].values:
        cv = final_importance[final_importance['Feature'] == feature]['CV'].values[0]
        reasons.append(f"Alta variabilidade entre modelos (CV={cv:.2f})")
    
    # Verificar se é redundante com outra feature
    redundant_with = None
    for f, better_f, corr, imp, better_imp in features_to_remove_corr:
        if f == feature:
            original_better_f = reverse_rename_dict.get(better_f, better_f)
            reasons.append(f"Altamente correlacionada (r={corr:.2f}) com {original_better_f} que tem maior importância ({better_imp:.4f} vs {imp:.4f})")
            redundant_with = original_better_f
            break
    
    # Se não encontrou razão específica
    if not reasons:
        reasons.append("Baixa contribuição geral para o modelo")
    
    unrecommended_with_reasons.append({
        'Feature': original_feature,
        'Reasons': "; ".join(reasons),
        'Redundant_With': redundant_with,
        'Importance': final_importance[final_importance['Feature'] == feature]['Mean_Importance'].values[0] if feature in final_importance['Feature'].values else 0
    })

# Converter para DataFrame e salvar
unrecommended_df = pd.DataFrame(unrecommended_with_reasons)
unrecommended_df = unrecommended_df.sort_values('Importance', ascending=False)
unrecommended_df.to_csv(os.path.join(analysis_dir, 'unrecommended_features.csv'), index=False)

# Criar arquivo de texto com explicações detalhadas
with open(os.path.join(analysis_dir, 'unrecommended_features_explanation.txt'), 'w') as f:
    f.write("# Features Não Recomendadas e Justificativas\n\n")
    f.write(f"Total de features analisadas: {len(numeric_cols)}\n")
    f.write(f"Features recomendadas: {len(relevant_features)}\n")
    f.write(f"Features não recomendadas: {len(unrecommended_features)}\n\n")
    
    f.write("## Razões para remoção:\n\n")
    f.write("1. **Baixa importância preditiva**: Features com importância menor que 0.1% da importância total.\n")
    f.write("2. **Alta variabilidade entre modelos**: Features cujo coeficiente de variação entre diferentes modelos é maior que 1.5.\n")
    f.write("3. **Redundância**: Features altamente correlacionadas (r > 0.8) com outras de maior importância.\n\n")
    
    f.write("## Lista de features não recomendadas:\n\n")
    
    for i, row in unrecommended_df.iterrows():
        f.write(f"### {i+1}. {row['Feature']}\n")
        f.write(f"   - **Razões**: {row['Reasons']}\n")
        if pd.notna(row['Redundant_With']):
            f.write(f"   - **Redundante com**: {row['Redundant_With']}\n")
        f.write(f"   - **Importância**: {row['Importance']:.6f}\n\n")

print(f"Documentação detalhada de features não recomendadas salva em {os.path.join(analysis_dir, 'unrecommended_features_explanation.txt')}")

# 14 - Aplicar a seleção de features aos datasets de entrada
print("\n--- Aplicando seleção de features a todos os datasets ---")

# Verificar se existem features que aparecem em alguns datasets mas não em outros
print("Verificando consistência das features entre os datasets...")
train_columns = set(train_df.columns)
val_columns = set(val_df.columns)
test_columns = set(test_df.columns)

# Features exclusivas de cada dataset
train_only = train_columns - (val_columns.union(test_columns))
val_only = val_columns - (train_columns.union(test_columns))
test_only = test_columns - (train_columns.union(val_columns))

# Encontrar features comuns a todos os datasets
common_features = train_columns.intersection(val_columns).intersection(test_columns)

# Features selecionadas em comum
selected_common_features = [f for f in original_relevant_features if f in common_features]

if train_only:
    print(f"Features encontradas apenas no dataset de treino ({len(train_only)}):")
    print(f"  {sorted(list(train_only))[:5]}{'...' if len(train_only) > 5 else ''}")

if val_only:
    print(f"Features encontradas apenas no dataset de validação ({len(val_only)}):")
    print(f"  {sorted(list(val_only))[:5]}{'...' if len(val_only) > 5 else ''}")

if test_only:
    print(f"Features encontradas apenas no dataset de teste ({len(test_only)}):")
    print(f"  {sorted(list(test_only))[:5]}{'...' if len(test_only) > 5 else ''}")

# Resolver o problema garantindo consistência entre datasets
print(f"\nGarantindo consistência entre datasets...")
print(f"Features selecionadas originalmente: {len(original_relevant_features)}")
print(f"Features comuns a todos os datasets: {len(common_features)}")
print(f"Features selecionadas comuns a todos os datasets: {len(selected_common_features)}")

# Função para aplicar seleção de features a um DataFrame
def apply_features_selection(df, selected_features, target_col):
    """Aplica a seleção de features a um DataFrame"""
    # Selecionamos apenas as features que existem no DataFrame
    available_features = [col for col in selected_features if col in df.columns]
    
    # Selecionar colunas
    if target_col in df.columns:
        selected_columns = available_features + [target_col]
    else:
        selected_columns = available_features
    
    # Criar novo DataFrame
    return df[selected_columns]

# Aplicar seleção a cada dataset usando APENAS as features que existem em todos os datasets
print(f"Aplicando seleção a {len(selected_common_features)} features comuns a todos os datasets...")

# Processar dataset de treino
train_selected = apply_features_selection(train_df, selected_common_features, target_col)
print(f"Dataset de treino: {train_df.shape[1]} colunas -> {train_selected.shape[1]} colunas selecionadas")

# Processar dataset de validação
val_selected = apply_features_selection(val_df, selected_common_features, target_col)
print(f"Dataset de validação: {val_df.shape[1]} colunas -> {val_selected.shape[1]} colunas selecionadas")

# Processar dataset de teste
test_selected = apply_features_selection(test_df, selected_common_features, target_col)
print(f"Dataset de teste: {test_df.shape[1]} colunas -> {test_selected.shape[1]} colunas selecionadas")

# Verificar se todos os datasets agora têm o mesmo número de colunas
if train_selected.shape[1] == val_selected.shape[1] == test_selected.shape[1]:
    print(f"Sucesso! Todos os datasets agora têm exatamente {train_selected.shape[1]} colunas.")
else:
    print("AVISO: Os datasets ainda têm números diferentes de colunas:")
    print(f"  - Treino: {train_selected.shape[1]} colunas")
    print(f"  - Validação: {val_selected.shape[1]} colunas")
    print(f"  - Teste: {test_selected.shape[1]} colunas")

# Antes de salvar os datasets, garantir que todas as colunas são numéricas
def ensure_numeric_columns(df):
    """Garante que todas as colunas são numéricas, convertendo strings para valores numéricos."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Verificar valores únicos
            unique_vals = df[col].unique()
            print(f"Convertendo coluna {col} de {df[col].dtype} para numérica. Valores únicos: {unique_vals[:5]}")
            
            # Se for categórica, fazer codificação simples
            value_map = {val: idx for idx, val in enumerate(unique_vals) if pd.notna(val)}
            df[col] = df[col].map(value_map).fillna(-1)
    
    return df

# Aplicar aos datasets finais
train_selected = ensure_numeric_columns(train_selected)
val_selected = ensure_numeric_columns(val_selected)
test_selected = ensure_numeric_columns(test_selected)

# Verificar se ainda há tipos não numéricos
non_numeric_cols = {col: train_selected[col].dtype for col in train_selected.columns 
                  if not pd.api.types.is_numeric_dtype(train_selected[col].dtype)}

if non_numeric_cols:
    print(f"AVISO: Ainda existem {len(non_numeric_cols)} colunas não numéricas:")
    for col, dtype in non_numeric_cols.items():
        print(f"  - {col}: {dtype}")
    
    # Remover colunas não numéricas como último recurso
    cols_to_drop = list(non_numeric_cols.keys())
    train_selected = train_selected.drop(columns=cols_to_drop)
    val_selected = val_selected.drop(columns=cols_to_drop)
    test_selected = test_selected.drop(columns=cols_to_drop)
    print(f"Removidas {len(cols_to_drop)} colunas não numéricas.")
    
# 15 - Salvar os datasets processados
print("\n--- Salvando datasets com features selecionadas ---")

# Salvar datasets
train_selected.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_selected.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
test_selected.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print(f"Datasets com features selecionadas salvos em: {output_dir}")

# 16 - Resumo das análises realizadas
print("\n=== RESUMO DAS ANÁLISES ===")
print(f"Total de features analisadas: {len(numeric_cols)}")
print(f"Features textuais processadas: {len(text_derived_cols)}")
print(f"Pares de features altamente correlacionadas: {len(high_corr_pairs)}")
print(f"Features potencialmente irrelevantes: {len(potentially_irrelevant)}")
print(f"Features recomendadas após filtragem: {len(relevant_features)}")
print(f"Features comuns a todos os datasets: {len(common_features)}")
print(f"Features selecionadas e comuns a todos os datasets: {len(selected_common_features)}")

# Exibir top 10 features mais importantes
print("\nTop 10 features mais importantes para previsão de conversão:")
for i, row in final_importance.head(10).iterrows():
    print(f"{i+1}. {row['Feature']}: {row['Mean_Importance']:.2f}")

# Categorizar features por tipo
feature_categories = {
    'Features Textuais': text_derived_cols,
    'UTM/Campaign': [col for col in numeric_cols if any(term in col.lower() for term in ['utm', 'campaign', 'camp'])],
    'Dados Geográficos': [col for col in numeric_cols if any(term in col.lower() for term in ['country', 'pais'])],
    'Tempo/Data': [col for col in numeric_cols if any(term in col.lower() for term in ['time', 'hour', 'day', 'month', 'year'])],
    'Demografia': [col for col in numeric_cols if any(term in col.lower() for term in ['age', 'gender', 'edad'])],
    'Profissão': [col for col in numeric_cols if any(term in col.lower() for term in ['profes', 'profession', 'work'])]
}

# Calcular importância por categoria
categories_importance = {}
for category, cols in feature_categories.items():
    category_features = [col for col in cols if col in final_importance['Feature'].values]
    if category_features:
        category_importance = final_importance[final_importance['Feature'].isin(category_features)]['Mean_Importance'].sum()
        categories_importance[category] = category_importance

# Ordenar categorias por importância
sorted_categories = sorted(categories_importance.items(), key=lambda x: x[1], reverse=True)

print("\nImportância por categoria de features:")
for category, importance in sorted_categories:
    category_pct = (importance / final_importance['Mean_Importance'].sum()) * 100
    print(f"{category}: {importance:.2f} ({category_pct:.1f}%)")

print("\nAnálise de importância de features concluída com sucesso!")
print(f"Resultados de análise salvos em: {analysis_dir}")
print(f"Datasets com features selecionadas salvos em: {output_dir}")