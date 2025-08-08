"""
Feature importance analysis module.

This module provides functions to analyze feature importance using multiple models
and combine their results. It also handles multicollinearity analysis and provides
recommendations for feature selection.
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

# Suppress warnings
warnings.filterwarnings('ignore')

def create_output_directory(output_dir):
    """Create directory for saving results if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Pasta '{output_dir}' criada para salvar resultados")
    else:
        print(f"Pasta '{output_dir}' já existe")
    return output_dir

def load_dataset(train_path, alt_paths=None):
    """Load dataset from the specified path or alternative paths if primary fails."""
    if alt_paths is None:
        alt_paths = []
    
    try:
        df = pd.read_csv(train_path)
        print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return df
    except Exception as e:
        print(f"Arquivo não encontrado, tentando alternativas... Erro: {e}")
        
        for path in alt_paths:
            try:
                df = pd.read_csv(path)
                print(f"Dataset alternativo carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
                return df
            except:
                print(f"Arquivo alternativo {path} não encontrado.")
        
        raise FileNotFoundError("Nenhum dos caminhos de arquivo fornecidos foi encontrado.")

def identify_launch_column(df, specified_col='lançamento'):
    """Identify the launch column in the dataframe."""
    if specified_col in df.columns:
        print(f"Coluna de lançamento encontrada: '{specified_col}'")
        n_launches = df[specified_col].nunique()
        print(f"Número de lançamentos: {n_launches}")
        print(f"Lançamentos identificados: {sorted(df[specified_col].unique())}")
        print(f"Distribuição de lançamentos:\n{df[specified_col].value_counts(normalize=True)*100}")
        return specified_col
    else:
        print(f"Coluna '{specified_col}' não encontrada. Verificando alternativas...")
        # Procurar colunas alternativas
        alt_launch_cols = [col for col in df.columns if 'lanc' in col.lower() or 'launch' in col.lower()]
        if alt_launch_cols:
            launch_col = alt_launch_cols[0]
            print(f"Usando coluna alternativa: '{launch_col}'")
            print(f"Número de lançamentos: {df[launch_col].nunique()}")
            print(f"Distribuição de lançamentos:\n{df[launch_col].value_counts(normalize=True)*100}")
            return launch_col
        else:
            print("Nenhuma coluna de lançamento identificada.")
            return None

def identify_target_column(df, default_target='target'):
    """Identify the target column in the dataframe."""
    if default_target in df.columns:
        return default_target
    
    print("Coluna 'target' não encontrada. Verificando alternativas...")
    target_cols = [col for col in df.columns if col.lower() in ['target', 'comprou', 'converted', 'conversion']]
    if target_cols:
        target_col = target_cols[0]
        print(f"Usando '{target_col}' como target.")
        return target_col
    else:
        raise ValueError("Não foi possível encontrar uma coluna target.")

def select_numeric_features(df, target_col):
    """Select numeric columns for analysis and filter out problematic ones."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Remove columns with high missing values
    missing_pct = df[numeric_cols].isna().mean()
    high_missing_cols = missing_pct[missing_pct > 0.9].index.tolist()
    if high_missing_cols:
        print(f"Removendo {len(high_missing_cols)} colunas com mais de 90% de valores ausentes")
        numeric_cols = [col for col in numeric_cols if col not in high_missing_cols]
    
    # Remove columns with zero variance
    try:
        selector = VarianceThreshold(threshold=0)
        selector.fit(df[numeric_cols].fillna(0))
        zero_var_cols = [numeric_cols[i] for i, var in enumerate(selector.variances_) if var == 0]
        if zero_var_cols:
            print(f"Removendo {len(zero_var_cols)} colunas com variância zero")
            numeric_cols = [col for col in numeric_cols if col not in zero_var_cols]
    except Exception as e:
        print(f"Erro ao verificar variância: {e}")
    
    return numeric_cols

def identify_text_derived_columns(numeric_cols):
    """Identify columns derived from text features."""
    text_indicators = ['_tfidf_', '_sentiment', '_word_count', '_length', '_motiv_', '_has_question']
    text_derived_cols = [col for col in numeric_cols if any(indicator in col for indicator in text_indicators)]
    
    print(f"Features derivadas de texto identificadas: {len(text_derived_cols)}")
    if text_derived_cols:
        print("Exemplos de features textuais:")
        for col in text_derived_cols[:5]:  # Mostrar alguns exemplos
            print(f"  - {col}")
        if len(text_derived_cols) > 5:
            print(f"  - ... e mais {len(text_derived_cols) - 5} features textuais")
    
    return text_derived_cols

def analyze_multicollinearity(X, threshold=0.8):
    """Identify pairs of features with high correlation."""
    print("--- Análise de Multicolinearidade ---")
    corr_matrix = X.corr()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    # Ordenar por correlação absoluta
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    print(f"Encontrados {len(high_corr_pairs)} pares de features com correlação > {threshold}:")
    for i, pair in enumerate(high_corr_pairs[:10]):  # Mostrar os 10 primeiros
        print(f"{i+1}. {pair['feature1']} & {pair['feature2']}: {pair['correlation']:.4f}")
    
    if len(high_corr_pairs) > 10:
        print(f"... e mais {len(high_corr_pairs) - 10} pares.")
    
    return high_corr_pairs

def compare_country_encodings(X, y):
    """Compare different country encoding methods."""
    print("--- Análise de Redundância: country_freq vs country_encoded ---")
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
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'corr_target_freq': corr_target_freq,
            'corr_target_encoded': corr_target_encoded,
            'recommendation': recommendation
        }
    else:
        print("Colunas country_freq e/ou country_encoded não encontradas.")
        return None

def evaluate_model(model, X, y, feature_names=None):
    """Evaluate model using metrics suitable for imbalanced data."""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:  # For models like XGBoost with DMatrix
        y_proba = model.predict(X)
    
    # Calculate metrics
    # AUC - evaluates ranking regardless of threshold
    auc = roc_auc_score(y, y_proba)
    
    # Average Precision - weighted average of precisions at different thresholds
    ap = average_precision_score(y, y_proba)
    
    # Find best F1-score by adjusting threshold
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

def analyze_rf_importance(X, y, numeric_cols, n_folds=5):
    """Analyze feature importance using RandomForest with cross-validation."""
    print("Analisando com RandomForest e validação cruzada para dados desbalanceados...")
    try:
        # Use stratified cross-validation to handle imbalance
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        rf_importances = np.zeros(len(numeric_cols))
        rf_metrics = {'auc': [], 'ap': [], 'f1': []}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold+1}/{n_folds}")
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Calculate class_weight
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
            
            # Evaluate model
            metrics = evaluate_model(rf_model, X_fold_val, y_fold_val, numeric_cols)
            for key in rf_metrics:
                rf_metrics[key].append(metrics[key])
            
            # Accumulate importances
            rf_importances += rf_model.feature_importances_
        
        # Calculate average importances
        rf_importances /= n_folds
        
        # Create importance dataframe
        rf_importance = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance_RF': rf_importances
        }).sort_values(by='Importance_RF', ascending=False)
        
        print("\nMétricas médias da validação cruzada (RandomForest):")
        for key, values in rf_metrics.items():
            print(f"  {key.upper()}: {np.mean(values):.4f} (±{np.std(values):.4f})")
        
        print("\nTop 15 features (RandomForest):")
        print(rf_importance.head(15))
        
        return rf_importance, rf_metrics
    except Exception as e:
        print(f"Erro ao executar RandomForest: {e}")
        # Create empty dataframe in case of error
        rf_importance = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance_RF': [0] * len(numeric_cols)
        })
        return rf_importance, {'auc': [], 'ap': [], 'f1': []}

def analyze_lgb_importance(X, y, numeric_cols, n_folds=5):
    """Analyze feature importance using LightGBM with cross-validation."""
    print("Analisando com LightGBM...")
    try:
        import lightgbm as lgb
        
        # Cross-validation with LightGBM for imbalanced data
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        lgb_importances = np.zeros(len(numeric_cols))
        lgb_metrics = {'auc': [], 'ap': [], 'f1': []}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold+1}/{n_folds}")
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Calculate scale_pos_weight
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
            
            # Train model
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0)]
            
            lgb_model = lgb.train(
                params, 
                train_data, 
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=callbacks
            )
            
            # Evaluate model
            metrics = evaluate_model(lgb_model, X_fold_val, y_fold_val, numeric_cols)
            for key in lgb_metrics:
                lgb_metrics[key].append(metrics[key])
            
            # Accumulate importances
            fold_importance = lgb_model.feature_importance(importance_type='gain')
            lgb_importances += fold_importance
        
        # Calculate average importances
        lgb_importances /= n_folds
        
        # Create importance dataframe
        lgb_importance = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance_LGB': lgb_importances
        }).sort_values(by='Importance_LGB', ascending=False)
        
        print("\nMétricas médias da validação cruzada (LightGBM):")
        for key, values in lgb_metrics.items():
            print(f"  {key.upper()}: {np.mean(values):.4f} (±{np.std(values):.4f})")

        print("\nTop 15 features (LightGBM):")
        print(lgb_importance.head(15))
        
        return lgb_importance, lgb_metrics
    except Exception as e:
        print(f"Erro ao executar LightGBM: {e}")
        print("Criando dataframe de importância vazio para LightGBM")
        # Create empty dataframe in case of error
        lgb_importance = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance_LGB': [0] * len(numeric_cols)
        })
        return lgb_importance, {'auc': [], 'ap': [], 'f1': []}

def analyze_xgb_importance(X, y, numeric_cols, n_folds=5):
    """Analyze feature importance using XGBoost with cross-validation."""
    print("Analisando com XGBoost...")
    try:
        import xgboost as xgb
        
        # Cross-validation with XGBoost for imbalanced data
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        xgb_importances = {}  # Dictionary to accumulate importances
        xgb_metrics = {'auc': [], 'ap': [], 'f1': []}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold+1}/{n_folds}")
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Calculate scale_pos_weight
            pos_scale = (y_fold_train == 0).sum() / max(1, (y_fold_train == 1).sum())
            
            # Prepare data
            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train, feature_names=numeric_cols)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val, feature_names=numeric_cols)
            
            # Configuration for imbalanced data
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'scale_pos_weight': pos_scale,
                'learning_rate': 0.05,
                'seed': 42 + fold,
                'tree_method': 'hist'
            }
            
            # Train model
            xgb_model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Evaluate model
            dval_pred = xgb.DMatrix(X_fold_val, feature_names=numeric_cols)
            y_pred = xgb_model.predict(dval_pred)
            metrics = {
                'auc': roc_auc_score(y_fold_val, y_pred),
                'ap': average_precision_score(y_fold_val, y_pred)
            }
            
            # Find best F1-score by adjusting threshold
            precisions, recalls, thresholds = precision_recall_curve(y_fold_val, y_pred)
            f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = 0 if len(thresholds) == 0 else thresholds[min(best_threshold_idx, len(thresholds)-1)]
            best_f1 = f1_scores[best_threshold_idx]
            metrics['f1'] = best_f1
            
            for key in xgb_metrics:
                xgb_metrics[key].append(metrics[key])
            
            # Accumulate importances
            importance_dict = xgb_model.get_score(importance_type='gain')
            for feat, score in importance_dict.items():
                if feat in xgb_importances:
                    xgb_importances[feat] += score
                else:
                    xgb_importances[feat] = score
        
        # Calculate average importances
        for feat in xgb_importances:
            xgb_importances[feat] /= n_folds
        
        # Create importance dataframe
        xgb_features = []
        xgb_scores = []
        
        for feat, score in xgb_importances.items():
            xgb_features.append(feat)
            xgb_scores.append(score)
        
        xgb_importance = pd.DataFrame({
            'Feature': xgb_features,
            'Importance_XGB': xgb_scores
        }).sort_values(by='Importance_XGB', ascending=False)
        
        # Add missing features
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
        
        return xgb_importance, xgb_metrics
        
    except Exception as e:
        print(f"Erro ao executar XGBoost: {e}")
        print("Criando dataframe de importância vazio para XGBoost")
        # Create empty dataframe in case of error
        xgb_importance = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance_XGB': [0] * len(numeric_cols)
        })
        return xgb_importance, {'auc': [], 'ap': [], 'f1': []}

def combine_importance_results(rf_importance, lgb_importance, xgb_importance):
    """Combine and normalize importance results from different models."""
    print("Combinando resultados de diferentes métodos...")

    # Normalize importances for comparability
    for df_imp, col in [(rf_importance, 'Importance_RF'), 
                        (lgb_importance, 'Importance_LGB'), 
                        (xgb_importance, 'Importance_XGB')]:
        if df_imp[col].sum() > 0:  # Avoid division by zero
            df_imp[col] = df_imp[col] / df_imp[col].sum() * 100

    # Merge results
    try:
        combined = pd.merge(rf_importance, lgb_importance, on='Feature', how='outer')
        combined = pd.merge(combined, xgb_importance, on='Feature', how='outer')
        combined = combined.fillna(0)
    except Exception as e:
        print(f"Erro ao combinar resultados: {e}")
        # Alternative: use only the model that worked
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

    # Calculate mean and standard deviation of importances
    combined['Mean_Importance'] = combined[['Importance_RF', 'Importance_LGB', 'Importance_XGB']].mean(axis=1)
    combined['Std_Importance'] = combined[['Importance_RF', 'Importance_LGB', 'Importance_XGB']].std(axis=1)
    combined['CV'] = combined['Std_Importance'] / combined['Mean_Importance'].replace(0, 1e-10)

    # Sort by mean importance
    final_importance = combined.sort_values(by='Mean_Importance', ascending=False)

    print("\nImportância combinada (top 20 features):")
    print(final_importance[['Feature', 'Mean_Importance', 'Std_Importance', 'CV']].head(20))
    
    return final_importance

def analyze_launch_robustness(df, X, y, numeric_cols, launch_col, rename_dict=None, final_importance=None):
    """Analyze feature importance stability across different launches."""
    if not launch_col or df[launch_col].nunique() < 2:
        print("\nAnálise de robustez entre lançamentos não realizada (coluna de lançamento não identificada ou insuficiente)")
        return None, None
    
    print("\n--- Análise de Robustez entre Lançamentos ---")
    
    # Check if the launch column was renamed
    if rename_dict and launch_col in rename_dict:
        launch_col = rename_dict[launch_col]
    
    launch_imp_results = {}
    
    # Select main launches for analysis (top 6)
    main_launches = df[launch_col].value_counts().nlargest(6).index.tolist()
    
    for launch in main_launches:
        launch_mask = df[launch_col] == launch
        if launch_mask.sum() < 100:  # Skip very small launches
            print(f"Pulando lançamento {launch} (menos de 100 amostras)")
            continue
            
        print(f"\nAnalisando lançamento: {launch} ({launch_mask.sum()} amostras)")
        
        # Separate data for this launch
        X_launch = X[launch_mask]
        y_launch = y[launch_mask]
        
        # Check if there are enough positive samples
        n_pos = y_launch.sum()
        if n_pos < 5:
            print(f"  Pulando: apenas {n_pos} amostras positivas")
            continue
            
        # Create proportional split
        test_size = min(0.2, 100/len(y_launch))
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_launch, y_launch, 
                                                  test_size=test_size, 
                                                  random_state=42,
                                                  stratify=y_launch)
        
        # Try RandomForest (more robust to errors)
        try:
            # Calculate class_weight
            n_samples = len(y_tr)
            n_pos = y_tr.sum()
            n_neg = n_samples - n_pos
            weight_pos = (n_samples / (2 * n_pos)) if n_pos > 0 else 1.0
            weight_neg = (n_samples / (2 * n_neg)) if n_neg > 0 else 1.0
            
            # Train model only for this launch
            rf_model_launch = RandomForestClassifier(
                n_estimators=50, 
                class_weight={0: weight_neg, 1: weight_pos},
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            rf_model_launch.fit(X_tr, y_tr)
            
            # Get importance
            launch_imp = pd.DataFrame({
                'Feature': numeric_cols,
                f'Imp_{launch}': rf_model_launch.feature_importances_
            })
            
            # Normalize importance
            if launch_imp[f'Imp_{launch}'].sum() > 0:
                launch_imp[f'Imp_{launch}'] = launch_imp[f'Imp_{launch}'] / launch_imp[f'Imp_{launch}'].sum() * 100
            
            # Save results
            launch_imp_results[launch] = launch_imp
        except Exception as e:
            print(f"  Erro ao analisar lançamento {launch}: {e}")
    
    # Combine results from different launches
    if launch_imp_results:
        combined_launch_imp = launch_imp_results[list(launch_imp_results.keys())[0]].copy()
        
        for launch, imp_df in list(launch_imp_results.items())[1:]:
            combined_launch_imp = pd.merge(combined_launch_imp, imp_df, on='Feature', how='outer')
        
        combined_launch_imp = combined_launch_imp.fillna(0)
        
        # Calculate mean and standard deviation across launches
        imp_cols = [col for col in combined_launch_imp.columns if col.startswith('Imp_')]
        combined_launch_imp['Mean_Launch_Imp'] = combined_launch_imp[imp_cols].mean(axis=1)
        combined_launch_imp['Std_Launch_Imp'] = combined_launch_imp[imp_cols].std(axis=1)
        combined_launch_imp['CV_Launch'] = combined_launch_imp['Std_Launch_Imp'] / combined_launch_imp['Mean_Launch_Imp'].replace(0, 1e-10)
        
        # Sort by mean importance
        launch_importance = combined_launch_imp.sort_values(by='Mean_Launch_Imp', ascending=False)
        
        print("\nImportância média entre lançamentos (top 15 features):")
        print(launch_importance[['Feature', 'Mean_Launch_Imp', 'Std_Launch_Imp', 'CV_Launch']].head(15))
        
        # Identify features with high variability across launches
        unstable_features = launch_importance[
            (launch_importance['CV_Launch'] > 1.2) & 
            (launch_importance['Mean_Launch_Imp'] > 0.5)
        ].sort_values(by='CV_Launch', ascending=False)
        
        print("\nFeatures com alta variabilidade entre lançamentos:")
        print(unstable_features[['Feature', 'Mean_Launch_Imp', 'CV_Launch']].head(10))
        
        # Merge with global importance for comparison (if provided)
        if final_importance is not None:
            launch_vs_global = pd.merge(
                launch_importance[['Feature', 'Mean_Launch_Imp', 'CV_Launch']],
                final_importance[['Feature', 'Mean_Importance']],
                on='Feature', how='inner'
            )
            
            # Identify consistently important features
            consistent_features = launch_vs_global[
                (launch_vs_global['Mean_Launch_Imp'] > launch_vs_global['Mean_Launch_Imp'].median()) &
                (launch_vs_global['Mean_Importance'] > launch_vs_global['Mean_Importance'].median()) &
                (launch_vs_global['CV_Launch'] < 1.0)
            ].sort_values(by='Mean_Importance', ascending=False)
            
            print("\nFeatures consistentemente importantes entre lançamentos:")
            print(consistent_features[['Feature', 'Mean_Importance', 'Mean_Launch_Imp', 'CV_Launch']].head(15))
            
            return launch_importance, unstable_features, launch_vs_global, consistent_features
        
        return launch_importance, unstable_features, None, None
    
    return None, None, None, None