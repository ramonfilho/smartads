#!/usr/bin/env python
"""
Feature Selection Script for Smart Ads Project

Este script realiza seleção de features usando múltiplos modelos e técnicas
para identificar as features mais relevantes para predição de conversão.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from scipy.stats import pearsonr
import re
import warnings
import argparse
import json
from datetime import datetime
import logging
import time

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar projeto
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Importar módulos do projeto se disponíveis
try:
    from src.evaluation.feature_importance import (
        create_output_directory,
        load_dataset,
        identify_launch_column,
        identify_target_column,
        select_numeric_features,
        identify_text_derived_columns,
        sanitize_column_names,
        analyze_multicollinearity,
        compare_country_encodings,
        evaluate_model,
        analyze_rf_importance,
        analyze_lgb_importance,
        analyze_xgb_importance,
        combine_importance_results,
        analyze_launch_robustness
    )
    from src.evaluation.feature_selector import (
        identify_irrelevant_features,
        analyze_text_features,
        select_final_features,
        document_feature_selections,
        create_selected_dataset,
        summarize_feature_categories
    )
    logger.info("Módulos do projeto carregados com sucesso")
    USE_PROJECT_MODULES = True
except ImportError as e:
    logger.warning(f"Não foi possível importar módulos do projeto: {e}")
    logger.info("Usando implementação local")
    USE_PROJECT_MODULES = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Feature Selection for Smart Ads Project',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Diretórios
    parser.add_argument(
        '--input-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data/new/03_feature_engineering_1'),
        help='Diretório de entrada com os datasets processados'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data/new/04_feature_selection'),
        help='Diretório de saída para datasets com features selecionadas'
    )
    
    parser.add_argument(
        '--analysis-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'reports/feature_importance_results'),
        help='Diretório para salvar resultados de análise'
    )
    
    # Parâmetros
    parser.add_argument(
        '--importance-threshold',
        type=float,
        default=0.1,
        help='Threshold de importância mínima (em %% da importância total)'
    )
    
    parser.add_argument(
        '--cv-threshold',
        type=float,
        default=1.5,
        help='Threshold de coeficiente de variação para features instáveis'
    )
    
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.95,
        help='Threshold de correlação para identificar features redundantes'
    )
    
    parser.add_argument(
        '--n-folds',
        type=int,
        default=3,  # Reduzido de 5 para 3 para acelerar
        help='Número de folds para cross-validation'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Tamanho do batch para processamento (economia de memória)'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=300,  # Mudado de 700 para 300 como padrão
        help='Número máximo de features a selecionar'
    )
    
    parser.add_argument(
        '--fast-mode',
        action='store_true',
        help='Modo rápido: usa apenas RandomForest sem CV'
    )
    
    # Flags
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Usar cache de análises anteriores se disponível'
    )
    
    parser.add_argument(
        '--skip-launch-analysis',
        action='store_true',
        default=True,
        help='Pular análise de robustez entre lançamentos'
    )
    
    parser.add_argument(
        '--keep-all-features',
        action='store_true',
        help='Manter todas as features (apenas gerar análise)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Arquivo JSON com configurações (sobrescreve outros argumentos)'
    )
    
    args = parser.parse_args()
    
    # Carregar configurações de arquivo se fornecido
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        logger.info(f"Configurações carregadas de {args.config}")
    
    return args

def create_directories(args):
    """Criar diretórios necessários."""
    for dir_path in [args.output_dir, args.analysis_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Diretório criado: {dir_path}")
        else:
            logger.info(f"Diretório já existe: {dir_path}")

def load_datasets(args):
    """Carregar datasets de treino, validação e teste."""
    datasets = {}
    
    for dataset_name in ['train', 'validation', 'test']:
        file_path = os.path.join(args.input_dir, f"{dataset_name}.csv")
        
        try:
            # Tentar carregar em chunks se o arquivo for muito grande
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            if file_size_mb > 500:  # Se maior que 500MB
                logger.info(f"Arquivo {dataset_name}.csv grande ({file_size_mb:.1f}MB), carregando em chunks...")
                
                # Ler em chunks e concatenar
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=args.batch_size):
                    chunks.append(chunk)
                
                datasets[dataset_name] = pd.concat(chunks, ignore_index=True)
            else:
                datasets[dataset_name] = pd.read_csv(file_path)
            
            logger.info(f"Dataset {dataset_name} carregado: {datasets[dataset_name].shape}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar {dataset_name}: {e}")
            raise
    
    return datasets['train'], datasets['validation'], datasets['test']

def save_configuration(args, output_info):
    """Salvar configuração usada e informações de output."""
    config = {
        'timestamp': datetime.now().isoformat(),
        'arguments': vars(args),
        'output_info': output_info,
        'project_root': PROJECT_ROOT,
        'python_version': sys.version,
        'modules_used': 'project' if USE_PROJECT_MODULES else 'local'
    }
    
    config_path = os.path.join(args.analysis_dir, 'feature_selection_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuração salva em {config_path}")

def remove_perfect_correlations(X, threshold=0.999):
    """
    Remove features com correlação perfeita ou quase perfeita.
    Mantém a primeira de cada par.
    """
    # Calcular matriz de correlação
    corr_matrix = X.corr().abs()
    
    # Máscara triangular superior
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Encontrar features para remover
    to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
    
    if to_drop:
        logger.info(f"Removendo {len(to_drop)} features com correlação >= {threshold}")
        logger.info(f"Exemplos: {to_drop[:5]}")
    
    return X.drop(columns=to_drop), to_drop

def fast_feature_selection(X, y, n_features=500):
    """
    Seleção rápida de features usando RandomForest + estatísticas básicas.
    """
    logger.info(f"\nModo rápido: selecionando top {n_features} features")
    
    # 1. Remover features com baixa variância
    selector = VarianceThreshold(threshold=0.01)
    X_var = pd.DataFrame(
        selector.fit_transform(X),
        columns=X.columns[selector.get_support()],
        index=X.index
    )
    logger.info(f"Features após filtro de variância: {X_var.shape[1]}")
    
    # 2. F-score para ranking inicial rápido
    if X_var.shape[1] > n_features * 2:
        selector = SelectKBest(f_classif, k=min(n_features * 2, X_var.shape[1]))
        X_f = pd.DataFrame(
            selector.fit_transform(X_var, y),
            columns=X_var.columns[selector.get_support()],
            index=X_var.index
        )
        logger.info(f"Features após F-score: {X_f.shape[1]}")
    else:
        X_f = X_var
    
    # 3. RandomForest para importância final
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    # Usar amostra se dataset muito grande
    if len(X_f) > 50000:
        X_sample, _, y_sample, _ = train_test_split(
            X_f, y, train_size=50000/len(X_f), stratify=y, random_state=42
        )
        rf.fit(X_sample, y_sample)
    else:
        rf.fit(X_f, y)
    
    # Criar DataFrame de importância
    importance_df = pd.DataFrame({
        'feature': X_f.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Selecionar top N
    selected_features = importance_df.head(n_features)['feature'].tolist()
    
    return selected_features, importance_df

def handle_multicollinearity(X, y, numeric_cols, correlation_threshold=0.95):
    """
    Remove features altamente correlacionadas antes da análise.
    Para pares correlacionados, mantém a que tem maior correlação com target.
    """
    logger.info(f"\nRemovendo features com correlação > {correlation_threshold}")
    start_time = time.time()
    
    # Primeiro, remover correlações perfeitas (mais rápido)
    X_cleaned, perfect_corr_removed = remove_perfect_correlations(X, threshold=0.999)
    numeric_cols = [col for col in numeric_cols if col in X_cleaned.columns]
    
    # Calcular correlações apenas se necessário
    if correlation_threshold < 0.999:
        corr_matrix = X_cleaned.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Calcular correlação com target
        target_corr = X_cleaned.corrwith(y).abs()
        
        features_to_remove = set()
        correlation_groups = {}
        
        # Encontrar pares com alta correlação
        high_corr_indices = np.where(upper > correlation_threshold)
        
        for idx in range(len(high_corr_indices[0])):
            i = high_corr_indices[0][idx]
            j = high_corr_indices[1][idx]
            
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = upper.iloc[i, j]
            
            # Para alta correlação (não perfeita), manter a com maior correlação com target
            if target_corr[col1] < target_corr[col2]:
                features_to_remove.add(col1)
            else:
                features_to_remove.add(col2)
            
            # Registrar para análise
            key = tuple(sorted([col1, col2]))
            correlation_groups[key] = {
                'correlation': corr_value,
                'target_corr_1': target_corr[col1],
                'target_corr_2': target_corr[col2]
            }
        
        # Remover features adicionais
        if features_to_remove:
            X_cleaned = X_cleaned.drop(columns=list(features_to_remove))
            numeric_cols = [col for col in numeric_cols if col not in features_to_remove]
    else:
        features_to_remove = set()
        correlation_groups = {}
    
    total_removed = len(perfect_corr_removed) + len(features_to_remove)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Correlações processadas em {elapsed_time:.1f} segundos")
    logger.info(f"Features removidas: {total_removed} (perfeitas: {len(perfect_corr_removed)}, altas: {len(features_to_remove)})")
    logger.info(f"Features restantes: {len(numeric_cols)}")
    
    return X_cleaned, numeric_cols, features_to_remove.union(set(perfect_corr_removed)), correlation_groups

def main():
    """Função principal."""
    total_start_time = time.time()
    
    # Parse argumentos
    args = parse_arguments()
    
    logger.info("=== INICIANDO FEATURE SELECTION ===")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Modo: {'RÁPIDO' if args.fast_mode else 'COMPLETO'}")
    logger.info(f"Max features: {args.max_features}")
    
    # Criar diretórios
    create_directories(args)
    
    # Carregar datasets
    logger.info("\nCarregando datasets...")
    train_df, val_df, test_df = load_datasets(args)
    
    # Usar dataset de treino para análise
    df = train_df
    
    # Identificar coluna de lançamento
    launch_col = identify_launch_column(df) if not args.skip_launch_analysis else None
    
    # Identificar coluna target
    target_col = identify_target_column(df)
    
    # Selecionar features numéricas
    logger.info("\nPreparando dados para análise...")
    numeric_cols = select_numeric_features(df, target_col)
    initial_n_features = len(numeric_cols)
    
    # Identificar features de texto
    text_derived_cols = identify_text_derived_columns(numeric_cols)
    logger.info(f"Features derivadas de texto: {len(text_derived_cols)}")
    
    # Sanitizar nomes se necessário
    rename_dict = sanitize_column_names(numeric_cols)
    if rename_dict:
        logger.info(f"Renomeando {len(rename_dict)} colunas")
        df = df.rename(columns=rename_dict)
        numeric_cols = [rename_dict.get(col, col) for col in numeric_cols]
        text_derived_cols = [rename_dict.get(col, col) for col in text_derived_cols]
        if launch_col and launch_col in rename_dict:
            launch_col = rename_dict[launch_col]
    
    # Preparar dados
    X = df[numeric_cols].fillna(0)
    y = df[target_col]
    
    logger.info(f"Analisando {len(numeric_cols)} features")
    logger.info(f"Distribuição do target: {y.value_counts(normalize=True) * 100}")
    
    # PASSO 1: Remover correlações altas (ANTES de qualquer análise)
    X_cleaned, numeric_cols_cleaned, removed_features, corr_groups = \
        handle_multicollinearity(X, y, numeric_cols, args.correlation_threshold)
    
    X = X_cleaned
    numeric_cols = numeric_cols_cleaned
    text_derived_cols = [col for col in text_derived_cols if col in numeric_cols]
    
    # Salvar features removidas
    if removed_features:
        removed_path = os.path.join(args.analysis_dir, 'removed_correlations.txt')
        with open(removed_path, 'w') as f:
            f.write(f"Total removidas por correlação: {len(removed_features)}\n\n")
            for feat in sorted(removed_features):
                f.write(f"- {feat}\n")
    
    # PASSO 2: Análise de importância
    if args.fast_mode:
        # Modo rápido
        selected_features, importance_df = fast_feature_selection(X, y, args.max_features)
        
        # Criar estrutura compatível
        final_importance = pd.DataFrame({
            'Feature': importance_df['feature'],
            'Mean_Importance': importance_df['importance'] * 100,
            'Importance_RF': importance_df['importance'] * 100,
            'Importance_LGB': 0,
            'Importance_XGB': 0,
            'Std_Importance': 0,
            'CV': 0
        })
        
    else:
        # Modo completo - verificar cache primeiro
        cache_path = os.path.join(args.analysis_dir, 'importance_cache.pkl')
        
        if args.use_cache and os.path.exists(cache_path):
            try:
                import joblib
                cache_data = joblib.load(cache_path)
                logger.info("Cache carregado com sucesso")
                final_importance = cache_data.get('final_importance')
                
                # Filtrar apenas features que ainda existem
                final_importance = final_importance[final_importance['Feature'].isin(numeric_cols)]
                
            except Exception as e:
                logger.warning(f"Erro ao carregar cache: {e}")
                args.use_cache = False
        
        if not args.use_cache or 'final_importance' not in locals():
            # Separar dados
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Análise com múltiplos modelos
            logger.info("\nIniciando análise de importância com múltiplos modelos...")
            
            # RandomForest
            rf_importance, rf_metrics = analyze_rf_importance(
                X_train, y_train, numeric_cols, n_folds=args.n_folds
            )
            
            # LightGBM
            lgb_importance, lgb_metrics = analyze_lgb_importance(
                X_train, y_train, numeric_cols, n_folds=args.n_folds
            )
            
            # XGBoost
            xgb_importance, xgb_metrics = analyze_xgb_importance(
                X_train, y_train, numeric_cols, n_folds=args.n_folds
            )
            
            # Combinar resultados
            final_importance = combine_importance_results(
                rf_importance, lgb_importance, xgb_importance
            )
            
            # Salvar cache
            if not args.fast_mode:
                try:
                    import joblib
                    cache_data = {
                        'final_importance': final_importance,
                        'timestamp': datetime.now()
                    }
                    joblib.dump(cache_data, cache_path)
                    logger.info("Cache salvo com sucesso")
                except Exception as e:
                    logger.warning(f"Erro ao salvar cache: {e}")
    
    # Salvar importância das features
    importance_path = os.path.join(args.analysis_dir, 'feature_importance_combined.csv')
    final_importance.to_csv(importance_path, index=False)
    logger.info(f"Importância das features salva em {importance_path}")
    
    # PASSO 3: Seleção das top features baseado em importância
    logger.info(f"\n📊 Selecionando top {args.max_features} features baseado em Mean_Importance")
    
    # Ordenar por importância e pegar as top N
    top_features_df = final_importance.nlargest(args.max_features, 'Mean_Importance')
    top_features_sanitized = top_features_df['Feature'].tolist()
    
    # Converter de volta para nomes originais
    if rename_dict:
        reverse_rename_dict = {v: k for k, v in rename_dict.items()}
        top_features_original = []
        for feat in top_features_sanitized:
            original_name = reverse_rename_dict.get(feat, feat)
            top_features_original.append(original_name)
    else:
        top_features_original = top_features_sanitized
    
    logger.info(f"✅ Top {args.max_features} features selecionadas")
    logger.info(f"   Importância média das top features: {top_features_df['Mean_Importance'].mean():.4f}")
    logger.info(f"   Top 5 features: {top_features_original[:5]}")
    
    # Documentar features selecionadas
    recommended_path = os.path.join(args.analysis_dir, 'recommended_features.txt')
    with open(recommended_path, 'w') as f:
        for feature in top_features_original:
            f.write(f"{feature}\n")
    logger.info(f"Lista de features recomendadas salva em {recommended_path}")
    
    # Salvar também um CSV com as top features e suas importâncias
    top_features_info = top_features_df.copy()
    if rename_dict:
        # Adicionar coluna com nomes originais
        top_features_info['Original_Feature'] = [
            reverse_rename_dict.get(feat, feat) for feat in top_features_info['Feature']
        ]
    top_features_info.to_csv(
        os.path.join(args.analysis_dir, 'top_features_importance.csv'), 
        index=False
    )
    
    if not args.keep_all_features:
        # PASSO 4: Aplicar seleção aos datasets, salvando APENAS as top features
        logger.info(f"\n📁 Salvando datasets com apenas as top {args.max_features} features...")
        
        for df_name, df_data in [('train', train_df), ('validation', val_df), ('test', test_df)]:
            # Verificar quais features existem no dataset
            features_in_dataset = [f for f in top_features_original if f in df_data.columns]
            missing_features = set(top_features_original) - set(features_in_dataset)
            
            if missing_features:
                logger.warning(f"  {df_name}: {len(missing_features)} features não encontradas")
            
            # Adicionar target se existir
            columns_to_keep = features_in_dataset.copy()
            if target_col in df_data.columns:
                columns_to_keep.append(target_col)
            
            # Criar dataset selecionado
            df_selected = df_data[columns_to_keep]
            
            # Salvar
            output_path = os.path.join(args.output_dir, f"{df_name}.csv")
            df_selected.to_csv(output_path, index=False)
            
            logger.info(f"✅ {df_name}: {df_data.shape[1]} → {df_selected.shape[1]} colunas")
            logger.info(f"   Features: {len(features_in_dataset)}, Target: {'Sim' if target_col in columns_to_keep else 'Não'}")
        
        # Informações de output
        output_info = {
            'features_selected': args.max_features,
            'features_removed': initial_n_features - args.max_features,
            'original_features': initial_n_features,
            'features_after_correlation_removal': len(numeric_cols),
            'datasets_processed': ['train', 'validation', 'test'],
            'mode': 'fast' if args.fast_mode else 'complete',
            'top_5_features': top_features_original[:5]
        }
    else:
        logger.info("\nModo --keep-all-features ativado. Apenas gerando análise.")
        output_info = {
            'analysis_only': True,
            'total_features': len(numeric_cols),
            'text_features': len(text_derived_cols)
        }
    
    # Análise de features textuais
    if text_derived_cols and not args.fast_mode:
        text_importance = analyze_text_features(final_importance, text_derived_cols)
        if text_importance is not None:
            text_path = os.path.join(args.analysis_dir, 'text_features_importance.csv')
            text_importance.to_csv(text_path, index=False)
    
    # Sumarizar categorias
    if not args.fast_mode:
        summarize_feature_categories(numeric_cols, final_importance, text_derived_cols)
    
    # Salvar configuração
    save_configuration(args, output_info)
    
    # Tempo total
    total_time = time.time() - total_start_time
    logger.info(f"\n=== FEATURE SELECTION CONCLUÍDO EM {total_time:.1f} SEGUNDOS ===")
    logger.info(f"Resultados salvos em: {args.analysis_dir}")
    if not args.keep_all_features:
        logger.info(f"Datasets selecionados salvos em: {args.output_dir}")
        logger.info(f"Cada dataset contém apenas as top {args.max_features} features + target")

if __name__ == "__main__":
    main()