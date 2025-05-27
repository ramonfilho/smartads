#!/usr/bin/env python
"""
Script corrigido para treinar modelos baseline e garantir o salvamento correto do modelo RandomForest.
Este script resolve problemas de configuração do MLflow para evitar a falha no salvamento do modelo.
"""

import os
import sys
import mlflow
import pandas as pd
import numpy as np
import argparse
import unicodedata
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score

# Caminho absoluto para o diretório raiz do projeto
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"

# ========== FUNÇÕES PARA TOP FEATURES ==========
# Configuração para usar apenas top features
USE_TOP_FEATURES = True  # Flag para ativar/desativar
N_TOP_FEATURES = 300     # Número de features a usar

def get_top_features(n_features=300):
    """
    Carrega as top N features baseado na importância média com mapeamento flexível.
    """
    importance_path = os.path.join(
        PROJECT_ROOT, 
        "reports/feature_importance_results/feature_importance_combined.csv"
    )
    
    if not os.path.exists(importance_path):
        print(f"AVISO: Arquivo de importância não encontrado: {importance_path}")
        print("Continuando com todas as features...")
        return None, {}
    
    # Carregar importâncias
    importance_df = pd.read_csv(importance_path)
    
    # Ordenar por Mean_Importance (já deve estar ordenado, mas garantir)
    importance_df = importance_df.sort_values('Mean_Importance', ascending=False)
    
    # Pegar top N features
    top_features = importance_df.head(n_features)['Feature'].tolist()
    
    print(f"✓ Selecionadas top {len(top_features)} features baseado em Mean_Importance")
    print(f"  Top 5: {top_features[:5]}")
    print(f"  Importância média das selecionadas: {importance_df.head(n_features)['Mean_Importance'].mean():.4f}")
    
    # Adicionar mapeamento alternativo para features comuns
    feature_mapping = {}
    for feat in top_features:
        # Adicionar versão original
        feature_mapping[feat] = feat
        
        # Adicionar versões alternativas comuns
        # Para features de texto que podem ter sido encurtadas
        if len(feat) > 30:
            # Versão truncada
            truncated = feat[:30]
            feature_mapping[truncated] = feat
            
        # Para underscores duplos que podem ter virado simples
        if '__' in feat:
            single_underscore = feat.replace('__', '_')
            feature_mapping[single_underscore] = feat
            
        # Para caracteres especiais que podem ter sido alterados
        if any(c in feat for c in ['ñ', 'á', 'é', 'í', 'ó', 'ú']):
            # Versão sem acentos
            normalized = unicodedata.normalize('NFKD', feat)
            ascii_version = normalized.encode('ascii', 'ignore').decode('ascii')
            feature_mapping[ascii_version] = feat
    
    return top_features, feature_mapping

# Carregar lista de features se habilitado
if USE_TOP_FEATURES:
    TOP_FEATURES_LIST, FEATURE_MAPPING = get_top_features(N_TOP_FEATURES)
else:
    TOP_FEATURES_LIST = None
    FEATURE_MAPPING = {}
# ========== FIM DAS FUNÇÕES PARA TOP FEATURES ==========

sys.path.insert(0, PROJECT_ROOT)

# Adicionar TODOS os paths necessários ao PYTHONPATH ANTES de qualquer import
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "evaluation"))

# Importar módulos
from src.evaluation.baseline_model import run_baseline_model_training

# ========== PARSE DE ARGUMENTOS ==========
def parse_arguments():
    parser = argparse.ArgumentParser(description='Treinar modelos baseline')
    parser.add_argument('--n-features', type=int, default=300, 
                        help='Número de top features a usar')
    parser.add_argument('--models', nargs='+', default=['all'],
                        choices=['rf', 'lgb', 'xgb', 'all'],
                        help='Modelos a treinar')
    parser.add_argument('--optimize-threshold', action='store_true',
                        help='Otimizar threshold para F1')
    return parser.parse_args()

# Parse argumentos (mas não usar ainda para manter compatibilidade)
# args = parse_arguments()
# ========== FIM DO PARSE DE ARGUMENTOS ==========

# Configuração de caminhos - CORRIGIDO para usar pasta 04_feature_selection
train_path = os.path.join(PROJECT_ROOT, "data/new/04_feature_selection/train.csv")
val_path = os.path.join(PROJECT_ROOT, "data/new/04_feature_selection/validation.csv")

# Diretório MLflow para tracking - usando caminho absoluto
mlflow_dir = os.path.join(PROJECT_ROOT, "mlflow")
if not os.path.exists(mlflow_dir):
    os.makedirs(mlflow_dir, exist_ok=True)

# Diretório permanente para artefatos
artifact_dir = os.path.join(PROJECT_ROOT, "models/mlflow_artifacts")
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir, exist_ok=True)

# Configurar MLflow corretamente
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_dir}"
mlflow.set_tracking_uri(f"file://{mlflow_dir}")
print(f"MLflow tracking URI configurado para: {mlflow.get_tracking_uri()}")

# Verificar experimento
experiment_name = "smart_ads_baseline"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Criado novo experimento: {experiment_name} (ID: {experiment_id})")
else:
    experiment_id = experiment.experiment_id
    print(f"Usando experimento existente: {experiment_name} (ID: {experiment_id})")

# ========== FUNÇÃO CORRIGIDA PARA ANÁLISE POR DECIL ==========
def analyze_deciles(y_true, y_proba):
    """Analisa performance por decil"""
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    
    # Ordenar por probabilidade decrescente
    df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
    
    # Criar decis baseado na posição
    n_samples = len(df)
    decile_size = n_samples // 10
    
    # Atribuir decil baseado na posição
    df['decile'] = 10  # Inicializar todos como decil 10
    for i in range(10):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < 9 else n_samples
        df.loc[start_idx:end_idx-1, 'decile'] = 10 - i  # Decil 10 = maior probabilidade
    
    stats = df.groupby('decile').agg({
        'y_true': ['sum', 'count', 'mean']
    })
    stats.columns = ['conversions', 'total', 'rate']
    
    # Calcular lift apenas onde há dados
    overall_rate = df['y_true'].mean()
    stats['lift'] = stats['rate'] / overall_rate
    
    # Garantir que temos índice correto (1-10)
    stats = stats.sort_index()
    
    return stats
# ========== FIM DA FUNÇÃO ANALYZE_DECILES ==========

# ========== IMPORTAR E MODIFICAR FUNÇÕES NECESSÁRIAS ==========
# Importar as funções originais
from src.evaluation.baseline_model import (
    prepare_data_for_training as original_prepare_data,
    get_baseline_models as original_get_baseline_models,
    train_and_evaluate_model as original_train_and_evaluate
)

# Wrapper para prepare_data_for_training com suporte a top features
def prepare_data_for_training(train_path, val_path=None):
    """Wrapper com suporte para top features"""
    # Chamar função original
    data = original_prepare_data(train_path, val_path)
    
    # Aplicar filtro de top features se configurado
    if TOP_FEATURES_LIST is not None:
        feature_cols = data['feature_cols']
        column_mapping = data['column_mapping']
        
        # Carregar um sample do dataset para ver as colunas disponíveis
        sample_df = pd.read_csv(train_path, nrows=5)
        available_cols = [col for col in sample_df.columns if col != 'target']
        
        print(f"\nDEBUG - Features no dataset: {len(available_cols)}")
        print(f"DEBUG - Top features solicitadas: {len(TOP_FEATURES_LIST)}")
        
        # Tentar matching mais flexível
        found_features = []
        not_found = []
        
        for top_feat in TOP_FEATURES_LIST:
            # Tentar encontrar a feature no dataset
            found = False
            
            # 1. Match exato
            if top_feat in available_cols:
                found_features.append(top_feat)
                found = True
            else:
                # 2. Verificar se existe no column_mapping
                for orig_col, mapped_col in column_mapping.items():
                    if top_feat == orig_col and mapped_col in available_cols:
                        found_features.append(mapped_col)
                        found = True
                        break
                
                # 3. Verificar match parcial (primeiros 30 caracteres)
                if not found:
                    for col in available_cols:
                        if (len(top_feat) > 30 and len(col) > 30 and 
                            top_feat[:30] == col[:30]):
                            found_features.append(col)
                            found = True
                            break
                        
                        # 4. Verificar sem caracteres especiais
                        if not found:
                            top_feat_clean = ''.join(c for c in top_feat if c.isalnum() or c == '_')
                            col_clean = ''.join(c for c in col if c.isalnum() or c == '_')
                            if top_feat_clean == col_clean:
                                found_features.append(col)
                                found = True
                                break
            
            if not found:
                not_found.append(top_feat)
        
        # Remover duplicatas mantendo ordem
        found_features = list(dict.fromkeys(found_features))
        
        print(f"DEBUG - Features encontradas: {len(found_features)}")
        print(f"DEBUG - Features não encontradas: {len(not_found)}")
        if not_found and len(not_found) <= 10:
            print(f"  Exemplos não encontrados: {not_found[:5]}")
        
        # Atualizar dados com features encontradas
        if found_features:
            data['feature_cols'] = found_features
            data['X_train'] = data['X_train'][found_features]
            if data['X_val'] is not None:
                data['X_val'] = data['X_val'][found_features]
            
            print(f"\n📊 Usando apenas top {len(found_features)} features (de {len(available_cols)} disponíveis)")
            print(f"Novo shape - treino: {data['X_train'].shape}")
            if data['X_val'] is not None:
                print(f"validação: {data['X_val'].shape}")
    
    return data

# Wrapper para get_baseline_models com scale_pos_weight dinâmico
def get_baseline_models(y_train, class_weight_rf='balanced'):
    """Wrapper com cálculo dinâmico de scale_pos_weight"""
    # Calcular ratio real
    neg_samples = (y_train == 0).sum()
    pos_samples = (y_train == 1).sum()
    scale_pos_weight = neg_samples / pos_samples
    print(f"Scale pos weight calculado: {scale_pos_weight:.2f}")
    
    return original_get_baseline_models(class_weight_rf, scale_pos_weight)

# Wrapper para train_and_evaluate_model com métricas adicionais
def train_and_evaluate_model(model, name, X_train, y_train, X_val, y_val, 
                           experiment_id, artifact_dir, feature_cols,
                           train_data_hash, val_data_hash, integer_columns,
                           generate_learning_curves=False):
    """Wrapper com métricas adicionais para dados desbalanceados"""
    
    # Chamar função original
    results = original_train_and_evaluate(
        model, name, X_train, y_train, X_val, y_val,
        experiment_id, artifact_dir, feature_cols,
        train_data_hash, val_data_hash, integer_columns,
        generate_learning_curves
    )
    
    # Adicionar análise por decil dentro do MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{name}_decile_analysis"):
        # Fazer predições
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= results['threshold']).astype(int)
        
        # Métricas adicionais
        mcc = matthews_corrcoef(y_val, y_pred)
        kappa = cohen_kappa_score(y_val, y_pred)
        
        mlflow.set_tag("model_type", f"{name}_decile")
        mlflow.log_metric("mcc", mcc)
        mlflow.log_metric("cohen_kappa", kappa)
        
        # Análise por decil
        try:
            decile_stats = analyze_deciles(y_val, y_pred_proba)
            decile_path = os.path.join(artifact_dir, f"decile_analysis_{name}.csv")
            decile_stats.to_csv(decile_path)
            mlflow.log_artifact(decile_path)
            
            # Log métricas de decil com verificação
            if 10 in decile_stats.index:
                mlflow.log_metric("top_decile_lift", decile_stats.loc[10, 'lift'])
                mlflow.log_metric("top_decile_conversion_rate", decile_stats.loc[10, 'rate'])
            else:
                # Pegar o decil mais alto disponível
                max_decile = decile_stats.index.max()
                mlflow.log_metric("top_decile_lift", decile_stats.loc[max_decile, 'lift'])
                mlflow.log_metric("top_decile_conversion_rate", decile_stats.loc[max_decile, 'rate'])
                print(f"  ⚠️ Aviso: Usando decil {max_decile} como top (10 não disponível)")
            
            # Top 2 deciles com verificação
            top_deciles = [d for d in [9, 10] if d in decile_stats.index]
            if top_deciles:
                top_2deciles_conversions = decile_stats.loc[top_deciles, 'conversions'].sum()
                mlflow.log_metric("top_2deciles_recall", top_2deciles_conversions / y_val.sum())
                mlflow.log_metric("top_2deciles_conversions", int(top_2deciles_conversions))
            
            print(f"\n📊 Análise por Decil - {name}:")
            print(decile_stats[['conversions', 'total', 'rate', 'lift']])
            
            # Adicionar métricas ao resultado
            results['mcc'] = mcc
            results['cohen_kappa'] = kappa
            if 10 in decile_stats.index:
                results['top_decile_lift'] = decile_stats.loc[10, 'lift']
            
            # Limpar arquivo temporário
            if os.path.exists(decile_path):
                os.remove(decile_path)
                
        except Exception as e:
            print(f"  ❌ Erro na análise por decil: {e}")
            import traceback
            traceback.print_exc()
    
    return results
# ========== FIM DAS MODIFICAÇÕES ==========

# Executar o treinamento
print(f"\nTreinando modelos usando dados de: {train_path}")
print(f"Artefatos serão salvos em: {artifact_dir}")

# Preparar dados com possível filtro de features
data = prepare_data_for_training(train_path, val_path)

# Aplicar filtro adicional se necessário (compatibilidade com versão anterior)
if USE_TOP_FEATURES and TOP_FEATURES_LIST is not None and 'feature_cols' in data:
    print(f"\n📊 Filtrando para top {len(data['feature_cols'])} features")
    print(f"Novo shape após filtro - treino: {data['X_train'].shape}")

# Importar função para calcular hash dos dados
from src.utils.mlflow_utils import get_data_hash

# Calcular hashes
train_hash = get_data_hash(data['train_df'])
val_hash = get_data_hash(data['val_df']) if data['val_df'] is not None else None

# Obter modelos com scale_pos_weight dinâmico
models = get_baseline_models(data['y_train'])

# Treinar cada modelo
all_results = {}
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Treinando {name}...")
    print(f"{'='*60}")
    
    try:
        results = train_and_evaluate_model(
            model, name,
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            experiment_id, artifact_dir,
            data['feature_cols'], 
            train_hash, val_hash,
            data['integer_columns'],
            generate_learning_curves=False
        )
        
        # Armazenar resultados
        if results:
            for key, value in results.items():
                all_results[f"{name}_{key}"] = value
                
    except Exception as e:
        print(f"❌ Erro ao treinar {name}: {e}")
        import traceback
        traceback.print_exc()

print("\nTreinamento de modelos baseline concluído.")

# Verificar se o Random Forest foi salvo
rf_model_uri = all_results.get('random_forest_model_uri')
if rf_model_uri:
    print(f"\nRandomForest model URI: {rf_model_uri}")
    
    # Opcional: Salvar o URI do modelo para referência futura
    model_uri_path = os.path.join(PROJECT_ROOT, "models/rf_model_uri.txt")
    with open(model_uri_path, "w") as f:
        f.write(rf_model_uri)
else:
    print("\nAVISO: Não foi possível confirmar o salvamento do modelo RandomForest.")

# Resumo final
print("\n" + "="*60)
print("RESUMO FINAL DOS MODELOS")
print("="*60)
for model_name in models.keys():
    if f"{model_name}_f1" in all_results:
        print(f"\n{model_name.upper()}:")
        print(f"  F1-Score: {all_results[f'{model_name}_f1']:.4f}")
        print(f"  Precisão: {all_results[f'{model_name}_precision']:.4f}")
        print(f"  Recall: {all_results[f'{model_name}_recall']:.4f}")
        print(f"  AUC: {all_results[f'{model_name}_auc']:.4f}")
        print(f"  PR-AUC: {all_results[f'{model_name}_pr_auc']:.4f}")
        print(f"  Threshold: {all_results[f'{model_name}_threshold']:.4f}")
        
        # Adicionar métricas extras se disponíveis
        if f"{model_name}_mcc" in all_results:
            print(f"  MCC: {all_results[f'{model_name}_mcc']:.4f}")
        if f"{model_name}_cohen_kappa" in all_results:
            print(f"  Cohen's Kappa: {all_results[f'{model_name}_cohen_kappa']:.4f}")
        if f"{model_name}_top_decile_lift" in all_results:
            print(f"  Top Decile Lift: {all_results[f'{model_name}_top_decile_lift']:.2f}x")

print("="*60)