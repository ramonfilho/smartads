#!/usr/bin/env python
"""
Script para realizar seleção de features baseada em um modelo já treinado.

Este script carrega um modelo RandomForest com boa performance e o utiliza
para avaliar e selecionar features do conjunto completo de features disponíveis.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import argparse
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import mlflow

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

def get_latest_random_forest_run(mlflow_dir):
    """
    Obtém o run_id do modelo RandomForest mais recente.
    
    Args:
        mlflow_dir: Diretório do MLflow tracking
        
    Returns:
        Tuple com (run_id, threshold, model_uri) ou (None, None, None) se não encontrado
    """
    # Configurar MLflow
    if os.path.exists(mlflow_dir):
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    else:
        print(f"AVISO: Diretório MLflow não encontrado: {mlflow_dir}")
        return None, None, None
    
    # Inicializar cliente MLflow
    client = mlflow.tracking.MlflowClient()
    
    # Procurar todos os experimentos
    experiments = client.search_experiments()
    
    for experiment in experiments:
        print(f"Verificando experimento: {experiment.name} (ID: {experiment.experiment_id})")
        
        # Buscar runs específicos do RandomForest ordenados pelo mais recente
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_type = 'random_forest'",
            order_by=["attribute.start_time DESC"]
        )
        
        if not runs:
            # Se não achou pela tag, procurar pelo nome do artefato
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"]
            )
        
        for run in runs:
            run_id = run.info.run_id
            print(f"  Encontrado run: {run_id}")
            
            # Verificar artefatos
            artifacts = client.list_artifacts(run_id)
            rf_artifact = None
            
            for artifact in artifacts:
                if artifact.is_dir and artifact.path == 'random_forest':
                    rf_artifact = artifact
                    break
            
            if rf_artifact:
                # Extrair o threshold das métricas
                threshold = run.data.metrics.get('threshold', 0.17)  # Fallback para 0.17 se não encontrar
                model_uri = f"runs:/{run_id}/random_forest"
                
                print(f"  Usando modelo RandomForest de {run.info.start_time}")
                print(f"  Run ID: {run_id}")
                print(f"  Model URI: {model_uri}")
                print(f"  Threshold: {threshold}")
                
                # Mostrar métricas registradas no MLflow
                precision = run.data.metrics.get('precision', None)
                recall = run.data.metrics.get('recall', None)
                f1 = run.data.metrics.get('f1', None)
                
                if precision and recall and f1:
                    print(f"  Métricas do MLflow: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                
                return run_id, threshold, model_uri
    
    print("Nenhum modelo RandomForest encontrado em MLflow.")
    return None, None, None

def load_model_from_mlflow(model_uri):
    """
    Carrega um modelo a partir do MLflow usando seu URI.
    
    Args:
        model_uri: URI do modelo no formato 'runs:/<run_id>/<artifact_path>'
        
    Returns:
        Modelo carregado ou None se falhar
    """
    try:
        print(f"Carregando modelo de: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        return None

def load_datasets(input_dir=None):
    """
    Carrega os datasets de treino, validação e teste.
    
    Args:
        input_dir: Caminho para o diretório com os datasets
        
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    # Definir caminhos padrão se não fornecidos
    if not input_dir:
        input_dir = os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/02_4_processed_text_fn_code7")
    
    train_path = os.path.join(input_dir, "train.csv")
    val_path = os.path.join(input_dir, "validation.csv")
    test_path = os.path.join(input_dir, "test.csv")
    
    # Verificar se os arquivos existem
    print(f"Verificando existência dos arquivos de entrada:")
    print(f"  Train path: {train_path} - Existe: {os.path.exists(train_path)}")
    print(f"  Val path: {val_path} - Existe: {os.path.exists(val_path)}")
    print(f"  Test path: {test_path} - Existe: {os.path.exists(test_path)}")
    
    # Carregar os dados
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    
    if train_df is None:
        raise ValueError(f"Arquivo de treino não encontrado: {train_path}")
    
    print(f"Dados carregados: ")
    print(f"  Treino: {train_df.shape}")
    if val_df is not None:
        print(f"  Validação: {val_df.shape}")
    if test_df is not None:
        print(f"  Teste: {test_df.shape}")
    
    return train_df, val_df, test_df

def get_model_selected_features(model):
    """
    Extrai as features que o modelo atual está usando.
    
    Args:
        model: Modelo treinado
        
    Returns:
        Lista de features usadas pelo modelo
    """
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    else:
        print("AVISO: Modelo não tem atributo feature_names_in_")
        return []

def evaluate_feature_importance_with_model(model, X, y, feature_names, target_col='target'):
    """
    Avalia a importância de cada feature usando o modelo existente.
    
    Esta função utiliza o modelo para predizer com cada feature e calcula o ganho
    de performance comparado com o modelo base.
    
    Args:
        model: Modelo treinado
        X: DataFrame com todas as features
        y: Vetor target
        feature_names: Lista com todas as features disponíveis
        target_col: Nome da coluna target
        
    Returns:
        DataFrame com importância de cada feature
    """
    print("Avaliando importância das features com o modelo existente...")
    
    # Obter features usadas pelo modelo
    model_features = get_model_selected_features(model)
    print(f"O modelo usa {len(model_features)} features")
    
    # Verificar quais features do modelo estão presentes nos dados
    missing_model_features = [f for f in model_features if f not in X.columns]
    if missing_model_features:
        print(f"AVISO: {len(missing_model_features)} features usadas pelo modelo não estão nos dados:")
        for f in missing_model_features[:10]:
            print(f"  - {f}")
        if len(missing_model_features) > 10:
            print(f"  - ... e mais {len(missing_model_features) - 10} features")
        
        print("Criando features ausentes preenchidas com zeros...")
        for f in missing_model_features:
            X[f] = 0
    
    # Features disponíveis que não estão no modelo
    new_features = [f for f in feature_names if f not in model_features and f != target_col]
    print(f"Avaliando {len(new_features)} features adicionais")
    
    # Converter todas as colunas para tipos numéricos quando possível
    # Isso evita erros relacionados a strings que não podem ser convertidas para números
    print("Convertendo dados para formato numérico...")
    for col in X.columns:
        # Tentar converter para números, preenchendo não-numéricos com NaN
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            # Se falhar completamente, criar uma nova coluna one-hot encoded
            if X[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(X[col]):
                try:
                    print(f"  Convertendo coluna categórica: {col}")
                    # Identificar valores únicos (limitando a 100 para evitar explosão)
                    unique_vals = X[col].dropna().unique()
                    if len(unique_vals) <= 100:
                        # Criar dummies apenas para colunas com cardinalidade razoável
                        dummies = pd.get_dummies(X[col], prefix=col, dummy_na=False)
                        # Adicionar as colunas dummies
                        for dummy_col in dummies.columns:
                            if dummy_col not in X.columns:
                                X[dummy_col] = dummies[dummy_col]
                    
                    # Remover a coluna original se conseguiu criar dummies
                    if col in new_features:
                        new_features.remove(col)
                        # Adicionar as novas colunas dummies à lista de features
                        new_features.extend([c for c in dummies.columns if c not in model_features])
                except Exception as e:
                    print(f"    Erro ao converter coluna {col}: {str(e)}")
                    # Se der erro, tentar remover a coluna da análise
                    if col in new_features:
                        new_features.remove(col)
    
    # Preencher valores ausentes com 0
    X = X.fillna(0)
    
    # Obter baseline de performance (usando apenas as features do modelo)
    X_base = X[model_features]
    y_pred_base = model.predict_proba(X_base)[:, 1]
    base_auc = roc_auc_score(y, y_pred_base)
    print(f"Performance base (AUC): {base_auc:.4f}")
    
    # Lista para armazenar resultados
    feature_importance = []
    
    # Avaliar cada feature adicional
    for i, feature in enumerate(new_features):
        if i % 50 == 0:
            print(f"  Avaliando feature {i+1}/{len(new_features)}")
        
        # Criar matriz com features do modelo + a nova feature
        if feature in X.columns:
            try:
                # Verificar se a feature é numérica e tem variância não-zero
                is_valid = True
                
                if not pd.api.types.is_numeric_dtype(X[feature]):
                    is_valid = False
                elif X[feature].var() == 0:
                    is_valid = False
                
                if not is_valid:
                    feature_importance.append({
                        'feature': feature,
                        'auc_gain': 0,
                        'is_valid': False
                    })
                    continue
                
                # Criar DataFrame com features base + a nova feature
                X_extended = X[model_features + [feature]]
                
                # Criar um novo modelo com a mesma configuração
                new_model = RandomForestClassifier(n_estimators=100, random_state=42)
                new_model.fit(X_extended, y)
                
                # Avaliar performance
                y_pred_new = new_model.predict_proba(X_extended)[:, 1]
                new_auc = roc_auc_score(y, y_pred_new)
                
                # Calcular ganho
                auc_gain = new_auc - base_auc
                
                feature_importance.append({
                    'feature': feature,
                    'auc_gain': auc_gain,
                    'is_valid': True
                })
            except Exception as e:
                print(f"  Erro ao avaliar feature {feature}: {str(e)}")
                feature_importance.append({
                    'feature': feature,
                    'auc_gain': 0,
                    'is_valid': False
                })
    
    # Adicionar features originais do modelo com valor fixo alto
    for feature in model_features:
        feature_importance.append({
            'feature': feature,
            'auc_gain': 0.1,  # Valor arbitrário alto para manter features originais
            'is_valid': True,
            'in_current_model': True
        })
    
    # Converter para DataFrame e ordenar
    importance_df = pd.DataFrame(feature_importance).sort_values('auc_gain', ascending=False)
    
    return importance_df

def select_top_features(importance_df, original_model_features, n_features=200):
    """
    Seleciona as top features com base na importância.
    
    Args:
        importance_df: DataFrame com importância das features
        original_model_features: Lista de features do modelo original
        n_features: Número de features a selecionar
        
    Returns:
        Lista de features selecionadas
    """
    print(f"\nSelecionando top {n_features} features...")
    
    # Garantir que as features do modelo original sejam incluídas
    model_features_in_df = [f for f in original_model_features if f in importance_df['feature'].values]
    
    # Contar quantas features do modelo original temos
    n_original = len(model_features_in_df)
    print(f"  Mantendo {n_original} features do modelo original")
    
    # Selecionar features adicionais com base no ganho de AUC
    additional_features = importance_df[
        ~importance_df['feature'].isin(original_model_features) & 
        importance_df['is_valid']
    ].sort_values('auc_gain', ascending=False)
    
    # Selecionar as top features adicionais
    n_additional = min(n_features - n_original, len(additional_features))
    top_additional = additional_features.head(n_additional)['feature'].tolist()
    
    print(f"  Adicionando {len(top_additional)} novas features com maior ganho")
    
    # Combinar features originais e novas
    selected_features = list(model_features_in_df) + top_additional
    
    return selected_features

def apply_feature_selection(df, selected_features, target_col='target'):
    """
    Aplica a seleção de features a um DataFrame.
    
    Args:
        df: DataFrame a processar
        selected_features: Lista de features selecionadas
        target_col: Nome da coluna target
        
    Returns:
        DataFrame com as features selecionadas
    """
    # Verificar quais features existem no DataFrame
    available_features = [col for col in selected_features if col in df.columns]
    missing_features = list(set(selected_features) - set(available_features))
    
    # Criar DataFrame com features disponíveis + target
    columns_to_keep = available_features + [target_col] if target_col in df.columns else available_features
    df_selected = df[columns_to_keep].copy()
    
    # Adicionar features faltantes preenchidas com zeros
    if missing_features:
        print(f"Adicionando {len(missing_features)} features faltantes, preenchidas com zeros")
        
        # Criar DataFrame com as features faltantes
        missing_df = pd.DataFrame(0, index=df.index, columns=missing_features)
        
        # Concatenar com o DataFrame existente
        df_selected = pd.concat([df_selected, missing_df], axis=1)
    
    print(f"DataFrame final: {df.shape[1]} colunas originais → {df_selected.shape[1]} colunas selecionadas")
    return df_selected

def plot_feature_importance(importance_df, output_dir, top_n=50):
    """
    Plota e salva o gráfico de importância das features.
    
    Args:
        importance_df: DataFrame com importância das features
        output_dir: Diretório para salvar o gráfico
        top_n: Número de top features para mostrar
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar apenas features válidas
    valid_importance = importance_df[importance_df['is_valid']].copy()
    
    # Marcar features do modelo original
    has_model_column = 'in_current_model' in valid_importance.columns
    
    plt.figure(figsize=(12, 10))
    
    # Selecionar top features para visualização
    top_features = valid_importance.head(top_n).copy()
    
    # Criar cores diferentes para features novas vs. originais
    if has_model_column:
        colors = ['royalblue' if in_model else 'green' 
                 for in_model in top_features['in_current_model'].fillna(False)]
    else:
        colors = ['royalblue'] * len(top_features)
    
    # Ordenar para visualização (maior para menor)
    top_features = top_features.sort_values('auc_gain')
    
    # Criar barras horizontais
    plt.barh(range(len(top_features)), top_features['auc_gain'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Ganho de AUC')
    plt.title(f'Top {top_n} Features por Ganho de Performance')
    
    # Adicionar legenda se tivermos a coluna de modelo
    if has_model_column:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='royalblue', label='Features do modelo original'),
            Patch(facecolor='green', label='Novas features')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    # Salvar também como CSV
    valid_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

def main(args):
    """
    Função principal do script.
    """
    # 1. Configurar diretórios de saída
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'model_selected_features'), exist_ok=True)
    
    # 2. Carregar modelo
    run_id, threshold, model_uri = get_latest_random_forest_run(args.mlflow_dir)
    
    if model_uri is None:
        if args.model_path:
            # Tentar carregar do caminho específico
            try:
                model = joblib.load(args.model_path)
                print(f"Modelo carregado de {args.model_path}")
            except Exception as e:
                print(f"Erro ao carregar modelo do caminho especificado: {str(e)}")
                return
        else:
            print("Nenhum modelo encontrado no MLflow e nenhum caminho de modelo fornecido.")
            return
    else:
        model = load_model_from_mlflow(model_uri)
        if model is None:
            print("Falha ao carregar o modelo do MLflow.")
            return
    
    # 3. Carregar datasets
    train_df, val_df, test_df = load_datasets(args.input_dir)
    
    # 4. Identificar features usadas pelo modelo atual
    current_model_features = get_model_selected_features(model)
    
    # 5. Preparar dados para avaliação
    # Separar features e target
    target_col = args.target_col
    if target_col not in train_df.columns:
        # Tentar detectar coluna target automaticamente
        if 'target' in train_df.columns:
            target_col = 'target'
        else:
            print("Erro: Coluna target não encontrada.")
            return
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    # 6. Avaliar importância das features usando o modelo
    importance_df = evaluate_feature_importance_with_model(
        model, X_train, y_train, 
        list(X_train.columns), 
        target_col
    )
    
    # 7. Selecionar top features com base na importância
    selected_features = select_top_features(
        importance_df, 
        current_model_features, 
        n_features=args.n_features
    )
    
    # 8. Visualizar e salvar importância das features
    plot_feature_importance(
        importance_df, 
        os.path.join(args.output_dir, 'model_selected_features'),
        top_n=50
    )
    
    # 9. Aplicar seleção de features aos datasets
    print("\nAplicando seleção de features aos datasets...")
    train_selected = apply_feature_selection(train_df, selected_features, target_col)
    
    if val_df is not None:
        val_selected = apply_feature_selection(val_df, selected_features, target_col)
    else:
        val_selected = None
        
    if test_df is not None:
        test_selected = apply_feature_selection(test_df, selected_features, target_col)
    else:
        test_selected = None
    
    # 10. Salvar datasets com features selecionadas
    train_selected.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    print(f"Dataset de treino com features selecionadas salvo em {os.path.join(args.output_dir, 'train.csv')}")
    
    if val_selected is not None:
        val_selected.to_csv(os.path.join(args.output_dir, 'validation.csv'), index=False)
        print(f"Dataset de validação com features selecionadas salvo em {os.path.join(args.output_dir, 'validation.csv')}")
    
    if test_selected is not None:
        test_selected.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
        print(f"Dataset de teste com features selecionadas salvo em {os.path.join(args.output_dir, 'test.csv')}")
    
    # 11. Salvar lista de features selecionadas
    feature_list_path = os.path.join(args.output_dir, 'model_selected_features', 'selected_features.txt')
    with open(feature_list_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print(f"Lista de features selecionadas salva em {feature_list_path}")
    
    # 12. Salvar metadados do processo
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_uri': model_uri,
        'model_features_count': len(current_model_features),
        'selected_features_count': len(selected_features),
        'new_features_count': len([f for f in selected_features if f not in current_model_features])
    }
    
    metadata_path = os.path.join(args.output_dir, 'model_selected_features', 'selection_metadata.joblib')
    joblib.dump(metadata, metadata_path)
    print(f"Metadados salvos em {metadata_path}")
    
    print("\nProcesso de seleção de features baseada no modelo concluído!")
    print(f"Total de {len(selected_features)} features selecionadas")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seleção de features baseada em modelo existente.")
    parser.add_argument("--input-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/02_4_processed_text_fn_code7"), 
                        help="Diretório contendo os arquivos de entrada (train.csv, validation.csv, test.csv)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/03_4_feature_selection_rf"), 
                        help="Diretório para salvar os arquivos processados")
    parser.add_argument("--mlflow-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/models/mlflow"), 
                        help="Diretório do MLflow tracking")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Caminho direto para o modelo salvo (opcional, usado se MLflow falhar)")
    parser.add_argument("--target-col", type=str, default="target",
                        help="Nome da coluna target")
    parser.add_argument("--n-features", type=int, default=200,
                        help="Número de features a serem selecionadas")
    
    args = parser.parse_args()
    
    main(args)