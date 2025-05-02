#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para selecionar features otimizadas para detecção de anomalias.

Este script:
1. Carrega o conjunto de dados completo
2. Prepara os dados tratando valores não-numéricos e NaN
3. Seleciona features otimizadas para detecção de anomalias
4. Salva os novos datasets com features selecionadas e imputação
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Adicionar caminho do projeto ao PATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Constantes
RANDOM_STATE = 42
INPUT_DIR = os.path.join(project_root, "data/02_3_processed_text_code6")
OUTPUT_DIR = os.path.join(project_root, "data/03_4_feature_selection_anomaly")
REPORTS_DIR = os.path.join(project_root, "reports")

# Configurações
N_FEATURES_TO_SELECT = 200  # Número aproximado de features para selecionar
SAMPLE_SIZE_FOR_FEATURE_IMPORTANCE = 5000  # Quantas amostras usar para calcular importância

# Criar diretórios necessários
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data(input_dir):
    """Carrega os conjuntos de dados de treino, validação e teste."""
    train_path = os.path.join(input_dir, "train.csv")
    val_path = os.path.join(input_dir, "validation.csv")
    test_path = os.path.join(input_dir, "test.csv")
    
    print(f"Carregando dados de {input_dir}...")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Dados carregados: treino={train_df.shape}, validação={val_df.shape}, teste={test_df.shape}")
    
    return train_df, val_df, test_df


def prepare_data_for_modeling(df):
    """
    Prepara os dados para modelagem, identificando e tratando diferentes tipos de colunas.
    
    Retorna:
        - X_numeric: features numéricas
        - feature_names: nomes das features numéricas
        - y: target
    """
    # Identificar a coluna target
    target_col = 'target' if 'target' in df.columns else None
    
    if target_col is None:
        raise ValueError("Coluna target não encontrada no DataFrame")
    
    # Copiar o DataFrame para evitar modificar o original
    df_copy = df.copy()
    
    # Separar target
    y = df_copy[target_col]
    df_copy = df_copy.drop(columns=[target_col])
    
    # Detectar e remover colunas de data e texto (não-numéricas)
    non_numeric_cols = []
    for col in df_copy.columns:
        # Verificar se a coluna é não-numérica
        if df_copy[col].dtype == 'object':
            # Tentar converter para numérico
            try:
                pd.to_numeric(df_copy[col])
            except:
                non_numeric_cols.append(col)
    
    print(f"Detectadas {len(non_numeric_cols)} colunas não-numéricas que serão removidas")
    
    # Remover colunas não-numéricas
    X = df_copy.drop(columns=non_numeric_cols)
    
    # Converter todas as colunas restantes para float
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Preencher valores NaN com a mediana
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Converter de volta para DataFrame
    X_numeric = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Lista de nomes de features
    feature_names = X_numeric.columns.tolist()
    
    print(f"Dataset preparado com {len(feature_names)} features numéricas após imputação de valores ausentes")
    
    return X_numeric, feature_names, y


def select_features_for_anomaly(X_train, y_train, feature_names, n_features=200):
    """
    Seleciona features otimizadas para detecção de anomalias usando múltiplas técnicas.
    
    Abordagens:
    1. Variância - features com alta variância
    2. Informação Mútua - relação não-linear com o target
    3. Importância baseada em Isolation Forest para anomalias
    """
    print(f"\nSelecionando features para detecção de anomalias...")
    
    # Escalar os dados para análises baseadas em distância
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Número de features a selecionar por método
    n_per_method = n_features // 3
    
    results = {}
    selected_features = []
    
    # 1. Seleção por Variância
    print("Calculando variância das features...")
    variances = np.var(X_scaled, axis=0)
    var_indices = np.argsort(variances)[::-1][:n_per_method]
    var_features = [feature_names[i] for i in var_indices]
    results['variance'] = var_features
    
    # 2. Seleção por Informação Mútua
    print("Calculando informação mútua com o target...")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    mi_indices = np.argsort(mi_scores)[::-1][:n_per_method]
    mi_features = [feature_names[i] for i in mi_indices]
    results['mutual_info'] = mi_features
    
    # 3. Seleção baseada em Isolation Forest para features de anomalias
    print("Treinando Isolation Forest para selecionar features...")
    X_normal = X_scaled[y_train == 0]  # Apenas amostras negativas
    iso_forest = IsolationForest(
        random_state=RANDOM_STATE, 
        contamination=0.015,
        n_estimators=100
    )
    iso_forest.fit(X_normal)
    
    # Calcular desvio do score de anomalia quando cada feature é permutada
    print("Calculando importância de features para anomalias...")
    anomaly_scores = -iso_forest.score_samples(X_scaled)
    feature_importance = np.zeros(X_scaled.shape[1])
    
    # Limitar número de amostras para acelerar
    n_samples = min(SAMPLE_SIZE_FOR_FEATURE_IMPORTANCE, X_scaled.shape[0])
    sample_indices = np.random.choice(X_scaled.shape[0], n_samples, replace=False)
    X_sample = X_scaled[sample_indices]
    base_scores = -iso_forest.score_samples(X_sample)
    
    for i in range(X_scaled.shape[1]):
        if i % 100 == 0:
            print(f"Processando feature {i} de {X_scaled.shape[1]}...")
        
        X_permuted = X_sample.copy()
        np.random.shuffle(X_permuted[:, i])
        permuted_scores = -iso_forest.score_samples(X_permuted)
        
        # A importância é o quanto o score muda quando a feature é permutada
        feature_importance[i] = np.mean(np.abs(permuted_scores - base_scores))
    
    # Selecionar features com maior importância para anomalias
    iso_indices = np.argsort(feature_importance)[::-1][:n_per_method]
    iso_features = [feature_names[i] for i in iso_indices]
    results['isolation_forest'] = iso_features
    
    # Combinar todas as features selecionadas
    for method, features in results.items():
        selected_features.extend(features)
    
    # Remover duplicatas mantendo a ordem
    selected_features = list(dict.fromkeys(selected_features))
    
    # Truncar para o número desejado de features
    selected_features = selected_features[:n_features]
    
    # Salvar info de feature importance para relatório
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'variance': variances,
        'mutual_info': mi_scores,
        'isolation_forest': feature_importance
    })
    
    # Ordenar por média de ranks
    feature_importance_df['variance_rank'] = feature_importance_df['variance'].rank(ascending=False)
    feature_importance_df['mutual_info_rank'] = feature_importance_df['mutual_info'].rank(ascending=False)
    feature_importance_df['isolation_forest_rank'] = feature_importance_df['isolation_forest'].rank(ascending=False)
    feature_importance_df['avg_rank'] = (
        feature_importance_df['variance_rank'] + 
        feature_importance_df['mutual_info_rank'] + 
        feature_importance_df['isolation_forest_rank']
    ) / 3
    
    # Salvar importâncias
    importance_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    feature_importance_df.sort_values('avg_rank').to_csv(importance_path, index=False)
    
    print(f"Selecionadas {len(selected_features)} features para detecção de anomalias")
    print(f"Importância de features salva em {importance_path}")
    
    return selected_features, results


def save_selected_datasets(train_df, val_df, test_df, selected_features, output_dir):
    """Salva os conjuntos de dados com as features selecionadas e imputação de valores ausentes."""
    # Garantir que o target esteja incluso
    target_col = 'target'
    all_cols = selected_features + [target_col]
    
    print(f"\nPreparando conjuntos de dados com {len(selected_features)} features selecionadas...")
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Identificar quais colunas realmente existem nos DataFrames
    train_cols = [col for col in all_cols if col in train_df.columns]
    val_cols = [col for col in all_cols if col in val_df.columns]
    test_cols = [col for col in all_cols if col in test_df.columns]
    
    # Encontrar a interseção das colunas disponíveis em todos os conjuntos
    common_cols = list(set(train_cols) & set(val_cols) & set(test_cols))
    
    if target_col not in common_cols:
        common_cols.append(target_col)
    
    feature_cols = [col for col in common_cols if col != target_col]
    print(f"Processando {len(feature_cols)} colunas numéricas comuns entre os conjuntos de dados")
    
    # Extrair as colunas selecionadas
    train_selected = train_df[common_cols].copy()
    val_selected = val_df[common_cols].copy()
    test_selected = test_df[common_cols].copy()
    
    # Verificar valores NaN antes da imputação
    train_nan_count = train_selected[feature_cols].isna().sum().sum()
    val_nan_count = val_selected[feature_cols].isna().sum().sum()
    test_nan_count = test_selected[feature_cols].isna().sum().sum()
    
    print(f"Valores NaN antes da imputação: treino={train_nan_count}, validação={val_nan_count}, teste={test_nan_count}")
    
    # Aplicar imputação de valores ausentes usando a mediana
    imputer = SimpleImputer(strategy='median')
    
    # Treinar o imputer apenas no conjunto de treino
    imputer.fit(train_selected[feature_cols])
    
    # Aplicar em todos os conjuntos
    train_selected[feature_cols] = imputer.transform(train_selected[feature_cols])
    val_selected[feature_cols] = imputer.transform(val_selected[feature_cols])
    test_selected[feature_cols] = imputer.transform(test_selected[feature_cols])
    
    # Verificar se ainda existem valores NaN após imputação
    train_nan_count = train_selected[feature_cols].isna().sum().sum()
    val_nan_count = val_selected[feature_cols].isna().sum().sum()
    test_nan_count = test_selected[feature_cols].isna().sum().sum()
    
    print(f"Valores NaN após imputação: treino={train_nan_count}, validação={val_nan_count}, teste={test_nan_count}")
    
    # Salvar o imputer para uso futuro
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(imputer, os.path.join(models_dir, "feature_imputer.joblib"))
    
    # Salvar datasets
    train_selected.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_selected.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_selected.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # Salvar lista de features
    pd.Series(feature_cols).to_csv(os.path.join(output_dir, "selected_features.csv"), index=False)
    
    print(f"Conjuntos de dados com imputação de valores ausentes salvos em {output_dir}")
    
    return feature_cols


def create_feature_report(feature_cols, results_dict, output_dir):
    """Cria relatório sobre as features selecionadas."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"feature_selection_report_{timestamp}.html")
    
    # Organizar features por método de seleção
    feature_by_method = {}
    total_unique_features = set()
    
    for method, features in results_dict.items():
        feature_by_method[method] = features
        total_unique_features.update(features)
    
    # Gerar relatório HTML
    html_content = f"""
    <html>
    <head>
        <title>Relatório de Seleção de Features para Detecção de Anomalias</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .header {{ background-color: #4CAF50; color: white; padding: 15px; }}
            .section {{ margin-top: 20px; }}
            .summary {{ background-color: #e7f3fe; border-left: 6px solid #2196F3; padding: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Relatório de Seleção de Features para Detecção de Anomalias</h1>
            <p>Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="section summary">
            <h2>Resumo</h2>
            <p>Total de features selecionadas: <b>{len(feature_cols)}</b></p>
            <p>Total de features únicas consideradas por método:</p>
            <ul>
    """
    
    for method, features in feature_by_method.items():
        html_content += f"<li>{method}: <b>{len(features)}</b> features</li>\n"
    
    html_content += f"""
            </ul>
            <p>Total de features únicas em todos os métodos: <b>{len(total_unique_features)}</b></p>
        </div>
        
        <div class="section">
            <h2>Features por Método de Seleção</h2>
    """
    
    for method, features in feature_by_method.items():
        html_content += f"""
            <h3>{method}</h3>
            <table>
                <tr>
                    <th>#</th>
                    <th>Feature</th>
                </tr>
        """
        
        for i, feature in enumerate(features[:30]):  # Mostrar apenas 30 primeiras
            html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{feature}</td>
                </tr>
            """
        
        html_content += "</table>\n"
        
        if len(features) > 30:
            html_content += f"<p>... e mais {len(features) - 30} features.</p>\n"
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Todas as Features Selecionadas</h2>
            <table>
                <tr>
                    <th>#</th>
                    <th>Feature</th>
                </tr>
    """
    
    for i, feature in enumerate(feature_cols):
        html_content += f"""
            <tr>
                <td>{i+1}</td>
                <td>{feature}</td>
            </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Relatório de seleção de features salvo em: {report_path}")


def main():
    """Função principal do script."""
    start_time = time.time()
    
    # 1. Carregar dados
    train_df, val_df, test_df = load_data(INPUT_DIR)
    
    # 2. Preparar dados para modelagem (extraindo apenas features numéricas)
    X_train, feature_names, y_train = prepare_data_for_modeling(train_df)
    
    # 3. Selecionar features para detecção de anomalias
    selected_features, feature_info = select_features_for_anomaly(
        X_train, y_train, feature_names, n_features=N_FEATURES_TO_SELECT
    )
    
    # 4. Salvar datasets com features selecionadas e imputação de valores ausentes
    feature_cols = save_selected_datasets(train_df, val_df, test_df, selected_features, OUTPUT_DIR)
    
    # 5. Criar relatório sobre as features selecionadas
    create_feature_report(feature_cols, feature_info, REPORTS_DIR)
    
    total_time = time.time() - start_time
    print(f"\nProcesso de seleção de features concluído em {total_time/60:.2f} minutos.")
    print(f"Os novos datasets estão em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()