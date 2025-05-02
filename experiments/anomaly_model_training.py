#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para treinar e avaliar diferentes algoritmos de detecção de anomalias.

Este script:
1. Carrega os datasets preprocessados com features selecionadas
2. Treina diferentes detectores de anomalias (Isolation Forest, One-Class SVM, LOF)
3. Avalia desempenho com foco em atingir recall >= 50% mantendo precisão >= 70%
4. Salva o melhor modelo e gera relatórios detalhados
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Adicionar caminho do projeto ao PATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Constantes
RANDOM_STATE = 42
INPUT_DIR = os.path.join(project_root, "data/03_4_feature_selection_anomaly")
REPORTS_DIR = os.path.join(project_root, "reports")
MODELS_DIR = os.path.join(INPUT_DIR, "models")

# Configurações ajustáveis
MIN_PRECISION_TARGET = 0.7                # Precisão mínima desejada
MIN_RECALL_TARGET = 0.5                   # Recall mínimo desejado
CONTAMINATION_RANGE = [0.005, 0.01, 0.015, 0.02, 0.03]  # Valores de contaminação para testar
ISOLATION_FOREST_ESTIMATORS = [100, 200, 300]  # Número de estimadores para testar

# Criar diretórios necessários
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_processed_data(input_dir):
    """Carrega os conjuntos de dados já preprocessados."""
    train_path = os.path.join(input_dir, "train.csv")
    val_path = os.path.join(input_dir, "validation.csv")
    
    print(f"Carregando dados processados de {input_dir}...")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Verificar se existem valores NaN
    train_nan = train_df.isna().sum().sum()
    val_nan = val_df.isna().sum().sum()
    
    if train_nan > 0 or val_nan > 0:
        print(f"AVISO: Encontrados valores NaN: treino={train_nan}, validação={val_nan}")
        
        # Carregar imputer se disponível
        imputer_path = os.path.join(input_dir, "models", "feature_imputer.joblib")
        if os.path.exists(imputer_path):
            print("Carregando imputer para tratar valores ausentes...")
            imputer = joblib.load(imputer_path)
            
            # Separar features e target
            target_col = 'target'
            feature_cols = [col for col in train_df.columns if col != target_col]
            
            # Aplicar imputer
            train_df[feature_cols] = imputer.transform(train_df[feature_cols])
            val_df[feature_cols] = imputer.transform(val_df[feature_cols])
        else:
            print("Nenhum imputer encontrado. Preenchendo NaN com 0...")
            train_df = train_df.fillna(0)
            val_df = val_df.fillna(0)
    
    print(f"Dados carregados: treino={train_df.shape}, validação={val_df.shape}")
    
    return train_df, val_df


def custom_threshold_search(anomaly_scores, y_val, target_recall=0.5, min_precision=0.7, n_thresholds=1000):
    """
    Busca um threshold que atinja o recall desejado com pelo menos a precisão mínima.
    Se não encontrar, retorna o threshold com maior F1.
    """
    thresholds = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores), n_thresholds)
    best_f1 = 0
    best_threshold = None
    best_precision = 0
    best_recall = 0
    
    # Threshold que atinge o recall desejado com a maior precisão
    target_threshold = None
    target_precision = 0
    target_f1 = 0
    
    for threshold in thresholds:
        y_pred = (anomaly_scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='binary', zero_division=0
        )
        
        # Verificar se atinge o recall desejado com pelo menos a precisão mínima
        if recall >= target_recall and precision >= min_precision and precision > target_precision:
            target_threshold = threshold
            target_precision = precision
            target_f1 = f1
        
        # Também acompanhar o melhor F1 como fallback
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    # Se encontrou um threshold que atinge o recall desejado, use-o
    if target_threshold is not None:
        return target_threshold, target_precision, target_recall, target_f1
    
    # Caso contrário, retorne o threshold com melhor F1
    return best_threshold, best_precision, best_recall, best_f1


def train_isolation_forest(X_train_normal, X_val, y_val, name_prefix=""):
    """
    Treina e avalia modelos Isolation Forest com diferentes parâmetros.
    """
    results = []
    models = {}
    
    print("Treinando Isolation Forest...")
    
    for n_estimators in ISOLATION_FOREST_ESTIMATORS:
        for contamination in CONTAMINATION_RANGE:
            model_name = f"{name_prefix}isolation_forest_n{n_estimators}_c{contamination}"
            print(f"  Treinando {model_name}...")
            
            start_time = time.time()
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=RANDOM_STATE,
                n_estimators=n_estimators,
                n_jobs=-1
            )
            iso_forest.fit(X_train_normal)
            train_time = time.time() - start_time
            
            models[model_name] = iso_forest
            
            # Calcular scores de anomalia (valores mais altos = mais anômalo)
            anomaly_scores = -iso_forest.score_samples(X_val)
            
            # Buscar threshold otimizado para recall >= 50% e precisão >= 70%
            threshold, precision, recall, f1 = custom_threshold_search(
                anomaly_scores, y_val, 
                target_recall=MIN_RECALL_TARGET, 
                min_precision=MIN_PRECISION_TARGET
            )
            
            # Aplicar threshold e calcular métricas finais
            y_pred = (anomaly_scores >= threshold).astype(int)
            cm = confusion_matrix(y_val, y_pred)
            
            # Armazenar resultados
            result = {
                'model': model_name,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'threshold': threshold,
                'train_time': train_time,
                'positive_count': y_pred.sum(),
                'positive_rate': y_pred.sum() / len(y_pred),
                'true_positive': cm[1, 1] if cm.shape == (2, 2) else 0,
                'false_positive': cm[0, 1] if cm.shape == (2, 2) else 0,
                'false_negative': cm[1, 0] if cm.shape == (2, 2) else 0,
                'true_negative': cm[0, 0] if cm.shape == (2, 2) else 0
            }
            
            results.append(result)
            
            print(f"  {model_name} - F1: {f1:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}")
            if cm.shape == (2, 2):
                print(f"  {model_name} - TP: {cm[1, 1]}, FP: {cm[0, 1]}, FN: {cm[1, 0]}, TN: {cm[0, 0]}")
            else:
                print(f"  {model_name} - Matriz de confusão com forma inválida: {cm.shape}")
    
    return results, models


def train_one_class_svm(X_train_normal, X_val, y_val, name_prefix=""):
    """
    Treina e avalia modelos One-Class SVM com diferentes parâmetros.
    """
    results = []
    models = {}
    
    print("\nTreinando One-Class SVM...")
    nu_values = [0.01, 0.02, 0.05]
    gamma_values = ['scale', 'auto']
    
    for nu in nu_values:
        for gamma in gamma_values:
            model_name = f"{name_prefix}ocsvm_nu{nu}_gamma{gamma}"
            print(f"  Treinando {model_name}...")
            
            try:
                start_time = time.time()
                ocsvm = OneClassSVM(nu=nu, gamma=gamma, kernel='rbf')
                ocsvm.fit(X_train_normal)
                train_time = time.time() - start_time
                
                models[model_name] = ocsvm
                
                # Calcular scores de anomalia
                anomaly_scores = -ocsvm.decision_function(X_val)
                
                # Buscar threshold otimizado
                threshold, precision, recall, f1 = custom_threshold_search(
                    anomaly_scores, y_val, 
                    target_recall=MIN_RECALL_TARGET, 
                    min_precision=MIN_PRECISION_TARGET
                )
                
                # Aplicar threshold e calcular métricas finais
                y_pred = (anomaly_scores >= threshold).astype(int)
                cm = confusion_matrix(y_val, y_pred)
                
                # Armazenar resultados
                result = {
                    'model': model_name,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'threshold': threshold,
                    'train_time': train_time,
                    'positive_count': y_pred.sum(),
                    'positive_rate': y_pred.sum() / len(y_pred),
                    'true_positive': cm[1, 1] if cm.shape == (2, 2) else 0,
                    'false_positive': cm[0, 1] if cm.shape == (2, 2) else 0,
                    'false_negative': cm[1, 0] if cm.shape == (2, 2) else 0,
                    'true_negative': cm[0, 0] if cm.shape == (2, 2) else 0
                }
                
                results.append(result)
                
                print(f"  {model_name} - F1: {f1:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}")
                if cm.shape == (2, 2):
                    print(f"  {model_name} - TP: {cm[1, 1]}, FP: {cm[0, 1]}, FN: {cm[1, 0]}, TN: {cm[0, 0]}")
                else:
                    print(f"  {model_name} - Matriz de confusão com forma inválida: {cm.shape}")
                
            except Exception as e:
                print(f"  Erro ao treinar {model_name}: {str(e)}")
    
    return results, models


def train_local_outlier_factor(X_train_normal, X_val, y_val, name_prefix=""):
    """
    Treina e avalia modelos Local Outlier Factor com diferentes parâmetros.
    """
    results = []
    models = {}
    
    print("\nTreinando Local Outlier Factor...")
    n_neighbors_values = [20, 50, 100]
    contamination_values = [0.01, 0.02, 0.05]
    
    for n_neighbors in n_neighbors_values:
        for contamination in contamination_values:
            model_name = f"{name_prefix}lof_n{n_neighbors}_c{contamination}"
            print(f"  Treinando {model_name}...")
            
            try:
                start_time = time.time()
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=contamination,
                    novelty=True,
                    n_jobs=-1
                )
                lof.fit(X_train_normal)
                train_time = time.time() - start_time
                
                models[model_name] = lof
                
                # Calcular scores de anomalia
                anomaly_scores = -lof.decision_function(X_val)
                
                # Buscar threshold otimizado
                threshold, precision, recall, f1 = custom_threshold_search(
                    anomaly_scores, y_val, 
                    target_recall=MIN_RECALL_TARGET, 
                    min_precision=MIN_PRECISION_TARGET
                )
                
                # Aplicar threshold e calcular métricas finais
                y_pred = (anomaly_scores >= threshold).astype(int)
                cm = confusion_matrix(y_val, y_pred)
                
                # Armazenar resultados
                result = {
                    'model': model_name,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'threshold': threshold,
                    'train_time': train_time,
                    'positive_count': y_pred.sum(),
                    'positive_rate': y_pred.sum() / len(y_pred),
                    'true_positive': cm[1, 1] if cm.shape == (2, 2) else 0,
                    'false_positive': cm[0, 1] if cm.shape == (2, 2) else 0,
                    'false_negative': cm[1, 0] if cm.shape == (2, 2) else 0,
                    'true_negative': cm[0, 0] if cm.shape == (2, 2) else 0
                }
                
                results.append(result)
                
                print(f"  {model_name} - F1: {f1:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}")
                if cm.shape == (2, 2):
                    print(f"  {model_name} - TP: {cm[1, 1]}, FP: {cm[0, 1]}, FN: {cm[1, 0]}, TN: {cm[0, 0]}")
                else:
                    print(f"  {model_name} - Matriz de confusão com forma inválida: {cm.shape}")
                
            except Exception as e:
                print(f"  Erro ao treinar {model_name}: {str(e)}")
    
    return results, models


def train_anomaly_detectors(train_df, val_df, name_prefix=""):
    """
    Treina diferentes detectores de anomalias e avalia seu desempenho.
    """
    print("\nPreparando dados para detectores de anomalias...")
    
    # Separar features e target
    target_col = 'target'
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Preparar os dados
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    # Dados de treino com apenas amostras negativas (não-conversões)
    X_train_normal = X_train[y_train == 0]
    print(f"Usando {X_train_normal.shape[0]} amostras negativas para treinamento")
    
    # Escalar os dados
    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)
    X_val_scaled = scaler.transform(X_val)
    
    # Salvar o scaler para uso futuro
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    # Dicionário para armazenar todos os resultados e modelos
    all_results = []
    all_models = {}
    
    # 1. Isolation Forest
    iso_results, iso_models = train_isolation_forest(
        X_train_normal_scaled, X_val_scaled, y_val, name_prefix
    )
    all_results.extend(iso_results)
    all_models.update(iso_models)
    
    # 2. One-Class SVM
    ocsvm_results, ocsvm_models = train_one_class_svm(
        X_train_normal_scaled, X_val_scaled, y_val, name_prefix
    )
    all_results.extend(ocsvm_results)
    all_models.update(ocsvm_models)
    
    # 3. Local Outlier Factor
    lof_results, lof_models = train_local_outlier_factor(
        X_train_normal_scaled, X_val_scaled, y_val, name_prefix
    )
    all_results.extend(lof_results)
    all_models.update(lof_models)
    
    # Converter resultados para DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Adicionar coluna indicando se o modelo atingiu os objetivos
    results_df['objetivo_atingido'] = ((results_df['precision'] >= MIN_PRECISION_TARGET) & 
                                       (results_df['recall'] >= MIN_RECALL_TARGET))
    
    # Calcular Taxa de Sucesso como % de casos atingindo os objetivos
    target_success = results_df['objetivo_atingido'].mean()
    print(f"\nTaxa de Sucesso (% modelos atingindo objetivos): {target_success*100:.1f}%")
    
    # Salvar o melhor modelo individual
    if not results_df.empty:
        try:
            best_model_idx = results_df['f1'].idxmax()
            best_model_row = results_df.iloc[best_model_idx]
            best_model_name = best_model_row['model']
            
            # Salvar melhor modelo individual
            best_model = all_models[best_model_name]
            joblib.dump(best_model, os.path.join(MODELS_DIR, f"{best_model_name}.joblib"))
            
            # Salvar parâmetros importantes
            params = {
                'model_type': best_model_name.split('_')[0],
                'threshold': float(best_model_row['threshold']),
                'precision': float(best_model_row['precision']),
                'recall': float(best_model_row['recall']),
                'f1': float(best_model_row['f1']),
                'train_time': float(best_model_row['train_time']),
                'min_precision_target': MIN_PRECISION_TARGET,
                'min_recall_target': MIN_RECALL_TARGET
            }
            
            with open(os.path.join(MODELS_DIR, f"{best_model_name}_params.json"), 'w') as f:
                json.dump(params, f, indent=4)
        
            print(f"Melhor modelo individual salvo: {best_model_name}")
            print(f"  F1: {best_model_row['f1']:.4f}, Precisão: {best_model_row['precision']:.4f}, Recall: {best_model_row['recall']:.4f}")
            
            # Também salvar o melhor modelo por tipo
            model_types = ['isolation_forest', 'ocsvm', 'lof']
            for model_type in model_types:
                type_models = results_df[results_df['model'].str.contains(model_type)]
                if not type_models.empty:
                    best_type_idx = type_models['f1'].idxmax()
                    best_type_row = results_df.iloc[best_type_idx]
                    best_type_name = best_type_row['model']
                    
                    if best_type_name != best_model_name:  # Evitar duplicatas
                        # Salvar melhor modelo deste tipo
                        joblib.dump(all_models[best_type_name], os.path.join(MODELS_DIR, f"{best_type_name}.joblib"))
                        
                        # Salvar parâmetros do modelo
                        type_params = {
                            'model_type': best_type_name.split('_')[0],
                            'threshold': float(best_type_row['threshold']),
                            'precision': float(best_type_row['precision']),
                            'recall': float(best_type_row['recall']),
                            'f1': float(best_type_row['f1']),
                            'train_time': float(best_type_row['train_time']),
                            'min_precision_target': MIN_PRECISION_TARGET,
                            'min_recall_target': MIN_RECALL_TARGET
                        }
                        
                        with open(os.path.join(MODELS_DIR, f"{best_type_name}_params.json"), 'w') as f:
                            json.dump(type_params, f, indent=4)
                        
                        print(f"Melhor modelo {model_type} salvo: {best_type_name}")
                        print(f"  F1: {best_type_row['f1']:.4f}, Precisão: {best_type_row['precision']:.4f}, "
                              f"Recall: {best_type_row['recall']:.4f}")
        except Exception as e:
            print(f"Erro ao salvar melhor modelo: {str(e)}")
    
    return results_df, all_models


def generate_model_reports(results_df, output_dir):
    """Gera relatórios visuais e HTML dos resultados dos modelos."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar CSV com resultados
    report_path = os.path.join(output_dir, f"anomaly_model_results_{timestamp}.csv")
    results_df.to_csv(report_path, index=False)
    
    # Criar relatório visual
    report_fig_path = os.path.join(output_dir, f"anomaly_model_visual_{timestamp}.png")
    
    try:
        plt.figure(figsize=(14, 12))
        
        # Subplot para precisão vs recall
        plt.subplot(2, 2, 1)
        scatter = sns.scatterplot(
            data=results_df, 
            x='recall', 
            y='precision', 
            hue='model', 
            s=100,
            style='objetivo_atingido'
        )
        
        # Ajustar legenda para não ficar muito grande
        handles, labels = scatter.get_legend_handles_labels()
        scatter.legend(handles[:8], labels[:8], loc='upper right')
        
        plt.title('Precisão vs Recall para Detectores de Anomalias')
        plt.axhline(y=MIN_PRECISION_TARGET, color='r', linestyle='--', 
                    label=f'Precisão Mínima = {MIN_PRECISION_TARGET}')
        plt.axvline(x=MIN_RECALL_TARGET, color='g', linestyle='--', 
                    label=f'Recall Mínimo = {MIN_RECALL_TARGET}')
        plt.grid(True, alpha=0.3)
        
        # Subplot para F1-score por modelo
        plt.subplot(2, 2, 2)
        model_types = results_df['model'].apply(lambda x: x.split('_')[0] if '_' in x else x)
        df_plot = results_df.copy()
        df_plot['model_type'] = model_types
        
        sns.boxplot(data=df_plot, x='model_type', y='f1')
        plt.title('F1-Score por Tipo de Modelo')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Subplot para distribuição de TP, FP, FN, TN
        plt.subplot(2, 2, 3)
        top_models = results_df.sort_values('f1', ascending=False).head(5)
        
        # Melting para formato longo
        melt_df = pd.melt(
            top_models,
            id_vars=['model'],
            value_vars=['true_positive', 'false_positive', 'false_negative', 'true_negative'],
            var_name='metric',
            value_name='count'
        )
        
        sns.barplot(data=melt_df, x='model', y='count', hue='metric')
        plt.title('Distribuição de TP, FP, FN, TN nos Top 5 Modelos')
        plt.xticks(rotation=45)
        plt.legend(loc='upper right')
        
        # Subplot para threshold vs F1
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=results_df, x='threshold', y='f1', hue='model', alpha=0.7)
        plt.title('Threshold vs F1-Score')
        
        plt.tight_layout()
        plt.savefig(report_fig_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Erro ao criar visualização: {str(e)}")
    
    # Criar relatório detalhado em HTML
    html_report = os.path.join(output_dir, f"anomaly_model_report_{timestamp}.html")
    
    try:
        # Filtrar para recuperar diferentes subconjuntos de modelos
        successful_models = results_df[results_df['objetivo_atingido']]
        best_f1 = results_df.sort_values('f1', ascending=False).head(5)
        best_precision = results_df[results_df['recall'] >= 0.25].sort_values('precision', ascending=False).head(5)
        best_recall = results_df[results_df['precision'] >= 0.5].sort_values('recall', ascending=False).head(5)
        
        # Estatísticas para o relatório
        stats = {
            'total_models': len(results_df),
            'successful_models': len(successful_models),
            'success_rate': len(successful_models) / len(results_df) * 100 if len(results_df) > 0 else 0,
            'best_f1': results_df['f1'].max() if not results_df.empty else 0,
            'best_precision': results_df['precision'].max() if not results_df.empty else 0,
            'best_recall': results_df['recall'].max() if not results_df.empty else 0,
        }
        
        # Tabela de resultados por tipo de modelo
        model_types = {'isolation_forest': 'Isolation Forest', 
                      'ocsvm': 'One-Class SVM', 
                      'lof': 'Local Outlier Factor'}
        
        type_stats = []
        for key, name in model_types.items():
            type_df = results_df[results_df['model'].str.contains(key)]
            if not type_df.empty:
                type_stats.append({
                    'tipo': name,
                    'total': len(type_df),
                    'media_f1': type_df['f1'].mean(),
                    'max_f1': type_df['f1'].max(),
                    'media_precision': type_df['precision'].mean(),
                    'media_recall': type_df['recall'].mean(),
                    'sucesso': type_df['objetivo_atingido'].mean() * 100,
                })
        
        type_stats_df = pd.DataFrame(type_stats)
        
        html_content = f"""
        <html>
        <head>
            <title>Relatório de Modelos de Detecção de Anomalias</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .header {{ background-color: #4CAF50; color: white; padding: 15px; }}
                .section {{ margin-top: 20px; }}
                .summary {{ background-color: #e7f3fe; border-left: 6px solid #2196F3; padding: 10px; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Relatório de Modelos de Detecção de Anomalias</h1>
                <p>Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section summary">
                <h2>Resumo</h2>
                <p>Total de modelos avaliados: <b>{stats['total_models']}</b></p>
                <p>Modelos atingindo os objetivos (Precisão ≥ {MIN_PRECISION_TARGET}, Recall ≥ {MIN_RECALL_TARGET}): 
                   <b class="{('success' if stats['success_rate'] >= 50 else 'warning')}">{stats['successful_models']} ({stats['success_rate']:.1f}%)</b></p>
                <p>Melhor F1-Score: <b>{stats['best_f1']:.4f}</b></p>
                <p>Melhor Precisão: <b>{stats['best_precision']:.4f}</b></p>
                <p>Melhor Recall: <b>{stats['best_recall']:.4f}</b></p>
            </div>
            
            <div class="section">
                <h2>Desempenho por Tipo de Modelo</h2>
                {type_stats_df.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x)}
            </div>
            
            <div class="section">
                <h2>Modelos que Atingiram os Objetivos</h2>
                {successful_models.to_html(index=False) if not successful_models.empty else "<p>Nenhum modelo atingiu os objetivos definidos.</p>"}
            </div>
            
            <div class="section">
                <h2>Melhores Modelos por Métrica</h2>
                
                <h3>Melhor F1-Score</h3>
                {best_f1.to_html(index=False) if not best_f1.empty else "<p>Sem dados disponíveis.</p>"}
                
                <h3>Melhor Precisão (com Recall > 25%)</h3>
                {best_precision.to_html(index=False) if not best_precision.empty else "<p>Sem dados disponíveis.</p>"}
                
                <h3>Melhor Recall (com Precisão > 50%)</h3>
                {best_recall.to_html(index=False) if not best_recall.empty else "<p>Sem dados disponíveis.</p>"}
            </div>
            
            <div class="section">
                <h2>Todos os Resultados</h2>
                {results_df.to_html(index=False)}
            </div>
        </body>
        </html>
        """
        
        with open(html_report, 'w') as f:
            f.write(html_content)
    except Exception as e:
        print(f"Erro ao criar relatório HTML: {str(e)}")
    
    print(f"\nRelatórios salvos em:")
    print(f"  CSV: {report_path}")
    print(f"  Visualização: {report_fig_path}")
    print(f"  HTML: {html_report}")


def create_predictor_script():
    """Cria um script para fazer previsões usando o modelo treinado."""
    script_path = os.path.join(INPUT_DIR, "predict_anomalies.py")
    
    script_content = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Script para fazer previsões usando os modelos de detecção de anomalias treinados.

Este script carrega o melhor modelo e faz previsões em novos dados.
\"\"\"

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json

def load_model_and_params(models_dir):
    \"\"\"
    Carrega modelo e parâmetros necessários para previsão.
    \"\"\"
    # Verificar se o diretório existe
    if not os.path.exists(models_dir):
        raise ValueError(f"Diretório de modelos não encontrado: {models_dir}")
    
    # Procurar pelo melhor modelo individual
    json_files = [f for f in os.listdir(models_dir) if f.endswith('_params.json')]
    
    if not json_files:
        raise ValueError("Nenhum arquivo de parâmetros encontrado")
    
    # Ordenar para pegar o melhor modelo (assumindo que os nomes de arquivo são consistentes)
    json_file = sorted(json_files)[0]
    
    # Carregar parâmetros do modelo
    params_path = os.path.join(models_dir, json_file)
    model_name = json_file.replace('_params.json', '')
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Carregar modelo
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise ValueError(f"Modelo {model_name} não encontrado")
    
    model = joblib.load(model_path)
    
    # Carregar scaler
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        raise ValueError("Scaler não encontrado")
    
    # Carregar imputer
    imputer_path = os.path.join(models_dir, "feature_imputer.joblib")
    if os.path.exists(imputer_path):
        imputer = joblib.load(imputer_path)
    else:
        print("Aviso: Imputer não encontrado. Será usado preenchimento com 0.")
        imputer = None
    
    return {
        'model': model,
        'model_name': model_name,
        'params': params,
        'scaler': scaler,
        'imputer': imputer
    }

def predict_anomalies(data, model_info, feature_cols):
    \"\"\"
    Faz previsões usando o modelo carregado.
    
    Args:
        data: DataFrame com os dados para previsão
        model_info: Dicionário com informações do modelo
        feature_cols: Lista de features usadas pelo modelo
    
    Returns:
        DataFrame com as previsões
    \"\"\"
    # Verificar features disponíveis
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        print(f"Aviso: {len(missing_features)} features estão faltando nos dados: {missing_features[:5]}")
        
    # Selecionar apenas as features relevantes disponíveis
    available_features = [col for col in feature_cols if col in data.columns]
    X = data[available_features].copy()
    
    # Preencher valores NaN (usando imputer ou mediana)
    if model_info['imputer'] is not None:
        X_filled = model_info['imputer'].transform(X)
    else:
        X_filled = X.fillna(0).values
    
    # Normalizar os dados
    X_scaled = model_info['scaler'].transform(X_filled)
    
    # Fazer previsões
    model = model_info['model']
    threshold = model_info['params']['threshold']
    
    # Calcular scores de anomalia
    if hasattr(model, 'score_samples'):
        anomaly_scores = -model.score_samples(X_scaled)
    else:
        anomaly_scores = -model.decision_function(X_scaled)
    
    # Aplicar threshold
    predictions = (anomaly_scores >= threshold).astype(int)
    
    # Criar DataFrame de resultado
    result_df = data.copy()
    result_df['anomaly_score'] = anomaly_scores
    result_df['anomaly_prediction'] = predictions
    
    # Estatísticas básicas
    n_anomalies = predictions.sum()
    total = len(predictions)
    anomaly_rate = n_anomalies / total if total > 0 else 0
    
    print(f"Previsões concluídas: {n_anomalies} anomalias detectadas em {total} amostras ({anomaly_rate:.2%})")
    
    return result_df

def main():
    # Caminhos para os diretórios e arquivos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    features_path = os.path.join(base_dir, "selected_features.csv")
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print(f"Uso: {sys.argv[0]} <caminho_para_dados>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        print(f"Erro: Arquivo de dados não encontrado: {data_path}")
        sys.exit(1)
    
    # Carregar lista de features
    if os.path.exists(features_path):
        feature_cols = pd.read_csv(features_path).iloc[:, 0].tolist()
    else:
        print("Aviso: Lista de features não encontrada. Serão usadas todas as colunas numéricas.")
        feature_cols = None
    
    # Carregar modelo e parâmetros
    try:
        model_info = load_model_and_params(models_dir)
        print(f"Modelo carregado: {model_info['model_name']}")
        print(f"Threshold: {model_info['params']['threshold']:.4f}")
        print(f"Métricas no conjunto de validação: F1={model_info['params']['f1']:.4f}, "
              f"Precision={model_info['params']['precision']:.4f}, Recall={model_info['params']['recall']:.4f}")
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        sys.exit(1)
    
    # Carregar dados
    try:
        data = pd.read_csv(data_path)
        print(f"Dados carregados: {data.shape[0]} amostras, {data.shape[1]} colunas")
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        sys.exit(1)
    
    # Se não temos lista de features, usar todas as colunas numéricas
    if feature_cols is None:
        feature_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        print(f"Usando {len(feature_cols)} colunas numéricas como features")
    else:
        print(f"Usando {len(feature_cols)} features pré-selecionadas")
    
    # Fazer previsões
    result_df = predict_anomalies(data, model_info, feature_cols)
    
    # Salvar resultados
    output_path = data_path.replace('.csv', '_anomalias.csv')
    if output_path == data_path:
        output_path = os.path.join(os.path.dirname(data_path), f"anomalias_{os.path.basename(data_path)}")
    
    result_df.to_csv(output_path, index=False)
    print(f"Resultados salvos em: {output_path}")

if __name__ == "__main__":
    main()
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Tornar executável no Unix
    try:
        import stat
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)
    except:
        pass
    
    print(f"\nScript de previsão criado: {script_path}")
    print(f"Use: python {script_path} <caminho_para_dados>")


def main():
    """Função principal do script."""
    start_time = time.time()
    
    # 1. Carregar dados preprocessados
    train_df, val_df = load_processed_data(INPUT_DIR)
    
    # 2. Treinar e avaliar modelos de detecção de anomalias
    name_prefix = "anomaly_"  # Prefixo para identificar modelos
    results_df, models = train_anomaly_detectors(train_df, val_df, name_prefix)
    
    # 3. Gerar relatórios
    generate_model_reports(results_df, REPORTS_DIR)
    
    # 4. Criar script de previsão
    create_predictor_script()
    
    total_time = time.time() - start_time
    print(f"\nProcesso de treinamento concluído em {total_time/60:.2f} minutos.")
    print(f"Todos os modelos estão em: {MODELS_DIR}")


if __name__ == "__main__":
    main()