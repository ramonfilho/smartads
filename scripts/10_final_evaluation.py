#!/usr/bin/env python
"""
Script para avaliar a calibração e performance de dois modelos no conjunto de teste:
1. RF com valores ausentes não tratados nas últimas features
2. GMM

Utiliza as mesmas funções de sanitização de nomes do módulo baseline_model.py.
"""

import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

# Adicionar o diretório raiz do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# Importar a classe GMM_Wrapper do módulo compartilhado
from src.modeling.gmm_wrapper import GMM_Wrapper

# Caminhos absolutos
BASE_DIR = "/Users/ramonmoreira/desktop/smart_ads"
DATA_DIR_GMM = "/Users/ramonmoreira/desktop/smart_ads/data/04_feature_engineering_2"
DATA_DIR_RF_UNTREATED = "/Users/ramonmoreira/desktop/smart_ads/data/05_feature_selection"
TEST_PATH_GMM = os.path.join(DATA_DIR_GMM, "test.csv")
TEST_PATH_RF_UNTREATED = os.path.join(DATA_DIR_RF_UNTREATED, "test.csv")

# Caminhos para os modelos calibrados
RF_UNTREATED_CALIB_DIR = "/Users/ramonmoreira/Desktop/smart_ads/models/calibrated/rf_calibrated_20250509_071445"
GMM_CALIB_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250518_152543"

# Diretório para salvar resultados
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "calibration_validation_two_models")

# Funções de tratamento de dados do baseline_model.py
def sanitize_column_names(df):
    """
    Sanitiza os nomes das colunas para evitar problemas com caracteres especiais.
    
    Args:
        df: Pandas DataFrame com colunas para sanitizar
        
    Returns:
        Dictionary mapping original column names to sanitized names
    """
    sanitized_columns = {}
    for col in df.columns:
        new_col = re.sub(r'[^\w\s]', '_', col)
        new_col = re.sub(r'\s+', '_', new_col)
        if new_col in sanitized_columns.values():
            new_col = f"{new_col}_{df.columns.get_loc(col)}"
        sanitized_columns[col] = new_col
    df.rename(columns=sanitized_columns, inplace=True)
    return sanitized_columns

def convert_integer_columns_to_float(df):
    """
    Converte colunas inteiras para float para compatibilidade com alguns modelos.
    
    Args:
        df: DataFrame para converter
        
    Returns:
        List of column names that were converted
    """
    print("Convertendo colunas inteiras para float...")
    integer_columns = []
    
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col].dtype):
            df[col] = df[col].astype(float)
            integer_columns.append(col)
    
    return integer_columns

def prepare_data_for_rf_model(model, df, target_col="target"):
    """
    Prepara os dados para serem compatíveis com o modelo RandomForest.
    Usa as funções de sanitização de baseline_model.py.
    """
    print("Preparando dados para o modelo RandomForest...")
    
    # Extrair features e target
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    y = df[target_col] if target_col in df.columns else None
    
    # Aplicar a mesma sanitização de nomes de colunas usada no treinamento
    print("Sanitizando nomes das colunas...")
    sanitize_column_names(X)
    
    # Converter inteiros para float
    convert_integer_columns_to_float(X)
    
    # Verificar features do modelo
    if hasattr(model, 'feature_names_in_'):
        expected_features = set(model.feature_names_in_)
        actual_features = set(X.columns)
        
        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features
        
        if missing_features:
            print(f"AVISO: Faltam {len(missing_features)} features que o modelo espera")
            print(f"  Exemplos: {list(missing_features)[:5]}")
            
            # Criar DataFrame vazio com as colunas corretas para minimizar a fragmentação
            missing_cols = list(missing_features)
            missing_dict = {col: [0] * len(X) for col in missing_cols}
            missing_df = pd.DataFrame(missing_dict)
            
            # Concatenar em vez de adicionar uma por uma
            X = pd.concat([X, missing_df], axis=1)
        
        if extra_features:
            print(f"AVISO: Removendo {len(extra_features)} features extras")
            print(f"  Exemplos: {list(extra_features)[:5]}")
            X = X.drop(columns=list(extra_features))
        
        # Garantir a ordem correta das colunas
        X = X[model.feature_names_in_]
    
    # Double-check para garantir que todas as colunas ainda são numéricas
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col].dtype):
            print(f"Convertendo coluna não-numérica para float: {col}")
            X[col] = X[col].astype(float)
            
    # Verificar por valores NaN
    if X.isna().any().any():
        print("Substituindo valores NaN por zeros...")
        X = X.fillna(0)
    
    return X, y

def load_rf_model(model_dir, model_type_name):
    """
    Carrega um modelo RandomForest calibrado.
    
    Args:
        model_dir: Diretório do modelo
        model_type_name: Nome/descrição do tipo de modelo (para logs)
        
    Returns:
        Dict com modelo e threshold ou None se houver erro
    """
    model_path = os.path.join(model_dir, "rf_calibrated.joblib")
    threshold_path = os.path.join(model_dir, "threshold.txt")
    
    try:
        # Carregar modelo
        loaded_data = joblib.load(model_path)
        
        # Verificar se o modelo carregado é um dicionário
        if isinstance(loaded_data, dict) and 'model' in loaded_data:
            # Se for um dicionário com chave 'model', use o modelo dentro dele
            rf_model = loaded_data['model']
            print(f"Modelo RF {model_type_name} extraído do dicionário carregado.")
        else:
            # Caso contrário, assumir que é o modelo diretamente
            rf_model = loaded_data
        
        # Verificar se o modelo carregado tem o método predict_proba
        if not hasattr(rf_model, 'predict_proba'):
            print(f"AVISO: Objeto carregado para RF {model_type_name} não parece ser um modelo válido.")
            print(f"Tipo do objeto: {type(rf_model)}")
            if isinstance(rf_model, dict):
                print(f"Chaves disponíveis: {list(rf_model.keys())}")
                # Tentar encontrar o modelo real em alguma chave do dicionário
                for key, value in rf_model.items():
                    if hasattr(value, 'predict_proba'):
                        rf_model = value
                        print(f"Modelo encontrado na chave '{key}'.")
                        break
        
        # Tentar carregar threshold do arquivo, se falhar, usar valor fixo de 0.09
        try:
            with open(threshold_path, 'r') as f:
                rf_threshold = float(f.read().strip())
        except (FileNotFoundError, IOError):
            print(f"Arquivo de threshold para RF {model_type_name} não encontrado. Usando valor fixo de 0.09.")
            rf_threshold = 0.09
        
        # Verificar novamente se temos um modelo válido
        if hasattr(rf_model, 'predict_proba'):
            print(f"Random Forest {model_type_name} carregado com threshold: {rf_threshold:.4f}")
            return {'model': rf_model, 'threshold': rf_threshold}
        else:
            print(f"ERRO: Não foi possível encontrar um modelo RandomForest {model_type_name} válido.")
            return None
    except Exception as e:
        print(f"Erro ao carregar Random Forest {model_type_name}: {e}")
        return None

def load_gmm_model(model_dir):
    """
    Carrega um modelo GMM calibrado.
    
    Args:
        model_dir: Diretório do modelo
        
    Returns:
        Dict com modelo e threshold ou None se houver erro
    """
    model_path = os.path.join(model_dir, "gmm_calibrated.joblib")
    threshold_path = os.path.join(model_dir, "threshold.txt")
    
    try:
        gmm_model = joblib.load(model_path)
        with open(threshold_path, 'r') as f:
            gmm_threshold = float(f.read().strip())
        print(f"GMM calibrado carregado com threshold: {gmm_threshold:.4f}")
        return {'model': gmm_model, 'threshold': gmm_threshold}
    except Exception as e:
        print(f"Erro ao carregar GMM: {e}")
        return None

def load_models():
    """
    Carrega os dois modelos calibrados.
    """
    print("Carregando modelos calibrados...")
    models = {}
    
    # Random Forest com valores ausentes não tratados
    rf_untreated = load_rf_model(RF_UNTREATED_CALIB_DIR, "VALORES AUSENTES NÃO TRATADOS")
    if rf_untreated:
        models['RF_Untreated'] = rf_untreated
    
    # GMM
    gmm_model = load_gmm_model(GMM_CALIB_DIR)
    if gmm_model:
        models['GMM'] = gmm_model
    
    return models

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calcula o Expected Calibration Error (ECE).
    """
    # Criar bins de probabilidade
    bin_indices = np.digitize(y_prob, np.linspace(0, 1, n_bins + 1)) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Inicializar arrays para estatísticas
    bin_accs = np.zeros(n_bins)  # Accuracy média por bin
    bin_confs = np.zeros(n_bins)  # Confiança média por bin
    bin_sizes = np.zeros(n_bins)  # Número de amostras por bin
    
    # Calcular estatísticas por bin
    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        if np.any(mask):
            bin_sizes[bin_idx] = mask.sum()
            bin_accs[bin_idx] = y_true[mask].mean()
            bin_confs[bin_idx] = y_prob[mask].mean()
    
    # Calcular ECE
    ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_sizes / len(y_true)))
    
    return ece, {
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_sizes': bin_sizes
    }

def plot_calibration_curve(y_true, y_prob, model_name, output_dir):
    """
    Plota a curva de calibração.
    """
    plt.figure(figsize=(10, 6))
    
    # Calcular curva de calibração
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    # Calcular ECE
    ece, _ = calculate_ece(y_true, y_prob)
    
    # Plotar curva
    plt.plot(prob_pred, prob_true, "s-", label=f'{model_name} (ECE={ece:.4f})')
    plt.plot([0, 1], [0, 1], "k--", label="Perfeitamente Calibrado")
    
    plt.xlabel('Probabilidade Média Predita')
    plt.ylabel('Fração de Positivos')
    plt.title(f'Curva de Calibração - {model_name}')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    # Salvar figura
    plt.savefig(os.path.join(output_dir, f"calibration_curve_{model_name}.png"))
    plt.close()
    
    return ece

def plot_combined_calibration_curves(models_results, output_dir):
    """
    Plota as curvas de calibração de todos os modelos juntos para comparação.
    
    Args:
        models_results: Dicionário com resultados de cada modelo
        output_dir: Diretório para salvar o gráfico
    """
    plt.figure(figsize=(12, 8))
    
    # Linha de referência (calibração perfeita)
    plt.plot([0, 1], [0, 1], "k--", label="Perfeitamente Calibrado")
    
    # Cores para cada modelo
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Plotar cada modelo
    for i, (model_name, result) in enumerate(models_results.items()):
        if 'y_true' in result and 'y_prob' in result:
            # Calcular curva de calibração
            prob_true, prob_pred = calibration_curve(result['y_true'], result['y_prob'], n_bins=10)
            
            # Plotar curva
            color = colors[i % len(colors)]
            plt.plot(prob_pred, prob_true, "s-", color=color, 
                     label=f'{model_name} (ECE={result["ece"]:.4f})')
    
    plt.xlabel('Probabilidade Média Predita', fontsize=12)
    plt.ylabel('Fração de Positivos', fontsize=12)
    plt.title('Comparação das Curvas de Calibração', fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Salvar figura
    plt.savefig(os.path.join(output_dir, "combined_calibration_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_and_plot_decile_analysis(y_true, y_prob, model_name, output_dir):
    """
    Calcula e plota a análise por decil de probabilidade.
    
    Args:
        y_true: Array com valores reais
        y_prob: Array com probabilidades preditas
        model_name: Nome do modelo
        output_dir: Diretório para salvar resultados
        
    Returns:
        DataFrame com análise por decil
    """
    # Criar decis de probabilidade
    deciles = np.percentile(y_prob, np.arange(0, 101, 10))
    
    # Garantir que o último decil inclua o valor máximo
    deciles[-1] = deciles[-1] + 0.0001
    
    # Inicializar dataframe para armazenar resultados
    decile_df = pd.DataFrame(columns=[
        'Decil', 'Limite_Inferior', 'Limite_Superior', 'Num_Amostras', 
        'Num_Positivos', 'Taxa_Positivos', 'Prob_Media'
    ])
    
    # Calcular estatísticas para cada decil
    for i in range(len(deciles) - 1):
        lower = deciles[i]
        upper = deciles[i+1]
        
        # Selecionar amostras neste decil
        mask = (y_prob >= lower) & (y_prob < upper)
        
        # Calcular estatísticas
        n_samples = mask.sum()
        n_positive = y_true[mask].sum()
        positive_rate = n_positive / n_samples if n_samples > 0 else 0
        mean_prob = y_prob[mask].mean() if n_samples > 0 else 0
        
        # Adicionar linha ao dataframe
        decile_df.loc[i] = {
            'Decil': i+1,
            'Limite_Inferior': lower,
            'Limite_Superior': upper,
            'Num_Amostras': n_samples,
            'Num_Positivos': n_positive,
            'Taxa_Positivos': positive_rate,
            'Prob_Media': mean_prob
        }
    
    # Salvar estatísticas
    decile_df.to_csv(os.path.join(output_dir, f"{model_name}_decile_analysis.csv"), index=False)
    
    # Plotar análise por decil
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(decile_df['Decil'], decile_df['Taxa_Positivos'], alpha=0.7, label='Taxa de Positivos')
    
    # Adicionar linha de probabilidade média
    plt.plot(decile_df['Decil'], decile_df['Prob_Media'], 'ro-', label='Probabilidade Média')
    
    # Adicionar labels dentro das barras com o número de amostras
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                 f"n={int(decile_df.loc[i, 'Num_Amostras'])}", 
                 ha='center', va='center', color='white', fontweight='bold')
    
    plt.xlabel('Decil de Probabilidade')
    plt.ylabel('Taxa de Positivos / Probabilidade Média')
    plt.title(f'Análise por Decil de Probabilidade - {model_name}')
    plt.xticks(decile_df['Decil'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar figura
    plt.savefig(os.path.join(output_dir, f"{model_name}_decile_analysis.png"))
    plt.close()
    
    return decile_df

def evaluate_rf_model(model_info, test_df, model_name, output_dir):
    """
    Avalia um modelo RandomForest.
    
    Args:
        model_info: Dicionário com modelo e threshold
        test_df: DataFrame de teste
        model_name: Nome do modelo
        output_dir: Diretório para salvar resultados
        
    Returns:
        Dicionário com métricas e resultados
    """
    print(f"\nAvaliando {model_name}...")
    
    rf_model = model_info['model']
    rf_threshold = model_info['threshold']
    
    # Preparar dados
    X_test, y_test = prepare_data_for_rf_model(rf_model, test_df)
    
    print(f"Gerando probabilidades para {len(X_test)} instâncias...")
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= rf_threshold).astype(int)
    
    # Calcular métricas
    rf_precision = precision_score(y_test, rf_preds)
    rf_recall = recall_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_ap = average_precision_score(y_test, rf_probs)  # Precision-recall AUC
    
    print(f"Métricas do {model_name}:")
    print(f"  Precision: {rf_precision:.4f}")
    print(f"  Recall: {rf_recall:.4f}")
    print(f"  F1 Score: {rf_f1:.4f}")
    print(f"  AUC-ROC: {rf_auc:.4f}")
    print(f"  Average Precision: {rf_ap:.4f}")
    
    # Avaliar calibração
    print(f"Avaliando calibração do {model_name}...")
    rf_ece = plot_calibration_curve(y_test, rf_probs, model_name, output_dir)
    print(f"Expected Calibration Error (ECE): {rf_ece:.4f}")
    
    # Análise por decis
    print(f"Realizando análise por decil para {model_name}...")
    decile_df = calculate_and_plot_decile_analysis(y_test, rf_probs, model_name, output_dir)
    
    # Salvar resultados
    rf_results = pd.DataFrame({
        'true': y_test,
        'prediction': rf_preds,
        'probability': rf_probs
    })
    rf_results.to_csv(os.path.join(output_dir, f"{model_name}_test_results.csv"), index=False)
    
    return {
        'precision': rf_precision,
        'recall': rf_recall,
        'f1': rf_f1,
        'auc': rf_auc,
        'ap': rf_ap,
        'ece': rf_ece,
        'threshold': rf_threshold,
        'y_true': y_test,
        'y_prob': rf_probs,
        'y_pred': rf_preds,
        'decile_analysis': decile_df
    }

def evaluate_gmm_model(model_info, test_df, output_dir):
    """
    Avalia o modelo GMM.
    
    Args:
        model_info: Dicionário com modelo e threshold
        test_df: DataFrame de teste
        output_dir: Diretório para salvar resultados
        
    Returns:
        Dicionário com métricas e resultados
    """
    print("\nAvaliando GMM calibrado...")
    
    gmm_model = model_info['model']
    gmm_threshold = model_info['threshold']
    
    # Separar features e target (preservando email para uso posterior)
    X_gmm_test = test_df.drop(columns=['target'])
    y_gmm_test = test_df['target']
    
    # Preservar o email ou outro identificador se existir
    has_email = 'email' in test_df.columns
    has_email_norm = 'email_norm' in test_df.columns
    
    print(f"Gerando probabilidades para {len(X_gmm_test)} instâncias...")
    gmm_probs = gmm_model.predict_proba(X_gmm_test)[:, 1]
    gmm_preds = (gmm_probs >= gmm_threshold).astype(int)
    
    # Calcular métricas
    gmm_precision = precision_score(y_gmm_test, gmm_preds)
    gmm_recall = recall_score(y_gmm_test, gmm_preds)
    gmm_f1 = f1_score(y_gmm_test, gmm_preds)
    gmm_auc = roc_auc_score(y_gmm_test, gmm_probs)
    gmm_ap = average_precision_score(y_gmm_test, gmm_probs)
    
    print("Métricas do GMM:")
    print(f"  Precision: {gmm_precision:.4f}")
    print(f"  Recall: {gmm_recall:.4f}")
    print(f"  F1 Score: {gmm_f1:.4f}")
    print(f"  AUC-ROC: {gmm_auc:.4f}")
    print(f"  Average Precision: {gmm_ap:.4f}")
    
    # Avaliar calibração
    print("Avaliando calibração do GMM...")
    gmm_ece = plot_calibration_curve(y_gmm_test, gmm_probs, "GMM", output_dir)
    print(f"Expected Calibration Error (ECE): {gmm_ece:.4f}")
    
    # Análise por decis
    print("Realizando análise por decil para GMM...")
    decile_df = calculate_and_plot_decile_analysis(y_gmm_test, gmm_probs, "GMM", output_dir)
    
    # Salvar resultados
    gmm_results = pd.DataFrame({
        'true': y_gmm_test,
        'prediction': gmm_preds,
        'probability': gmm_probs
    })
    
    # Adicionar email ou outro identificador se disponível
    if has_email:
        gmm_results['email'] = test_df['email'].values
    if has_email_norm:
        gmm_results['email_norm'] = test_df['email_norm'].values
    
    # Salvar com os identificadores incluídos
    gmm_results.to_csv(os.path.join(output_dir, "gmm_test_results.csv"), index=False)
    print(f"Resultados salvos em: {os.path.join(output_dir, 'gmm_test_results.csv')}")
    
    return {
        'precision': gmm_precision,
        'recall': gmm_recall,
        'f1': gmm_f1,
        'auc': gmm_auc,
        'ap': gmm_ap,
        'ece': gmm_ece,
        'threshold': gmm_threshold,
        'y_true': y_gmm_test,
        'y_prob': gmm_probs,
        'y_pred': gmm_preds,
        'decile_analysis': decile_df
    }

def main():
    # Criar diretório para resultados com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {results_dir}")
    
    # Carregar modelos
    models = load_models()
    
    # Dicionário para armazenar resultados de cada modelo
    model_results = {}
    
    # Avaliar Random Forest com valores ausentes não tratados
    if 'RF_Untreated' in models:
        print("\nCarregando dados de teste para RF com valores ausentes não tratados...")
        rf_untreated_test_df = pd.read_csv(TEST_PATH_RF_UNTREATED)
        model_results['RF_Untreated'] = evaluate_rf_model(
            models['RF_Untreated'], 
            rf_untreated_test_df, 
            "RF_Untreated", 
            results_dir
        )
    
    # Avaliar GMM
    if 'GMM' in models:
        print("\nCarregando dados de teste para GMM...")
        gmm_test_df = pd.read_csv(TEST_PATH_GMM)
        model_results['GMM'] = evaluate_gmm_model(models['GMM'], gmm_test_df, results_dir)
    
    # Criar tabela comparativa
    metrics = ['precision', 'recall', 'f1', 'auc', 'ap', 'ece', 'threshold']
    metrics_labels = ['Precision', 'Recall', 'F1', 'AUC-ROC', 'Average Precision', 'ECE', 'Threshold']
    
    metrics_df = pd.DataFrame({'Metric': metrics_labels})
    
    # Adicionar resultados de cada modelo
    for model_name, results in model_results.items():
        metrics_df[model_name] = [results.get(metric, 'N/A') for metric in metrics]
    
    # Formatar métricas para maior clareza
    for model_name in model_results.keys():
        for i, metric in enumerate(metrics):
            if metric != 'threshold' and metric in model_results[model_name]:
                metrics_df.loc[i, model_name] = f"{metrics_df.loc[i, model_name]:.4f}"
    
    # Salvar comparação
    metrics_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    print(f"\nComparação de modelos salva em: {os.path.join(results_dir, 'model_comparison.csv')}")
    
    # Criar gráfico comparativo de calibração
    plot_combined_calibration_curves(model_results, results_dir)
    
    # Exportar resultados detalhados em formato JSON
    results_summary = {
        'timestamp': timestamp,
        'model_paths': {
            'RF_Untreated': RF_UNTREATED_CALIB_DIR,
            'GMM': GMM_CALIB_DIR
        },
        'data_paths': {
            'RF_Untreated': TEST_PATH_RF_UNTREATED,
            'GMM': TEST_PATH_GMM
        },
        'metrics': {}
    }
    
    # Adicionar métricas sem os arrays grandes
    for model_name, results in model_results.items():
        results_summary['metrics'][model_name] = {
            metric: results[metric] 
            for metric in ['precision', 'recall', 'f1', 'auc', 'ap', 'ece', 'threshold']
            if metric in results
        }
    
    # Salvar relatório em JSON
    with open(os.path.join(results_dir, "results_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("\nAvaliação de modelos concluída!")
    print(f"Resultados e visualizações salvas em: {results_dir}")
    
    # Imprimir tabela comparativa para visualização imediata
    print("\n=== TABELA COMPARATIVA DE MÉTRICAS ===")
    print(metrics_df.to_string(index=False))
    print("\n=== CONCLUSÃO ===")
    
    # Determinar qual modelo tem melhor desempenho em cada métrica
    best_models = {}
    for metric in ['precision', 'recall', 'f1', 'auc', 'ap']:
        best_value = 0
        best_model = None
        for model_name, results in model_results.items():
            if metric in results and results[metric] > best_value:
                best_value = results[metric]
                best_model = model_name
        if best_model:
            best_models[metric] = (best_model, best_value)
    
    # Imprimir conclusões
    print("Melhores modelos por métrica:")
    for metric, (model, value) in best_models.items():
        print(f"- {metric.upper()}: {model} ({value:.4f})")
    
    # Verificar qual modelo é melhor em calibração (menor ECE)
    best_calibration = None
    best_ece = float('inf')
    for model_name, results in model_results.items():
        if 'ece' in results and results['ece'] < best_ece:
            best_ece = results['ece']
            best_calibration = model_name
    
    if best_calibration:
        print(f"- Melhor calibração (menor ECE): {best_calibration} ({best_ece:.4f})")

if __name__ == "__main__":
    main()