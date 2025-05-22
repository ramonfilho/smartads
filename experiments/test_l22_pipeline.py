#!/usr/bin/env python
"""
Script para testar a pipeline em dados novos do L22.
Realiza coleta, integração, inferência e avaliação de métricas
para simular o desempenho do modelo em produção.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Adicionar o diretório raiz do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# IMPORTANTE: Adicionar GMM_Wrapper ao namespace builtins para desserialização
from src.modeling.gmm_wrapper import GMM_Wrapper
import builtins
builtins.GMM_Wrapper = GMM_Wrapper  # Esta linha é crucial para o carregamento do modelo

# Importar módulos necessários
from src.preprocessing.local_storage import (
    connect_to_gcs, load_csv_or_excel, load_csv_with_auto_delimiter
)
from src.preprocessing.email_processing import normalize_emails_in_dataframe
from src.preprocessing.data_matching import match_surveys_with_buyers
from src.preprocessing.data_integration import create_target_variable, merge_datasets

# Importar a pipeline de inferência
from inference_v4.inference_pipeline import process_inference

# Configuração de caminhos
RAW_DATA_DIR = os.path.join(project_root, "data/00_raw_data")
OUTPUT_DIR = os.path.join(project_root, "data/L22_test")
RESULTS_DIR = os.path.join(project_root, "reports/L22_test_results")

# Arquivo de saída para dados processados e predições
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "L22_processed.csv")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "L22_predictions.csv")

# Arquivos específicos do L22
L22_FILES = {
    "survey": "Pesquisa_L22.xlsx",
    "buyers": "compradores_mario_L22.xlsx",
    "utm": "L22_UTMS.csv"
}

def find_email_column(df):
    """Encontra a coluna que contém emails em um DataFrame."""
    email_patterns = ['email', 'e-mail', 'correo', '@', 'mail']
    for col in df.columns:
        if any(pattern in col.lower() for pattern in email_patterns):
            return col
    return None

def process_l22_files():
    """
    Carrega e processa os arquivos específicos do L22.
    """
    print("\n=== Processando arquivos do lançamento L22 ===\n")
    
    # Verificar a existência dos arquivos
    for file_type, file_name in L22_FILES.items():
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"ERRO: Arquivo {file_type} não encontrado: {file_path}")
            return None, None, None
    
    # Criar conexão local (apenas para compatibilidade com funções existentes)
    bucket = connect_to_gcs("local_bucket")
    
    # Carregar arquivo de pesquisa
    print(f"Carregando dados de pesquisa: {L22_FILES['survey']}")
    survey_path = os.path.join(RAW_DATA_DIR, L22_FILES['survey'])
    surveys_df = load_csv_or_excel(bucket, survey_path)
    
    # Encontrar e renomear coluna de email na pesquisa
    email_col = find_email_column(surveys_df)
    if email_col and email_col != 'email':
        surveys_df = surveys_df.rename(columns={email_col: 'email'})
        print(f"  Coluna de email renomeada: {email_col} -> email")
    
    # Adicionar identificador de lançamento
    surveys_df['lançamento'] = 'L22'
    print(f"  Dados de pesquisa: {surveys_df.shape[0]} linhas, {surveys_df.shape[1]} colunas")
    
    # Carregar arquivo de compradores
    print(f"\nCarregando dados de compradores: {L22_FILES['buyers']}")
    buyers_path = os.path.join(RAW_DATA_DIR, L22_FILES['buyers'])
    buyers_df = load_csv_or_excel(bucket, buyers_path)
    
    # Encontrar e renomear coluna de email nos compradores
    email_col = find_email_column(buyers_df)
    if email_col and email_col != 'email':
        buyers_df = buyers_df.rename(columns={email_col: 'email'})
        print(f"  Coluna de email renomeada: {email_col} -> email")
    
    # Adicionar identificador de lançamento
    buyers_df['lançamento'] = 'L22'
    print(f"  Dados de compradores: {buyers_df.shape[0]} linhas, {buyers_df.shape[1]} colunas")
    
    # Carregar arquivo de UTM
    print(f"\nCarregando dados de UTM: {L22_FILES['utm']}")
    utm_path = os.path.join(RAW_DATA_DIR, L22_FILES['utm'])
    utm_df = load_csv_with_auto_delimiter(bucket, utm_path)
    
    # Encontrar e renomear coluna de email no UTM
    email_col = find_email_column(utm_df)
    if email_col and email_col != 'email':
        utm_df = utm_df.rename(columns={email_col: 'email'})
        print(f"  Coluna de email renomeada: {email_col} -> email")
    
    # Adicionar identificador de lançamento
    utm_df['lançamento'] = 'L22'
    print(f"  Dados de UTM: {utm_df.shape[0]} linhas, {utm_df.shape[1]} colunas")
    
    return surveys_df, buyers_df, utm_df

def integrate_l22_data(surveys_df, buyers_df, utm_df):
    """
    Integra os dados do L22 para criar um dataset consolidado.
    """
    print("\n=== Integrando dados do L22 ===\n")
    
    # 1. Normalizar emails
    print("Normalizando endereços de email...")
    surveys_df = normalize_emails_in_dataframe(surveys_df)
    buyers_df = normalize_emails_in_dataframe(buyers_df)
    if 'email' in utm_df.columns:
        utm_df = normalize_emails_in_dataframe(utm_df)
    
    # 2. Correspondência de pesquisas com dados de compradores
    print("\nRealizando correspondência entre pesquisas e compradores...")
    matches_df = match_surveys_with_buyers(surveys_df, buyers_df)
    print(f"  Correspondências encontradas: {len(matches_df)}")
    
    # 3. Criar variável alvo
    print("\nCriando variável alvo...")
    surveys_with_target = create_target_variable(surveys_df, matches_df)
    target_counts = surveys_with_target['target'].value_counts(dropna=False)
    print(f"  Distribuição de target: {target_counts.to_dict()}")
    
    # 4. Mesclar datasets
    print("\nMesclando datasets...")
    merged_data = merge_datasets(surveys_with_target, utm_df, buyers_df)
    print(f"  Dataset final: {merged_data.shape[0]} linhas, {merged_data.shape[1]} colunas")
    
    # Verificar se o resultado parece correto
    if merged_data.shape[0] == 0:
        print("ERRO: Nenhum dado resultante após integração. Verifique as colunas de email.")
        return None
    
    # Criar diretório de output se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Salvar dados processados
    merged_data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nDados integrados salvos em: {PROCESSED_DATA_PATH}")
    
    return merged_data

def run_inference_pipeline(data_df):
    """
    Executa a pipeline de inferência nos dados integrados.
    """
    print("\n=== Executando pipeline de inferência ===\n")
    
    # Verificar se temos dados
    if data_df is None or data_df.empty:
        print("ERRO: Sem dados para inferência.")
        return None
    
    # Chamada para a função de inferência
    print(f"Processando {data_df.shape[0]} registros...")
    start_time = datetime.now()
    
    # Executar a pipeline de inferência
    predictions_df = process_inference(
        data_df,
        params=None,  # Carregará automaticamente
        return_proba=True,
        threshold=None,  # Usar o valor padrão calibrado
        use_calibrated=True
    )
    
    if predictions_df is None:
        print("ERRO: Falha na pipeline de inferência.")
        return None
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"Inferência concluída em {elapsed_time:.2f} segundos.")
    
    # Salvar predições
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predições salvas em: {PREDICTIONS_PATH}")
    
    return predictions_df

def evaluate_predictions(predictions_df):
    """
    Avalia as predições contra os valores reais.
    """
    print("\n=== Avaliando predições ===\n")
    
    # Verificar se temos predições
    if predictions_df is None or 'prediction_class' not in predictions_df.columns:
        print("ERRO: Predições não disponíveis para avaliação.")
        return
    
    # Verificar se temos a variável alvo
    if 'target' not in predictions_df.columns:
        print("ERRO: Variável alvo 'target' não encontrada nos dados.")
        return
    
    # Criar diretório para resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Estatísticas básicas
    total_samples = len(predictions_df)
    positives = predictions_df['target'].sum()
    predicted_positives = predictions_df['prediction_class'].sum()
    
    print(f"Total de registros: {total_samples}")
    print(f"Positivos reais: {positives} ({positives/total_samples*100:.2f}%)")
    print(f"Positivos preditos: {predicted_positives} ({predicted_positives/total_samples*100:.2f}%)")
    
    # Calcular métricas
    y_true = predictions_df['target']
    y_pred = predictions_df['prediction_class']
    y_prob = predictions_df['prediction_probability']
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\nMétricas de desempenho:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusão:")
    print(cm)
    
    # Salvar resultados em CSV
    results_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1', 'True Positives', 'False Positives', 
                  'False Negatives', 'True Negatives', 'Positives', 'Predictions'],
        'Value': [precision, recall, f1, cm[1,1], cm[0,1], cm[1,0], cm[0,0], 
                 positives, predicted_positives]
    })
    
    results_path = os.path.join(RESULTS_DIR, "l22_metrics.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nMétricas salvas em: {results_path}")
    
    # Criar visualizações
    create_visualizations(y_true, y_pred, y_prob)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }

def create_visualizations(y_true, y_pred, y_prob):
    """
    Cria visualizações para analisar as predições.
    """
    # 1. Distribuição de probabilidades
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob, bins=50, alpha=0.7)
    plt.axvline(x=0.21, color='r', linestyle='--', label='Threshold (0.21)')
    plt.title("Distribuição de Probabilidades Preditas")
    plt.xlabel("Probabilidade")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "probability_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Análise por decis
    deciles = np.percentile(y_prob, np.arange(0, 101, 10))
    decile_labels = [f"D{i+1}" for i in range(10)]
    
    # Criar dataframe para análise por decil
    decile_df = pd.DataFrame(columns=['decile', 'range_min', 'range_max', 'count', 'positives', 'conversion_rate'])
    
    for i in range(9):
        mask = (y_prob >= deciles[i]) & (y_prob < deciles[i+1])
        count = mask.sum()
        positives = y_true[mask].sum()
        conv_rate = positives / count if count > 0 else 0
        
        decile_df.loc[i] = {
            'decile': decile_labels[i],
            'range_min': deciles[i],
            'range_max': deciles[i+1],
            'count': count,
            'positives': positives,
            'conversion_rate': conv_rate
        }
    
    # Adicionar o último decil
    mask = (y_prob >= deciles[9])
    count = mask.sum()
    positives = y_true[mask].sum() 
    conv_rate = positives / count if count > 0 else 0
    
    decile_df.loc[9] = {
        'decile': decile_labels[9],
        'range_min': deciles[9],
        'range_max': max(y_prob),
        'count': count,
        'positives': positives, 
        'conversion_rate': conv_rate
    }
    
    # Ordenar por decil
    decile_df = decile_df.sort_values('decile')
    
    # Salvar análise por decil
    decile_path = os.path.join(RESULTS_DIR, "decile_analysis.csv")
    decile_df.to_csv(decile_path, index=False)
    print(f"Análise por decil salva em: {decile_path}")
    
    # Gráfico de taxa de conversão por decil
    plt.figure(figsize=(12, 6))
    bars = plt.bar(decile_df['decile'], decile_df['conversion_rate'] * 100)
    
    # Adicionar labels de contagem
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 2, 
                 f"n={int(decile_df.iloc[i]['count'])}", 
                 ha='center', color='white', fontweight='bold')
        
        # Adicionar taxa de conversão no topo
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f"{decile_df.iloc[i]['conversion_rate']*100:.2f}%", 
                 ha='center', fontweight='bold')
    
    plt.title("Taxa de Conversão por Decil de Probabilidade")
    plt.xlabel("Decil")
    plt.ylabel("Taxa de Conversão (%)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "conversion_by_decile.png"), dpi=300)
    plt.close()
    
    # 3. Matriz de confusão visual
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    # Calcular percentuais
    cm_pct = cm / cm.sum() * 100
    
    # Plot da matriz
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    
    # Adicionar labels
    classes = ["Não Comprou", "Comprou"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Adicionar números
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()
    
    print(f"Visualizações salvas em: {RESULTS_DIR}")

def main():
    """Função principal para execução do teste L22."""
    print("=== Teste da Pipeline de Inferência com Dados do L22 ===\n")
    
    # Criar diretórios de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Carregar e processar arquivos do L22
    surveys_df, buyers_df, utm_df = process_l22_files()
    if surveys_df is None or buyers_df is None:
        print("ERRO: Falha ao processar arquivos do L22. Abortando.")
        return 1
    
    # 2. Integrar dados
    integrated_data = integrate_l22_data(surveys_df, buyers_df, utm_df)
    if integrated_data is None:
        print("ERRO: Falha na integração dos dados. Abortando.")
        return 1
    
    # 3. Executar pipeline de inferência
    predictions_df = run_inference_pipeline(integrated_data)
    if predictions_df is None:
        print("ERRO: Falha na pipeline de inferência. Abortando.")
        return 1
    
    # 4. Avaliar resultados
    evaluate_predictions(predictions_df)
    
    print("\n=== Teste da Pipeline Concluído com Sucesso ===")
    print(f"Dados processados: {PROCESSED_DATA_PATH}")
    print(f"Predições: {PREDICTIONS_PATH}")
    print(f"Resultados: {RESULTS_DIR}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())