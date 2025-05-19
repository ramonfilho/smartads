#!/usr/bin/env python
"""
Script para analisar diferenças entre as features geradas durante treinamento e inferência,
e investigar os 33 casos de falsos negativos (referência positiva, pipeline negativa).

O script:
1. Executa a pipeline de inferência adaptada para salvar features intermediárias
2. Compara com os datasets de treinamento previamente salvos
3. Analisa detalhadamente os 33 falsos negativos
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# Definir caminhos
DATA_DIR = os.path.join(project_root, "data")
INFERENCE_DIR = os.path.join(project_root, "inference/output/feature_analysis")
REPORTS_DIR = os.path.join(project_root, "reports/feature_analysis")
PARAMS_DIR = os.path.join(project_root, "inference/params")
REFERENCE_PREDICTIONS = os.path.join(project_root, "reports/calibration_validation_two_models/20250518_153608/gmm_test_results.csv")

# Criar diretórios de saída
os.makedirs(INFERENCE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Importar módulos da pipeline
try:
    from inference.modules.script2_module import apply_script2_transformations
    from inference.modules.script3_module import apply_script3_transformations
    from inference.modules.script4_module import apply_script4_transformations
    from src.modeling.gmm_wrapper import GMM_Wrapper
except Exception as e:
    print(f"ERRO AO IMPORTAR MÓDULOS: {str(e)}")
    print(traceback.format_exc())
    sys.exit(1)

def apply_inference_pipeline_with_feature_saving(input_path):
    """
    Executa a pipeline de inferência, salvando os resultados intermediários
    de cada etapa para comparação.
    """
    print(f"Executando pipeline de inferência com salvamento de features...")
    print(f"Usando arquivo de entrada: {input_path}")
    
    # Carregar dados de entrada
    input_data = pd.read_csv(input_path)
    print(f"Dados carregados: {input_data.shape}")
    
    # Etapa 1: Pré-processamento básico (script 2)
    print("\n--- Etapa 1: Pré-processamento básico ---")
    params_path = os.path.join(PARAMS_DIR, "02_params.joblib")
    step1_result = apply_script2_transformations(input_data, params_path)
    
    # Salvar resultado da etapa 1
    step1_output = os.path.join(INFERENCE_DIR, "step1_result.csv")
    step1_result.to_csv(step1_output, index=False)
    print(f"Resultado da etapa 1 salvo em: {step1_output}")
    
    # Etapa 2: Processamento de texto avançado (script 3)
    print("\n--- Etapa 2: Processamento de texto avançado ---")
    params_path = os.path.join(PARAMS_DIR, "dummy.joblib")  # Caminho base apenas
    step2_result = apply_script3_transformations(step1_result, params_path)
    
    # Salvar resultado da etapa 2
    step2_output = os.path.join(INFERENCE_DIR, "step2_result.csv")
    step2_result.to_csv(step2_output, index=False)
    print(f"Resultado da etapa 2 salvo em: {step2_output}")
    
    # Etapa 3: Features de motivação profissional (script 4)
    print("\n--- Etapa 3: Features de motivação profissional ---")
    params_path = os.path.join(PARAMS_DIR, "04_params.joblib")
    step3_result = apply_script4_transformations(step2_result, params_path)
    
    # Salvar resultado da etapa 3
    step3_output = os.path.join(INFERENCE_DIR, "step3_result.csv")
    step3_result.to_csv(step3_output, index=False)
    print(f"Resultado da etapa 3 salvo em: {step3_output}")
    
    # Etapa 4: Aplicação de GMM e predição
    print("\n--- Etapa 4: Aplicação de GMM e predição ---")
    step4_result = apply_gmm_and_predict(step3_result, PARAMS_DIR)
    
    # Salvar resultado da etapa 4
    step4_output = os.path.join(INFERENCE_DIR, "step4_result.csv")
    step4_result.to_csv(step4_output, index=False)
    print(f"Resultado da etapa 4 salvo em: {step4_output}")
    
    return {
        'step1': step1_result,
        'step2': step2_result,
        'step3': step3_result,
        'step4': step4_result,
        'output_paths': {
            'step1': step1_output,
            'step2': step2_output,
            'step3': step3_output,
            'step4': step4_output
        }
    }

def apply_gmm_and_predict(df, params_dir):
    """
    Aplica o modelo GMM calibrado para fazer predições.
    Versão adaptada da pipeline original.
    """
    print("\n=== Aplicando GMM calibrado e fazendo predição ===")
    print(f"Processando {len(df)} amostras...")
    
    # Copiar o DataFrame para não modificar o original
    result_df = df.copy()
    
    try:
        # Caminhos para os modelos
        GMM_CALIB_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250518_152543"
        model_path = os.path.join(GMM_CALIB_DIR, "gmm_calibrated.joblib")
        threshold_path = os.path.join(GMM_CALIB_DIR, "threshold.txt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        # Carregar threshold
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold = float(f.read().strip())
            print(f"Threshold carregado: {threshold}")
        else:
            threshold = 0.1
            print(f"Arquivo de threshold não encontrado. Usando valor padrão: {threshold}")
        
        # Carregar o modelo calibrado
        print(f"Carregando modelo GMM calibrado de: {model_path}")
        calibrated_model = joblib.load(model_path)
        
        # Preparar dados
        X_test = result_df.copy()
        if 'target' in X_test.columns:
            X_test = X_test.drop(columns=['target'])
        
        # Fazer predições
        print(f"Gerando probabilidades para {len(X_test)} instâncias...")
        start_time = datetime.now()
        
        # Obter probabilidades
        probabilities = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Aplicar threshold para classes
        predictions = (probabilities >= threshold).astype(int)
        
        # Adicionar ao DataFrame
        result_df['probability'] = probabilities
        result_df['prediction'] = predictions
        
        # Calcular estatísticas
        prediction_counts = dict(zip(*np.unique(predictions, return_counts=True)))
        print(f"  Distribuição de predições: {prediction_counts}")
        
        if 1 in prediction_counts:
            positive_rate = prediction_counts[1] / len(df)
            print(f"  Taxa de positivos: {positive_rate:.4f} ({prediction_counts.get(1, 0)} de {len(df)})")
        
        # Tempo de processamento
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"Predições concluídas em {elapsed_time:.2f} segundos.")
        
    except Exception as e:
        print(f"  ERRO durante predição: {e}")
        print(traceback.format_exc())
        
        # Em caso de erro, adicionar colunas com valores default
        result_df['prediction'] = 0
        result_df['probability'] = 0.0
    
    return result_df

def compare_feature_sets(inference_features, training_features, step_name):
    """
    Compara conjuntos de features entre o ambiente de inferência e treinamento.
    
    Args:
        inference_features: DataFrame com features geradas na inferência
        training_features: DataFrame com features geradas no treinamento
        step_name: Nome da etapa (para relatórios)
        
    Returns:
        Dicionário com estatísticas da comparação
    """
    print(f"\n=== Comparando features da etapa {step_name} ===")
    
    # Identificar colunas comuns, exclusivas de cada conjunto
    inference_cols = set(inference_features.columns)
    training_cols = set(training_features.columns)
    
    common_cols = inference_cols.intersection(training_cols)
    inference_only_cols = inference_cols - training_cols
    training_only_cols = training_cols - inference_cols
    
    print(f"Total de features na inferência: {len(inference_cols)}")
    print(f"Total de features no treinamento: {len(training_cols)}")
    print(f"Features comuns: {len(common_cols)}")
    print(f"Features exclusivas da inferência: {len(inference_only_cols)}")
    print(f"Features exclusivas do treinamento: {len(training_only_cols)}")
    
    # Comparar distribuições estatísticas das features comuns
    stats_comparison = []
    
    # Limitar o número de features para análise detalhada
    max_features_to_analyze = 50
    selected_common_cols = list(common_cols)[:max_features_to_analyze]
    
    for col in selected_common_cols:
        try:
            if pd.api.types.is_numeric_dtype(inference_features[col]) and pd.api.types.is_numeric_dtype(training_features[col]):
                inf_mean = inference_features[col].mean()
                train_mean = training_features[col].mean()
                inf_std = inference_features[col].std()
                train_std = training_features[col].std()
                
                # Calcular diferença em desvios padrão
                if train_std > 0:
                    mean_diff_in_std = abs(inf_mean - train_mean) / train_std
                else:
                    mean_diff_in_std = float('inf') if inf_mean != train_mean else 0
                
                stats_comparison.append({
                    'feature': col,
                    'inference_mean': inf_mean,
                    'training_mean': train_mean,
                    'mean_diff': abs(inf_mean - train_mean),
                    'mean_diff_pct': abs(inf_mean - train_mean) / (abs(train_mean) + 1e-10) * 100,
                    'inference_std': inf_std,
                    'training_std': train_std,
                    'std_diff': abs(inf_std - train_std),
                    'mean_diff_in_std': mean_diff_in_std
                })
        except Exception as e:
            print(f"  Erro ao analisar feature {col}: {e}")
    
    # Criar DataFrame para análise
    if stats_comparison:
        stats_df = pd.DataFrame(stats_comparison)
        
        # Ordenar por diferença normalizada
        stats_df = stats_df.sort_values('mean_diff_in_std', ascending=False)
        
        # Salvar resultados
        comparison_path = os.path.join(REPORTS_DIR, f"{step_name}_feature_comparison.csv")
        stats_df.to_csv(comparison_path, index=False)
        print(f"Comparação detalhada salva em: {comparison_path}")
        
        # Identificar features com maiores diferenças
        diff_threshold = 1.0  # diferença de 1 desvio padrão
        high_diff_features = stats_df[stats_df['mean_diff_in_std'] > diff_threshold]
        print(f"Features com diferenças significativas: {len(high_diff_features)}")
        
        if len(high_diff_features) > 0:
            print("Top 5 features com maiores diferenças:")
            for _, row in high_diff_features.head(5).iterrows():
                print(f"  {row['feature']}: {row['mean_diff_in_std']:.2f} desvios padrão")
        
        # Gerar gráfico de diferenças
        plt.figure(figsize=(10, 6))
        top_n = min(20, len(stats_df))
        plt.barh(stats_df['feature'].head(top_n), stats_df['mean_diff_in_std'].head(top_n))
        plt.xlabel('Diferença em Desvios Padrão')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Features com Maiores Diferenças - {step_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, f"{step_name}_top_differences.png"))
        plt.close()
    else:
        print("Nenhuma feature numérica comum encontrada para comparação.")
        stats_df = pd.DataFrame()
    
    # Salvar listas de colunas
    with open(os.path.join(REPORTS_DIR, f"{step_name}_column_analysis.txt"), 'w') as f:
        f.write(f"=== Análise de Colunas - Etapa {step_name} ===\n\n")
        f.write(f"Total de features na inferência: {len(inference_cols)}\n")
        f.write(f"Total de features no treinamento: {len(training_cols)}\n")
        f.write(f"Features comuns: {len(common_cols)}\n")
        f.write(f"Features exclusivas da inferência: {len(inference_only_cols)}\n")
        f.write(f"Features exclusivas do treinamento: {len(training_only_cols)}\n\n")
        
        f.write("=== Features exclusivas da inferência ===\n")
        for col in sorted(inference_only_cols):
            f.write(f"{col}\n")
        
        f.write("\n=== Features exclusivas do treinamento ===\n")
        for col in sorted(training_only_cols):
            f.write(f"{col}\n")
    
    return {
        'common_cols': common_cols,
        'inference_only_cols': inference_only_cols,
        'training_only_cols': training_only_cols,
        'stats_df': stats_df
    }

def analyze_false_negatives(pipeline_predictions, reference_predictions):
    """
    Realiza uma análise detalhada dos falsos negativos (casos onde a referência
    previu positivo, mas a pipeline previu negativo).
    
    Args:
        pipeline_predictions: DataFrame com predições da pipeline
        reference_predictions: DataFrame com predições de referência
    """
    print("\n=== Analisando os Falsos Negativos ===")
    
    # Garantir que temos colunas de email para alinhamento
    if 'email' not in pipeline_predictions.columns or 'email' not in reference_predictions.columns:
        print("ERRO: Coluna 'email' não encontrada em um dos arquivos de predições")
        if 'email' not in pipeline_predictions.columns:
            print("  Colunas no arquivo da pipeline:", pipeline_predictions.columns.tolist())
        if 'email' not in reference_predictions.columns:
            print("  Colunas no arquivo de referência:", reference_predictions.columns.tolist())
        return None
    
    # Remover duplicações no email para evitar problemas
    pipeline_unique = pipeline_predictions.drop_duplicates(subset=['email'])
    reference_unique = reference_predictions.drop_duplicates(subset=['email'])
    
    # Alinhar os DataFrames
    print("Alinhando predições da pipeline e referência pelo email...")
    merged_df = pd.merge(
        pipeline_unique[['email', 'prediction', 'probability']],
        reference_unique[['email', 'prediction', 'probability']],
        on='email',
        how='inner',
        suffixes=('_pipeline', '_reference')
    )
    
    print(f"Dados alinhados: {merged_df.shape[0]} registros")
    
    # Identificar falsos negativos
    false_negatives = merged_df[
        (merged_df['prediction_reference'] == 1) & 
        (merged_df['prediction_pipeline'] == 0)
    ]
    
    num_false_negatives = len(false_negatives)
    print(f"Falsos negativos identificados: {num_false_negatives}")
    
    if num_false_negatives == 0:
        print("Nenhum falso negativo encontrado para análise.")
        return None
    
    # Salvar os falsos negativos para análise detalhada
    fn_path = os.path.join(REPORTS_DIR, "false_negatives.csv")
    false_negatives.to_csv(fn_path, index=False)
    print(f"Falsos negativos salvos em: {fn_path}")
    
    # Obter emails dos falsos negativos
    fn_emails = false_negatives['email'].tolist()
    
    # Obter features completas para os falsos negativos
    print("Extraindo features completas para os falsos negativos...")
    fn_features = pipeline_predictions[pipeline_predictions['email'].isin(fn_emails)]
    
    # Salvar features completas
    fn_features_path = os.path.join(REPORTS_DIR, "false_negatives_features.csv")
    fn_features.to_csv(fn_features_path, index=False)
    print(f"Features completas dos falsos negativos salvas em: {fn_features_path}")
    
    # Análise de probabilidades
    print("\nAnálise de probabilidades dos falsos negativos:")
    prob_stats = false_negatives['probability_pipeline'].describe()
    print(prob_stats)
    
    # Análise de proximidade ao threshold
    threshold = 0.21  # Threshold do modelo
    proximity_ranges = [
        (0.0, 0.05),
        (0.05, 0.1),
        (0.1, 0.15),
        (0.15, 0.19),
        (0.19, 0.205),
        (0.205, 0.21),  # Extramamente próximo ao threshold
        (0.21, 0.215),  # Logo acima do threshold
        (0.215, 0.22),
        (0.22, 0.25),
        (0.25, 0.3),
        (0.3, 1.0)
    ]
    
    proximity_counts = {}
    for lower, upper in proximity_ranges:
        count = ((false_negatives['probability_pipeline'] >= lower) & 
                 (false_negatives['probability_pipeline'] < upper)).sum()
        proximity_counts[f"{lower:.3f}-{upper:.3f}"] = count
    
    print("\nDistribuição de proximidade ao threshold:")
    for range_label, count in proximity_counts.items():
        print(f"  {range_label}: {count}")
    
    # Analisar a diferença entre as probabilidades da pipeline e referência
    false_negatives['prob_diff'] = false_negatives['probability_reference'] - false_negatives['probability_pipeline']
    
    print("\nDiferença entre probabilidades (referência - pipeline):")
    diff_stats = false_negatives['prob_diff'].describe()
    print(diff_stats)
    
    # Gráfico de dispersão das probabilidades
    plt.figure(figsize=(10, 6))
    plt.scatter(false_negatives['probability_pipeline'], 
                false_negatives['probability_reference'],
                alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold da Referência ({threshold})')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold da Pipeline ({threshold})')
    plt.xlabel('Probabilidade da Pipeline')
    plt.ylabel('Probabilidade da Referência')
    plt.title('Comparação de Probabilidades nos Falsos Negativos')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(REPORTS_DIR, "false_negatives_probabilities.png"))
    plt.close()
    
    # Histograma da diferença de probabilidades
    plt.figure(figsize=(10, 6))
    plt.hist(false_negatives['prob_diff'], bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Sem diferença')
    plt.xlabel('Diferença de Probabilidade (Referência - Pipeline)')
    plt.ylabel('Frequência')
    plt.title('Diferença de Probabilidades nos Falsos Negativos')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(REPORTS_DIR, "false_negatives_prob_diff_hist.png"))
    plt.close()
    
    return {
        'num_false_negatives': num_false_negatives,
        'fn_emails': fn_emails,
        'prob_stats': prob_stats,
        'proximity_counts': proximity_counts,
        'diff_stats': diff_stats
    }

def analyze_near_threshold_cases(pipeline_predictions, threshold=0.21, margin=0.05):
    """
    Analisa casos com probabilidades próximas ao threshold.
    
    Args:
        pipeline_predictions: DataFrame com predições da pipeline
        threshold: Threshold aplicado
        margin: Margem de proximidade ao threshold
    """
    print(f"\n=== Analisando Casos Próximos ao Threshold ({threshold}) ===")
    
    lower_bound = threshold - margin
    upper_bound = threshold + margin
    
    near_threshold = pipeline_predictions[
        (pipeline_predictions['probability'] >= lower_bound) & 
        (pipeline_predictions['probability'] <= upper_bound)
    ]
    
    num_near = len(near_threshold)
    print(f"Casos próximos ao threshold ({lower_bound:.3f} a {upper_bound:.3f}): {num_near}")
    
    if num_near == 0:
        print("Nenhum caso próximo ao threshold encontrado.")
        return None
    
    # Distribuição dos casos por faixa menor
    small_ranges = [
        (lower_bound, threshold - margin/4),
        (threshold - margin/4, threshold - margin/8),
        (threshold - margin/8, threshold),
        (threshold, threshold + margin/8),
        (threshold + margin/8, threshold + margin/4),
        (threshold + margin/4, upper_bound)
    ]
    
    range_counts = {}
    for i, (lower, upper) in enumerate(small_ranges):
        count = ((near_threshold['probability'] >= lower) & 
                 (near_threshold['probability'] < upper)).sum()
        range_counts[f"Faixa {i+1}: {lower:.3f}-{upper:.3f}"] = count
    
    print("\nDistribuição em faixas menores:")
    for range_label, count in range_counts.items():
        print(f"  {range_label}: {count}")
    
    # Salvar casos próximos ao threshold
    near_path = os.path.join(REPORTS_DIR, "near_threshold_cases.csv")
    near_threshold.to_csv(near_path, index=False)
    print(f"Casos próximos ao threshold salvos em: {near_path}")
    
    # Histograma das probabilidades próximas ao threshold
    plt.figure(figsize=(10, 6))
    plt.hist(near_threshold['probability'], bins=20, alpha=0.7)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Probabilidade')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Probabilidades Próximas ao Threshold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(REPORTS_DIR, "near_threshold_histogram.png"))
    plt.close()
    
    return {
        'num_near_threshold': num_near,
        'range_counts': range_counts
    }

def main():
    # Caminho para o arquivo de teste
    test_path = os.path.join(DATA_DIR, "01_split/test.csv")
    
    # Verificar se existe o arquivo
    if not os.path.exists(test_path):
        print(f"ERRO: Arquivo de teste não encontrado: {test_path}")
        sys.exit(1)
    
    # Verificar se existem os arquivos de treinamento
    train_step1_path = os.path.join(DATA_DIR, "02_processed/train.csv")
    train_step2_path = os.path.join(DATA_DIR, "03_feature_engineering_1/train.csv")
    train_step3_path = os.path.join(DATA_DIR, "04_feature_engineering_2/train.csv")
    
    for path in [train_step1_path, train_step2_path, train_step3_path]:
        if not os.path.exists(path):
            print(f"AVISO: Arquivo de treinamento não encontrado: {path}")
    
    # Executar pipeline de inferência
    print("\n===== PARTE 1: EXECUTANDO PIPELINE DE INFERÊNCIA =====")
    inference_results = apply_inference_pipeline_with_feature_saving(test_path)
    
    # Comparar features com os datasets de treinamento
    print("\n===== PARTE 2: COMPARANDO FEATURES ENTRE TREINAMENTO E INFERÊNCIA =====")
    
    # Comparar features da etapa 1 (script 2)
    if os.path.exists(train_step1_path):
        train_step1 = pd.read_csv(train_step1_path)
        print(f"\nDataset de treinamento etapa 1 carregado: {train_step1.shape}")
        compare_feature_sets(inference_results['step1'], train_step1, "step1")
    
    # Comparar features da etapa 2 (script 3)
    if os.path.exists(train_step2_path):
        train_step2 = pd.read_csv(train_step2_path)
        print(f"\nDataset de treinamento etapa 2 carregado: {train_step2.shape}")
        compare_feature_sets(inference_results['step2'], train_step2, "step2")
    
    # Comparar features da etapa 3 (script 4)
    if os.path.exists(train_step3_path):
        train_step3 = pd.read_csv(train_step3_path)
        print(f"\nDataset de treinamento etapa 3 carregado: {train_step3.shape}")
        compare_feature_sets(inference_results['step3'], train_step3, "step3")
    
    # Analisar falsos negativos
    print("\n===== PARTE 3: ANALISANDO FALSOS NEGATIVOS =====")
    
    # Verificar se temos o arquivo de referência
    if not os.path.exists(REFERENCE_PREDICTIONS):
        print(f"ERRO: Arquivo de predições de referência não encontrado: {REFERENCE_PREDICTIONS}")
    else:
        # Carregar predições de referência
        reference_predictions = pd.read_csv(REFERENCE_PREDICTIONS)
        print(f"Arquivo de predições de referência carregado: {reference_predictions.shape}")
        
        # Analisar falsos negativos
        analyze_false_negatives(inference_results['step4'], reference_predictions)
    
    # Analisar casos próximos ao threshold
    print("\n===== PARTE 4: ANALISANDO CASOS PRÓXIMOS AO THRESHOLD =====")
    analyze_near_threshold_cases(inference_results['step4'])
    
    print("\n===== ANÁLISE CONCLUÍDA =====")
    print(f"Resultados salvos em: {REPORTS_DIR}")

if __name__ == "__main__":
    main()