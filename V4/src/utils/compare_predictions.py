#!/usr/bin/env python
"""
Script para comparar as predições da nova pipeline de inferência com 
as predições do modelo calibrado original avaliado pelo script 11_final_evaluation.py.
Versão otimizada para evitar problemas de memória.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Caminhos atualizados para os arquivos
PIPELINE_PREDICTIONS_PATH = "/Users/ramonmoreira/desktop/smart_ads/inference_v4/output/predictions.csv"
REFERENCE_PREDICTIONS_PATH = "/Users/ramonmoreira/desktop/smart_ads/reports/calibration_validation_two_models/20250518_153608/gmm_test_results.csv"

def load_predictions(pipeline_path, reference_path):
    """
    Carrega os arquivos de predições para comparação.
    """
    print(f"Carregando arquivos de predições...")
    
    # Verificar se os arquivos existem
    for path, name in [(pipeline_path, "Pipeline"), (reference_path, "Referência")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo de predições {name} não encontrado: {path}")
    
    # Carregar arquivo de pipeline
    # Usando colunas mínimas necessárias para economizar memória
    pipeline_columns = ['email', 'email_norm', 'prediction_probability', 'prediction_class']
    try:
        pipeline_df = pd.read_csv(pipeline_path, usecols=lambda x: x in pipeline_columns)
        
        # Renomear colunas para o padrão esperado pelo resto do script
        rename_map = {}
        if 'prediction_probability' in pipeline_df.columns and 'probability' not in pipeline_df.columns:
            rename_map['prediction_probability'] = 'probability'
        if 'prediction_class' in pipeline_df.columns and 'prediction' not in pipeline_df.columns:
            rename_map['prediction_class'] = 'prediction'
        
        if rename_map:
            pipeline_df = pipeline_df.rename(columns=rename_map)
            print(f"Colunas renomeadas: {rename_map}")
            
    except ValueError:
        print("Aviso: Não encontrou colunas específicas, carregando todas as colunas")
        pipeline_df = pd.read_csv(pipeline_path)
        
        # Tentar identificar e renomear colunas de predição
        if 'prediction_probability' in pipeline_df.columns:
            pipeline_df = pipeline_df.rename(columns={'prediction_probability': 'probability'})
        if 'prediction_class' in pipeline_df.columns:
            pipeline_df = pipeline_df.rename(columns={'prediction_class': 'prediction'})
        
        # Verificar se temos email ou email_norm
        if 'email' not in pipeline_df.columns:
            email_cols = [col for col in pipeline_df.columns if 'email' in col.lower()]
            if email_cols:
                pipeline_df = pipeline_df.rename(columns={email_cols[0]: 'email'})
                print(f"Renomeando coluna {email_cols[0]} para 'email'")
    
    # Carregar arquivo de referência
    reference_df = pd.read_csv(reference_path)
    
    print(f"Predições da pipeline: {pipeline_df.shape}")
    print(f"Predições de referência: {reference_df.shape}")
    
    # Verificar se temos as colunas necessárias
    for df, name in [(pipeline_df, "Pipeline"), (reference_df, "Referência")]:
        if 'prediction' not in df.columns or 'probability' not in df.columns:
            print(f"AVISO: Arquivo {name} não tem colunas 'prediction' e/ou 'probability'")
            print(f"Colunas disponíveis: {df.columns.tolist()}")
            
            # Tentar detectar colunas alternativas
            pred_cols = [col for col in df.columns if 'prediction' in col.lower() or 'class' in col.lower()]
            prob_cols = [col for col in df.columns if 'prob' in col.lower()]
            
            if pred_cols and 'prediction' not in df.columns:
                df.rename(columns={pred_cols[0]: 'prediction'}, inplace=True)
                print(f"Usando {pred_cols[0]} como coluna de predição")
                
            if prob_cols and 'probability' not in df.columns:
                df.rename(columns={prob_cols[0]: 'probability'}, inplace=True)
                print(f"Usando {prob_cols[0]} como coluna de probabilidade")
    
    return pipeline_df, reference_df

def align_predictions(pipeline_df, reference_df):
    """
    Alinha os dados de predições para garantir comparação correta.
    Versão otimizada para evitar problemas de memória.
    """
    print("Alinhando dados para comparação...")
    
    # Verificar se temos identificadores comuns (email ou similar)
    common_id = None
    for col in ['email', 'email_norm']:
        if col in pipeline_df.columns and col in reference_df.columns:
            common_id = col
            break
    
    if common_id:
        print(f"Usando coluna '{common_id}' como identificador para alinhamento")
        
        # Verificar valores únicos antes do merge
        pipeline_ids = set(pipeline_df[common_id].dropna())
        reference_ids = set(reference_df[common_id].dropna())
        
        common_ids = pipeline_ids.intersection(reference_ids)
        print(f"IDs em comum: {len(common_ids)}")
        
        if len(common_ids) < 100:
            print("AVISO: Muito poucos IDs em comum! Verifique se os dados são comparáveis.")
            if len(common_ids) == 0:
                print("Não há IDs em comum. Usando índices como alternativa.")
                # Se não há IDs em comum, usar índices
                min_rows = min(len(pipeline_df), len(reference_df))
                return create_aligned_df_by_index(pipeline_df, reference_df, min_rows)
        
        # Filtrar apenas IDs em comum ANTES de fazer o merge para evitar produto cartesiano
        pipeline_filtered = pipeline_df[pipeline_df[common_id].isin(common_ids)].copy()
        reference_filtered = reference_df[reference_df[common_id].isin(common_ids)].copy()
        
        # Verificar duplicações nos IDs, que causariam produto cartesiano
        pipeline_duplicates = pipeline_filtered[common_id].duplicated().sum()
        reference_duplicates = reference_filtered[common_id].duplicated().sum()
        
        if pipeline_duplicates > 0:
            print(f"AVISO: {pipeline_duplicates} duplicações no campo '{common_id}' na pipeline")
            print("Removendo duplicações para evitar produto cartesiano...")
            pipeline_filtered = pipeline_filtered.drop_duplicates(subset=[common_id])
        
        if reference_duplicates > 0:
            print(f"AVISO: {reference_duplicates} duplicações no campo '{common_id}' na referência")
            print("Removendo duplicações para evitar produto cartesiano...")
            reference_filtered = reference_filtered.drop_duplicates(subset=[common_id])
        
        # Mesclar DataFrames com inner join
        print(f"Realizando merge nos dados filtrados...")
        merged_df = pd.merge(
            pipeline_filtered[['prediction', 'probability', common_id]], 
            reference_filtered[['prediction', 'probability', common_id]],
            on=common_id,
            how='inner',  # Usar inner join para garantir apenas IDs em comum
            suffixes=('_pipeline', '_reference')
        )
        
        # Se "true" estiver disponível, incluir também
        if 'true' in reference_filtered.columns:
            true_df = reference_filtered[[common_id, 'true']].drop_duplicates(subset=[common_id])
            merged_df = pd.merge(merged_df, true_df, on=common_id, how='left')
        
        # Verificar se o tamanho do merge faz sentido
        expected_size = min(len(pipeline_filtered), len(reference_filtered))
        if len(merged_df) > expected_size * 1.1:  # 10% de margem
            print(f"AVISO: Merge gerou {len(merged_df)} linhas, mais que o esperado ({expected_size})")
            print("Isso pode indicar um produto cartesiano. Tentando abordagem alternativa...")
            return create_aligned_df_by_index(pipeline_df, reference_df, expected_size)
        
        print(f"Dados alinhados: {merged_df.shape}")
        return merged_df
    else:
        print("AVISO: Nenhum identificador comum encontrado para alinhamento")
        print("Assumindo que a ordem das linhas é a mesma em ambos os arquivos")
        
        # Verificar se temos o mesmo número de linhas
        min_rows = min(len(pipeline_df), len(reference_df))
        return create_aligned_df_by_index(pipeline_df, reference_df, min_rows)

def create_aligned_df_by_index(pipeline_df, reference_df, min_rows):
    """
    Cria um DataFrame alinhado usando índices em vez de chaves de junção.
    """
    # Criar DataFrame comparativo
    merged_df = pd.DataFrame({
        'prediction_pipeline': pipeline_df['prediction'].iloc[:min_rows].values,
        'probability_pipeline': pipeline_df['probability'].iloc[:min_rows].values,
        'prediction_reference': reference_df['prediction'].iloc[:min_rows].values,
        'probability_reference': reference_df['probability'].iloc[:min_rows].values
    })
    
    # Adicionar "true" se disponível
    if 'true' in reference_df.columns:
        merged_df['true'] = reference_df['true'].iloc[:min_rows].values
    
    print(f"Dados alinhados por índice: {merged_df.shape}")
    return merged_df

def analyze_agreement(comparison_df):
    """
    Analisa a concordância entre as predições da pipeline e da referência.
    """
    print("\nAnalisando concordância entre as predições...")
    
    # Verificar dados
    n_samples = len(comparison_df)
    print(f"Analisando {n_samples} amostras")
    
    if n_samples == 0:
        print("ERRO: Não há dados para analisar!")
        return {}
    
    # Concordância em predições binárias
    agreement = (comparison_df['prediction_pipeline'] == comparison_df['prediction_reference']).mean()
    print(f"Taxa de concordância (classificação): {agreement:.4f}")
    
    # Matriz de confusão
    cm = confusion_matrix(
        comparison_df['prediction_reference'], 
        comparison_df['prediction_pipeline']
    )
    print("\nMatriz de Confusão (Referência vs Pipeline):")
    print(cm)
    
    # Calcular distribuição com segurança
    total = cm.sum()
    
    if cm.shape == (2, 2):  # Confirmar formato 2x2
        true_neg = cm[0, 0] / total if total > 0 else 0
        false_pos = cm[0, 1] / total if total > 0 else 0
        false_neg = cm[1, 0] / total if total > 0 else 0
        true_pos = cm[1, 1] / total if total > 0 else 0
        
        print(f"Ambos predizem Negativo: {true_neg:.4f}")
        print(f"Referência Negativo, Pipeline Positivo: {false_pos:.4f}")
        print(f"Referência Positivo, Pipeline Negativo: {false_neg:.4f}")
        print(f"Ambos predizem Positivo: {true_pos:.4f}")
    
    # Diferença média nas probabilidades
    prob_diff = (comparison_df['probability_pipeline'] - comparison_df['probability_reference']).abs().mean()
    print(f"Diferença média nas probabilidades: {prob_diff:.4f}")
    
    # Correlação entre probabilidades
    prob_corr = comparison_df['probability_pipeline'].corr(comparison_df['probability_reference'])
    print(f"Correlação entre probabilidades: {prob_corr:.4f}")
    
    return {
        'agreement': agreement,
        'confusion_matrix': cm,
        'prob_diff': prob_diff,
        'prob_corr': prob_corr
    }

def evaluate_predictions(comparison_df, reference_df):
    """
    Avalia as predições contra valores reais, se disponíveis.
    """
    if 'true' not in reference_df.columns and 'true' not in comparison_df.columns:
        print("\nAviso: Valores reais ('true') não encontrados no arquivo de referência")
        return
    
    print("\nAvaliando predições contra valores reais...")
    
    try:
        # Limitando o tamanho para evitar problemas de memória
        max_samples = min(len(comparison_df), 10000)  # Limitar a 10K amostras
        if len(comparison_df) > max_samples:
            print(f"Usando uma amostra de {max_samples} registros para avaliação")
            comparison_sample = comparison_df.sample(max_samples, random_state=42)
        else:
            comparison_sample = comparison_df
        
        # Se já temos 'true' no DataFrame de comparação, usá-lo diretamente
        if 'true' in comparison_sample.columns:
            print("Usando valores 'true' já presentes nos dados alinhados.")
        # Caso contrário, tentar obter de reference_df
        elif 'true' in reference_df.columns:
            # Tentar juntar usando identificador comum se disponível
            if 'email' in comparison_sample.columns and 'email' in reference_df.columns:
                # Usando email como chave para obter valores reais
                ref_subset = reference_df[['email', 'true']].drop_duplicates(subset=['email'])
                comparison_sample = pd.merge(
                    comparison_sample,
                    ref_subset,
                    on='email',
                    how='left'
                )
            # Tentar usar índices como alternativa
            else:
                try:
                    comparison_sample['true'] = reference_df['true'].iloc[:len(comparison_sample)].values
                    print("Valores 'true' adicionados usando índices.")
                except Exception as e:
                    print(f"Erro ao adicionar valores 'true': {e}")
                    print("Pulando avaliação contra valores reais.")
                    return
        else:
            print("Valores 'true' não encontrados. Pulando avaliação contra valores reais.")
            return
        
        # Verificar se temos true em comparison_sample
        if 'true' not in comparison_sample.columns:
            print("Aviso: Não foi possível obter valores reais para avaliação")
            return
        
        # Calcular métricas para cada modelo
        metrics = {}
        for model in ['reference', 'pipeline']:
            y_true = comparison_sample['true']
            y_pred = comparison_sample[f'prediction_{model}']
            
            metrics[model] = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            }
        
        # Imprimir comparativo
        print("\nMétricas de desempenho:")
        print(f"{'Métrica':<10} {'Referência':<12} {'Pipeline':<12} {'Diferença':<10}")
        print("-" * 44)
        
        for metric in ['precision', 'recall', 'f1']:
            ref_val = metrics['reference'][metric]
            pipe_val = metrics['pipeline'][metric]
            diff = pipe_val - ref_val
            diff_pct = diff / ref_val * 100 if ref_val != 0 else float('inf')
            
            print(f"{metric:<10} {ref_val:.4f}       {pipe_val:.4f}       {diff:.4f} ({diff_pct:+.1f}%)")
    
    except Exception as e:
        print(f"Erro durante avaliação: {e}")
        import traceback
        traceback.print_exc()

def visualize_comparison(comparison_df, output_dir="."):
    """
    Cria visualizações para comparar as predições.
    """
    print("\nCriando visualizações...")
    
    try:
        # Criar diretório para gráficos se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Limitar o tamanho para visualização
        max_samples = min(len(comparison_df), 50000)  # Limitar a 50K amostras para visualização
        if len(comparison_df) > max_samples:
            print(f"Usando uma amostra de {max_samples} registros para visualização")
            viz_sample = comparison_df.sample(max_samples, random_state=42)
        else:
            viz_sample = comparison_df
        
        # 1. Histograma de diferenças nas probabilidades
        plt.figure(figsize=(10, 6))
        prob_diff = viz_sample['probability_pipeline'] - viz_sample['probability_reference']
        plt.hist(prob_diff, bins=50, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title("Diferenças nas Probabilidades (Pipeline - Referência)")
        plt.xlabel("Diferença")
        plt.ylabel("Frequência")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "probability_differences.png"), dpi=300)
        plt.close()
        
        # 2. Scatter plot das probabilidades
        plt.figure(figsize=(10, 10))
        plt.scatter(
            viz_sample['probability_reference'], 
            viz_sample['probability_pipeline'],
            alpha=0.5, s=10
        )
        plt.plot([0, 1], [0, 1], 'r--')  # Linha de igualdade
        plt.title("Probabilidades: Pipeline vs Referência")
        plt.xlabel("Probabilidade (Referência)")
        plt.ylabel("Probabilidade (Pipeline)")
        plt.grid(alpha=0.3)
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, "probability_scatter.png"), dpi=300)
        plt.close()
        
        # 3. Barras para taxa de concordância
        plt.figure(figsize=(8, 6))
        labels = ['Concordância', 'Discordância']
        agreement = (viz_sample['prediction_pipeline'] == viz_sample['prediction_reference']).mean()
        values = [agreement, 1-agreement]
        
        plt.bar(labels, values, color=['green', 'red'])
        plt.title(f"Taxa de Concordância nas Predições: {agreement:.2%}")
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Adicionar percentuais nas barras
        for i, v in enumerate(values):
            plt.text(i, v/2, f"{v:.1%}", ha='center', fontweight='bold', color='white')
        
        plt.savefig(os.path.join(output_dir, "agreement_rate.png"), dpi=300)
        plt.close()
        
        print(f"Visualizações salvas em: {output_dir}")
    
    except Exception as e:
        print(f"Erro durante visualização: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Solicitar caminhos dos arquivos
    pipeline_path = input(f"Caminho para o arquivo de predições da pipeline [{PIPELINE_PREDICTIONS_PATH}]: ")
    pipeline_path = pipeline_path.strip() if pipeline_path.strip() else PIPELINE_PREDICTIONS_PATH
    
    reference_path = input(f"Caminho para o arquivo de predições de referência [{REFERENCE_PREDICTIONS_PATH}]: ")
    reference_path = reference_path.strip() if reference_path.strip() else REFERENCE_PREDICTIONS_PATH
    
    # Carregar dados
    try:
        pipeline_df, reference_df = load_predictions(pipeline_path, reference_path)
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        return
    
    # Alinhar dados
    comparison_df = align_predictions(pipeline_df, reference_df)
    
    # Analisar concordância
    results = analyze_agreement(comparison_df)
    
    # Avaliar contra valores reais
    evaluate_predictions(comparison_df, reference_df)
    
    # Visualizar
    output_dir = os.path.join(os.path.dirname(pipeline_path), "comparison_results")
    visualize_comparison(comparison_df, output_dir)
    
    print("\nComparação concluída com sucesso!")
    print(f"Resultados salvos em: {output_dir}")

if __name__ == "__main__":
    main()