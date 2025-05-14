#!/usr/bin/env python
"""
Script completo para testar a pipeline de inferência do Smart Ads.

Este script:
1. Executa a pipeline nos dados de teste após o script 01
2. Gera resultados da pipeline
3. Compara os resultados com o modelo de treinamento original

Foco na comparação de probabilidades para classificação por decis.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Adicionar diretório do projeto ao path para poder importar os outros scripts
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.append(PROJECT_ROOT)

# Importar componentes da pipeline
from src.inference.EmailNormalizationTransformer import EmailNormalizationTransformer
from src.inference.CompletePreprocessingTransformer import CompletePreprocessingTransformer
from src.inference.TextFeatureEngineeringTransformer import TextFeatureEngineeringTransformer
from src.inference.GMM_InferenceWrapper import GMM_InferenceWrapper

# Configuração de caminhos
DATA_DIR = os.path.join(PROJECT_ROOT, "data/01_split")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models/artifacts/gmm_optimized")
PARAMS_DIR = os.path.join(PROJECT_ROOT, "src/preprocessing/preprocessing_params")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports/pipeline_testing")
ORIGINAL_RESULTS = os.path.join(PROJECT_ROOT, "reports/calibration_validation_three_models/20250512_073645/gmm_test_results.csv")

# Criar o diretório de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Função para colorir diferenças significativas
def color_diff(val):
    if abs(val) < 0.001:
        return 'background-color: green; color: white'
    elif abs(val) < 0.01:
        return 'background-color: yellow'
    else:
        return 'background-color: red; color: white'

def create_inference_pipeline(models_dir=MODELS_DIR, params_dir=PARAMS_DIR):
    """
    Cria a pipeline de inferência.
    
    Args:
        models_dir: Diretório contendo componentes do modelo
        params_dir: Diretório contendo parâmetros salvos
        
    Returns:
        Pipeline configurada
    """
    # Verificar existência dos diretórios
    if not os.path.exists(models_dir):
        raise ValueError(f"Diretório de modelos não encontrado: {models_dir}")
    if not os.path.exists(params_dir):
        raise ValueError(f"Diretório de parâmetros não encontrado: {params_dir}")
    
    # Definir caminhos para parâmetros
    preprocessing_params_path = os.path.join(params_dir, "all_preprocessing_params.joblib")
    script03_params_path = os.path.join(params_dir, "script03_params.joblib")
    
    # Verificar se os parâmetros existem
    if not os.path.exists(preprocessing_params_path):
        raise ValueError(f"Parâmetros de pré-processamento não encontrados: {preprocessing_params_path}")
    
    # Verificar se temos parâmetros do script 3
    if os.path.exists(script03_params_path):
        print(f"Parâmetros do script 3 encontrados em: {script03_params_path}")
    else:
        print(f"AVISO: Parâmetros do script 3 não encontrados em: {script03_params_path}")
        print("A pipeline usará apenas os parâmetros padrão.")
    
    # Determinar se existe modelo calibrado
    use_calibrated = False
    calibrated_dir = os.path.join(PROJECT_ROOT, "models/calibrated")
    if os.path.exists(calibrated_dir):
        # Procurar pelo diretório mais recente
        calibrated_dirs = [d for d in os.listdir(calibrated_dir) 
                          if os.path.isdir(os.path.join(calibrated_dir, d)) 
                          and d.startswith("gmm_calibrated_")]
        
        if calibrated_dirs:
            # Ordenar por data de modificação (mais recente primeiro)
            calibrated_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(calibrated_dir, d)), 
                              reverse=True)
            
            calibrated_model_dir = os.path.join(calibrated_dir, calibrated_dirs[0])
            if os.path.exists(os.path.join(calibrated_model_dir, "gmm_calibrated.joblib")):
                models_dir = calibrated_model_dir
                use_calibrated = True
                print(f"Usando modelo GMM calibrado de: {calibrated_model_dir}")
    
    # Criar componentes individuais
    email_transformer = EmailNormalizationTransformer(email_col='email')
    preprocessing_transformer = CompletePreprocessingTransformer(params_path=preprocessing_params_path)
    text_transformer = TextFeatureEngineeringTransformer(
        params_path=preprocessing_params_path,
        script03_params_path=script03_params_path
    )
    predictor = GMM_InferenceWrapper(models_dir=models_dir)
    
    # Criar pipeline
    pipeline = Pipeline([
        ('email_normalization', email_transformer),
        ('preprocessing', preprocessing_transformer),
        ('text_features', text_transformer),
        ('predictor', predictor)
    ])
    
    return pipeline, use_calibrated

def run_pipeline():
    """
    Executa a pipeline nos dados de teste e gera resultados.
    
    Returns:
        DataFrame com resultados
    """
    print("\n===== EXECUTANDO PIPELINE DE INFERÊNCIA =====")
    
    # Verificar se o arquivo de teste existe
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {TEST_FILE}")
    
    # Carregar dados de teste
    print(f"Carregando dados de teste de: {TEST_FILE}")
    test_df = pd.read_csv(TEST_FILE)
    print(f"Dados carregados: {test_df.shape[0]} linhas, {test_df.shape[1]} colunas")
    
    # Criar pipeline
    print("\nCriando pipeline de inferência...")
    pipeline, use_calibrated = create_inference_pipeline()
    print(f"Pipeline criada. Usando modelo {'calibrado' if use_calibrated else 'não calibrado'}")
    
    # Aplicar pipeline
    print("\nAplicando pipeline para previsão...")
    start_time = datetime.now()
    
    # IMPORTANTE: Primeiro ajustar (fit) a pipeline antes de fazer previsões
    print("Ajustando a pipeline nos dados...")
    # Criar um conjunto de dados dummy para o target se necessário
    if 'target' in test_df.columns:
        y_dummy = test_df['target']
    else:
        y_dummy = np.zeros(len(test_df))
    
    # Ajustar cada componente da pipeline individualmente para evitar problemas
    print("1. Ajustando transformador de emails...")
    email_transformer = pipeline.named_steps['email_normalization']
    test_df_email = email_transformer.fit_transform(test_df.copy())
    
    print("2. Ajustando transformador de pré-processamento...")
    preproc_transformer = pipeline.named_steps['preprocessing']
    preproc_transformer.fit(test_df_email)
    test_df_preproc = preproc_transformer.transform(test_df_email)
    
    print("3. Ajustando transformador de features de texto...")
    text_transformer = pipeline.named_steps['text_features']
    text_transformer.fit(test_df_preproc)
    test_df_text = text_transformer.transform(test_df_preproc)
    
    print("4. Ajustando preditor GMM...")
    predictor = pipeline.named_steps['predictor']
    predictor.fit()
    
    # Agora fazer as previsões usando os dados pré-processados
    print("Gerando previsões...")
    probas = predictor.predict_proba(test_df_text)
    predictions = predictor.predict(test_df_text)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    print(f"Processamento concluído em {processing_time:.2f} segundos")
    
    # Criar DataFrame de resultados
    results_df = pd.DataFrame()
    results_df['email'] = test_df['email']
    if 'email_norm' in test_df_email.columns:
        results_df['email_norm'] = test_df_email['email_norm']
    results_df['probability'] = probas[:, 1]
    results_df['prediction'] = predictions
    
    # Adicionar decil de probabilidade
    # Usar duplicates='drop' para lidar com valores duplicados nas bordas
    try:
        results_df['probability_decile'] = pd.qcut(results_df['probability'], 10, labels=False, duplicates='drop') + 1
    except ValueError as e:
        # Se ainda houver problemas, use um método alternativo de quantis
        print(f"AVISO: Não foi possível criar decis exatos devido à distribuição dos dados: {e}")
        print("Usando método alternativo para criar categorias de probabilidade...")
        
        # Contar valores únicos
        unique_probs = results_df['probability'].nunique()
        print(f"Existem apenas {unique_probs} valores únicos de probabilidade")
        
        if unique_probs < 10:
            # Se houver menos de 10 valores únicos, usar rank diretamente
            results_df['probability_decile'] = results_df['probability'].rank(method='dense').astype(int)
            max_decile = results_df['probability_decile'].max()
            if max_decile < 10:
                results_df['probability_decile'] = results_df['probability_decile'].apply(
                    lambda x: int(np.ceil(x * 10 / max_decile))
                )
            print(f"Criados {results_df['probability_decile'].nunique()} grupos de probabilidade")
        else:
            # Método alternativo: classificar e dividir em grupos aproximados
            n_bins = min(10, unique_probs)
            results_df['probability_decile'] = np.ceil(results_df['probability'].rank(pct=True) * n_bins).astype(int)
            results_df.loc[results_df['probability_decile'] == 0, 'probability_decile'] = 1  # Garantir mínimo de 1
    
    # Salvar resultados
    output_file = os.path.join(OUTPUT_DIR, "pipeline_test_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Resultados salvos em: {output_file}")
    
    # Estatísticas básicas
    qualified_leads = predictions.sum()
    qualified_pct = (qualified_leads / len(test_df)) * 100
    
    print(f"\nEstatísticas de previsão:")
    print(f"Total de leads processados: {len(test_df)}")
    print(f"Leads qualificados: {int(qualified_leads)} ({qualified_pct:.2f}%)")
    
    # Estatísticas por decil ou grupo de probabilidade
    decile_stats = results_df.groupby('probability_decile').agg(
        count=('email', 'count'),
        avg_prob=('probability', 'mean'),
        qualified=('prediction', 'sum'),
        pct_qualified=('prediction', lambda x: x.mean() * 100)
    ).reset_index()
    
    print("\nEstatísticas por grupo de probabilidade:")
    print(decile_stats)
    
    # Salvar estatísticas por decil
    decile_stats_file = os.path.join(OUTPUT_DIR, "decile_statistics.csv")
    decile_stats.to_csv(decile_stats_file, index=False)
    print(f"Estatísticas por grupo salvas em: {decile_stats_file}")
    
    return results_df

def compare_with_original(new_results):
    """
    Compara os resultados da pipeline com os resultados originais do treinamento.
    
    Args:
        new_results: DataFrame com resultados da pipeline
    """
    print("\n===== COMPARANDO RESULTADOS COM MODELO ORIGINAL =====")
    
    # Verificar se o arquivo de resultados originais existe
    if not os.path.exists(ORIGINAL_RESULTS):
        print(f"ERRO: Arquivo de resultados originais não encontrado: {ORIGINAL_RESULTS}")
        return
    
    # Carregar resultados originais
    print(f"Carregando resultados originais de: {ORIGINAL_RESULTS}")
    original_results = pd.read_csv(ORIGINAL_RESULTS)
    print(f"Resultados originais carregados: {original_results.shape[0]} linhas")
    
    # Verificar colunas nos resultados originais
    print(f"Colunas nos resultados originais: {original_results.columns.tolist()}")
    
    # Identificar colunas para probabilidade e email
    original_prob_col = None
    original_email_col = None
    
    # Procurar colunas específicas nos resultados originais
    if 'probability' in original_results.columns:
        original_prob_col = 'probability'
    elif 'prob' in original_results.columns:
        original_prob_col = 'prob'
    else:
        # Buscar qualquer coluna que contenha 'prob'
        prob_cols = [col for col in original_results.columns if 'prob' in col.lower()]
        if prob_cols:
            original_prob_col = prob_cols[0]
    
    if 'email' in original_results.columns:
        original_email_col = 'email'
    elif 'email_norm' in original_results.columns:
        original_email_col = 'email_norm'
    
    if not original_prob_col or not original_email_col:
        print("ERRO: Colunas necessárias não encontradas nos resultados originais.")
        print(f"Colunas disponíveis: {original_results.columns.tolist()}")
        return
    
    print(f"Usando coluna '{original_prob_col}' para probabilidades originais")
    print(f"Usando coluna '{original_email_col}' para identificação")
    
    # Colunas nos novos resultados
    new_prob_col = 'probability'
    new_email_col = original_email_col  # Usar o mesmo campo para junção
    
    # Verificar se temos a coluna de email nos novos resultados
    if new_email_col not in new_results.columns:
        print(f"ERRO: Coluna '{new_email_col}' não encontrada nos novos resultados.")
        print(f"Colunas disponíveis: {new_results.columns.tolist()}")
        return
    
    # Verificar valores duplicados que podem causar problemas no merge
    orig_duplicated = original_results[original_email_col].duplicated().sum()
    new_duplicated = new_results[new_email_col].duplicated().sum()
    
    if orig_duplicated > 0:
        print(f"AVISO: {orig_duplicated} emails duplicados nos resultados originais")
        print("Removendo duplicatas para evitar explosão do número de registros...")
        original_results = original_results.drop_duplicates(subset=[original_email_col])
    
    if new_duplicated > 0:
        print(f"AVISO: {new_duplicated} emails duplicados nos novos resultados")
        print("Removendo duplicatas para evitar explosão do número de registros...")
        new_results = new_results.drop_duplicates(subset=[new_email_col])
    
    # Mesclar resultados pelo campo de email identificado (com tratamento de duplicatas)
    print(f"Mesclando resultados por '{original_email_col}'...")
    
    # Extrair apenas as colunas necessárias para reduzir uso de memória
    original_subset = original_results[[original_email_col, original_prob_col]].copy()
    new_subset = new_results[[new_email_col, new_prob_col]].copy()
    
    # Executar o merge com verificações adicionais
    try:
        # Tentar realizar o merge com diagnóstico de tamanho
        print(f"Tamanho do dataframe original: {len(original_subset)}")
        print(f"Tamanho do dataframe novo: {len(new_subset)}")
        
        # Renomear colunas para clareza
        original_subset.rename(columns={original_prob_col: 'original_probability'}, inplace=True)
        new_subset.rename(columns={new_prob_col: 'new_probability'}, inplace=True)
        
        # Realizar o merge
        merged = pd.merge(
            original_subset,
            new_subset,
            on=original_email_col, 
            how='inner'
        )
        
        # Verificar o tamanho após o merge
        merge_count = len(merged)
        original_count = len(original_subset)
        new_count = len(new_subset)
        
        # Verificar se o tamanho é razoável (não deve ser maior que o produto das entradas)
        max_expected = min(original_count, new_count)
        if merge_count > max_expected * 1.1:  # Permitir uma margem de 10%
            print(f"ALERTA: Número de registros mesclados ({merge_count}) é muito maior que o esperado.")
            print("Isso sugere um problema com valores duplicados nas chaves de mesclagem.")
            print("Aplicando técnica alternativa de mesclagem...")
            
            # Redefinir e tentar uma abordagem mais restritiva
            merged = pd.merge(
                original_subset.drop_duplicates(subset=original_email_col),
                new_subset.drop_duplicates(subset=new_email_col),
                on=original_email_col, 
                how='inner'
            )
            merge_count = len(merged)
        
        print(f"Registros mesclados: {merge_count} de {original_count} originais e {new_count} novos")
        
    except Exception as e:
        print(f"ERRO durante a mesclagem: {e}")
        print("Tentando abordagem alternativa...")
        
        # Tentar uma abordagem diferente
        try:
            # Usar pandas concat e groupby como estratégia alternativa
            original_subset = original_subset.set_index(original_email_col)
            new_subset = new_subset.set_index(new_email_col)
            
            # Unir por índice e agrupar pelo índice
            combined = pd.concat([original_subset, new_subset], axis=1, join='inner')
            merge_count = len(combined)
            
            if merge_count > 0:
                print(f"Mesclagem alternativa bem-sucedida: {merge_count} registros")
                merged = combined.reset_index()
                original_count = len(original_subset)
                new_count = len(new_subset)
            else:
                print("ERRO: Nenhum registro mesclado usando método alternativo.")
                return
        except Exception as e2:
            print(f"ERRO na abordagem alternativa: {e2}")
            print("Não foi possível realizar a comparação.")
            return
    
    if merge_count == 0:
        print("ERRO: Nenhum registro comum encontrado para comparação.")
        return
    
    # Calcular diferenças de probabilidade (usar novos nomes de coluna)
    merged['prob_diff'] = merged['original_probability'] - merged['new_probability']
    merged['prob_diff_abs'] = merged['prob_diff'].abs()
    
    # Adicionar decil de probabilidade para ambos os conjuntos
    try:
        # Usar duplicates='drop' para lidar com valores duplicados nas bordas
        merged['original_decile'] = pd.qcut(merged['original_probability'], 10, labels=False, duplicates='drop') + 1
    except ValueError:
        # Se falhar, usar método alternativo
        print("AVISO: Não foi possível criar decis exatos para probabilidades originais.")
        merged['original_decile'] = np.ceil(merged['original_probability'].rank(pct=True) * 10).astype(int)
        
    try:
        merged['new_decile'] = pd.qcut(merged['new_probability'], 10, labels=False, duplicates='drop') + 1
    except ValueError:
        # Se falhar, usar método alternativo
        print("AVISO: Não foi possível criar decis exatos para novas probabilidades.")
        merged['new_decile'] = np.ceil(merged['new_probability'].rank(pct=True) * 10).astype(int)
    
    merged['decile_diff'] = merged['original_decile'] - merged['new_decile']
    
    # Estatísticas de diferença
    prob_diff_mean = merged['prob_diff'].mean()
    prob_diff_abs_mean = merged['prob_diff_abs'].mean()
    prob_diff_median = merged['prob_diff'].median()
    prob_diff_std = merged['prob_diff'].std()
    prob_diff_max = merged['prob_diff_abs'].max()
    
    # Estatísticas de decil
    same_decile = (merged['decile_diff'] == 0).sum()
    same_decile_pct = (same_decile / merge_count) * 100
    
    one_decile_diff = (merged['decile_diff'].abs() <= 1).sum() - same_decile
    one_decile_diff_pct = (one_decile_diff / merge_count) * 100
    
    two_decile_diff = (merged['decile_diff'].abs() <= 2).sum() - same_decile - one_decile_diff
    two_decile_diff_pct = (two_decile_diff / merge_count) * 100
    
    # Calcular limites para diferenças
    small_diffs = (merged['prob_diff_abs'] < 0.001).sum()
    medium_diffs = ((merged['prob_diff_abs'] >= 0.001) & (merged['prob_diff_abs'] < 0.01)).sum()
    large_diffs = (merged['prob_diff_abs'] >= 0.01).sum()
    
    small_diffs_pct = (small_diffs / merge_count) * 100
    medium_diffs_pct = (medium_diffs / merge_count) * 100
    large_diffs_pct = (large_diffs / merge_count) * 100
    
    # Calcular correlação entre probabilidades originais e novas
    prob_correlation = merged[['original_probability', 'new_probability']].corr().iloc[0, 1]
    
    # Exibir resultados
    print("\n===== ESTATÍSTICAS DE COMPARAÇÃO DE PROBABILIDADES =====")
    print(f"Registros comparados: {merge_count}")
    
    print(f"\nDiferença média de probabilidade: {prob_diff_mean:.6f}")
    print(f"Diferença média absoluta: {prob_diff_abs_mean:.6f}")
    print(f"Desvio padrão das diferenças: {prob_diff_std:.6f}")
    print(f"Diferença máxima absoluta: {prob_diff_max:.6f}")
    print(f"Correlação entre probabilidades: {prob_correlation:.4f}")
    
    print(f"\nDistribuição das diferenças:")
    print(f"  Pequenas (<0.001): {small_diffs} ({small_diffs_pct:.2f}%)")
    print(f"  Médias (0.001-0.01): {medium_diffs} ({medium_diffs_pct:.2f}%)")
    print(f"  Grandes (>0.01): {large_diffs} ({large_diffs_pct:.2f}%)")
    
    print("\n===== ANÁLISE DE DECIS =====")
    print(f"Mesmo decil: {same_decile} ({same_decile_pct:.2f}%)")
    print(f"Diferença de 1 decil: {one_decile_diff} ({one_decile_diff_pct:.2f}%)")
    print(f"Diferença de 2 decis: {two_decile_diff} ({two_decile_diff_pct:.2f}%)")
    print(f"Diferença maior que 2 decis: {merge_count - same_decile - one_decile_diff - two_decile_diff} "
          f"({100 - same_decile_pct - one_decile_diff_pct - two_decile_diff_pct:.2f}%)")
    
    # Avaliar compatibilidade geral com foco em probabilidades
    print("\n===== AVALIAÇÃO DE COMPATIBILIDADE =====")
    if prob_diff_max < 0.001 and same_decile_pct > 99:
        print("A pipeline de inferência é 100% compatível com o modelo original!")
    elif prob_diff_max < 0.01 and same_decile_pct + one_decile_diff_pct > 95:
        print("A pipeline de inferência é praticamente compatível com o modelo original.")
        print("Pequenas diferenças numéricas são esperadas devido à precisão de ponto flutuante.")
    elif prob_correlation > 0.95 and same_decile_pct + one_decile_diff_pct + two_decile_diff_pct > 90:
        print("A pipeline de inferência é aceitavelmente compatível com o modelo original.")
        print("As diferenças observadas não devem impactar significativamente a classificação por decis.")
    else:
        print("Há discrepâncias significativas entre a pipeline de inferência e o modelo original.")
        print("Revise a implementação para garantir processamento idêntico.")
    
    # Visualizações e análises adicionais
    try:
        # 1. Histograma das diferenças de probabilidade
        plt.figure(figsize=(10, 6))
        sns.histplot(merged['prob_diff'], bins=50, kde=True)
        plt.title('Distribuição das Diferenças de Probabilidade')
        plt.xlabel('Diferença (Original - Pipeline)')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        hist_file = os.path.join(OUTPUT_DIR, 'probability_difference_histogram.png')
        plt.savefig(hist_file)
        print(f"\nHistograma de diferenças salvo em: {hist_file}")
        
        # 2. Gráfico de dispersão das probabilidades
        plt.figure(figsize=(10, 10))
        plt.scatter(merged['original_probability'], merged['new_probability'], alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')  # Linha de referência
        plt.title('Comparação de Probabilidades')
        plt.xlabel('Probabilidades Originais')
        plt.ylabel('Probabilidades da Pipeline')
        plt.grid(True, alpha=0.3)
        scatter_file = os.path.join(OUTPUT_DIR, 'probability_comparison_scatter.png')
        plt.savefig(scatter_file)
        print(f"Gráfico de dispersão salvo em: {scatter_file}")
        
        # 3. Heatmap de transição entre decis
        plt.figure(figsize=(12, 8))
        transition_matrix = pd.crosstab(
            merged['original_decile'], 
            merged['new_decile'], 
            normalize='index'
        )
        sns.heatmap(transition_matrix, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Proporção'})
        plt.title('Matriz de Transição entre Decis de Probabilidade')
        plt.xlabel('Decil (Pipeline)')
        plt.ylabel('Decil (Original)')
        heatmap_file = os.path.join(OUTPUT_DIR, 'decile_transition_heatmap.png')
        plt.savefig(heatmap_file)
        print(f"Heatmap de transição entre decis salvo em: {heatmap_file}")
        
        # 4. Amostras com maiores diferenças
        if large_diffs > 0:
            print("\n===== AMOSTRAS COM MAIORES DIFERENÇAS =====")
            largest_diffs = merged.sort_values(by='prob_diff_abs', ascending=False).head(10)
            pd.set_option('display.precision', 6)
            print(largest_diffs[[original_email_col, 'original_probability', 'new_probability', 'prob_diff', 'original_decile', 'new_decile']])
    except Exception as viz_error:
        print(f"\nAVISO: Erro ao gerar visualizações: {viz_error}")
    
    try:
        # 5. Salvar análise completa
        comparison_file = os.path.join(OUTPUT_DIR, 'pipeline_comparison_results.csv')
        merged.to_csv(comparison_file, index=False)
        print(f"\nAnálise completa salva em: {comparison_file}")
        
        # 6. Salvar estatísticas de resumo
        summary_stats = {
            'metric': [
                'registros_comparados', 'diferenca_media', 'diferenca_abs_media', 
                'diferenca_maxima', 'correlacao_probabilidades',
                'pct_pequenas_diferencas', 'pct_medias_diferencas', 'pct_grandes_diferencas',
                'pct_mesmo_decil', 'pct_um_decil_diff', 'pct_dois_decil_diff'
            ],
            'value': [
                merge_count, prob_diff_mean, prob_diff_abs_mean,
                prob_diff_max, prob_correlation,
                small_diffs_pct, medium_diffs_pct, large_diffs_pct,
                same_decile_pct, one_decile_diff_pct, two_decile_diff_pct
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(OUTPUT_DIR, 'comparison_summary_stats.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Estatísticas resumidas salvas em: {summary_file}")
        
        # 7. Salvar versão HTML estilizada
        try:
            styled_sample = merged.sample(min(20, len(merged)))
            styled_df = styled_sample[[original_email_col, 'original_probability', 'new_probability', 'prob_diff', 'original_decile', 'new_decile']]
            styled_df = styled_df.style.applymap(color_diff, subset=['prob_diff'])
            
            html_file = os.path.join(OUTPUT_DIR, 'styled_comparison_samples.html')
            with open(html_file, 'w') as f:
                f.write("<h1>Amostra de Comparação Pipeline vs Original</h1>")
                f.write(styled_df.to_html())
            print(f"Amostra estilizada salva em: {html_file}")
        except Exception as style_error:
            print(f"AVISO: Erro ao gerar amostra estilizada: {style_error}")
    except Exception as output_error:
        print(f"\nAVISO: Erro ao salvar resultados: {output_error}")

def main():
    """Função principal"""
    print("=" * 60)
    print("TESTE DA PIPELINE DE INFERÊNCIA DO SMART ADS")
    print("=" * 60)
    
    try:
        # 1. Executar a pipeline nos dados de teste
        new_results = run_pipeline()
        
        # 2. Comparar com resultados originais
        compare_with_original(new_results)
        
        print("\n===== PROCESSO CONCLUÍDO COM SUCESSO =====")
        print(f"Todos os resultados foram salvos em: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\nERRO durante o processo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()