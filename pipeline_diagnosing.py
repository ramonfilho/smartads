#!/usr/bin/env python
"""
Script de diagnóstico para identificar discrepâncias entre o pipeline de inferência
e o pipeline de treinamento do projeto Smart Ads.

Este script compara os resultados intermediários após cada etapa de transformação
com os dados salvos durante o treinamento.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar diretório do projeto ao path
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.append(PROJECT_ROOT)

# Importar componentes da pipeline
from src.inference.EmailNormalizationTransformer import EmailNormalizationTransformer
from src.inference.CompletePreprocessingTransformer import CompletePreprocessingTransformer
from src.inference.TextFeatureEngineeringTransformer import TextFeatureEngineeringTransformer
from src.inference.GMM_InferenceWrapper import GMM_InferenceWrapper

# Configuração de caminhos
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_FILE = os.path.join(DATA_DIR, "01_split", "test.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models/artifacts/gmm_optimized")
PARAMS_DIR = os.path.join(PROJECT_ROOT, "src/preprocessing/preprocessing_params")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports/pipeline_diagnostics")

# Criar o diretório de saída
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_datasets():
    """
    Carrega os datasets originais e processados para comparação.
    
    Returns:
        Dictionary com os DataFrames carregados
    """
    datasets = {}
    
    # Dataset de teste original
    print(f"Carregando dados de teste originais de: {TEST_FILE}")
    datasets['original'] = pd.read_csv(TEST_FILE)
    
    # Datasets processados durante o treinamento
    try:
        # Após preprocessamento básico
        preproc_path = os.path.join(DATA_DIR, "02_1_processed", "test.csv")
        if os.path.exists(preproc_path):
            datasets['train_preproc'] = pd.read_csv(preproc_path)
            print(f"Dados de teste pré-processados durante treino carregados: {preproc_path}")
        
        # Após processamento do script 03
        nlp_path = os.path.join(DATA_DIR, "02_2_reprocessed", "test.csv")
        if os.path.exists(nlp_path):
            datasets['train_nlp'] = pd.read_csv(nlp_path)
            print(f"Dados de teste com NLP do script 03 carregados: {nlp_path}")
        
        # Após processamento do script 04
        final_path = os.path.join(DATA_DIR, "02_3_processed", "test.csv")
        if os.path.exists(final_path):
            datasets['train_final'] = pd.read_csv(final_path)
            print(f"Dados de teste com processamento completo carregados: {final_path}")
    
    except Exception as e:
        print(f"Erro ao carregar dados processados: {e}")
    
    return datasets

def compare_dataframes(df1, df2, name1="DF1", name2="DF2", prefix="comparison"):
    """
    Compara dois DataFrames e gera estatísticas e visualizações das diferenças.
    
    Args:
        df1: Primeiro DataFrame
        df2: Segundo DataFrame
        name1: Nome do primeiro DataFrame para referência
        name2: Nome do segundo DataFrame para referência
        prefix: Prefixo para arquivos de saída
        
    Returns:
        Dictionary com estatísticas de comparação
    """
    print(f"\n===== Comparando {name1} vs {name2} =====")
    
    # Verificar dimensões
    print(f"{name1}: {df1.shape[0]} linhas, {df1.shape[1]} colunas")
    print(f"{name2}: {df2.shape[0]} linhas, {df2.shape[1]} colunas")
    
    # Encontrar colunas comuns
    common_cols = set(df1.columns).intersection(set(df2.columns))
    print(f"Colunas comuns: {len(common_cols)}")
    
    # Colunas em df1 mas não em df2
    cols_only_in_df1 = set(df1.columns) - set(df2.columns)
    print(f"Colunas exclusivas de {name1}: {len(cols_only_in_df1)}")
    if len(cols_only_in_df1) > 0:
        print(f"  Exemplos: {list(cols_only_in_df1)[:5]}")
    
    # Colunas em df2 mas não em df1
    cols_only_in_df2 = set(df2.columns) - set(df1.columns)
    print(f"Colunas exclusivas de {name2}: {len(cols_only_in_df2)}")
    if len(cols_only_in_df2) > 0:
        print(f"  Exemplos: {list(cols_only_in_df2)[:5]}")
    
    # Se não há colunas comuns, não podemos continuar a comparação
    if len(common_cols) == 0:
        print("Sem colunas comuns para comparar!")
        return {
            "common_cols": 0,
            "cols_only_in_df1": len(cols_only_in_df1),
            "cols_only_in_df2": len(cols_only_in_df2)
        }
    
    # Verificar se temos uma coluna 'email' para alinhar os dados
    join_col = None
    for col in ['email', 'email_norm']:
        if col in common_cols:
            join_col = col
            break
    
    # Se temos uma coluna de email, tentamos juntar os DataFrames
    if join_col:
        print(f"Usando coluna '{join_col}' para alinhar registros")
        
        # Remover duplicatas para evitar problemas no merge
        df1_dedup = df1.drop_duplicates(subset=[join_col]).copy()
        df2_dedup = df2.drop_duplicates(subset=[join_col]).copy()
        
        # Merge dos DataFrames usando a coluna de email
        merged = pd.merge(
            df1_dedup,
            df2_dedup,
            on=join_col,
            how='inner',
            suffixes=(f'_{name1}', f'_{name2}')
        )
        
        print(f"Registros mesclados: {len(merged)} de {len(df1_dedup)} em {name1} e {len(df2_dedup)} em {name2}")
        
        # Comparar colunas numéricas
        numeric_diffs = {}
        high_diff_cols = []
        
        for col in common_cols:
            if col == join_col:
                continue
                
            col1 = f"{col}_{name1}" if col != join_col else col
            col2 = f"{col}_{name2}" if col != join_col else col
            
            # Verificar se ambas as colunas existem após o merge
            if col1 not in merged.columns or col2 not in merged.columns:
                continue
            
            # Verificar se as colunas são numéricas
            if pd.api.types.is_numeric_dtype(merged[col1]) and pd.api.types.is_numeric_dtype(merged[col2]):
                # Pular colunas booleanas para evitar erros ao calcular diferenças
                if merged[col1].dtype == bool or merged[col2].dtype == bool:
                    # Para colunas booleanas, calcular a % de valores iguais
                    equality = (merged[col1] == merged[col2])
                    match_pct = equality.mean() * 100
                    numeric_diffs[col] = {
                        'is_boolean': True,
                        'match_percentage': match_pct,
                        'values_same': match_pct
                    }
                    # Se houver discrepância significativa, adicionar à lista
                    if match_pct < 99:
                        high_diff_cols.append((col, 100-match_pct, 1.0))
                    continue
                
                # Calcular diferenças para colunas numéricas não-booleanas
                diff = merged[col1] - merged[col2]
                abs_diff = diff.abs()
                
                # Estatísticas das diferenças
                numeric_diffs[col] = {
                    'mean_diff': diff.mean(),
                    'mean_abs_diff': abs_diff.mean(),
                    'max_abs_diff': abs_diff.max(),
                    'std_diff': diff.std(),
                    'values_same': (abs_diff < 1e-10).mean() * 100  # % valores idênticos
                }
                
                # Identificar colunas com grandes diferenças
                if abs_diff.mean() > 0.01 or abs_diff.max() > 0.1:
                    high_diff_cols.append((col, abs_diff.mean(), abs_diff.max()))
        
        # Ordenar colunas por diferença média absoluta
        if high_diff_cols:
            high_diff_cols.sort(key=lambda x: x[1], reverse=True)
            
            print("\nColunas com diferenças significativas:")
            for col, mean_diff, max_diff in high_diff_cols[:10]:  # Top 10
                print(f"  {col}: Média abs diff = {mean_diff:.6f}, Max abs diff = {max_diff:.6f}")
            
            # Salvar colunas com alta diferença em arquivo
            high_diff_df = pd.DataFrame(high_diff_cols, columns=['column', 'mean_abs_diff', 'max_abs_diff'])
            high_diff_df.to_csv(os.path.join(OUTPUT_DIR, f"{prefix}_high_diff_columns.csv"), index=False)
            
            # Visualizar distribuição das diferenças para top 5 colunas
            plt.figure(figsize=(12, 10))
            
            for i, (col, _, _) in enumerate(high_diff_cols[:5]):
                plt.subplot(3, 2, i+1)
                
                col1 = f"{col}_{name1}"
                col2 = f"{col}_{name2}"
                
                diff = merged[col1] - merged[col2]
                
                sns.histplot(diff, kde=True)
                plt.title(f"Diferenças para {col}")
                plt.xlabel(f"{col1} - {col2}")
                plt.tight_layout()
            
            plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_diff_distributions.png"))
            
            # Visualizar correlação entre valores originais e processados
            plt.figure(figsize=(15, 10))
            
            for i, (col, _, _) in enumerate(high_diff_cols[:5]):
                plt.subplot(2, 3, i+1)
                
                col1 = f"{col}_{name1}"
                col2 = f"{col}_{name2}"
                
                plt.scatter(merged[col1], merged[col2], alpha=0.5)
                plt.plot([merged[col1].min(), merged[col1].max()], 
                         [merged[col1].min(), merged[col1].max()], 'r--')
                plt.title(f"Correlação para {col}")
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.tight_layout()
            
            plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_value_correlations.png"))
        
        # Retornar estatísticas
        return {
            "common_cols": len(common_cols),
            "cols_only_in_df1": len(cols_only_in_df1),
            "cols_only_in_df2": len(cols_only_in_df2),
            "merged_records": len(merged),
            "numeric_diffs": numeric_diffs,
            "high_diff_cols": high_diff_cols
        }
    
    else:
        print("Sem coluna de email para alinhar registros!")
        return {
            "common_cols": len(common_cols),
            "cols_only_in_df1": len(cols_only_in_df1),
            "cols_only_in_df2": len(cols_only_in_df2)
        }

def run_pipeline_stages_with_diagnostics():
    """
    Executa cada etapa da pipeline individualmente e compara com dados de treinamento.
    """
    print("\n===== EXECUTANDO PIPELINE POR ETAPAS COM DIAGNÓSTICO =====")
    
    # Carregar datasets de referência do treinamento
    datasets = load_datasets()
    
    # Carregar dados de teste originais
    test_df = datasets['original']
    
    print(f"\n===== ETAPA 1: NORMALIZAÇÃO DE EMAIL =====")
    # Configurar e ajustar transformador de email
    email_transformer = EmailNormalizationTransformer(email_col='email')
    email_transformer.fit(test_df)
    
    # Aplicar transformação
    df_email = email_transformer.transform(test_df.copy())
    
    # Salvar resultado intermediário
    email_path = os.path.join(OUTPUT_DIR, "1_email_normalized.csv")
    df_email.to_csv(email_path, index=False)
    print(f"Dados após normalização de email salvos em: {email_path}")
    
    print(f"\n===== ETAPA 2: PRÉ-PROCESSAMENTO BÁSICO =====")
    # Configurar e ajustar transformador de pré-processamento
    params_path = os.path.join(PARAMS_DIR, "all_preprocessing_params.joblib")
    preproc_transformer = CompletePreprocessingTransformer(params_path=params_path)
    preproc_transformer.fit(df_email)
    
    # Aplicar transformação
    df_preproc = preproc_transformer.transform(df_email.copy())
    
    # Salvar resultado intermediário
    preproc_path = os.path.join(OUTPUT_DIR, "2_preprocessed.csv")
    df_preproc.to_csv(preproc_path, index=False)
    print(f"Dados após pré-processamento básico salvos em: {preproc_path}")
    
    # Comparar com dados pré-processados durante treino
    if 'train_preproc' in datasets:
        compare_dataframes(
            df_preproc, 
            datasets['train_preproc'],
            name1="Inferência Preproc", 
            name2="Treino Preproc",
            prefix="2_preproc"
        )
    
    print(f"\n===== ETAPA 3: PROCESSAMENTO DE TEXTO =====")
    # Configurar e ajustar transformador de texto
    script03_params_path = os.path.join(PARAMS_DIR, "script03_params.joblib")
    text_transformer = TextFeatureEngineeringTransformer(
        params_path=params_path,
        script03_params_path=script03_params_path
    )
    text_transformer.fit(df_preproc)
    
    # Aplicar transformação
    df_text = text_transformer.transform(df_preproc.copy())
    
    # Salvar resultado intermediário
    text_path = os.path.join(OUTPUT_DIR, "3_text_processed.csv")
    df_text.to_csv(text_path, index=False)
    print(f"Dados após processamento de texto salvos em: {text_path}")
    
    # Comparar com dados processados durante treino após script 03
    if 'train_nlp' in datasets:
        compare_dataframes(
            df_text, 
            datasets['train_nlp'],
            name1="Inferência NLP", 
            name2="Treino NLP (Script 03)",
            prefix="3_nlp"
        )
    
    # Comparar com dados processados durante treino após script 04
    if 'train_final' in datasets:
        compare_dataframes(
            df_text, 
            datasets['train_final'],
            name1="Inferência Text Final", 
            name2="Treino Text Final (Script 04)",
            prefix="4_text_final"
        )
    
    print(f"\n===== ETAPA 4: PREDIÇÃO GMM =====")
    # Configurar e ajustar preditor GMM
    predictor = GMM_InferenceWrapper(models_dir=MODELS_DIR)
    predictor.fit()
    
    # Gerar probabilidades
    probas = predictor.predict_proba(df_text)
    
    # Criar DataFrame com resultados
    result_df = pd.DataFrame({
        'email': df_text['email'] if 'email' in df_text.columns else None,
        'probability': probas[:, 1],
        'prediction': (probas[:, 1] >= predictor.threshold).astype(int)
    })
    
    # Salvar resultado final
    result_path = os.path.join(OUTPUT_DIR, "4_predictions.csv")
    result_df.to_csv(result_path, index=False)
    print(f"Predições finais salvas em: {result_path}")
    
    # Carregar e comparar com resultados originais
    original_results_path = os.path.join(PROJECT_ROOT, "reports/calibration_validation_three_models/20250512_073645/gmm_test_results.csv")
    if os.path.exists(original_results_path):
        original_results = pd.read_csv(original_results_path)
        
        compare_dataframes(
            result_df,
            original_results,
            name1="Inferência Final",
            name2="Treino Final",
            prefix="5_final"
        )
    
    print("\n===== DIAGNÓSTICO COMPLETO =====")
    print(f"Todos os arquivos de diagnóstico foram salvos em: {OUTPUT_DIR}")

def examine_model_components():
    """
    Examina os componentes do modelo (PCA, GMM, Scaler) para verificar configurações.
    """
    print("\n===== EXAMINANDO COMPONENTES DO MODELO =====")
    
    # Verificar componentes do modelo
    pca_path = os.path.join(MODELS_DIR, "pca_model.joblib")
    gmm_path = os.path.join(MODELS_DIR, "gmm_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler_model.joblib")
    
    # Examinar PCA
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        print(f"\nPCA:")
        print(f"  Número de componentes: {pca.n_components_}")
        print(f"  Variância explicada: {pca.explained_variance_ratio_.sum():.2f}")
        
        # Salvar gráfico da variância explicada acumulada
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.title('Variância Acumulada Explicada pelo PCA')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Variância Acumulada Explicada')
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "pca_variance.png"))
    
    # Examinar GMM
    if os.path.exists(gmm_path):
        gmm = joblib.load(gmm_path)
        print(f"\nGMM:")
        print(f"  Número de componentes: {gmm.n_components}")
        print(f"  Covariância tipo: {gmm.covariance_type}")
    
    # Examinar Scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"\nScaler:")
        print(f"  Média de algumas features: {scaler.mean_[:5]}")
        print(f"  Escala de algumas features: {scaler.scale_[:5]}")
        
        # Salvar histogramas das médias e desvios
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(scaler.mean_, bins=30)
        plt.title('Distribuição das Médias do Scaler')
        plt.xlabel('Valor da Média')
        plt.ylabel('Frequência')
        
        plt.subplot(1, 2, 2)
        plt.hist(scaler.scale_, bins=30)
        plt.title('Distribuição dos Desvios Padrão do Scaler')
        plt.xlabel('Valor do Desvio Padrão')
        plt.ylabel('Frequência')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "scaler_stats.png"))

def examine_tfidf_vectorizers():
    """
    Examina os vetorizadores TF-IDF para verificar compatibilidade.
    """
    print("\n===== EXAMINANDO VETORIZADORES TF-IDF =====")
    
    # Verificar vetorizadores do script 03
    script03_vectorizers_path = os.path.join(DATA_DIR, "02_2_reprocessed/models/tfidf_vectorizers.joblib")
    
    if os.path.exists(script03_vectorizers_path):
        vectorizers = joblib.load(script03_vectorizers_path)
        print(f"Vetorizadores script 03 carregados de: {script03_vectorizers_path}")
        print(f"  Número de vetorizadores: {len(vectorizers)}")
        print(f"  Chaves: {list(vectorizers.keys())}")
        
        # Verificar cada vetorizador
        for key, vectorizer in vectorizers.items():
            print(f"\n  Vetorizador para '{key}':")
            print(f"    Número de features: {len(vectorizer.get_feature_names_out())}")
            print(f"    Algumas features: {vectorizer.get_feature_names_out()[:5]}")
    else:
        print(f"Arquivo de vetorizadores não encontrado: {script03_vectorizers_path}")
    
    # Verificar vetorizadores do script 04 (nos parâmetros)
    print("\n=== Parâmetros do script 04 ===")
    params_path = os.path.join(PARAMS_DIR, "script03_params.joblib")
    
    if os.path.exists(params_path):
        params = joblib.load(params_path)
        print(f"Parâmetros carregados de: {params_path}")
        
        if 'vectorizers' in params:
            print(f"  Número de vetorizadores nos parâmetros: {len(params['vectorizers'])}")
            print(f"  Chaves: {list(params['vectorizers'].keys())}")
        else:
            print("  Chave 'vectorizers' não encontrada nos parâmetros")
    else:
        print(f"Arquivo de parâmetros não encontrado: {params_path}")

def main():
    """Função principal do diagnóstico"""
    print("=" * 60)
    print("DIAGNÓSTICO DA PIPELINE DE INFERÊNCIA DO SMART ADS")
    print("=" * 60)
    
    try:
        # Examinar componentes do modelo
        examine_model_components()
        
        # Examinar vetorizadores TF-IDF
        examine_tfidf_vectorizers()
        
        # Executar pipeline com diagnóstico
        run_pipeline_stages_with_diagnostics()
        
        print("\n===== DIAGNÓSTICO CONCLUÍDO COM SUCESSO =====")
        print(f"Todos os resultados foram salvos em: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\nERRO durante o processo de diagnóstico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()