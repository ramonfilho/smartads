#!/usr/bin/env python
"""
Script para executar a pipeline completa de análise de importância de features
e seleção de features relevantes para o modelo, garantindo consistência entre
os conjuntos de treino, validação e teste.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import argparse
import warnings

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

# Imports dos módulos de avaliação
from src.evaluation import feature_importance as fi
from src.evaluation import feature_selector as fs

def create_output_directories(output_dir, params_dir):
    """Cria diretórios para salvar resultados e parâmetros."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "importance_results"), exist_ok=True)
    if params_dir:
        os.makedirs(params_dir, exist_ok=True)
    print(f"Diretórios de saída criados em '{output_dir}'")

def run_feature_importance_analysis(train_df, target_col, output_dir, params_dir=None):
    """
    Executa a análise de importância das features e retorna features selecionadas.
    
    Args:
        train_df: DataFrame de treinamento
        target_col: Nome da coluna target
        output_dir: Diretório para salvar resultados
        params_dir: Diretório para salvar parâmetros (opcional)
        
    Returns:
        Lista de features selecionadas e parâmetros
    """
    print("\n=== Executando análise de importância de features ===")
    
    # 1. Identificar coluna de lançamento (se existir)
    launch_col = fi.identify_launch_column(train_df)
    
    # 2. Selecionar features numéricas para análise
    numeric_cols = fi.select_numeric_features(train_df, target_col)
    
    # 3. Identificar colunas derivadas de texto
    text_derived_cols = fi.identify_text_derived_columns(numeric_cols)
    
    # 4. Sanitizar nomes das colunas
    rename_dict = fi.sanitize_column_names(numeric_cols)
    
    # Aplicar renaming se necessário
    if rename_dict:
        print(f"Renomeando {len(rename_dict)} colunas para evitar erros com caracteres especiais")
        train_df = train_df.rename(columns=rename_dict)
        
        # Atualizar listas
        numeric_cols = [rename_dict.get(col, col) for col in numeric_cols]
        text_derived_cols = [rename_dict.get(col, col) for col in text_derived_cols]
        if launch_col in rename_dict:
            launch_col = rename_dict[launch_col]
    
    # 5. Preparar dados para modelagem
    X = train_df[numeric_cols].fillna(0)
    y = train_df[target_col]
    
    print(f"Usando {len(numeric_cols)} features numéricas para análise")
    print(f"Distribuição do target: {y.value_counts(normalize=True) * 100}")
    
    # 6. Análise de multicolinearidade
    high_corr_pairs = fi.analyze_multicollinearity(X)
    
    # 7. Análise específica: codificação de países (se aplicável)
    fi.compare_country_encodings(X, y)
    
    # 8. Análise de importância com múltiplos modelos
    print("\n--- Iniciando análise de importância de features ---")
    
    # 8.1 - RandomForest
    rf_importance, rf_metrics = fi.analyze_rf_importance(X, y, numeric_cols)
    
    # 8.2 - LightGBM
    lgb_importance, lgb_metrics = fi.analyze_lgb_importance(X, y, numeric_cols)
    
    # 8.3 - XGBoost
    xgb_importance, xgb_metrics = fi.analyze_xgb_importance(X, y, numeric_cols)
    
    # 9. Combinar resultados de importância
    final_importance = fi.combine_importance_results(rf_importance, lgb_importance, xgb_importance)
    
    # 10. Salvar resultados de importância combinados
    importance_results_path = os.path.join(output_dir, "importance_results", "feature_importance_combined.csv")
    final_importance.to_csv(importance_results_path, index=False)
    print(f"\nImportância das features salva em {importance_results_path}")
    
    # 11. Análise de robustez entre lançamentos (se aplicável)
    if launch_col:
        launch_importance, unstable_features, launch_vs_global, consistent_features = fi.analyze_launch_robustness(
            train_df, X, y, numeric_cols, launch_col, rename_dict, final_importance
        )
        
        # Salvar análise de robustez entre lançamentos
        if launch_vs_global is not None:
            robustness_path = os.path.join(output_dir, "importance_results", "feature_robustness_analysis.csv")
            launch_vs_global.to_csv(robustness_path, index=False)
            print(f"\nAnálise de robustez entre lançamentos salva em {robustness_path}")
    
    # 12. Identificar features potencialmente irrelevantes
    potentially_irrelevant, irrelevant_by_importance, irrelevant_by_variance, irrelevant_by_correlation = (
        fs.identify_irrelevant_features(final_importance, high_corr_pairs)
    )
    
    # 13. Análise de features textuais
    text_importance = fs.analyze_text_features(final_importance, text_derived_cols)
    
    # Salvar análise de features textuais (se aplicável)
    if text_importance is not None:
        text_analysis_path = os.path.join(output_dir, "importance_results", "text_features_importance.csv")
        text_importance.to_csv(text_analysis_path, index=False)
        print(f"Análise de features textuais salva em {text_analysis_path}")
    
    # 14. Selecionar features finais
    selected_features, features_to_remove_corr, unrecommended_features = fs.select_final_features(
        final_importance, high_corr_pairs, numeric_cols, rename_dict
    )
    
    # 15. Documentar seleção de features
    fs.document_feature_selections(
        selected_features, unrecommended_features, 
        final_importance, high_corr_pairs, rename_dict, 
        os.path.join(output_dir, "importance_results")
    )
    
    # 16. Resumo por categorias
    fs.summarize_feature_categories(numeric_cols, final_importance, text_derived_cols)
    
    # 17. Salvar parâmetros para uso futuro
    selection_params = {
        'selected_features': selected_features,
        'rename_dict': rename_dict,
        'target_col': target_col
    }
    
    if params_dir:
        params_path = os.path.join(params_dir, "feature_selection_params.joblib")
        joblib.dump(selection_params, params_path)
        print(f"\nParâmetros de seleção de features salvos em {params_path}")
    
    print(f"\nTotal de {len(selected_features)} features selecionadas para o modelo.")
    return selected_features, selection_params

def apply_feature_selection(df, selected_features, target_col):
    """
    Aplica a seleção de features a um DataFrame.
    
    Args:
        df: DataFrame a processar
        selected_features: Lista de features selecionadas
        target_col: Nome da coluna target
        
    Returns:
        DataFrame com apenas as features selecionadas e coluna target
    """
    # Verificar quais features selecionadas existem no DataFrame
    available_features = [col for col in selected_features if col in df.columns]
    missing_features = set(selected_features) - set(available_features)
    
    if missing_features:
        print(f"Aviso: {len(missing_features)} features selecionadas não foram encontradas no DataFrame")
        if len(missing_features) <= 10:
            print(f"Features ausentes: {list(missing_features)}")
        else:
            print(f"Exemplos de features ausentes: {list(missing_features)[:10]}...")
    
    # Selecionar apenas as features disponíveis + target
    columns_to_keep = available_features + [target_col]
    df_selected = df[columns_to_keep]
    
    print(f"DataFrame reduzido: de {df.shape[1]} para {df_selected.shape[1]} colunas")
    return df_selected

def process_datasets(input_dir, output_dir, params_dir=None):
    """
    Função principal que processa todos os conjuntos de dados.
    
    Args:
        input_dir: Diretório contendo os arquivos de entrada
        output_dir: Diretório para salvar os arquivos processados
        params_dir: Diretório para salvar os parâmetros (opcional)
    
    Returns:
        Dicionário com os DataFrames processados e parâmetros
    """
    # 1. Criar diretórios de saída
    create_output_directories(output_dir, params_dir)
    
    # 2. Definir caminhos dos datasets
    train_path = os.path.join(input_dir, "train.csv")
    cv_path = os.path.join(input_dir, "validation.csv")
    test_path = os.path.join(input_dir, "test.csv")
    
    # Verificar se os arquivos existem
    print(f"Verificando existência dos arquivos de entrada:")
    print(f"  Train path: {train_path} - Existe: {os.path.exists(train_path)}")
    print(f"  CV path: {cv_path} - Existe: {os.path.exists(cv_path)}")
    print(f"  Test path: {test_path} - Existe: {os.path.exists(test_path)}")
    
    if not os.path.exists(train_path):
        print("ERRO: Arquivo de treinamento não encontrado!")
        return None
    
    # 3. Carregar o dataset de treinamento
    print(f"Carregando dataset de treinamento de {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Dataset de treinamento carregado: {train_df.shape[0]} linhas, {train_df.shape[1]} colunas")
    
    # 4. Identificar coluna target
    target_col = fi.identify_target_column(train_df)
    
    # 5. Analisar importância de features e selecionar as relevantes
    selected_features, selection_params = run_feature_importance_analysis(
        train_df, target_col, output_dir, params_dir
    )
    
    # 6. Aplicar seleção de features ao conjunto de treinamento
    print("\n--- Aplicando seleção de features ao dataset de treinamento ---")
    train_selected = apply_feature_selection(train_df, selected_features, target_col)
    
    # 7. Salvar conjunto de treino processado
    train_output_path = os.path.join(output_dir, "train.csv")
    train_selected.to_csv(train_output_path, index=False)
    print(f"Dataset de treino com features selecionadas salvo em {train_output_path}")
    
    # 8. Processar e salvar conjuntos de validação e teste (se existirem)
    if os.path.exists(cv_path):
        print("\n--- Aplicando seleção de features ao dataset de validação ---")
        cv_df = pd.read_csv(cv_path)
        cv_selected = apply_feature_selection(cv_df, selected_features, target_col)
        cv_output_path = os.path.join(output_dir, "validation.csv")
        cv_selected.to_csv(cv_output_path, index=False)
        print(f"Dataset de validação com features selecionadas salvo em {cv_output_path}")
    
    if os.path.exists(test_path):
        print("\n--- Aplicando seleção de features ao dataset de teste ---")
        test_df = pd.read_csv(test_path)
        test_selected = apply_feature_selection(test_df, selected_features, target_col)
        test_output_path = os.path.join(output_dir, "test.csv")
        test_selected.to_csv(test_output_path, index=False)
        print(f"Dataset de teste com features selecionadas salvo em {test_output_path}")
    
    print("\nProcesso de seleção de features concluído!")
    print(f"Os datasets com features selecionadas foram salvos em {output_dir}/")
    
    return {
        'train': train_selected,
        'selected_features': selected_features,
        'params': selection_params
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de análise de importância e seleção de features.")
    parser.add_argument("--input-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/processed"), 
                        help="Diretório contendo os arquivos de entrada (train.csv, validation.csv, test.csv)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/feature_selection"), 
                        help="Diretório para salvar os arquivos processados")
    parser.add_argument("--params-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/src/evaluation/"), 
                        help="Diretório para salvar os parâmetros aprendidos")
    
    args = parser.parse_args()
    
    # Chamada da função principal
    results = process_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        params_dir=args.params_dir
    )
    
    if results is None:
        sys.exit(1)  # Sair com código de erro