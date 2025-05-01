#!/usr/bin/env python
"""
Script para executar a pipeline de análise de importância e seleção de features
usando Random Forest para cada grupo de países gerado anteriormente.

Este script:
1. Itera sobre todos os subdiretórios no diretório de entrada
2. Aplica a seleção de features para cada conjunto de dados
3. Salva apenas os datasets com features selecionadas no diretório de saída
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import argparse
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import shutil

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

# Imports dos módulos de avaliação
try:
    from src.evaluation import feature_importance as fi
    from src.evaluation import feature_selector as fs
except ImportError:
    print("Aviso: Não foi possível importar os módulos de avaliação. Algumas funcionalidades podem não estar disponíveis.")
    # Implementações básicas para quando os módulos não estão disponíveis
    class FallbackModule:
        @staticmethod
        def identify_launch_column(df):
            """Tenta identificar coluna de lançamento."""
            for col in df.columns:
                if 'launch' in col.lower() or 'lançamento' in col.lower():
                    return col
            return None
            
        @staticmethod
        def select_numeric_features(df, target_col):
            """Seleciona colunas numéricas, excluindo o target."""
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            return numeric_cols
            
        @staticmethod
        def identify_text_derived_columns(cols):
            """Identifica colunas derivadas de texto."""
            text_cols = []
            for col in cols:
                if any(kw in col.lower() for kw in ['text', 'tfidf', 'word', 'sentiment', 'topic', 'embed']):
                    text_cols.append(col)
            return text_cols
            
        @staticmethod
        def sanitize_column_names(cols):
            """Sanitiza nomes de colunas para evitar problemas."""
            rename_dict = {}
            for col in cols:
                if any(c in col for c in [' ', '/', '\\', '?', '%', '*', ':', '|', '"', '<', '>', '.', ',']):
                    new_name = col.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    new_name = new_name.replace('?', '').replace('%', 'pct').replace('*', '_')
                    new_name = new_name.replace(':', '_').replace('|', '_').replace('"', '')
                    new_name = new_name.replace('<', 'lt').replace('>', 'gt').replace('.', '_')
                    new_name = new_name.replace(',', '_')
                    rename_dict[col] = new_name
            return rename_dict
            
        @staticmethod
        def analyze_multicollinearity(X, threshold=0.8):
            """Analisa multicolinearidade entre features."""
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = []
            for col in upper.columns:
                for idx, val in upper[col].items():
                    if val >= threshold:
                        high_corr_pairs.append((idx, col, val))
            return high_corr_pairs
            
        @staticmethod
        def compare_country_encodings(X, y):
            """Compara diferentes codificações de país (stub)."""
            pass
            
        @staticmethod
        def analyze_launch_robustness(train_df, X, y, numeric_cols, launch_col, rename_dict, importance_df):
            """Analisa robustez entre lançamentos (stub)."""
            return None, None, None, None
            
        @staticmethod
        def identify_target_column(df):
            """Identifica a coluna target."""
            if 'target' in df.columns:
                return 'target'
            for col in df.columns:
                if col.lower() == 'target' or col.lower() == 'label' or col.lower() == 'y':
                    return col
            return None
    
    # Atribuir o módulo de fallback para uso posterior
    fi = FallbackModule()
    fs = FallbackModule()

def analyze_rf_importance(X, y, feature_names, cv=5):
    """
    Análise de importância de features com Random Forest usando validação cruzada.
    
    Args:
        X: Matriz de features
        y: Vetor target
        feature_names: Lista com nomes das features
        cv: Número de folds para validação cruzada
        
    Returns:
        DataFrame com importância das features e métricas
    """
    print("  Analisando importância de features...")
    
    # Configuração do modelo com classe balanceada para tratar desbalanceamento
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5, 
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    # Validação cruzada para avaliação robusta
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    importance_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Treinar modelo
        rf.fit(X_train_fold, y_train_fold)
        
        # Armazenar importância
        fold_importance = rf.feature_importances_
        importance_scores.append(fold_importance)
        
        # Avaliar
        y_pred_proba = rf.predict_proba(X_val_fold)[:, 1]
        threshold = 0.17  # Threshold otimizado baseado em execuções anteriores
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Métricas
        precision = precision_score(y_val_fold, y_pred)
        recall = recall_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred)
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)
    
    # Calcular importância média e desvio padrão entre folds
    mean_importance = np.mean(importance_scores, axis=0)
    std_importance = np.std(importance_scores, axis=0)
    
    # Criar DataFrame com resultados
    results = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importance,
        'importance_std': std_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Calcular médias das métricas
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_auc = np.mean(auc_scores)
    
    metrics = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'auc': avg_auc
    }
    
    print(f"  Métricas CV: F1={avg_f1:.4f}, AUC={avg_auc:.4f}")
    
    # Treinar modelo final em todos os dados
    rf.fit(X, y)
    
    return results, metrics, rf

def analyze_feature_stability(X, y, feature_names, results, n_iterations=5):
    """
    Analisa a estabilidade da importância das features através de múltiplas 
    execuções com diferentes seeds.
    
    Args:
        X: Matriz de features
        y: Vetor target
        feature_names: Lista com nomes das features
        results: DataFrame com importância das features
        n_iterations: Número de iterações para testar estabilidade
        
    Returns:
        DataFrame com medidas de estabilidade
    """
    print("  Analisando estabilidade das features...")
    
    # Obter as top features do resultado principal
    top_k = min(50, len(feature_names))  # Top 50 ou menos se não houver suficientes
    top_features = results.head(top_k)['feature'].tolist()
    
    # Matriz para armazenar rankings
    rank_matrix = np.zeros((n_iterations, len(feature_names)))
    
    # Executar múltiplas iterações
    for i in range(n_iterations):
        # Criar modelo com seed diferente
        rf = RandomForestClassifier(
            n_estimators=100,  # Reduzido para performance
            max_depth=None,
            min_samples_split=5, 
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=i*10+42,  # Diferentes seeds
            class_weight='balanced'
        )
        
        # Treinar modelo
        rf.fit(X, y)
        
        # Obter importância e ranking
        importance = rf.feature_importances_
        ranks = np.zeros(len(feature_names))
        sorted_idx = np.argsort(importance)[::-1]
        
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
            
        rank_matrix[i, :] = ranks
    
    # Calcular estatísticas de estabilidade
    mean_rank = np.mean(rank_matrix, axis=0)
    std_rank = np.std(rank_matrix, axis=0)
    
    # Criar DataFrame com resultados
    stability_results = pd.DataFrame({
        'feature': feature_names,
        'mean_rank': mean_rank,
        'std_rank': std_rank,
        'in_top_k': [1 if feat in top_features else 0 for feat in feature_names],
        'original_importance': [results[results['feature'] == feat]['importance'].values[0] 
                               if feat in results['feature'].values else 0 
                               for feat in feature_names]
    })
    
    # Ordenar por ranking médio
    stability_results = stability_results.sort_values('mean_rank').reset_index(drop=True)
    
    # Identificar features estáveis vs instáveis
    threshold = 0.6  # Limiar para considerar uma feature estável
    stability_results['is_stable'] = stability_results['std_rank'] <= threshold * stability_results['mean_rank']
    
    # Contar quantas das top features são estáveis
    stable_top_features = sum((stability_results['in_top_k'] == 1) & (stability_results['is_stable'] == True))
    
    return stability_results

def select_features_rf(X, y, feature_names, importance_results, stability_results=None):
    """
    Seleciona features baseado nos resultados de importância do Random Forest.
    
    Args:
        X: Matriz de features
        y: Vetor target
        feature_names: Lista com nomes das features
        importance_results: DataFrame com importância das features
        stability_results: DataFrame com estabilidade das features (opcional)
        
    Returns:
        Lista de features selecionadas
    """
    print("  Selecionando features...")
    
    # Determinar limiar de importância automaticamente
    importances = importance_results['importance'].values
    sorted_imp = np.sort(importances)[::-1]
    
    # Calcular diferenças entre importâncias consecutivas
    diff = np.diff(sorted_imp)
    
    # Verificar se há um declínio significativo na importância
    max_diff_idx = np.argmax(diff)
    if diff[max_diff_idx] < 0.001:  # Se a maior diferença é pequena
        # Usar percentil para selecionar as top 15% features
        percentile_threshold = np.percentile(importances, 85)
        threshold = max(percentile_threshold, 0.001)
    else:
        # Encontrar o maior declínio na importância
        elbow_idx = max_diff_idx + 1
        
        # Definir threshold como o valor do cotovelo
        threshold = sorted_imp[elbow_idx]
    
    # Garantir pelo menos um número mínimo de features
    min_features = min(100, len(feature_names) // 8)
    if sum(importances >= threshold) < min_features:
        # Ajustar threshold para incluir pelo menos min_features
        threshold = sorted_imp[min_features-1]
    
    # Garantir um número máximo de features para evitar overfitting
    max_features = 250
    if sum(importances >= threshold) > max_features:
        # Ajustar threshold para limitar a max_features
        threshold = sorted_imp[max_features-1]
    
    # Selecionar features acima do threshold
    selected = importance_results[importance_results['importance'] >= threshold]
    selected_features = selected['feature'].tolist()
    
    # Se temos dados de estabilidade, dar preferência a features estáveis
    if stability_results is not None:
        # Adicionar features estáveis que estão próximas do threshold
        stable_features = stability_results[stability_results['is_stable'] == True]['feature'].tolist()
        almost_selected = importance_results[
            (importance_results['importance'] >= threshold * 0.8) & 
            (importance_results['importance'] < threshold)
        ]
        
        for _, row in almost_selected.iterrows():
            if row['feature'] in stable_features and row['feature'] not in selected_features:
                selected_features.append(row['feature'])
    
    print(f"  Selecionadas {len(selected_features)} features")
    
    return selected_features

def handle_collinearity_in_selection(selected_features, high_corr_pairs, importance_results, X):
    """
    Remove features colineares da seleção final, mantendo as mais importantes.
    
    Args:
        selected_features: Lista de features selecionadas
        high_corr_pairs: Lista de pares de features altamente correlacionadas
        importance_results: DataFrame com importância das features
        X: Matriz de features
        
    Returns:
        Tuple com (lista de features final, lista de features removidas)
    """
    print("  Removendo features colineares...")
    
    # Criar um dicionário de importância para fácil acesso
    importance_dict = dict(zip(
        importance_results['feature'], 
        importance_results['importance']
    ))
    
    # Se não temos pares correlacionados identificados, computá-los diretamente
    formatted_pairs = []
    if not high_corr_pairs or len(high_corr_pairs) == 0:
        # Calcular matriz de correlação para as features selecionadas
        if len(selected_features) > 1:
            X_selected = X[selected_features]
            corr_matrix = X_selected.corr().abs()
            
            # Extrair pares altamente correlacionados (threshold = 0.8)
            for i in range(len(selected_features)):
                for j in range(i+1, len(selected_features)):
                    feat1 = selected_features[i]
                    feat2 = selected_features[j]
                    corr = corr_matrix.iloc[i, j]
                    if corr >= 0.8:
                        formatted_pairs.append((feat1, feat2, corr))
    else:
        # Converter high_corr_pairs ao formato esperado se necessário
        for pair in high_corr_pairs:
            if isinstance(pair, tuple) and len(pair) == 3:
                formatted_pairs.append(pair)
            elif isinstance(pair, list) and len(pair) == 2:
                formatted_pairs.append((pair[0], pair[1], 0.8))
        
        # Se ainda não temos pares formatados, verificar se é um DataFrame
        if not formatted_pairs and isinstance(high_corr_pairs, pd.DataFrame):
            if 'feature1' in high_corr_pairs.columns and 'feature2' in high_corr_pairs.columns:
                for _, row in high_corr_pairs.iterrows():
                    corr = row.get('correlation', 0.8)
                    formatted_pairs.append((row['feature1'], row['feature2'], corr))
    
    # Identificar pares correlacionados onde ambas features estão na seleção
    correlated_selected_pairs = []
    for feat1, feat2, corr in formatted_pairs:
        if feat1 in selected_features and feat2 in selected_features:
            # Ordenar par pela importância (mais importante primeiro)
            if importance_dict.get(feat1, 0) >= importance_dict.get(feat2, 0):
                correlated_selected_pairs.append((feat1, feat2, corr))
            else:
                correlated_selected_pairs.append((feat2, feat1, corr))
    
    # Remover features menos importantes em cada par
    removed_features = set()
    for feat1, feat2, corr in correlated_selected_pairs:
        # Se a segunda feature não foi removida por outro par
        if feat2 not in removed_features:
            removed_features.add(feat2)
    
    # Criar lista final de features
    final_features = [f for f in selected_features if f not in removed_features]
    
    print(f"  Seleção final: {len(final_features)} features (removidas {len(removed_features)} devido à correlação)")
    
    return final_features, list(removed_features)

def run_feature_importance_analysis_rf(train_df, target_col):
    """
    Executa a análise de importância das features usando apenas Random Forest
    e retorna features selecionadas.
    
    Args:
        train_df: DataFrame de treinamento
        target_col: Nome da coluna target
        
    Returns:
        Lista de features selecionadas e parâmetros
    """
    print("\n=== Executando análise de importância de features com Random Forest ===")
    
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
    
    # 6. Análise de multicolinearidade (simplificada)
    high_corr_pairs = fi.analyze_multicollinearity(X)
    
    # 7. Análise de importância com Random Forest
    importance_results, metrics, rf_model = analyze_rf_importance(
        X, y, numeric_cols, cv=5
    )
    
    # 8. Análise de estabilidade das features
    stability_results = analyze_feature_stability(X, y, numeric_cols, importance_results)
    
    # 9. Selecionar features finais
    selected_features = select_features_rf(X, y, numeric_cols, importance_results, stability_results)
    
    # 10. Ajustar seleção final com base na multicolinearidade
    final_selected_features, removed_due_to_correlation = handle_collinearity_in_selection(
        selected_features, high_corr_pairs, importance_results, X
    )
    
    # 11. Preparar parâmetros para retorno
    selection_params = {
        'selected_features': final_selected_features,
        'rename_dict': rename_dict,
        'target_col': target_col,
        'metrics': metrics
    }
    
    print(f"Total de {len(final_selected_features)} features selecionadas para o modelo.")
    return final_selected_features, selection_params

def apply_feature_selection(df, selected_features, target_col):
    """
    Aplica a seleção de features a um DataFrame, adicionando features faltantes quando necessário.
    
    Args:
        df: DataFrame a processar
        selected_features: Lista de features selecionadas
        target_col: Nome da coluna target
        
    Returns:
        DataFrame com as features selecionadas e coluna target
    """
    # Verificar quais features selecionadas existem no DataFrame
    available_features = [col for col in selected_features if col in df.columns]
    missing_features = set(selected_features) - set(available_features)
    
    # Criar o DataFrame com as features disponíveis + target
    columns_to_keep = available_features + [target_col]
    df_selected = df[columns_to_keep]
    
    # Adicionar features faltantes preenchidas com zeros
    if missing_features:
        print(f"Adicionando {len(missing_features)} features faltantes, preenchidas com zeros")
        
        # Criar DataFrame com as features faltantes preenchidas com zeros
        missing_df = pd.DataFrame(0, index=df.index, columns=list(missing_features))
        
        # Concatenar com o DataFrame existente
        df_selected = pd.concat([df_selected, missing_df], axis=1)
        
        # Garantir que as colunas estejam na mesma ordem que as features selecionadas
        all_columns = selected_features + [target_col]
        df_selected = df_selected[all_columns]
    
    print(f"DataFrame processado: {df.shape[1]} colunas originais → {df_selected.shape[1]} colunas selecionadas")
    return df_selected

def process_country_directory(country_dir, output_base_dir):
    """
    Processa um diretório de país, aplicando seleção de features e salvando resultados.
    
    Args:
        country_dir: Caminho para o diretório do país
        output_base_dir: Diretório base para salvar os resultados
        
    Returns:
        True se bem-sucedido, False caso contrário
    """
    country_name = os.path.basename(country_dir)
    print(f"\n\n{'='*80}")
    print(f"Processando país/grupo: {country_name}")
    print(f"{'='*80}")
    
    # Definir diretório de saída para este país/grupo
    output_dir = os.path.join(output_base_dir, country_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar existência dos arquivos
    train_path = os.path.join(country_dir, "train.csv")
    val_path = os.path.join(country_dir, "validation.csv")
    test_path = os.path.join(country_dir, "test.csv")
    
    if not os.path.exists(train_path):
        print(f"Erro: Arquivo de treinamento não encontrado em {train_path}")
        return False
    
    # Carregar o dataset de treinamento
    print(f"Carregando dataset de treinamento de {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Dataset de treinamento carregado: {train_df.shape[0]} linhas, {train_df.shape[1]} colunas")
    
    # Identificar coluna target
    target_col = 'target'  # Assumimos que a coluna target é 'target'
    if target_col not in train_df.columns:
        # Tentar identificar automaticamente
        target_col = fi.identify_target_column(train_df)
        if not target_col:
            print(f"Erro: Coluna target não encontrada no dataset")
            return False
    
    try:
        # Analisar importância de features e selecionar as relevantes
        selected_features, selection_params = run_feature_importance_analysis_rf(
            train_df, target_col
        )
        
        # Aplicar seleção de features ao conjunto de treinamento
        print("\n--- Aplicando seleção de features ao dataset de treinamento ---")
        train_selected = apply_feature_selection(train_df, selected_features, target_col)
        
        # Salvar conjunto de treino processado
        train_output_path = os.path.join(output_dir, "train.csv")
        train_selected.to_csv(train_output_path, index=False)
        print(f"Dataset de treino com features selecionadas salvo em {train_output_path}")
        
        # Processar e salvar conjuntos de validação e teste (se existirem)
        if os.path.exists(val_path):
            print("\n--- Aplicando seleção de features ao dataset de validação ---")
            val_df = pd.read_csv(val_path)
            val_selected = apply_feature_selection(val_df, selected_features, target_col)
            val_output_path = os.path.join(output_dir, "validation.csv")
            val_selected.to_csv(val_output_path, index=False)
            print(f"Dataset de validação com features selecionadas salvo em {val_output_path}")
        
        if os.path.exists(test_path):
            print("\n--- Aplicando seleção de features ao dataset de teste ---")
            test_df = pd.read_csv(test_path)
            test_selected = apply_feature_selection(test_df, selected_features, target_col)
            test_output_path = os.path.join(output_dir, "test.csv")
            test_selected.to_csv(test_output_path, index=False)
            print(f"Dataset de teste com features selecionadas salvo em {test_output_path}")
        
        # Copiar arquivo de grupo (se existir)
        group_info_path = os.path.join(country_dir, "group_info.json")
        if os.path.exists(group_info_path):
            shutil.copy2(group_info_path, os.path.join(output_dir, "group_info.json"))
        
        print(f"\nProcessamento concluído para {country_name}!")
        return True
        
    except Exception as e:
        print(f"Erro ao processar {country_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Pipeline de seleção de features por país.")
    parser.add_argument("--input-dir", type=str, 
                        default="data/02_4_processed_text_selection_per_country", 
                        help="Diretório base contendo os diretórios por país")
    parser.add_argument("--output-dir", type=str, 
                        default="data/03_4_feature_selection_text_per_country", 
                        help="Diretório base para salvar os resultados")
    
    args = parser.parse_args()
    
    # Criar diretório de saída base
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Listar subdiretórios (países/grupos)
    country_dirs = []
    for item in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path):
            country_dirs.append(item_path)
    
    if not country_dirs:
        print(f"Erro: Nenhum subdiretório encontrado em {args.input_dir}")
        return 1
    
    print(f"Encontrados {len(country_dirs)} países/grupos para processar")
    
    # Processar cada país/grupo
    results = {}
    for country_dir in country_dirs:
        country_name = os.path.basename(country_dir)
        success = process_country_directory(country_dir, args.output_dir)
        results[country_name] = success
    
    # Resumo final
    print("\n\n====== RESUMO DO PROCESSAMENTO ======")
    successful = sum(1 for success in results.values() if success)
    print(f"Total de países/grupos processados com sucesso: {successful}/{len(results)}")
    
    # Listar falhas (se houver)
    failures = [country for country, success in results.items() if not success]
    if failures:
        print(f"Falhas ({len(failures)}): {', '.join(failures)}")
    
    print(f"\nResultados salvos em {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())