#!/usr/bin/env python
"""
Script para executar a pipeline de análise de importância de features
usando apenas o Random Forest para selecionar features relevantes para o modelo,
garantindo consistência entre os conjuntos de treino, validação e teste.

Esta versão é otimizada para usar apenas o modelo Random Forest na seleção de features,
mantendo outras verificações importantes como análise de multicolinearidade
e análise de robustez entre lançamentos.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import argparse
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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

def analyze_rf_importance_extended(X, y, feature_names, cv=5, output_dir=None):
    """
    Análise estendida de importância de features com Random Forest usando validação cruzada.
    
    Args:
        X: Matriz de features
        y: Vetor target
        feature_names: Lista com nomes das features
        cv: Número de folds para validação cruzada
        output_dir: Diretório para salvar os gráficos
        
    Returns:
        DataFrame com importância das features e métricas
    """
    print("\n--- Análise de importância com Random Forest ---")
    
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
        
        print(f"  Fold {fold+1}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
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
    
    print(f"\n  Métricas médias: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}, AUC={avg_auc:.4f}")
    
    # Gerar gráfico com as top features
    if output_dir:
        plt.figure(figsize=(12, 10))
        top_n = min(30, len(results))
        top_features = results.head(top_n)
        
        # Criar barras com desvio padrão
        plt.barh(
            np.arange(top_n),
            top_features['importance'],
            xerr=top_features['importance_std'],
            align='center',
            alpha=0.8
        )
        
        plt.yticks(np.arange(top_n), top_features['feature'])
        plt.xlabel('Importância')
        plt.title(f'Top {top_n} Features - Random Forest')
        plt.gca().invert_yaxis()  # Maior importância no topo
        plt.tight_layout()
        
        # Salvar gráfico
        plt_path = os.path.join(output_dir, "importance_results", "rf_feature_importance.png")
        plt.savefig(plt_path)
        plt.close()
        print(f"  Gráfico de importância salvo em {plt_path}")
        
        # Salvar também em CSV
        csv_path = os.path.join(output_dir, "importance_results", "rf_feature_importance.csv")
        results.to_csv(csv_path, index=False)
        print(f"  Resultados de importância salvos em {csv_path}")
    
    # Salvar modelo final treinado em todos os dados
    rf.fit(X, y)
    metrics = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'auc': avg_auc
    }
    
    return results, metrics, rf

def analyze_feature_stability(X, y, feature_names, results, n_iterations=10, threshold=0.6):
    """
    Analisa a estabilidade da importância das features através de múltiplas 
    execuções com diferentes seeds.
    
    Args:
        X: Matriz de features
        y: Vetor target
        feature_names: Lista com nomes das features
        results: DataFrame com importância das features
        n_iterations: Número de iterações para testar estabilidade
        threshold: Limiar para considerar uma feature estável
        
    Returns:
        DataFrame com medidas de estabilidade
    """
    print("\n--- Analisando estabilidade das features ---")
    
    # Obter as top features do resultado principal
    top_k = min(50, len(feature_names))  # Top 50 ou menos se não houver suficientes
    top_features = results.head(top_k)['feature'].tolist()
    
    # Matriz para armazenar rankings
    rank_matrix = np.zeros((n_iterations, len(feature_names)))
    
    # Executar múltiplas iterações
    for i in range(n_iterations):
        # Criar modelo com seed diferente
        rf = RandomForestClassifier(
            n_estimators=200,
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
    stability_results['is_stable'] = stability_results['std_rank'] <= threshold * stability_results['mean_rank']
    
    # Contar quantas das top features são estáveis
    stable_top_features = sum((stability_results['in_top_k'] == 1) & (stability_results['is_stable'] == True))
    print(f"  {stable_top_features} de {top_k} top features são estáveis ({stable_top_features/top_k*100:.1f}%)")
    
    return stability_results

def select_features_rf(X, y, feature_names, importance_results, stability_results=None, threshold=None):
    """
    Seleciona features baseado nos resultados de importância do Random Forest.
    
    Args:
        X: Matriz de features
        y: Vetor target
        feature_names: Lista com nomes das features
        importance_results: DataFrame com importância das features
        stability_results: DataFrame com estabilidade das features (opcional)
        threshold: Limiar de importância (opcional, se None será calculado automaticamente)
        
    Returns:
        Lista de features selecionadas
    """
    print("\n--- Selecionando features baseado em Random Forest ---")
    
    # Determinar limiar de importância
    if threshold is None:
        # Usa método do cotovelo modificado para determinar threshold
        importances = importance_results['importance'].values
        sorted_imp = np.sort(importances)[::-1]
        
        # Calcular diferenças entre importâncias consecutivas
        diff = np.diff(sorted_imp)
        
        # Verificar se há um declínio significativo na importância
        # Se não houver declínio claro, usar abordagem de percentil
        max_diff_idx = np.argmax(diff)
        if diff[max_diff_idx] < 0.001:  # Se a maior diferença é pequena
            # Usar percentil para selecionar as top 25% features (para aumentar para ~200 features)
            percentile_threshold = np.percentile(importances, 75)  # Reduzido para aumentar número de features
            threshold = max(percentile_threshold, 0.0008)  # Threshold menor para incluir mais features
            print(f"  Usando threshold baseado em percentil: {threshold:.6f} (top 25% features)")
        else:
            # Encontrar o maior declínio na importância
            elbow_idx = max_diff_idx + 1
            
            # Definir threshold como o valor do cotovelo
            threshold = sorted_imp[elbow_idx]
            print(f"  Usando threshold baseado no método do cotovelo: {threshold:.6f}")
        
        # Garantir pelo menos um número mínimo de features
        min_features = min(200, len(feature_names) // 5)  # Aumentado para 200
        if sum(importances >= threshold) < min_features:
            # Ajustar threshold para incluir pelo menos min_features
            threshold = sorted_imp[min_features-1]
            print(f"  Ajustando threshold para incluir pelo menos {min_features} features: {threshold:.6f}")
        
        # Garantir um número máximo de features para evitar overfitting
        max_features = 300  # Aumentado para 300
        if sum(importances >= threshold) > max_features:
            # Ajustar threshold para limitar a max_features
            threshold = sorted_imp[max_features-1]
            print(f"  Limitando seleção a no máximo {max_features} features: {threshold:.6f}")
    
    # Selecionar features acima do threshold
    selected = importance_results[importance_results['importance'] >= threshold]
    selected_features = selected['feature'].tolist()
    
    # Se temos dados de estabilidade, dar preferência a features estáveis
    if stability_results is not None:
        # Adicionar features estáveis que estão próximas do threshold
        stable_features = stability_results[stability_results['is_stable'] == True]['feature'].tolist()
        almost_selected = importance_results[
            (importance_results['importance'] >= threshold * 0.7) &  # Reduzido para incluir mais features
            (importance_results['importance'] < threshold)
        ]
        
        for _, row in almost_selected.iterrows():
            if row['feature'] in stable_features and row['feature'] not in selected_features:
                selected_features.append(row['feature'])
                print(f"  Adicionando feature estável próxima do threshold: {row['feature']}")
    
    print(f"  Selecionadas {len(selected_features)} features com threshold de importância {threshold:.6f}")
    
    return selected_features

def run_feature_importance_analysis_rf(train_df, target_col, output_dir, params_dir=None):
    """
    Executa a análise de importância das features usando apenas Random Forest
    e retorna features selecionadas.
    
    Args:
        train_df: DataFrame de treinamento
        target_col: Nome da coluna target
        output_dir: Diretório para salvar resultados
        params_dir: Diretório para salvar parâmetros (opcional)
        
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
    
    # 6. Análise de multicolinearidade
    high_corr_pairs = fi.analyze_multicollinearity(X)
    
    # 7. Análise específica: codificação de países (se aplicável)
    fi.compare_country_encodings(X, y)
    
    # 8. Análise de importância com Random Forest
    print("\n--- Iniciando análise de importância de features com Random Forest ---")
    importance_results, metrics, rf_model = analyze_rf_importance_extended(
        X, y, numeric_cols, cv=5, output_dir=output_dir
    )
    
    # 9. Análise de estabilidade das features
    stability_results = analyze_feature_stability(X, y, numeric_cols, importance_results)
    
    # 10. Salvar resultados de estabilidade
    stability_path = os.path.join(output_dir, "importance_results", "feature_stability_analysis.csv")
    stability_results.to_csv(stability_path, index=False)
    print(f"\nAnálise de estabilidade das features salva em {stability_path}")
    
    # 11. Análise de robustez entre lançamentos (se aplicável)
    if launch_col:
        # Converter o formato do DataFrame para compatibilidade com a função analyze_launch_robustness
        # O formato esperado pela função tem colunas 'Feature' e 'Mean_Importance'
        compatible_importance = pd.DataFrame({
            'Feature': importance_results['feature'],
            'Mean_Importance': importance_results['importance']
        })
        
        launch_importance, unstable_features, launch_vs_global, consistent_features = fi.analyze_launch_robustness(
            train_df, X, y, numeric_cols, launch_col, rename_dict, compatible_importance
        )
        
        # Salvar análise de robustez entre lançamentos
        if launch_vs_global is not None:
            robustness_path = os.path.join(output_dir, "importance_results", "feature_robustness_analysis.csv")
            launch_vs_global.to_csv(robustness_path, index=False)
            print(f"\nAnálise de robustez entre lançamentos salva em {robustness_path}")
    
    # 12. Selecionar features finais
    selected_features = select_features_rf(X, y, numeric_cols, importance_results, stability_results)
    
    # 13. Verificar features textuais importantes
    text_importance = None
    if text_derived_cols:
        # Filtrar apenas features textuais do resultado de importância
        text_features_in_results = [f for f in importance_results['feature'] if f in text_derived_cols]
        
        if text_features_in_results:
            text_importance = importance_results[importance_results['feature'].isin(text_derived_cols)]
            text_importance = text_importance.sort_values('importance', ascending=False).reset_index(drop=True)
            
            # Salvar análise de features textuais
            text_analysis_path = os.path.join(output_dir, "importance_results", "text_features_importance.csv")
            text_importance.to_csv(text_analysis_path, index=False)
            print(f"Análise de features textuais salva em {text_analysis_path}")
            
            # Imprimir top features textuais
            top_text_features = text_importance.head(10)
            print("\nTop 10 features textuais por importância:")
            for i, (_, row) in enumerate(top_text_features.iterrows(), 1):
                print(f"  {i}. {row['feature']} (importância: {row['importance']:.6f})")
    
    # 14. Ajustar seleção final com base na multicolinearidade
    # Implementação local da função handle_collinearity_in_selection já que ela não existe no módulo
    def handle_collinearity_in_selection(selected_features, high_corr_pairs, importance_results):
        """
        Remove features colineares da seleção final, mantendo as mais importantes.
        
        Args:
            selected_features: Lista de features selecionadas
            high_corr_pairs: Lista de pares de features altamente correlacionadas
            importance_results: DataFrame com importância das features
            
        Returns:
            Tuple com (lista de features final, lista de features removidas)
        """
        print("\n--- Ajustando seleção final com base na multicolinearidade ---")
        
        # Criar um dicionário de importância para fácil acesso
        importance_dict = dict(zip(
            importance_results['feature'], 
            importance_results['importance']
        ))
        
        # Converter high_corr_pairs para o formato correto se necessário
        formatted_pairs = []
        for pair in high_corr_pairs:
            if isinstance(pair, tuple) and len(pair) == 3:
                # Já está no formato correto (feat1, feat2, corr)
                formatted_pairs.append(pair)
            elif isinstance(pair, list) and len(pair) == 2:
                # Formato [feat1, feat2], adicionar correlação fictícia
                formatted_pairs.append((pair[0], pair[1], 0.8))
        
        # Se não há pares formatados, verificar se é um DataFrame
        if not formatted_pairs and isinstance(high_corr_pairs, pd.DataFrame):
            if 'feature1' in high_corr_pairs.columns and 'feature2' in high_corr_pairs.columns:
                for _, row in high_corr_pairs.iterrows():
                    corr = row.get('correlation', 0.8)
                    formatted_pairs.append((row['feature1'], row['feature2'], corr))
        
        if not formatted_pairs:
            print("  Aviso: Formato de pares correlacionados não reconhecido.")
            print("  Usando função alternativa para detectar multicolinearidade nos dados...")
            
            # Usar matriz de correlação diretamente dos dados
            if len(selected_features) > 1:
                X_selected = X[selected_features]
                corr_matrix = X_selected.corr().abs()
                
                # Extrair pares correlacionados (supondo que estamos usando um limiar de 0.8)
                for i in range(len(selected_features)):
                    for j in range(i+1, len(selected_features)):
                        feat1 = selected_features[i]
                        feat2 = selected_features[j]
                        corr = corr_matrix.iloc[i, j]
                        if corr >= 0.8:
                            formatted_pairs.append((feat1, feat2, corr))
        
        # Identificar pares correlacionados onde ambas features estão na seleção
        correlated_selected_pairs = []
        for feat1, feat2, corr in formatted_pairs:
            if feat1 in selected_features and feat2 in selected_features:
                # Ordenar par pela importância (mais importante primeiro)
                if importance_dict.get(feat1, 0) >= importance_dict.get(feat2, 0):
                    correlated_selected_pairs.append((feat1, feat2, corr))
                else:
                    correlated_selected_pairs.append((feat2, feat1, corr))
        
        print(f"  Encontrados {len(correlated_selected_pairs)} pares correlacionados na seleção final")
        
        # Remover features menos importantes em cada par
        removed_features = set()
        for feat1, feat2, corr in correlated_selected_pairs:
            # Se a segunda feature não foi removida por outro par
            if feat2 not in removed_features:
                removed_features.add(feat2)
                print(f"  Removendo {feat2} (correlação {corr:.4f} com {feat1})")
        
        # Criar lista final de features
        final_features = [f for f in selected_features if f not in removed_features]
        
        print(f"  Seleção final: {len(final_features)} features (removidas {len(removed_features)} devido à correlação)")
        
        return final_features, list(removed_features)
    
    final_selected_features, removed_due_to_correlation = handle_collinearity_in_selection(
        selected_features, high_corr_pairs, importance_results
    )
    
    # 15. Documentar seleção de features
    docs_dir = os.path.join(output_dir, "importance_results")
    
    # Criar relatório detalhado
    with open(os.path.join(docs_dir, "feature_selection_report.md"), 'w') as f:
        f.write("# Relatório de Seleção de Features com Random Forest\n\n")
        
        f.write("## Sumário\n\n")
        f.write(f"- Total de features consideradas: {len(numeric_cols)}\n")
        f.write(f"- Features selecionadas: {len(final_selected_features)}\n")
        f.write(f"- Features removidas devido à correlação: {len(removed_due_to_correlation)}\n\n")
        
        f.write("## Métricas do Modelo\n\n")
        f.write(f"- Precision: {metrics['precision']:.4f}\n")
        f.write(f"- Recall: {metrics['recall']:.4f}\n")
        f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"- AUC: {metrics['auc']:.4f}\n\n")
        
        f.write("## Lista de Features Selecionadas\n\n")
        for i, feat in enumerate(final_selected_features, 1):
            orig_name = next((k for k, v in rename_dict.items() if v == feat), feat)
            importance_val = importance_results[importance_results['feature'] == feat]['importance'].values
            importance_str = f" (importância: {importance_val[0]:.6f})" if len(importance_val) > 0 else ""
            f.write(f"{i}. `{orig_name}`{importance_str}\n")
        
        f.write("\n\n## Features Removidas Devido à Correlação\n\n")
        for i, feat in enumerate(removed_due_to_correlation, 1):
            orig_name = next((k for k, v in rename_dict.items() if v == feat), feat)
            f.write(f"{i}. `{orig_name}`\n")
    
    print(f"\nRelatório de seleção de features salvo em {docs_dir}/feature_selection_report.md")
    
    # 16. Salvar lista completa e simplificada de features
    feature_list_path = os.path.join(docs_dir, "selected_features.txt")
    with open(feature_list_path, 'w') as f:
        for feat in final_selected_features:
            f.write(f"{feat}\n")
    print(f"Lista de features selecionadas salva em {feature_list_path}")
    
    # 17. Salvar parâmetros para uso futuro
    selection_params = {
        'selected_features': final_selected_features,
        'rename_dict': rename_dict,
        'target_col': target_col,
        'metrics': metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Salvar o modelo Random Forest para reutilização
    if params_dir:
        # Adicionar timestamp ao nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        params_filename = f"feature_selection_rf_params_{timestamp}.joblib"
        params_path = os.path.join(params_dir, params_filename)
        
        # Salvar a versão atual com timestamp
        joblib.dump(selection_params, params_path)
        print(f"\nParâmetros de seleção de features salvos em {params_path}")
        
        # Opcionalmente, salvar também como "latest" para facilitar referência
        latest_path = os.path.join(params_dir, "feature_selection_rf_params_latest.joblib")
        joblib.dump(selection_params, latest_path)
        print(f"Parâmetros também salvos como versão mais recente em {latest_path}")
        
        # Salvar o modelo treinado
        model_path = os.path.join(params_dir, f"rf_feature_selector_{timestamp}.joblib")
        joblib.dump(rf_model, model_path)
        print(f"Modelo Random Forest salvo em {model_path}")
    
    print(f"\nTotal de {len(final_selected_features)} features selecionadas para o modelo.")
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
        print(f"Adicionando {len(missing_features)} features faltantes ao DataFrame, preenchidas com zeros")
        if len(missing_features) <= 10:
            print(f"Features adicionadas: {list(missing_features)}")
        else:
            print(f"Exemplos de features adicionadas: {list(missing_features)[:10]}...")
        
        # Criar DataFrame com as features faltantes preenchidas com zeros
        missing_df = pd.DataFrame(0, index=df.index, columns=list(missing_features))
        
        # Concatenar com o DataFrame existente
        df_selected = pd.concat([df_selected, missing_df], axis=1)
        
        # Garantir que as colunas estejam na mesma ordem que as features selecionadas
        all_columns = selected_features + [target_col]
        df_selected = df_selected[all_columns]
    
    print(f"DataFrame processado: {df.shape[1]} colunas originais → {df_selected.shape[1]} colunas selecionadas")
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
    
    # 5. Analisar importância de features com Random Forest e selecionar as relevantes
    selected_features, selection_params = run_feature_importance_analysis_rf(
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
    parser = argparse.ArgumentParser(description="Pipeline de análise de importância e seleção de features usando Random Forest.")
    parser.add_argument("--input-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/02_4_processed_text_fn_code7"), 
                        help="Diretório contendo os arquivos de entrada (train.csv, validation.csv, test.csv)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.expanduser("~"), "desktop/smart_ads/data/03_4_feature_selection_rf"), 
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