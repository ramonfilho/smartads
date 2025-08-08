"""
Feature selection module based on importance analysis.

This module provides functions to identify irrelevant features and select
the most relevant ones based on importance metrics and other criteria.
"""

import pandas as pd
import numpy as np
import os

def identify_irrelevant_features(final_importance, high_corr_pairs, threshold_importance=0.1, threshold_cv=1.5):
    """
    Identify potentially irrelevant features based on various criteria.
    
    Args:
        final_importance: DataFrame with combined feature importance
        high_corr_pairs: List of highly correlated feature pairs
        threshold_importance: Threshold for low importance (percentage)
        threshold_cv: Threshold for high coefficient of variation
        
    Returns:
        DataFrame of potentially irrelevant features
    """
    print("\nIdentificando features potencialmente irrelevantes...")

    # Criteria to consider a feature as potentially irrelevant:
    # 1. Low average importance (< threshold_importance % of total importance)
    irrelevant_by_importance = final_importance[final_importance['Mean_Importance'] < threshold_importance]

    # 2. High variability between models (coef. variation > threshold_cv)
    irrelevant_by_variance = final_importance[
        (final_importance['CV'] > threshold_cv) & 
        (final_importance['Mean_Importance'] < final_importance['Mean_Importance'].median())
    ]

    # 3. Features highly correlated with other more important ones
    irrelevant_by_correlation = []
    for pair in high_corr_pairs:
        f1, f2 = pair['feature1'], pair['feature2']
        f1_imp = final_importance[final_importance['Feature'] == f1]['Mean_Importance'].values[0] if f1 in final_importance['Feature'].values else 0
        f2_imp = final_importance[final_importance['Feature'] == f2]['Mean_Importance'].values[0] if f2 in final_importance['Feature'].values else 0
        
        # The less important feature is considered irrelevant
        if f1_imp < f2_imp:
            irrelevant_by_correlation.append({
                'Feature': f1, 
                'Correlation_With': f2, 
                'Correlation': pair['correlation'],
                'Mean_Importance': f1_imp,
                'Better_Feature_Importance': f2_imp
            })
        else:
            irrelevant_by_correlation.append({
                'Feature': f2, 
                'Correlation_With': f1, 
                'Correlation': pair['correlation'],
                'Mean_Importance': f2_imp,
                'Better_Feature_Importance': f1_imp
            })

    # Convert to DataFrame
    irrelevant_by_correlation_df = pd.DataFrame(irrelevant_by_correlation)

    # Combine criteria
    potentially_irrelevant = pd.concat([
        irrelevant_by_importance[['Feature', 'Mean_Importance', 'CV']],
        irrelevant_by_variance[['Feature', 'Mean_Importance', 'CV']]
    ]).drop_duplicates().sort_values(by='Mean_Importance')

    print(f"\nFeatures potencialmente irrelevantes ({len(potentially_irrelevant)}):")
    print(potentially_irrelevant[['Feature', 'Mean_Importance', 'CV']].head(20))

    if len(potentially_irrelevant) > 20:
        print(f"... e mais {len(potentially_irrelevant) - 20} features.")
        
    return potentially_irrelevant, irrelevant_by_importance, irrelevant_by_variance, irrelevant_by_correlation_df

def analyze_text_features(final_importance, text_derived_cols):
    """
    Analyze the importance of text-derived features.
    
    Args:
        final_importance: DataFrame with combined feature importance
        text_derived_cols: List of text-derived feature columns
        
    Returns:
        DataFrame with text features importance
    """
    if not text_derived_cols:
        print("\nNenhuma feature textual encontrada para análise.")
        return None
        
    print("\n--- Análise Específica de Features Textuais ---")
    
    # Extract only text features from importance dataframe
    text_importance = final_importance[final_importance['Feature'].isin(text_derived_cols)].copy()
    
    # Group by feature type
    text_feature_types = {
        'Comprimento Texto': [col for col in text_derived_cols if '_length' in col or '_word_count' in col],
        'Sentimento': [col for col in text_derived_cols if '_sentiment' in col],
        'Motivação': [col for col in text_derived_cols if '_motiv_' in col],
        'Características': [col for col in text_derived_cols if '_has_' in col],
        'TF-IDF': [col for col in text_derived_cols if '_tfidf_' in col]
    }
    
    print("\nImportância das features textuais por categoria:")
    for category, cols in text_feature_types.items():
        if cols:
            category_importance = text_importance[text_importance['Feature'].isin(cols)]
            avg_importance = category_importance['Mean_Importance'].mean() if not category_importance.empty else 0
            print(f"{category}: {len(cols)} features, importância média: {avg_importance:.2f}")
            
            # Top 3 features in this category
            top_features = category_importance.head(3)
            if not top_features.empty:
                print("  Top features nesta categoria:")
                for i, row in top_features.iterrows():
                    print(f"    - {row['Feature']}: {row['Mean_Importance']:.2f}")
    
    # Top 10 text features overall
    print("\nTop 10 features textuais:")
    print(text_importance[['Feature', 'Mean_Importance']].head(10))
    
    # Proportion of importance from text features
    text_importance_sum = text_importance['Mean_Importance'].sum()
    total_importance_sum = final_importance['Mean_Importance'].sum()
    text_proportion = (text_importance_sum / total_importance_sum) * 100 if total_importance_sum > 0 else 0
    
    print(f"\nContribuição total das features textuais: {text_proportion:.2f}% da importância total")
    
    return text_importance

def select_final_features(final_importance, high_corr_pairs, numeric_cols, rename_dict=None, importance_threshold=None):
    """
    Create a list of recommended features based on importance and redundancy analysis.
    
    Args:
        final_importance: DataFrame with combined feature importance
        high_corr_pairs: List of highly correlated feature pairs
        numeric_cols: List of all numeric columns
        rename_dict: Dictionary mapping sanitized column names to original names
        importance_threshold: Threshold for minimum importance (as % of total)
        
    Returns:
        Tuple of (relevant_features, features_to_remove_corr, unrecommended_with_reasons)
    """
    print("\n--- Preparando Recomendações Finais ---")

    # Define importance threshold if not provided
    if importance_threshold is None:
        importance_threshold = final_importance['Mean_Importance'].sum() * 0.001  # 0.1% of total importance

    # Filter relevant and non-redundant features
    relevant_features = final_importance[final_importance['Mean_Importance'] > importance_threshold]['Feature'].tolist()

    # Remove one feature from each highly correlated pair (keep the more important one)
    features_to_remove_corr = []
    if high_corr_pairs:
        for pair in high_corr_pairs:
            f1, f2 = pair['feature1'], pair['feature2']
            if f1 in relevant_features and f2 in relevant_features:
                f1_imp = final_importance[final_importance['Feature'] == f1]['Mean_Importance'].values[0] 
                f2_imp = final_importance[final_importance['Feature'] == f2]['Mean_Importance'].values[0]
                # Remove the less important feature
                if f1_imp < f2_imp and f1 in relevant_features:
                    relevant_features.remove(f1)
                    features_to_remove_corr.append((f1, f2, pair['correlation'], f1_imp, f2_imp))
                elif f2 in relevant_features:
                    relevant_features.remove(f2)
                    features_to_remove_corr.append((f2, f1, pair['correlation'], f2_imp, f1_imp))

    # Create set of unrecommended features with reasons
    unrecommended_features = set(numeric_cols) - set(relevant_features)
    
    # Convert column names back to original if rename_dict is provided
    if rename_dict:
        # Create reverse mapping
        reverse_rename_dict = {v: k for k, v in rename_dict.items()}
        original_relevant_features = [reverse_rename_dict.get(feature, feature) for feature in relevant_features]
        unrecommended_features_original = [reverse_rename_dict.get(feature, feature) for feature in unrecommended_features]
    else:
        original_relevant_features = relevant_features
        unrecommended_features_original = list(unrecommended_features)
    
    print(f"\nTotal de features analisadas: {len(numeric_cols)}")
    print(f"Features recomendadas após filtragem: {len(original_relevant_features)}")
    
    return original_relevant_features, features_to_remove_corr, unrecommended_features_original

def document_feature_selections(original_relevant_features, unrecommended_features, 
                               final_importance, high_corr_pairs, rename_dict, 
                               output_dir='eda_results/feature_importance_results'):
    """
    Create documentation files for recommended and unrecommended features.
    
    Args:
        original_relevant_features: List of recommended features (original names)
        unrecommended_features: List of unrecommended features (original names)
        final_importance: DataFrame with combined feature importance
        high_corr_pairs: List of highly correlated feature pairs
        rename_dict: Dictionary mapping sanitized column names to original names
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Reverse rename dictionary for lookups
    if rename_dict:
        reverse_rename_dict = {v: k for k, v in rename_dict.items()}
    else:
        reverse_rename_dict = {}
    
    # Save list of recommended features
    with open(os.path.join(output_dir, 'recommended_features.txt'), 'w') as f:
        for feature in original_relevant_features:
            f.write(f"{feature}\n")
    
    print(f"\nLista de {len(original_relevant_features)} features recomendadas salva em {os.path.join(output_dir, 'recommended_features.txt')}")
    
    # Prepare justifications for each unrecommended feature
    unrecommended_with_reasons = []
    
    for feature in unrecommended_features:
        # Get sanitized name if it exists
        sanitized_feature = None
        for orig, sanitized in rename_dict.items() if rename_dict else []:
            if orig == feature:
                sanitized_feature = sanitized
                break
        
        if not sanitized_feature:
            sanitized_feature = feature
        
        reasons = []
        
        # Check if has low importance
        if sanitized_feature in final_importance['Feature'].values:
            imp = final_importance[final_importance['Feature'] == sanitized_feature]['Mean_Importance'].values[0]
            if imp < final_importance['Mean_Importance'].sum() * 0.001:  # Less than 0.1% of total
                reasons.append(f"Baixa importância preditiva ({imp:.4f})")
        
        # Check if has high variability between models
        if sanitized_feature in final_importance['Feature'].values:
            cv = final_importance[final_importance['Feature'] == sanitized_feature]['CV'].values[0]
            if cv > 1.5:
                reasons.append(f"Alta variabilidade entre modelos (CV={cv:.2f})")
        
        # Check if redundant with another feature
        redundant_with = None
        for pair in high_corr_pairs:
            f1, f2 = pair['feature1'], pair['feature2']
            f1_orig = reverse_rename_dict.get(f1, f1) if reverse_rename_dict else f1
            f2_orig = reverse_rename_dict.get(f2, f2) if reverse_rename_dict else f2
            
            if f1_orig == feature or f2_orig == feature:
                other_f = f2_orig if f1_orig == feature else f1_orig
                f1_imp = final_importance[final_importance['Feature'] == f1]['Mean_Importance'].values[0] if f1 in final_importance['Feature'].values else 0
                f2_imp = final_importance[final_importance['Feature'] == f2]['Mean_Importance'].values[0] if f2 in final_importance['Feature'].values else 0
                
                if (f1_orig == feature and f1_imp < f2_imp) or (f2_orig == feature and f2_imp < f1_imp):
                    other_imp = f2_imp if f1_orig == feature else f1_imp
                    this_imp = f1_imp if f1_orig == feature else f2_imp
                    reasons.append(f"Altamente correlacionada (r={pair['correlation']:.2f}) com {other_f} que tem maior importância ({other_imp:.4f} vs {this_imp:.4f})")
                    redundant_with = other_f
                    break
        
        # If no specific reason found
        if not reasons:
            reasons.append("Baixa contribuição geral para o modelo")
        
        # Get importance
        importance = 0
        if sanitized_feature in final_importance['Feature'].values:
            importance = final_importance[final_importance['Feature'] == sanitized_feature]['Mean_Importance'].values[0]
        
        unrecommended_with_reasons.append({
            'Feature': feature,
            'Reasons': "; ".join(reasons),
            'Redundant_With': redundant_with,
            'Importance': importance
        })
    
    # Convert to DataFrame and save
    unrecommended_df = pd.DataFrame(unrecommended_with_reasons)
    unrecommended_df = unrecommended_df.sort_values('Importance', ascending=False)
    unrecommended_df.to_csv(os.path.join(output_dir, 'unrecommended_features.csv'), index=False)
    
    # Create text file with detailed explanations
    with open(os.path.join(output_dir, 'unrecommended_features_explanation.txt'), 'w') as f:
        f.write("# Features Não Recomendadas e Justificativas\n\n")
        f.write(f"Total de features analisadas: {len(original_relevant_features) + len(unrecommended_features)}\n")
        f.write(f"Features recomendadas: {len(original_relevant_features)}\n")
        f.write(f"Features não recomendadas: {len(unrecommended_features)}\n\n")
        
        f.write("## Razões para remoção:\n\n")
        f.write("1. **Baixa importância preditiva**: Features com importância menor que 0.1% da importância total.\n")
        f.write("2. **Alta variabilidade entre modelos**: Features cujo coeficiente de variação entre diferentes modelos é maior que 1.5.\n")
        f.write("3. **Redundância**: Features altamente correlacionadas (r > 0.8) com outras de maior importância.\n\n")
        
        f.write("## Lista de features não recomendadas:\n\n")
        
        for i, row in unrecommended_df.iterrows():
            f.write(f"### {i+1}. {row['Feature']}\n")
            f.write(f"   - **Razões**: {row['Reasons']}\n")
            if pd.notna(row['Redundant_With']):
                f.write(f"   - **Redundante com**: {row['Redundant_With']}\n")
            f.write(f"   - **Importância**: {row['Importance']:.6f}\n\n")
    
    print(f"Documentação detalhada de features não recomendadas salva em {os.path.join(output_dir, 'unrecommended_features_explanation.txt')}")

def create_selected_dataset(original_relevant_features, target_col, input_path, output_path):
    """
    Create and save a dataset with only the selected features.
    
    Args:
        original_relevant_features: List of recommended features (original names)
        target_col: Name of the target column
        input_path: Path to the original dataset
        output_path: Path to save the new dataset
    """
    print("\n--- Gerando Dataset com Features Selecionadas ---")
    
    # Load original dataset
    try:
        original_df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Erro ao carregar dataset original: {e}")
        return
    
    # Check if the recommended features exist in the original dataset
    available_features = [col for col in original_relevant_features if col in original_df.columns]
    missing_features = set(original_relevant_features) - set(available_features)
    
    if missing_features:
        print(f"Aviso: {len(missing_features)} features recomendadas não foram encontradas no dataset original.")
        print(f"Exemplos: {list(missing_features)[:5]}...")
    
    # Select relevant features + target
    selected_columns = available_features + [target_col]
    print(f"Selecionando {len(available_features)} features + target para o novo dataset")
    
    # Create new dataset
    selected_df = original_df[selected_columns]
    
    # Save dataset with selected features
    try:
        selected_df.to_csv(output_path, index=False)
        print(f"Dataset com features selecionadas salvo em {output_path}")
    except Exception as e:
        print(f"Erro ao salvar dataset com features selecionadas: {e}")

def summarize_feature_categories(numeric_cols, final_importance, text_derived_cols):
    """
    Summarize feature importance by category and show top features.
    
    Args:
        numeric_cols: List of all numeric column names
        final_importance: DataFrame with combined feature importance
        text_derived_cols: List of text-derived features
    """
    print("\n=== RESUMO DAS ANÁLISES ===")
    print(f"Total de features analisadas: {len(numeric_cols)}")
    print(f"Features textuais processadas: {len(text_derived_cols) if text_derived_cols else 0}")
    
    # Show top 10 most important features
    print("\nTop 10 features mais importantes para previsão de conversão:")
    for i, row in final_importance.head(10).iterrows():
        print(f"{i+1}. {row['Feature']}: {row['Mean_Importance']:.2f}")
    
    # Categorize features by type
    feature_categories = {
        'Features Textuais': text_derived_cols if text_derived_cols else [],
        'UTM/Campaign': [col for col in numeric_cols if any(term in col.lower() for term in ['utm', 'campaign', 'camp'])],
        'Dados Geográficos': [col for col in numeric_cols if any(term in col.lower() for term in ['country', 'pais'])],
        'Tempo/Data': [col for col in numeric_cols if any(term in col.lower() for term in ['time', 'hour', 'day', 'month', 'year'])],
        'Demografia': [col for col in numeric_cols if any(term in col.lower() for term in ['age', 'gender', 'edad'])],
        'Profissão': [col for col in numeric_cols if any(term in col.lower() for term in ['profes', 'profession', 'work'])]
    }
    
    # Calculate importance by category
    categories_importance = {}
    for category, cols in feature_categories.items():
        category_features = [col for col in cols if col in final_importance['Feature'].values]
        if category_features:
            category_importance = final_importance[final_importance['Feature'].isin(category_features)]['Mean_Importance'].sum()
            categories_importance[category] = category_importance
    
    # Sort categories by importance
    sorted_categories = sorted(categories_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nImportância por categoria de features:")
    for category, importance in sorted_categories:
        category_pct = (importance / final_importance['Mean_Importance'].sum()) * 100
        print(f"{category}: {importance:.2f} ({category_pct:.1f}%)")