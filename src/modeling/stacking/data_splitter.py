"""
Módulo para dividir features do Smart Ads em grupos especializados para stacking.

Este módulo contém funções específicas para categorizar as features do projeto
Smart Ads, com regras adaptadas ao conjunto de dados existente.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict

# Definições de categorias específicas para o Smart Ads
FEATURE_CATEGORIES = {
    'demographic': [
        # Features demográficas baseadas em análise do CSV
        'country', 'age', 'gender', 'job_title', 'current_salary', 'desired_salary',
        'education_level', 'experience_years', 'language_proficiency', 'industry',
        'marital_status', 'children_count', 'has_loan', 'income_range',
        'work_status', 'device_type', 'browser', 'location'
    ],
    'temporal': [
        # Features temporais baseadas em análise do CSV
        'day_of_week', 'hour_of_day', 'month', 'time_spent', 'session_duration',
        'days_since_last_visit', 'time_on_page', 'lead_time', 'registration_date',
        'last_activity', 'time_to_convert', 'response_time', 'first_visit_date'
    ],
    'text': [
        # Features textuais baseadas em análise do CSV
        'motivation', 'goals', 'expectations', 'challenges', 'interests',
        'career_objectives', 'comments', 'feedback', 'search_terms',
        'message', 'notes', 'job_description', 'bio', 'reasons',
        'tfidf_', 'nlp_', 'text_', 'embed_', 'topic_', 'sentiment_'
    ],
    'campaign': [
        # Features de campanha baseadas em análise do CSV
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term',
        'ad_id', 'campaign_id', 'landing_page', 'referrer', 'channel',
        'source', 'medium', 'campaign', 'cpc', 'cpm', 'ctr', 'conversion_rate'
    ],
    'behavioral': [
        # Features comportamentais baseadas em análise do CSV
        'clicks', 'views', 'engagement', 'pages_visited', 'actions_taken',
        'downloads', 'form_completions', 'scroll_depth', 'bounce_rate',
        'exit_intent', 'email_opens', 'email_clicks', 'video_views',
        'feature_usage', 'activity_score'
    ]
}

# Mapeamento específico para colunas TF-IDF que aparecem no CSV
TFIDF_COLUMNS = [
    # Lista de colunas TF-IDF baseadas na análise do CSV
    'tfidf_money', 'tfidf_income', 'tfidf_salary', 'tfidf_job', 'tfidf_work',
    'tfidf_career', 'tfidf_professional', 'tfidf_opportunity', 'tfidf_business',
    'tfidf_financial', 'tfidf_family', 'tfidf_children', 'tfidf_better', 'tfidf_life',
    'tfidf_time', 'tfidf_growth', 'tfidf_learn', 'tfidf_knowledge'
]

def categorize_smart_ads_features(df, target_col="target"):
    """
    Categoriza features do Smart Ads em grupos específicos.
    
    Args:
        df: DataFrame com todas as features
        target_col: Nome da coluna target
        
    Returns:
        Dicionário com listas de features por categoria
    """
    # Inicializar grupos de features
    feature_groups = {
        'demographic': [],
        'temporal': [],
        'text': [],
        'campaign': [],
        'behavioral': [],
        'other': []
    }
    
    # Criar conjuntos de padrões para cada categoria para busca rápida
    category_patterns = {}
    for category, patterns in FEATURE_CATEGORIES.items():
        category_patterns[category] = [re.compile(pattern.lower()) for pattern in patterns]
    
    # Processar cada coluna
    for col in df.columns:
        if col == target_col:
            continue
        
        col_lower = col.lower()
        
        # Atribuir diretamente as colunas TF-IDF
        if any(tfidf_col.lower() in col_lower for tfidf_col in TFIDF_COLUMNS):
            feature_groups['text'].append(col)
            continue
        
        # Tentar corresponder a cada categoria
        matched = False
        for category, patterns in category_patterns.items():
            if any(pattern.search(col_lower) for pattern in patterns):
                feature_groups[category].append(col)
                matched = True
                break
            
            # Verificar correspondência parcial para padrões específicos
            if category == 'text' and any(kw in col_lower for kw in ['motivation', 'goal', 'reason', 'comment', 'text']):
                feature_groups['text'].append(col)
                matched = True
                break
            
            if category == 'campaign' and any(kw in col_lower for kw in ['utm', 'ad_', 'campaign', 'source']):
                feature_groups['campaign'].append(col)
                matched = True
                break
        
        # Se não corresponder a nenhuma categoria, verificar tipo de dados
        if not matched:
            # Verificar tipo de dados
            if pd.api.types.is_numeric_dtype(df[col]):
                # Se for numérico, verificar se é binário
                unique_values = df[col].dropna().unique()
                if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, 0.0, 1.0}):
                    feature_groups['demographic'].append(col)
                else:
                    # Para outros numéricos, verificar nomes comuns
                    if any(kw in col_lower for kw in ['count', 'total', 'sum', 'avg', 'number', 'score']):
                        feature_groups['behavioral'].append(col)
                    else:
                        feature_groups['other'].append(col)
            elif pd.api.types.is_string_dtype(df[col]):
                # Verificar se tem mais de 3 palavras em média = texto
                try:
                    avg_words = df[col].astype(str).str.split().str.len().mean()
                    if avg_words >= 3:
                        feature_groups['text'].append(col)
                    else:
                        feature_groups['demographic'].append(col)
                except:
                    feature_groups['demographic'].append(col)
            else:
                feature_groups['other'].append(col)
    
    return feature_groups

def prepare_data_for_specialists(df, feature_groups, target_col="target"):
    """
    Prepara os dados para os modelos especialistas.
    
    Args:
        df: DataFrame com todas as features
        feature_groups: Dicionário com grupos de features
        target_col: Nome da coluna target
        
    Returns:
        Tupla (X, y) onde X é um dicionário com DataFrames para cada tipo de feature
    """
    # Extrair target
    y = df[target_col] if target_col in df.columns else None
    
    # Criar dicionário de features
    X = {}
    
    for group_name, features in feature_groups.items():
        if features:  # Verificar se a lista não está vazia
            # Garantir que todas as features existem no DataFrame
            valid_features = [f for f in features if f in df.columns]
            if valid_features:
                X[group_name] = df[valid_features].copy()
    
    return X, y

def print_feature_group_stats(feature_groups, df=None):
    """
    Imprime estatísticas sobre os grupos de features.
    
    Args:
        feature_groups: Dicionário com grupos de features
        df: DataFrame com dados (opcional, para estatísticas adicionais)
    """
    print("\n=== Estatísticas de Grupos de Features ===")
    
    total_features = sum(len(features) for features in feature_groups.values())
    
    print(f"Total de features: {total_features}")
    print("\nDistribuição por grupo:")
    
    for group, features in sorted(feature_groups.items()):
        count = len(features)
        percent = (count / total_features * 100) if total_features > 0 else 0
        print(f"  {group}: {count} features ({percent:.1f}%)")
        
        # Se df fornecido, mostrar exemplos de cada grupo
        if df is not None and count > 0:
            example_count = min(3, count)
            examples = features[:example_count]
            print(f"    Exemplos: {', '.join(examples)}")
    
    print("\n")

def validate_feature_groups(feature_groups, df):
    """
    Valida se todos os grupos de features existem no DataFrame.
    
    Args:
        feature_groups: Dicionário com grupos de features
        df: DataFrame com dados
        
    Returns:
        Dicionário com grupos validados (removendo features inexistentes)
    """
    validated = {}
    df_columns = set(df.columns)
    
    for group, features in feature_groups.items():
        valid_features = [f for f in features if f in df_columns]
        validated[group] = valid_features
    
    return validated

def adjust_feature_groups_manually(feature_groups, adjustments):
    """
    Ajusta manualmente grupos de features com base em um dicionário de ajustes.
    
    Args:
        feature_groups: Dicionário original com grupos de features
        adjustments: Dicionário com ajustes no formato:
                     {'move': [('feature', 'from_group', 'to_group'), ...],
                      'add': [('feature', 'group'), ...],
                      'remove': [('feature', 'group'), ...]}
        
    Returns:
        Dicionário ajustado com grupos de features
    """
    # Cria uma cópia para não modificar o original
    adjusted = {group: list(features) for group, features in feature_groups.items()}
    
    # Processar movimentações
    if 'move' in adjustments:
        for feature, from_group, to_group in adjustments['move']:
            if feature in adjusted.get(from_group, []):
                adjusted[from_group].remove(feature)
                if to_group not in adjusted:
                    adjusted[to_group] = []
                adjusted[to_group].append(feature)
    
    # Processar adições
    if 'add' in adjustments:
        for feature, group in adjustments['add']:
            if group not in adjusted:
                adjusted[group] = []
            if feature not in adjusted[group]:
                adjusted[group].append(feature)
    
    # Processar remoções
    if 'remove' in adjustments:
        for feature, group in adjustments['remove']:
            if feature in adjusted.get(group, []):
                adjusted[group].remove(feature)
    
    return adjusted

def save_feature_groups(feature_groups, filepath):
    """
    Salva grupos de features em um arquivo CSV.
    
    Args:
        feature_groups: Dicionário com grupos de features
        filepath: Caminho para salvar o arquivo
    """
    import csv
    import os
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['group', 'feature'])
        
        for group, features in feature_groups.items():
            for feature in features:
                writer.writerow([group, feature])

def load_feature_groups(filepath):
    """
    Carrega grupos de features de um arquivo CSV.
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Dicionário com grupos de features
    """
    import csv
    from collections import defaultdict
    
    feature_groups = defaultdict(list)
    
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Pular cabeçalho
        
        for row in reader:
            group, feature = row
            feature_groups[group].append(feature)
    
    return dict(feature_groups)

def get_feature_categories(df, target_col="target"):
    """
    Função wrapper para manter a API consistente com o código original.
    
    Args:
        df: DataFrame com todas as features
        target_col: Nome da coluna target
        
    Returns:
        Dicionário com listas de features por categoria
    """
    return categorize_smart_ads_features(df, target_col)