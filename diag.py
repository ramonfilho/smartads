#!/usr/bin/env python
"""
Script de diagnóstico para identificar inconsistências de nomes entre scripts
"""
import pandas as pd
import joblib
import os
import sys
import json

PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, PROJECT_ROOT)

def diagnose_feature_inconsistencies():
    """Diagnostica problemas de nomes de features entre scripts"""
    
    print("=== DIAGNÓSTICO DE INCONSISTÊNCIAS DE FEATURES ===\n")
    
    # 1. Carregar dataset de treino final
    train_path = os.path.join(PROJECT_ROOT, "data/new/04_feature_selection/train.csv")
    train_df = pd.read_csv(train_path)
    print(f"1. Dataset de treino carregado: {train_df.shape}")
    
    # 2. Carregar features recomendadas
    features_path = os.path.join(PROJECT_ROOT, "reports/feature_importance_results/recommended_features.txt")
    with open(features_path, 'r') as f:
        recommended_features = [line.strip() for line in f.readlines()]
    print(f"2. Features recomendadas: {len(recommended_features)}")
    
    # 3. Analisar parâmetros salvos
    params_paths = {
        'preprocessing': os.path.join(PROJECT_ROOT, "src/preprocessing/params/new/02_preprocessing_params/all_preprocessing_params.joblib"),
        'professional': os.path.join(PROJECT_ROOT, "src/preprocessing/params/new/03_params/03_professional_features_params.joblib"),
        'tfidf': os.path.join(PROJECT_ROOT, "src/preprocessing/params/new/03_params/03_tfidf_vectorizers.joblib"),
        'lda': os.path.join(PROJECT_ROOT, "src/preprocessing/params/new/03_params/03_lda_models.joblib")
    }
    
    params_info = {}
    for name, path in params_paths.items():
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                params_info[name] = {
                    'exists': True,
                    'type': type(data).__name__,
                    'size': len(data) if hasattr(data, '__len__') else 'N/A'
                }
            except Exception as e:
                params_info[name] = {'exists': True, 'error': str(e)}
        else:
            params_info[name] = {'exists': False}
    
    print("\n3. Status dos arquivos de parâmetros:")
    for name, info in params_info.items():
        print(f"   {name}: {info}")
    
    # 4. Analisar colunas de texto e suas transformações
    text_patterns = {
        'original': ['Cuando hables inglés con fluidez', '¿Qué esperas aprender', 'Déjame un mensaje'],
        'sanitized': ['cuando_hables', 'que_esperas', 'dejame_un']
    }
    
    print("\n4. Análise de colunas de texto:")
    text_features = {}
    
    for col in train_df.columns:
        # Identificar features derivadas de texto
        if any(pattern in col for pattern in ['_tfidf_', '_topic_', '_motivation', '_sentiment', '_commitment']):
            # Tentar identificar a coluna de origem
            for pattern in text_patterns['original'] + text_patterns['sanitized']:
                if pattern.lower() in col.lower():
                    origin = pattern
                    if origin not in text_features:
                        text_features[origin] = []
                    text_features[origin].append(col)
                    break
    
    for origin, features in text_features.items():
        print(f"\n   Origem: {origin[:50]}...")
        print(f"   Features derivadas: {len(features)}")
        print(f"   Exemplos: {features[:3]}")
    
    # 5. Verificar features ausentes
    train_features = set(train_df.columns)
    recommended_set = set(recommended_features)
    
    missing_in_train = recommended_set - train_features
    extra_in_train = train_features - recommended_set - {'target'}
    
    print(f"\n5. Comparação com features recomendadas:")
    print(f"   Features faltando no treino: {len(missing_in_train)}")
    print(f"   Features extras no treino: {len(extra_in_train)}")
    
    # Categorizar features problemáticas
    problematic_features = {
        'tfidf': [],
        'topic': [],
        'professional': [],
        'advanced': [],
        'other': []
    }
    
    for feat in list(missing_in_train) + list(extra_in_train):
        if '_tfidf_' in feat:
            problematic_features['tfidf'].append(feat)
        elif '_topic_' in feat:
            problematic_features['topic'].append(feat)
        elif any(x in feat for x in ['motivation', 'commitment', 'career', 'aspiration']):
            problematic_features['professional'].append(feat)
        elif any(x in feat for x in ['embedding', 'refined', 'interaction']):
            problematic_features['advanced'].append(feat)
        else:
            problematic_features['other'].append(feat)
    
    print("\n6. Features problemáticas por categoria:")
    for category, features in problematic_features.items():
        if features:
            print(f"\n   {category.upper()} ({len(features)} features):")
            for feat in features[:5]:
                print(f"      - {feat}")
            if len(features) > 5:
                print(f"      ... e mais {len(features) - 5}")
    
    # Salvar relatório
    report = {
        'train_shape': train_df.shape,
        'recommended_features_count': len(recommended_features),
        'params_info': params_info,
        'text_features_mapping': {k: len(v) for k, v in text_features.items()},
        'missing_features': list(missing_in_train),
        'extra_features': list(extra_in_train),
        'problematic_by_category': {k: len(v) for k, v in problematic_features.items()}
    }
    
    report_path = os.path.join(PROJECT_ROOT, "reports/feature_name_diagnostic.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Relatório completo salvo em: {report_path}")
    
    return report

if __name__ == "__main__":
    diagnose_feature_inconsistencies()