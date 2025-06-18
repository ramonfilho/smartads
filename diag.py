#!/usr/bin/env python
"""
Script para comparar features entre modelo original e novo pipeline OOP.
Identifica diferenças críticas que podem explicar a diferença de performance.
"""

import json
import pandas as pd
import joblib
from pathlib import Path
import sys

# Adicionar ao path
sys.path.insert(0, '/Users/ramonmoreira/desktop/smart_ads')

def compare_model_features():
    """Compara features entre modelo original e novo."""
    
    print("=== ANÁLISE COMPARATIVA DE FEATURES ===\n")
    
    # 1. Carregar features do modelo original
    print("1. Carregando features do modelo original...")
    with open('/Users/ramonmoreira/desktop/smart_ads/models/artifacts/direct_ranking/model_config.json', 'r') as f:
        original_config = json.load(f)
    
    features_original = set(original_config['feature_columns'])
    print(f"   Features originais: {len(features_original)}")
    
    # 2. Carregar features do novo modelo
    print("\n2. Carregando features do novo modelo...")
    try:
        # Tentar carregar do teste
        importance_df = pd.read_csv('/tmp/smart_ads_model_test/feature_importance.csv')
        features_novo = set(importance_df['feature'].tolist())
        print(f"   Features novo modelo: {len(features_novo)}")
    except:
        print("   ❌ Arquivo de feature importance não encontrado")
        return
    
    # 3. Análise de diferenças
    print("\n3. ANÁLISE DE DIFERENÇAS:")
    print("-" * 60)
    
    # Features únicas em cada modelo
    only_original = features_original - features_novo
    only_novo = features_novo - features_original
    common = features_original & features_novo
    
    print(f"Features em comum: {len(common)} ({len(common)/len(features_original)*100:.1f}%)")
    print(f"Features apenas no original: {len(only_original)}")
    print(f"Features apenas no novo: {len(only_novo)}")
    
    # 4. Análise qualitativa das features perdidas
    print("\n4. ANÁLISE QUALITATIVA DAS FEATURES PERDIDAS:")
    print("-" * 60)
    
    # Categorizar features perdidas
    categories = {
        'salary': [],
        'country': [],
        'age': [],
        'utm': [],
        'text_tfidf': [],
        'topic': [],
        'professional': [],
        'temporal': [],
        'other': []
    }
    
    for feat in only_original:
        feat_lower = feat.lower()
        if 'salary' in feat_lower or 'sueldo' in feat_lower:
            categories['salary'].append(feat)
        elif 'country' in feat_lower or 'pais' in feat_lower:
            categories['country'].append(feat)
        elif 'age' in feat_lower or 'edad' in feat_lower:
            categories['age'].append(feat)
        elif 'utm' in feat_lower:
            categories['utm'].append(feat)
        elif '_tfidf_' in feat_lower:
            categories['text_tfidf'].append(feat)
        elif '_topic_' in feat_lower:
            categories['topic'].append(feat)
        elif any(term in feat_lower for term in ['professional', 'career', 'motiv']):
            categories['professional'].append(feat)
        elif any(term in feat_lower for term in ['hour', 'day', 'month', 'morning']):
            categories['temporal'].append(feat)
        else:
            categories['other'].append(feat)
    
    # Mostrar resumo por categoria
    for cat, features in categories.items():
        if features:
            print(f"\n{cat.upper()} ({len(features)} features perdidas):")
            # Mostrar até 5 exemplos
            for i, feat in enumerate(features[:5]):
                print(f"   - {feat}")
            if len(features) > 5:
                print(f"   ... e mais {len(features)-5} features")
    
    # 5. Top features críticas perdidas
    print("\n5. TOP 20 FEATURES CRÍTICAS DO MODELO ORIGINAL:")
    print("-" * 60)
    
    # As primeiras features no config geralmente são as mais importantes
    top_20_original = original_config['feature_columns'][:20]
    
    missing_critical = []
    for i, feat in enumerate(top_20_original):
        status = "✅" if feat in features_novo else "❌"
        print(f"{i+1:2d}. {status} {feat}")
        if feat not in features_novo:
            missing_critical.append(feat)
    
    # 6. Análise de padrões de nomenclatura
    print("\n6. ANÁLISE DE PADRÕES DE NOMENCLATURA:")
    print("-" * 60)
    
    # Verificar caracteres especiais nas features originais
    special_chars = set()
    for feat in features_original:
        for char in feat:
            if not (char.isalnum() or char == '_'):
                special_chars.add(char)
    
    if special_chars:
        print(f"Caracteres especiais encontrados nas features originais: {special_chars}")
    
    # Comparar um exemplo de nomenclatura
    print("\nExemplo de diferenças de nomenclatura:")
    
    # Procurar features similares mas com nomes diferentes
    for orig_feat in list(only_original)[:5]:
        # Tentar encontrar correspondente no novo
        base_name = orig_feat.lower().replace('é', 'e').replace('á', 'a').replace('í', 'i').replace('ñ', 'n')
        possible_matches = [f for f in features_novo if base_name[:20] in f.lower()]
        if possible_matches:
            print(f"\nOriginal: {orig_feat}")
            print(f"Possível match: {possible_matches[0]}")
    
    # 7. Salvar análise completa
    analysis_path = '/tmp/feature_comparison_analysis.json'
    analysis = {
        'summary': {
            'total_features_original': len(features_original),
            'total_features_novo': len(features_novo),
            'common_features': len(common),
            'only_original': len(only_original),
            'only_novo': len(only_novo),
            'overlap_percentage': len(common)/len(features_original)*100
        },
        'missing_critical_features': missing_critical,
        'features_only_original': list(only_original),
        'features_only_novo': list(only_novo),
        'category_analysis': {k: len(v) for k, v in categories.items() if v}
    }
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✅ Análise completa salva em: {analysis_path}")
    
    # 8. Recomendações
    print("\n8. DIAGNÓSTICO E RECOMENDAÇÕES:")
    print("=" * 60)
    
    overlap_pct = len(common)/len(features_original)*100
    
    if overlap_pct < 50:
        print("🚨 PROBLEMA CRÍTICO: Menos de 50% de overlap nas features!")
        print("   Isso explica completamente a diferença de performance.")
        print("\n   CAUSA PROVÁVEL:")
        print("   - Mudança na padronização de nomes (acentos, caracteres especiais)")
        print("   - Diferente processo de feature engineering")
        print("   - Perda de features críticas durante o processamento")
    elif overlap_pct < 80:
        print("⚠️  PROBLEMA SIGNIFICATIVO: Overlap de features abaixo de 80%")
        print("   Isso pode explicar parte da diferença de performance.")
    else:
        print("✅ Overlap de features aceitável, procurar outras causas")
    
    print("\n   PRÓXIMOS PASSOS:")
    print("   1. Revisar o processo de padronização de nomes")
    print("   2. Garantir que TODAS as etapas de feature engineering foram aplicadas")
    print("   3. Verificar se o FeatureSelector está usando os mesmos critérios")
    
    return analysis

if __name__ == "__main__":
    analysis = compare_model_features()
    
    print("\n" + "="*60)
    print("CONCLUSÃO DA ANÁLISE")
    print("="*60)