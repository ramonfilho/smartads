#!/usr/bin/env python
"""
Script para comparar diretamente o pipeline antigo com o novo.
Identifica exatamente quais features foram perdidas e onde.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict

PROJECT_ROOT = "/Users/ramonmoreira/Desktop/smart_ads"
OLD_DATA_PATH = "/Users/ramonmoreira/Desktop/smart_ads/data/V4/04_feature_engineering_2"
NEW_DATA_PATH = "/Users/ramonmoreira/Desktop/smart_ads/data/V5/04_feature_engineering_2"

def deep_feature_analysis():
    """Análise profunda das features perdidas."""
    
    print("="*80)
    print("ANÁLISE PROFUNDA: PIPELINE ANTIGO vs NOVO")
    print("="*80)
    
    # Paths dos datasets finais - tentar múltiplas opções
    v4_options = [
        os.path.join(OLD_DATA_PATH, "train.csv"),
        os.path.join(OLD_DATA_PATH.replace("04_feature_engineering_2", "04_feature_engineering_2_2"), "train.csv"),
        os.path.join(OLD_DATA_PATH.replace("04_feature_engineering_2", "05_feature_selection"), "train.csv")
    ]
    
    v5_options = [
        os.path.join(NEW_DATA_PATH, "train.csv"),
        os.path.join(NEW_DATA_PATH.replace("04_feature_engineering_2", "04_feature_engineering_2_2"), "train.csv")
    ]
    
    # Encontrar o arquivo V4
    old_final = None
    for path in v4_options:
        if os.path.exists(path):
            old_final = path
            print(f"✅ Encontrado arquivo V4 em: {path}")
            break
    
    # Encontrar o arquivo V5
    new_final = None
    for path in v5_options:
        if os.path.exists(path):
            new_final = path
            print(f"✅ Encontrado arquivo V5 em: {path}")
            break
    
    if not old_final:
        print(f"❌ Nenhum arquivo V4 encontrado. Tentei: {v4_options}")
        return
    
    if not new_final:
        print(f"❌ Nenhum arquivo V5 encontrado. Tentei: {v5_options}")
        return
    
    try:
        # Tentar carregar o dataset V4 (antigo)
        print("Carregando dataset V4 (antigo)...")
        old_df = pd.read_csv(old_final)
        print(f"✅ Dataset V4 carregado: {old_df.shape}")
    except Exception as e:
        print(f"❌ Erro ao carregar dataset V4: {e}")
        return analyze_without_old_data(new_final)
    
    # Carregar dataset V5 (novo)
    print("Carregando dataset V5 (novo)...")
    new_df = pd.read_csv(new_final)
    print(f"✅ Dataset V5 carregado: {new_df.shape}")
    
    # Análise detalhada
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    
    lost_features = old_cols - new_cols
    added_features = new_cols - old_cols
    kept_features = old_cols & new_cols
    
    print(f"\n📊 RESUMO:")
    print(f"   Features no pipeline V4 (antigo): {len(old_cols)}")
    print(f"   Features no pipeline V5 (novo): {len(new_cols)}")
    print(f"   Features perdidas: {len(lost_features)}")
    print(f"   Features adicionadas: {len(added_features)}")
    print(f"   Features mantidas: {len(kept_features)}")
    
    # Categorizar features perdidas
    lost_categories = defaultdict(list)
    
    for feature in lost_features:
        # TF-IDF features
        if 'tfidf_' in feature.lower():
            lost_categories['TF-IDF'].append(feature)
        # BOW features
        elif 'bow_' in feature.lower():
            lost_categories['Bag of Words'].append(feature)
        # Sentiment features
        elif 'sentiment' in feature.lower():
            lost_categories['Sentiment Analysis'].append(feature)
        # Length/count features
        elif any(x in feature.lower() for x in ['length', 'count', 'n_']):
            lost_categories['Text Statistics'].append(feature)
        # Categorical encodings
        elif any(x in feature for x in ['_encoded', '_dummy', '_onehot']):
            lost_categories['Categorical Encodings'].append(feature)
        # Interaction features
        elif '_x_' in feature or '_interaction' in feature:
            lost_categories['Feature Interactions'].append(feature)
        # Numerical transformations
        elif any(x in feature for x in ['_log', '_sqrt', '_squared', '_binned']):
            lost_categories['Numerical Transformations'].append(feature)
        # Campaign/launch specific
        elif 'lançamento' in feature or 'campaign' in feature.lower():
            lost_categories['Campaign Features'].append(feature)
        # Time-based features
        elif any(x in feature.lower() for x in ['day', 'month', 'week', 'time']):
            lost_categories['Time Features'].append(feature)
        # Other
        else:
            lost_categories['Other'].append(feature)
    
    # Imprimir análise por categoria
    print("\n🔍 FEATURES PERDIDAS POR CATEGORIA:")
    print("-"*80)
    
    for category, features in sorted(lost_categories.items(), key=lambda x: -len(x[1])):
        print(f"\n{category}: {len(features)} features")
        # Mostrar exemplos
        for i, feature in enumerate(features[:10]):
            print(f"   - {feature}")
        if len(features) > 10:
            print(f"   ... e mais {len(features) - 10} features")
    
    # Salvar lista completa de features perdidas
    lost_features_file = os.path.join(PROJECT_ROOT, "lost_features_analysis.csv")
    lost_df = pd.DataFrame([
        {'category': cat, 'feature': feat}
        for cat, feats in lost_categories.items()
        for feat in feats
    ])
    lost_df.to_csv(lost_features_file, index=False)
    print(f"\n💾 Lista completa de features perdidas salva em: {lost_features_file}")
    
    return lost_categories, old_df, new_df

def analyze_without_old_data(new_final):
    """Análise quando não temos os dados antigos."""
    
    new_df = pd.read_csv(new_final)
    
    print(f"\n🔍 ANÁLISE DO DATASET ATUAL:")
    print(f"   Total de features: {len(new_df.columns)}")
    print(f"   Features esperadas (baseado no histórico): 916")
    print(f"   Diferença: -{916 - len(new_df.columns)} features")
    
    # Analisar o que temos atualmente
    current_features = defaultdict(int)
    
    for col in new_df.columns:
        if 'tfidf_' in col.lower():
            current_features['TF-IDF'] += 1
        elif 'bow_' in col.lower():
            current_features['Bag of Words'] += 1
        elif 'sentiment' in col.lower():
            current_features['Sentiment'] += 1
        elif any(x in col.lower() for x in ['length', 'count']):
            current_features['Text Stats'] += 1
        elif col.startswith(('¿', 'Quando', 'Déjame')):
            current_features['Original Text'] += 1
        else:
            current_features['Other'] += 1
    
    print("\n📊 FEATURES ATUAIS POR TIPO:")
    for feat_type, count in sorted(current_features.items(), key=lambda x: -x[1]):
        print(f"   {feat_type}: {count}")
    
    # Identificar possíveis problemas
    print("\n⚠️  POSSÍVEIS PROBLEMAS IDENTIFICADOS:")
    
    if current_features['TF-IDF'] < 100:
        print("   - Poucas features TF-IDF (esperado: 200+)")
        print("     → Verificar max_features no TfidfVectorizer")
    
    if current_features['Text Stats'] < 20:
        print("   - Poucas estatísticas de texto")
        print("     → Verificar processamento no script 03")
    
    if 'lançamento' not in new_df.columns:
        print("   - Coluna 'lançamento' ausente")
        print("     → Impossível criar features por campanha")

def check_text_processing():
    """Verifica especificamente o processamento de texto."""
    
    print("\n\n📝 ANÁLISE DO PROCESSAMENTO DE TEXTO")
    print("-"*80)
    
    # Colunas de texto esperadas
    text_columns = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
        'Déjame un mensaje'
    ]
    
    # Verificar em cada estágio (V5)
    stages = [
        ('Raw', '01_split/train.csv'),
        ('Preprocessed', '02_preprocessed/train.csv'),
        ('Feature Eng 1', '03_feature_engineering_1/train.csv'),
        ('Feature Eng 2', '04_feature_engineering_2/train.csv')
    ]
    
    for stage_name, file_path in stages:
        full_path = os.path.join(NEW_DATA_PATH.replace('04_feature_engineering_2', ''), file_path)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            print(f"\n{stage_name} (V5):")
            
            # Verificar presença das colunas de texto
            for text_col in text_columns:
                if text_col in df.columns:
                    # Verificar se tem dados
                    non_null = df[text_col].notna().sum()
                    print(f"   ✅ {text_col[:50]}... ({non_null} valores)")
                else:
                    print(f"   ❌ {text_col[:50]}... (AUSENTE)")
            
            # Verificar features NLP geradas
            nlp_features = [col for col in df.columns if any(x in col for x in ['tfidf_', 'bow_', 'sentiment_'])]
            print(f"   📊 Features NLP geradas: {len(nlp_features)}")

if __name__ == "__main__":
    # 1. Análise profunda
    print("Iniciando análise profunda...")
    lost_categories, old_df, new_df = deep_feature_analysis()
    
    # 2. Verificar processamento de texto
    check_text_processing()
    
    # 3. Gerar recomendações específicas
    print("\n\n💡 RECOMENDAÇÕES ESPECÍFICAS:")
    print("-"*80)
    
    print("\n1. RECUPERAR FEATURES DE TEXTO:")
    print("   - Verificar parâmetros do TfidfVectorizer nos scripts 03/04")
    print("   - Aumentar max_features de 100 para 300-500")
    print("   - Verificar se min_df não está muito alto")
    
    print("\n2. PRESERVAR METADADOS:")
    print("   - Adicionar 'lançamento' como feature auxiliar no script 01")
    print("   - Manter no pipeline mesmo que não esteja em produção")
    print("   - Usar para criar features agregadas")
    
    print("\n3. IMPLEMENTAR PIPELINE DUPLA:")
    print("   ```python")
    print("   # No script 01:")
    print("   def prepare_dataset(df, mode='training'):")
    print("       if mode == 'training':")
    print("           # Manter todas as features úteis")
    print("           preserve_cols = ['lançamento', 'utm_campaign_original']")
    print("       else:  # mode == 'production'")
    print("           # Filtrar apenas colunas de produção")
    print("           df = filter_to_inference_columns(df)")
    print("   ```")
    
    print("\n✅ Análise concluída!")