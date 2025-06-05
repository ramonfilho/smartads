import joblib
import pandas as pd
import numpy as np
from unified_pipeline import unified_data_pipeline

print("=== DIAGNÓSTICO DE PROBLEMAS COM MODELOS ===\n")

# 1. Carregar os parâmetros salvos
params_path = "/Users/ramonmoreira/desktop/smart_ads/src/preprocessing/params/unified_v3/all_preprocessing_params.joblib"
params = joblib.load(params_path)

print("1. ESTRUTURA DOS PARÂMETROS:")
print(f"   Chaves principais: {list(params.keys())}")

# 2. Verificar parâmetros de texto
if 'text_processing' in params:
    text_params = params['text_processing']
    print("\n2. PARÂMETROS DE PROCESSAMENTO DE TEXTO:")
    print(f"   Chaves: {list(text_params.keys())}")
    
    # Verificar TF-IDF
    if 'tfidf' in text_params:
        print(f"\n   TF-IDF vetorizadores salvos:")
        for col, data in text_params['tfidf'].items():
            print(f"      - {col}:")
            print(f"        Tem vetorizador? {'vectorizer' in data}")
            print(f"        Número de features: {len(data.get('feature_names', []))}")

# 3. Verificar parâmetros profissionais
if 'professional_features' in params:
    prof_params = params['professional_features']
    print("\n3. PARÂMETROS DE FEATURES PROFISSIONAIS:")
    print(f"   Chaves: {list(prof_params.keys())}")
    
    # Verificar vetorizadores de carreira
    if 'career_tfidf' in prof_params:
        print(f"\n   Career TF-IDF vetorizadores:")
        for col, data in prof_params.get('career_tfidf', {}).items():
            if isinstance(data, dict):
                print(f"      - {col}:")
                print(f"        Tem vetorizador? {'vectorizer' in data}")
                print(f"        Features: {len(data.get('feature_names', []))}")
    
    # Verificar LDA
    if 'lda' in prof_params:
        print(f"\n   Modelos LDA:")
        for col, data in prof_params.get('lda', {}).items():
            print(f"      - {col}:")
            print(f"        Tem modelo? {'model' in data}")
            print(f"        Tópicos: {data.get('n_topics', 0)}")

# 4. Carregar datasets processados e verificar valores
print("\n4. VERIFICAÇÃO DE VALORES NOS DATASETS:")

# Carregar um dos datasets
train_df = pd.read_csv("/Users/ramonmoreira/desktop/smart_ads/data/processed_data/train.csv")
val_df = pd.read_csv("/Users/ramonmoreira/desktop/smart_ads/data/processed_data/validation.csv")

# Verificar TF-IDF features
tfidf_cols = [col for col in train_df.columns if '_tfidf_' in col]
print(f"\n   Features TF-IDF encontradas: {len(tfidf_cols)}")

# Verificar se têm valores não-zero
train_tfidf_nonzero = (train_df[tfidf_cols] != 0).sum().sum()
val_tfidf_nonzero = (val_df[tfidf_cols] != 0).sum().sum()

print(f"   Valores não-zero no TRAIN: {train_tfidf_nonzero}")
print(f"   Valores não-zero no VAL: {val_tfidf_nonzero}")

# Verificar topic features
topic_cols = [col for col in train_df.columns if '_topic_' in col]
print(f"\n   Features de tópicos encontradas: {len(topic_cols)}")

train_topic_nonzero = (train_df[topic_cols] != 0).sum().sum()
val_topic_nonzero = (val_df[topic_cols] != 0).sum().sum()

print(f"   Valores não-zero no TRAIN: {train_topic_nonzero}")
print(f"   Valores não-zero no VAL: {val_topic_nonzero}")

# 5. Teste rápido de recuperação de modelos
print("\n5. TESTE DE RECUPERAÇÃO DE MODELOS:")

# Tentar acessar um vetorizador
try:
    if 'vectorizers' in params and 'career_tfidf' in params['vectorizers']:
        career_tfidf = params['vectorizers']['career_tfidf']
        first_key = list(career_tfidf.keys())[0] if career_tfidf else None
        if first_key:
            vectorizer = career_tfidf[first_key].get('vectorizer')
            print(f"   ✓ Vetorizador recuperado para '{first_key}'")
            print(f"     Tipo: {type(vectorizer)}")
except Exception as e:
    print(f"   ✗ Erro ao recuperar vetorizador: {e}")