# check_data_state.py
# Script para verificar em que estado os dados estão

import sys
import os
import pandas as pd

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)


def check_data_files():
    """Verifica os arquivos de dados disponíveis e seu estado."""
    
    data_dirs = {
        'raw': '/Users/ramonmoreira/desktop/smart_ads/data/raw_data',
        'split': '/Users/ramonmoreira/desktop/smart_ads/data/new/01_split',
        'preprocessed': '/Users/ramonmoreira/desktop/smart_ads/data/new/02_processed',
        'features': '/Users/ramonmoreira/desktop/smart_ads/data/new/03_feature_engineering',
        'unified': '/Users/ramonmoreira/desktop/smart_ads/data/unified_v1',
        "processed": '/Users/ramonmoreira/desktop/smart_ads/data/processed_data'
    }
    
    print("Verificando arquivos de dados disponíveis:\n")
    
    for stage, path in data_dirs.items():
        print(f"=== {stage.upper()} ===")
        print(f"Path: {path}")
        
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.csv')]
            print(f"✓ Diretório existe")
            print(f"  Arquivos CSV: {files}")
            
            # Se houver train.csv, verificar colunas
            train_path = os.path.join(path, 'train.csv')
            if os.path.exists(train_path):
                df = pd.read_csv(train_path, nrows=5)
                print(f"  Shape: {df.shape}")
                print(f"  Primeiras colunas: {list(df.columns)[:10]}")
                
                # Verificar se tem colunas padronizadas
                padronized_patterns = ['cuando_hables_ingles', 'que_esperas_aprender', 'dejame_un_mensaje']
                has_padronized = any(any(pattern in col for pattern in padronized_patterns) 
                                   for col in df.columns)
                
                if has_padronized:
                    print(f"  ✓ Tem colunas padronizadas (snake_case)")
                else:
                    print(f"  ✗ Sem colunas padronizadas (ainda com acentos/caracteres especiais)")
                
                # Verificar se tem features de texto processadas
                text_features = [col for col in df.columns if any(pattern in col for pattern in 
                               ['_tfidf_', '_sentiment', '_length', '_word_count'])]
                
                if text_features:
                    print(f"  ✓ Tem features de texto: {len(text_features)} features")
                else:
                    print(f"  ✗ Sem features de texto processadas")
                    
        else:
            print(f"✗ Diretório não existe")
        
        print()


def analyze_best_dataset():
    """Analisa qual dataset é melhor para testar ProfessionalFeatures."""
    
    print("\n=== RECOMENDAÇÃO ===")
    
    # Verificar se existe o diretório de features processadas
    features_path = '/Users/ramonmoreira/desktop/smart_ads/data/new/03_features/train.csv'
    
    if os.path.exists(features_path):
        df = pd.read_csv(features_path, nrows=100)
        
        # Verificar colunas de texto padronizadas
        text_cols = []
        patterns = ['cuando_hables_ingles', 'que_esperas_aprender', 'dejame_un_mensaje', 
                   'cual_es_tu_profesion', 'cual_es_tu_instagram']
        
        for col in df.columns:
            if any(pattern in col.lower() for pattern in patterns):
                # Verificar se não é uma feature já processada
                if not any(suffix in col for suffix in ['_tfidf_', '_sentiment', '_encoded']):
                    text_cols.append(col)
        
        print(f"Dataset recomendado: {features_path}")
        print(f"Colunas de texto encontradas: {len(text_cols)}")
        for col in text_cols:
            print(f"  - {col}")
        
        return features_path
    
    else:
        print("⚠️  Não encontrei dados já processados.")
        print("Você precisará:")
        print("1. Carregar dados brutos")
        print("2. Aplicar DataPreprocessor")
        print("3. Aplicar FeatureEngineer")
        print("4. Aplicar TextProcessor")
        print("5. Então aplicar ProfessionalFeatures")
        
        return None


def create_test_with_real_data(data_path):
    """Cria script de teste usando dados reais."""
    
    print("\n=== CÓDIGO DE TESTE COM DADOS REAIS ===")
    print(f"""
# Teste com dados reais
import pandas as pd
from smart_ads_pipeline.components import ProfessionalFeatures

# Carregar dados
df = pd.read_csv('{data_path}')
print(f"Dados carregados: {{df.shape}}")

# Aplicar ProfessionalFeatures
processor = ProfessionalFeatures(n_topics=5)
df_transformed = processor.fit_transform(df)

# Verificar resultados
new_features = set(df_transformed.columns) - set(df.columns)
print(f"\\nFeatures criadas: {{len(new_features)}}")

# Categorizar
categories = {{'motivation': 0, 'aspiration': 0, 'commitment': 0, 
             'career_term': 0, 'career_tfidf': 0, 'topic': 0}}

for feat in new_features:
    if 'motivation' in feat: categories['motivation'] += 1
    elif 'aspiration' in feat: categories['aspiration'] += 1
    elif 'commitment' in feat: categories['commitment'] += 1
    elif 'career_tfidf' in feat: categories['career_tfidf'] += 1
    elif 'career_term' in feat: categories['career_term'] += 1
    elif 'topic_' in feat or 'dominant_topic' in feat: categories['topic'] += 1

print("\\nFeatures por categoria:")
for cat, count in categories.items():
    if count > 0:
        print(f"  {{cat}}: {{count}} features")
""")


if __name__ == "__main__":
    check_data_files()
    best_dataset = analyze_best_dataset()
    
    if best_dataset:
        create_test_with_real_data(best_dataset)