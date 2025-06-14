# test_professional_features_real_data.py
# Script para testar ProfessionalFeatures com dados reais

import sys
import os
import pandas as pd
import numpy as np
import time
import warnings
import logging

# Silenciar warnings específicos
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Configurar logging para reduzir verbosidade
logging.getLogger('src.utils.column_type_classifier').setLevel(logging.WARNING)

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

from smart_ads_pipeline.components.professional_features import ProfessionalFeatures
from smart_ads_pipeline.core import ExtendedParameterManager


def test_with_real_data():
    """Testa ProfessionalFeatures com dados reais já processados."""
    print("=== Testando ProfessionalFeatures com Dados Reais ===\n")
    
    # Caminho para os dados processados
    data_path = "/Users/ramonmoreira/desktop/smart_ads/data/processed_data/train.csv"
    
    print(f"Carregando dados de: {data_path}")
    
    # Carregar dados (limitando para teste rápido)
    df = pd.read_csv(data_path, nrows=1000)  # Usar 1000 linhas para teste
    print(f"✓ Dados carregados: {df.shape}")
    
    # Verificar colunas de texto disponíveis
    text_patterns = ['cuando_hables_ingles', 'que_esperas_aprender', 'dejame_un_mensaje', 
                    'cual_es_tu_profesion', 'cual_es_tu_instagram']
    
    text_cols_found = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in text_patterns):
            # Verificar se não é uma feature já processada
            if not any(suffix in col for suffix in ['_tfidf_', '_sentiment', '_encoded', '_length']):
                text_cols_found.append(col)
    
    print(f"\nColunas de texto encontradas: {len(text_cols_found)}")
    for col in text_cols_found:
        print(f"  - {col}")
    
    # Criar e aplicar ProfessionalFeatures
    print("\n--- Aplicando ProfessionalFeatures ---")
    
    # Medir tempo
    start_time = time.time()
    
    processor = ProfessionalFeatures(n_topics=5)
    df_transformed = processor.fit_transform(df)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Processamento concluído em {elapsed_time:.2f} segundos")
    print(f"  Shape original: {df.shape}")
    print(f"  Shape final: {df_transformed.shape}")
    print(f"  Features criadas: {df_transformed.shape[1] - df.shape[1]}")
    
    # Analisar features criadas
    new_features = set(df_transformed.columns) - set(df.columns)
    
    # Categorizar
    categories = {
        'motivation': [],
        'aspiration': [],
        'commitment': [],
        'career_term': [],
        'career_tfidf': [],
        'topic': [],
        'dominant_topic': []
    }
    
    for feat in new_features:
        if 'professional_motivation' in feat or 'career_keyword' in feat:
            categories['motivation'].append(feat)
        elif 'aspiration' in feat:
            categories['aspiration'].append(feat)
        elif 'commitment' in feat:
            categories['commitment'].append(feat)
        elif 'career_tfidf' in feat:
            categories['career_tfidf'].append(feat)
        elif 'career_term' in feat and 'career_tfidf' not in feat:
            categories['career_term'].append(feat)
        elif 'dominant_topic' in feat:
            categories['dominant_topic'].append(feat)
        elif 'topic_' in feat:
            categories['topic'].append(feat)
    
    print("\n--- Features por Categoria ---")
    total_categorized = 0
    for cat, features in categories.items():
        if features:
            print(f"\n{cat.upper()} ({len(features)} features):")
            # Mostrar até 3 exemplos
            for feat in features[:3]:
                print(f"  • {feat}")
            if len(features) > 3:
                print(f"  ... e mais {len(features) - 3}")
            total_categorized += len(features)
    
    print(f"\nTotal categorizado: {total_categorized} de {len(new_features)}")
    
    # Verificar se LDA foi aplicado
    lda_features = [f for f in new_features if 'topic_' in f or 'dominant_topic' in f]
    if lda_features:
        print(f"\n✓ LDA aplicado com sucesso: {len(lda_features)} features de tópicos")
        
        # Mostrar distribuição de um tópico dominante
        dom_topic_cols = [col for col in df_transformed.columns if 'dominant_topic' in col]
        if dom_topic_cols:
            for col in dom_topic_cols[:1]:  # Primeiro tópico dominante
                topic_dist = df_transformed[col].value_counts().head()
                print(f"\nDistribuição de {col}:")
                print(topic_dist)
    
    # Testar save/load
    print("\n--- Testando Save/Load ---")
    param_manager = ExtendedParameterManager()
    processor.save_params(param_manager)
    
    # Criar novo processor e carregar
    processor2 = ProfessionalFeatures()
    processor2.load_params(param_manager)
    
    # Transformar subset para verificar
    df_subset = df.iloc[:100].copy()
    df_transformed2 = processor2.transform(df_subset)
    
    print(f"✓ Save/Load funcionando")
    print(f"  Transform em subset: {df_transformed2.shape}")
    
    # Mostrar informações do processador
    info = processor.get_feature_info()
    print("\n--- Informações do Processador ---")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return df_transformed


def analyze_memory_usage(df_original, df_transformed):
    """Analisa uso de memória antes e depois."""
    print("\n--- Análise de Memória ---")
    
    mem_original = df_original.memory_usage(deep=True).sum() / 1024**2  # MB
    mem_transformed = df_transformed.memory_usage(deep=True).sum() / 1024**2  # MB
    
    print(f"Memória original: {mem_original:.2f} MB")
    print(f"Memória após processamento: {mem_transformed:.2f} MB")
    print(f"Aumento: {mem_transformed - mem_original:.2f} MB ({(mem_transformed/mem_original - 1)*100:.1f}%)")


def test_fit_transform_consistency():
    """Testa se fit e transform criam o mesmo número de features."""
    print("\n=== Testando Consistência Fit/Transform ===")
    
    data_path = "/Users/ramonmoreira/desktop/smart_ads/data/processed_data/train.csv"
    
    # Carregar dois subsets diferentes
    df_fit = pd.read_csv(data_path, nrows=200)
    df_transform = pd.read_csv(data_path, skiprows=range(1, 201), nrows=100)
    
    # Aplicar fit
    processor = ProfessionalFeatures(n_topics=3)
    processor.fit(df_fit)
    
    # Aplicar transform em dados diferentes
    df_fit_transformed = processor.transform(df_fit)
    df_transform_transformed = processor.transform(df_transform)
    
    # Comparar features criadas
    features_fit = set(df_fit_transformed.columns) - set(df_fit.columns)
    features_transform = set(df_transform_transformed.columns) - set(df_transform.columns)
    
    print(f"\nFeatures no fit: {len(features_fit)}")
    print(f"Features no transform: {len(features_transform)}")
    
    # Verificar diferenças
    missing_in_transform = features_fit - features_transform
    extra_in_transform = features_transform - features_fit
    
    if missing_in_transform:
        print(f"\n⚠️  ALERTA: {len(missing_in_transform)} features faltando no transform!")
        print("Exemplos:", list(missing_in_transform)[:5])
    
    if extra_in_transform:
        print(f"\n⚠️  ALERTA: {len(extra_in_transform)} features extras no transform!")
        print("Exemplos:", list(extra_in_transform)[:5])
    
    if not missing_in_transform and not extra_in_transform:
        print("\n✅ Fit e Transform criam exatamente as mesmas features!")
    
    return len(features_fit) == len(features_transform)


def test_on_different_sizes():
    """Testa com diferentes tamanhos de dados."""
    print("\n=== Testando Performance com Diferentes Tamanhos ===")
    
    data_path = "/Users/ramonmoreira/desktop/smart_ads/data/processed_data/train.csv"
    
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n--- Testando com {size} linhas ---")
        
        df = pd.read_csv(data_path, nrows=size)
        
        start_time = time.time()
        processor = ProfessionalFeatures(n_topics=3)  # Menos tópicos para teste
        df_transformed = processor.fit_transform(df)
        elapsed = time.time() - start_time
        
        features_created = df_transformed.shape[1] - df.shape[1]
        print(f"  Tempo: {elapsed:.2f}s")
        print(f"  Features criadas: {features_created}")
        print(f"  Tempo por linha: {elapsed/size*1000:.2f}ms")


def main():
    """Executa todos os testes com dados reais."""
    print("Testando ProfessionalFeatures com Dados Reais do Smart Ads\n")
    
    try:
        # Teste principal
        df_original = pd.read_csv("/Users/ramonmoreira/desktop/smart_ads/data/processed_data/train.csv", nrows=1000)
        df_transformed = test_with_real_data()
        
        # Análise de memória
        if df_transformed is not None:
            analyze_memory_usage(df_original, df_transformed)
        
        # Teste de consistência fit/transform (NOVO!)
        test_fit_transform_consistency()
        
        # Teste de performance
        test_on_different_sizes()
        
        print("\n✅ Todos os testes com dados reais passaram!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()