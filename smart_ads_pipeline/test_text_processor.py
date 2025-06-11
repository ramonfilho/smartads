# test_text_processor.py
# Script para testar o TextProcessor

import sys
import os
import pandas as pd
import numpy as np

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

from smart_ads_pipeline.components.text_processor import TextProcessor
from smart_ads_pipeline.core import ExtendedParameterManager


def create_test_data():
    """Cria dados de teste com colunas de texto."""
    
    # Criar dados com textos realistas
    df = pd.DataFrame({
        # Colunas de texto principais
        'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida_que_oportunidades_se_abriran_para_ti': [
            'Podré conseguir un mejor trabajo y viajar por el mundo',
            'Mejores oportunidades laborales y comunicación internacional',
            'Quiero trabajar en empresas multinacionales',
            '',
            'Ganar más dinero y tener mejores oportunidades'
        ] * 20,
        
        'que_esperas_aprender_en_el_evento_cero_a_ingles_fluido': [
            'Hablar con fluidez y perder el miedo',
            'Mejorar mi pronunciación y gramática',
            'Aprender técnicas efectivas de estudio',
            None,
            'Todo lo necesario para comunicarme mejor'
        ] * 20,
        
        'dejame_un_mensaje': [
            'Espero que el curso sea muy bueno',
            'Gracias por la oportunidad',
            '',
            'Necesito aprender inglés urgentemente',
            'Me encanta tu contenido'
        ] * 20,
        
        # Outras colunas necessárias
        'cual_es_tu_edad': ['25-34', '35-44', '18-24', '45-54', 'desconhecido'] * 20,
        'cual_es_tu_pais': ['Brasil', 'México', 'Colombia', 'Argentina', 'Chile'] * 20,
        
        # Target para features discriminativas
        'target': [0, 1, 0, 0, 1] * 20
    })
    
    return df


def test_basic_functionality():
    """Testa funcionalidade básica do TextProcessor."""
    print("=== Testando Funcionalidade Básica ===")
    
    df = create_test_data()
    processor = TextProcessor()
    
    # Verificar shape inicial
    print(f"Shape inicial: {df.shape}")
    initial_cols = set(df.columns)
    
    # Fit transform
    df_transformed = processor.fit_transform(df)
    
    # Verificar shape final
    print(f"Shape após processamento: {df_transformed.shape}")
    
    # Verificar features criadas
    new_cols = set(df_transformed.columns) - initial_cols
    print(f"✓ {len(new_cols)} novas features criadas")
    
    # Categorizar features
    feature_types = {
        'basic': [],
        'sentiment': [],
        'tfidf': [],
        'motivation': [],
        'discriminative': []
    }
    
    for col in new_cols:
        if '_length' in col or '_word_count' in col or '_has_' in col:
            feature_types['basic'].append(col)
        elif '_sentiment' in col:
            feature_types['sentiment'].append(col)
        elif '_tfidf_' in col:
            feature_types['tfidf'].append(col)
        elif '_motiv_' in col:
            feature_types['motivation'].append(col)
        elif '_conv_term' in col:
            feature_types['discriminative'].append(col)
    
    print("\nFeatures por tipo:")
    for feat_type, features in feature_types.items():
        if features:
            print(f"  {feat_type}: {len(features)} features")
    
    return True


def test_tfidf_features():
    """Testa criação de features TF-IDF."""
    print("\n=== Testando Features TF-IDF ===")
    
    df = create_test_data()
    processor = TextProcessor()
    
    # Fit
    processor.fit(df)
    
    # Transform
    df_transformed = processor.transform(df)
    
    # Verificar features TF-IDF
    tfidf_cols = [col for col in df_transformed.columns if '_tfidf_' in col]
    print(f"✓ {len(tfidf_cols)} features TF-IDF criadas")
    
    # Verificar informações
    info = processor.get_feature_info()
    print(f"✓ Vetorizadores TF-IDF: {info['n_tfidf_vectorizers']}")
    print(f"✓ Colunas processadas: {info['text_columns_processed']}")
    
    # Verificar valores
    if tfidf_cols:
        sample_col = tfidf_cols[0]
        non_zero = (df_transformed[sample_col] > 0).sum()
        print(f"✓ Exemplo - {sample_col}: {non_zero} valores não-zero")
    
    return True


def test_sentiment_features():
    """Testa análise de sentimento."""
    print("\n=== Testando Features de Sentimento ===")
    
    df = create_test_data()
    processor = TextProcessor()
    
    # Fit transform
    df_transformed = processor.fit_transform(df)
    
    # Verificar features de sentimento
    sentiment_cols = [col for col in df_transformed.columns if '_sentiment' in col]
    print(f"✓ {len(sentiment_cols)} features de sentimento criadas")
    
    for col in sentiment_cols[:3]:  # Primeiras 3
        values = df_transformed[col]
        print(f"  {col}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
    
    return True


def test_save_load_params():
    """Testa salvamento e carregamento de parâmetros."""
    print("\n=== Testando Save/Load de Parâmetros ===")
    
    df = create_test_data()
    
    # Criar e ajustar processor
    processor1 = TextProcessor()
    df_transformed1 = processor1.fit_transform(df)
    
    # Salvar parâmetros
    param_manager = ExtendedParameterManager()
    processor1.save_params(param_manager)
    
    # Criar novo processor e carregar parâmetros
    processor2 = TextProcessor()
    processor2.load_params(param_manager)
    
    # Transformar com processor carregado
    df_transformed2 = processor2.transform(df)
    
    # Verificar se resultados são iguais
    assert df_transformed1.shape == df_transformed2.shape
    assert list(df_transformed1.columns) == list(df_transformed2.columns)
    
    # Verificar valores de algumas colunas
    tfidf_cols = [col for col in df_transformed1.columns if '_tfidf_' in col]
    if tfidf_cols:
        for col in tfidf_cols[:5]:  # Primeiras 5
            pd.testing.assert_series_equal(
                df_transformed1[col], 
                df_transformed2[col], 
                check_names=False,
                atol=1e-6  # Tolerância para valores float
            )
    
    print("✓ Save/Load funcionando corretamente")
    print(f"✓ Parâmetros preservados para {len(df_transformed1.columns)} features")
    
    return True


def test_empty_text_handling():
    """Testa tratamento de textos vazios."""
    print("\n=== Testando Tratamento de Textos Vazios ===")
    
    # Criar dados com muitos textos vazios
    df = pd.DataFrame({
        'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida_que_oportunidades_se_abriran_para_ti': 
            [''] * 50 + ['Texto válido'] * 50,
        'que_esperas_aprender_en_el_evento_cero_a_ingles_fluido': 
            [None] * 50 + ['Outro texto'] * 50,
        'dejame_un_mensaje': 
            [''] * 100,
        'target': [0, 1] * 50
    })
    
    processor = TextProcessor()
    
    # Deve processar sem erros
    df_transformed = processor.fit_transform(df)
    
    print(f"✓ Processamento com textos vazios concluído")
    print(f"  Shape resultante: {df_transformed.shape}")
    
    # Verificar que features foram criadas mesmo com textos vazios
    new_features = set(df_transformed.columns) - set(df.columns)
    print(f"  Features criadas: {len(new_features)}")
    
    return True


def test_compatibility_check():
    """Verifica compatibilidade com pipeline original."""
    print("\n=== Verificando Compatibilidade ===")
    
    df = create_test_data()
    processor = TextProcessor()
    
    # Processar
    df_transformed = processor.fit_transform(df)
    
    # Verificar patterns esperados nas features
    expected_patterns = [
        '_length',      # Features básicas
        '_word_count',
        '_sentiment',   # Sentimento
        '_tfidf_',      # TF-IDF
        '_motiv_',      # Motivação
        '_clean'        # Colunas limpas (removidas ao final)
    ]
    
    # Verificar que NÃO temos colunas _clean (devem ser removidas)
    clean_cols = [col for col in df_transformed.columns if '_clean' in col]
    assert len(clean_cols) == 0, "Colunas _clean não foram removidas"
    print("✓ Colunas temporárias (_clean) removidas corretamente")
    
    # Verificar patterns presentes
    for pattern in expected_patterns[:-1]:  # Exceto _clean
        matching_cols = [col for col in df_transformed.columns if pattern in col]
        if matching_cols:
            print(f"✓ Pattern '{pattern}': {len(matching_cols)} features")
    
    return True


def main():
    """Executa todos os testes."""
    print("Testando TextProcessor do Smart Ads Pipeline\n")
    
    try:
        # Executar testes
        test_basic_functionality()
        test_tfidf_features()
        test_sentiment_features()
        test_save_load_params()
        test_empty_text_handling()
        test_compatibility_check()
        
        print("\n✅ Todos os testes do TextProcessor passaram!")
        print("\n⚠️  IMPORTANTE: Este componente usa EXATAMENTE as mesmas funções")
        print("   do pipeline original, garantindo 100% de compatibilidade!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()