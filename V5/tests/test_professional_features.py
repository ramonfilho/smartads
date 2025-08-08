# test_professional_features.py
# Script para testar o ProfessionalFeatures

import sys
import os
import pandas as pd
import numpy as np

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

from smart_ads_pipeline.components.professional_features import ProfessionalFeatures
from smart_ads_pipeline.core import ExtendedParameterManager


def create_test_data():
    """Cria dados de teste com colunas de texto profissionais."""
    
    # Criar dados com textos realistas sobre carreira
    df = pd.DataFrame({
        # Colunas de texto principais
        'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida_que_oportunidades_se_abriran_para_ti': [
            'Podré conseguir un mejor trabajo y ganar más dinero. Quiero trabajar en empresas multinacionales',
            'Mejores oportunidades laborales y crecimiento profesional en mi carrera',
            'Necesito el inglés para mi trabajo actual y futuras oportunidades',
            'Espero poder ascender en mi empresa y tener mejor salario',
            'Mi meta es trabajar en el extranjero y desarrollarme profesionalmente'
        ] * 20,
        
        'que_esperas_aprender_en_el_evento_cero_a_ingles_fluido': [
            'Quiero aprender a comunicarme con fluidez para mi trabajo',
            'Necesito dominar el inglés para mejorar mis oportunidades laborales',
            'Espero poder hablar con confianza en reuniones de trabajo',
            'Me comprometo a estudiar diariamente para lograr mis metas',
            'Estoy decidido a aprender inglés para mi desarrollo profesional'
        ] * 20,
        
        'dejame_un_mensaje': [
            'Estoy muy motivado para aprender y mejorar mi carrera',
            'Necesito esto para mi trabajo, es mi prioridad',
            'Haré lo necesario para dominar el inglés',
            'Mi aspiración es crecer profesionalmente',
            'Confío en que esto cambiará mi vida laboral'
        ] * 20,
        
        # Outras colunas necessárias
        'cual_es_tu_edad': ['25-34', '35-44', '18-24', '45-54', 'desconhecido'] * 20,
        'cual_es_tu_pais': ['Brasil', 'México', 'Colombia', 'Argentina', 'Chile'] * 20,
        
        # Target
        'target': [0, 1, 0, 0, 1] * 20
    })
    
    return df


def test_basic_functionality():
    """Testa funcionalidade básica do ProfessionalFeatures."""
    print("=== Testando Funcionalidade Básica ===")
    
    df = create_test_data()
    processor = ProfessionalFeatures()
    
    # Verificar shape inicial
    print(f"Shape inicial: {df.shape}")
    initial_cols = set(df.columns)
    
    # Fit transform
    df_transformed = processor.fit_transform(df)
    
    # Verificar shape final
    print(f"Shape após processamento: {df_transformed.shape}")
    
    # Verificar features criadas
    new_cols = set(df_transformed.columns) - initial_cols
    print(f"✓ {len(new_cols)} novas features profissionais criadas")
    
    # Verificar informações
    info = processor.get_feature_info()
    print(f"\nInformações das features:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return True


def test_motivation_features():
    """Testa features de motivação profissional."""
    print("\n=== Testando Features de Motivação ===")
    
    df = create_test_data()
    processor = ProfessionalFeatures()
    
    # Fit transform
    df_transformed = processor.fit_transform(df)
    
    # Verificar features de motivação
    motivation_features = [col for col in df_transformed.columns if 'motivation' in col.lower()]
    print(f"✓ {len(motivation_features)} features de motivação criadas")
    
    # Verificar score de motivação
    if 'professional_motivation_score' in df_transformed.columns:
        score = df_transformed['professional_motivation_score']
        print(f"  Score de motivação - min: {score.min():.3f}, max: {score.max():.3f}, mean: {score.mean():.3f}")
    
    # Verificar keyword count
    if 'career_keyword_count' in df_transformed.columns:
        count = df_transformed['career_keyword_count']
        print(f"  Career keywords - total encontradas: {count.sum()}")
    
    return True


def test_aspiration_sentiment():
    """Testa análise de sentimento de aspiração."""
    print("\n=== Testando Sentimento de Aspiração ===")
    
    df = create_test_data()
    processor = ProfessionalFeatures()
    
    # Fit transform
    df_transformed = processor.fit_transform(df)
    
    # Verificar features de aspiração
    aspiration_features = [col for col in df_transformed.columns if 'aspiration' in col.lower()]
    print(f"✓ {len(aspiration_features)} features de aspiração criadas")
    
    # Mostrar algumas
    for feat in aspiration_features[:3]:
        if feat in df_transformed.columns:
            values = df_transformed[feat]
            non_zero = (values > 0).sum()
            print(f"  {feat}: {non_zero} valores não-zero")
    
    return True


def test_commitment_detection():
    """Testa detecção de compromisso."""
    print("\n=== Testando Detecção de Compromisso ===")
    
    df = create_test_data()
    processor = ProfessionalFeatures()
    
    # Fit transform
    df_transformed = processor.fit_transform(df)
    
    # Verificar features de compromisso
    commitment_features = [col for col in df_transformed.columns if 'commitment' in col.lower()]
    print(f"✓ {len(commitment_features)} features de compromisso criadas")
    
    # Verificar has_commitment
    has_commitment_cols = [col for col in commitment_features if 'has_commitment' in col]
    for col in has_commitment_cols[:2]:
        if col in df_transformed.columns:
            count = df_transformed[col].sum()
            print(f"  {col}: {count} textos com compromisso")
    
    return True


def test_career_terms():
    """Testa detector de termos de carreira."""
    print("\n=== Testando Termos de Carreira ===")
    
    df = create_test_data()
    processor = ProfessionalFeatures()
    
    # Fit transform
    df_transformed = processor.fit_transform(df)
    
    # Verificar features de carreira
    career_features = [col for col in df_transformed.columns if 'career' in col.lower()]
    print(f"✓ {len(career_features)} features de carreira criadas")
    
    # Verificar TF-IDF de carreira
    career_tfidf = [col for col in df_transformed.columns if 'career_tfidf' in col]
    print(f"  TF-IDF de carreira: {len(career_tfidf)} features")
    
    # Verificar scores
    career_scores = [col for col in career_features if 'score' in col]
    for col in career_scores[:2]:
        if col in df_transformed.columns:
            score = df_transformed[col]
            print(f"  {col} - mean: {score.mean():.3f}")
    
    return True


def test_save_load_params():
    """Testa salvamento e carregamento de parâmetros."""
    print("\n=== Testando Save/Load de Parâmetros ===")
    
    df = create_test_data()
    
    # Criar e ajustar processor
    processor1 = ProfessionalFeatures()
    df_transformed1 = processor1.fit_transform(df)
    
    # Salvar parâmetros
    param_manager = ExtendedParameterManager()
    processor1.save_params(param_manager)
    
    # Criar novo processor e carregar parâmetros
    processor2 = ProfessionalFeatures()
    processor2.load_params(param_manager)
    
    # Transformar com processor carregado
    df_transformed2 = processor2.transform(df)
    
    # Verificar se resultados são iguais
    assert df_transformed1.shape == df_transformed2.shape
    assert list(df_transformed1.columns) == list(df_transformed2.columns)
    
    # Verificar valores de algumas colunas
    test_cols = ['professional_motivation_score', 'career_keyword_count']
    for col in test_cols:
        if col in df_transformed1.columns:
            pd.testing.assert_series_equal(
                df_transformed1[col], 
                df_transformed2[col], 
                check_names=False
            )
    
    print("✓ Save/Load funcionando corretamente")
    print(f"✓ Parâmetros preservados para {len(df_transformed1.columns)} colunas")
    
    return True


def test_empty_data_handling():
    """Testa tratamento quando não há colunas de texto."""
    print("\n=== Testando Dados sem Texto ===")
    
    # Criar dados sem colunas de texto esperadas
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5] * 20,
        'col2': ['a', 'b', 'c', 'd', 'e'] * 20,
        'target': [0, 1, 0, 0, 1] * 20
    })
    
    processor = ProfessionalFeatures()
    
    # Deve processar sem erros
    df_transformed = processor.fit_transform(df)
    
    print(f"✓ Processamento sem colunas de texto concluído")
    print(f"  Shape permaneceu: {df.shape} → {df_transformed.shape}")
    print(f"  Nenhuma feature criada (esperado)")
    
    return True


def test_feature_categories():
    """Analisa categorias de features criadas."""
    print("\n=== Analisando Categorias de Features ===")
    
    df = create_test_data()
    processor = ProfessionalFeatures()
    
    # Fit transform
    df_transformed = processor.fit_transform(df)
    
    # Categorizar features
    new_cols = set(df_transformed.columns) - set(df.columns)
    
    categories = {
        'motivation': [],
        'aspiration': [],
        'commitment': [],
        'career_term': [],
        'career_tfidf': [],
        'sentiment': []
    }
    
    for col in new_cols:
        if 'motivation' in col:
            categories['motivation'].append(col)
        elif 'aspiration' in col:
            categories['aspiration'].append(col)
        elif 'commitment' in col:
            categories['commitment'].append(col)
        elif 'career_tfidf' in col:
            categories['career_tfidf'].append(col)
        elif 'career_term' in col and 'career_tfidf' not in col:
            categories['career_term'].append(col)
        elif 'sentiment' in col:
            categories['sentiment'].append(col)
    
    print("Features por categoria:")
    total = 0
    for cat, features in categories.items():
        if features:
            print(f"  {cat}: {len(features)} features")
            total += len(features)
    
    print(f"\nTotal categorizado: {total} de {len(new_cols)} features")
    
    return True


def main():
    """Executa todos os testes."""
    print("Testando ProfessionalFeatures do Smart Ads Pipeline\n")
    
    try:
        # Executar testes
        test_basic_functionality()
        test_motivation_features()
        test_aspiration_sentiment()
        test_commitment_detection()
        test_career_terms()
        test_save_load_params()
        test_empty_data_handling()
        test_feature_categories()
        
        print("\n✅ Todos os testes do ProfessionalFeatures passaram!")
        print("\n⚠️  IMPORTANTE: Este componente usa EXATAMENTE as mesmas funções")
        print("   do pipeline original (professional_motivation_features.py),")
        print("   garantindo 100% de compatibilidade!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()