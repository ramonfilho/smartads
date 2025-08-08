# test_feature_engineer.py
# Script para testar o FeatureEngineer

import sys
import os
import pandas as pd
import numpy as np

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

from smart_ads_pipeline.components.feature_engineer import FeatureEngineer
from smart_ads_pipeline.core import ExtendedParameterManager


def create_test_data():
    """Cria dados de teste para feature engineering."""
    
    # Criar dados mais completos
    df = pd.DataFrame({
        # Dados de identidade
        'como_te_llamas': ['João Silva', 'Maria Santos', 'Pedro Costa', None, 'Ana Lima'] * 20,
        'cual_es_tu_telefono': ['+55 11 98765-4321', '123456', None, '987654321', 'invalido'] * 20,
        'cual_es_tu_instagram': ['@joao', None, '@pedro_costa', '', '@ana.lima'] * 20,
        'cual_es_tu_profesion': ['Engenheiro', 'Médica', 'Professor', None, 'Designer'] * 20,
        
        # Dados temporais
        'marca_temporal': pd.date_range('2024-01-01 08:00:00', periods=100, freq='3H'),
        'data': pd.date_range('2024-01-01', periods=100, freq='6H'),
        
        # Dados categóricos ordinais
        'cual_es_tu_edad': ['25 años a 34 años', '35 años a 44 años', '18 años a 24 años', 
                           'Mas de 54', 'desconhecido'] * 20,
        'hace_quanto_tiempo_me_conoces': ['Te sigo desde hace 1 año', 'Te acabo de conocer a través del anuncio.', 
                                         'Te sigo hace más de 2 años', 'desconhecido', 'Te sigo desde hace 3 meses'] * 20,
        'cual_es_tu_disponibilidad_de_tiempo_para_estudiar_ingles': ['2 horas al día', '1 hora al día', 
                                                                     'Más de 3 horas al día', 'desconhecido', 
                                                                     'Menos de 1 hora al día'] * 20,
        'cual_es_tu_sueldo_anual_en_dolares': ['US$3000 a US$5000', 'Menos de US$3000', 'US$10000 o más', 
                                               'desconhecido', 'US$5000 o más'] * 20,
        'cuanto_te_gustaria_ganar_al_ano': ['Más de US$10000 por año', 'Al menos US$ 3000 por año', 
                                            'Más de US$20000 por año', 'desconhecido', 'Más de US$5000 por año'] * 20,
        'crees_que_aprender_ingles_te_acercaria_mas_al_salario_que_mencionaste_anteriormente': 
            ['¡Sí, sin duda!', 'Tal vez', 'Creo que no...', 'desconhecido', '¡Si por su puesto!'] * 20,
        'crees_que_aprender_ingles_puede_ayudarte_en_el_trabajo_o_en_tu_vida_diaria':
            ['¡Sí, sin duda!', 'Tal vez', 'desconhecido', 'Creo que no...', '¡Si por su puesto!'] * 20,
        'cual_es_tu_genero': ['Hombre', 'Mujer', 'desconhecido', 'Mujer', 'Hombre'] * 20,
        
        # País (alta cardinalidade)
        'cual_es_tu_pais': ['Brasil', 'México', 'Colombia', 'Argentina', 'Chile', 
                           'Peru', 'España', 'Uruguay', 'Venezuela', 'Ecuador'] * 10,
        
        # UTMs
        'utm_source': ['google', 'facebook', 'instagram', 'unknown', 'organic'] * 20,
        'utm_medium': ['cpc', 'social', 'email', 'unknown', 'referral'] * 20,
        'utm_campaign': ['campaign_' + str(i % 15) for i in range(100)],  # Alta cardinalidade
        'gclid': [None] * 50 + ['gclid_123456'] * 50,
        
        # Target
        'target': [0, 1, 0, 0, 1] * 20
    })
    
    return df


def test_identity_features():
    """Testa criação de features de identidade."""
    print("=== Testando Features de Identidade ===")
    
    df = create_test_data()
    engineer = FeatureEngineer()
    
    # Verificar colunas antes
    print(f"Colunas antes: {len(df.columns)}")
    
    # Fit transform
    df_transformed = engineer.fit_transform(df)
    
    # Verificar features criadas
    identity_features = ['name_length', 'name_word_count', 'valid_phone', 'has_instagram']
    for feat in identity_features:
        assert feat in df_transformed.columns, f"Feature {feat} não foi criada"
    
    print("✓ Features de identidade criadas:")
    print(f"  - name_length: média = {df_transformed['name_length'].mean():.1f}")
    print(f"  - name_word_count: média = {df_transformed['name_word_count'].mean():.1f}")
    print(f"  - valid_phone: {df_transformed['valid_phone'].sum()} válidos")
    print(f"  - has_instagram: {df_transformed['has_instagram'].sum()} com instagram")
    
    # Verificar remoção de colunas originais
    assert 'como_te_llamas' not in df_transformed.columns
    assert 'cual_es_tu_telefono' not in df_transformed.columns
    assert 'cual_es_tu_instagram' not in df_transformed.columns
    assert 'cual_es_tu_profesion' not in df_transformed.columns
    print("✓ Colunas originais removidas")
    
    return True


def test_temporal_features():
    """Testa criação de features temporais."""
    print("\n=== Testando Features Temporais ===")
    
    df = create_test_data()
    engineer = FeatureEngineer()
    
    # Fit transform
    df_transformed = engineer.fit_transform(df)
    
    # Verificar features temporais
    temporal_features = ['hour', 'day_of_week', 'month', 'year', 
                        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'period_of_day',
                        'utm_hour', 'utm_day_of_week', 'utm_month', 'utm_year']
    
    created_features = [f for f in temporal_features if f in df_transformed.columns]
    print(f"✓ {len(created_features)} features temporais criadas:")
    
    for feat in ['hour', 'day_of_week', 'month', 'year']:
        if feat in df_transformed.columns:
            print(f"  - {feat}: min={df_transformed[feat].min()}, max={df_transformed[feat].max()}")
    
    # Verificar features cíclicas
    if 'hour_sin' in df_transformed.columns:
        assert -1.1 <= df_transformed['hour_sin'].min() <= 1.1
        assert -1.1 <= df_transformed['hour_cos'].max() <= 1.1
        print("✓ Features cíclicas no intervalo correto")
    
    # Verificar período do dia
    if 'period_of_day' in df_transformed.columns:
        periods = df_transformed['period_of_day'].value_counts()
        print(f"✓ Períodos do dia: {periods.to_dict()}")
    
    return True


def test_categorical_encoding():
    """Testa encoding de variáveis categóricas."""
    print("\n=== Testando Encoding Categórico ===")
    
    df = create_test_data()
    engineer = FeatureEngineer()
    
    # Fit transform
    df_transformed = engineer.fit_transform(df)
    
    # Verificar encodings ordinais
    ordinal_features = ['age_encoded', 'time_known_encoded', 'availability_encoded',
                       'current_salary_encoded', 'desired_salary_encoded', 
                       'belief_salary_encoded', 'belief_work_encoded', 'gender_encoded']
    
    for feat in ordinal_features:
        if feat in df_transformed.columns:
            unique_values = df_transformed[feat].unique()
            print(f"✓ {feat}: valores únicos = {sorted(unique_values)}")
    
    # Verificar encoding de país
    assert 'country_freq' in df_transformed.columns
    assert 'country_encoded' in df_transformed.columns
    print(f"✓ País - frequency encoding: min={df_transformed['country_freq'].min():.3f}, "
          f"max={df_transformed['country_freq'].max():.3f}")
    print(f"✓ País - label encoding: {df_transformed['country_encoded'].nunique()} categorias")
    
    # Verificar UTM encoding
    utm_encoded = [col for col in df_transformed.columns if 'utm_' in col and '_encoded' in col]
    utm_freq = [col for col in df_transformed.columns if 'utm_' in col and '_freq' in col]
    print(f"✓ UTMs - {len(utm_encoded)} com label encoding, {len(utm_freq)} com frequency encoding")
    
    # Verificar GCLID
    assert 'has_gclid' in df_transformed.columns
    print(f"✓ GCLID: {df_transformed['has_gclid'].sum()} com gclid")
    
    return True


def test_save_load_params():
    """Testa salvamento e carregamento de parâmetros."""
    print("\n=== Testando Save/Load de Parâmetros ===")
    
    df = create_test_data()
    
    # Criar e ajustar engineer
    engineer1 = FeatureEngineer()
    df_transformed1 = engineer1.fit_transform(df)
    
    # Salvar parâmetros
    param_manager = ExtendedParameterManager()
    engineer1.save_params(param_manager)
    
    # Criar novo engineer e carregar parâmetros
    engineer2 = FeatureEngineer()
    engineer2.load_params(param_manager)
    
    # Transformar com engineer carregado
    df_transformed2 = engineer2.transform(df)
    
    # Verificar se resultados são iguais
    assert df_transformed1.shape == df_transformed2.shape
    assert list(df_transformed1.columns) == list(df_transformed2.columns)
    
    # Verificar alguns valores
    for col in ['age_encoded', 'country_freq', 'has_gclid']:
        if col in df_transformed1.columns:
            pd.testing.assert_series_equal(df_transformed1[col], df_transformed2[col], check_names=False)
    
    print("✓ Save/Load funcionando corretamente")
    print(f"  Parâmetros categóricos salvos: {len(engineer1.categorical_params)} items")
    
    return True


def test_preserved_columns():
    """Testa se colunas são preservadas durante fit."""
    print("\n=== Testando Preservação de Colunas ===")
    
    df = create_test_data()
    engineer = FeatureEngineer()
    
    # Fit
    engineer.fit(df)
    
    # Verificar colunas preservadas
    print(f"✓ Colunas preservadas para processamento posterior: {list(engineer.preserved_columns.keys())}")
    
    for col, data in engineer.preserved_columns.items():
        print(f"  - {col}: {len(data)} valores preservados")
    
    return True


def main():
    """Executa todos os testes."""
    print("Testando FeatureEngineer do Smart Ads Pipeline\n")
    
    try:
        # Executar testes
        test_identity_features()
        test_temporal_features()
        test_categorical_encoding()
        test_save_load_params()
        test_preserved_columns()
        
        print("\n✅ Todos os testes do FeatureEngineer passaram!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()