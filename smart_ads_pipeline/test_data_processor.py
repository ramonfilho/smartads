# test_data_preprocessor.py
# Script para testar o DataPreprocessor

import sys
import os
import pandas as pd
import numpy as np

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# Debug: verificar sys.path
print(f"Python path includes: {project_root}")
print(f"Current working directory: {os.getcwd()}")

try:
    from smart_ads_pipeline.components.data_preprocessor import DataPreprocessor
    from smart_ads_pipeline.core import ExtendedParameterManager
    print("✓ Imports bem-sucedidos!")
except ImportError as e:
    print(f"❌ Erro de import: {e}")
    print("Verificando estrutura de diretórios...")
    import os
    for root, dirs, files in os.walk(os.path.join(project_root, "smart_ads_pipeline")):
        level = root.replace(project_root, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py'):
                print(f"{subindent}{file}")
    sys.exit(1)


def create_test_data():
    """Cria dados de teste com problemas típicos para preprocessamento."""
    
    # Criar dados com problemas intencionais
    df = pd.DataFrame({
        # Dados normais
        'e_mail': ['user1@gmail.com', 'user2@gmail.com', None, 'user4@gmail.com', 'user5@gmail.com'] * 20,
        'cual_es_tu_edad': ['25-34', None, '35-44', '25-34', 'desconhecido'] * 20,
        'cual_es_tu_pais': ['Brasil', 'México', None, 'Brasil', 'Argentina'] * 20,
        
        # Colunas de qualidade (múltiplas variantes)
        'Qualidade (Número)': [8, 9, None, 7, 10] * 20,
        'Qualidade (Nome)': ['Bom', 'Ótimo', 'Regular', None, 'Excelente'] * 20,
        
        # Coluna com muitos missing (>95%)
        'coluna_vazia': [None] * 96 + [1, 2, 3, 4],
        
        # Dados numéricos com outliers
        'idade_numerica': list(range(18, 38)) * 4 + [150, 200, -10, 999, 0] * 4,
        
        # Texto
        'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida': [
            'Mejores oportunidades', None, '', 'Viajar más', 'Nuevo trabajo'
        ] * 20,
        
        # UTM
        'utm_source': ['google', 'facebook', None, 'instagram', 'organic'] * 20,
        
        # Temporal
        'marca_temporal': ['2024-01-15 10:30:00', '2024-01-16 14:45:00', None, 
                          '2024-01-17 09:15:00', '2024-01-18 16:20:00'] * 20,
        
        # Target
        'target': [0, 1, 0, 0, 1] * 20
    })
    
    return df


def test_consolidate_quality_columns():
    """Testa consolidação de colunas de qualidade."""
    print("=== Testando Consolidação de Colunas de Qualidade ===")
    
    df = create_test_data()
    preprocessor = DataPreprocessor()
    
    # Verificar colunas originais
    quality_cols_before = [col for col in df.columns if 'Qualidade' in col]
    print(f"Colunas de qualidade antes: {quality_cols_before}")
    
    # Fit transform
    df_transformed = preprocessor.fit_transform(df)
    
    # Verificar colunas depois
    quality_cols_after = [col for col in df_transformed.columns if 'qualidade' in col.lower()]
    print(f"Colunas de qualidade depois: {quality_cols_after}")
    
    # Verificar se consolidou corretamente
    assert 'qualidade_numerica' in df_transformed.columns
    assert 'qualidade_textual' in df_transformed.columns
    assert 'Qualidade (Número)' not in df_transformed.columns
    assert 'Qualidade (Nome)' not in df_transformed.columns
    
    print("✓ Consolidação funcionou corretamente")
    
    # Verificar valores
    print(f"  Valores numéricos não nulos: {df_transformed['qualidade_numerica'].notna().sum()}")
    print(f"  Valores textuais não nulos: {df_transformed['qualidade_textual'].notna().sum()}")
    
    return True


def test_missing_values():
    """Testa tratamento de valores ausentes."""
    print("\n=== Testando Tratamento de Missing Values ===")
    
    df = create_test_data()
    preprocessor = DataPreprocessor()
    
    # Contar missing antes
    missing_before = df.isnull().sum()
    print(f"Missing values antes (amostra):")
    print(missing_before[missing_before > 0].head())
    
    # Fit transform
    df_transformed = preprocessor.fit_transform(df)
    
    # Verificar remoção de coluna com >95% missing
    assert 'coluna_vazia' not in df_transformed.columns
    print("✓ Coluna com >95% missing removida")
    
    # Verificar preenchimento de UTMs
    assert df_transformed['utm_source'].isna().sum() == 0
    assert (df_transformed['utm_source'] == 'unknown').sum() > 0
    print("✓ UTMs preenchidas com 'unknown'")
    
    # Verificar preenchimento de textos
    text_col = 'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida'
    assert df_transformed[text_col].isna().sum() == 0
    assert (df_transformed[text_col] == '').sum() > 0
    print("✓ Textos preenchidos com string vazia")
    
    # Verificar medianas numéricas
    assert df_transformed['idade_numerica'].isna().sum() == 0
    print("✓ Valores numéricos preenchidos com mediana")
    
    return True


def test_outliers():
    """Testa tratamento de outliers."""
    print("\n=== Testando Tratamento de Outliers ===")
    
    df = create_test_data()
    preprocessor = DataPreprocessor()
    
    # Valores extremos antes
    print(f"Idade numérica - Min antes: {df['idade_numerica'].min()}")
    print(f"Idade numérica - Max antes: {df['idade_numerica'].max()}")
    
    # Fit transform
    df_transformed = preprocessor.fit_transform(df)
    
    # Valores após tratamento
    print(f"Idade numérica - Min depois: {df_transformed['idade_numerica'].min():.2f}")
    print(f"Idade numérica - Max depois: {df_transformed['idade_numerica'].max():.2f}")
    
    # Verificar se outliers foram cortados
    assert df_transformed['idade_numerica'].min() > -50  # Não deve ter valores muito negativos
    assert df_transformed['idade_numerica'].max() < 100  # Não deve ter valores muito altos
    
    print("✓ Outliers tratados com sucesso")
    
    return True


def test_normalization():
    """Testa normalização de valores."""
    print("\n=== Testando Normalização ===")
    
    df = create_test_data()
    preprocessor = DataPreprocessor()
    
    # Fit transform
    df_transformed = preprocessor.fit_transform(df)
    
    # Verificar normalização
    numeric_cols = df_transformed.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col != 'target']
    
    for col in numeric_cols:
        if df_transformed[col].std() > 0:  # Só verificar colunas com variabilidade
            mean = df_transformed[col].mean()
            std = df_transformed[col].std()
            print(f"{col} - Média: {mean:.6f}, Desvio: {std:.6f}")
            
            # Verificar se está aproximadamente normalizado
            assert abs(mean) < 0.1  # Média próxima de 0
            assert 0.9 < std < 1.1  # Desvio próximo de 1
    
    print("✓ Normalização funcionando corretamente")
    
    return True


def test_save_load_params():
    """Testa salvamento e carregamento de parâmetros."""
    print("\n=== Testando Save/Load de Parâmetros ===")
    
    df = create_test_data()
    
    # Criar e ajustar preprocessor
    preprocessor1 = DataPreprocessor()
    df_transformed1 = preprocessor1.fit_transform(df)
    
    # Salvar parâmetros
    param_manager = ExtendedParameterManager()
    preprocessor1.save_params(param_manager)
    
    # Criar novo preprocessor e carregar parâmetros
    preprocessor2 = DataPreprocessor()
    preprocessor2.load_params(param_manager)
    
    # Transformar com preprocessor carregado
    df_transformed2 = preprocessor2.transform(df)
    
    # Verificar se resultados são iguais
    pd.testing.assert_frame_equal(df_transformed1, df_transformed2)
    
    print("✓ Save/Load funcionando corretamente")
    print(f"  Parâmetros salvos: {list(preprocessor1.quality_params.keys())}")
    
    return True


def test_data_types():
    """Testa conversão de tipos de dados."""
    print("\n=== Testando Conversão de Tipos ===")
    
    df = create_test_data()
    preprocessor = DataPreprocessor()
    
    # Verificar tipo antes
    print(f"Tipo de 'marca_temporal' antes: {df['marca_temporal'].dtype}")
    
    # Fit transform
    df_transformed = preprocessor.fit_transform(df)
    
    # Verificar tipo depois
    print(f"Tipo de 'marca_temporal' depois: {df_transformed['marca_temporal'].dtype}")
    
    # Deve ser datetime
    assert pd.api.types.is_datetime64_any_dtype(df_transformed['marca_temporal'])
    
    print("✓ Conversão de tipos funcionando")
    
    return True


def main():
    """Executa todos os testes."""
    print("Testando DataPreprocessor do Smart Ads Pipeline\n")
    
    try:
        # Executar testes
        test_consolidate_quality_columns()
        test_missing_values()
        test_outliers()
        test_normalization()
        test_save_load_params()
        test_data_types()
        
        print("\n✅ Todos os testes do DataPreprocessor passaram!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()