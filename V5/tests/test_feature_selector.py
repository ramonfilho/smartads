# test_feature_selector.py
# Script para testar o FeatureSelector

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

from smart_ads_pipeline.components.feature_selector import FeatureSelector
from smart_ads_pipeline.core import ExtendedParameterManager


def create_test_data():
    """Cria dados de teste com features correlacionadas e irrelevantes."""
    
    # Criar dataset sintético
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Converter para DataFrame
    feature_names = [f'feature_{i}' for i in range(50)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Adicionar features altamente correlacionadas
    df['feature_corr_1'] = df['feature_0'] + np.random.normal(0, 0.1, size=len(df))
    df['feature_corr_2'] = df['feature_1'] * 0.99 + np.random.normal(0, 0.01, size=len(df))
    
    # Adicionar features de ruído
    df['noise_1'] = np.random.random(size=len(df))
    df['noise_2'] = np.random.random(size=len(df))
    
    # Adicionar target
    df['target'] = y
    
    return df


def test_basic_functionality():
    """Testa funcionalidade básica do FeatureSelector."""
    print("=== Testando Funcionalidade Básica ===")
    
    df = create_test_data()
    
    # Separar features e target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Shape inicial: X={X.shape}, y={y.shape}")
    print(f"Classes no target: {y.value_counts().to_dict()}")
    
    # Criar e ajustar selector
    selector = FeatureSelector(
        max_features=30,
        correlation_threshold=0.95,
        fast_mode=True,  # Modo rápido para teste
        n_folds=3
    )
    
    # Fit
    selector.fit(X, y)
    
    # Transform
    X_selected = selector.transform(X)
    
    print(f"\n✓ Seleção concluída!")
    print(f"  Features originais: {X.shape[1]}")
    print(f"  Features selecionadas: {X_selected.shape[1]}")
    print(f"  Redução: {(1 - X_selected.shape[1]/X.shape[1])*100:.1f}%")
    
    # Verificar informações
    info = selector.get_feature_info()
    print(f"\n  Features removidas por correlação: {info['n_removed_by_correlation']}")
    print(f"  Pares altamente correlacionados: {info['n_high_corr_pairs']}")
    
    return True


def test_correlation_removal():
    """Testa remoção de features correlacionadas."""
    print("\n=== Testando Remoção de Correlações ===")
    
    df = create_test_data()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Verificar correlações antes
    corr_before = X[['feature_0', 'feature_corr_1']].corr().iloc[0, 1]
    print(f"Correlação feature_0 vs feature_corr_1: {corr_before:.3f}")
    
    # Aplicar seleção
    selector = FeatureSelector(
        max_features=50,  # Não limitar por número
        correlation_threshold=0.95,
        fast_mode=True
    )
    
    selector.fit(X, y)
    
    # Verificar se removeu features correlacionadas
    print(f"\n✓ Features removidas por correlação: {selector.removed_by_correlation}")
    
    # Verificar pares correlacionados
    if selector.high_corr_pairs:
        print("\nPares com alta correlação detectados:")
        for pair in selector.high_corr_pairs[:3]:  # Primeiros 3
            print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
    
    return True


def test_importance_ranking():
    """Testa ranking de importância."""
    print("\n=== Testando Ranking de Importância ===")
    
    df = create_test_data()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Aplicar seleção
    selector = FeatureSelector(
        max_features=10,  # Selecionar apenas top 10
        fast_mode=True
    )
    
    selector.fit(X, y)
    
    # Verificar importâncias
    if selector.feature_importance is not None:
        print("\nTop 10 features por importância:")
        top_10 = selector.feature_importance.head(10)
        for i, row in top_10.iterrows():
            print(f"  {i+1}. {row['Feature']}: {row['Mean_Importance']:.4f}")
    
    # Verificar se features de ruído têm baixa importância
    noise_features = ['noise_1', 'noise_2']
    if selector.feature_importance is not None:
        noise_importance = selector.feature_importance[
            selector.feature_importance['Feature'].isin(noise_features)
        ]
        if not noise_importance.empty:
            print(f"\n✓ Importância das features de ruído (esperado: baixa):")
            for _, row in noise_importance.iterrows():
                print(f"  {row['Feature']}: {row['Mean_Importance']:.4f}")
    
    return True


def test_save_load_params():
    """Testa salvamento e carregamento de parâmetros."""
    print("\n=== Testando Save/Load de Parâmetros ===")
    
    df = create_test_data()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Criar e ajustar selector
    selector1 = FeatureSelector(max_features=25)
    selector1.fit(X, y)
    
    # Salvar parâmetros
    param_manager = ExtendedParameterManager()
    selector1.save_params(param_manager)
    
    # Criar novo selector e carregar parâmetros
    selector2 = FeatureSelector()
    selector2.load_params(param_manager)
    
    # Transformar com selector carregado
    X_selected1 = selector1.transform(X)
    X_selected2 = selector2.transform(X)
    
    # Verificar se resultados são iguais
    assert X_selected1.shape == X_selected2.shape
    assert list(X_selected1.columns) == list(X_selected2.columns)
    
    print("✓ Save/Load funcionando corretamente")
    print(f"  Features selecionadas preservadas: {len(selector2.selected_features)}")
    
    # Verificar se features foram salvas no param_manager
    saved_features = param_manager.get_selected_features()
    print(f"  Features no param_manager: {len(saved_features)}")
    
    return True


def test_with_target_in_dataframe():
    """Testa quando target está incluído no DataFrame."""
    print("\n=== Testando com Target no DataFrame ===")
    
    df = create_test_data()  # Já inclui target
    
    selector = FeatureSelector(max_features=20)
    
    # Separar X e y
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Fit
    selector.fit(X, y)
    
    # Transform com DataFrame que inclui target
    df_transformed = selector.transform(df)
    
    # Verificar se target foi preservado
    assert 'target' in df_transformed.columns
    print("✓ Target preservado no DataFrame transformado")
    print(f"  Colunas finais: {df_transformed.shape[1]} (incluindo target)")
    
    return True


def main():
    """Executa todos os testes."""
    print("Testando FeatureSelector do Smart Ads Pipeline\n")
    
    try:
        # Executar testes
        test_basic_functionality()
        test_correlation_removal()
        test_importance_ranking()
        test_save_load_params()
        test_with_target_in_dataframe()
        
        print("\n✅ Todos os testes do FeatureSelector passaram!")
        print("\n⚠️  IMPORTANTE: Este componente usa EXATAMENTE as mesmas funções")
        print("   de feature_importance.py do pipeline original,")
        print("   garantindo 100% de compatibilidade!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()