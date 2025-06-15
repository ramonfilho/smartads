# test_base_structure.py
# Script para testar a estrutura base implementada

import sys
import pandas as pd
import numpy as np

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

from smart_ads_pipeline.core import BaseComponent, PipelineState, ExtendedParameterManager


# Criar um componente de exemplo para teste
class ExampleComponent(BaseComponent):
    """Componente de exemplo para testar a estrutura base."""
    
    def __init__(self):
        super().__init__(name="ExampleComponent")
        self.mean_value = None
    
    def fit(self, X: pd.DataFrame, y=None):
        self._validate_input(X)
        # Aprender algo simples: média de uma coluna numérica
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.mean_value = X[numeric_cols[0]].mean()
        else:
            self.mean_value = 0
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame):
        self._check_is_fitted()
        self._validate_input(X)
        # Transformação simples: adicionar coluna com mean_value
        X_transformed = X.copy()
        X_transformed['example_feature'] = self.mean_value
        return X_transformed
    
    def _save_component_params(self, param_manager):
        param_manager.save_component_params(
            self.name,
            {'mean_value': self.mean_value}
        )
    
    def _load_component_params(self, param_manager):
        params = param_manager.get_component_params(self.name)
        self.mean_value = params.get('mean_value', 0)


def test_base_component():
    """Testa a funcionalidade do BaseComponent."""
    print("=== Testando BaseComponent ===")
    
    # Criar dados de teste
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e']
    })
    
    # Criar e testar componente
    component = ExampleComponent()
    
    # Testar fit_transform
    df_transformed = component.fit_transform(df)
    print(f"✓ fit_transform executado")
    print(f"  Nova coluna adicionada: {df_transformed.columns.tolist()}")
    print(f"  Valor aprendido: {component.mean_value}")
    
    # Testar save/load com ParameterManager
    param_manager = ExtendedParameterManager()
    component.save_params(param_manager)
    print(f"✓ Parâmetros salvos")
    
    # Criar novo componente e carregar parâmetros
    component2 = ExampleComponent()
    component2.load_params(param_manager)
    print(f"✓ Parâmetros carregados")
    print(f"  Valor recuperado: {component2.mean_value}")
    
    return True


def test_pipeline_state():
    """Testa a funcionalidade do PipelineState."""
    print("\n=== Testando PipelineState ===")
    
    # Criar estado
    state = PipelineState()
    
    # Criar dados de teste
    train_df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'target': [0, 1, 0, 1, 0]
    })
    
    val_df = pd.DataFrame({
        'feature1': [6, 7],
        'target': [1, 0]
    })
    
    # Atualizar estado
    state.update_dataframes(train=train_df, val=val_df)
    state.log_step("data_loading", {"source": "test"})
    
    # Testar validação
    state.validate_for_training()
    print("✓ Validação para treino passou")
    
    # Testar resumo
    summary = state.get_summary()
    print(f"✓ Resumo do estado:")
    print(f"  Shapes: {summary['data_shapes']}")
    print(f"  Steps executados: {summary['n_steps_executed']}")
    
    return True


def test_extended_parameter_manager():
    """Testa a funcionalidade do ExtendedParameterManager."""
    print("\n=== Testando ExtendedParameterManager ===")
    
    param_manager = ExtendedParameterManager()
    
    # Salvar parâmetros de componente
    param_manager.save_component_params(
        'test_component',
        {'param1': 10, 'param2': 'teste'}
    )
    
    # Salvar features selecionadas
    param_manager.save_selected_features(['feat1', 'feat2', 'feat3'])
    
    # Recuperar e verificar
    comp_params = param_manager.get_component_params('test_component')
    selected_feats = param_manager.get_selected_features()
    
    print(f"✓ Parâmetros do componente: {comp_params}")
    print(f"✓ Features selecionadas: {selected_feats}")
    
    # Testar resumo
    summary = param_manager.get_component_summary()
    print(f"✓ Resumo dos componentes: {summary}")
    
    # Testar validação
    issues = param_manager.validate_for_prediction()
    print(f"✓ Issues encontradas: {len(issues)}")
    if issues:
        print("  Issues (esperadas neste momento):")
        for issue in issues:
            print(f"    - {issue}")
    
    return True


def main():
    """Executa todos os testes."""
    print("Testando estrutura base do Smart Ads Pipeline OOP\n")
    
    try:
        # Executar testes
        test_base_component()
        test_pipeline_state()
        test_extended_parameter_manager()
        
        print("\n✅ Todos os testes passaram! A estrutura base está funcionando.")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()