# test_feature_alignment.py
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/ramonmoreira/desktop/smart_ads')

from smart_ads_pipeline.pipelines.training_pipeline import TrainingPipeline

def test_feature_alignment():
    """Testa se o alinhamento de features está funcionando corretamente"""
    
    print("=== Teste de Alinhamento de Features ===\n")
    
    # Configurar pipeline com amostra pequena para teste rápido
    config = {
        'data_path': '/Users/ramonmoreira/desktop/smart_ads/data/raw_data',
        'output_dir': '/tmp/smart_ads_alignment_test',
        'test_size': 0.3,
        'val_size': 0.5,
        'random_state': 42,
        'max_features': 50,
        'fast_mode': True,
        'sample_fraction': 0.01,  # Usar apenas 1% dos dados
        'train_model': False
    }
    
    print("Configuração do teste:")
    print(f"  - Usando {config['sample_fraction']*100}% dos dados")
    print(f"  - Fast mode: {config['fast_mode']}")
    print(f"  - Max features: {config['max_features']}\n")
    
    # Criar e executar pipeline
    pipeline = TrainingPipeline()
    
    try:
        results = pipeline.run(config)
        
        # O pipeline retorna um dicionário com os datasets processados
        # Vamos verificar a estrutura do retorno
        print(f"\n🔍 Estrutura do resultado:")
        print(f"   Chaves disponíveis: {list(results.keys())}")
        
        # Verificar se temos os datasets
        if 'train' in results and 'val' in results and 'test' in results:
            train_shape = results['train'].shape
            val_shape = results['val'].shape
            test_shape = results['test'].shape
            
            print(f"\n📊 Shapes dos datasets:")
            print(f"  Train: {train_shape}")
            print(f"  Val:   {val_shape}")
            print(f"  Test:  {test_shape}")
            
            # Verificar se número de colunas é igual
            if train_shape[1] == val_shape[1] == test_shape[1]:
                print(f"\n✅ Alinhamento correto! Todos têm {train_shape[1]} colunas")
            else:
                print(f"\n❌ Problema no alinhamento!")
                print(f"   Train: {train_shape[1]} colunas")
                print(f"   Val: {val_shape[1]} colunas")
                print(f"   Test: {test_shape[1]} colunas")
                
                # Diagnosticar diferenças
                train_cols = set(results['train'].columns)
                val_cols = set(results['val'].columns)
                test_cols = set(results['test'].columns)
                
                # Colunas extras em val/test
                val_extra = val_cols - train_cols
                test_extra = test_cols - train_cols
                
                # Colunas faltando em val/test
                val_missing = train_cols - val_cols
                test_missing = train_cols - test_cols
                
                if val_extra or val_missing:
                    print(f"\n   Validação:")
                    if val_extra:
                        print(f"     - Colunas extras: {list(val_extra)[:5]}...")
                    if val_missing:
                        print(f"     - Colunas faltando: {list(val_missing)[:5]}...")
                
                if test_extra or test_missing:
                    print(f"\n   Teste:")
                    if test_extra:
                        print(f"     - Colunas extras: {list(test_extra)[:5]}...")
                    if test_missing:
                        print(f"     - Colunas faltando: {list(test_missing)[:5]}...")
        
        # Verificar features selecionadas
        if 'selected_features' in results:
            print(f"\n📋 Features selecionadas: {len(results['selected_features'])}")
            print(f"   Top 5: {results['selected_features'][:5]}")
        
        # Verificar parâmetros
        if 'param_manager' in results:
            print(f"\n📁 Parâmetros salvos com sucesso")
            selected = results['param_manager'].get_selected_features()
            if selected:
                print(f"   Features no param_manager: {len(selected)}")
        
        # Mostrar estatísticas finais
        print(f"\n📈 Estatísticas finais:")
        if 'state' in pipeline.__dict__:
            state_summary = pipeline.state.get_summary()
            print(f"   Passos executados: {state_summary.get('n_steps_executed', 'N/A')}")
            if 'data_shapes' in state_summary:
                print(f"   Shapes finais: {state_summary['data_shapes']}")
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_alignment()
    print("\n✅ Teste de alinhamento concluído!")