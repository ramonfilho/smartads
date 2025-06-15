# test_pipeline_300_features.py
import sys
sys.path.insert(0, '/Users/ramonmoreira/desktop/smart_ads')

from smart_ads_pipeline.pipelines.training_pipeline import TrainingPipeline

def test_with_300_features():
    """Testa o pipeline com 300 features como o original"""
    
    print("=== Teste com 300 Features ===\n")
    
    config = {
        'data_path': '/Users/ramonmoreira/desktop/smart_ads/data/raw_data',
        'output_dir': '/tmp/smart_ads_300_features',
        'test_size': 0.3,
        'val_size': 0.5,
        'random_state': 42,
        'max_features': 300,  # MUDANÇA PRINCIPAL
        'fast_mode': False,   # Usar todos os modelos para importância
        'sample_fraction': 1.0,  # Usar todos os dados
        'train_model': True,  # Treinar modelo também
        'use_checkpoints': False
    }
    
    pipeline = TrainingPipeline()
    
    try:
        results = pipeline.run(config)
        
        print(f"\n📊 Resultados:")
        print(f"   Features selecionadas: {len(results['selected_features'])}")
        print(f"   Tempo total: {results['summary']['total_time']:.1f}s")
        
        # Verificar se temos ~300 features
        if len(results['selected_features']) >= 250:
            print(f"   ✅ Sucesso! Pipeline gerou {len(results['selected_features'])} features")
        else:
            print(f"   ⚠️  Apenas {len(results['selected_features'])} features geradas")
        
        # Salvar lista de features para comparação
        with open('/tmp/features_300.txt', 'w') as f:
            for feat in results['selected_features']:
                f.write(f"{feat}\n")
                
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_300_features()