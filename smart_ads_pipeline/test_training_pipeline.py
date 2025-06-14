# test_training_pipeline.py
# Script para testar o TrainingPipeline

import sys
import os
import tempfile
import shutil
from datetime import datetime

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

from smart_ads_pipeline.pipelines import TrainingPipeline


def test_training_pipeline_minimal():
    """Testa o pipeline de treino com configuração mínima."""
    print("=== Testando TrainingPipeline (Minimal) ===")
    
    # Criar diretório temporário para output
    output_dir = tempfile.mkdtemp(prefix="smart_ads_test_")
    
    try:
        # Configuração mínima
        config = {
            'data_path': "/Users/ramonmoreira/desktop/smart_ads/data/raw_data",
            'output_dir': output_dir,
            'test_size': 0.3,
            'val_size': 0.5,
            'random_state': 42,
            'max_features': 50,  # Menos features para teste rápido
            'fast_mode': True,
            'use_checkpoints': False,  # Sem checkpoints no teste
            'clear_cache': False,
            'train_model': False,  # Sem treinar modelo no teste básico
            'sample_fraction': 0.1  # NOVO: Usar apenas 10% dos dados
        }
        
        # Executar pipeline
        pipeline = TrainingPipeline()
        results = pipeline.run(config)
        
        # Verificar resultados
        assert results['success'], f"Pipeline falhou: {results.get('error')}"
        
        print("\n✓ Pipeline executado com sucesso!")
        print(f"  Parâmetros salvos em: {results['params_path']}")
        print(f"  Shapes finais:")
        print(f"    Train: {results['train_shape']}")
        print(f"    Val: {results['val_shape']}")
        print(f"    Test: {results['test_shape']}")
        print(f"  Features selecionadas: {len(results['selected_features'])}")
        
        # Verificar arquivos criados
        expected_files = [
            'pipeline_params.joblib',
            'train.csv',
            'val.csv',
            'test.csv',
            'feature_importance.csv'
        ]
        
        for file in expected_files:
            file_path = os.path.join(output_dir, file)
            assert os.path.exists(file_path), f"Arquivo esperado não encontrado: {file}"
        
        print("\n✓ Todos os arquivos esperados foram criados")
        
        return True
        
    finally:
        # Limpar diretório temporário
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"\n✓ Diretório temporário removido: {output_dir}")


def test_training_pipeline_with_model():
    """Testa o pipeline incluindo treinamento de modelo."""
    print("\n=== Testando TrainingPipeline com Modelo ===")
    
    # Criar diretório temporário
    output_dir = tempfile.mkdtemp(prefix="smart_ads_model_test_")
    
    try:
        config = {
            'data_path': "/Users/ramonmoreira/desktop/smart_ads/data/raw_data",
            'output_dir': output_dir,
            'test_size': 0.3,
            'val_size': 0.5,
            'random_state': 42,
            'max_features': 30,  # Poucas features para teste rápido
            'fast_mode': True,
            'use_checkpoints': False,
            'train_model': True  # TREINAR MODELO
        }
        
        # Executar pipeline
        pipeline = TrainingPipeline()
        results = pipeline.run(config)
        
        assert results['success'], f"Pipeline falhou: {results.get('error')}"
        
        print("\n✓ Pipeline com modelo executado com sucesso!")
        print(f"  Modelo salvo em: {results.get('model_path')}")
        print(f"  Métricas: {results.get('metrics')}")
        
        # Verificar se modelo foi criado
        model_path = os.path.join(output_dir, 'model.pkl')
        assert os.path.exists(model_path), "Modelo não foi salvo"
        
        # Verificar tamanho do modelo
        model_size = os.path.getsize(model_path) / 1024  # KB
        print(f"  Tamanho do modelo: {model_size:.1f} KB")
        
        return True
        
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def test_training_pipeline_with_checkpoints():
    """Testa o pipeline com checkpoints habilitados."""
    print("\n=== Testando TrainingPipeline com Checkpoints ===")
    
    output_dir = tempfile.mkdtemp(prefix="smart_ads_checkpoint_test_")
    cache_dir = os.path.join(output_dir, "cache")
    
    try:
        # Configurar para usar cache personalizado
        os.environ['SMART_ADS_CACHE_DIR'] = cache_dir
        
        config = {
            'data_path': "/Users/ramonmoreira/desktop/smart_ads/data/raw_data",
            'output_dir': output_dir,
            'test_size': 0.3,
            'val_size': 0.5,
            'max_features': 20,
            'fast_mode': True,
            'use_checkpoints': True,  # USAR CHECKPOINTS
            'clear_cache': True,
            'train_model': False
        }
        
        # Primeira execução
        print("\nPrimeira execução (criando checkpoints)...")
        pipeline1 = TrainingPipeline()
        results1 = pipeline1.run(config)
        assert results1['success']
        
        # Verificar que checkpoints foram criados
        checkpoint_files = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
        print(f"\n✓ Checkpoints criados: {len(checkpoint_files)} arquivos")
        
        # Segunda execução (usando checkpoints)
        print("\nSegunda execução (usando checkpoints)...")
        config['clear_cache'] = False  # Não limpar cache
        
        pipeline2 = TrainingPipeline()
        results2 = pipeline2.run(config)
        assert results2['success']
        
        # Comparar tempos (segunda deve ser mais rápida)
        time1 = pipeline1.state.get_summary()['execution_time_seconds']
        time2 = pipeline2.state.get_summary()['execution_time_seconds']
        
        print(f"\n✓ Tempo primeira execução: {time1:.1f}s")
        print(f"✓ Tempo segunda execução: {time2:.1f}s")
        print(f"✓ Speedup com checkpoints: {time1/time2:.1f}x")
        
        return True
        
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def test_data_path_validation():
    """Verifica se o caminho de dados existe."""
    print("\n=== Verificando Caminho de Dados ===")
    
    data_path = "/Users/ramonmoreira/desktop/smart_ads/data/raw_data"
    
    if os.path.exists(data_path):
        print(f"✓ Caminho de dados existe: {data_path}")
        
        # Listar alguns arquivos
        files = [f for f in os.listdir(data_path) if f.endswith(('.csv', '.xlsx'))]
        print(f"  Arquivos encontrados: {len(files)}")
        if files:
            print(f"  Exemplos: {files[:3]}")
        
        return True
    else:
        print(f"⚠️  Caminho de dados não encontrado: {data_path}")
        print("  O pipeline precisa de dados reais para funcionar")
        return False


def main():
    """Executa todos os testes."""
    print("Testando TrainingPipeline do Smart Ads\n")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # Verificar dados primeiro
        if not test_data_path_validation():
            print("\n❌ Não é possível executar testes sem dados")
            return
        
        # Executar testes
        test_training_pipeline_minimal()
        test_training_pipeline_with_model()
        # test_training_pipeline_with_checkpoints()  # Comentado por enquanto
        
        print("\n✅ Todos os testes do TrainingPipeline passaram!")
        print("\n📊 RESUMO:")
        print("  - Pipeline básico: OK")
        print("  - Pipeline com modelo: OK")
        print("  - Salvamento de parâmetros: OK")
        print("  - Geração de datasets: OK")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()