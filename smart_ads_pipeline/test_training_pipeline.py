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


def test_training_pipeline(use_checkpoints=False, sample_fraction=0.1, train_model=False):
    """
    Testa o pipeline de treino com configurações parametrizáveis.
    
    Args:
        use_checkpoints: Se deve usar checkpoints
        sample_fraction: Fração dos dados para usar (0.1 = 10%)
        train_model: Se deve treinar um modelo LightGBM
    """
    print(f"=== Testando TrainingPipeline ===")
    print(f"Configurações:")
    print(f"  - Checkpoints: {'Sim' if use_checkpoints else 'Não'}")
    print(f"  - Amostragem: {sample_fraction*100:.0f}%")
    print(f"  - Treinar modelo: {'Sim' if train_model else 'Não'}")
    print()
    
    # Criar diretório temporário para output
    output_dir = tempfile.mkdtemp(prefix="smart_ads_test_")
    
    try:
        # Configuração
        config = {
            'data_path': "/Users/ramonmoreira/desktop/smart_ads/data/raw_data",
            'output_dir': output_dir,
            'test_size': 0.3,
            'val_size': 0.5,
            'random_state': 42,
            'max_features': 50,  # Menos features para teste rápido
            'fast_mode': True,
            'use_checkpoints': use_checkpoints,
            'clear_cache': False,
            'train_model': train_model,
            'sample_fraction': sample_fraction
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
        
        if train_model and 'metrics' in results:
            print(f"  Métricas do modelo:")
            print(f"    AUC Validação: {results['metrics']['auc_val']:.4f}")
        
        # Verificar arquivos criados
        expected_files = [
            'pipeline_params.joblib',
            'train.csv',
            'val.csv',
            'test.csv',
            'feature_importance.csv'
        ]
        
        if train_model:
            expected_files.append('model.pkl')
        
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
    """Executa o teste com configurações padrão ou da linha de comando."""
    print("Testando TrainingPipeline do Smart Ads\n")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Parâmetros padrão
    use_checkpoints = False
    sample_fraction = 0.1
    train_model = False
    
    # Parse argumentos da linha de comando (simples)
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '--checkpoints':
                use_checkpoints = True
            elif arg == '--full':
                sample_fraction = 1.0
            elif arg == '--model':
                train_model = True
            elif arg.startswith('--sample='):
                sample_fraction = float(arg.split('=')[1])
    
    try:
        # Verificar dados primeiro
        if not test_data_path_validation():
            print("\n❌ Não é possível executar testes sem dados")
            return
        
        # Executar teste
        test_training_pipeline(
            use_checkpoints=use_checkpoints,
            sample_fraction=sample_fraction,
            train_model=train_model
        )
        
        print("\n✅ Teste do TrainingPipeline concluído!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()