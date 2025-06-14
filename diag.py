# diagnose_email_error.py
# Script para diagnosticar onde está ocorrendo o erro com 'e_mail'

import sys
import os

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

from smart_ads_pipeline.pipelines import TrainingPipeline

def test_with_debug():
    """Executa o pipeline com debug detalhado"""
    print("=== Diagnóstico do Erro 'e_mail' ===\n")
    
    # Configuração mínima
    config = {
        'data_path': "/Users/ramonmoreira/desktop/smart_ads/data/raw_data",
        'output_dir': "/tmp/smart_ads_debug",
        'test_size': 0.3,
        'val_size': 0.5,
        'random_state': 42,
        'max_features': 50,
        'fast_mode': True,
        'use_checkpoints': False,
        'clear_cache': False,
        'train_model': False,
        'sample_fraction': 0.01  # Apenas 1% para debug rápido
    }
    
    try:
        pipeline = TrainingPipeline()
        results = pipeline.run(config)
        
        if results['success']:
            print("✓ Pipeline executado com sucesso!")
        else:
            print(f"✗ Pipeline falhou: {results.get('error')}")
            print(f"\nÚltimo passo executado: {pipeline.state.get_last_step()}")
            print(f"\nPassos completados:")
            for step in pipeline.state.steps:
                print(f"  - {step['name']}: {step.get('status', 'unknown')}")
    
    except Exception as e:
        print(f"❌ Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Tentar identificar onde parou
        print("\n=== Estado do Pipeline ===")
        try:
            summary = pipeline.state.get_summary()
            print(f"Tempo de execução: {summary.get('execution_time_seconds', 'N/A')}s")
            print(f"Passos executados: {summary.get('steps_completed', 'N/A')}")
            print(f"Último passo: {summary.get('last_step', 'N/A')}")
        except:
            print("Não foi possível obter o estado do pipeline")


def check_data_columns():
    """Verifica as colunas após cada etapa"""
    print("\n=== Verificando Colunas em Cada Etapa ===\n")
    
    from smart_ads_pipeline.data_handlers import DataLoader, DataMatcher
    
    # 1. Carregar dados
    print("1. Carregando dados...")
    loader = DataLoader("/Users/ramonmoreira/desktop/smart_ads/data/raw_data")
    data_dict = loader.load_training_data()
    
    print(f"\nColunas em surveys: {list(data_dict['surveys'].columns)[:10]}...")
    print(f"Tem 'e_mail'? {'e_mail' in data_dict['surveys'].columns}")
    print(f"Tem 'E-MAIL'? {'E-MAIL' in data_dict['surveys'].columns}")
    
    # 2. Matching
    print("\n2. Aplicando matching...")
    matcher = DataMatcher()
    try:
        final_df = matcher.match_and_create_target(data_dict)
        print(f"\nColunas após matching: {list(final_df.columns)}")
        print(f"Tem 'e_mail'? {'e_mail' in final_df.columns}")
        print(f"Shape: {final_df.shape}")
    except Exception as e:
        print(f"❌ Erro no matching: {e}")
        
        # Verificar se o erro é antes ou depois da filtragem
        print("\nVerificando etapas do matching...")
        surveys = data_dict['surveys']
        buyers = data_dict['buyers']
        
        print(f"Surveys shape: {surveys.shape}")
        print(f"Buyers shape: {buyers.shape}")
        
        # Verificar normalização
        if 'e_mail' in surveys.columns:
            print("✓ 'e_mail' existe em surveys")
        elif 'E-MAIL' in surveys.columns:
            print("⚠️ Coluna está como 'E-MAIL' (maiúscula)")
        else:
            print("❌ Nenhuma coluna de email encontrada em surveys")


if __name__ == "__main__":
    # Primeiro verificar as colunas
    check_data_columns()
    
    # Depois tentar executar com debug
    print("\n" + "="*60 + "\n")
    test_with_debug()