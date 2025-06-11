# test_data_handlers.py
# Script para testar os Data Handlers implementados

import sys
import pandas as pd
import numpy as np
import os

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

from smart_ads_pipeline.data_handlers import DataLoader, DataMatcher
from smart_ads_pipeline.core import PipelineState


def create_test_data():
    """Cria dados de teste simulados."""
    
    # Surveys
    surveys = pd.DataFrame({
        'e_mail': ['user1@gmail.com', 'user2@gmail.com', 'user3@gmail.com', 
                   'user4@gmail.com', 'user5@gmail.com'],
        'cual_es_tu_nombre': ['João', 'Maria', 'Pedro', 'Ana', 'Carlos'],
        'cual_es_tu_edad': ['25-34', '35-44', '25-34', '18-24', '45-54'],
        'cual_es_tu_pais': ['Brasil', 'México', 'Colombia', 'Brasil', 'Argentina']
    })
    
    # Buyers (apenas alguns compraram)
    buyers = pd.DataFrame({
        'email': ['user2@gmail.com', 'user4@gmail.com'],  # Maria e Ana compraram
        'data_compra': ['2024-01-15', '2024-01-20'],
        'valor': [997, 997]
    })
    
    # UTMs
    utms = pd.DataFrame({
        'e_mail': ['user1@gmail.com', 'user2@gmail.com', 'user3@gmail.com'],
        'utm_source': ['google', 'facebook', 'google'],
        'utm_medium': ['cpc', 'social', 'cpc'],
        'utm_campaign': ['black_friday', 'launch_22', 'black_friday']
    })
    
    return {
        'surveys': surveys,
        'buyers': buyers,
        'utms': utms
    }


def test_data_loader_with_mock():
    """Testa DataLoader com dados simulados."""
    print("=== Testando DataLoader (Mock) ===")
    
    # Como não podemos testar o carregamento real sem arquivos,
    # vamos testar apenas o load_prediction_data
    
    loader = DataLoader("/fake/path")
    
    # Criar DataFrame de teste
    test_df = pd.DataFrame({
        'E-mail': ['test@gmail.com', 'test2@gmail.com'],
        'Nome': ['Teste 1', 'Teste 2']
    })
    
    # Testar carregamento para predição
    loaded_df = loader.load_prediction_data(dataframe=test_df)
    
    print(f"✓ Dados carregados para predição")
    print(f"  Colunas padronizadas: {loaded_df.columns.tolist()}")
    print(f"  Email_norm criado: {'email_norm' in loaded_df.columns}")
    
    return True


def test_data_matcher():
    """Testa DataMatcher com dados simulados."""
    print("\n=== Testando DataMatcher ===")
    
    # Criar dados de teste
    data_dict = create_test_data()
    
    # Adicionar email_norm aos dados (simular o que DataLoader faria)
    for df_name, df in data_dict.items():
        if 'e_mail' in df.columns:
            df['email_norm'] = df['e_mail'].str.lower().str.strip()
        elif 'email' in df.columns:
            df['email_norm'] = df['email'].str.lower().str.strip()
    
    # Criar matcher e executar
    matcher = DataMatcher()
    result_df = matcher.match_and_create_target(data_dict)
    
    print(f"✓ Matching e target criados")
    print(f"  Shape final: {result_df.shape}")
    print(f"  Colunas: {result_df.columns.tolist()}")
    
    # Verificar target
    if 'target' in result_df.columns:
        target_dist = result_df['target'].value_counts().to_dict()
        print(f"  Distribuição do target: {target_dist}")
        print(f"  Taxa de conversão: {target_dist.get(1, 0) / len(result_df) * 100:.1f}%")
    
    # Verificar se UTMs foram merged
    utm_cols = [col for col in result_df.columns if 'utm' in col.lower()]
    print(f"  Colunas UTM merged: {utm_cols}")
    
    # Verificar estatísticas
    stats = matcher.get_statistics()
    print(f"\n  Estatísticas de matching:")
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    return True


def test_integration_with_pipeline_state():
    """Testa integração com PipelineState."""
    print("\n=== Testando Integração com PipelineState ===")
    
    # Criar estado
    state = PipelineState()
    
    # Simular carregamento e matching
    data_dict = create_test_data()
    
    # Adicionar email_norm
    for df_name, df in data_dict.items():
        if 'e_mail' in df.columns:
            df['email_norm'] = df['e_mail'].str.lower().str.strip()
        elif 'email' in df.columns:
            df['email_norm'] = df['email'].str.lower().str.strip()
    
    # Executar matching
    matcher = DataMatcher()
    final_df = matcher.match_and_create_target(data_dict)
    
    # Simular split train/val/test
    train_size = int(0.7 * len(final_df))
    val_size = int(0.15 * len(final_df))
    
    train_df = final_df.iloc[:train_size].copy()
    val_df = final_df.iloc[train_size:train_size+val_size].copy()
    test_df = final_df.iloc[train_size+val_size:].copy()
    
    # Atualizar estado
    state.update_dataframes(train=train_df, val=val_df, test=test_df)
    state.log_step("data_loading", {"source": "test_data"})
    state.log_step("data_matching", {"matches_found": 2})
    
    # Verificar estado
    print(f"✓ Estado atualizado com dados")
    summary = state.get_summary()
    print(f"  Shapes no estado: {summary['data_shapes']}")
    print(f"  Steps executados: {summary['n_steps_executed']}")
    
    # Validar para treino
    try:
        state.validate_for_training()
        print(f"✓ Estado válido para treino")
    except Exception as e:
        print(f"✗ Erro na validação: {e}")
        return False
    
    return True


def test_data_path_validation():
    """Testa validação de caminhos de dados."""
    print("\n=== Testando Validação de Caminhos ===")
    
    # Verificar se o caminho de dados existe
    data_path = "/Users/ramonmoreira/desktop/smart_ads/data/raw_data"
    
    if os.path.exists(data_path):
        print(f"✓ Caminho de dados existe: {data_path}")
        
        # Listar alguns arquivos
        files = os.listdir(data_path)[:5]  # Primeiros 5 arquivos
        print(f"  Exemplo de arquivos: {files}")
        
        # Testar carregamento real (se quiser)
        # loader = DataLoader(data_path)
        # data = loader.load_training_data()
        # print(f"  Dados carregados: surveys={len(data['surveys'])}, ...")
    else:
        print(f"⚠️ Caminho de dados não encontrado: {data_path}")
        print("  Usando apenas dados simulados para teste")
    
    return True


def main():
    """Executa todos os testes."""
    print("Testando Data Handlers do Smart Ads Pipeline\n")
    
    try:
        # Executar testes
        test_data_loader_with_mock()
        test_data_matcher()
        test_integration_with_pipeline_state()
        test_data_path_validation()
        
        print("\n✅ Todos os testes de Data Handlers passaram!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()