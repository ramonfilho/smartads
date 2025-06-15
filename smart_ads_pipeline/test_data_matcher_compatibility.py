# test_data_matcher_production_compatibility.py
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, '/Users/ramonmoreira/desktop/smart_ads')

from smart_ads_pipeline.data_handlers import DataLoader, DataMatcher
from smart_ads_pipeline.core import ExtendedParameterManager

def test_production_filter_with_real_data():
    """Testa o filtro de compatibilidade usando dados reais do projeto"""
    
    print("=== Teste de Filtro de Compatibilidade com Dados Reais ===\n")
    
    # Configurar caminhos
    data_path = '/Users/ramonmoreira/desktop/smart_ads/data/raw_data'
    
    if not os.path.exists(data_path):
        print(f"❌ Erro: Diretório de dados não encontrado: {data_path}")
        return
    
    print(f"📁 Usando dados de: {data_path}")
    
    # Passo 1: Carregar dados usando DataLoader
    print("\n1. Carregando dados com DataLoader...")
    data_loader = DataLoader(data_path)
    
    try:
        data_dict = data_loader.load_training_data()
        
        print(f"   ✓ Surveys carregados: {data_dict['surveys'].shape}")
        print(f"   ✓ Buyers carregados: {data_dict['buyers'].shape}")
        print(f"   ✓ UTMs carregados: {data_dict['utms'].shape}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return
    
    # Passo 2: Aplicar DataMatcher (matching + target + merge)
    print("\n2. Aplicando DataMatcher...")
    data_matcher = DataMatcher()
    
    # Guardar shape antes do processamento
    original_surveys = data_dict['surveys'].shape[1]
    
    try:
        # Executar matching completo
        final_df = data_matcher.match_and_create_target(data_dict)
        
        print(f"   ✓ Dataset após matching: {final_df.shape}")
        print(f"   ✓ Target distribution: {final_df['target'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Erro no matching: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Passo 3: Análise de colunas removidas
    print("\n3. Análise de Compatibilidade com Produção:")
    
    # Listar todas as colunas do dataset final
    final_columns = set(final_df.columns)
    
    # Colunas que esperamos que sejam removidas (baseado no unified_pipeline)
    expected_removed = {
        'campanhas', 'class', 'email_norm', 'lancamento', 'lancamento_utm',
        'prediction', 'qualidade', 'teste', 'unnamed_10', 'unnamed_11',
        'unnamed_12', 'unnamed_8', 'unnamed_9', 'utm_campaign2',
        'utm_campaign_correcta', 'utm_campaign_vini'
    }
    
    # Verificar quais das colunas problemáticas ainda estão presentes
    problematic_still_present = expected_removed & final_columns
    
    if problematic_still_present:
        print(f"   ⚠️  Colunas problemáticas ainda presentes: {problematic_still_present}")
    else:
        print(f"   ✓ Todas as colunas problemáticas foram removidas!")
    
    # Verificar colunas de INFERENCE_COLUMNS
    print(f"\n4. Verificação de INFERENCE_COLUMNS:")
    
    missing_inference_cols = []
    present_inference_cols = []
    
    for col in data_matcher.INFERENCE_COLUMNS:
        if col in final_df.columns:
            present_inference_cols.append(col)
        else:
            missing_inference_cols.append(col)
    
    print(f"   ✓ Colunas de inferência presentes: {len(present_inference_cols)}/{len(data_matcher.INFERENCE_COLUMNS)}")
    
    if missing_inference_cols:
        print(f"   ℹ️  Colunas de inferência ausentes (serão NaN em produção):")
        for col in missing_inference_cols[:5]:
            print(f"      - {col}")
        if len(missing_inference_cols) > 5:
            print(f"      ... e mais {len(missing_inference_cols) - 5} colunas")
    
    # Passo 4: Validar compatibilidade
    print(f"\n5. Validação de Compatibilidade:")
    
    is_compatible, report = data_matcher.validate_production_compatibility(final_df)
    
    print(f"   Status: {'✅ COMPATÍVEL' if is_compatible else '❌ INCOMPATÍVEL'}")
    
    if report['errors']:
        print(f"\n   Erros encontrados:")
        for error in report['errors']:
            print(f"   ❌ {error}")
    
    if report['warnings']:
        print(f"\n   Avisos:")
        for warning in report['warnings'][:3]:
            print(f"   ⚠️  {warning}")
        if len(report['warnings']) > 3:
            print(f"   ... e mais {len(report['warnings']) - 3} avisos")
    
    # Passo 5: Estatísticas finais
    print(f"\n6. Estatísticas Finais:")
    print(f"   - Shape original (surveys): {data_dict['surveys'].shape}")
    print(f"   - Shape final: {final_df.shape}")
    print(f"   - Colunas removidas: {original_surveys - final_df.shape[1] + len(missing_inference_cols)}")
    print(f"   - Taxa de positivos: {(final_df['target'] == 1).sum() / len(final_df) * 100:.2f}%")
    
    # Verificar colunas críticas
    print(f"\n7. Verificação de Colunas Críticas:")
    critical_cols = ['e_mail', 'cual_es_tu_e_mail', 'data', 'marca_temporal']
    
    for col in critical_cols:
        if col in final_df.columns:
            coverage = final_df[col].notna().sum() / len(final_df) * 100
            print(f"   ✓ {col}: {coverage:.1f}% cobertura")
        else:
            print(f"   ❌ {col}: não encontrada")
    
    # Passo 6: Comparar com comportamento esperado do unified_pipeline
    print(f"\n8. Comparação com unified_pipeline.py:")
    
    # No unified_pipeline, após prepare_final_dataset, temos 31 colunas
    expected_columns = 31  # Baseado no log do unified_pipeline
    
    if final_df.shape[1] == expected_columns:
        print(f"   ✅ Número de colunas correto: {final_df.shape[1]}")
    else:
        print(f"   ⚠️  Número de colunas diferente: {final_df.shape[1]} (esperado: {expected_columns})")
    
    # Salvar amostra para inspeção
    output_path = '/tmp/smart_ads_test_output.csv'
    final_df.head(100).to_csv(output_path, index=False)
    print(f"\n📄 Amostra dos dados salvos em: {output_path}")
    
    return final_df

def test_with_sample_data():
    """Testa com uma amostra menor dos dados para debugging"""
    
    print("\n=== Teste com Amostra Reduzida ===\n")
    
    # Configurar para usar apenas uma amostra
    data_path = '/Users/ramonmoreira/desktop/smart_ads/data/raw_data'
    
    # Carregar dados
    data_loader = DataLoader(data_path)
    data_dict = data_loader.load_training_data()
    
    # Usar apenas primeiras 1000 linhas de surveys
    print(f"Reduzindo surveys de {len(data_dict['surveys'])} para 1000 linhas...")
    data_dict['surveys'] = data_dict['surveys'].head(1000)
    
    # Aplicar DataMatcher
    data_matcher = DataMatcher()
    
    print("\nColunas antes do processamento:")
    print(f"  Total: {len(data_dict['surveys'].columns)}")
    print(f"  Exemplos: {list(data_dict['surveys'].columns)[:10]}...")
    
    # Processar
    final_df = data_matcher.match_and_create_target(data_dict)
    
    print("\nColunas após processamento:")
    print(f"  Total: {len(final_df.columns)}")
    print(f"  Exemplos: {list(final_df.columns)[:10]}...")
    
    # Mostrar colunas removidas
    if hasattr(data_matcher, 'match_stats') and 'removed_columns' in data_matcher.match_stats:
        removed = data_matcher.match_stats['removed_columns']
        print(f"\nColunas removidas ({len(removed)}):")
        for col in sorted(removed)[:20]:
            print(f"  - {col}")
        if len(removed) > 20:
            print(f"  ... e mais {len(removed) - 20} colunas")

if __name__ == "__main__":
    # Teste principal com dados completos
    print("🚀 Iniciando teste com dados reais...\n")
    
    try:
        df_result = test_production_filter_with_real_data()
        
        # Teste adicional com amostra se desejar debugging
        # test_with_sample_data()
        
        print("\n✅ Todos os testes concluídos!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()