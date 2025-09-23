"""
Teste de integração do pipeline completo.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import LeadScoringPipeline
import pandas as pd
import json


def test_model_configuration(model_name: str, com_utm: bool, versao: str, expected_features: int, features_file: str, test_file: str):
    """
    Testa uma configuração específica do pipeline com validação detalhada de features.

    Args:
        model_name: Nome do modelo para identificação
        com_utm: Se deve manter features UTM
        versao: "v1" ou "v2"
        expected_features: Número esperado de features
        features_file: Caminho para arquivo JSON com features esperadas
        test_file: Caminho para arquivo de teste
    """
    print(f"\n{'='*60}")
    print(f"🔬 TESTANDO: {model_name}")
    print(f"   Parâmetros: com_utm={com_utm}, versao={versao}")
    print(f"   Features esperadas: {expected_features}")
    print(f"{'='*60}")

    # Carregar features esperadas
    try:
        with open(features_file, 'r', encoding='utf-8') as f:
            expected_features_data = json.load(f)
            expected_feature_names = expected_features_data['feature_names']
    except Exception as e:
        print(f"❌ ERRO ao carregar features esperadas: {str(e)}")
        return False, None

    # Criar pipeline com configuração fixa (V1 com UTM)
    pipeline = LeadScoringPipeline()

    try:
        # Executar pipeline
        result = pipeline.run(test_file)

        # Obter features produzidas
        actual_feature_names = list(result.columns)
        actual_features = len(actual_feature_names)

        print(f"\n📊 RESULTADO:")
        print(f"   Features produzidas: {actual_features}")
        print(f"   Features esperadas: {expected_features}")

        # 1. Verificar número de features
        features_count_ok = actual_features == expected_features
        if features_count_ok:
            print(f"   ✅ Quantidade de features: CORRETO")
        else:
            print(f"   ❌ Quantidade de features: INCORRETO (diff: {actual_features - expected_features:+d})")

        # 2. Verificar nomes das features (ordem e conteúdo)
        features_names_ok = actual_feature_names == expected_feature_names
        if features_names_ok:
            print(f"   ✅ Nomes e ordem das features: CORRETO")
        else:
            print(f"   ❌ Nomes e ordem das features: INCORRETO")

            # Mostrar diferenças detalhadas
            missing_features = set(expected_feature_names) - set(actual_feature_names)
            extra_features = set(actual_feature_names) - set(expected_feature_names)

            # DEBUG: Confirmar que os conjuntos são idênticos
            conjuntos_identicos = missing_features == set() and extra_features == set()
            print(f"   🔍 CONJUNTOS DE FEATURES IDÊNTICOS: {'✅ SIM' if conjuntos_identicos else '❌ NÃO'}")

            if missing_features:
                print(f"   🔍 Features FALTANDO ({len(missing_features)}):")
                for feat in sorted(missing_features)[:10]:  # Mostrar primeiras 10
                    print(f"      - {feat}")
                if len(missing_features) > 10:
                    print(f"      ... e mais {len(missing_features)-10} features")

            if extra_features:
                print(f"   🔍 Features EXTRAS ({len(extra_features)}):")
                for feat in sorted(extra_features)[:10]:  # Mostrar primeiras 10
                    print(f"      + {feat}")
                if len(extra_features) > 10:
                    print(f"      ... e mais {len(extra_features)-10} features")

            # Mostrar comparação ordenada se mesmo conjunto mas ordem diferente
            if not missing_features and not extra_features:
                print(f"   🔍 MESMAS FEATURES, ORDEM DIFERENTE:")
                print(f"   Verificando todas as {len(actual_feature_names)} features:")

                # Encontrar primeira divergência
                primeira_divergencia = None
                for i in range(len(actual_feature_names)):
                    esperada = expected_feature_names[i] if i < len(expected_feature_names) else "N/A"
                    atual = actual_feature_names[i] if i < len(actual_feature_names) else "N/A"
                    if atual != esperada:
                        primeira_divergencia = i
                        break

                if primeira_divergencia is None:
                    print(f"      ✅ TODAS AS FEATURES ESTÃO EM ORDEM CORRETA!")
                else:
                    print(f"      ❌ Primeira divergência na posição {primeira_divergencia+1}")
                    print(f"      Comparação ao redor da divergência:")
                    start = max(0, primeira_divergencia - 2)
                    end = min(len(actual_feature_names), primeira_divergencia + 8)

                    for i in range(start, end):
                        esperada = expected_feature_names[i] if i < len(expected_feature_names) else "N/A"
                        atual = actual_feature_names[i] if i < len(actual_feature_names) else "N/A"
                        match = "✓" if atual == esperada else "✗"
                        destaque = " <<<" if i == primeira_divergencia else ""
                        print(f"      {i+1:2d}. {match} {atual[:50]:<52} | {esperada[:50]}{destaque}")

        # 3. Verificar tipos de dados das features
        print(f"\n🔍 TIPOS DE DADOS:")
        type_issues = []
        for i, feature in enumerate(actual_feature_names[:10]):  # Verificar primeiras 10
            dtype = str(result[feature].dtype)
            null_count = result[feature].isnull().sum()
            print(f"   {feature}: {dtype} (nulls: {null_count})")

            # Verificar se há problemas óbvios
            if dtype == 'object' and feature not in ['nome_comprimento', 'dia_semana']:
                # Features categóricas codificadas deveriam ser numéricas
                if any(keyword in feature for keyword in ['_N_o', '_Sim', '_Feminino', '_Masculino']):
                    type_issues.append(f"{feature} deveria ser numérico, mas é {dtype}")

        if len(actual_feature_names) > 10:
            print(f"   ... e mais {len(actual_feature_names)-10} features")

        if type_issues:
            print(f"   ⚠️  Possíveis problemas de tipo:")
            for issue in type_issues[:3]:
                print(f"      - {issue}")

        # Status final
        status = features_count_ok and features_names_ok
        if status:
            print(f"\n   🎉 SUCESSO COMPLETO: Features exatamente como esperadas!")
        else:
            print(f"\n   ❌ FALHA: Verifique as diferenças acima")

        return status, result

    except Exception as e:
        print(f"❌ ERRO durante execução: {str(e)}")
        return False, None


def test_pipeline():
    """Testa o pipeline completo com as 4 configurações de modelos."""

    print("=== TESTE DO PIPELINE COMPLETO ===\n")

    # Arquivo de teste conforme especificado no PROJECT_GUIDE.md
    test_file = '../Lead score LF 24.xlsx'

    # Verificar se arquivo existe
    if not os.path.exists(test_file):
        print(f"❌ Arquivo não encontrado: {test_file}")
        print("Certifique-se de que o arquivo de teste está no local correto.")
        return None

    # Listar arquivos de modelos disponíveis
    import glob
    model_files = glob.glob("../arquivos_modelo/features_ordenadas_*.json")

    if not model_files:
        print("❌ Nenhum arquivo de modelo encontrado em ../arquivos_modelo/")
        return None

    print(f"📁 Arquivos de modelo encontrados: {len(model_files)}")
    for i, file in enumerate(model_files, 1):
        filename = file.split('/')[-1]
        print(f"  {i}. {filename}")

    # Usar o primeiro arquivo encontrado
    features_file = model_files[0]
    print(f"\n🎯 Usando arquivo: {features_file.split('/')[-1]}")

    # Carregar features esperadas
    try:
        with open(features_file, 'r', encoding='utf-8') as f:
            expected_features_data = json.load(f)
            expected_feature_names = expected_features_data['feature_names']
            expected_features = len(expected_feature_names)
    except Exception as e:
        print(f"❌ ERRO ao carregar features esperadas: {str(e)}")
        return None

    # Configuração única - V1 com UTM
    test_configs = [
        {
            "model_name": "V1 DEVCLUB com UTM",
            "expected_features": expected_features,
            "features_file": features_file
        }
    ]

    # Executar todos os testes
    results = []
    result_dfs = []

    for config in test_configs:
        success, result_df = test_model_configuration(
            config["model_name"], True, "v1",
            config["expected_features"], config["features_file"], test_file
        )
        results.append((config["model_name"], success))
        if result_df is not None:
            result_dfs.append((config["model_name"], result_df))

    # Resumo final dos testes
    print(f"\n{'='*60}")
    print("📋 RESUMO DOS TESTES")
    print(f"{'='*60}")

    for model_name, success in results:
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {model_name}")

    total_success = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\n🎯 RESULTADO GERAL: {total_success}/{total_tests} testes aprovados")

    if total_success == total_tests:
        print("🎉 TODOS OS TESTES PASSARAM! Pipeline configurado corretamente.")
    else:
        print("⚠️  ALGUNS TESTES FALHARAM. Verifique a configuração do pipeline.")

    # Retornar o primeiro resultado para compatibilidade
    return result_dfs[0][1] if result_dfs else None


if __name__ == "__main__":
    df_result = test_pipeline()
    if df_result is not None:
        print(f"\n📊 Pipeline testado com sucesso!")
        print(f"DataFrame final: {len(df_result)} registros, {len(df_result.columns)} colunas")
    else:
        print("\n❌ Teste falhou - verifique o caminho do arquivo")
        sys.exit(1)