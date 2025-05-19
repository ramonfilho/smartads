#!/usr/bin/env python
"""
Script para verificar os parâmetros necessários para a pipeline de inferência.
Verifica se todos os arquivos de parâmetros existem, se estão no formato correto,
e se contêm as chaves necessárias para a execução da pipeline.
"""

import os
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Caminhos absolutos para os diretórios de parâmetros
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
PARAMS_PATHS = {
    'stage_02': os.path.join(PROJECT_ROOT, "src/preprocessing/02_params/02_params.joblib"),
    'stage_03_tfidf': os.path.join(PROJECT_ROOT, "src/preprocessing/03_params/03_tfidf_vectorizers.joblib"),
    'stage_03_lda': os.path.join(PROJECT_ROOT, "src/preprocessing/03_params/03_lda_models.joblib"),
    'stage_04': os.path.join(PROJECT_ROOT, "src/preprocessing/04_params/04_params.joblib")
}

# Definir requisitos esperados para cada arquivo de parâmetros
PARAM_REQUIREMENTS = {
    'stage_02': {
        'expected_type': dict,
        'expected_keys': ['quality_columns', 'missing_values', 'outliers', 'normalization', 
                         'feature_engineering', 'text_processing', 'advanced_features']
    },
    'stage_03_tfidf': {
        'expected_type': dict,
        'expected_keys': []  # Verificaremos se há pelo menos uma chave
    },
    'stage_03_lda': {
        'expected_type': dict,
        'expected_keys': []  # Verificaremos se há pelo menos uma chave
    },
    'stage_04': {
        'expected_type': dict,
        'expected_keys': ['professional_motivation', 'vectorizers', 'aspiration_sentiment', 
                         'commitment', 'career']
    }
}

def check_param_file(param_name, path, requirements):
    """
    Verifica se um arquivo de parâmetros existe, pode ser carregado,
    e se contém as chaves esperadas.
    
    Args:
        param_name: Nome do parâmetro para exibição
        path: Caminho para o arquivo
        requirements: Requisitos esperados (tipo e chaves)
        
    Returns:
        Tuple (existe, válido, mensagem)
    """
    print(f"Verificando {param_name}... ", end="")
    
    # Verificar existência
    if not os.path.exists(path):
        print(f"ERRO: Arquivo não encontrado em {path}")
        return False, False, f"Arquivo não encontrado: {path}"
    
    # Verificar se pode ser carregado
    try:
        params = joblib.load(path)
        print(f"Carregado com sucesso, tipo: {type(params).__name__}")
    except Exception as e:
        print(f"ERRO: Falha ao carregar - {str(e)}")
        return True, False, f"Falha ao carregar: {str(e)}"
    
    # Verificar tipo
    expected_type = requirements['expected_type']
    if not isinstance(params, expected_type):
        print(f"AVISO: Tipo incorreto, esperado {expected_type.__name__}, encontrado {type(params).__name__}")
        return True, False, f"Tipo incorreto: esperado {expected_type.__name__}, encontrado {type(params).__name__}"
    
    # Verificar chaves
    if isinstance(params, dict):
        expected_keys = requirements['expected_keys']
        
        # Se não temos chaves esperadas, apenas verificar se há alguma chave
        if not expected_keys and len(params) == 0:
            print(f"AVISO: Dicionário vazio, esperava pelo menos uma chave")
            return True, False, "Dicionário vazio"
        
        # Verificar chaves específicas
        missing_keys = [key for key in expected_keys if key not in params]
        if missing_keys:
            print(f"AVISO: {len(missing_keys)} chaves obrigatórias ausentes: {missing_keys}")
            return True, False, f"Chaves ausentes: {missing_keys}"
        
        # Para TF-IDF e LDA, verificar se há pelo menos uma chave de modelo
        if param_name in ['stage_03_tfidf', 'stage_03_lda'] and len(params) == 0:
            print(f"AVISO: Nenhum modelo encontrado em {param_name}")
            return True, False, "Nenhum modelo encontrado"
            
        # Informações adicionais para tfidf
        if param_name == 'stage_03_tfidf':
            print(f"  Encontrados {len(params)} vetorizadores TF-IDF")
            for i, (col_key, vectorizer) in enumerate(list(params.items())[:3]):
                try:
                    n_features = len(vectorizer.get_feature_names_out())
                    print(f"  - Vectorizer {i+1}: '{col_key}' com {n_features} features")
                except Exception as e:
                    print(f"  - Vectorizer {i+1}: '{col_key}' (falha ao obter features: {e})")
            if len(params) > 3:
                print(f"  ... e mais {len(params) - 3} vetorizadores")
        
        # Informações adicionais para lda
        if param_name == 'stage_03_lda':
            print(f"  Encontrados {len(params)} modelos LDA")
            for i, (col_key, model_info) in enumerate(list(params.items())[:3]):
                try:
                    lda_model = model_info.get('lda_model')
                    vectorizer = model_info.get('vectorizer')
                    n_topics = lda_model.n_components if hasattr(lda_model, 'n_components') else 'desconhecido'
                    has_vectorizer = "sim" if vectorizer else "não"
                    print(f"  - Modelo LDA {i+1}: '{col_key}' com {n_topics} tópicos, vectorizer: {has_vectorizer}")
                except Exception as e:
                    print(f"  - Modelo LDA {i+1}: '{col_key}' (falha ao obter informações: {e})")
            if len(params) > 3:
                print(f"  ... e mais {len(params) - 3} modelos")
    
    return True, True, "OK"

def main():
    """Função principal para verificar todos os parâmetros."""
    print("=" * 80)
    print(f"VERIFICAÇÃO DE PARÂMETROS PARA PIPELINE DE INFERÊNCIA")
    print(f"Data e hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Resultados
    all_valid = True
    results = {}
    
    # Verificar cada arquivo de parâmetros
    for param_name, path in PARAMS_PATHS.items():
        requirements = PARAM_REQUIREMENTS[param_name]
        exists, valid, message = check_param_file(param_name, path, requirements)
        
        # Registrar resultados
        results[param_name] = {
            'exists': exists,
            'valid': valid,
            'message': message,
            'path': path
        }
        
        if not exists or not valid:
            all_valid = False
        
        print("-" * 80)
    
    # Resumo final
    print("\nRESUMO DA VERIFICAÇÃO:")
    for param_name, result in results.items():
        status = "✓ OK" if result['exists'] and result['valid'] else "✗ FALHA"
        print(f"{param_name}: {status} - {result['message']}")
    
    # Verificar se todos os parâmetros estão disponíveis
    if all_valid:
        print("\n✅ TODOS OS PARÂMETROS ESTÃO DISPONÍVEIS E VÁLIDOS!")
        print("\nCaminhos a serem usados pela pipeline:")
        for param_name, path in PARAMS_PATHS.items():
            print(f"- {param_name}: {path}")
    else:
        print("\n❌ ATENÇÃO: ALGUNS PARÂMETROS ESTÃO AUSENTES OU INVÁLIDOS!")
        print("Resolva os problemas acima antes de executar a pipeline de inferência.")
    
    return all_valid

if __name__ == "__main__":
    main()