#!/usr/bin/env python
"""
Script para verificar os parâmetros necessários para a pipeline de inferência.
Verifica se todos os arquivos de parâmetros existem, se estão no formato correto,
e se contêm as chaves necessárias para a execução da pipeline.
"""

import os
import joblib
import warnings
import numpy as np
from datetime import datetime
import sys

# Adicionar o diretório raiz do projeto ao path para importar módulos do projeto
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.append(project_root)

# Importar a classe GMM_Wrapper para garantir que esteja disponível durante o carregamento
try:
    from src.modeling.gmm_wrapper import GMM_Wrapper
    print("Classe GMM_Wrapper importada com sucesso de src.modeling.gmm_wrapper")
except ImportError as e:
    print(f"AVISO: Não foi possível importar GMM_Wrapper: {e}")
    # Não definiremos uma versão básica aqui, pois queremos verificar realmente se a
    # classe correta está disponível e funcional

warnings.filterwarnings('ignore')

# Caminhos absolutos para os diretórios de parâmetros
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
PARAMS_PATHS = {
    'stage_02': os.path.join(PROJECT_ROOT, "src/preprocessing/02_params/02_params.joblib"),
    'stage_03_tfidf': os.path.join(PROJECT_ROOT, "src/preprocessing/03_params/03_tfidf_vectorizers.joblib"),
    'stage_03_lda': os.path.join(PROJECT_ROOT, "src/preprocessing/03_params/03_lda_models.joblib"),
    'stage_04': os.path.join(PROJECT_ROOT, "src/preprocessing/04_params/04_params.joblib")
}

# Caminhos para os modelos GMM
GMM_PATHS = {
    'original': {
        'pca_model': os.path.join(PROJECT_ROOT, "models/artifacts/gmm_optimized/pca_model.joblib"),
        'scaler_model': os.path.join(PROJECT_ROOT, "models/artifacts/gmm_optimized/scaler_model.joblib"),
        'gmm_model': os.path.join(PROJECT_ROOT, "models/artifacts/gmm_optimized/gmm_model.joblib"),
        'cluster_models': [
            os.path.join(PROJECT_ROOT, "models/artifacts/gmm_optimized/cluster_0_model.joblib"),
            os.path.join(PROJECT_ROOT, "models/artifacts/gmm_optimized/cluster_1_model.joblib"),
            os.path.join(PROJECT_ROOT, "models/artifacts/gmm_optimized/cluster_2_model.joblib")
        ]
    },
    'calibrated': {
        'gmm_calibrated': os.path.join(PROJECT_ROOT, "models/calibrated/gmm_calibrated_20250518_152543/gmm_calibrated.joblib"),
        'threshold': os.path.join(PROJECT_ROOT, "models/calibrated/gmm_calibrated_20250518_152543/threshold.txt")
    }
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

def check_param_file(param_name, path, requirements=None):
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
        # Para arquivos de texto (threshold)
        if path.endswith('.txt'):
            with open(path, 'r') as f:
                content = f.read().strip()
                # Verificar se é um número
                try:
                    value = float(content)
                    print(f"Carregado com sucesso, valor: {value}")
                    return True, True, f"Valor: {value}"
                except:
                    print(f"ERRO: Conteúdo não é um número - {content}")
                    return True, False, f"Conteúdo inválido: {content}"
        else:
            # Para arquivos joblib
            params = joblib.load(path)
            print(f"Carregado com sucesso, tipo: {type(params).__name__}")
    except Exception as e:
        print(f"ERRO: Falha ao carregar - {str(e)}")
        return True, False, f"Falha ao carregar: {str(e)}"
    
    # Se não temos requisitos específicos, consideramos válido
    if requirements is None:
        return True, True, "Carregado com sucesso"
    
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

def check_gmm_models():
    """
    Verifica os modelos GMM e seus componentes.
    
    Returns:
        Tuple (original_valid, calibrated_valid)
    """
    print("\n" + "=" * 80)
    print("VERIFICAÇÃO DOS MODELOS GMM")
    print("=" * 80)
    
    # Verificar GMM original
    print("\nVerificando GMM original (componentes individuais do modelo):")
    original_valid = True
    
    # Componentes críticos para inferência
    critical_components = ['pca_model', 'scaler_model', 'gmm_model']
    cluster_valid = True
    
    for component, path in GMM_PATHS['original'].items():
        if component == 'cluster_models':
            # Verificar modelos de cluster
            for i, cluster_path in enumerate(path):
                exists, valid, message = check_param_file(f"cluster_{i}_model", cluster_path)
                if not exists or not valid:
                    cluster_valid = False
        else:
            # Verificar outros componentes
            exists, valid, message = check_param_file(component, path)
            if not exists or not valid and component in critical_components:
                original_valid = False
    
    if not cluster_valid:
        print("\nAVISO: Alguns modelos de cluster do GMM original estão ausentes ou inválidos.")
        print("Isso pode afetar a qualidade das previsões, mas não impede a execução.")
        # Não falhar completamente se apenas alguns clusters estiverem inválidos
    
    # Verificar GMM calibrado (este é o que realmente usamos para inferência)
    print("\nVerificando GMM calibrado (modelo para inferência):")
    calibrated_valid = True
    
    for component, path in GMM_PATHS['calibrated'].items():
        exists, valid, message = check_param_file(component, path)
        if not exists or not valid:
            calibrated_valid = False
    
    # Verificar se o modelo GMM calibrado é utilizável
    if calibrated_valid:
        try:
            model_path = GMM_PATHS['calibrated']['gmm_calibrated']
            print(f"Tentando carregar o modelo calibrado para verificação... ", end="")
            model = joblib.load(model_path)
            
            # Verificar se tem os métodos necessários
            has_predict = hasattr(model, 'predict')
            has_predict_proba = hasattr(model, 'predict_proba')
            
            if has_predict and has_predict_proba:
                print(f"OK (tipo: {type(model).__name__})")
                
                # Tentar verificar se tem a estrutura esperada de um CalibratedClassifierCV
                if hasattr(model, 'calibrated_classifiers_'):
                    print(f"  Estrutura: CalibratedClassifierCV com {len(model.calibrated_classifiers_)} calibradores")
                else:
                    print(f"  Estrutura: modelo com API compatível, mas não é um CalibratedClassifierCV padrão")
            else:
                print(f"AVISO: Modelo não tem APIs necessárias (predict: {has_predict}, predict_proba: {has_predict_proba})")
                calibrated_valid = False
        except Exception as e:
            print(f"ERRO ao carregar o modelo calibrado: {e}")
            calibrated_valid = False
    
    # Resumo
    print("\nResumo dos modelos GMM:")
    print(f"- GMM original (componentes): {'✓ Disponível' if original_valid else '✗ Alguns componentes indisponíveis'}")
    print(f"- GMM calibrado (para inferência): {'✓ Disponível' if calibrated_valid else '✗ Indisponível ou inválido'}")
    
    return original_valid, calibrated_valid

def main():
    """Função principal para verificar todos os parâmetros."""
    print("=" * 80)
    print(f"VERIFICAÇÃO DE PARÂMETROS PARA PIPELINE DE INFERÊNCIA")
    print(f"Data e hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Resultados
    preprocessing_params_valid = True
    results = {}
    
    # Verificar cada arquivo de parâmetros de pré-processamento
    print("\nVERIFICANDO PARÂMETROS DE PRÉ-PROCESSAMENTO:")
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
            preprocessing_params_valid = False
        
        print("-" * 80)
    
    # Verificar modelos GMM
    original_valid, calibrated_valid = check_gmm_models()
    
    # Considerar válido se pelo menos um dos modelos GMM estiver disponível
    model_valid = calibrated_valid  # Priorizar o modelo calibrado para inferência
    
    # Resumo final
    print("\nRESUMO DA VERIFICAÇÃO DE PARÂMETROS DE PRÉ-PROCESSAMENTO:")
    for param_name, result in results.items():
        status = "✓ OK" if result['exists'] and result['valid'] else "✗ FALHA"
        print(f"{param_name}: {status} - {result['message']}")
    
    print("\nRESUMO DA VERIFICAÇÃO DE MODELOS GMM:")
    print(f"GMM original (componentes): {'✓ OK' if original_valid else '✗ FALHA'}")
    print(f"GMM calibrado (para inferência): {'✓ OK' if calibrated_valid else '✗ FALHA'}")
    
    # Verificar se os parâmetros essenciais estão disponíveis
    if preprocessing_params_valid and calibrated_valid:
        print("\n✅ TODOS OS PARÂMETROS NECESSÁRIOS ESTÃO DISPONÍVEIS E VÁLIDOS!")
        print("\nA pipeline de inferência pode ser executada com o modelo GMM calibrado.")
        
        print("\nCaminhos a serem usados pela pipeline:")
        for param_name, path in PARAMS_PATHS.items():
            print(f"- {param_name}: {path}")
        
        print("\nModelo GMM calibrado:")
        for component, path in GMM_PATHS['calibrated'].items():
            print(f"- {component}: {path}")
    else:
        print("\n❌ ATENÇÃO: ALGUNS PARÂMETROS ESTÃO AUSENTES OU INVÁLIDOS!")
        
        if not preprocessing_params_valid:
            print("- Problemas nos parâmetros de pré-processamento: verifique as mensagens acima.")
        
        if not calibrated_valid:
            print("- Problemas no modelo GMM calibrado: verifique as mensagens acima.")
        
        print("\nResolva os problemas acima antes de executar a pipeline de inferência.")
    
    return preprocessing_params_valid and calibrated_valid

if __name__ == "__main__":
    main()