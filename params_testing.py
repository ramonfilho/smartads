#!/usr/bin/env python
"""
Script para examinar os arquivos de parâmetros do script 3 e identificar
quais vetorizadores TF-IDF estão faltando para o processamento de texto.
"""

import os
import sys
import joblib
import pandas as pd
import json
from pprint import pprint

# Configuração de caminhos absolutos
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
PARAMS_DIR = os.path.join(PROJECT_ROOT, "src/preprocessing/preprocessing_params")

# Adicionar diretório do projeto ao path para poder importar os outros scripts
sys.path.append(PROJECT_ROOT)

def analyze_params_file(file_path):
    """
    Analisa um arquivo de parâmetros e extrai informações relevantes.
    
    Args:
        file_path: Caminho para o arquivo de parâmetros
        
    Returns:
        Dict com informações analisadas
    """
    if not os.path.exists(file_path):
        return {"error": f"Arquivo não encontrado: {file_path}"}
    
    try:
        params = joblib.load(file_path)
        
        # Informações básicas
        info = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path) / (1024 * 1024),  # tamanho em MB
            "top_level_keys": list(params.keys())
        }
        
        # Verificar se há vetorizadores
        if 'vectorizers' in params:
            vectorizers = params['vectorizers']
            info["vectorizers"] = {
                "total_count": len(vectorizers),
                "column_names": list(vectorizers.keys())
            }
            
            # Verificar vetorizadores específicos que estão faltando na saída
            missing_columns = [
                "cuando_hables_inglés_con_fluid",
                "qué_esperas_aprender_en_la",
                "déjame_un_mensaje"
            ]
            
            info["missing_vectorizers"] = {}
            for col in missing_columns:
                # Verificar se o nome exato existe
                exists_exact = col in vectorizers
                
                # Verificar nomes similares
                similar_names = [name for name in vectorizers.keys() if col in name]
                
                info["missing_vectorizers"][col] = {
                    "exists_exact": exists_exact,
                    "similar_names": similar_names
                }
        
        # Verificar outros parâmetros relacionados ao processamento de texto
        text_related_keys = ['professional_motivation', 'aspiration_sentiment', 
                            'commitment', 'career', 'text_processing']
        
        info["text_params"] = {}
        for key in text_related_keys:
            if key in params:
                info["text_params"][key] = {
                    "exists": True,
                    "keys": list(params[key].keys()) if isinstance(params[key], dict) else "not a dict"
                }
            else:
                info["text_params"][key] = {"exists": False}
        
        return info
    
    except Exception as e:
        return {"error": f"Erro ao analisar o arquivo: {str(e)}"}

def main():
    """Função principal"""
    print("=" * 60)
    print("ANÁLISE DE PARÂMETROS DO SCRIPT 3")
    print("=" * 60)
    
    # Listar todos os arquivos de parâmetros no diretório
    print(f"\nArquivos no diretório {PARAMS_DIR}:")
    files = os.listdir(PARAMS_DIR)
    param_files = [f for f in files if f.endswith('.joblib')]
    
    for idx, file in enumerate(param_files):
        print(f"  {idx+1}. {file} - {os.path.getsize(os.path.join(PARAMS_DIR, file)) / (1024 * 1024):.2f} MB")
    
    # Analisar cada arquivo de parâmetros
    print("\n" + "=" * 60)
    print("ANÁLISE DETALHADA DOS ARQUIVOS DE PARÂMETROS")
    
    for file in param_files:
        file_path = os.path.join(PARAMS_DIR, file)
        print("\n" + "-" * 60)
        print(f"Analisando: {file}")
        
        info = analyze_params_file(file_path)
        
        if "error" in info:
            print(f"ERRO: {info['error']}")
            continue
        
        print(f"Tamanho do arquivo: {info['file_size']:.2f} MB")
        print("Chaves de nível superior:")
        for key in info['top_level_keys']:
            print(f"  - {key}")
        
        # Mostrar informações sobre vetorizadores
        if "vectorizers" in info:
            vec_info = info["vectorizers"]
            print(f"\nVetorizadores: {vec_info['total_count']} encontrados")
            
            # Se houver muitos, mostrar apenas alguns
            column_names = vec_info['column_names']
            if len(column_names) > 10:
                print(f"Primeiros 5 nomes de colunas: {column_names[:5]}")
                print(f"Últimos 5 nomes de colunas: {column_names[-5:]}")
            else:
                print(f"Nomes de colunas: {column_names}")
            
            # Verificar vetorizadores faltantes
            print("\nVerificação de vetorizadores específicos:")
            for col, status in info["missing_vectorizers"].items():
                if status["exists_exact"]:
                    print(f"  ✓ '{col}' existe exatamente")
                else:
                    print(f"  ✗ '{col}' não existe exatamente")
                
                if status["similar_names"]:
                    print(f"    Nomes similares encontrados: {status['similar_names']}")
        
        # Mostrar informações sobre parâmetros de texto
        print("\nParâmetros relacionados ao processamento de texto:")
        for key, status in info["text_params"].items():
            if status["exists"]:
                print(f"  ✓ '{key}' existe")
                if "keys" in status and isinstance(status["keys"], list):
                    if len(status["keys"]) > 10:
                        print(f"    Primeiras 5 sub-chaves: {status['keys'][:5]}")
                    else:
                        print(f"    Sub-chaves: {status['keys']}")
            else:
                print(f"  ✗ '{key}' não existe")
    
    # Verificar especificamente o dicionário de vetorizadores no script03_params.joblib
    print("\n" + "=" * 60)
    print("ANÁLISE ESPECÍFICA DOS VETORIZADORES NO SCRIPT03_PARAMS.JOBLIB")
    
    script03_path = os.path.join(PARAMS_DIR, "script03_params.joblib")
    if os.path.exists(script03_path):
        try:
            params = joblib.load(script03_path)
            
            if 'vectorizers' in params:
                vectorizers = params['vectorizers']
                print(f"Total de vetorizadores: {len(vectorizers)}")
                
                # Verificar os nomes exatos de colunas
                column_names = list(vectorizers.keys())
                print("\nNomes de colunas no parâmetro 'vectorizers':")
                for idx, col in enumerate(column_names):
                    print(f"  {idx+1}. '{col}'")
                
                # Verificar colunas específicas que aparecem nos avisos
                problem_columns = [
                    "cuando_hables_inglés_con_fluid",
                    "qué_esperas_aprender_en_la",
                    "déjame_un_mensaje"
                ]
                
                print("\nBuscando colunas problemáticas:")
                for col in problem_columns:
                    # Buscar correspondências exatas
                    if col in vectorizers:
                        print(f"  ✓ '{col}' encontrada exatamente")
                    else:
                        print(f"  ✗ '{col}' não encontrada exatamente")
                        
                        # Buscar correspondências parciais
                        partial_matches = [name for name in column_names if col in name]
                        if partial_matches:
                            print(f"    Correspondências parciais: {partial_matches}")
                        
                        # Buscar correspondências por similaridade (primeiros caracteres)
                        similar_matches = [name for name in column_names 
                                        if name.startswith(col[:15]) or col.startswith(name[:15])]
                        if similar_matches and similar_matches != partial_matches:
                            print(f"    Correspondências por similaridade: {similar_matches}")
                
                # Criar mapeamento de nomes reais para os nomes problemáticos
                print("\nMapeamento sugerido:")
                mapping = {}
                for problem_col in problem_columns:
                    best_match = None
                    max_common_prefix = 0
                    
                    for col_name in column_names:
                        # Encontrar o maior prefixo comum
                        i = 0
                        min_len = min(len(problem_col), len(col_name))
                        while i < min_len and problem_col[i].lower() == col_name[i].lower():
                            i += 1
                        
                        if i > max_common_prefix:
                            max_common_prefix = i
                            best_match = col_name
                    
                    mapping[problem_col] = best_match
                    
                    if best_match:
                        print(f"  '{problem_col}' → '{best_match}' ({max_common_prefix} caracteres em comum)")
                    else:
                        print(f"  '{problem_col}' → Nenhuma correspondência encontrada")
                        
                # Extrair mais informações sobre as estruturas dos vetorizadores
                if len(vectorizers) > 0:
                    sample_key = list(vectorizers.keys())[0]
                    sample_vectorizer = vectorizers[sample_key]
                    
                    print("\nEstrutura de um exemplo de vetorizador:")
                    print(f"  Tipo de objeto: {type(sample_vectorizer)}")
                    
                    if isinstance(sample_vectorizer, dict):
                        print(f"  Chaves: {list(sample_vectorizer.keys())}")
                        
                        # Verificar se há feature_names dentro
                        if 'feature_names' in sample_vectorizer:
                            feature_names = sample_vectorizer['feature_names']
                            print(f"  Quantidade de feature_names: {len(feature_names) if isinstance(feature_names, list) else 'N/A'}")
                            if isinstance(feature_names, list) and len(feature_names) > 0:
                                print(f"  Primeiros feature_names: {feature_names[:5]}")
            else:
                print("Não há um dicionário 'vectorizers' no arquivo script03_params.joblib")
            
            # Verificar o dicionário 'professional_motivation' que também pode conter informações relacionadas
            if 'professional_motivation' in params:
                print("\nAnálise do dicionário 'professional_motivation':")
                prof_motivation = params['professional_motivation']
                print(f"  Chaves: {list(prof_motivation.keys())}")
                
                # Verificar work_keywords se existir
                if 'work_keywords' in prof_motivation:
                    work_keywords = prof_motivation['work_keywords']
                    print(f"  work_keywords: {len(work_keywords)} entradas")
                    print(f"  Primeiras 5 entradas: {list(work_keywords.items())[:5]}")
            
        except Exception as e:
            print(f"ERRO ao analisar script03_params.joblib: {str(e)}")
    else:
        print(f"Arquivo script03_params.joblib não encontrado em: {PARAMS_DIR}")
    
    print("\n" + "=" * 60)
    print("RECOMENDAÇÕES:")
    print("1. Atualize o TextFeatureEngineeringTransformer para usar os nomes de colunas corretos dos vetorizadores")
    print("2. Verifique se o arquivo script03_params.joblib contém todos os vetorizadores necessários")
    print("3. Se os vetorizadores estão faltando, pode ser necessário re-executar o script 03")

if __name__ == "__main__":
    main()