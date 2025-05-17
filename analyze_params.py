#!/usr/bin/env python
import joblib
import json
import sys

def analyze_params(filepath):
    """
    Analisa e imprime informações sobre o arquivo de parâmetros
    """
    try:
        # Carregar o arquivo de parâmetros
        params = joblib.load(filepath)
        
        # Imprimir estrutura básica
        print(f"\n=== Análise do arquivo de parâmetros: {filepath} ===\n")
        print(f"Tipo do objeto: {type(params)}")
        
        if isinstance(params, dict):
            print(f"Chaves de primeiro nível: {list(params.keys())}")
            
            # Imprimir detalhes para cada chave
            for key, value in params.items():
                print(f"\n--- Detalhe para chave '{key}' ---")
                
                if isinstance(value, dict):
                    print(f"Tipo: dict com {len(value)} itens")
                    print(f"Subchaves: {list(value.keys())[:10]}{'...' if len(value) > 10 else ''}")
                elif isinstance(value, list):
                    print(f"Tipo: list com {len(value)} itens")
                    print(f"Primeiros 5 itens: {value[:5]}{'...' if len(value) > 5 else ''}")
                else:
                    print(f"Tipo: {type(value)}")
                    
                    # Tentar imprimir alguns dados de exemplo
                    try:
                        print(f"Amostra: {str(value)[:100]}...")
                    except:
                        print("Não foi possível imprimir amostra")
        
        # Verificar se contém vetorizadores TF-IDF
        tfidf_keys = [k for k in params.keys() if 'tfidf' in k.lower()]
        if tfidf_keys:
            print("\n=== Análise de Vetorizadores TF-IDF ===\n")
            for key in tfidf_keys:
                print(f"Chave: {key}")
                if isinstance(params[key], dict):
                    for subkey, subvalue in params[key].items():
                        print(f"  Subchave: {subkey}")
                        if hasattr(subvalue, 'vocabulary_'):
                            print(f"  Tamanho do vocabulário: {len(subvalue.vocabulary_)}")
                            print(f"  Primeiros 5 termos: {list(subvalue.vocabulary_.keys())[:5]}")
        
        # Verificar se contém parâmetros de motivação profissional
        motivation_keys = [k for k in params.keys() if any(term in k.lower() for term in ['work', 'career', 'motivation', 'professional'])]
        if motivation_keys:
            print("\n=== Análise de Parâmetros de Motivação Profissional ===\n")
            for key in motivation_keys:
                print(f"Chave: {key}")
                if isinstance(params[key], dict):
                    print(f"  Tamanho: {len(params[key])}")
                    print(f"  Primeiros 5 itens: {list(params[key].items())[:5]}")
        
    except Exception as e:
        print(f"ERRO ao analisar arquivo: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # Usar o arquivo especificado como argumento ou o padrão
    filepath = sys.argv[1] if len(sys.argv) > 1 else "/Users/ramonmoreira/desktop/smart_ads/inference/params/04_script03_params.joblib"
    analyze_params(filepath)