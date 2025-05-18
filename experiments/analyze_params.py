#!/usr/bin/env python
"""
Script de diagnóstico para analisar parâmetros do script04
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Configurar caminhos
project_root = "/Users/ramonmoreira/desktop/smart_ads"
params_dir = os.path.join(project_root, "inference/params")
params_path = os.path.join(params_dir, "04_script03_params.joblib")
train_cols_path = os.path.join(params_dir, "04_train_columns.csv")
missing_cols_path = os.path.join(project_root, "reports/missing_columns.txt")
extra_cols_path = os.path.join(project_root, "reports/extra_columns.txt")

def load_missing_columns():
    """Carrega a lista de colunas faltantes do arquivo gerado na inferência"""
    if not os.path.exists(missing_cols_path):
        print(f"Arquivo de colunas faltantes não encontrado: {missing_cols_path}")
        return []
    
    with open(missing_cols_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def analyze_parameters(params_path):
    """Analisa a estrutura dos parâmetros salvos"""
    if not os.path.exists(params_path):
        print(f"Arquivo de parâmetros não encontrado: {params_path}")
        return None
    
    try:
        print(f"Carregando parâmetros de: {params_path}")
        params = joblib.load(params_path)
        print("Parâmetros carregados com sucesso.")
        
        # Analisar estrutura dos parâmetros
        print("\n=== Estrutura dos Parâmetros ===")
        for key in params.keys():
            if isinstance(params[key], dict):
                print(f"{key}: {len(params[key])} itens")
                # Exibir algumas chaves de exemplo
                subkeys = list(params[key].keys())
                print(f"  Exemplos: {subkeys[:3] if subkeys else 'Nenhum'}")
            else:
                print(f"{key}: {type(params[key])}")
        
        # Verificar vetorizadores específicos
        if 'vectorizers' in params:
            vectorizers = params['vectorizers']
            print(f"\n=== Vetorizadores ({len(vectorizers)}) ===")
            for key in vectorizers.keys():
                print(f"Vetorizador: {key}")
                vectorizer = vectorizers[key]
                
                # Obter termos TF-IDF para cada vetorizador
                if hasattr(vectorizer, 'get_feature_names_out'):
                    terms = vectorizer.get_feature_names_out()
                    print(f"  Termos: {len(terms)}")
                    print(f"  Exemplos: {terms[:5] if len(terms) > 0 else 'Nenhum'}")
                    
                    # Se for o vetorizador de déjame_un_mensaje, analisar em mais detalhes
                    if 'déjame' in key:
                        print("\n=== Termos para déjame_un_mensaje ===")
                        print("Termos disponíveis:")
                        for term in terms:
                            print(f"  - {term}")
        
        return params
    
    except Exception as e:
        print(f"Erro ao carregar parâmetros: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def check_missing_tfidf_terms(params, missing_columns):
    """Verifica se os termos TF-IDF faltantes estão nos parâmetros"""
    if 'vectorizers' not in params or not missing_columns:
        return
    
    # Extrair apenas os nomes dos termos TF-IDF de déjame_un_mensaje
    missing_terms = []
    for col in missing_columns:
        if col.startswith('déjame_un_mensaje_tfidf_'):
            term = col.replace('déjame_un_mensaje_tfidf_', '')
            missing_terms.append(term)
    
    if not missing_terms:
        print("\nNenhum termo TF-IDF faltante para déjame_un_mensaje")
        return
    
    print(f"\n=== Verificando {len(missing_terms)} termos TF-IDF faltantes ===")
    
    # Verificar se o vetorizador para déjame_un_mensaje existe
    déjame_vectorizer = None
    for key, vectorizer in params['vectorizers'].items():
        if 'déjame' in key:
            déjame_vectorizer = vectorizer
            break
    
    if déjame_vectorizer is None:
        print("Vetorizador para déjame_un_mensaje não encontrado!")
        return
    
    # Obter todos os termos do vetorizador
    available_terms = déjame_vectorizer.get_feature_names_out()
    
    # Verificar quais termos faltantes estão ou não no vetorizador
    found_terms = []
    not_found_terms = []
    
    for term in missing_terms:
        if term in available_terms:
            found_terms.append(term)
        else:
            not_found_terms.append(term)
    
    if found_terms:
        print(f"Termos encontrados no vetorizador mas faltantes na inferência ({len(found_terms)}):")
        for term in found_terms:
            print(f"  - {term}")
    
    if not_found_terms:
        print(f"\nTermos NÃO encontrados no vetorizador ({len(not_found_terms)}):")
        for term in not_found_terms:
            print(f"  - {term}")
    
    # Sugestão de solução
    if not_found_terms:
        print("\n=== Solução Sugerida ===")
        print("Os termos faltantes não estão no vetorizador. É necessário:")
        print("1. Regenerar os vetorizadores com todos os termos necessários")
        print("   OU")
        print("2. Modificar script4_module.py para adicionar manualmente esses termos")
    else:
        print("\n=== Solução Sugerida ===")
        print("Os termos estão no vetorizador mas não estão sendo incluídos na inferência.")
        print("Modifique script4_module.py para garantir que todos os termos do vetorizador")
        print("sejam incluídos para déjame_un_mensaje, especialmente ao usar o mapeamento de colunas.")

def main():
    # Carregar colunas faltantes
    missing_columns = load_missing_columns()
    print(f"Colunas faltantes carregadas: {len(missing_columns)}")
    
    # Analisar parâmetros
    params = analyze_parameters(params_path)
    
    if params:
        # Verificar termos TF-IDF faltantes
        check_missing_tfidf_terms(params, missing_columns)
        
        # Verificar se as colunas de treino existem
        if os.path.exists(train_cols_path):
            try:
                train_cols_df = pd.read_csv(train_cols_path)
                print(f"\nColunas de treino: {train_cols_df.shape}")
                
                # Verificar número de colunas relacionadas a déjame_un_mensaje
                déjame_cols = [col for col in train_cols_df.columns if 'déjame' in col]
                print(f"Colunas relacionadas a déjame_un_mensaje no treino: {len(déjame_cols)}")
                
                # Verificar TF-IDF cols
                tfidf_cols = [col for col in train_cols_df.columns if 'tfidf' in col]
                print(f"Colunas TF-IDF no treino: {len(tfidf_cols)}")
                
                # Verificar déjame TF-IDF cols
                déjame_tfidf = [col for col in train_cols_df.columns if 'déjame' in col and 'tfidf' in col]
                print(f"Colunas TF-IDF para déjame_un_mensaje no treino: {len(déjame_tfidf)}")
                if déjame_tfidf:
                    print("Exemplos:")
                    for col in déjame_tfidf[:5]:
                        print(f"  - {col}")
            except Exception as e:
                print(f"Erro ao analisar colunas de treino: {e}")
    
    print("\n=== Conclusão da Análise ===")
    print("Com base na análise, sugerimos:")
    print("1. Verificar se é necessário regenerar os parâmetros")
    print("2. Atualizar script4_module.py para lidar corretamente com os termos faltantes")

if __name__ == "__main__":
    main()