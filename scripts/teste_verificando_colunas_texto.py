#!/usr/bin/env python
"""
Script para diagnosticar o estado das features textuais nos datasets.
"""

import os
import pandas as pd
import numpy as np

def diagnose_text_features(dataset_path, output_path=None):
    """
    Analisa detalhadamente o dataset para buscar evidências de 
    colunas de texto ou suas derivações.
    
    Args:
        dataset_path: Caminho para o dataset CSV
        output_path: Opcional, caminho para salvar resultados detalhados
    """
    print(f"\n=== Diagnóstico de features textuais em: {dataset_path} ===\n")
    
    # Carregar o dataset
    df = pd.read_csv(dataset_path)
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # 1. Verificar se existem colunas do tipo object (potencialmente texto)
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Colunas do tipo object: {len(object_cols)}")
    if object_cols:
        print("Exemplos:", object_cols[:5])
        
        # Analisar detalhes das colunas object
        print("\nEstatísticas das colunas object:")
        for col in object_cols:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                avg_len = non_null.str.len().mean()
                max_len = non_null.str.len().max()
                print(f"  - '{col}': {non_null.nunique()} valores únicos, {non_null.count()} não-nulos")
                print(f"    Comprimento médio: {avg_len:.1f}, máximo: {max_len}")
                # Mostrar amostra dos primeiros valores não-nulos
                sample = non_null.iloc[:2].values
                print(f"    Amostra: {[s[:50] + '...' if len(s) > 50 else s for s in sample]}")
    
    # 2. Procurar evidências de colunas de texto processadas
    # Padrões comuns de sufixos de features derivadas
    text_feature_patterns = [
        '_length', '_word_count', '_avg_word_length', 
        '_tfidf_', '_sentiment', '_motiv_', 
        '_has_question', '_has_exclamation', '_topic_'
    ]
    
    derived_features = {}
    for pattern in text_feature_patterns:
        matches = [col for col in df.columns if pattern in col]
        if matches:
            derived_features[pattern] = matches
    
    print("\nFeatures derivadas de texto encontradas:")
    for pattern, cols in derived_features.items():
        print(f"  - {pattern}: {len(cols)} colunas")
        if len(cols) > 0:
            print(f"    Exemplos: {cols[:3]}")
    
    # 3. Inferir colunas de texto originais a partir das derivadas
    original_text_cols = set()
    for pattern, cols in derived_features.items():
        for col in cols:
            # Extrair nome base (tudo antes do sufixo)
            if pattern in ['_tfidf_', '_motiv_']:
                base = col.split(pattern)[0]
            else:
                base = col.rsplit(pattern, 1)[0]
            
            original_text_cols.add(base)
    
    print("\nPossíveis colunas de texto originais inferidas:")
    for col in original_text_cols:
        # Verificar se a coluna original existe
        exists = col in df.columns
        print(f"  - '{col}': {'Presente' if exists else 'Ausente'} no dataset")
        
        # Se existir, mostrar tipo de dados
        if exists:
            dtype = df[col].dtype
            n_unique = df[col].nunique()
            print(f"    Tipo: {dtype}, Valores únicos: {n_unique}")
            
            # Se for object, mostrar amostra
            if df[col].dtype == object:
                sample = df[col].dropna().iloc[:1].values
                if len(sample) > 0:
                    print(f"    Amostra: {sample[0][:50] + '...' if len(sample[0]) > 50 else sample[0]}")
    
    # 4. Verificar se temos as colunas indicadas na análise anterior
    expected_cols = [
        'Déjame un mensaje', 
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
    ]
    
    print("\nVerificando colunas específicas identificadas anteriormente:")
    for col in expected_cols:
        found = False
        # Verificar correspondência exata
        if col in df.columns:
            found = True
            status = "Presente (correspondência exata)"
        else:
            # Verificar correspondência parcial
            partial_matches = [c for c in df.columns if col.replace('?', '') in c or col.replace('¿', '') in c]
            if partial_matches:
                found = True
                status = f"Presente com nome similar: {partial_matches[0]}"
            else:
                # Verificar se há variações com espaços substituídos por underscores
                underscore_col = col.replace(' ', '_')
                if underscore_col in df.columns:
                    found = True
                    status = f"Presente com underscores: {underscore_col}"
                else:
                    # Verificar campos derivados
                    derivative_matches = [c for c in df.columns if col.replace(' ', '_') in c and any(p in c for p in text_feature_patterns)]
                    if derivative_matches:
                        found = True
                        status = f"Presente apenas como derivado: {derivative_matches[0]}"
                    else:
                        status = "Ausente"
        
        print(f"  - '{col}': {status}")
    
    # 5. Verificar se existem derivados dos nomes esperados com underscores
    expected_cols_underscores = [
        'Déjame_un_mensaje',
        'Cuando_hables_inglés_con_fluidez___qué_cambiará_en_tu_vida___Qué_oportunidades_se_abrirán_para_ti',
        '¿Qué_esperas_aprender_en_la_Semana_de_Cero_a_Inglés_Fluido?',
        '¿Qué_esperas_aprender_en_la_Inmersión_Desbloquea_Tu_Inglés_En_72_horas?'
    ]
    
    print("\nVerificando variações de nome com underscores:")
    for col in expected_cols_underscores:
        # Verificar correspondência exata ou como parte de nomes de colunas
        matches = [c for c in df.columns if col in c]
        if matches:
            print(f"  - '{col}': Encontrado como parte de {len(matches)} colunas")
            print(f"    Exemplos: {matches[:3]}")
        else:
            # Verificar correspondência parcial
            partial_matches = []
            parts = col.split('_')
            if len(parts) > 2:
                for c in df.columns:
                    if sum(1 for part in parts if part in c) >= min(3, len(parts) // 2):
                        partial_matches.append(c)
            
            if partial_matches:
                print(f"  - '{col}': Encontrado com correspondência parcial em {len(partial_matches)} colunas")
                print(f"    Exemplos: {partial_matches[:3]}")
            else:
                print(f"  - '{col}': Não encontrado")
    
    # Conclusão
    print("\n=== Conclusão do Diagnóstico ===")
    if not object_cols and original_text_cols:
        print("DIAGNÓSTICO: As colunas de texto originais parecem ter sido removidas,")
        print("            mas existem features derivadas que indicam seu processamento anterior.")
    elif object_cols:
        print("DIAGNÓSTICO: Existem colunas de texto (object) que podem ser usadas para feature engineering.")
    else:
        print("DIAGNÓSTICO: Não foram encontradas colunas de texto nem evidências claras de seu processamento.")
    
    # Salvar resultados detalhados se desejado
    if output_path:
        # Criar um DataFrame com as informações sobre as colunas
        results = pd.DataFrame({
            'column': df.columns,
            'dtype': df.dtypes.astype(str),
            'n_unique': [df[col].nunique() for col in df.columns],
            'n_nulls': [df[col].isna().sum() for col in df.columns],
            'is_derived': [any(p in col for p in text_feature_patterns) for col in df.columns]
        })
        
        # Identificar possíveis colunas base
        results['possible_base'] = results['column'].apply(
            lambda x: any(x.startswith(b) for b in original_text_cols)
        )
        
        # Salvar
        results.to_csv(output_path, index=False)
        print(f"\nResultados detalhados salvos em: {output_path}")
    
    return {
        'object_cols': object_cols,
        'derived_features': derived_features,
        'original_text_cols': original_text_cols
    }

if __name__ == "__main__":
    # Definir caminhos
    base_dir = os.path.expanduser("~")
    train_path = os.path.join(base_dir, "desktop/smart_ads/data/feature_selection/train.csv")
    val_path = os.path.join(base_dir, "desktop/smart_ads/data/feature_selection/validation.csv")
    output_dir = os.path.join(base_dir, "desktop/smart_ads/data/text_features_diagnostics")
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Analisar datasets
    train_results = diagnose_text_features(
        train_path, 
        os.path.join(output_dir, "train_text_features.csv")
    )
    
    val_results = diagnose_text_features(
        val_path,
        os.path.join(output_dir, "val_text_features.csv")
    )
    
    # Resumo final
    print("\n=== Resumo Final ===")
    print(f"Colunas de texto (object) no treino: {len(train_results['object_cols'])}")
    print(f"Colunas de texto (object) na validação: {len(val_results['object_cols'])}")
    
    print(f"Possíveis colunas de texto originais no treino: {len(train_results['original_text_cols'])}")
    print(f"Possíveis colunas de texto originais na validação: {len(val_results['original_text_cols'])}")
    
    if not train_results['object_cols'] and train_results['derived_features']:
        print("\nCONCLUSÃO: As colunas de texto originais provavelmente foram removidas durante o pré-processamento,")
        print("          mas existem features derivadas (TF-IDF, contagens, etc.) que podem ser usadas como base")
        print("          para criar novos embeddings, tópicos LDA e outras features avançadas.")