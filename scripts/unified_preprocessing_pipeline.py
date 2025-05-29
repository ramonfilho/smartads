#!/usr/bin/env python
"""
unified_preprocessing_pipeline.py - Pipeline unificada que garante consistência de nomes
Combina scripts 02 e 03 em um único processo com convenções de nomes padronizadas
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
import argparse
import re
import time
from datetime import datetime

# Configuração
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings('ignore')

# Imports dos módulos necessários
from src.preprocessing.email_processing import normalize_emails_in_dataframe
from src.preprocessing.data_cleaning import (
    consolidate_quality_columns,
    handle_missing_values,
    handle_outliers,
    normalize_values,
    convert_data_types
)
from src.preprocessing.feature_engineering import feature_engineering
from src.preprocessing.text_processing import text_feature_engineering
from src.preprocessing.advanced_feature_engineering import advanced_feature_engineering
from src.preprocessing.professional_motivation_features import (
    create_professional_motivation_score,
    analyze_aspiration_sentiment,
    detect_commitment_expressions,
    create_career_term_detector,
    enhance_tfidf_for_career_terms
)

# CRÍTICO: Convenções de nomes padronizadas
TEXT_COLUMN_STANDARDS = {
    'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?': 'cuando_hables',
    '¿Qué esperas aprender en el evento Cero a Inglés Fluido?': 'que_esperas',
    '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?': 'que_esperas',  # Variação
    '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?': 'que_esperas',  # Variação
    'Déjame un mensaje': 'dejame_mensaje'
}

def standardize_column_name(col_name, is_text_base=False):
    """
    Padroniza nomes de colunas para garantir consistência
    """
    if is_text_base and col_name in TEXT_COLUMN_STANDARDS:
        return TEXT_COLUMN_STANDARDS[col_name]
    
    # Para features derivadas, usar nome padrão
    standardized = col_name.lower()
    
    # Remover acentos
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n',
        'ü': 'u', 'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u'
    }
    for old, new in replacements.items():
        standardized = standardized.replace(old, new)
    
    # Remover caracteres especiais
    standardized = re.sub(r'[¿?¡!,.:;()\-\']', '', standardized)
    
    # Substituir espaços por underscore
    standardized = re.sub(r'\s+', '_', standardized)
    
    # Remover underscores múltiplos
    standardized = re.sub(r'_+', '_', standardized)
    
    # Remover underscores nas extremidades
    standardized = standardized.strip('_')
    
    # Truncar se muito longo (mas preservar sufixo importante)
    if len(standardized) > 50 and not any(suffix in standardized for suffix in ['_tfidf_', '_topic_', '_motiv_']):
        standardized = standardized[:50]
    
    return standardized

def apply_unified_preprocessing(df, params=None, fit=False, preserve_text=True):
    """
    Pipeline unificada de pré-processamento com nomes padronizados
    """
    if params is None:
        params = {
            'column_mapping': {},
            'feature_creation_log': []
        }
    
    print(f"\n=== PIPELINE UNIFICADA DE PRÉ-PROCESSAMENTO ===")
    print(f"Modo: {'FIT' if fit else 'TRANSFORM'}")
    print(f"Shape inicial: {df.shape}")
    
    # Registrar colunas originais
    original_columns = list(df.columns)
    
    # 1. Pré-processamento básico (scripts do 02)
    print("\n📋 FASE 1: Pré-processamento Básico")
    
    print("  1.1 Normalizando emails...")
    df = normalize_emails_in_dataframe(df, email_col='email')
    
    print("  1.2 Consolidando colunas de qualidade...")
    quality_params = params.get('quality_columns', {})
    df, quality_params = consolidate_quality_columns(df, fit=fit, params=quality_params)
    
    print("  1.3 Tratando valores ausentes...")
    missing_params = params.get('missing_values', {})
    df, missing_params = handle_missing_values(df, fit=fit, params=missing_params)
    
    print("  1.4 Tratando outliers...")
    outlier_params = params.get('outliers', {})
    df, outlier_params = handle_outliers(df, fit=fit, params=outlier_params)
    
    print("  1.5 Normalizando valores...")
    norm_params = params.get('normalization', {})
    df, norm_params = normalize_values(df, fit=fit, params=norm_params)
    
    print("  1.6 Convertendo tipos...")
    df, _ = convert_data_types(df, fit=fit)
    
    # 2. Identificar e padronizar colunas de texto
    print("\n📝 FASE 2: Padronização de Colunas de Texto")
    
    text_cols = []
    text_col_mapping = {}
    
    for col in df.columns:
        if col in TEXT_COLUMN_STANDARDS:
            standard_name = TEXT_COLUMN_STANDARDS[col]
            text_cols.append(col)
            text_col_mapping[col] = standard_name
            
            if fit:
                params['column_mapping'][col] = standard_name
            
            print(f"  ✓ {col[:50]}... → {standard_name}")
    
    # Preservar colunas originais
    if preserve_text and text_cols:
        print("\n  Preservando colunas de texto originais...")
        for col in text_cols:
            df[f"{col}_original"] = df[col].copy()
    
    # 3. Feature Engineering (do script 02)
    print("\n🔧 FASE 3: Feature Engineering Básica")
    
    feature_params = params.get('feature_engineering', {})
    df, feature_params = feature_engineering(df, fit=fit, params=feature_params)
    
    # 4. Processamento de Texto com Nomes Padronizados
    print("\n📊 FASE 4: Processamento de Texto")
    
    # IMPORTANTE: Criar cópia temporária com features padronizadas
    df_for_text = df.copy()
    
    # Antes de processar texto, renomear temporariamente as colunas base
    temp_renames = {}
    for orig_col, std_name in text_col_mapping.items():
        if orig_col in df_for_text.columns:
            temp_col_name = f"_temp_{std_name}"
            df_for_text[temp_col_name] = df_for_text[orig_col]
            temp_renames[temp_col_name] = std_name
    
    # Processar features de texto
    text_params = params.get('text_processing', {})
    df_text_features = pd.DataFrame(index=df.index)
    
    # 4.1 Features básicas de texto
    print("  4.1 Extraindo features básicas de texto...")
    for col in text_cols:
        std_name = text_col_mapping[col]
        
        # Comprimento
        df_text_features[f'{std_name}_length'] = df[col].str.len()
        
        # Contagem de palavras
        df_text_features[f'{std_name}_word_count'] = df[col].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Sentimento (simplificado)
        df_text_features[f'{std_name}_sentiment'] = 0  # Placeholder
    
    # 4.2 TF-IDF com nomes padronizados
    print("  4.2 Aplicando TF-IDF...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    for col in text_cols:
        std_name = text_col_mapping[col]
        
        if fit:
            vectorizer = TfidfVectorizer(
                max_features=20,  # Limitado para evitar explosão
                min_df=5,
                ngram_range=(1, 2)
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(df[col].fillna(''))
                
                # Salvar vetorizador
                if 'tfidf_vectorizers' not in params:
                    params['tfidf_vectorizers'] = {}
                params['tfidf_vectorizers'][std_name] = vectorizer
                
                # Criar features com nomes padronizados
                feature_names = vectorizer.get_feature_names_out()
                for i, term in enumerate(feature_names):
                    feat_name = f'{std_name}_tfidf_{standardize_column_name(term)}'
                    df_text_features[feat_name] = tfidf_matrix[:, i].toarray().ravel()
                
                print(f"    ✓ {std_name}: {len(feature_names)} features TF-IDF")
                
            except Exception as e:
                print(f"    ✗ Erro em TF-IDF para {std_name}: {e}")
        
        else:  # Transform
            if 'tfidf_vectorizers' in params and std_name in params['tfidf_vectorizers']:
                vectorizer = params['tfidf_vectorizers'][std_name]
                
                try:
                    tfidf_matrix = vectorizer.transform(df[col].fillna(''))
                    feature_names = vectorizer.get_feature_names_out()
                    
                    for i, term in enumerate(feature_names):
                        feat_name = f'{std_name}_tfidf_{standardize_column_name(term)}'
                        df_text_features[feat_name] = tfidf_matrix[:, i].toarray().ravel()
                    
                except Exception as e:
                    print(f"    ✗ Erro em transform TF-IDF para {std_name}: {e}")
    
    # 5. Features Profissionais (do script 03)
    print("\n💼 FASE 5: Features Profissionais")
    
    # 5.1 Motivação profissional
    print("  5.1 Score de motivação profissional...")
    prof_motivation_df, prof_params = create_professional_motivation_score(
        df, text_cols, fit=fit, params=params.get('professional_motivation')
    )
    
    # Padronizar nomes das features criadas
    for col in prof_motivation_df.columns:
        std_col_name = standardize_column_name(col)
        df_text_features[std_col_name] = prof_motivation_df[col]
    
    # 5.2 Análise de sentimento de aspiração
    print("  5.2 Sentimento de aspiração...")
    aspiration_df, asp_params = analyze_aspiration_sentiment(
        df, text_cols, fit=fit, params=params.get('aspiration_sentiment')
    )
    
    for col in aspiration_df.columns:
        std_col_name = standardize_column_name(col)
        df_text_features[std_col_name] = aspiration_df[col]
    
    # 5.3 Detecção de compromisso
    print("  5.3 Expressões de compromisso...")
    commitment_df, comm_params = detect_commitment_expressions(
        df, text_cols, fit=fit, params=params.get('commitment')
    )
    
    for col in commitment_df.columns:
        std_col_name = standardize_column_name(col)
        df_text_features[std_col_name] = commitment_df[col]
    
    # 5.4 Termos de carreira
    print("  5.4 Termos de carreira...")
    career_df, career_params = create_career_term_detector(
        df, text_cols, fit=fit, params=params.get('career_terms')
    )
    
    for col in career_df.columns:
        std_col_name = standardize_column_name(col)
        df_text_features[std_col_name] = career_df[col]
    
    # 6. LDA Topics com nomes padronizados
    print("\n🎯 FASE 6: Modelagem de Tópicos (LDA)")
    
    from sklearn.decomposition import LatentDirichletAllocation
    
    for col in text_cols:
        std_name = text_col_mapping[col]
        
        # Usar features TF-IDF existentes para LDA
        tfidf_cols = [c for c in df_text_features.columns if c.startswith(f'{std_name}_tfidf_')]
        
        if len(tfidf_cols) >= 10:  # Precisa de features suficientes
            print(f"  Aplicando LDA para {std_name}...")
            
            if fit:
                lda = LatentDirichletAllocation(
                    n_components=5,
                    max_iter=10,
                    random_state=42
                )
                
                try:
                    tfidf_data = df_text_features[tfidf_cols].values
                    topic_dist = lda.fit_transform(tfidf_data)
                    
                    # Salvar modelo
                    if 'lda_models' not in params:
                        params['lda_models'] = {}
                    params['lda_models'][std_name] = lda
                    
                    # Criar features de tópico com nomes padronizados
                    for i in range(5):
                        df_text_features[f'{std_name}_topic_{i+1}'] = topic_dist[:, i]
                    
                    print(f"    ✓ 5 tópicos criados")
                    
                except Exception as e:
                    print(f"    ✗ Erro em LDA: {e}")
            
            else:  # Transform
                if 'lda_models' in params and std_name in params['lda_models']:
                    lda = params['lda_models'][std_name]
                    
                    try:
                        tfidf_data = df_text_features[tfidf_cols].values
                        topic_dist = lda.transform(tfidf_data)
                        
                        for i in range(5):
                            df_text_features[f'{std_name}_topic_{i+1}'] = topic_dist[:, i]
                            
                    except Exception as e:
                        print(f"    ✗ Erro em transform LDA: {e}")
    
    # 7. Combinar todas as features
    print("\n✅ FASE 7: Combinando Features")
    
    # Remover colunas de texto originais (mantendo apenas _original se preserve_text)
    cols_to_drop = text_cols
    df_final = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Adicionar todas as features de texto
    for col in df_text_features.columns:
        df_final[col] = df_text_features[col]
    
    # 8. Feature engineering avançada
    print("\n🚀 FASE 8: Feature Engineering Avançada")
    advanced_params = params.get('advanced_features', {})
    df_final, advanced_params = advanced_feature_engineering(df_final, fit=fit, params=advanced_params)
    
    # Atualizar parâmetros
    params.update({
        'quality_columns': quality_params,
        'missing_values': missing_params,
        'outliers': outlier_params,
        'normalization': norm_params,
        'feature_engineering': feature_params,
        'text_processing': text_params,
        'professional_motivation': prof_params if fit else params.get('professional_motivation'),
        'aspiration_sentiment': asp_params if fit else params.get('aspiration_sentiment'),
        'commitment': comm_params if fit else params.get('commitment'),
        'career_terms': career_params if fit else params.get('career_terms'),
        'advanced_features': advanced_params
    })
    
    print(f"\n📊 Pipeline concluída!")
    print(f"   Shape inicial: {len(original_columns)} colunas")
    print(f"   Shape final: {df_final.shape[1]} colunas")
    print(f"   Features criadas: {df_final.shape[1] - len(original_columns)}")
    
    return df_final, params

def process_all_datasets():
    """
    Processa todos os datasets com a pipeline unificada
    """
    print("=== PROCESSAMENTO UNIFICADO DE DATASETS ===")
    print(f"Timestamp: {datetime.now()}")
    
    # Diretórios
    input_dir = os.path.join(PROJECT_ROOT, "data/new/01_split")
    output_dir = os.path.join(PROJECT_ROOT, "data/unified_v1/03_all_features")
    params_dir = os.path.join(PROJECT_ROOT, "src/preprocessing/params/unified_v1")
    
    # Criar diretórios
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    
    # 1. Processar treino (fit)
    print("\n>>> PROCESSANDO TREINO (FIT)")
    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    print(f"Shape original: {train_df.shape}")
    
    train_processed, params = apply_unified_preprocessing(train_df, fit=True)
    
    # Salvar
    train_processed.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    joblib.dump(params, os.path.join(params_dir, "unified_preprocessing_params.joblib"))
    
    print(f"\n✅ Treino processado e salvo")
    print(f"   Features totais: {train_processed.shape[1]}")
    
    # Salvar lista de features
    with open(os.path.join(params_dir, "feature_list.txt"), 'w') as f:
        for col in sorted(train_processed.columns):
            f.write(f"{col}\n")
    
    # 2. Processar validação (transform)
    print("\n>>> PROCESSANDO VALIDAÇÃO (TRANSFORM)")
    val_df = pd.read_csv(os.path.join(input_dir, "validation.csv"))
    val_processed, _ = apply_unified_preprocessing(val_df, params=params, fit=False)
    
    # Garantir mesmas colunas
    val_processed = val_processed.reindex(columns=train_processed.columns, fill_value=0)
    val_processed.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    
    print(f"✅ Validação processada: {val_processed.shape}")
    
    # 3. Processar teste (transform)
    print("\n>>> PROCESSANDO TESTE (TRANSFORM)")
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
    test_processed, _ = apply_unified_preprocessing(test_df, params=params, fit=False)
    
    # Garantir mesmas colunas
    test_processed = test_processed.reindex(columns=train_processed.columns, fill_value=0)
    test_processed.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"✅ Teste processado: {test_processed.shape}")
    
    # Relatório final
    print("\n" + "="*60)
    print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print(f"\nDatasets salvos em: {output_dir}")
    print(f"Parâmetros salvos em: {params_dir}")
    print(f"\nTodos os datasets têm exatamente {train_processed.shape[1]} colunas")
    print("Nomes de features são 100% consistentes entre treino/validação/teste")
    
    return True

if __name__ == "__main__":
    process_all_datasets()