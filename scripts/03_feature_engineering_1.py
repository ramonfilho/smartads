#!/usr/bin/env python
"""
Script complementar de processamento de texto para o projeto Smart Ads.
Este script aplica APENAS as features de NLP que NÃO são processadas pelo script 02.
"""

import pandas as pd
import numpy as np
import os
import sys
import re
import gc
import time
import joblib
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import nltk
import logging

# Silenciar avisos
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configurar projeto
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"

# Adicionar TODOS os paths necessários ao PYTHONPATH ANTES de qualquer import
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "preprocessing"))

# Verificar se o arquivo existe antes de tentar importar
professional_module_path = os.path.join(PROJECT_ROOT, "src", "preprocessing", "professional_motivation_features.py")
if not os.path.exists(professional_module_path):
    print(f"ERRO FATAL: Arquivo não encontrado: {professional_module_path}")
    print("O script não pode continuar sem este módulo.")
    sys.exit(1)

# Importar módulos obrigatórios - se falhar, o script para
try:
    from src.preprocessing.professional_motivation_features import (
        create_professional_motivation_score,
        analyze_aspiration_sentiment,
        detect_commitment_expressions,
        create_career_term_detector,
        enhance_tfidf_for_career_terms
    )
    print("✓ Módulos de motivação profissional carregados com sucesso!")
except ImportError as e:
    print(f"ERRO FATAL: Não foi possível importar os módulos necessários: {e}")
    print("\nVerifique:")
    print("1. Se o arquivo existe em: src/preprocessing/professional_motivation_features.py")
    print("2. Se não há erros de sintaxe no arquivo")
    print("3. Se todas as dependências estão instaladas")
    print(f"\nPYTHONPATH atual: {sys.path}")
    sys.exit(1)

# Diretórios
INPUT_DIR = os.path.join(PROJECT_ROOT, "data/new/02_processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/new/03_feature_engineering_1")
PARAMS_DIR = os.path.join(PROJECT_ROOT, "src/preprocessing/params/new/03_params")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "data/new/checkpoints/03_feature_engineering_1/03_professional")

# Criar diretórios necessários
for dir_path in [OUTPUT_DIR, PARAMS_DIR, CHECKPOINT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Otimização para processadores multi-core
os.environ["OMP_NUM_THREADS"] = str(cpu_count())

# Flag para habilitar/desabilitar multiprocessing
USE_MULTIPROCESSING = True

# Configurar NLTK silenciosamente
nltk_logger = logging.getLogger('nltk')
nltk_logger.setLevel(logging.CRITICAL)

def setup_nltk_resources():
    """Configura recursos NLTK necessários sem mensagens repetidas."""
    resources = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass

# Configurar NLTK uma única vez
setup_nltk_resources()

# Cache global para modelos
MODEL_CACHE = {}

def save_checkpoint(data, checkpoint_name):
    """Salva checkpoint do processamento."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    try:
        joblib.dump(data, checkpoint_path)
        print(f"  ✓ Checkpoint salvo: {checkpoint_name}")
    except Exception as e:
        print(f"  ✗ Erro ao salvar checkpoint: {e}")

def load_checkpoint(checkpoint_name):
    """Carrega checkpoint se existir."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    if os.path.exists(checkpoint_path):
        try:
            data = joblib.load(checkpoint_path)
            print(f"  ✓ Checkpoint carregado: {checkpoint_name}")
            return data
        except Exception as e:
            print(f"  ✗ Erro ao carregar checkpoint: {e}")
    return None

def identify_text_columns(df):
    """
    Identifica colunas de texto no DataFrame.
    """
    text_columns_normalized = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
        'Déjame un mensaje'
    ]
    
    text_columns = []
    for col in text_columns_normalized:
        if col in df.columns:
            text_columns.append(col)
            print(f"  ✓ Encontrada: {col[:60]}...")
    
    # Se não encontrou a versão normalizada, buscar variações
    if '¿Qué esperas aprender en el evento Cero a Inglés Fluido?' not in text_columns:
        variations = [
            '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
            '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
        ]
        for var in variations:
            if var in df.columns:
                text_columns.append(var)
                print(f"  ✓ Encontrada variação: {var[:60]}...")
    
    return text_columns

def process_professional_features_batch(df, text_columns, dataset_name, batch_size=5000, 
                                      fit=True, params=None):
    """
    Processa features profissionais em batches com checkpoints.
    """
    print(f"\n=== Processando features profissionais para {dataset_name} ===")
    start_time = time.time()
    
    # Verificar checkpoint
    checkpoint_data = load_checkpoint(f"{dataset_name}_professional")
    if checkpoint_data is not None:
        print(f"Retomando do checkpoint...")
        df = checkpoint_data['df']
        params = checkpoint_data['params']
        start_col = checkpoint_data['last_column'] + 1
        features_added = checkpoint_data.get('features_added', {})
    else:
        start_col = 0
        features_added = {}
        # Inicializar parâmetros
        if params is None:
            params = {
                'professional_motivation': {},
                'aspiration_sentiment': {},
                'commitment': {},
                'career_terms': {},
                'career_tfidf': {}
            }
    
    n_samples = len(df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"Processando {n_samples} amostras em {n_batches} batches (batch_size: {batch_size})")
    
    # Processar cada coluna de texto
    for col_idx in range(start_col, len(text_columns)):
        col = text_columns[col_idx]
        if col not in df.columns:
            continue
        
        print(f"\n[{col_idx+1}/{len(text_columns)}] Processando: {col[:60]}...")
        
        col_key = re.sub(r'[^\w]', '', col.replace(' ', '_'))[:30]
        features_added[col_key] = {
            'professional_motivation': 0,
            'aspiration': 0,
            'commitment': 0,
            'career_terms': 0,
            'career_tfidf': 0
        }
        
        # Processar em batches para economia de memória
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, n_samples)
            
            # Mostrar progresso
            if batch_idx % 5 == 0:
                elapsed = time.time() - start_time
                if batch_idx > 0:
                    avg_time_per_batch = elapsed / batch_idx
                    remaining_batches = n_batches - batch_idx
                    eta = avg_time_per_batch * remaining_batches
                    print(f"\r  Batch {batch_idx+1}/{n_batches} "
                          f"({(batch_idx+1)/n_batches*100:.1f}%) "
                          f"ETA: {eta/60:.1f} min", end='', flush=True)
            
            # Criar DataFrame do batch
            batch_df = df.iloc[batch_start:batch_end][[col]].copy()
            
            # 1. Score de motivação profissional
            if batch_idx == 0:
                print("\n  1. Calculando score de motivação profissional...")
            
            motiv_df, motiv_params = create_professional_motivation_score(
                batch_df, [col], 
                fit=fit and batch_idx == 0,
                params=params['professional_motivation'] if not (fit and batch_idx == 0) else None
            )
            if fit and batch_idx == 0:
                params['professional_motivation'] = motiv_params
            
            for motiv_col in motiv_df.columns:
                if motiv_col not in df.columns:
                    df[motiv_col] = np.nan
                df.loc[batch_start:batch_end-1, motiv_col] = motiv_df[motiv_col].values
                features_added[col_key]['professional_motivation'] += 1
            
            # 2. Análise de sentimento de aspiração
            if batch_idx == 0:
                print("\n  2. Analisando sentimento de aspiração...")
            
            asp_df, asp_params = analyze_aspiration_sentiment(
                batch_df, [col],
                fit=fit and batch_idx == 0,
                params=params['aspiration_sentiment'] if not (fit and batch_idx == 0) else None
            )
            if fit and batch_idx == 0:
                params['aspiration_sentiment'] = asp_params
            
            for asp_col in asp_df.columns:
                if asp_col not in df.columns:
                    df[asp_col] = np.nan
                df.loc[batch_start:batch_end-1, asp_col] = asp_df[asp_col].values
                features_added[col_key]['aspiration'] += 1
            
            # 3. Detecção de expressões de compromisso
            if batch_idx == 0:
                print("\n  3. Detectando expressões de compromisso...")
            
            comm_df, comm_params = detect_commitment_expressions(
                batch_df, [col],
                fit=fit and batch_idx == 0,
                params=params['commitment'] if not (fit and batch_idx == 0) else None
            )
            if fit and batch_idx == 0:
                params['commitment'] = comm_params
            
            for comm_col in comm_df.columns:
                if comm_col not in df.columns:
                    df[comm_col] = np.nan
                df.loc[batch_start:batch_end-1, comm_col] = comm_df[comm_col].values
                features_added[col_key]['commitment'] += 1
            
            # 4. Detector de termos de carreira
            if batch_idx == 0:
                print("\n  4. Detectando termos de carreira...")
            
            career_df, career_params = create_career_term_detector(
                batch_df, [col],
                fit=fit and batch_idx == 0,
                params=params['career_terms'] if not (fit and batch_idx == 0) else None
            )
            if fit and batch_idx == 0:
                params['career_terms'] = career_params
            
            for career_col in career_df.columns:
                if career_col not in df.columns:
                    df[career_col] = np.nan
                df.loc[batch_start:batch_end-1, career_col] = career_df[career_col].values
                features_added[col_key]['career_terms'] += 1
            
            # Limpar memória após cada batch
            del batch_df
            gc.collect()
        
        print()  # Nova linha após progresso
        
        # 5. TF-IDF aprimorado (processa coluna inteira)
        print("  5. Aplicando TF-IDF aprimorado para termos de carreira...")

        temp_df = df[[col]].copy()

        # CORREÇÃO: Usar col_key consistente
        tfidf_df, tfidf_params = enhance_tfidf_for_career_terms(
            temp_df, [col],
            fit=fit,
            params=params['career_tfidf'].get(col_key) if not fit else None  # ← AQUI
        )

        if fit:
            # CORREÇÃO: Salvar com col_key correto
            if 'career_tfidf' not in params:
                params['career_tfidf'] = {}
            params['career_tfidf'][col_key] = tfidf_params  # ← AQUI
        
        added_count = 0
        for tfidf_col in tfidf_df.columns:
            if tfidf_col not in df.columns and tfidf_col != col:
                df[tfidf_col] = tfidf_df[tfidf_col].values
                added_count += 1
        features_added[col_key]['career_tfidf'] = added_count
        print(f"     ✓ Adicionadas {added_count} features TF-IDF de carreira")
        
        # Salvar checkpoint após cada coluna
        save_checkpoint({
            'df': df,
            'params': params,
            'last_column': col_idx,
            'features_added': features_added
        }, f"{dataset_name}_professional")
    
    # Relatório final
    elapsed_time = time.time() - start_time
    print(f"\n✓ Processamento concluído em {elapsed_time/60:.1f} minutos")
    
    # Limpar checkpoint após conclusão
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name}_professional.pkl")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return df, params

# Adicionar esta função antes da função main():

def summarize_features(df, dataset_name, original_shape=None):
    """
    Sumariza as features criadas no DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"SUMÁRIO DE FEATURES - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    if original_shape:
        print(f"\n📊 DIMENSÕES:")
        print(f"   Original: {original_shape[0]} linhas × {original_shape[1]} colunas")
        print(f"   Atual:    {df.shape[0]} linhas × {df.shape[1]} colunas")
        print(f"   Features adicionadas: {df.shape[1] - original_shape[1]}")
    
    # Categorizar features por tipo
    feature_categories = {
        'professional_motivation': [],
        'aspiration': [],
        'commitment': [],
        'career_terms': [],
        'career_tfidf': [],
        'outras': []
    }
    
    new_features = [col for col in df.columns if any(pattern in col for pattern in [
        'professional_motivation', 'career_keyword', 'aspiration', 'commitment', 
        'career_term', 'career_tfidf'
    ])]
    
    for col in new_features:
        if 'professional_motivation' in col or 'career_keyword' in col:
            feature_categories['professional_motivation'].append(col)
        elif 'aspiration' in col:
            feature_categories['aspiration'].append(col)
        elif 'commitment' in col:
            feature_categories['commitment'].append(col)
        elif 'career_tfidf' in col:
            feature_categories['career_tfidf'].append(col)
        elif 'career_term' in col:
            feature_categories['career_terms'].append(col)
        else:
            feature_categories['outras'].append(col)
    
    print(f"\n📈 FEATURES CRIADAS POR CATEGORIA:")
    total_new = 0
    for category, features in feature_categories.items():
        if features:
            print(f"\n   {category.upper()} ({len(features)} features):")
            # Mostrar até 5 exemplos
            for i, feat in enumerate(features[:5]):
                print(f"      • {feat}")
            if len(features) > 5:
                print(f"      ... e mais {len(features) - 5} features")
            total_new += len(features)
    
    print(f"\n   TOTAL DE NOVAS FEATURES: {total_new}")
    
    # Estatísticas sobre valores ausentes nas novas features
    if new_features:
        print(f"\n📊 ESTATÍSTICAS DAS NOVAS FEATURES:")
        null_counts = df[new_features].isnull().sum()
        features_with_nulls = null_counts[null_counts > 0]
        if len(features_with_nulls) > 0:
            print(f"   Features com valores ausentes: {len(features_with_nulls)}/{len(new_features)}")
        else:
            print(f"   ✓ Todas as novas features estão completas (sem valores ausentes)")
    
    print(f"\n{'='*60}\n")

# Modificar a função main() para incluir os sumários:

def main():
    """Função principal."""
    
    print("=== PROCESSAMENTO DE FEATURES PROFISSIONAIS ===")
    print(f"Data e hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPUs disponíveis: {cpu_count()}")
    
    # Carregar datasets
    print("\nCarregando datasets processados pelo script 02...")
    try:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(INPUT_DIR, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
        
        print(f"✓ Datasets carregados:")
        print(f"  Train: {train_df.shape}")
        print(f"  Validation: {valid_df.shape}")
        print(f"  Test: {test_df.shape}")
    except Exception as e:
        print(f"ERRO FATAL: Não foi possível carregar os datasets: {e}")
        sys.exit(1)
    
    # Guardar shapes originais
    train_original_shape = train_df.shape
    valid_original_shape = valid_df.shape
    test_original_shape = test_df.shape
    
    # Verificar features já existentes
    print("\n📋 ANÁLISE INICIAL DAS FEATURES:")
    print("\nFeatures de texto já processadas pelo script 02:")
    existing_features = {
        'básicas': len([c for c in train_df.columns if any(x in c for x in ['_length', '_word_count', '_has_question'])]),
        'tfidf': len([c for c in train_df.columns if '_tfidf_' in c]),
        'sentiment': len([c for c in train_df.columns if '_sentiment' in c]),
        'motivation': len([c for c in train_df.columns if '_motiv_' in c and 'professional_motivation' not in c]),
        'topic': len([c for c in train_df.columns if '_topic_' in c]),
        'embedding': len([c for c in train_df.columns if '_embedding_' in c]),
        'discriminative': len([c for c in train_df.columns if '_high_conv_' in c or '_low_conv_' in c])
    }
    
    total_existing = 0
    for feature_type, count in existing_features.items():
        if count > 0:
            print(f"  {feature_type}: {count} features")
            total_existing += count
    print(f"\n  TOTAL de features de texto existentes: {total_existing}")
    print(f"  TOTAL de features no dataset: {train_df.shape[1]}")
    
    # Identificar colunas de texto
    print("\nIdentificando colunas de texto...")
    text_columns = identify_text_columns(train_df)
    
    if not text_columns:
        print("ERRO FATAL: Nenhuma coluna de texto encontrada!")
        sys.exit(1)
    
    print(f"\n✓ {len(text_columns)} colunas de texto identificadas")
    
    # Processar treino
    print("\n>>> PROCESSANDO TREINO <<<")
    train_processed, params = process_professional_features_batch(
        train_df, text_columns, 'train', batch_size=5000, fit=True
    )
    
    # Sumarizar features do treino
    summarize_features(train_processed, 'train', train_original_shape)
    
    # Salvar parâmetros
    print("\nSalvando parâmetros...")
    params_path = os.path.join(PARAMS_DIR, "03_professional_features_params.joblib")
    joblib.dump(params, params_path)
    
    # Processar validação
    print("\n>>> PROCESSANDO VALIDAÇÃO <<<")
    valid_processed, _ = process_professional_features_batch(
        valid_df, text_columns, 'validation', batch_size=5000, fit=False, params=params
    )
    
    # Sumarizar features da validação
    summarize_features(valid_processed, 'validation', valid_original_shape)
    
    # Processar teste
    print("\n>>> PROCESSANDO TESTE <<<")
    test_processed, _ = process_professional_features_batch(
        test_df, text_columns, 'test', batch_size=5000, fit=False, params=params
    )
    
    # Sumarizar features do teste
    summarize_features(test_processed, 'test', test_original_shape)
    
    # Verificar consistência de colunas
    print("\n🔍 VERIFICAÇÃO DE CONSISTÊNCIA:")
    train_cols = set(train_processed.columns)
    valid_cols = set(valid_processed.columns)
    test_cols = set(test_processed.columns)
    
    if train_cols == valid_cols == test_cols:
        print("✓ Todos os datasets têm exatamente as mesmas colunas")
        print(f"  Total de colunas: {len(train_cols)}")
    else:
        print("✗ AVISO: Inconsistência detectada nas colunas!")
        if train_cols - valid_cols:
            print(f"  Colunas em train mas não em valid: {len(train_cols - valid_cols)}")
        if train_cols - test_cols:
            print(f"  Colunas em train mas não em test: {len(train_cols - test_cols)}")
    
    # Salvar datasets
    print("\nSalvando datasets processados...")
    train_processed.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    valid_processed.to_csv(os.path.join(OUTPUT_DIR, 'validation.csv'), index=False)
    test_processed.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO FINAL DO PROCESSAMENTO")
    print(f"{'='*60}")
    print(f"\n📊 INCREMENTO DE FEATURES:")
    print(f"   Dataset   | Original | Final | Adicionadas")
    print(f"   ----------|----------|-------|------------")
    print(f"   Train     | {train_original_shape[1]:>8} | {train_processed.shape[1]:>5} | {train_processed.shape[1] - train_original_shape[1]:>11}")
    print(f"   Valid     | {valid_original_shape[1]:>8} | {valid_processed.shape[1]:>5} | {valid_processed.shape[1] - valid_original_shape[1]:>11}")
    print(f"   Test      | {test_original_shape[1]:>8} | {test_processed.shape[1]:>5} | {test_processed.shape[1] - test_original_shape[1]:>11}")
    
    print(f"\n📁 ARQUIVOS SALVOS:")
    print(f"   Datasets: {OUTPUT_DIR}")
    print(f"   Parâmetros: {PARAMS_DIR}")
    
    print(f"\n✅ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()