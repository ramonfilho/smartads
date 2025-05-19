#!/usr/bin/env python
"""
Script para processamento avançado de texto para o projeto Smart Ads.
Este script aplica engenharia de features focada em motivação profissional.

ESTRUTURA DE DADOS:
1. Carrega dados do Script 03 (features de texto básicas)
2. Carrega dados do Script 02 (features básicas)
3. Cria novas features de motivação profissional
4. Combina tudo em um único dataset final

Implementa processamento em lotes para economizar memória.
"""

import os
import sys
import pandas as pd
import numpy as np
import time
import joblib
import gc
import warnings
from datetime import datetime

# Adicionar o diretório raiz do projeto ao path do sistema
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importar módulos existentes
from src.preprocessing.professional_motivation_features import (
    create_professional_motivation_score,
    analyze_aspiration_sentiment,
    detect_commitment_expressions,
    create_career_term_detector,
    enhance_tfidf_for_career_terms
)

# Configurações
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "8"  # Otimiza para MacBook M1

# Define paths
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
INPUT_DIR_BASIC = os.path.join(PROJECT_ROOT, "data/02_processed") # Resultado do Script 02
INPUT_DIR_TEXT = os.path.join(PROJECT_ROOT, "data/03_feature_engineering_1") # Resultado do Script 03
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/04_feature_engineering_2_2") # Saída deste script
PARAMS_DIR = os.path.join(PROJECT_ROOT, "src/preprocessing/04_params") # Parâmetros do Script 04

# Criar diretórios se não existirem
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# Colunas de texto a processar
TEXT_COLUMNS = [
    'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
    '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?', 
    'Déjame un mensaje',
    '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
]

def load_datasets(dataset_name):
    """
    Carrega os datasets necessários para o processamento.
    
    Args:
        dataset_name: Nome do dataset (train, validation, test)
        
    Returns:
        Tuple (basic_df, text_df) com os DataFrames carregados
    """
    print(f"Carregando datasets para {dataset_name}...")
    
    # Caminhos dos arquivos
    basic_path = os.path.join(INPUT_DIR_BASIC, f"{dataset_name}.csv")
    text_path = os.path.join(INPUT_DIR_TEXT, f"{dataset_name}.csv")
    
    # Verificar se os arquivos existem
    for path in [basic_path, text_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    # Carregar datasets
    basic_df = pd.read_csv(basic_path, low_memory=False)
    text_df = pd.read_csv(text_path, low_memory=False)
    
    # Verificar se têm o mesmo número de linhas
    if len(basic_df) != len(text_df):
        raise ValueError(f"Datasets com número diferente de linhas: {len(basic_df)} vs {len(text_df)}")
    
    print(f"  Carregado basic_df: {basic_df.shape}")
    print(f"  Carregado text_df: {text_df.shape}")
    
    return basic_df, text_df

def process_dataset_in_batches(basic_df, text_df, text_columns, dataset_name, 
                              batch_size=10000, mode="fit", save_dir=None, 
                              params=None):
    """
    Processa um dataset em lotes para economizar memória.
    
    Args:
        basic_df: DataFrame com features básicas
        text_df: DataFrame com features de texto
        text_columns: Lista de colunas de texto para processar
        dataset_name: Nome do dataset (train, validation, test)
        batch_size: Tamanho do lote para processamento
        mode: "fit" para treinar parâmetros, "transform" para aplicar
        save_dir: Diretório para salvar resultado
        params: Parâmetros pré-existentes (para modo transform)
        
    Returns:
        DataFrame com todas as features
        Dicionário com parâmetros
    """
    print(f"\n=== Processando {dataset_name} em lotes ===")
    
    # Verificações iniciais
    n_samples = len(basic_df)
    if n_samples == 0:
        print("ERRO: Dataset vazio")
        return None, None
        
    # Calcular número de lotes
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(f"Processando {n_samples} amostras em {n_batches} lotes (batch_size: {batch_size})")
    
    # Inicializar parâmetros
    if params is None:
        params = {
            'professional_motivation': {},
            'vectorizers': {},
            'aspiration_sentiment': {},
            'commitment': {},
            'career': {}
        }
    
    # Estruturas para acumular dados para fit global (no modo fit)
    if mode == "fit":
        # Preparar para acumular dados para fit completo posteriormente
        all_text_samples = {col: [] for col in text_columns if col in basic_df.columns}
        text_counts = {col: 0 for col in text_columns if col in basic_df.columns}
    
    # DataFrame para resultado final
    all_results = []
    
    # Iniciar cronômetro
    start_time = time.time()
    
    # ======== ETAPA 1: Processamento por lotes ========
    for batch_idx in range(n_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min((batch_idx + 1) * batch_size, n_samples)
        batch_actual_size = batch_end_idx - batch_start_idx
        
        print(f"\nProcessando lote {batch_idx+1}/{n_batches} "
              f"(amostras {batch_start_idx+1}-{batch_end_idx}, {batch_actual_size} total)")
        
        # Extrair lote dos dados
        basic_batch = basic_df.iloc[batch_start_idx:batch_end_idx].copy()
        text_batch = text_df.iloc[batch_start_idx:batch_end_idx].copy()
        
        # No modo "fit", acumular textos para análise global posterior
        if mode == "fit":
            for col in text_columns:
                if col in basic_batch.columns:
                    valid_texts = basic_batch[col].dropna().astype(str)
                    # Acumular dados de texto
                    all_text_samples[col].extend(valid_texts.tolist())
                    text_counts[col] += len(valid_texts)
        
        # 1. Criar features de motivação profissional
        print("  1. Criando scores de motivação profissional...")
        motivation_df, _ = create_professional_motivation_score(
            basic_batch, text_columns, 
            fit=False,  # Sempre False nos lotes, faremos fit global depois
            params=params['professional_motivation']
        )
        
        # 2. Features de sentimento de aspiração
        print("  2. Analisando sentimento de aspiração...")
        sentiment_df, _ = analyze_aspiration_sentiment(
            basic_batch, text_columns,
            fit=False,  # Sempre False nos lotes
            params=params['aspiration_sentiment']
        )
        
        # 3. Features de expressões de compromisso
        print("  3. Detectando expressões de compromisso...")
        commitment_df, _ = detect_commitment_expressions(
            basic_batch, text_columns,
            fit=False,  # Sempre False nos lotes
            params=params['commitment']
        )
        
        # 4. Features de termos de carreira
        print("  4. Criando detector de termos de carreira...")
        career_df, _ = create_career_term_detector(
            basic_batch, text_columns,
            fit=False,  # Sempre False nos lotes
            params=params['career']
        )
        
        # 5. TF-IDF para carreira - aplicação em lote das transformações
        print("  5. Aplicando TF-IDF para termos de carreira...")
        # No modo transform, usa vetorizadores existentes
        # No modo fit, as features TF-IDF serão criadas no passo global
        if mode == "transform" and params.get('vectorizers'):
            tfidf_df, _ = enhance_tfidf_for_career_terms(
                basic_batch, text_columns,
                fit=False,
                params=params['vectorizers']
            )
        else:
            # No modo fit por lotes, criamos um DataFrame vazio para placeholder
            tfidf_df = pd.DataFrame(index=basic_batch.index)
        
        # Combinar todas as features deste lote
        print("  Combinando features do lote...")
        batch_features = pd.concat(
            [motivation_df, sentiment_df, commitment_df, career_df, tfidf_df],
            axis=1
        )
        
        # Combinar com o DataFrame de texto
        batch_result = pd.concat([text_batch, batch_features], axis=1)
        
        # Remover colunas duplicadas
        batch_result = batch_result.loc[:, ~batch_result.columns.duplicated()]
        
        # Armazenar resultados
        all_results.append(batch_result)
        
        # Relatório de progresso
        elapsed = time.time() - start_time
        remaining = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
        print(f"  Lote {batch_idx+1}/{n_batches} concluído. "
              f"Progresso: {(batch_idx+1)/n_batches*100:.1f}%. "
              f"Tempo restante estimado: {remaining/60:.1f} min")
        
        # Limpar memória
        del basic_batch, text_batch, motivation_df, sentiment_df, commitment_df, career_df, tfidf_df, batch_features, batch_result
        gc.collect()
    
    # ======== ETAPA 2: Ajuste global (apenas no modo fit) ========
    if mode == "fit":
        print("\n=== Realizando ajuste global dos parâmetros ===")
        
        # 1. Ajuste global para motivação profissional
        print("1. Ajuste global para motivação profissional...")
        # Criar DataFrame temporário com todos os textos acumulados
        temp_df = pd.DataFrame()
        for col in text_columns:
            if col in all_text_samples and all_text_samples[col]:
                # Usar amostra de textos para ajuste global
                temp_df[col] = all_text_samples[col][:10000]  # limitar para gestão de memória
                
        # Ajustar parâmetros globais
        _, params['professional_motivation'] = create_professional_motivation_score(
            temp_df, text_columns, fit=True
        )
        
        # 2. Ajuste global para aspiração e sentimento
        print("2. Ajuste global para aspiração e sentimento...")
        _, params['aspiration_sentiment'] = analyze_aspiration_sentiment(
            temp_df, text_columns, fit=True
        )
        
        # 3. Ajuste global para expressões de compromisso
        print("3. Ajuste global para expressões de compromisso...")
        _, params['commitment'] = detect_commitment_expressions(
            temp_df, text_columns, fit=True
        )
        
        # 4. Ajuste global para termos de carreira
        print("4. Ajuste global para termos de carreira...")
        _, params['career'] = create_career_term_detector(
            temp_df, text_columns, fit=True
        )
        
        # 5. Ajuste global para TF-IDF - mais importante
        print("5. Ajuste global para TF-IDF de carreira...")
        _, params['vectorizers'] = enhance_tfidf_for_career_terms(
            temp_df, text_columns, fit=True
        )
        
        # ADICIONAR: Verificar e extrair termos de carreira dos vetorizadores 
        # Esta é a parte crucial que estava faltando!
        if 'vectorizers' in params and 'career_tfidf' in params['vectorizers']:
            for col_key, vectorizer_info in params['vectorizers']['career_tfidf'].items():
                if 'career_terms' in vectorizer_info:
                    # Garantir que temos career_terms também em params['career']
                    if 'career' not in params:
                        params['career'] = {}
                    if 'career_terms' not in params['career'] or not params['career'].get('career_terms'):
                        params['career']['career_terms'] = vectorizer_info['career_terms']
                        print(f"  Career terms extraídos do vetorizador para '{col_key}'")
        
        # Aplicar TF-IDF globalmente em todos os lotes
        print("6. Aplicando TF-IDF global em todos os lotes...")
        tfidf_results = []
        
        for batch_idx, batch_df in enumerate(all_results):
            print(f"  Aplicando TF-IDF no lote {batch_idx+1}/{n_batches}...")
            
            # Extrair o batch original
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            basic_batch = basic_df.iloc[start_idx:end_idx].copy()
            
            # Aplicar TF-IDF
            tfidf_batch, _ = enhance_tfidf_for_career_terms(
                basic_batch, text_columns,
                fit=False,
                params=params['vectorizers']
            )
            
            # Combinar com o lote existente
            result_with_tfidf = pd.concat([batch_df, tfidf_batch], axis=1)
            result_with_tfidf = result_with_tfidf.loc[:, ~result_with_tfidf.columns.duplicated()]
            
            tfidf_results.append(result_with_tfidf)
            
            # Limpar memória
            del basic_batch, tfidf_batch, result_with_tfidf
            gc.collect()
        
        # Substituir resultados
        all_results = tfidf_results
        del tfidf_results
        gc.collect()
        
        # Limpar memória de ajuste global
        del temp_df, all_text_samples
        gc.collect()
    
    # ======== ETAPA 3: Combinar todos os lotes ========
    print("\n=== Combinando todos os lotes ===")
    final_result = pd.concat(all_results, axis=0)
    
    # Garantir que o índice está correto
    final_result.reset_index(drop=True, inplace=True)
    
    # Limpar variáveis não necessárias
    del all_results
    gc.collect()
    
    # Salvar resultado
    if save_dir:
        output_path = os.path.join(save_dir, f"{dataset_name}.csv")
        print(f"Salvando resultado em: {output_path}")
        final_result.to_csv(output_path, index=False)
    
    # Relatório final
    total_time = time.time() - start_time
    print(f"\nProcessamento concluído em {total_time/60:.1f} minutos")
    print(f"Dataset final: {final_result.shape[0]} linhas, {final_result.shape[1]} colunas")
    
    return final_result, params

def main():
    """Função principal"""
    print("=== INICIANDO ENGENHARIA DE FEATURES AVANÇADA ===")
    print(f"Data e hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuração do tamanho do lote - ajuste conforme necessário
    BATCH_SIZE = 10000
    
    # Parâmetros iniciais
    all_params = None
    
    try:
        # 1. Processar dataset de treino (modo fit)
        print("\n>>> PROCESSANDO CONJUNTO DE TREINO <<<")
        try:
            # Carregar datasets
            train_basic, train_text = load_datasets("train")
            
            # Processar em lotes
            train_result, all_params = process_dataset_in_batches(
                train_basic, train_text, TEXT_COLUMNS, 
                "train", batch_size=BATCH_SIZE, 
                mode="fit", save_dir=OUTPUT_DIR
            )
            
            print(f"✓ Conjunto de treino processado: {train_result.shape}")
            
        except Exception as e:
            print(f"ERRO no processamento do conjunto de treino: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 2. Processar dataset de validação (modo transform)
        print("\n>>> PROCESSANDO CONJUNTO DE VALIDAÇÃO <<<")
        try:
            # Carregar datasets
            val_basic, val_text = load_datasets("validation")
            
            # Processar em lotes
            val_result, _ = process_dataset_in_batches(
                val_basic, val_text, TEXT_COLUMNS, 
                "validation", batch_size=BATCH_SIZE, 
                mode="transform", save_dir=OUTPUT_DIR, 
                params=all_params
            )
            
            print(f"✓ Conjunto de validação processado: {val_result.shape}")
            
        except Exception as e:
            print(f"ERRO no processamento do conjunto de validação: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. Processar dataset de teste (modo transform)
        print("\n>>> PROCESSANDO CONJUNTO DE TESTE <<<")
        try:
            # Carregar datasets
            test_basic, test_text = load_datasets("test")
            
            # Processar em lotes
            test_result, _ = process_dataset_in_batches(
                test_basic, test_text, TEXT_COLUMNS, 
                "test", batch_size=BATCH_SIZE, 
                mode="transform", save_dir=OUTPUT_DIR, 
                params=all_params
            )
            
            print(f"✓ Conjunto de teste processado: {test_result.shape}")
            
        except Exception as e:
            print(f"ERRO no processamento do conjunto de teste: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. Salvar parâmetros
        if all_params:
            print("\n>>> SALVANDO PARÂMETROS <<<")
            try:
                # MODIFICAÇÃO: Reestruturar os parâmetros para formato consistente
                structured_params = {
                    'professional_motivation': {
                        'work_keywords': {}
                    },
                    'aspiration_sentiment': {
                        'aspiration_phrases': []
                    },
                    'commitment': {
                        'commitment_phrases': {}
                    },
                    'career': {
                        'career_terms': {}
                    },
                    'vectorizers': {}
                }
                
                # Examinar todos os parâmetros para debugging
                print("Inspecionando chaves do dicionário all_params:")
                for key in all_params.keys():
                    print(f"  Chave: {key}")
                    if isinstance(all_params[key], dict):
                        print(f"    Subchaves: {list(all_params[key].keys())}")
                
                # Extrair parâmetros das chaves específicas
                
                # 1. work_keywords (extraindo da localização correta)
                if 'work_keywords' in all_params:
                    # Chave direta
                    structured_params['professional_motivation']['work_keywords'] = all_params['work_keywords']
                    print("  work_keywords encontrado diretamente na raiz dos parâmetros")
                elif 'professional_motivation' in all_params and 'work_keywords' in all_params['professional_motivation']:
                    # Subchave within professional_motivation
                    structured_params['professional_motivation']['work_keywords'] = all_params['professional_motivation']['work_keywords']
                    print("  work_keywords encontrado em professional_motivation")
                else:
                    # Busca recursiva em todas as subchaves
                    found = False
                    for key, value in all_params.items():
                        if isinstance(value, dict) and 'work_keywords' in value:
                            structured_params['professional_motivation']['work_keywords'] = value['work_keywords']
                            print(f"  work_keywords encontrado em {key}")
                            found = True
                            break
                    
                    if not found:
                        print("AVISO: Parâmetro 'work_keywords' não encontrado. Usando valores padrão.")
                        structured_params['professional_motivation']['work_keywords'] = {
                            'trabajo': 1.0, 'empleo': 1.0, 'carrera': 1.2, 'profesional': 1.2,
                            'oportunidades': 1.0, 'mejor': 0.7, 'mejorar': 0.7
                        }
                
                # 2. aspiration_phrases
                if 'aspiration_phrases' in all_params:
                    structured_params['aspiration_sentiment']['aspiration_phrases'] = all_params['aspiration_phrases']
                    print("  aspiration_phrases encontrado diretamente na raiz dos parâmetros")
                elif 'aspiration_sentiment' in all_params and 'aspiration_phrases' in all_params['aspiration_sentiment']:
                    structured_params['aspiration_sentiment']['aspiration_phrases'] = all_params['aspiration_sentiment']['aspiration_phrases']
                    print("  aspiration_phrases encontrado em aspiration_sentiment")
                else:
                    # Busca recursiva
                    found = False
                    for key, value in all_params.items():
                        if isinstance(value, dict) and 'aspiration_phrases' in value:
                            structured_params['aspiration_sentiment']['aspiration_phrases'] = value['aspiration_phrases']
                            print(f"  aspiration_phrases encontrado em {key}")
                            found = True
                            break
                    
                    if not found:
                        print("AVISO: Parâmetro 'aspiration_phrases' não encontrado. Usando valores padrão.")
                        structured_params['aspiration_sentiment']['aspiration_phrases'] = [
                            'quiero ser', 'espero ser', 'mi meta es', 'mi objetivo es',
                            'en el futuro', 'me veo', 'me visualizo'
                        ]
                
                # 3. commitment_phrases
                if 'commitment_phrases' in all_params:
                    structured_params['commitment']['commitment_phrases'] = all_params['commitment_phrases']
                    print("  commitment_phrases encontrado diretamente na raiz dos parâmetros")
                elif 'commitment' in all_params and 'commitment_phrases' in all_params['commitment']:
                    structured_params['commitment']['commitment_phrases'] = all_params['commitment']['commitment_phrases']
                    print("  commitment_phrases encontrado em commitment")
                else:
                    # Busca recursiva
                    found = False
                    for key, value in all_params.items():
                        if isinstance(value, dict) and 'commitment_phrases' in value:
                            structured_params['commitment']['commitment_phrases'] = value['commitment_phrases']
                            print(f"  commitment_phrases encontrado em {key}")
                            found = True
                            break
                    
                    if not found:
                        print("AVISO: Parâmetro 'commitment_phrases' não encontrado. Usando valores padrão.")
                        structured_params['commitment']['commitment_phrases'] = {
                            'estoy decidido': 2.0, 'me comprometo': 2.0, 'quiero aprender': 1.0
                        }
                
                # 4. career_terms
                if 'career_terms' in all_params:
                    structured_params['career']['career_terms'] = all_params['career_terms']
                    print("  career_terms encontrado diretamente na raiz dos parâmetros")
                elif 'career' in all_params and 'career_terms' in all_params['career']:
                    structured_params['career']['career_terms'] = all_params['career']['career_terms']
                    print("  career_terms encontrado em career")
                else:
                    # Busca em vectorizers
                    found = False
                    if 'vectorizers' in all_params and 'career_tfidf' in all_params['vectorizers']:
                        for col_key, vectorizer_info in all_params['vectorizers']['career_tfidf'].items():
                            if isinstance(vectorizer_info, dict) and 'career_terms' in vectorizer_info:
                                structured_params['career']['career_terms'] = vectorizer_info['career_terms']
                                print(f"  career_terms encontrado em vectorizers > career_tfidf > {col_key}")
                                found = True
                                break
                    
                    # Busca recursiva geral
                    if not found:
                        for key, value in all_params.items():
                            if isinstance(value, dict) and 'career_terms' in value:
                                structured_params['career']['career_terms'] = value['career_terms']
                                print(f"  career_terms encontrado em {key}")
                                found = True
                                break
                    
                    if not found:
                        print("AVISO: Parâmetro 'career_terms' não encontrado. Usando valores padrão.")
                        structured_params['career']['career_terms'] = {
                            'crecimiento profesional': 2.0, 'desarrollo profesional': 2.0,
                            'oportunidades laborales': 2.0, 'mejor salario': 1.8
                        }
                
                # 5. vectorizers
                if 'vectorizers' in all_params:
                    structured_params['vectorizers'] = all_params['vectorizers']
                    print("  vectorizers encontrado na raiz dos parâmetros")
                elif 'career_tfidf' in all_params:
                    structured_params['vectorizers']['career_tfidf'] = all_params['career_tfidf']
                    print("  vectorizers extraído de career_tfidf na raiz dos parâmetros")
                else:
                    print("AVISO: Nenhum vectorizer encontrado nos parâmetros.")
                
                # Verificar quais parâmetros estão faltando
                missing_params = []
                if not structured_params['professional_motivation']['work_keywords']:
                    missing_params.append('work_keywords')
                if not structured_params['aspiration_sentiment']['aspiration_phrases']:
                    missing_params.append('aspiration_phrases')
                if not structured_params['commitment']['commitment_phrases']:
                    missing_params.append('commitment_phrases')
                if not structured_params['career']['career_terms']:
                    missing_params.append('career_terms')
                if not structured_params['vectorizers']:
                    missing_params.append('vectorizers')
                
                # Salvar parâmetros estruturados específicos do script
                script_params_path = os.path.join(PARAMS_DIR, "04_params.joblib")
                joblib.dump(structured_params, script_params_path)
                print(f"✓ Parâmetros do Script 04 salvos em: {script_params_path}")
                
                if missing_params:
                    print(f"AVISO: Alguns parâmetros não foram encontrados e foram substituídos por valores padrão: {missing_params}")
                
                # Tentar atualizar arquivo de parâmetros global
                all_params_path = os.path.join(PARAMS_DIR, "all_preprocessing_params.joblib")
                if os.path.exists(all_params_path):
                    try:
                        global_params = joblib.load(all_params_path)
                        global_params['script04_features'] = structured_params
                        
                        # Salvar versão atualizada
                        updated_params_path = os.path.join(PARAMS_DIR, "all_preprocessing_params_updated.joblib")
                        joblib.dump(global_params, updated_params_path)
                        print(f"✓ Parâmetros globais atualizados em: {updated_params_path}")
                        
                    except Exception as e:
                        print(f"AVISO: Não foi possível atualizar parâmetros globais: {e}")
            except Exception as e:
                print(f"ERRO ao salvar parâmetros: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n=== PROCESSAMENTO CONCLUÍDO COM SUCESSO ===")
        
    except Exception as e:
        print(f"ERRO FATAL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()