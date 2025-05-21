#!/usr/bin/env python
"""
Script adaptado para inferência, baseado no 04_feature_engineering_2.py original.
Função principal apply() recebe um DataFrame e retorna o DataFrame processado.
"""

import os
import sys
import pandas as pd
import numpy as np
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

# Definir colunas de texto a processar
TEXT_COLUMNS = [
    'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
    '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?', 
    'Déjame un mensaje',
    '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
]

def process_dataframe_for_inference(df, text_columns, params=None, batch_size=10000):
    """
    Processa um DataFrame para inferência, aplicando engenharia de features
    focada em motivação profissional.
    
    Args:
        df: DataFrame com dados a serem processados
        text_columns: Lista de colunas de texto para processar
        params: Parâmetros pré-treinados (se None, serão carregados do disco)
        batch_size: Tamanho do lote para processamento (para economia de memória)
        
    Returns:
        DataFrame processado com features adicionadas
    """
    print("\n=== Processando DataFrame para inferência ===")
    
    # Verificações iniciais
    n_samples = len(df)
    if n_samples == 0:
        print("ERRO: DataFrame vazio")
        return df
    
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
    
    # DataFrame para resultado final
    all_results = []
    
    # Iniciar cronômetro
    start_time = datetime.now()
    
    # Processamento por lotes
    for batch_idx in range(n_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min((batch_idx + 1) * batch_size, n_samples)
        batch_actual_size = batch_end_idx - batch_start_idx
        
        print(f"\nProcessando lote {batch_idx+1}/{n_batches} "
              f"(amostras {batch_start_idx+1}-{batch_end_idx}, {batch_actual_size} total)")
        
        # Extrair lote dos dados
        batch_df = df.iloc[batch_start_idx:batch_end_idx].copy()
        
        # 1. Criar features de motivação profissional
        print("  1. Aplicando scores de motivação profissional...")
        motivation_df, _ = create_professional_motivation_score(
            batch_df, text_columns, 
            fit=False,
            params=params['professional_motivation']
        )
        
        # 2. Features de sentimento de aspiração
        print("  2. Analisando sentimento de aspiração...")
        sentiment_df, _ = analyze_aspiration_sentiment(
            batch_df, text_columns,
            fit=False,
            params=params['aspiration_sentiment']
        )
        
        # 3. Features de expressões de compromisso
        print("  3. Detectando expressões de compromisso...")
        commitment_df, _ = detect_commitment_expressions(
            batch_df, text_columns,
            fit=False,
            params=params['commitment']
        )
        
        # 4. Features de termos de carreira
        print("  4. Aplicando detector de termos de carreira...")
        career_df, _ = create_career_term_detector(
            batch_df, text_columns,
            fit=False,
            params=params['career']
        )
        
        # 5. TF-IDF para carreira
        print("  5. Aplicando TF-IDF para termos de carreira...")
        if 'vectorizers' in params and params['vectorizers']:
            tfidf_df, _ = enhance_tfidf_for_career_terms(
                batch_df, text_columns,
                fit=False,
                params=params['vectorizers']
            )
        else:
            # Se não temos vetorizadores, criamos um DataFrame vazio para placeholder
            tfidf_df = pd.DataFrame(index=batch_df.index)
        
        # Combinar todas as features deste lote
        print("  Combinando features do lote...")
        batch_features = pd.concat(
            [motivation_df, sentiment_df, commitment_df, career_df, tfidf_df],
            axis=1
        )
        
        # Remover colunas duplicadas
        batch_features = batch_features.loc[:, ~batch_features.columns.duplicated()]
        
        # Combinar com o DataFrame original
        batch_result = pd.concat([batch_df, batch_features], axis=1)
        
        # Remover colunas duplicadas novamente
        batch_result = batch_result.loc[:, ~batch_result.columns.duplicated()]
        
        # Armazenar resultados
        all_results.append(batch_result)
        
        # Relatório de progresso
        elapsed = (datetime.now() - start_time).total_seconds()
        remaining = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
        print(f"  Lote {batch_idx+1}/{n_batches} concluído. "
              f"Progresso: {(batch_idx+1)/n_batches*100:.1f}%. "
              f"Tempo restante estimado: {remaining/60:.1f} min")
        
        # Limpar memória
        del batch_df, motivation_df, sentiment_df, commitment_df, career_df, tfidf_df, batch_features, batch_result
        gc.collect()
    
    # Combinar todos os lotes
    print("\n=== Combinando todos os lotes ===")
    final_result = pd.concat(all_results, axis=0)
    
    # Garantir que o índice está correto
    final_result.reset_index(drop=True, inplace=True)
    
    # Limpar variáveis não necessárias
    del all_results
    gc.collect()
    
    # Relatório final
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\nProcessamento concluído em {total_time/60:.1f} minutos")
    print(f"DataFrame final: {final_result.shape[0]} linhas, {final_result.shape[1]} colunas")
    
    return final_result

def apply(df, params=None):
    """
    Função principal para aplicar engenharia de features avançada em modo de inferência.
    
    Args:
        df: DataFrame com dados das etapas anteriores
        params: Parâmetros pré-treinados (se None, serão carregados do disco)
        
    Returns:
        DataFrame com features adicionadas
    """
    # Carregar parâmetros se não fornecidos
    if params is None:
        print("Parâmetros não fornecidos. Carregando do disco...")
        try:
            params_path = "/Users/ramonmoreira/desktop/smart_ads/src/preprocessing/04_params/04_params.joblib"
            print(f"Carregando parâmetros de: {params_path}")
            params = joblib.load(params_path)
        except Exception as e:
            print(f"ERRO ao carregar parâmetros: {e}")
            print("Prosseguindo sem parâmetros pré-treinados")
            params = {
                'professional_motivation': {},
                'vectorizers': {},
                'aspiration_sentiment': {},
                'commitment': {},
                'career': {}
            }
    
    # Verificar colunas de texto
    columns_to_process = [col for col in TEXT_COLUMNS if col in df.columns]
    
    if not columns_to_process:
        # Verificar versões originais
        columns_to_process = [col for col in df.columns if '_original' in col]
        print(f"Colunas de texto originais não encontradas. Usando {len(columns_to_process)} colunas com sufixo _original.")
    
    if not columns_to_process:
        print("AVISO: Nenhuma coluna de texto encontrada para processamento.")
        return df
    
    print(f"Processando {len(columns_to_process)} colunas de texto...")
    
    # Definir tamanho do lote adequado
    batch_size = min(10000, max(1000, len(df) // 10))
    
    # Processar DataFrame em lotes
    result_df = process_dataframe_for_inference(
        df,
        columns_to_process,
        params=params,
        batch_size=batch_size
    )
    
    return result_df

# Manter a funcionalidade original para uso direto do script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aplicar engenharia de features avançada em modo de inferência.")
    parser.add_argument("--input", type=str, required=True, 
                       help="Caminho para o arquivo CSV de entrada")
    parser.add_argument("--output", type=str, required=True,
                       help="Caminho para salvar o arquivo CSV processado")
    parser.add_argument("--batch-size", type=int, default=10000,
                       help="Tamanho do lote para processamento")
    
    args = parser.parse_args()
    
    # Carregar dados
    print(f"Carregando dados de: {args.input}")
    input_data = pd.read_csv(args.input)
    
    # Processar
    processed_data = apply(input_data)
    
    # Salvar resultado
    print(f"Salvando resultado em: {args.output}")
    processed_data.to_csv(args.output, index=False)
    
    print("Processamento concluído!")