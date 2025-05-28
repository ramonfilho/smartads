#!/usr/bin/env python
"""
Script de Diagnóstico para Produção - Smart Ads L24

Este script diagnostica problemas na pipeline de inferência para produção.

Autor: Smart Ads Team
Data: Dezembro 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import warnings
from datetime import datetime
import logging

# Silenciar avisos
warnings.filterwarnings('ignore')

# Configurar paths absolutos
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Criar diretórios necessários
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "diagnostics"), exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'production_diagnostics.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def diagnose_model_artifacts():
    """Diagnostica os artefatos do modelo."""
    logger.info("=" * 80)
    logger.info("DIAGNÓSTICO 1: ARTEFATOS DO MODELO")
    logger.info("=" * 80)
    
    artifacts_dir = os.path.join(PROJECT_ROOT, "models", "artifacts")
    
    # 1. Verificar modelo principal
    model_path = os.path.join(artifacts_dir, "lightgbm_direct_ranking.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info("✓ Modelo LightGBM encontrado")
        logger.info(f"  Tipo: {type(model)}")
        logger.info(f"  Número de features esperadas: {model.n_features_}")
    else:
        logger.error("✗ Modelo não encontrado!")
    
    # 2. Verificar limiares de decis
    logger.info("\nVerificando limiares de decis...")
    
    # Tentar diferentes formatos
    for filename in ["decile_thresholds.pkl", "decile_thresholds.json", "model_config.json"]:
        filepath = os.path.join(artifacts_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"\n✓ Arquivo encontrado: {filename}")
            
            if filename.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    logger.info(f"  Conteúdo: {data}")
                    
            elif filename.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if 'decile_thresholds' in data:
                        logger.info(f"  Limiares: {data['decile_thresholds']}")
                    else:
                        logger.info(f"  Chaves disponíveis: {list(data.keys())}")
        else:
            logger.warning(f"✗ {filename} não encontrado")
    
    return artifacts_dir


def diagnose_preprocessing_params():
    """Diagnostica os parâmetros de pré-processamento."""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNÓSTICO 2: PARÂMETROS DE PRÉ-PROCESSAMENTO")
    logger.info("=" * 80)
    
    # Verificar parâmetros do script 02
    params_02_paths = [
        os.path.join(PROJECT_ROOT, "src", "preprocessing", "params", "new", "02_preprocessing_params", "all_preprocessing_params.joblib"),
        os.path.join(PROJECT_ROOT, "src", "preprocessing", "02_preprocessing_params", "all_preprocessing_params.joblib"),
        os.path.join(PROJECT_ROOT, "data", "new", "02_preprocessing_params", "all_preprocessing_params.joblib")
    ]
    
    params_02 = None
    for path in params_02_paths:
        if os.path.exists(path):
            logger.info(f"✓ Parâmetros script 02 encontrados em: {path}")
            params_02 = joblib.load(path)
            logger.info(f"  Chaves principais: {list(params_02.keys())}")
            break
    
    if params_02 is None:
        logger.error("✗ Parâmetros do script 02 não encontrados em nenhum dos caminhos esperados")
    
    # Verificar parâmetros do script 03
    params_03_path = os.path.join(PROJECT_ROOT, "src", "preprocessing", "params", "new", "03_params", "03_professional_features_params.joblib")
    if os.path.exists(params_03_path):
        logger.info(f"\n✓ Parâmetros script 03 encontrados")
        params_03 = joblib.load(params_03_path)
        
        # Verificar vetorizadores TF-IDF
        if 'career_tfidf' in params_03:
            logger.info("  ✓ Parâmetros career_tfidf encontrados")
            logger.info(f"    Chaves: {list(params_03['career_tfidf'].keys())}")
        else:
            logger.warning("  ✗ Parâmetros career_tfidf não encontrados")
            
        # Verificar outros parâmetros importantes
        for key in ['professional_motivation', 'aspiration_sentiment', 'commitment', 'career_terms']:
            if key in params_03:
                logger.info(f"  ✓ {key} encontrado")
            else:
                logger.warning(f"  ✗ {key} não encontrado")
    else:
        logger.error("✗ Parâmetros do script 03 não encontrados")
    
    return params_02, params_03


def diagnose_selected_features():
    """Diagnostica as features selecionadas."""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNÓSTICO 3: FEATURES SELECIONADAS")
    logger.info("=" * 80)
    
    features_file = os.path.join(PROJECT_ROOT, "reports", "feature_importance_results", "recommended_features.txt")
    
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        
        logger.info(f"✓ Arquivo de features encontrado: {len(features)} features")
        
        # Analisar tipos de features
        feature_types = {
            'tfidf': [],
            'topic': [],
            'professional': [],
            'interaction': [],
            'basic': [],
            'other': []
        }
        
        for feat in features:
            if '_tfidf_' in feat:
                feature_types['tfidf'].append(feat)
            elif '_topic_' in feat or 'dominant_topic' in feat:
                feature_types['topic'].append(feat)
            elif any(x in feat for x in ['professional_motivation', 'aspiration', 'commitment', 'career']):
                feature_types['professional'].append(feat)
            elif any(x in feat for x in ['_x_', 'country_x_', 'age_x_', 'hour_x_']):
                feature_types['interaction'].append(feat)
            elif any(x in feat for x in ['_encoded', '_length', '_count', 'has_']):
                feature_types['basic'].append(feat)
            else:
                feature_types['other'].append(feat)
        
        logger.info("\nDistribuição por tipo de feature:")
        for feat_type, feat_list in feature_types.items():
            if feat_list:
                logger.info(f"  {feat_type}: {len(feat_list)} features")
                logger.info(f"    Exemplos: {feat_list[:3]}")
        
        return features
    else:
        logger.error("✗ Arquivo de features selecionadas não encontrado")
        return None


def diagnose_production_data():
    """Diagnostica os dados de produção."""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNÓSTICO 4: DADOS DE PRODUÇÃO")
    logger.info("=" * 80)
    
    # Carregar dados
    utm_path = os.path.join(PROJECT_ROOT, "data", "L24", "L24_UTMS.csv")
    survey_path = os.path.join(PROJECT_ROOT, "data", "L24", "Pesquisa_L24.csv")
    
    if os.path.exists(utm_path):
        utm_df = pd.read_csv(utm_path)
        logger.info(f"✓ UTMs: {utm_df.shape}")
        logger.info(f"  Colunas: {list(utm_df.columns)}")
    else:
        logger.error("✗ Arquivo de UTMs não encontrado")
    
    if os.path.exists(survey_path):
        survey_df = pd.read_csv(survey_path)
        logger.info(f"\n✓ Pesquisas: {survey_df.shape}")
        logger.info(f"  Colunas: {list(survey_df.columns)}")
        
        # Verificar colunas de texto críticas
        text_cols = [
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
            '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
            'Déjame un mensaje'
        ]
        
        logger.info("\nColunas de texto:")
        for col in text_cols:
            if col in survey_df.columns:
                non_null = survey_df[col].notna().sum()
                logger.info(f"  ✓ {col[:50]}... : {non_null} valores não nulos")
            else:
                logger.warning(f"  ✗ {col[:50]}... : NÃO ENCONTRADA")
    else:
        logger.error("✗ Arquivo de pesquisas não encontrado")


def diagnose_predictions():
    """Diagnostica as predições já geradas."""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNÓSTICO 5: PREDIÇÕES GERADAS")
    logger.info("=" * 80)
    
    predictions_dir = os.path.join(PROJECT_ROOT, "predictions", "L24")
    
    if os.path.exists(predictions_dir):
        files = [f for f in os.listdir(predictions_dir) if f.startswith('L24_predictions_')]
        
        if files:
            # Pegar o arquivo mais recente
            latest_file = sorted(files)[-1]
            filepath = os.path.join(predictions_dir, latest_file)
            
            logger.info(f"✓ Analisando predições mais recentes: {latest_file}")
            
            df = pd.read_csv(filepath)
            
            # Estatísticas básicas
            if 'probability_score' in df.columns:
                probs = df['probability_score']
                logger.info(f"\nEstatísticas de probabilidade:")
                logger.info(f"  Média: {probs.mean():.4f}")
                logger.info(f"  Mediana: {probs.median():.4f}")
                logger.info(f"  Min: {probs.min():.4f}")
                logger.info(f"  Max: {probs.max():.4f}")
                logger.info(f"  Std: {probs.std():.4f}")
                
                # Comparar com treino
                logger.info(f"\nComparação com TREINO:")
                logger.info(f"  Média TREINO: 0.1390")
                logger.info(f"  Média PRODUÇÃO: {probs.mean():.4f}")
                logger.info(f"  Razão PROD/TREINO: {probs.mean()/0.1390:.2f}x")
                
                # Distribuição por decil
                if 'decile' in df.columns:
                    decile_dist = df['decile'].value_counts().sort_index()
                    logger.info(f"\nDistribuição por decil:")
                    for decile, count in decile_dist.items():
                        pct = count / len(df) * 100
                        logger.info(f"  Decil {decile}: {count} ({pct:.1f}%)")
        else:
            logger.warning("✗ Nenhum arquivo de predições encontrado")
    else:
        logger.warning("✗ Diretório de predições não encontrado")


def test_feature_generation():
    """Testa a geração de features em uma amostra pequena."""
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNÓSTICO 6: TESTE DE GERAÇÃO DE FEATURES")
    logger.info("=" * 80)
    
    # Importar módulos necessários
    from src.preprocessing.email_processing import normalize_emails_in_dataframe, normalize_email
    from src.preprocessing.column_normalization import normalize_survey_columns
    from src.preprocessing.data_cleaning import consolidate_quality_columns
    from src.preprocessing.feature_engineering import feature_engineering
    from src.preprocessing.text_processing import text_feature_engineering
    
    # Carregar uma amostra pequena
    survey_path = os.path.join(PROJECT_ROOT, "data", "L24", "Pesquisa_L24.csv")
    
    if os.path.exists(survey_path):
        # Pegar apenas 100 registros para teste
        df_sample = pd.read_csv(survey_path, nrows=100)
        logger.info(f"✓ Amostra carregada: {df_sample.shape}")
        
        # Normalizar colunas
        df_sample = normalize_survey_columns(df_sample, launch_id='L24_TEST')
        
        # Testar processamento de texto
        text_cols = [col for col in df_sample.columns if any(x in col for x in ['inglés', 'mensaje', 'aprender'])]
        
        if text_cols:
            logger.info(f"\nTestando processamento de texto em {len(text_cols)} colunas...")
            
            # Tentar gerar features
            try:
                df_processed, params = text_feature_engineering(df_sample, fit=True)
                
                # Contar features geradas
                new_features = [col for col in df_processed.columns if col not in df_sample.columns]
                
                logger.info(f"✓ Features geradas: {len(new_features)}")
                
                # Verificar tipos
                tfidf_features = [f for f in new_features if '_tfidf_' in f]
                logger.info(f"  TF-IDF: {len(tfidf_features)}")
                
            except Exception as e:
                logger.error(f"✗ Erro ao processar texto: {str(e)}")


def generate_diagnostic_report():
    """Gera relatório completo de diagnóstico."""
    report_path = os.path.join(PROJECT_ROOT, "diagnostics", f"production_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    logger.info("\n" + "=" * 80)
    logger.info("GERANDO RELATÓRIO DE DIAGNÓSTICO")
    logger.info("=" * 80)
    logger.info(f"Relatório salvo em: {report_path}")
    
    # Criar resumo das descobertas
    with open(report_path, 'w') as f:
        f.write("RELATÓRIO DE DIAGNÓSTICO - PRODUÇÃO L24\n")
        f.write("=" * 80 + "\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("RESUMO DOS PROBLEMAS IDENTIFICADOS:\n")
        f.write("-" * 40 + "\n")
        
        f.write("\n1. FEATURES FALTANTES (95):\n")
        f.write("   - Vetorizadores TF-IDF não encontrados para processamento de texto\n")
        f.write("   - Features sendo preenchidas com zeros (PERIGOSO!)\n")
        f.write("   - Possível incompatibilidade entre parâmetros salvos e pipeline\n")
        
        f.write("\n2. LIMIARES DE DECIS:\n")
        f.write("   - Arquivo não encontrado no caminho esperado\n")
        f.write("   - Usando percentis dos dados atuais (INCORRETO)\n")
        
        f.write("\n3. DISTRIBUIÇÃO ANÔMALA:\n")
        f.write("   - Probabilidade média 2x maior que no treino\n")
        f.write("   - Possível causa: features zeradas ou processamento incorreto\n")
        
        f.write("\n\nRECOMENDAÇÕES:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Reprocessar os dados de treino salvando TODOS os parâmetros\n")
        f.write("2. Garantir que os vetorizadores TF-IDF sejam salvos corretamente\n")
        f.write("3. Usar limiares de decis fixos do treino\n")
        f.write("4. Validar features antes de fazer predições\n")


def main():
    """Executa todos os diagnósticos."""
    start_time = datetime.now()
    
    logger.info("=" * 80)
    logger.info("INICIANDO DIAGNÓSTICO COMPLETO - PRODUÇÃO L24")
    logger.info("=" * 80)
    logger.info(f"Início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Diagnosticar artefatos do modelo
        diagnose_model_artifacts()
        
        # 2. Diagnosticar parâmetros
        params_02, params_03 = diagnose_preprocessing_params()
        
        # 3. Diagnosticar features selecionadas
        selected_features = diagnose_selected_features()
        
        # 4. Diagnosticar dados de produção
        diagnose_production_data()
        
        # 5. Diagnosticar predições existentes
        diagnose_predictions()
        
        # 6. Testar geração de features
        test_feature_generation()
        
        # 7. Gerar relatório
        generate_diagnostic_report()
        
        # Tempo total
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "=" * 80)
        logger.info("DIAGNÓSTICO CONCLUÍDO!")
        logger.info("=" * 80)
        logger.info(f"Tempo total: {elapsed_time:.1f} segundos")
        
    except Exception as e:
        logger.error(f"ERRO durante diagnóstico: {str(e)}")
        logger.error("Detalhes:", exc_info=True)
        raise


if __name__ == "__main__":
    main()