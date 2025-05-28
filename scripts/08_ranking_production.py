#!/usr/bin/env python
"""
Script de Inferência para Produção - Smart Ads L24

Este script processa os dados de produção do lançamento L24 através de toda a pipeline
e gera predições usando o modelo LightGBM treinado.

Autor: Smart Ads Team
Data: Dezembro 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
from datetime import datetime
import json
import logging

# Silenciar avisos
warnings.filterwarnings('ignore')

# Configurar paths absolutos
PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Criar diretórios necessários ANTES de configurar logging
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "predictions", "L24"), exist_ok=True)

# Configurar logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
       logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'production_inference.log')),
       logging.StreamHandler()
   ]
)
logger = logging.getLogger(__name__)

# Importar módulos necessários
from src.preprocessing.email_processing import normalize_emails_in_dataframe, normalize_email
from src.preprocessing.column_normalization import normalize_survey_columns
from src.preprocessing.data_cleaning import (
   consolidate_quality_columns,
   handle_missing_values,
   handle_outliers,
   normalize_values,
   convert_data_types
)
from src.preprocessing.feature_engineering import feature_engineering
from src.preprocessing.text_processing import text_feature_engineering
from src.preprocessing.professional_motivation_features import enhance_professional_features

# Importar as funções corretas do advanced_feature_engineering
from src.preprocessing.advanced_feature_engineering import (
   advanced_feature_engineering,
   refine_tfidf_weights,
   create_text_embeddings_simple,
   perform_topic_modeling,
   identify_text_columns,
   # CORREÇÃO: Importar as funções _fixed daqui
   refine_tfidf_weights_fixed,
   create_text_embeddings_simple_fixed
)

# Importar apenas a função que realmente existe no script 03
from scripts.feature_engineering_03 import (
   perform_topic_modeling_fixed
)

# Caminhos dos arquivos
PRODUCTION_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "L24")
UTM_FILE = os.path.join(PRODUCTION_DATA_DIR, "L24_UTMS.csv")
SURVEY_FILE = os.path.join(PRODUCTION_DATA_DIR, "Pesquisa_L24.csv")

# Caminhos dos artefatos
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "artifacts", "lightgbm_direct_ranking.joblib")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "predictions", "L24")

# Features selecionadas (top 300)
SELECTED_FEATURES_FILE = os.path.join(PROJECT_ROOT, "reports", "feature_importance_results", "recommended_features.txt")


def load_production_data():
   """Carrega os dados de produção do L24."""
   logger.info("=" * 80)
   logger.info("CARREGANDO DADOS DE PRODUÇÃO L24")
   logger.info("=" * 80)
   
   # Carregar UTMs
   logger.info(f"Carregando UTMs de: {UTM_FILE}")
   utm_df = pd.read_csv(UTM_FILE)
   
   # Remover colunas Unnamed
   unnamed_cols = [col for col in utm_df.columns if col.startswith('Unnamed')]
   if unnamed_cols:
       utm_df = utm_df.drop(columns=unnamed_cols)
       logger.info(f"  Removidas {len(unnamed_cols)} colunas 'Unnamed'")
   
   logger.info(f"  UTMs carregadas: {utm_df.shape[0]:,} registros, {utm_df.shape[1]} colunas")
   
   # Carregar Pesquisas
   logger.info(f"Carregando pesquisas de: {SURVEY_FILE}")
   survey_df = pd.read_csv(SURVEY_FILE)
   logger.info(f"  Pesquisas carregadas: {survey_df.shape[0]:,} registros, {survey_df.shape[1]} colunas")
   
   return utm_df, survey_df


def integrate_data(utm_df, survey_df):
   """Integra dados de UTM e pesquisa (similar ao script 01)."""
   logger.info("\n" + "=" * 80)
   logger.info("INTEGRANDO DADOS")
   logger.info("=" * 80)
   
   # Normalizar colunas da pesquisa
   logger.info("Normalizando colunas da pesquisa...")
   survey_df = normalize_survey_columns(survey_df, launch_id='L24')
   
   # Normalizar emails
   logger.info("Normalizando emails...")
   
   # Para pesquisas
   if '¿Cuál es tu e-mail?' in survey_df.columns:
       survey_df['email_norm'] = survey_df['¿Cuál es tu e-mail?'].apply(normalize_email)
   else:
       logger.warning("Coluna de email da pesquisa não encontrada!")
   
   # Para UTMs
   if 'E-MAIL' in utm_df.columns:
       utm_df['email_norm'] = utm_df['E-MAIL'].apply(normalize_email)
   else:
       logger.warning("Coluna E-MAIL das UTMs não encontrada!")
   
   # Remover duplicatas de UTM antes do merge
   logger.info(f"UTMs antes da deduplicação: {len(utm_df):,}")
   utm_df_dedup = utm_df.drop_duplicates(subset=['email_norm'])
   logger.info(f"UTMs após deduplicação: {len(utm_df_dedup):,}")
   
   # Fazer merge
   logger.info("Mesclando dados de pesquisa com UTM...")
   merged_df = pd.merge(
       survey_df,
       utm_df_dedup,
       on='email_norm',
       how='left',
       suffixes=('', '_utm')
   )
   
   logger.info(f"Dados integrados: {merged_df.shape[0]:,} registros, {merged_df.shape[1]} colunas")
   
   # Remover colunas duplicadas e desnecessárias
   cols_to_remove = ['email_norm', 'email']
   cols_to_remove = [col for col in cols_to_remove if col in merged_df.columns]
   if cols_to_remove:
       merged_df = merged_df.drop(columns=cols_to_remove)
   
   return merged_df


def load_preprocessing_params():
   """Carrega todos os parâmetros de pré-processamento salvos durante o treinamento."""
   logger.info("\n" + "=" * 80)
   logger.info("CARREGANDO PARÂMETROS DE PRÉ-PROCESSAMENTO")
   logger.info("=" * 80)
   
   params = {}
   
   # Parâmetros do script 02 - CAMINHO CORRIGIDO
   params_02_path = os.path.join(PROJECT_ROOT, "src", "preprocessing", "params", "new", "02_preprocessing_params", "all_preprocessing_params.joblib")
   if os.path.exists(params_02_path):
       params['preprocessing'] = joblib.load(params_02_path)
       logger.info("✓ Parâmetros do script 02 carregados")
   else:
       logger.warning(f"Parâmetros do script 02 não encontrados em: {params_02_path}")
   
   # Parâmetros do script 03
   params_03_path = os.path.join(PROJECT_ROOT, "src", "preprocessing", "params", "new", "03_params", "03_professional_features_params.joblib")
   if os.path.exists(params_03_path):
       params['professional'] = joblib.load(params_03_path)
       logger.info("✓ Parâmetros do script 03 carregados")
   else:
       logger.warning(f"Parâmetros do script 03 não encontrados em: {params_03_path}")
   
   return params


def apply_preprocessing_pipeline(df, params):
   """Aplica toda a pipeline de pré-processamento (script 02)."""
   logger.info("\n" + "=" * 80)
   logger.info("APLICANDO PRÉ-PROCESSAMENTO")
   logger.info("=" * 80)
   
   df_result = df.copy()
   preprocessing_params = params.get('preprocessing', {})
   
   # 1. Normalizar emails
   logger.info("1. Normalizando emails...")
   df_result = normalize_emails_in_dataframe(df_result, email_col='email')
   
   # 2. Consolidar colunas de qualidade
   logger.info("2. Consolidando colunas de qualidade...")
   quality_params = preprocessing_params.get('quality_columns', {})
   df_result, _ = consolidate_quality_columns(df_result, fit=False, params=quality_params)
   
   # 3. Tratamento de valores ausentes
   logger.info("3. Tratando valores ausentes...")
   missing_params = preprocessing_params.get('missing_values', {})
   df_result, _ = handle_missing_values(df_result, fit=False, params=missing_params)
   
   # 4. Tratamento de outliers
   logger.info("4. Tratando outliers...")
   outlier_params = preprocessing_params.get('outliers', {})
   df_result, _ = handle_outliers(df_result, fit=False, params=outlier_params)
   
   # 5. Normalização de valores
   logger.info("5. Normalizando valores numéricos...")
   norm_params = preprocessing_params.get('normalization', {})
   df_result, _ = normalize_values(df_result, fit=False, params=norm_params)
   
   # 6. Converter tipos de dados
   logger.info("6. Convertendo tipos de dados...")
   df_result, _ = convert_data_types(df_result, fit=False)
   
   # Identificar colunas de texto ANTES do processamento
   text_cols = [
       col for col in df_result.columns 
       if df_result[col].dtype == 'object' and any(term in col for term in [
           'mensaje', 'inglés', 'vida', 'oportunidades', 'esperas', 'aprender', 
           'Semana', 'Inmersión', 'Déjame', 'fluidez'
       ])
   ]
   logger.info(f"Colunas de texto identificadas: {len(text_cols)}")
   
   # Preservar colunas de texto originais
   for col in text_cols:
       df_result[f"{col}_original"] = df_result[col].copy()
   
   # 7. Feature engineering não-textual
   logger.info("7. Aplicando feature engineering não-textual...")
   feature_params = preprocessing_params.get('feature_engineering', {})
   df_result, _ = feature_engineering(df_result, fit=False, params=feature_params)
   
   # 8. Processamento de texto
   logger.info("8. Processando features textuais...")
   text_params = preprocessing_params.get('text_processing', {})
   df_result, _ = text_feature_engineering(df_result, fit=False, params=text_params)
   
   # 9. Feature engineering avançada
   logger.info("9. Aplicando feature engineering avançada...")
   advanced_params = preprocessing_params.get('advanced_features', {})
   df_result, _ = advanced_feature_engineering(df_result, fit=False, params=advanced_params)
   
   logger.info(f"Pré-processamento concluído: {df_result.shape}")
   
   return df_result


def apply_professional_features_complete(df, params):
   """Aplica features profissionais COMPLETAS (script 03)."""
   logger.info("\n" + "=" * 80)
   logger.info("APLICANDO FEATURES PROFISSIONAIS (SCRIPT 03)")
   logger.info("=" * 80)
   
   professional_params = params.get('professional', {})
   
   # Identificar colunas de texto
   text_columns = [
       'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
       '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
       'Déjame un mensaje'
   ]
   
   # Filtrar apenas colunas existentes
   text_columns = [col for col in text_columns if col in df.columns]
   logger.info(f"Colunas de texto encontradas: {len(text_columns)}")
   
   if text_columns:
       # 1. Features profissionais básicas
       logger.info("1. Aplicando features profissionais básicas...")
       df_result, _ = enhance_professional_features(df, text_columns, fit=False, params=professional_params)
       
       # 2. TF-IDF refinado (versão _fixed do advanced_feature_engineering)
       logger.info("2. Aplicando TF-IDF refinado...")
       df_result, _ = refine_tfidf_weights_fixed(df_result, text_columns, fit=False, params=professional_params)
       
       # 3. Embeddings simples (versão _fixed do advanced_feature_engineering)
       logger.info("3. Criando embeddings de texto...")
       df_result, _ = create_text_embeddings_simple_fixed(df_result, text_columns, fit=False, params=professional_params)
       
       # 4. Topic modeling (versão _fixed do script 03) - CRÍTICO!
       logger.info("4. Aplicando modelagem de tópicos (LDA)...")
       df_result, _ = perform_topic_modeling_fixed(df_result, text_columns, n_topics=5, fit=False, params=professional_params)
       
       logger.info(f"Features profissionais completas aplicadas: {df_result.shape}")
       
       # Verificar se as features de tópicos foram criadas
       topic_features = [col for col in df_result.columns if '_topic_' in col]
       logger.info(f"Features de tópicos criadas: {len(topic_features)}")
       if topic_features:
           logger.info(f"Exemplos: {topic_features[:5]}")
       
       return df_result
   else:
       logger.warning("Nenhuma coluna de texto encontrada para features profissionais")
       return df


def load_selected_features():
   """Carrega a lista de features selecionadas."""
   logger.info("\n" + "=" * 80)
   logger.info("CARREGANDO FEATURES SELECIONADAS")
   logger.info("=" * 80)
   
   if os.path.exists(SELECTED_FEATURES_FILE):
       with open(SELECTED_FEATURES_FILE, 'r') as f:
           features = [line.strip() for line in f.readlines()]
       logger.info(f"✓ {len(features)} features selecionadas carregadas")
       return features
   else:
       logger.error(f"Arquivo de features selecionadas não encontrado: {SELECTED_FEATURES_FILE}")
       raise FileNotFoundError("Arquivo de features selecionadas não encontrado")


def validate_and_fix_features(df, selected_features):
   """Valida e corrige features antes da predição."""
   logger.info("\n" + "=" * 80)
   logger.info("VALIDANDO E CORRIGINDO FEATURES")
   logger.info("=" * 80)
   
   # Verificar quais features existem
   existing_features = [f for f in selected_features if f in df.columns]
   missing_features = set(selected_features) - set(existing_features)
   
   logger.info(f"Features existentes: {len(existing_features)}/{len(selected_features)}")
   
   if missing_features:
       logger.warning(f"Features ausentes: {len(missing_features)}")
       
       # Analisar tipos de features ausentes
       missing_types = {
           'tfidf': [],
           'topic': [],
           'professional': [],
           'other': []
       }
       
       for feat in missing_features:
           if '_tfidf_' in feat:
               missing_types['tfidf'].append(feat)
           elif '_topic_' in feat:
               missing_types['topic'].append(feat)
           elif any(x in feat for x in ['professional', 'aspiration', 'commitment', 'career']):
               missing_types['professional'].append(feat)
           else:
               missing_types['other'].append(feat)
       
       for feat_type, feats in missing_types.items():
           if feats:
               logger.warning(f"  {feat_type}: {len(feats)} features ausentes")
               logger.warning(f"    Exemplos: {feats[:3]}")
   
   # Criar DataFrame com features selecionadas
   df_filtered = df[existing_features].copy()
   
   # Preencher features ausentes com valores apropriados
   logger.info("Preenchendo features ausentes com valores apropriados...")
   
   for feature in missing_features:
       if '_tfidf_' in feature:
           # Para TF-IDF, usar 0 é apropriado (termo não aparece)
           df_filtered[feature] = 0
       elif '_topic_' in feature:
           # Para tópicos, usar distribuição uniforme
           df_filtered[feature] = 1.0 / 5  # Assumindo 5 tópicos
       else:
           # Para outras features, usar média das features do mesmo tipo
           similar_features = [f for f in existing_features if f.split('_')[0] == feature.split('_')[0]]
           if similar_features:
               df_filtered[feature] = df[similar_features].mean(axis=1)
           else:
               df_filtered[feature] = 0
   
   # Reordenar colunas conforme a lista original
   df_filtered = df_filtered[selected_features]
   
   # Validar valores
   logger.info("\nValidando valores das features:")
   zero_cols = (df_filtered == 0).all()
   num_zero_cols = zero_cols.sum()
   
   if num_zero_cols > 0:
       logger.warning(f"  ⚠️  {num_zero_cols} features estão completamente zeradas")
       zero_features = df_filtered.columns[zero_cols].tolist()
       logger.warning(f"    Exemplos: {zero_features[:5]}")
   else:
       logger.info("  ✓ Nenhuma feature completamente zerada")
   
   # Estatísticas
   logger.info(f"\nEstatísticas das features:")
   logger.info(f"  Média geral: {df_filtered.mean().mean():.4f}")
   logger.info(f"  Desvio padrão médio: {df_filtered.std().mean():.4f}")
   logger.info(f"  Valores NaN: {df_filtered.isna().sum().sum()}")
   
   logger.info(f"\nDataFrame final: {df_filtered.shape}")
   
   return df_filtered


def make_predictions(df, model_path):
   """Faz predições usando o modelo treinado."""
   logger.info("\n" + "=" * 80)
   logger.info("FAZENDO PREDIÇÕES")
   logger.info("=" * 80)
   
   # Carregar modelo
   logger.info(f"Carregando modelo de: {model_path}")
   model = joblib.load(model_path)
   
   # Fazer predições
   logger.info("Gerando probabilidades...")
   probabilities = model.predict_proba(df)[:, 1]
   
   # Carregar limiares de decis
   deciles_path = os.path.join(PROJECT_ROOT, "models", "artifacts", "decile_thresholds.pkl")
   if os.path.exists(deciles_path):
       with open(deciles_path, 'rb') as f:
           decile_thresholds = pickle.load(f)
       logger.info("✓ Limiares de decis carregados do arquivo .pkl")
       logger.info(f"  Limiares: {decile_thresholds}")
       
       # Calcular decis
       deciles = np.digitize(probabilities, decile_thresholds) + 1
       deciles = np.clip(deciles, 1, 10)
   else:
       logger.warning("Limiares de decis não encontrados, usando valores do treino")
       decile_thresholds = np.array([0.00524785, 0.01170815, 0.02155099, 0.03593944, 
                                     0.0577897, 0.09347356, 0.15549475, 0.25659115, 
                                     0.40992528])
       deciles = np.digitize(probabilities, decile_thresholds) + 1
       deciles = np.clip(deciles, 1, 10)
   
   return probabilities, deciles


def generate_output(original_df, probabilities, deciles):
   """Gera arquivos de saída com as predições."""
   logger.info("\n" + "=" * 80)
   logger.info("GERANDO ARQUIVOS DE SAÍDA")
   logger.info("=" * 80)
   
   # Criar DataFrame de resultados
   results_df = original_df.copy()
   results_df['probability_score'] = probabilities
   results_df['decile'] = deciles
   results_df['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   
   # Ordenar por probabilidade decrescente
   results_df = results_df.sort_values('probability_score', ascending=False)
   
   # Salvar arquivo completo
   output_file = os.path.join(OUTPUT_DIR, f"L24_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
   results_df.to_csv(output_file, index=False)
   logger.info(f"✓ Predições salvas em: {output_file}")
   
   # Gerar resumo por decil
   decile_summary = results_df.groupby('decile').agg({
       'probability_score': ['count', 'mean', 'min', 'max']
   }).round(4)
   
   decile_summary.columns = ['count', 'avg_prob', 'min_prob', 'max_prob']
   decile_summary['pct_of_total'] = (decile_summary['count'] / len(results_df) * 100).round(2)
   
   logger.info("\nRESUMO POR DECIL:")
   logger.info("-" * 60)
   logger.info(decile_summary)
   
   # Salvar resumo
   summary_file = os.path.join(OUTPUT_DIR, f"L24_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
   decile_summary.to_csv(summary_file)
   
   # Análise por UTM Campaign
   if 'UTM_CAMPAING' in results_df.columns:
       utm_summary = results_df.groupby('UTM_CAMPAING').agg({
           'probability_score': ['count', 'mean'],
           'decile': lambda x: (x <= 3).sum()  # Quantos no top 30%
       }).round(4)
       
       utm_summary.columns = ['total_leads', 'avg_probability', 'top_30_percent_count']
       utm_summary['top_30_percent_rate'] = (utm_summary['top_30_percent_count'] / utm_summary['total_leads'] * 100).round(2)
       utm_summary = utm_summary.sort_values('avg_probability', ascending=False)
       
       logger.info("\nTOP 10 CAMPANHAS POR PROBABILIDADE MÉDIA:")
       logger.info("-" * 80)
       logger.info(utm_summary.head(10))
       
       utm_file = os.path.join(OUTPUT_DIR, f"L24_utm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
       utm_summary.to_csv(utm_file)
   
   # Comparação com treino
   logger.info("\nCOMPARAÇÃO COM TREINO:")
   logger.info(f"  Probabilidade média TREINO: 0.1390")
   logger.info(f"  Probabilidade média PRODUÇÃO: {probabilities.mean():.4f}")
   logger.info(f"  Razão PROD/TREINO: {probabilities.mean()/0.1390:.2f}x")
   
   if probabilities.mean() > 0.2:
       logger.warning("  ⚠️  Distribuição significativamente diferente do treino!")
   
   return results_df


def main():
   """Função principal de execução."""
   start_time = datetime.now()
   
   logger.info("=" * 80)
   logger.info("INICIANDO INFERÊNCIA PARA PRODUÇÃO - LANÇAMENTO L24")
   logger.info("=" * 80)
   logger.info(f"Início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
   
   try:
       # 1. Carregar dados de produção
       utm_df, survey_df = load_production_data()
       
       # 2. Integrar dados
       integrated_df = integrate_data(utm_df, survey_df)
       
       # Guardar cópia do DataFrame original para o output final
       original_df = integrated_df.copy()
       
       # 3. Carregar parâmetros
       params = load_preprocessing_params()
       
       # 4. Aplicar pré-processamento (script 02)
       preprocessed_df = apply_preprocessing_pipeline(integrated_df, params)
       
       # 5. Aplicar features profissionais COMPLETAS (script 03)
       with_professional_df = apply_professional_features_complete(preprocessed_df, params)
       
       # 6. Carregar features selecionadas
       selected_features = load_selected_features()
       
       # 7. Validar e corrigir features
       final_df = validate_and_fix_features(with_professional_df, selected_features)
       
       # 8. Fazer predições
       probabilities, deciles = make_predictions(final_df, MODEL_PATH)
       
       # 9. Gerar outputs
       results_df = generate_output(original_df, probabilities, deciles)
       
       # Tempo total
       end_time = datetime.now()
       elapsed_time = (end_time - start_time).total_seconds()
       
       logger.info("\n" + "=" * 80)
       logger.info("INFERÊNCIA CONCLUÍDA COM SUCESSO!")
       logger.info("=" * 80)
       logger.info(f"Tempo total: {elapsed_time:.1f} segundos")
       logger.info(f"Total de predições: {len(results_df):,}")
       logger.info(f"Arquivos salvos em: {OUTPUT_DIR}")
       
       # Estatísticas finais
       logger.info("\nESTATÍSTICAS FINAIS:")
       logger.info(f"  Probabilidade média: {probabilities.mean():.4f}")
       logger.info(f"  Probabilidade mediana: {np.median(probabilities):.4f}")
       logger.info(f"  Probabilidade mínima: {probabilities.min():.4f}")
       logger.info(f"  Probabilidade máxima: {probabilities.max():.4f}")
       
       top_10_pct = (deciles == 1).sum()
       top_30_pct = (deciles <= 3).sum()
       logger.info(f"\n  Leads no decil 1 (top 10%): {top_10_pct:,} ({top_10_pct/len(results_df)*100:.1f}%)")
       logger.info(f"  Leads nos decis 1-3 (top 30%): {top_30_pct:,} ({top_30_pct/len(results_df)*100:.1f}%)")
       
   except Exception as e:
       logger.error(f"ERRO durante a execução: {str(e)}")
       logger.error("Detalhes do erro:", exc_info=True)
       raise
   

if __name__ == "__main__":
   main()