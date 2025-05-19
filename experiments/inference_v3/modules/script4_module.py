"""
Módulo para aplicar transformações equivalentes ao script 04_feature_engineering_2.py

Este módulo implementa features relacionadas à motivação profissional e aspirações de carreira,
usando técnicas como análise de sentimento, detecção de expressões de compromisso e termos de carreira.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import re
import warnings
import traceback
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Adicionar o diretório raiz do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

# Suprimir avisos
warnings.filterwarnings('ignore')

# Garantir que os recursos NLTK necessários estejam disponíveis
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print("Baixando recursos NLTK necessários...")
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)

def normalize_text(text):
    """
    Normaliza texto para processamento.
    
    Args:
        text: Texto para normalizar
        
    Returns:
        Texto normalizado ou string vazia se inválido
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Converter para minúsculas e normalizar espaços
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text

def enhance_tfidf_for_career_terms(df, text_columns, vectorizers, column_mapping=None):
    """
    Enhance TF-IDF weights for career-related terms.
    Versão modificada para a pipeline de inferência.
    
    Args:
        df: Input DataFrame
        text_columns: List of text columns to process
        vectorizers: Dictionary with existing vectorizers
        column_mapping: Dictionary mapping column names to vectorizer keys
    
    Returns:
        DataFrame with enhanced TF-IDF features
    """
    # Career-related terms to boost
    career_terms = [
        'trabajo', 'empleo', 'profesional', 'oportunidades', 'laboral',
        'carrera', 'mejor trabajo', 'oportunidades laborales', 'profesión',
        'mejor', 'comunicación', 'viajar', 'mejorar'
    ]
    
    if vectorizers is None:
        vectorizers = {}
    
    if column_mapping is None:
        column_mapping = {}
    
    # Dictionary to store enhanced TF-IDF features
    enhanced_tfidf_features = {}
    
    # Debug: mostrar as chaves disponíveis em vectorizers
    print(f"  Vetorizadores disponíveis: {list(vectorizers.keys())}")
    
    for col in text_columns:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # MODIFICAÇÃO: Tratamento especial para déjame_un_mensaje
        if 'déjame' in col_clean:
            print(f"  Processando features especiais para déjame_un_mensaje...")
            
            # Obter o vetorizador de déjame_un_mensaje
            déjame_key = 'déjame_un_mensaje'
            if 'career_tfidf' in vectorizers and déjame_key in vectorizers['career_tfidf']:
                vectorizer_info = vectorizers['career_tfidf'][déjame_key]
                feature_names = vectorizer_info['feature_names']
                
                print(f"    Vetorizador contém {len(feature_names)} termos")
                
                # Criar features para todos os termos do vetorizador
                clean_name = 'déjame_un_mensaje'  # Usar nome sem _original
                
                for term in feature_names:
                    feature_name = f"{clean_name}_tfidf_{term}"
                    
                    # Calcular valores baseados no texto (abordagem simplificada)
                    values = []
                    for text in df[col]:
                        text = normalize_text(text) if pd.notna(text) else ""
                        if term in text:
                            values.append(0.5)  # valor arbitrário para presença do termo
                        else:
                            values.append(0.0)
                    enhanced_tfidf_features[feature_name] = pd.Series(values)
                
                # Adicionar features de sentimento/motivação com zeros
                sentiment_features = [
                    'aspiration_count', 'aspiration_score', 
                    'career_term_count', 'career_term_score', 
                    'commitment_count', 'commitment_score', 
                    'has_career_terms', 'has_commitment', 
                    'sentiment_compound', 'sentiment_neg', 'sentiment_pos'
                ]
                
                for feature in sentiment_features:
                    feature_name = f"{clean_name}_{feature}"
                    enhanced_tfidf_features[feature_name] = pd.Series(np.zeros(len(df)))
                
                print(f"    Adicionadas {len(feature_names)} features TF-IDF específicas")
                print(f"    Adicionadas {len(sentiment_features)} features de sentimento/motivação")
            else:
                print(f"    AVISO: Vetorizador para déjame_un_mensaje não encontrado!")
            
            # Pular o processamento normal para esta coluna
            continue
        
        # Determinar qual vetorizador usar
        key_to_use = None
        
        # 1. Verificar mapeamento explícito
        if col in column_mapping and 'career_tfidf' in vectorizers and column_mapping[col] in vectorizers['career_tfidf']:
            key_to_use = column_mapping[col]
            print(f"  Aplicando TF-IDF para {col_clean} (usando vetorizador para '{key_to_use}')")
        else:
            # 2. Tentar correspondência por nome limpo
            clean_key = col_clean.replace('_original', '')
            if 'career_tfidf' in vectorizers and clean_key in vectorizers['career_tfidf']:
                key_to_use = clean_key
                print(f"  Aplicando TF-IDF para {col_clean} (correspondência por nome limpo)")
            # 3. Tentar correspondência parcial
            else:
                if 'career_tfidf' in vectorizers:
                    for key in vectorizers['career_tfidf'].keys():
                        if key in col_clean or col_clean in key:
                            key_to_use = key
                            print(f"  Aplicando TF-IDF para {col_clean} (correspondência parcial com '{key}')")
                            break
        
        # Skip if no vectorizer for this column
        if key_to_use is None:
            print(f"  Pulando TF-IDF para {col_clean} (nenhum vetorizador encontrado)")
            continue
            
        # Obter informações do vetorizador
        if 'career_tfidf' in vectorizers and key_to_use in vectorizers['career_tfidf']:
            vectorizer_info = vectorizers['career_tfidf'][key_to_use]
            
            if 'feature_names' in vectorizer_info:
                feature_names = vectorizer_info['feature_names']
                
                # Criar features TF-IDF simples (sem usar o vetorizador real)
                for term in feature_names:
                    feature_name = f"{col_clean.replace('_original', '')}_tfidf_{term}"
                    
                    values = []
                    for text in df[col]:
                        text = normalize_text(text) if pd.notna(text) else ""
                        if term in text:
                            # Valor arbitrário mas com boost para termos de carreira
                            if any(career_term in term for career_term in career_terms):
                                values.append(0.75)  # Boost de 50%
                            else:
                                values.append(0.5)
                        else:
                            values.append(0.0)
                    
                    enhanced_tfidf_features[feature_name] = pd.Series(values)
                
                print(f"    Geradas {len(feature_names)} features TF-IDF simplificadas")
            else:
                print(f"    AVISO: Nenhum feature_names encontrado para {key_to_use}")
        else:
            print(f"    AVISO: Vetorizador {key_to_use} não encontrado em career_tfidf")
    
    # Adicionar features TF-IDF especiais para qué_esperas_aprender_en_la
    qué_specials = ['los', 'mucho', 'pueda', 'todo lo']
    for term in qué_specials:
        feature_name = f"qué_esperas_aprender_en_la_tfidf_{term}"
        if feature_name not in enhanced_tfidf_features:
            enhanced_tfidf_features[feature_name] = pd.Series(np.zeros(len(df)))
    
    return pd.DataFrame(enhanced_tfidf_features, index=df.index)

def clean_column_name(col_name):
    """
    Limpa nomes de colunas para facilitar o manuseio.
    
    Args:
        col_name: Nome da coluna para limpar
        
    Returns:
        Nome de coluna limpo e curto
    """
    # Criar versão curta e limpa do nome da coluna
    col_clean = col_name.lower()
    col_clean = re.sub(r'[^\w\s]', '', col_clean)
    col_clean = re.sub(r'\s+', '_', col_clean)
    
    # Pegar as primeiras palavras para brevidade
    words = col_clean.split('_')
    short_name = '_'.join(words[:min(5, len(words))])
    
    return short_name[:30]  # Limitar comprimento

def create_professional_motivation_score(df, text_columns, work_keywords):
    """
    Cria um score agregado de motivação profissional baseado em palavras-chave.
    Versão modificada para a pipeline de inferência.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        work_keywords: Dicionário de palavras-chave com pesos
        
    Returns:
        DataFrame com features de motivação profissional adicionadas
    """
    result_df = pd.DataFrame(index=df.index)
    
    # Verificar se work_keywords está vazio
    if not work_keywords:
        # Usar valores padrão
        work_keywords = {
            'trabajo': 1.0, 'empleo': 1.0, 'carrera': 1.2, 'profesional': 1.2, 'puesto': 0.8,
            'laboral': 1.0, 'sueldo': 0.8, 'salario': 0.8, 'trabajar': 0.9, 'profesión': 1.2,
            'oportunidades': 1.0, 'mejor': 0.7, 'mejorar': 0.7
        }
    
    # Inicializar arrays para scores
    motivation_scores = np.zeros(len(df))
    keyword_counts = np.zeros(len(df))
    
    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df.columns:
            continue
            
        # Processar cada texto
        for idx, text in enumerate(df[col]):
            text = normalize_text(text)
            if not text:
                continue
                
            # Contar ocorrências ponderadas de palavras-chave
            col_score = 0
            col_count = 0
            
            for keyword, weight in work_keywords.items():
                occurrences = text.count(keyword)
                if occurrences > 0:
                    col_score += occurrences * weight
                    col_count += occurrences
            
            # Adicionar aos scores totais
            motivation_scores[idx] += col_score
            keyword_counts[idx] += col_count
    
    # Normalizar scores (0-1)
    max_score = np.max(motivation_scores) if np.max(motivation_scores) > 0 else 1
    normalized_scores = motivation_scores / max_score
    
    # Adicionar features ao DataFrame
    result_df['professional_motivation_score'] = normalized_scores
    result_df['career_keyword_count'] = keyword_counts
    
    return result_df

def analyze_aspiration_sentiment(df, text_columns, aspiration_phrases):
    """
    Implementa análise de sentimento específica para linguagem de aspiração.
    Versão modificada para a pipeline de inferência.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        aspiration_phrases: Lista de frases de aspiração para detectar
        
    Returns:
        DataFrame com features de sentimento de aspiração adicionadas
    """
    result_df = pd.DataFrame(index=df.index)
    
    # Verificar se aspiration_phrases está vazio
    if not aspiration_phrases:
        # Usar valores padrão
        aspiration_phrases = [
            'quiero ser', 'espero ser', 'mi meta es', 'mi objetivo es',
            'en el futuro', 'me veo', 'me visualizo'
        ]
    
    # Inicializar analisador de sentimento
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Erro ao inicializar analisador de sentimento: {e}")
        sia = None
    
    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Análise de sentimento padrão
        sentiment_scores = []
        for text in df[col]:
            text = normalize_text(text)
            if not text:
                sentiment_scores.append({"pos": 0, "neg": 0, "neu": 0, "compound": 0})
                continue
                
            if sia:
                try:
                    sentiment_scores.append(sia.polarity_scores(text))
                except:
                    sentiment_scores.append({"pos": 0, "neg": 0, "neu": 0, "compound": 0})
            else:
                sentiment_scores.append({"pos": 0, "neg": 0, "neu": 0, "compound": 0})
        
        # Extrair componentes
        result_df[f"{col_clean}_sentiment_pos"] = [score["pos"] for score in sentiment_scores]
        result_df[f"{col_clean}_sentiment_neg"] = [score["neg"] for score in sentiment_scores]
        result_df[f"{col_clean}_sentiment_compound"] = [score["compound"] for score in sentiment_scores]
        
        # Detecção de aspiração
        aspiration_counts = []
        for text in df[col]:
            text = normalize_text(text)
            count = sum(text.count(phrase) for phrase in aspiration_phrases)
            aspiration_counts.append(count)
        
        result_df[f"{col_clean}_aspiration_count"] = aspiration_counts
        
        # Score de aspiração: combinação de sentimento positivo e frases de aspiração
        result_df[f"{col_clean}_aspiration_score"] = np.array(aspiration_counts) * \
                                                  result_df[f"{col_clean}_sentiment_pos"]
    
    return result_df

def detect_commitment_expressions(df, text_columns, commitment_phrases):
    """
    Cria features para expressões de compromisso e determinação.
    Versão modificada para a pipeline de inferência.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        commitment_phrases: Dicionário de frases de compromisso com pesos
        
    Returns:
        DataFrame com features de expressões de compromisso adicionadas
    """
    result_df = pd.DataFrame(index=df.index)
    
    # Verificar se commitment_phrases está vazio
    if not commitment_phrases:
        # Usar valores padrão
        commitment_phrases = {
            'estoy decidido': 2.0,
            'estoy comprometido': 2.0,
            'me comprometo': 2.0,
            'quiero aprender': 1.0,
            'necesito aprender': 1.5,
            'debo aprender': 1.2
        }
    
    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Listas para features de compromisso
        commitment_scores = []
        has_commitment = []
        commitment_counts = []
        
        # Processar cada texto
        for text in df[col]:
            text = normalize_text(text)
            
            # Contar e pontuar frases de compromisso
            score = 0
            count = 0
            
            for phrase, weight in commitment_phrases.items():
                occurrences = text.count(phrase)
                if occurrences > 0:
                    score += occurrences * weight
                    count += occurrences
            
            commitment_scores.append(score)
            has_commitment.append(1 if count > 0 else 0)
            commitment_counts.append(count)
        
        # Adicionar features ao dicionário
        result_df[f"{col_clean}_commitment_score"] = commitment_scores
        result_df[f"{col_clean}_has_commitment"] = has_commitment
        result_df[f"{col_clean}_commitment_count"] = commitment_counts
    
    return result_df

def create_career_term_detector(df, text_columns, career_terms):
    """
    Cria um detector para termos relacionados à carreira com pesos de importância.
    Versão modificada para a pipeline de inferência.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        career_terms: Dicionário de termos relacionados à carreira com pesos
        
    Returns:
        DataFrame com features de termos de carreira adicionadas
    """
    result_df = pd.DataFrame(index=df.index)
    
    # Verificar se career_terms está vazio
    if not career_terms:
        # Usar valores padrão
        career_terms = {
            'crecimiento profesional': 2.0,
            'desarrollo profesional': 2.0,
            'oportunidades laborales': 2.0,
            'mejor salario': 1.8,
            'trabajo': 1.0,
            'empleo': 1.0,
            'profesional': 1.2,
            'oportunidades': 1.0,
            'laboral': 1.0,
            'carrera': 1.2
        }
    
    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Listas para features de termos de carreira
        career_scores = []
        has_career_terms = []
        career_term_counts = []
        
        # Processar cada texto
        for text in df[col]:
            text = normalize_text(text)
            
            # Contar e pontuar termos de carreira
            score = 0
            count = 0
            
            for term, weight in career_terms.items():
                occurrences = text.count(term)
                if occurrences > 0:
                    score += occurrences * weight
                    count += occurrences
            
            career_scores.append(score)
            has_career_terms.append(1 if count > 0 else 0)
            career_term_counts.append(count)
        
        # Adicionar features ao DataFrame
        result_df[f"{col_clean}_career_term_score"] = career_scores
        result_df[f"{col_clean}_has_career_terms"] = has_career_terms
        result_df[f"{col_clean}_career_term_count"] = career_term_counts
    
    return result_df

def process_dataset(df, params):
    """
    Aplica todo o pipeline do script 4 a um DataFrame.
    
    Args:
        df: DataFrame de entrada
        params: Dicionário com parâmetros pré-treinados
        
    Returns:
        DataFrame com features adicionadas
    """
    print("\nAplicando engenharia de features profissionais...")
    
    # Identificar colunas de texto a processar
    text_cols = [
        col for col in df.columns 
        if col.endswith('_original') and any(term in col for term in [
            'mensaje', 'inglés', 'vida', 'oportunidades', 'esperas', 'aprender', 
            'Semana', 'Inmersión', 'Déjame', 'fluidez'
        ])
    ]
    
    if not text_cols:
        # Tentar encontrar colunas de texto originais sem sufixo
        text_cols = [
            col for col in df.columns 
            if df[col].dtype == 'object' and any(term in col for term in [
                'mensaje', 'inglés', 'vida', 'oportunidades', 'esperas', 'aprender', 
                'Semana', 'Inmersión', 'Déjame', 'fluidez'
            ])
        ]
    
    if len(text_cols) == 0:
        print("  AVISO: Nenhuma coluna de texto encontrada para engenharia de features profissionais.")
        return df
    
    print(f"  Processando {len(text_cols)} colunas de texto.")
    
    # Extrair parâmetros necessários
    work_keywords = params.get('professional_motivation', {}).get('work_keywords', {})
    aspiration_phrases = params.get('aspiration_sentiment', {}).get('aspiration_phrases', [])
    commitment_phrases = params.get('commitment', {}).get('commitment_phrases', {})
    career_terms = params.get('career', {}).get('career_terms', {})
    vectorizers = params.get('vectorizers', {})
    
    # 1. Criar score de motivação profissional
    print("  1. Criando score de motivação profissional...")
    motivation_df = create_professional_motivation_score(df, text_cols, work_keywords)
    
    # 2. Aplicar TF-IDF para termos de carreira (ADICIONADO)
    print("  2. Aplicando TF-IDF para termos de carreira...")
    tfidf_df = enhance_tfidf_for_career_terms(df, text_cols, vectorizers)
    
    # 3. Analisar sentimento de aspiração
    print("  3. Analisando sentimento de aspiração...")
    aspiration_df = analyze_aspiration_sentiment(df, text_cols, aspiration_phrases)
    
    # 4. Detectar expressões de compromisso
    print("  4. Detectando expressões de compromisso...")
    commitment_df = detect_commitment_expressions(df, text_cols, commitment_phrases)
    
    # 5. Criar detector de termos de carreira
    print("  5. Criando detector de termos de carreira...")
    career_df = create_career_term_detector(df, text_cols, career_terms)
    
    # Combinar todas as features
    print("  Combinando todas as features...")
    dfs = [df, motivation_df, tfidf_df, aspiration_df, commitment_df, career_df]
    result_df = pd.concat(dfs, axis=1)
    
    # Remover colunas duplicadas
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    
    print(f"  Processamento completo: {result_df.shape[1] - df.shape[1]} novas features adicionadas.")
    
    return result_df

def apply_script4_transformations(df, params_path):
    """
    Função principal para aplicar transformações do script 4 (features de motivação profissional).
    Atualizada para usar os parâmetros do novo caminho.
    
    Args:
        df: DataFrame de entrada (output do script 3)
        params_path: Caminho base para o arquivo de parâmetros (não usado diretamente)
        
    Returns:
        DataFrame processado
    """
    print(f"\n=== Aplicando transformações do script 4 (features de motivação profissional) ===")
    
    # Definir o caminho correto para o arquivo de parâmetros
    project_root = "/Users/ramonmoreira/desktop/smart_ads"
    
    # ATUALIZAÇÃO: Novo caminho para os parâmetros
    motivation_params_path = os.path.join(project_root, "src/preprocessing/04_params/04_params.joblib")
    
    # Caminho alternativo (fallback)
    alt_params_path = os.path.join(project_root, "inference/params/04_params.joblib")
    
    # Verificar qual caminho usar
    if os.path.exists(motivation_params_path):
        print(f"Carregando parâmetros de: {motivation_params_path}")
        params_file = motivation_params_path
    elif os.path.exists(alt_params_path):
        print(f"Carregando parâmetros de: {alt_params_path}")
        params_file = alt_params_path
    else:
        print(f"AVISO: Arquivos de parâmetros não encontrados. Usando valores padrão.")
        params_file = None
    
    # Carregar parâmetros
    try:
        if params_file:
            params = joblib.load(params_file)
            print(f"Parâmetros carregados com sucesso.")
            
            # Verificar a estrutura dos parâmetros
            expected_keys = ['professional_motivation', 'aspiration_sentiment', 'commitment', 'career', 'vectorizers']
            missing_keys = [key for key in expected_keys if key not in params]
            
            if missing_keys:
                print(f"AVISO: Estrutura de parâmetros incompleta. Faltando: {missing_keys}")
        else:
            # Valores padrão básicos para fallback
            params = {
                'professional_motivation': {
                    'work_keywords': {
                        'trabajo': 1.0, 'empleo': 1.0, 'carrera': 1.2, 'profesional': 1.2,
                        'oportunidades': 1.0, 'mejor': 0.7, 'mejorar': 0.7
                    }
                },
                'aspiration_sentiment': {
                    'aspiration_phrases': [
                        'quiero ser', 'espero ser', 'mi meta es', 'mi objetivo es',
                        'en el futuro', 'me veo', 'me visualizo'
                    ]
                },
                'commitment': {
                    'commitment_phrases': {
                        'estoy decidido': 2.0, 'me comprometo': 2.0, 'quiero aprender': 1.0
                    }
                },
                'career': {
                    'career_terms': {
                        'crecimiento profesional': 2.0, 'desarrollo profesional': 2.0,
                        'oportunidades laborales': 2.0, 'mejor salario': 1.8
                    }
                },
                'vectorizers': {}
            }
    except Exception as e:
        print(f"ERRO ao carregar parâmetros: {e}")
        print(traceback.format_exc())
        
        # Valores padrão para fallback
        params = {
            'professional_motivation': {'work_keywords': {}},
            'aspiration_sentiment': {'aspiration_phrases': []},
            'commitment': {'commitment_phrases': {}},
            'career': {'career_terms': {}},
            'vectorizers': {}
        }
    
    # Extrair parâmetros específicos que serão usados nas funções
    work_keywords = params['professional_motivation'].get('work_keywords', {})
    aspiration_phrases = params['aspiration_sentiment'].get('aspiration_phrases', [])
    commitment_phrases = params['commitment'].get('commitment_phrases', {})
    career_terms = params['career'].get('career_terms', {})
    vectorizers = params.get('vectorizers', {})
    
    # Mostrar resumo dos parâmetros carregados
    print(f"Parâmetros carregados:")
    print(f"  work_keywords: {len(work_keywords)} termos")
    print(f"  aspiration_phrases: {len(aspiration_phrases)} frases")
    print(f"  commitment_phrases: {len(commitment_phrases)} frases")
    print(f"  career_terms: {len(career_terms)} termos")
    print(f"  vectorizers: {list(vectorizers.keys()) if vectorizers else 'Nenhum'}")
    
    # Identificar as colunas de texto originais
    text_cols = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?_original',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?_original', 
        'Déjame un mensaje_original',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?_original'
    ]
    
    # Verificar quais colunas existem no DataFrame
    existing_text_cols = [col for col in text_cols if col in df.columns]
    
    if len(existing_text_cols) == 0:
        print("AVISO: Nenhuma das colunas de texto esperadas foi encontrada.")
        # Tentar identificar colunas alternativas
        alternative_cols = [
            col for col in df.columns 
            if col.endswith('_original') and any(term in col for term in [
                'mensaje', 'inglés', 'vida', 'oportunidades', 'esperas', 'aprender', 
                'Semana', 'Inmersión', 'Déjame', 'fluidez'
            ])
        ]
        
        if alternative_cols:
            print(f"Encontradas {len(alternative_cols)} colunas de texto alternativas:")
            for col in alternative_cols:
                print(f"  - {col}")
            text_cols = alternative_cols
        else:
            print("ERRO: Não foi possível encontrar colunas de texto para processamento.")
            return df
    else:
        print(f"Encontradas {len(existing_text_cols)} das {len(text_cols)} colunas de texto esperadas.")
        text_cols = existing_text_cols
    
    print("\nAplicando engenharia de features profissionais...")
    print(f"  Processando {len(text_cols)} colunas de texto.")
    
    # Mapeamento preciso para vetorizadores
    column_to_key_map = {
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?_original': 'cuando_hables_inglés_con_fluid',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?_original': 'qué_esperas_aprender_en_la',
        'Déjame un mensaje_original': 'déjame_un_mensaje',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?_original': 'qué_esperas_aprender_en_la'
    }
    
    # 1. Criar score de motivação profissional
    print("  1. Criando score de motivação profissional...")
    try:
        motivation_df = create_professional_motivation_score(df, text_cols, work_keywords)
    except Exception as e:
        print(f"ERRO ao criar score de motivação profissional: {e}")
        print(traceback.format_exc())
        motivation_df = pd.DataFrame(index=df.index)
    
    # 2. Aplicar TF-IDF para termos de carreira
    print("  2. Aplicando TF-IDF para termos de carreira...")
    try:
        tfidf_df = enhance_tfidf_for_career_terms(df, text_cols, vectorizers, column_to_key_map)
    except Exception as e:
        print(f"ERRO ao aplicar TF-IDF: {e}")
        print(traceback.format_exc())
        tfidf_df = pd.DataFrame(index=df.index)
    
    # 3. Analisar sentimento de aspiração
    print("  3. Analisando sentimento de aspiração...")
    try:
        aspiration_df = analyze_aspiration_sentiment(df, text_cols, aspiration_phrases)
    except Exception as e:
        print(f"ERRO ao analisar sentimento de aspiração: {e}")
        print(traceback.format_exc())
        aspiration_df = pd.DataFrame(index=df.index)
    
    # 4. Detectar expressões de compromisso
    print("  4. Detectando expressões de compromisso...")
    try:
        commitment_df = detect_commitment_expressions(df, text_cols, commitment_phrases)
    except Exception as e:
        print(f"ERRO ao detectar expressões de compromisso: {e}")
        print(traceback.format_exc())
        commitment_df = pd.DataFrame(index=df.index)
    
    # 5. Criar detector de termos de carreira
    print("  5. Criando detector de termos de carreira...")
    try:
        career_df = create_career_term_detector(df, text_cols, career_terms)
    except Exception as e:
        print(f"ERRO ao criar detector de termos de carreira: {e}")
        print(traceback.format_exc())
        career_df = pd.DataFrame(index=df.index)
    
    # Combinar todas as features
    print("  Combinando todas as features...")
    dfs = [df, motivation_df, tfidf_df, aspiration_df, commitment_df, career_df]
    result_df = pd.concat(dfs, axis=1)
    
    # Remover colunas duplicadas
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    
    print(f"  Processamento completo: {result_df.shape[1] - df.shape[1]} novas features adicionadas.")
    
    # Alinhar com as colunas do treinamento
    train_cols_path = os.path.join(project_root, "inference/params/04_train_columns.csv")
    
    # Buscar em locais alternativos
    if not os.path.exists(train_cols_path):
        alt_train_cols_path = os.path.join(project_root, "reports/04_train_columns.csv")
        if os.path.exists(alt_train_cols_path):
            train_cols_path = alt_train_cols_path
    
    if os.path.exists(train_cols_path):
        try:
            print(f"Usando arquivo de colunas do estágio 4: {train_cols_path}")
            
            # Ler colunas de treinamento
            train_columns_df = pd.read_csv(train_cols_path)
            
            # Determinar o formato do arquivo
            if 'column_name' in train_columns_df.columns:
                # Formato coluna/valor
                train_columns = train_columns_df['column_name'].tolist()
            else:
                # Formato em que os nomes de colunas são as próprias colunas do CSV
                train_columns = train_columns_df.columns.tolist()
                
                # Se o primeiro valor parece ser um índice, use a primeira linha
                if train_columns_df.shape[0] > 0 and len(train_columns) > 0 and train_columns[0].isdigit():
                    train_columns = train_columns_df.iloc[0].tolist()
            
            print(f"Carregadas {len(train_columns)} colunas do dataset de treino (estágio 4).")
            
            # Eliminar None ou NaN da lista
            train_columns = [col for col in train_columns if col and pd.notna(col)]
            
            # Identificar colunas extras e faltantes
            extra_cols = set(result_df.columns) - set(train_columns)
            missing_cols = set(train_columns) - set(result_df.columns)
            
            # Remover a coluna target da comparação
            if 'target' in missing_cols:
                missing_cols.remove('target')
            
            if extra_cols:
                print(f"Removendo {len(extra_cols)} colunas extras não presentes no treino:")
                
                # Agrupar por prefixo para melhor diagnóstico
                prefixes = {}
                for col in extra_cols:
                    prefix = col.split('_')[0] if '_' in col else col
                    if prefix not in prefixes:
                        prefixes[prefix] = []
                    prefixes[prefix].append(col)
                
                for prefix, cols in prefixes.items():
                    print(f"  Grupo '{prefix}': {len(cols)} colunas")
                    if len(cols) > 0:
                        print(f"    Exemplos: {cols[:min(3, len(cols))]}")
                
                # Salvar detalhes para análise
                reports_dir = os.path.join(project_root, "reports")
                os.makedirs(reports_dir, exist_ok=True)
                with open(os.path.join(reports_dir, 'extra_columns.txt'), 'w') as f:
                    for col in sorted(extra_cols):
                        f.write(f"{col}\n")
                
                # Remover colunas extras
                result_df = result_df.drop(columns=list(extra_cols))
            
            if missing_cols:
                print(f"AVISO: Adicionando {len(missing_cols)} colunas faltantes com valores zero.")
                
                # Agrupar por prefixo para melhor diagnóstico
                prefixes = {}
                for col in missing_cols:
                    prefix = col.split('_')[0] if '_' in col else col
                    if prefix not in prefixes:
                        prefixes[prefix] = []
                    prefixes[prefix].append(col)
                
                for prefix, cols in prefixes.items():
                    print(f"  Grupo '{prefix}': {len(cols)} colunas")
                    if len(cols) > 0:
                        print(f"    Exemplos: {cols[:min(3, len(cols))]}")
                
                # Salvar detalhes para análise
                with open(os.path.join(reports_dir, 'missing_columns.txt'), 'w') as f:
                    for col in sorted(missing_cols):
                        f.write(f"{col}\n")
                
                # Adicionar colunas faltantes
                for col in missing_cols:
                    result_df[col] = 0.0
            
            # Garantir a mesma ordem das colunas
            train_cols_without_target = [col for col in train_columns if col != 'target']
            result_df = result_df[train_cols_without_target]
            
            print(f"Dataset alinhado com o treino (estágio 4): {result_df.shape}")
        except Exception as e:
            print(f"ERRO ao alinhar colunas com estágio 4: {e}")
            print(traceback.format_exc())
    else:
        print("AVISO: Arquivo de colunas do estágio 4 não encontrado.")
    
    print(f"Transformações do script 4 concluídas. Dimensões finais: {result_df.shape}")
    
    return result_df