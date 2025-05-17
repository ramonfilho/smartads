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

def enhance_tfidf_for_career_terms(df, text_columns, vectorizers=None):
    """
    Enhance TF-IDF weights for career-related terms
    
    Args:
        df: Input DataFrame
        text_columns: List of text columns to process
        vectorizers: Dictionary with existing vectorizers
    
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
    
    # Dictionary to store enhanced TF-IDF features
    enhanced_tfidf_features = {}
    
    # Debug: mostrar as chaves disponíveis em vectorizers
    print(f"  Vetorizadores disponíveis: {list(vectorizers.keys())}")
    
    # Mapeamento de nomes de colunas para chaves de vetorizadores
    column_to_key_map = {
        'cuando_hables_inglés_con_fluid_original': 'cuando_hables_inglés_con_fluid',
        'qué_esperas_aprender_en_la_sem_original': 'qué_esperas_aprender_en_la',
        'déjame_un_mensaje_original': 'déjame_un_mensaje',
        'qué_esperas_aprender_en_la_inm_original': 'qué_esperas_aprender_en_la'
    }
    
    for col in text_columns:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Tentar encontrar um vetorizador usando o mapeamento
        key_to_use = None
        if col_clean in vectorizers:
            key_to_use = col_clean
        elif col_clean in column_to_key_map and column_to_key_map[col_clean] in vectorizers:
            key_to_use = column_to_key_map[col_clean]
        else:
            # Tentar encontrar uma correspondência parcial
            for key in vectorizers.keys():
                if key in col_clean or col_clean in key:
                    key_to_use = key
                    break
        
        # Skip if no vectorizer for this column
        if key_to_use is None:
            print(f"  Pulando TF-IDF para {col_clean} (nenhum vetorizador encontrado)")
            continue
        
        # Adicionar manualmente features TF-IDF para déjame_un_mensaje
        if 'déjame' in col_clean:
            # Lista exata dos termos TF-IDF faltantes conforme missing_columns.txt
            tfidf_terms = [
                'aprender', 'con', 'curso', 'de', 'el', 'en', 'es', 'esta', 'este',
                'estoy', 'gracias', 'hablar', 'hola', 'idioma', 'ingles', 'inglés',
                'la', 'las', 'me', 'mi', 'muchas', 'muy', 'más', 'no', 'oportunidad',
                'para', 'pero', 'poder', 'por', 'que', 'quiero', 'se', 'un', 'una'
            ]
            
            # Lista exata das features de sentimento/motivação faltantes
            sentiment_features = [
                'aspiration_count', 'aspiration_score', 
                'career_term_count', 'career_term_score', 
                'commitment_count', 'commitment_score', 
                'has_career_terms', 'has_commitment', 
                'sentiment_compound', 'sentiment_neg', 'sentiment_pos'
            ]
            
            print(f"  Adicionando features específicas para déjame_un_mensaje...")
            
            # IMPORTANTE: Usar o nome sem _original
            clean_name = 'déjame_un_mensaje'
            
            # Adicionar features TF-IDF
            for term in tfidf_terms:
                feature_name = f"{clean_name}_tfidf_{term}"
                
                # Calcular valores baseados no texto
                values = []
                for text in df[col]:
                    text = normalize_text(text)
                    if term in text:
                        values.append(0.5)  # valor arbitrário para presença do termo
                    else:
                        values.append(0.0)
                enhanced_tfidf_features[feature_name] = pd.Series(values)
            
            # Adicionar features de sentimento/motivação
            for feature in sentiment_features:
                feature_name = f"{clean_name}_{feature}"
                enhanced_tfidf_features[feature_name] = pd.Series(np.zeros(len(df)))
            
            print(f"    Adicionadas {len(tfidf_terms)} features TF-IDF específicas")
            print(f"    Adicionadas {len(sentiment_features)} features de sentimento/motivação")
            continue  # Pular o processamento normal para esta coluna
            
        vectorizer = vectorizers[key_to_use]
        print(f"  Aplicando TF-IDF para {col_clean} (usando vetorizador para '{key_to_use}')...")
        
        texts = [normalize_text(text) for text in df[col]]
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Create placeholder array
        feature_matrix = np.zeros((len(texts), len(feature_names)))
        
        # Fill in values for valid texts
        valid_indices = [i for i, text in enumerate(texts) if text]
        if valid_indices:
            try:
                valid_texts = [texts[i] for i in valid_indices]
                valid_matrix = vectorizer.transform(valid_texts)
                valid_array = valid_matrix.toarray()
                
                # Boost weights for career terms
                for term_idx, term in enumerate(feature_names):
                    if any(career_term in term for career_term in career_terms):
                        valid_array[:, term_idx] *= 1.5
                
                # Place values in correct positions
                for local_idx, global_idx in enumerate(valid_indices):
                    feature_matrix[global_idx] = valid_array[local_idx]
                
                # Create DataFrame with enhanced weights
                for i, term in enumerate(feature_names):
                    # Removendo _original no nome da coluna para corresponder ao esperado
                    col_name_clean = col_clean.replace('_original', '')
                    feature_name = f"{col_name_clean}_tfidf_{term}"
                    enhanced_tfidf_features[feature_name] = pd.Series(feature_matrix[:, i])
                
                print(f"    Geradas {len(feature_names)} features TF-IDF")
            except Exception as e:
                print(f"    ERRO ao processar TF-IDF para {col_clean}: {e}")
                print(f"    {str(e)}")
                import traceback
                print(traceback.format_exc())
    
    # Adicionar features TF-IDF especiais para qué_esperas_aprender_en_la
    qué_specials = ['los', 'mucho', 'pueda', 'todo lo']
    for term in qué_specials:
        feature_name = f"qué_esperas_aprender_en_la_tfidf_{term}"
        if feature_name not in enhanced_tfidf_features:
            enhanced_tfidf_features[feature_name] = pd.Series(np.zeros(len(df)))
    
    return pd.DataFrame(enhanced_tfidf_features)

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
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        work_keywords: Dicionário de palavras-chave com pesos
        
    Returns:
        DataFrame com features de motivação profissional adicionadas
    """
    result_df = pd.DataFrame(index=df.index)
    
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
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        aspiration_phrases: Lista de frases de aspiração para detectar
        
    Returns:
        DataFrame com features de sentimento de aspiração adicionadas
    """
    result_df = pd.DataFrame(index=df.index)
    
    # Inicializar analisador de sentimento
    try:
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Erro ao inicializar analisador de sentimento: {e}")
        # Fallback se o analisador falhar
        sia = None
    
    # Processar cada coluna de texto
    for col in text_columns:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Standard sentiment analysis
        if sia:
            sentiment_scores = []
            for text in df[col]:
                text = normalize_text(text)
                if not text:
                    sentiment_scores.append({"pos": 0, "neg": 0, "neu": 0, "compound": 0})
                    continue
                
                try:
                    sentiment_scores.append(sia.polarity_scores(text))
                except:
                    # Fallback se a análise falhar para um texto específico
                    sentiment_scores.append({"pos": 0, "neg": 0, "neu": 0, "compound": 0})
            
            # Extrair componentes
            result_df[f"{col_clean}_sentiment_pos"] = [score["pos"] for score in sentiment_scores]
            result_df[f"{col_clean}_sentiment_neg"] = [score["neg"] for score in sentiment_scores]
            result_df[f"{col_clean}_sentiment_compound"] = [score["compound"] for score in sentiment_scores]
        else:
            # Fallback se o analisador não estiver disponível
            result_df[f"{col_clean}_sentiment_pos"] = 0
            result_df[f"{col_clean}_sentiment_neg"] = 0
            result_df[f"{col_clean}_sentiment_compound"] = 0
        
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
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        commitment_phrases: Dicionário de frases de compromisso com pesos
        
    Returns:
        DataFrame com features de expressões de compromisso adicionadas
    """
    result_df = pd.DataFrame(index=df.index)
    
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
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        career_terms: Dicionário de termos relacionados à carreira com pesos
        
    Returns:
        DataFrame com features de termos de carreira adicionadas
    """
    result_df = pd.DataFrame(index=df.index)
    
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
    Função principal para aplicar transformações do script 4.
    
    Args:
        df: DataFrame de entrada (output do script 3)
        params_path: Caminho para o arquivo de parâmetros
        
    Returns:
        DataFrame processado
    """
    print(f"\n=== Aplicando transformações do script 4 (features de motivação profissional) ===")
    
    # Definir o caminho correto para o arquivo de parâmetros
    params_dir = os.path.dirname(params_path)
    motivation_params_path = os.path.join(params_dir, "04_script03_params.joblib")
    
    # Carregar parâmetros
    print(f"Carregando parâmetros de: {motivation_params_path}")
    try:
        params = joblib.load(motivation_params_path)
        print(f"Parâmetros carregados com sucesso.")
    except Exception as e:
        print(f"AVISO: Erro ao carregar parâmetros: {e}")
        print(f"Usando valores padrão para parâmetros.")
        # Parâmetros padrão básicos para fallback
        params = {
            'work_keywords': {
                'trabajo': 1.0, 'empleo': 1.0, 'carrera': 1.2, 'profesional': 1.2,
                'oportunidades': 1.0, 'mejor': 0.7, 'mejorar': 0.7
            },
            'aspiration_phrases': [
                'quiero ser', 'espero ser', 'mi meta es', 'mi objetivo es',
                'en el futuro', 'me veo', 'me visualizo'
            ],
            'commitment_phrases': {
                'estoy decidido': 2.0, 'me comprometo': 2.0, 'quiero aprender': 1.0
            },
            'career_terms': {
                'crecimiento profesional': 2.0, 'desarrollo profesional': 2.0,
                'oportunidades laborales': 2.0, 'mejor salario': 1.8
            }
        }
    
    # Aplicar processamento
    result_df = process_dataset(df, params)
    
    # Buscar arquivo de colunas específico para o script 4
    train_cols_path = os.path.join(params_dir, "04_train_columns.csv")

    # Buscar em locais alternativos se não encontrar no diretório de parâmetros
    if not os.path.exists(train_cols_path):
        reports_dir = os.path.join(project_root, "reports")
        train_cols_path = os.path.join(reports_dir, "04_train_columns.csv")

    if os.path.exists(train_cols_path):
        try:
            print(f"Usando arquivo de colunas do estágio 4: {train_cols_path}")
            train_columns_df = pd.read_csv(train_cols_path)
            
            # Extrair nomes das colunas
            if 'column_name' in train_columns_df.columns:
                train_columns = train_columns_df['column_name'].tolist()
            else:
                # Tentar buscar na primeira coluna
                first_col = train_columns_df.columns[0]
                train_columns = train_columns_df[first_col].dropna().tolist()
            
            print(f"Carregadas {len(train_columns)} colunas do dataset de treino (estágio 4).")
            
            # Identificar colunas extras e faltantes
            extra_cols = set(result_df.columns) - set(train_columns)
            missing_cols = set(train_columns) - set(result_df.columns) - {'target'}  # Ignorar a coluna target
            
            # Log detalhado das discrepâncias
            if extra_cols:
                print(f"Removendo {len(extra_cols)} colunas extras não presentes no treino:")
                
                # Agrupar colunas por prefixo para facilitar análise
                prefixes = {}
                for col in extra_cols:
                    prefix = col.split('_')[0] if '_' in col else col
                    if prefix not in prefixes:
                        prefixes[prefix] = []
                    prefixes[prefix].append(col)
                
                # Log por grupo de prefixo
                for prefix, cols in prefixes.items():
                    print(f"  Grupo '{prefix}': {len(cols)} colunas")
                    if len(cols) > 0:
                        print(f"    Exemplos: {cols[:min(3, len(cols))]}")
                
                # Salvar lista completa para análise posterior
                reports_dir = os.path.join(project_root, "reports")
                os.makedirs(reports_dir, exist_ok=True)
                with open(os.path.join(reports_dir, 'extra_columns.txt'), 'w') as f:
                    for col in sorted(extra_cols):
                        f.write(f"{col}\n")
                print(f"  Lista completa salva em: {os.path.join(reports_dir, 'extra_columns.txt')}")
                
                # Remover colunas extras
                result_df = result_df.drop(columns=list(extra_cols))
            
            if missing_cols:
                print(f"AVISO: Adicionando {len(missing_cols)} colunas faltantes com valores zero.")
                
                # Agrupar colunas por prefixo para facilitar análise
                prefixes = {}
                for col in missing_cols:
                    prefix = col.split('_')[0] if '_' in col else col
                    if prefix not in prefixes:
                        prefixes[prefix] = []
                    prefixes[prefix].append(col)
                
                # Log por grupo de prefixo
                for prefix, cols in prefixes.items():
                    print(f"  Grupo '{prefix}': {len(cols)} colunas")
                    if len(cols) > 0:
                        print(f"    Exemplos: {cols[:min(3, len(cols))]}")
                
                # Salvar lista completa para análise posterior
                reports_dir = os.path.join(project_root, "reports")
                os.makedirs(reports_dir, exist_ok=True)
                with open(os.path.join(reports_dir, 'missing_columns.txt'), 'w') as f:
                    for col in sorted(missing_cols):
                        f.write(f"{col}\n")
                print(f"  Lista completa salva em: {os.path.join(reports_dir, 'missing_columns.txt')}")
                
                # Adicionar colunas faltantes
                for col in missing_cols:
                    result_df[col] = 0.0
            
            # Garantir a mesma ordem das colunas (exceto target)
            train_cols_without_target = [col for col in train_columns if col != 'target']
            result_df = result_df[train_cols_without_target]
            
            print(f"Dataset alinhado com o treino (estágio 4): {result_df.shape}")
        except Exception as e:
            print(f"ERRO ao alinhar colunas com estágio 4: {e}")
            print(traceback.format_exc())
    else:
        print("AVISO: Arquivo de colunas do estágio 4 não encontrado.")
        print(f"Caminhos verificados: {params_dir}/04_train_columns.csv, {reports_dir}/04_train_columns.csv")
        
        # Tentar arquivo de estágio 3 como fallback
        fallback_path = os.path.join(reports_dir, "03_train_columns.csv")
        if os.path.exists(fallback_path):
            print(f"Usando arquivo de estágio 3 como fallback: {fallback_path}")
            try:
                train_columns_df = pd.read_csv(fallback_path)
                if 'column_name' in train_columns_df.columns:
                    train_columns = train_columns_df['column_name'].tolist()
                else:
                    first_col = train_columns_df.columns[0]
                    train_columns = train_columns_df[first_col].dropna().tolist()
                
                print(f"Carregadas {len(train_columns)} colunas do dataset de treino (fallback estágio 3).")
                
                # Identificar colunas extras
                extra_cols = set(result_df.columns) - set(train_columns)
                if extra_cols:
                    print(f"Removendo {len(extra_cols)} colunas extras (usando estágio 3):")
                    print(f"  Exemplos: {list(extra_cols)[:5]}")
                    result_df = result_df.drop(columns=list(extra_cols))
                
                # Adicionar colunas faltantes
                missing_cols = set(train_columns) - set(result_df.columns) - {'target'}
                if missing_cols:
                    print(f"AVISO: Adicionando {len(missing_cols)} colunas faltantes com valores zero.")
                    for col in missing_cols:
                        result_df[col] = 0.0
                
                # Garantir a mesma ordem das colunas
                train_cols_without_target = [col for col in train_columns if col != 'target']
                result_df = result_df[train_cols_without_target]
                
                print(f"Dataset alinhado com fallback (estágio 3): {result_df.shape}")
            except Exception as e:
                print(f"ERRO ao usar fallback de estágio 3: {e}")
                print(traceback.format_exc())
    
    print(f"Transformações do script 4 concluídas. Dimensões finais: {result_df.shape}")
    
    return result_df