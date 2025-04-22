#!/usr/bin/env python
"""
Advanced Text Feature Engineering Script
Focus: Work Motivation and Career Aspirations

This script enhances a dataset with advanced text features related to:
1. Professional motivation scoring
2. Enhanced TF-IDF for career-related terms
3. Aspiration sentiment analysis
4. Commitment expression detection
5. Career-focused term detection

Input: Two datasets (with and without basic text processing)
Output: Enhanced dataset with new text features
"""

import os
import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

# Define paths
INPUT_DIR_BASIC = "/Users/ramonmoreira/desktop/smart_ads/data/02_processed"
INPUT_DIR_TEXT = "/Users/ramonmoreira/desktop/smart_ads/data/02_processed_text"
OUTPUT_DIR = "/Users/ramonmoreira/desktop/smart_ads/data/02_processed_text_v2"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define text columns to process
TEXT_COLUMNS = [
    'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
    '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?', 
    'Déjame un mensaje',
    '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
]

def clean_column_name(col_name):
    """
    Clean column names for easier handling
    """
    # Create a short, clean version of column name
    col_clean = col_name.lower()
    col_clean = re.sub(r'[^\w\s]', '', col_clean)
    col_clean = re.sub(r'\s+', '_', col_clean)
    
    # Take first few words for brevity
    words = col_clean.split('_')
    short_name = '_'.join(words[:min(5, len(words))])
    
    return short_name[:30]  # Limit length

def normalize_text(text):
    """
    Basic text normalization
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Lowercase and normalize whitespace
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text

def get_vader_sentiment_scores(text, sia):
    """
    Get sentiment scores using VADER SentimentIntensityAnalyzer
    """
    if not text:
        return {"pos": 0, "neg": 0, "neu": 0, "compound": 0}
    
    return sia.polarity_scores(text)

def create_professional_motivation_score(df, text_columns):
    """
    Create an aggregate professional motivation score based on career-related keywords
    """
    # Professional and career-related keywords in Spanish
    work_keywords = {
        # Termos básicos de trabalho/emprego
        'trabajo': 1.0, 'empleo': 1.0, 'carrera': 1.2, 'profesional': 1.2, 'puesto': 0.8,
        'laboral': 1.0, 'sueldo': 0.8, 'salario': 0.8, 'trabajar': 0.9, 'profesión': 1.2,
        'ocupación': 0.9, 'oficio': 0.8, 'cargo': 1.0, 'posición': 0.8, 'rol': 0.7,
        'empleador': 0.7, 'compañía': 0.6, 'empresa': 0.7, 'industria': 0.6, 'sector': 0.6,
        'contrato': 0.7, 'ingresos': 0.9, 'remuneración': 0.9, 'ganancias': 0.8,
        
        # Termos de busca/mudança de trabalho
        'curriculum': 1.0, 'cv': 0.9, 'entrevista': 1.1, 'aplicar': 0.8, 'solicitar': 0.8,
        'contratación': 0.9, 'reclutamiento': 0.8, 'oferta laboral': 1.0, 'vacante': 0.9,
        'búsqueda de empleo': 1.1, 'cambio de trabajo': 1.2, 'nuevo trabajo': 1.2,
        'despido': 0.7, 'renuncia': 0.7, 'desempleado': 0.7, 'experiencia laboral': 1.0,
        
        # Termos de oportunidade
        'oportunidades': 1.0, 'oportunidad': 1.0, 'ascenso': 1.5, 'promoción': 1.5,
        'crecimiento': 1.2, 'mejorar': 0.7, 'avanzar': 0.9, 'progreso': 1.0, 'éxito': 0.8,
        'logro': 0.9, 'meta profesional': 1.3, 'objetivo laboral': 1.3, 'aspiración': 1.1,
        'potencial': 0.9, 'desarrollo': 1.1, 'avance': 1.0, 'evolución': 0.9, 'prosperar': 0.9,
        'ambición': 1.0, 'superación': 1.1, 'posibilidades': 0.8, 'proyección': 0.9,
        
        # Termos de competências
        'habilidades': 0.8, 'competencias': 0.9, 'capacidades': 0.8, 'aptitudes': 0.8,
        'cualificación': 1.0, 'formación': 0.8, 'especialización': 1.0, 'conocimientos': 0.7,
        'experiencia': 0.9, 'experto': 1.0, 'certificación': 0.9, 'título': 0.8,
        'educación': 0.7, 'aprendizaje': 0.7, 'capacitación': 0.9, 'entrenamiento': 0.8,
        
        # Termos de networking/mercado
        'contactos': 0.8, 'networking': 1.0, 'conexiones': 0.8, 'mercado laboral': 1.1,
        'competencia': 0.8, 'ventaja competitiva': 1.2, 'perfil profesional': 1.1,
        'marca personal': 1.0, 'reputación': 0.9, 'profesionalismo': 1.0,
        'presencia online': 0.8, 'linkedin': 0.9, 'red profesional': 1.0,
        
        # Termos internacionais
        'global': 0.9, 'internacional': 1.0, 'extranjero': 0.9, 'expatriado': 1.1,
        'multinacional': 0.9, 'traslado': 0.8, 'reubicación': 0.9, 'viajar': 0.7,
        'idioma inglés trabajo': 1.4, 'inglés laboral': 1.4, 'negocios internacionales': 1.2,
        'comunicación empresarial': 1.0, 'entorno multicultural': 1.0,
        
        # Combinações mais fortes (com pesos mais altos)
        'mejor trabajo': 1.8, 'oportunidades laborales': 2.0, 'crecimiento profesional': 1.8,
        'desarrollo profesional': 1.8, 'mejorar profesionalmente': 1.7, 'futuro profesional': 1.7,
        'carrera internacional': 1.9, 'éxito profesional': 1.7, 'mercado laboral global': 1.8,
        'proyección internacional': 1.7, 'perfil competitivo': 1.6, 'destacar profesionalmente': 1.7,
        'mejor posición': 1.6, 'cargo superior': 1.7, 'mejor remuneración': 1.8,
        'mejores oportunidades': 1.8, 'trabajo en el extranjero': 1.9, 'avance profesional': 1.7,
        'posición de liderazgo': 1.7, 'ascenso laboral': 1.8, 'mejor calidad de vida': 1.5,
        'estabilidad laboral': 1.6, 'satisfacción profesional': 1.6, 'equilibrio trabajo-vida': 1.5,
        'negociar salario': 1.6, 'aumento de sueldo': 1.7, 'empleo bien remunerado': 1.8,
        
        # Frases específicas sobre inglês e trabalho (extremamente relevantes)
        'inglés para trabajo': 2.0, 'inglés profesional': 2.0, 'inglés negocios': 2.0,
        'inglés laboral': 2.0, 'comunicación profesional en inglés': 2.1,
        'entrevista en inglés': 2.0, 'reuniones en inglés': 1.9, 'presentaciones en inglés': 1.9,
        'correos en inglés': 1.8, 'llamadas en inglés': 1.9, 'negociaciones en inglés': 2.0,
        'requisito inglés': 2.1, 'inglés fluido trabajo': 2.1, 'inglés nivel profesional': 2.0,
        'inglés comercial': 1.9, 'inglés corporativo': 1.9, 'inglés técnico': 1.8
    }
    
    # Initialize score array
    motivation_scores = np.zeros(len(df))
    keyword_counts = np.zeros(len(df))
    
    # Process each text column
    for col in text_columns:
        if col not in df.columns:
            continue
            
        # Process each text entry
        for idx, text in enumerate(df[col]):
            text = normalize_text(text)
            if not text:
                continue
                
            # Count weighted occurrences of career keywords
            col_score = 0
            col_count = 0
            
            for keyword, weight in work_keywords.items():
                occurrences = text.count(keyword)
                if occurrences > 0:
                    col_score += occurrences * weight
                    col_count += occurrences
            
            # Add to total scores
            motivation_scores[idx] += col_score
            keyword_counts[idx] += col_count
    
    # Normalize scores (0-1 range)
    max_score = np.max(motivation_scores) if np.max(motivation_scores) > 0 else 1
    normalized_scores = motivation_scores / max_score
    
    return pd.DataFrame({
        'professional_motivation_score': normalized_scores,
        'career_keyword_count': keyword_counts
    })

def enhance_tfidf_for_career_terms(df, text_columns, fit_mode=True, vectorizers=None):
    """
    Enhance TF-IDF weights for career-related terms
    """
    # Career-related terms to boost
    career_terms = [
        'trabajo', 'empleo', 'profesional', 'oportunidades', 'laboral',
        'carrera', 'mejor trabajo', 'oportunidades laborales', 'profesión'
    ]
    
    if vectorizers is None:
        vectorizers = {}
    
    # Dictionary to store enhanced TF-IDF features
    enhanced_tfidf_features = {}
    
    for col in text_columns:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        texts = df[col].apply(normalize_text)
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text]
        
        if len(valid_texts) < 10:  # Skip if too few valid texts
            print(f"Skipping TF-IDF for {col_clean} (too few valid texts)")
            continue
        
        if fit_mode:
            # Create and fit vectorizer
            vectorizer = TfidfVectorizer(
                max_features=50,
                min_df=3,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Store vectorizer for future use
            vectorizers[col_clean] = vectorizer
            
            # Convert to array for manipulation
            tfidf_array = tfidf_matrix.toarray()
            
            # Boost weights for career terms
            for term_idx, term in enumerate(feature_names):
                if any(career_term in term for career_term in career_terms):
                    # Boost career-related terms by 50%
                    tfidf_array[:, term_idx] *= 1.5
            
            # Create DataFrame with enhanced weights
            for i, term in enumerate(feature_names):
                feature_name = f"{col_clean}_tfidf_{term}"
                enhanced_tfidf_features[feature_name] = pd.Series(
                    [tfidf_array[valid_texts.index(text), i] if text in valid_texts else 0 
                     for text in texts]
                )
        
        elif col_clean in vectorizers:
            # Transform using existing vectorizer
            vectorizer = vectorizers[col_clean]
            feature_names = vectorizer.get_feature_names_out()
            
            # Create placeholder array
            feature_matrix = np.zeros((len(texts), len(feature_names)))
            
            # Fill in values for valid texts
            valid_indices = [i for i, text in enumerate(texts) if text]
            if valid_indices:
                valid_matrix = vectorizer.transform([texts[i] for i in valid_indices])
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
                feature_name = f"{col_clean}_tfidf_{term}"
                enhanced_tfidf_features[feature_name] = pd.Series(feature_matrix[:, i])
    
    return pd.DataFrame(enhanced_tfidf_features), vectorizers

def analyze_aspiration_sentiment(df, text_columns):
    """
    Implement sentiment analysis specifically for aspirational language
    """
    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    # Aspirational and ambitious phrases to detect
    aspiration_phrases = [
    # Metas e objetivos explícitos
    'quiero ser', 'espero ser', 'mi meta es', 'mi objetivo es', 'mi propósito es',
    'aspiro a', 'deseo lograr', 'planeo', 'mi sueño es', 'mi ambición es',
    
    # Projeções futuras
    'en el futuro', 'me veo', 'me visualizo', 'en cinco años', 'más adelante',
    'cuando domine el inglés', 'una vez que aprenda', 'después de dominar',
    
    # Verbos de conquista no futuro
    'lograré', 'conseguiré', 'alcanzaré', 'obtendré', 'realizaré', 'cumpliré',
    'conquistaré', 'dominaré', 'superaré', 'progresaré', 'avanzaré',
    
    # Expressões de capacitação
    'me ayudará a', 'me permitirá', 'me dará la oportunidad de', 'me capacitará para',
    'me facilitará', 'me abrirá puertas para', 'me habilitará para', 'podré',
    
    # Expressões de compromisso
    'estoy determinado a', 'estoy comprometido a', 'haré lo necesario para',
    'no me detendré hasta', 'persistiré hasta', 'me esforzaré por',
    
    # Expressões com timeline específico
    'en los próximos meses', 'dentro de un año', 'para fin de año',
    'en corto plazo', 'a mediano plazo', 'a largo plazo',
    
    # Frases completas de planejamento
    'tengo un plan para', 'he establecido como meta', 'mi estrategia es',
    'estoy trabajando para', 'estoy en camino a', 'voy a dedicarme a'
    ]
    
    # Initialize features dictionary
    features = {}
    
    for col in text_columns:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Standard sentiment analysis
        sentiment_scores = df[col].apply(
            lambda text: get_vader_sentiment_scores(
                normalize_text(text) if isinstance(text, str) and not pd.isna(text) else "", 
                sia
            )
        )
        
        # Extract components
        features[f"{col_clean}_sentiment_pos"] = sentiment_scores.apply(lambda x: x["pos"])
        features[f"{col_clean}_sentiment_neg"] = sentiment_scores.apply(lambda x: x["neg"])
        features[f"{col_clean}_sentiment_compound"] = sentiment_scores.apply(lambda x: x["compound"])
        
        # Aspiration detection
        aspiration_counts = []
        for text in df[col]:
            text = normalize_text(text)
            count = sum(text.count(phrase) for phrase in aspiration_phrases)
            aspiration_counts.append(count)
        
        features[f"{col_clean}_aspiration_count"] = aspiration_counts
        
        # Aspiration score: combination of positive sentiment and aspiration phrases
        features[f"{col_clean}_aspiration_score"] = np.array(aspiration_counts) * \
                                                 features[f"{col_clean}_sentiment_pos"]
    
    return pd.DataFrame(features)

def detect_commitment_expressions(df, text_columns):
    """
    Create features for commitment and determination expressions
    """
    # Commitment phrases in Spanish
    commitment_phrases = {
    # Expressões de forte comprometimento
    'estoy decidido': 2.0,
    'estoy comprometido': 2.0,
    'me comprometo': 2.0,
    'haré lo que sea': 2.0,
    'haré lo necesario': 2.0,
    'cueste lo que cueste': 2.0,
    'sin duda': 1.5,
    'estoy determinado': 2.0,
    'daré mi máximo': 1.8,
    'no me rendiré': 1.9,
    'persistiré': 1.7,
    'me esforzaré al máximo': 1.8,
    'pondré todo de mi': 1.8,
    'no descansaré hasta': 1.9,
    'haré todo lo posible': 1.8,
    
    # Expressões de compromisso com aprendizado
    'quiero aprender': 1.0,
    'necesito aprender': 1.5,
    'debo aprender': 1.2,
    'tengo que aprender': 1.2,
    'es importante aprender': 1.0,
    'es necesario aprender': 1.2,
    'estoy dispuesto a aprender': 1.4,
    'me interesa aprender': 0.9,
    'voy a aprender': 1.1,
    'aprenderé': 1.1,
    'quiero dominar': 1.3,
    'necesito dominar': 1.6,
    'estoy enfocado en aprender': 1.4,
    'mi prioridad es aprender': 1.5,
    'estoy listo para aprender': 1.3,
    
    # Expressões de expectativa
    'espero poder': 0.8,
    'espero lograr': 1.0,
    'espero conseguir': 1.0,
    'confío en que podré': 1.1,
    'tengo la expectativa de': 0.9,
    'aspiro a': 1.0,
    'anhelo': 0.9,
    'aguardo con interés': 0.8,
    'tengo la confianza de': 1.0,
    'sé que lograré': 1.2,
    
    # Necessidade relacionada ao trabalho
    'necesito para mi trabajo': 1.8,
    'necesito para mi carrera': 1.8,
    'es esencial para mi trabajo': 1.8,
    'lo requiere mi trabajo': 1.5,
    'lo exige mi trabajo': 1.5,
    'mi puesto depende de': 1.7,
    'mi empleo exige': 1.6,
    'para mejorar en mi trabajo': 1.7,
    'para avanzar profesionalmente': 1.8,
    'mi jefe requiere': 1.6,
    'mi profesión demanda': 1.7,
    'indispensable para mi trabajo': 1.9,
    'fundamental para mi carrera': 1.8,
    'clave para mi desempeño laboral': 1.7,
    'vital para mi desarrollo profesional': 1.8,
    
    # Compromisso de tempo
    'dedicaré tiempo': 1.2,
    'invertir tiempo': 1.2,
    'voy a dedicar': 1.2,
    'destinaré tiempo': 1.2,
    'apartaré tiempo': 1.1,
    'haré espacio en mi agenda': 1.3,
    'dedicaré horas': 1.3,
    'reservaré tiempo': 1.2,
    'comprometeré tiempo': 1.3,
    'tiempo diario': 1.4,
    'práctica constante': 1.3,
    'estudio diario': 1.4,
    'todos los días': 1.3,
    'rutina de estudio': 1.4,
    'horario establecido': 1.3
    }
    
    # Initialize features dictionary
    features = {}
    
    # Process each text column
    for col in text_columns:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Lists for commitment features
        commitment_scores = []
        has_commitment = []
        commitment_counts = []
        
        # Process each text entry
        for text in df[col]:
            text = normalize_text(text)
            
            # Count and score commitment phrases
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
        
        # Add features to dictionary
        features[f"{col_clean}_commitment_score"] = commitment_scores
        features[f"{col_clean}_has_commitment"] = has_commitment
        features[f"{col_clean}_commitment_count"] = commitment_counts
    
    return pd.DataFrame(features)

def create_career_term_detector(df, text_columns):
    """
    Create a detector for career-related terms with weighted importance
    """
    # Career-related terms with importance weights
    career_terms = {
    # Termos de avanço na carreira
    'crecimiento profesional': 2.0,
    'desarrollo profesional': 2.0,
    'avance profesional': 2.0,
    'progreso profesional': 2.0,
    'ascenso': 1.8,
    'promoción': 1.8,
    'evolución profesional': 1.9,
    'trayectoria profesional': 1.7,
    'plan de carrera': 1.9,
    'mejora profesional': 1.8,
    'superación profesional': 1.9,
    'especialización': 1.7,
    'adquirir experiencia': 1.6,
    
    # Termos de oportunidade na carreira
    'oportunidades laborales': 2.0,
    'oportunidades de trabajo': 2.0,
    'mejores trabajos': 1.8,
    'opciones de trabajo': 1.5,
    'posibilidades laborales': 1.5,
    'mercado laboral': 1.6,
    'campo profesional': 1.5,
    'sector laboral': 1.5,
    'industria': 1.4,
    'área profesional': 1.5,
    'nuevas posiciones': 1.7,
    'vacantes': 1.4,
    'puestos disponibles': 1.5,
    
    # Termos de salário/compensação
    'mejor salario': 1.8,
    'mayor sueldo': 1.8,
    'mejor remuneración': 1.7,
    'ganar más': 1.6,
    'aumentar ingresos': 1.6,
    'incremento salarial': 1.7,
    'mejores condiciones': 1.6,
    'compensación competitiva': 1.7,
    'paquete de beneficios': 1.5,
    'incentivos económicos': 1.6,
    'mejor nivel de vida': 1.7,
    'estabilidad económica': 1.7,
    'seguridad financiera': 1.6,
    
    # Termos de trabalho internacional
    'trabajo internacional': 1.7,
    'empleo internacional': 1.7,
    'trabajar en el extranjero': 1.7,
    'oportunidades internacionales': 1.7,
    'empresa multinacional': 1.6,
    'carrera global': 1.8,
    'expatriación': 1.6,
    'trabajar en otro país': 1.7,
    'experiencia internacional': 1.7,
    'proyección internacional': 1.7,
    'comunicación global': 1.6,
    'negocios internacionales': 1.7,
    'mercado global': 1.6,
    
    # Termos específicos para inglês no trabalho
    'comunicación en inglés': 1.9,
    'inglés para negocios': 1.9,
    'inglés profesional': 1.9,
    'inglés corporativo': 1.8,
    'requisito de inglés': 1.9,
    'inglés como requisito': 1.9,
    'inglés es necesario': 1.8,
    'empresas requieren inglés': 2.0,
    'entorno internacional': 1.8,
    'ambiente corporativo': 1.7,
    
    # Termos de busca de emprego
    'buscar trabajo': 1.5,
    'encontrar trabajo': 1.5,
    'conseguir empleo': 1.5,
    'entrevista': 1.4,
    'curriculum': 1.3,
    'cv': 1.3,
    'aplicar a puestos': 1.4,
    'postular': 1.4,
    'selección laboral': 1.4,
    'reclutamiento': 1.4,
    'oportunidad laboral': 1.6,
    'proceso de selección': 1.4,
    'perfil profesional': 1.5,
    'red de contactos': 1.3,
    'networking': 1.4
    }
    
    # Initialize features dictionary
    features = {}
    
    # Process each text column
    for col in text_columns:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Lists for career term features
        career_scores = []
        has_career_terms = []
        career_term_counts = []
        
        # Process each text entry
        for text in df[col]:
            text = normalize_text(text)
            
            # Count and score career terms
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
        
        # Add features to dictionary
        features[f"{col_clean}_career_term_score"] = career_scores
        features[f"{col_clean}_has_career_terms"] = has_career_terms
        features[f"{col_clean}_career_term_count"] = career_term_counts
    
    return pd.DataFrame(features)

def process_dataset(dataset_name, input_dir_basic, input_dir_text, output_dir, 
                    text_columns, mode="fit", vectorizers=None):
    """
    Process a dataset (train, validation, or test) to add new text features
    """
    print(f"\n=== Processing {dataset_name} dataset ===")
    
    # Load datasets
    basic_path = os.path.join(input_dir_basic, f"{dataset_name}.csv")
    text_path = os.path.join(input_dir_text, f"{dataset_name}.csv")
    
    if not os.path.exists(basic_path):
        print(f"Error: Basic dataset not found at {basic_path}")
        return None, None
    
    if not os.path.exists(text_path):
        print(f"Error: Text dataset not found at {text_path}")
        return None, None
    
    print(f"Loading datasets from {basic_path} and {text_path}")
    df_basic = pd.read_csv(basic_path)
    df_text = pd.read_csv(text_path)
    
    # Verify the datasets have the same number of rows
    if len(df_basic) != len(df_text):
        print(f"Error: Datasets have different sizes - Basic: {len(df_basic)}, Text: {len(df_text)}")
        return None, None
    
    print(f"Successfully loaded datasets with {len(df_basic)} rows")
    
    # Create feature DataFrames
    print("Creating professional motivation score...")
    motivation_df = create_professional_motivation_score(df_basic, text_columns)
    
    print("Enhancing TF-IDF for career terms...")
    tfidf_df, updated_vectorizers = enhance_tfidf_for_career_terms(
        df_basic, text_columns, 
        fit_mode=(mode == "fit"), 
        vectorizers=vectorizers
    )
    
    print("Analyzing aspiration sentiment...")
    sentiment_df = analyze_aspiration_sentiment(df_basic, text_columns)
    
    print("Detecting commitment expressions...")
    commitment_df = detect_commitment_expressions(df_basic, text_columns)
    
    print("Creating career term detector...")
    career_df = create_career_term_detector(df_basic, text_columns)
    
    # Combine all new features
    print("Combining all new features...")
    new_features = pd.concat(
        [motivation_df, tfidf_df, sentiment_df, commitment_df, career_df], 
        axis=1
    )
    
    print(f"Created {len(new_features.columns)} new features")
    
    # Combine with advanced text features
    print("Merging with existing features...")
    result_df = pd.concat([df_text, new_features], axis=1)
    
    # Save the result
    output_path = os.path.join(output_dir, f"{dataset_name}.csv")
    result_df.to_csv(output_path, index=False)
    print(f"Saved enhanced dataset to {output_path}")
    
    return result_df, updated_vectorizers

def main():
    """
    Main function to process all datasets
    """
    start_time = time.time()
    print("Starting advanced text feature engineering process...")
    
    # Process training dataset
    _, vectorizers = process_dataset(
        "train", INPUT_DIR_BASIC, INPUT_DIR_TEXT, OUTPUT_DIR, 
        TEXT_COLUMNS, mode="fit"
    )
    
    # Process validation dataset
    process_dataset(
        "validation", INPUT_DIR_BASIC, INPUT_DIR_TEXT, OUTPUT_DIR, 
        TEXT_COLUMNS, mode="transform", vectorizers=vectorizers
    )
    
    # Process test dataset
    process_dataset(
        "test", INPUT_DIR_BASIC, INPUT_DIR_TEXT, OUTPUT_DIR, 
        TEXT_COLUMNS, mode="transform", vectorizers=vectorizers
    )
    
    total_time = time.time() - start_time
    print(f"\nAdvanced text feature engineering completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()