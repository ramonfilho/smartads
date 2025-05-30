"""
Módulo para extração de features relacionadas à motivação profissional.
Contém funcionalidades para detectar e quantificar aspectos relacionados 
à carreira, compromisso e aspirações profissionais nos textos.
"""

import pandas as pd
import numpy as np
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Garantir que os recursos NLTK necessários estejam disponíveis
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)

def normalize_text(text):
    """Normaliza texto para processamento.
    
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

def clean_column_name(col_name):
    """Limpa nomes de colunas para facilitar o manuseio.
    
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

def create_professional_motivation_score(df, text_columns, fit=True, params=None):
    """
    Cria um score agregado de motivação profissional baseado em palavras-chave.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit
        params: Dicionário com parâmetros
        
    Returns:
        DataFrame com features de motivação profissional adicionadas
        Parâmetros atualizados
    """
    if params is None:
        params = {}
    
    result_df = pd.DataFrame(index=df.index)
    
    # Palavras-chave relacionadas à motivação profissional com pesos
    # Se estamos no modo fit, criar o dicionário de palavras-chave
    if fit or 'work_keywords' not in params:
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
            
            # Combinações mais fortes (com pesos mais altos)
            'mejor trabajo': 1.8, 'oportunidades laborales': 2.0, 'crecimiento profesional': 1.8,
            'desarrollo profesional': 1.8, 'mejorar profesionalmente': 1.7, 'futuro profesional': 1.7,
            'carrera internacional': 1.9, 'éxito profesional': 1.7, 'mercado laboral global': 1.8,
            'proyección internacional': 1.7, 'perfil competitivo': 1.6, 'destacar profesionalmente': 1.7,
            'mejor posición': 1.6, 'cargo superior': 1.7, 'mejor remuneración': 1.8,
            'mejores oportunidades': 1.8, 'trabajo en el extranjero': 1.9, 'avance profesional': 1.7,
            'posición de liderazgo': 1.7, 'ascenso laboral': 1.8, 'mejor calidad de vida': 1.5
        }
        
        # IMPORTANTE: Armazenar explicitamente no dicionário de parâmetros
        params['work_keywords'] = work_keywords
    else:
        # No modo transform, usar as palavras-chave existentes
        work_keywords = params['work_keywords']
    
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
    
    return result_df, params

def analyze_aspiration_sentiment(df, text_columns, fit=True, params=None):
    """
    Implementa análise de sentimento específica para linguagem de aspiração.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit
        params: Dicionário com parâmetros
        
    Returns:
        DataFrame com features de sentimento de aspiração adicionadas
        Parâmetros atualizados
    """
    if params is None:
        params = {}
    
    result_df = pd.DataFrame(index=df.index)
    
    # Inicializar analisador de sentimento
    try:
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Erro ao inicializar analisador de sentimento: {e}")
        sia = None
    
    # Frases de aspiração para detectar
    if fit or 'aspiration_phrases' not in params:
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
        
        # IMPORTANTE: Armazenar explicitamente no dicionário de parâmetros
        params['aspiration_phrases'] = aspiration_phrases
    else:
        # No modo transform, usar as frases existentes
        aspiration_phrases = params['aspiration_phrases']
    
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
    
    return result_df, params

def detect_commitment_expressions(df, text_columns, fit=True, params=None):
    """
    Cria features para expressões de compromisso e determinação.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit
        params: Dicionário com parâmetros
        
    Returns:
        DataFrame com features de expressões de compromisso adicionadas
        Parâmetros atualizados
    """
    if params is None:
        params = {}
    
    result_df = pd.DataFrame(index=df.index)
    
    # Frases de compromisso em espanhol com pesos
    if fit or 'commitment_phrases' not in params:
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
        
        # IMPORTANTE: Armazenar explicitamente no dicionário de parâmetros
        params['commitment_phrases'] = commitment_phrases
    else:
        # No modo transform, usar as frases existentes
        commitment_phrases = params['commitment_phrases']
    
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
    
    return result_df, params

def create_career_term_detector(df, text_columns, fit=True, params=None):
    """
    Cria um detector para termos relacionados à carreira com pesos de importância.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit
        params: Dicionário com parâmetros
        
    Returns:
        DataFrame com features de termos de carreira adicionadas
        Parâmetros atualizados
    """
    if params is None:
        params = {}
    
    result_df = pd.DataFrame(index=df.index)
    
    # Termos relacionados à carreira com pesos
    if fit or 'career_terms' not in params:
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
            
            # Termos básicos de carreira
            'trabajo': 1.0,
            'empleo': 1.0,
            'profesional': 1.2,
            'oportunidades': 1.0,
            'laboral': 1.0,
            'carrera': 1.2,
            'mejor trabajo': 1.8,
            'profesión': 1.0,
            'mejor': 0.7,
            'comunicación': 0.7,
            'viajar': 0.6,
            'mejorar': 0.7
        }
        
        # IMPORTANTE: Armazenar explicitamente no dicionário de parâmetros
        params['career_terms'] = career_terms
    else:
        # No modo transform, usar os termos existentes
        career_terms = params['career_terms']
    
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
    
    return result_df, params

def enhance_tfidf_for_career_terms(df, text_cols, fit=True, params=None):
    """
    Aprimora pesos TF-IDF para termos relacionados à carreira.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto para processamento
        fit: Se True, ajusta vetorizadores, caso contrário usa existentes
        params: Dicionário com parâmetros para transform
        
    Returns:
        DataFrame com features TF-IDF aprimoradas
        Dicionário com parâmetros atualizados
    """
    if params is None:
        params = {}
    
    if 'career_tfidf' not in params:
        params['career_tfidf'] = {}
    
    df_result = pd.DataFrame(index=df.index)
    
    # Career-related terms to boost
    career_terms = [
        'trabajo', 'empleo', 'profesional', 'oportunidades', 'laboral',
        'carrera', 'mejor trabajo', 'oportunidades laborales', 'profesión',
        'mejor', 'comunicación', 'viajar', 'mejorar'
    ]
    
    # NOVO: Garantir que career_terms seja acessível nos parâmetros
    if fit:
        # Criar um mapeamento com pesos padrão para cada termo
        career_terms_dict = {term: 1.0 for term in career_terms}
        
        # Atribuir pesos específicos para termos mais importantes
        special_terms = {
            'mejor trabajo': 1.8, 
            'oportunidades laborales': 2.0, 
            'desarrollo profesional': 2.0,
            'crecimiento profesional': 2.0
        }
        
        # Atualizar o dicionário com os pesos específicos
        career_terms_dict.update(special_terms)
    
    # Inicializar analisador de sentimento
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Erro ao inicializar analisador de sentimento: {e}")
        sia = None
    
    # Processar cada coluna de texto
    for col in text_cols:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Normalizar textos e remover vazios
        texts = df[col].apply(normalize_text)
        valid_texts = texts[texts != ""]
        
        if len(valid_texts) < 10:  # Pular se houver poucos textos válidos
            print(f"Pulando TF-IDF para {col_clean} (poucos textos válidos)")
            continue
        
        if fit:
            # Criar e ajustar vetorizador
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Lista de stopwords em espanhol
            spanish_stopwords = [
                'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra',
                'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e', 'el', 'ella',
                'ellas', 'ellos', 'en', 'entre', 'era', 'erais', 'eran', 'eras', 'eres', 'es',
                'esa', 'esas', 'ese', 'eso', 'esos', 'esta', 'estaba', 'estabais', 'estaban',
                'estabas', 'estad', 'estada', 'estadas', 'estado', 'estados', 'estamos', 'estando',
                'estar', 'estaremos', 'estará', 'estarán', 'estarás', 'estaré', 'estaréis'
            ]
            
            vectorizer = TfidfVectorizer(
                max_features=50,
                min_df=3,
                ngram_range=(1, 2),
                stop_words=spanish_stopwords
            )
            
            # Ajustar aos textos válidos
            vectorizer.fit(valid_texts)
            
            # Armazenar vetorizador para uso futuro
            params['career_tfidf'][col_clean] = {
                'vectorizer': vectorizer,
                'feature_names': vectorizer.get_feature_names_out().tolist(),
                'career_terms': career_terms_dict  # NOVO: Armazenar o dicionário de termos de carreira
            }
        
        # Verificar se temos um vetorizador para esta coluna
        if col_clean not in params['career_tfidf']:
            print(f"Vetorizador não encontrado para {col_clean}, pulando...")
            continue
            
        # Obter vetorizador e nomes de features
        vectorizer = params['career_tfidf'][col_clean]['vectorizer']
        feature_names = params['career_tfidf'][col_clean]['feature_names']
        
        # Inicializar matriz de zeros para os resultados
        result_matrix = np.zeros((len(df), len(feature_names)))
        
        # Preencher valores para textos válidos
        valid_indices = texts.index[texts != ""]
        if len(valid_indices) > 0:
            valid_matrix = vectorizer.transform(texts[valid_indices]).toarray()
            
            # Aplicar boost para termos relacionados à carreira
            for term_idx, term in enumerate(feature_names):
                if any(career_term in term for career_term in career_terms):
                    valid_matrix[:, term_idx] *= 1.5  # Boost de 50%
            
            # Colocar valores nas posições corretas
            for i, (idx, _) in enumerate(valid_indices):
                result_matrix[i] = valid_matrix[i]
        
        # Adicionar features ao DataFrame
        for i, term in enumerate(feature_names):
            feature_name = f"{col_clean}_tfidf_{term}"
            df_result[feature_name] = result_matrix[:, i]
    
    # Adicionar colunas com score agregado de carreira
    for col in text_cols:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Encontrar colunas TF-IDF para esta coluna de texto
        tfidf_cols = [c for c in df_result.columns if c.startswith(f"{col_clean}_tfidf_")]
        
        if tfidf_cols:
            # Calcular score agregado: média de todas as colunas TF-IDF relacionadas à carreira
            career_cols = [c for c in tfidf_cols if any(term in c for term in career_terms)]
            
            if career_cols:
                df_result[f"{col_clean}_career_tfidf_score"] = df_result[career_cols].mean(axis=1)
    
    return df_result, params

def enhance_professional_features(df, text_cols, fit=True, params=None):
    """Função principal que executa toda a pipeline de features profissionais.
    
    Args:
        df: DataFrame pandas
        text_cols: Lista de colunas de texto para processamento
        fit: Se True, aprende parâmetros, caso contrário usa existentes
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame com features profissionais adicionadas
        Dicionário com parâmetros atualizados
    """
    if params is None:
        params = {}
    
    if 'professional_features' not in params:
        params['professional_features'] = {}
    
    # Filtrar apenas colunas existentes no dataframe
    text_cols = [col for col in text_cols if col in df.columns]
    
    if not text_cols:
        return df, params
    
    print(f"Aplicando engenharia de features profissionais a {len(text_cols)} colunas de texto...")
    
    # Salvar colunas originais antes de aplicar as transformações
    original_cols = df.columns.tolist()
    
    # 1. Score de motivação profissional
    print("1. Criando score de motivação profissional...")
    df_result, motivation_params = create_professional_motivation_score(
        df, text_cols, fit, params
    )
    
    # NOVO: Armazenar parâmetro de cada etapa separadamente para facilitar acesso
    params['professional_motivation'] = motivation_params
    
    # 2. Aprimorar TF-IDF para termos de carreira
    print("2. Aprimorando TF-IDF para termos de carreira...")
    df_tfidf, vectorizer_params = enhance_tfidf_for_career_terms(
        df, text_cols, fit, params
    )
    
    # NOVO: Armazenar vetorizadores
    params['vectorizers'] = vectorizer_params
    
    # 3. Análise de sentimento de aspiração
    print("3. Analisando sentimento de aspiração...")
    df_aspiration, aspiration_params = analyze_aspiration_sentiment(
        df, text_cols, fit, params
    )
    
    # NOVO: Armazenar parâmetros de aspiração
    params['aspiration_sentiment'] = aspiration_params
    
    # 4. Detecção de expressões de compromisso
    print("4. Detectando expressões de compromisso...")
    df_commitment, commitment_params = detect_commitment_expressions(
        df, text_cols, fit, params
    )
    
    # NOVO: Armazenar parâmetros de compromisso
    params['commitment'] = commitment_params
    
    # 5. Detector de termos de carreira
    print("5. Criando detector de termos de carreira...")
    df_career, career_params = create_career_term_detector(
        df, text_cols, fit, params
    )
    
    # NOVO: Armazenar parâmetros de carreira
    params['career'] = career_params
    
    # NOVO: Garantir que temos career_terms a partir dos vetorizadores se não tiver sido definido explicitamente
    if ('career' in params and 'career_terms' not in params['career']) or ('career' in params and not params['career'].get('career_terms')):
        if 'vectorizers' in params and 'career_tfidf' in params['vectorizers']:
            for col_key, vectorizer_info in params['vectorizers']['career_tfidf'].items():
                if 'career_terms' in vectorizer_info:
                    params['career']['career_terms'] = vectorizer_info['career_terms']
                    print(f"  Career terms extraídos do vetorizador para '{col_key}'")
                    break
    
    # Combinar todas as features
    feature_dfs = [df, df_result, df_tfidf, df_aspiration, df_commitment, df_career]
    all_features = pd.concat(feature_dfs, axis=1)
    
    # Remover colunas duplicadas
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    
    # Armazenar lista de features adicionadas
    if fit:
        added_cols = [col for col in all_features.columns if col not in original_cols]
        params['professional_features']['added_columns'] = added_cols
    
    print(f"Total de features profissionais adicionadas: {len(all_features.columns) - len(original_cols)}")
    return all_features, params