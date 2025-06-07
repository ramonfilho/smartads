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
from src.utils.feature_naming import standardize_feature_name
from src.utils.parameter_manager import ParameterManager

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

def create_professional_motivation_score(df, text_columns, fit=True, params=None, param_manager=None):
    """
    Cria um score agregado de motivação profissional baseado em palavras-chave.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit
        params: Dicionário com parâmetros (deprecated - usar param_manager)
        param_manager: Instância do ParameterManager
        
    Returns:
        DataFrame com features de motivação profissional adicionadas
        ParameterManager atualizado
    """
    if param_manager is None:
        param_manager = ParameterManager()
    
    result_df = pd.DataFrame(index=df.index)
    
    # Se estamos no modo fit, criar o dicionário de palavras-chave
    if fit:
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
        
        # MUDANÇA: Salvar usando param_manager
        param_manager.save_professional_params('motivation_keywords', work_keywords)
    else:
        # MUDANÇA: Recuperar usando param_manager
        work_keywords = param_manager.get_professional_params('motivation_keywords')
        if not work_keywords:
            raise ValueError("Keywords de motivação não encontradas nos parâmetros!")
    
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
    result_df[standardize_feature_name('professional_motivation_score')] = normalized_scores
    result_df[standardize_feature_name('career_keyword_count')] = keyword_counts
    
    return result_df, param_manager

def analyze_aspiration_sentiment(df, text_columns, fit=True, params=None, param_manager=None):
    """
    Implementa análise de sentimento específica para linguagem de aspiração.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit
        params: Dicionário com parâmetros (deprecated - usar param_manager)
        param_manager: Instância do ParameterManager
        
    Returns:
        DataFrame com features de sentimento de aspiração adicionadas
        ParameterManager atualizado
    """
    if param_manager is None:
        param_manager = ParameterManager()
    
    result_df = pd.DataFrame(index=df.index)
    
    # Inicializar analisador de sentimento
    try:
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Erro ao inicializar analisador de sentimento: {e}")
        sia = None
    
    # Frases de aspiração para detectar
    if fit:
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
        
        # MUDANÇA: Salvar usando param_manager
        param_manager.save_professional_params('aspiration_phrases', aspiration_phrases)
    else:
        # MUDANÇA: Recuperar usando param_manager
        aspiration_phrases = param_manager.get_professional_params('aspiration_phrases')
        if not aspiration_phrases:
            raise ValueError("Frases de aspiração não encontradas nos parâmetros!")
    
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
        result_df[standardize_feature_name(f"{col_clean}_sentiment_pos")] = [score["pos"] for score in sentiment_scores]
        result_df[standardize_feature_name(f"{col_clean}_sentiment_neg")] = [score["neg"] for score in sentiment_scores]
        result_df[standardize_feature_name(f"{col_clean}_sentiment_compound")] = [score["compound"] for score in sentiment_scores]
        
        # Detecção de aspiração
        aspiration_counts = []
        for text in df[col]:
            text = normalize_text(text)
            count = sum(text.count(phrase) for phrase in aspiration_phrases)
            aspiration_counts.append(count)
        
        result_df[standardize_feature_name(f"{col_clean}_aspiration_count")] = aspiration_counts
        
        # Score de aspiração: combinação de sentimento positivo e frases de aspiração
        result_df[standardize_feature_name(f"{col_clean}_aspiration_score")] = np.array(aspiration_counts) * \
                                                  result_df[f"{col_clean}_sentiment_pos"]
    
    return result_df, param_manager

def detect_commitment_expressions(df, text_columns, fit=True, params=None, param_manager=None):
    """
    Cria features para expressões de compromisso e determinação.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit
        params: Dicionário com parâmetros (deprecated - usar param_manager)
        param_manager: Instância do ParameterManager
        
    Returns:
        DataFrame com features de expressões de compromisso adicionadas
        ParameterManager atualizado
    """
    if param_manager is None:
        param_manager = ParameterManager()
    
    result_df = pd.DataFrame(index=df.index)
    
    # Frases de compromisso em espanhol com pesos
    if fit:
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
        
        # MUDANÇA: Salvar usando param_manager
        param_manager.save_professional_params('commitment_phrases', commitment_phrases)
    else:
        # MUDANÇA: Recuperar usando param_manager
        commitment_phrases = param_manager.get_professional_params('commitment_phrases')
        if not commitment_phrases:
            raise ValueError("Frases de compromisso não encontradas nos parâmetros!")
    
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
        result_df[standardize_feature_name(f"{col_clean}_commitment_score")] = commitment_scores
        result_df[standardize_feature_name(f"{col_clean}_has_commitment")] = has_commitment
        result_df[standardize_feature_name(f"{col_clean}_commitment_count")] = commitment_counts
    
    return result_df, param_manager

def create_career_term_detector(df, text_columns, fit=True, params=None, param_manager=None):
    """
    Cria um detector para termos relacionados à carreira com pesos de importância.
    
    Args:
        df: DataFrame pandas
        text_columns: Lista de colunas de texto para processamento
        fit: Flag para indicar se estamos no modo fit
        params: Dicionário com parâmetros (deprecated - usar param_manager)
        param_manager: Instância do ParameterManager
        
    Returns:
        DataFrame com features de termos de carreira adicionadas
        ParameterManager atualizado
    """
    if param_manager is None:
        param_manager = ParameterManager()
    
    result_df = pd.DataFrame(index=df.index)
    
    # Termos relacionados à carreira com pesos
    if fit:
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
        
        # MUDANÇA: Salvar usando param_manager
        param_manager.save_professional_params('career_terms', career_terms)
    else:
        # MUDANÇA: Recuperar usando param_manager
        career_terms = param_manager.get_professional_params('career_terms')
        if not career_terms:
            raise ValueError("Termos de carreira não encontrados nos parâmetros!")
    
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
        result_df[standardize_feature_name(f"{col_clean}_career_term_score")] = career_scores
        result_df[standardize_feature_name(f"{col_clean}_has_career_terms")] = has_career_terms
        result_df[standardize_feature_name(f"{col_clean}_career_term_count")] = career_term_counts
    
    return result_df, param_manager

def enhance_tfidf_for_career_terms(df, text_cols, fit=True, params=None, param_manager=None):
    """Aprimora pesos TF-IDF para termos de carreira - COM PARAMETER MANAGER"""
    
    if param_manager is None:
        param_manager = ParameterManager()
    
    df_result = pd.DataFrame(index=df.index)
    
    # Career-related terms
    career_terms = [
        'trabajo', 'empleo', 'profesional', 'oportunidades', 'laboral',
        'carrera', 'mejor trabajo', 'oportunidades laborales', 'profesión',
        'mejor', 'comunicación', 'viajar', 'mejorar'
    ]
    
    for col in text_cols:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Normalizar textos
        texts = df[col].apply(normalize_text)
        valid_mask = texts != ""
        valid_count = valid_mask.sum()
        
        if valid_count < 10:
            print(f"  ⚠️ Poucos textos válidos para {col} ({valid_count}), pulando...")
            continue
        
        if fit:
            # MODO FIT: Criar e treinar vetorizador
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            spanish_stopwords = [
                'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra',
                'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e', 'el', 'ella'
            ]
            
            vectorizer = TfidfVectorizer(
                max_features=50,
                min_df=3,
                ngram_range=(1, 2),
                stop_words=spanish_stopwords
            )
            
            try:
                print(f"  🔄 Criando Career TF-IDF para '{col}'...")
                
                # Fit apenas em textos válidos
                valid_texts = texts[valid_mask]
                vectorizer.fit(valid_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # MUDANÇA: Salvar usando param_manager
                param_manager.save_vectorizer(
                    {
                        'vectorizer': vectorizer,
                        'feature_names': feature_names.tolist(),
                        'career_terms': career_terms
                    },
                    name=col_clean,
                    category='career_tfidf'
                )
                
                # Transform e criar features
                tfidf_matrix_sparse = vectorizer.transform(texts)
                tfidf_matrix = tfidf_matrix_sparse.toarray()
                
                # Aplicar boost para termos de carreira
                for term_idx, term in enumerate(feature_names):
                    if any(career_term in term for career_term in career_terms):
                        tfidf_matrix[:, term_idx] *= 1.5
                
                # Adicionar features ao DataFrame
                for i, term in enumerate(feature_names):
                    feature_name = standardize_feature_name(f"{col_clean}_career_tfidf_{term}")
                    df_result[feature_name] = tfidf_matrix[:, i]
                
                # Rastrear features criadas
                feature_names_list = [standardize_feature_name(f"{col_clean}_career_tfidf_{term}") 
                                     for term in feature_names]
                param_manager.track_created_features(feature_names_list)
                
                print(f"    ✓ {len(feature_names)} features Career TF-IDF criadas")
                
            except Exception as e:
                print(f"  ❌ Erro ao criar Career TF-IDF para '{col}': {e}")
                
        else:
            # MODO TRANSFORM: Usar vetorizador existente
            # MUDANÇA: Recuperar usando param_manager
            vectorizer_data = param_manager.get_vectorizer(col_clean, 'career_tfidf')
            
            if not vectorizer_data:
                print(f"  ⚠️ Vetorizador Career TF-IDF não encontrado para '{col_clean}'")
                continue
                
            try:
                vectorizer = vectorizer_data['vectorizer']
                feature_names = vectorizer_data['feature_names']
                saved_career_terms = vectorizer_data.get('career_terms', career_terms)
                
                print(f"  🔄 Aplicando Career TF-IDF para '{col}'...")
                
                # Transform
                tfidf_matrix_sparse = vectorizer.transform(texts)
                tfidf_matrix = tfidf_matrix_sparse.toarray()
                
                # Aplicar boost
                for term_idx, term in enumerate(feature_names):
                    if any(career_term in term for career_term in saved_career_terms):
                        tfidf_matrix[:, term_idx] *= 1.5
                
                # Adicionar features
                for i, term in enumerate(feature_names):
                    feature_name = standardize_feature_name(f"{col_clean}_career_tfidf_{term}")
                    df_result[feature_name] = tfidf_matrix[:, i]
                
                print(f"    ✓ {len(feature_names)} features Career TF-IDF aplicadas")
                
            except Exception as e:
                print(f"  ❌ Erro ao aplicar Career TF-IDF para '{col}': {e}")
    
    # Adicionar score agregado
    for col in text_cols:
        if col not in df.columns:
            continue
            
        col_clean = clean_column_name(col)
        
        # Encontrar colunas career tfidf
        career_tfidf_cols = [c for c in df_result.columns if f"{col_clean}_career_tfidf_" in c]
        
        if career_tfidf_cols:
            # Score agregado
            df_result[standardize_feature_name(f"{col_clean}_career_tfidf_score")] = df_result[career_tfidf_cols].mean(axis=1)
            print(f"    ✓ Score agregado criado para {col_clean}")
    
    return df_result, param_manager

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
    excluded_cols = params.get('excluded_from_text_processing', [])
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