import pandas as pd
from difflib import SequenceMatcher

def normalize_email(email):
    """Normaliza endereços de email para facilitar correspondência.
    
    Args:
        email: Email para normalizar
        
    Returns:
        Email normalizado ou None se inválido
    """
    if pd.isna(email) or email is None:
        return None
    
    email_str = str(email).lower().strip().replace(" ", "")
    if '@' not in email_str:
        return None
    
    username, domain = email_str.split('@', 1)
    
    # Gmail-specific normalization
    if domain == 'gmail.com':
        username = username.replace('.', '')
    
    # Common domain corrections
    domain_corrections = {
        'gmial.com': 'gmail.com', 'gmail.con': 'gmail.com', 'hotmial.com': 'hotmail.com',
        'outlook.con': 'outlook.com', 'yahoo.con': 'yahoo.com'
    }
    
    if domain in domain_corrections:
        domain = domain_corrections[domain]
    
    return f"{username}@{domain}"

def similarity_score(str1, str2):
    """Calcula um score de similaridade entre duas strings.
    
    Args:
        str1: Primeira string
        str2: Segunda string
        
    Returns:
        Score de similaridade entre 0 e 1
    """
    if not str1 or not str2:
        return 0
    return SequenceMatcher(None, str1, str2).ratio()

def normalize_emails_in_dataframe(df, email_col='email'):
    """Adiciona uma coluna de emails normalizados a um DataFrame.
    
    Args:
        df: DataFrame a ser processado
        email_col: Nome da coluna contendo emails
        
    Returns:
        DataFrame com coluna 'email_norm' adicionada
    """
    if email_col in df.columns:
        df['email_norm'] = df[email_col].apply(normalize_email)
    return df