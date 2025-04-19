import pandas as pd
import time

def match_surveys_with_buyers(surveys, buyers):
    """Realiza correspondência entre pesquisas e compradores.
    
    Args:
        surveys: DataFrame com dados de pesquisas
        buyers: DataFrame com dados de compradores
        
    Returns:
        DataFrame com correspondências encontradas
    """
    print("\nMatching surveys with buyers...")
    start_time = time.time()
    
    # Verificar se podemos prosseguir com a correspondência
    if surveys.empty or buyers.empty or 'email_norm' not in buyers.columns or 'email_norm' not in surveys.columns:
        print("Warning: Cannot perform matching. Either surveys or buyers data is empty or missing email_norm column.")
        return pd.DataFrame(columns=['buyer_id', 'survey_id', 'match_type', 'score'])
    
    # Dicionários de consulta para correspondência mais rápida
    survey_emails_dict = dict(zip(surveys['email_norm'], surveys.index))
    survey_emails_set = set(surveys['email_norm'].dropna())
    
    # Primeiro: correspondência exata
    matches = []
    buyers_with_exact_match = buyers[buyers['email_norm'].isin(survey_emails_set)]
    for idx, buyer in buyers_with_exact_match.iterrows():
        if pd.isna(buyer['email_norm']):
            continue
            
        survey_idx = survey_emails_dict.get(buyer['email_norm'])
        if survey_idx is not None:
            match_data = {
                'buyer_id': idx,
                'survey_id': survey_idx,
                'match_type': 'exact',
                'score': 1.0
            }
            
            # Adicionar informação de lançamento se disponível
            if 'lançamento' in buyer and not pd.isna(buyer['lançamento']):
                match_data['lançamento'] = buyer['lançamento']
                
            matches.append(match_data)
    
    # Reportar resultados
    print(f"Found {len(matches)} exact matches out of {len(buyers)} buyers.")
    matches_df = pd.DataFrame(matches)
    
    # Calcular tempo gasto
    end_time = time.time()
    print(f"Matching completed in {end_time - start_time:.2f} seconds.")
    
    return matches_df