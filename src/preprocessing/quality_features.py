import pandas as pd
import numpy as np

def detect_quality_columns(df):
    """Detecta colunas relacionadas à qualidade no DataFrame.
    
    Args:
        df: DataFrame pandas com dados
        
    Returns:
        dict: Informação sobre colunas de qualidade encontradas
    """
    # Verificar se as colunas consolidadas já existem
    has_numeric = 'qualidade_numerica' in df.columns
    has_textual = 'qualidade_textual' in df.columns
    
    # Listar todas as variantes das colunas de qualidade
    quality_variants = [
        'Qualidade (Nome)', 'Qualidade (Número)', 'Qualidade (Numero)', 
        'Qualidade (nome)', 'Qualidade (número)', 'Qualidade (Nombre)', 
        'Qualidade', 'Qualidade (Número) ', 'Qualidade (Nome) '
    ]
    
    # Checar quais variantes existem no dataset
    existing_variants = [col for col in quality_variants if col in df.columns]
    
    # Verificar possíveis colunas renomeadas
    if len(existing_variants) == 0 and not (has_numeric or has_textual):
        possible_renamed = [col for col in df.columns if 'qual' in col.lower()]
    else:
        possible_renamed = []
    
    return {
        'has_numeric': has_numeric,
        'has_textual': has_textual,
        'existing_variants': existing_variants,
        'possible_renamed': possible_renamed
    }

def consolidate_quality_columns(df, fit=True, params=None):
    """Consolida múltiplas colunas de qualidade em colunas normalizadas.
    
    Args:
        df: DataFrame pandas com dados
        fit: Se True, realiza processo de fit, caso contrário utiliza params
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame pandas com colunas de qualidade consolidadas
        Dicionário com parâmetros aprendidos (se fit=True)
    """
    # Cria uma cópia do dataframe para não modificar o original
    df_result = df.copy()
    
    # Detectar colunas de qualidade
    quality_info = detect_quality_columns(df_result)
    
    # Se estamos no modo transform e temos parâmetros, usar configuração salva
    if not fit and params and 'quality_columns' in params:
        # No modo transform, só criar colunas se elas não existirem
        if not quality_info['has_numeric']:
            df_result['qualidade_numerica'] = np.nan
        if not quality_info['has_textual']:
            df_result['qualidade_textual'] = None
            
        # Remover colunas originais se ainda existirem
        variants_to_remove = [col for col in quality_info['existing_variants'] 
                             if col in df_result.columns]
        if variants_to_remove:
            df_result = df_result.drop(columns=variants_to_remove)
            
        return df_result, params
    
    # A partir daqui estamos no modo fit ou sem parâmetros
    
    # Inicializar parâmetros se necessário
    if fit:
        if params is None:
            params = {}
        params['quality_columns'] = {
            'numeric_cols': [],
            'text_cols': []
        }
    
    # Caso 1: Nenhuma coluna encontrada, criar colunas vazias
    if len(quality_info['existing_variants']) == 0 and not (quality_info['has_numeric'] or quality_info['has_textual']):
        df_result['qualidade_numerica'] = np.nan
        df_result['qualidade_textual'] = None
        
        return df_result, params
        
    # Caso 2: Colunas já existem, remover redundâncias
    elif quality_info['has_numeric'] and quality_info['has_textual']:
        # Remover as colunas originais se ainda existirem
        if quality_info['existing_variants']:
            df_result = df_result.drop(columns=quality_info['existing_variants'])
            
        return df_result, params
        
    # Caso 3: Processar e consolidar as colunas
    else:
        # Separar colunas numéricas e textuais
        numeric_cols = []
        text_cols = []
        
        for col in quality_info['existing_variants']:
            if 'nome' in col.lower() or 'name' in col.lower() or 'nombre' in col.lower():
                text_cols.append(col)
            else:
                # Tentar converter para confirmar se é numérica
                try:
                    numeric_test = pd.to_numeric(df_result[col], errors='coerce')
                    # Se pelo menos 50% dos valores não-nulos converteram, considerar numérica
                    not_null = df_result[col].notna().sum()
                    if not_null > 0 and numeric_test.notna().sum() / not_null >= 0.5:
                        numeric_cols.append(col)
                    else:
                        text_cols.append(col)
                except:
                    text_cols.append(col)
        
        # Guardar a classificação nos parâmetros
        if fit:
            params['quality_columns']['numeric_cols'] = numeric_cols
            params['quality_columns']['text_cols'] = text_cols
        
        # Consolidar colunas numéricas
        if numeric_cols:
            df_result['qualidade_numerica'] = np.nan
            # Preencher em ordem de prioridade (mais valores não-nulos primeiro)
            numeric_cols_sorted = sorted(numeric_cols, key=lambda x: df_result[x].notna().sum(), reverse=True)
            
            for col in numeric_cols_sorted:
                mask = df_result['qualidade_numerica'].isna() & df_result[col].notna()
                if mask.sum() > 0:
                    df_result.loc[mask, 'qualidade_numerica'] = pd.to_numeric(df_result.loc[mask, col], errors='coerce')
        
        # Consolidar colunas textuais
        if text_cols:
            df_result['qualidade_textual'] = None
            # Preencher em ordem de prioridade
            text_cols_sorted = sorted(text_cols, key=lambda x: df_result[x].notna().sum(), reverse=True)
            
            for col in text_cols_sorted:
                mask = df_result['qualidade_textual'].isna() & df_result[col].notna()
                if mask.sum() > 0:
                    df_result.loc[mask, 'qualidade_textual'] = df_result.loc[mask, col]
        
        # Remover as colunas originais
        df_result = df_result.drop(columns=quality_info['existing_variants'])
        
        return df_result, params