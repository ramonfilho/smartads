import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

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
    # Importar a função de padronização
    from src.utils.feature_naming import standardize_feature_name
    
    # Cria uma cópia do dataframe para não modificar o original
    df_result = df.copy()
    
    # Inicializar parâmetros se necessário
    if params is None:
        params = {}
    if 'quality_columns' not in params:
        params['quality_columns'] = {}
    
    # Detectar colunas de qualidade
    quality_info = detect_quality_columns(df_result)
    
    # Definir nomes padronizados para as novas colunas
    numeric_col_name = standardize_feature_name('qualidade_numerica')  # → 'qualidade_numerica'
    text_col_name = standardize_feature_name('qualidade_textual')      # → 'qualidade_textual'
    
    # Se estamos no modo transform e temos parâmetros, usar configuração salva
    if not fit and 'numeric_cols' in params['quality_columns'] and 'text_cols' in params['quality_columns']:
        # No modo transform, só criar colunas se elas não existirem
        if not quality_info['has_numeric']:
            df_result[numeric_col_name] = np.nan
        if not quality_info['has_textual']:
            df_result[text_col_name] = None
            
        # Remover colunas originais se ainda existirem
        variants_to_remove = [col for col in quality_info['existing_variants'] 
                             if col in df_result.columns]
        if variants_to_remove:
            df_result = df_result.drop(columns=variants_to_remove)
            
        return df_result, params
    
    # Caso 1: Nenhuma coluna encontrada, criar colunas vazias
    if len(quality_info['existing_variants']) == 0 and not (quality_info['has_numeric'] or quality_info['has_textual']):
        df_result[numeric_col_name] = np.nan
        df_result[text_col_name] = None
        
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
            # Verificar se o nome da coluna indica que é textual
            col_lower = col.lower()
            if 'nome' in col_lower or 'name' in col_lower or 'nombre' in col_lower:
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
        params['quality_columns']['numeric_cols'] = numeric_cols
        params['quality_columns']['text_cols'] = text_cols
        params['quality_columns']['numeric_col_name'] = numeric_col_name
        params['quality_columns']['text_col_name'] = text_col_name
        
        # Consolidar colunas numéricas
        if numeric_cols:
            df_result[numeric_col_name] = np.nan
            # Preencher em ordem de prioridade (mais valores não-nulos primeiro)
            numeric_cols_sorted = sorted(numeric_cols, key=lambda x: df_result[x].notna().sum(), reverse=True)
            
            for col in numeric_cols_sorted:
                mask = df_result[numeric_col_name].isna() & df_result[col].notna()
                if mask.sum() > 0:
                    df_result.loc[mask, numeric_col_name] = pd.to_numeric(df_result.loc[mask, col], errors='coerce')
        elif not quality_info['has_numeric']:
            # Se não há colunas numéricas, criar coluna vazia
            df_result[numeric_col_name] = np.nan
        
        # Consolidar colunas textuais
        if text_cols:
            df_result[text_col_name] = None
            # Preencher em ordem de prioridade
            text_cols_sorted = sorted(text_cols, key=lambda x: df_result[x].notna().sum(), reverse=True)
            
            for col in text_cols_sorted:
                mask = df_result[text_col_name].isna() & df_result[col].notna()
                if mask.sum() > 0:
                    df_result.loc[mask, text_col_name] = df_result.loc[mask, col]
        elif not quality_info['has_textual']:
            # Se não há colunas textuais, criar coluna vazia
            df_result[text_col_name] = None
        
        # Remover as colunas originais
        df_result = df_result.drop(columns=quality_info['existing_variants'])
        
        return df_result, params

def handle_duplicates(df, fit=True, params=None):
    """Remove seletivamente duplicatas, preservando compradores (target=1).
    
    Args:
        df: DataFrame pandas com dados
        fit: Se True, realiza processo de fit, caso contrário apenas transforma
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame sem duplicatas
        Dicionário com parâmetros aprendidos (se fit=True)
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    # Se não tem coluna target ou email_norm, retornar sem mudanças
    if 'target' not in df.columns or 'email_norm' not in df.columns:
        return df, params
    
    # No modo transform, não fazemos remoção de duplicatas
    # pois precisamos prever para todos os registros
    if not fit:
        return df, params
    
    # A partir daqui estamos no modo fit
    # Identificar duplicatas
    duplicated_emails = df.duplicated('email_norm', keep=False)
    
    # Separar compradores e não-compradores
    buyers = df[df['target'] == 1].copy()
    non_buyers = df[df['target'] == 0].copy()
    
    # Remover apenas duplicatas entre não-compradores
    non_buyers_dedup = non_buyers.sort_values('Marca temporal').drop_duplicates(subset=['email_norm'], keep='first')
    
    # Recombinar dataset
    result = pd.concat([buyers, non_buyers_dedup], ignore_index=True)
    
    return result, params

def handle_missing_values(df, fit=True, params=None):
    """Trata valores ausentes com estratégias específicas por tipo de coluna.
    
    Args:
        df: DataFrame pandas com dados
        fit: Se True, aprende parâmetros, caso contrário usa parâmetros existentes
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame com valores ausentes tratados
        Dicionário com parâmetros aprendidos (se fit=True)
    """
    # Inicializar ou usar parâmetros existentes
    if params is None:
        params = {}
    if 'missing_values' not in params:
        params['missing_values'] = {}
    
    # Criar cópia para não modificar o original
    df_result = df.copy()
    
    # 1. Remover colunas com alta ausência (>95%)
    if fit:
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        high_missing_cols = missing_pct[missing_pct > 95].index.tolist()
        params['missing_values']['high_missing_cols'] = high_missing_cols
    else:
        high_missing_cols = params['missing_values'].get('high_missing_cols', [])
    
    # Remover apenas as colunas que existem no DataFrame
    cols_to_drop = [col for col in high_missing_cols if col in df_result.columns]
    if cols_to_drop:
        df_result = df_result.drop(columns=cols_to_drop)
    
    # 2. Colunas UTM (tratar para análise de marketing)
    utm_cols = [col for col in df_result.columns if col.startswith('UTM_') or 'utm' in col.lower()]
    for col in utm_cols:
        df_result[col] = df_result[col].fillna('unknown')
    
    # 3. Dados categóricos (preencher com 'desconhecido')
    cat_cols = [
        '¿Cuál es tu género?', '¿Cuál es tu edad?', '¿Cual es tu país?',
        '¿Hace quánto tiempo me conoces?', '¿Cuál es tu disponibilidad de tiempo para estudiar inglés?',
        '¿Cuál es tu sueldo anual? (en dólares)', '¿Cuánto te gustaría ganar al año?',
        '¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?',
        '¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?',
        'Qualidade', 'lançamento'
    ]
    
    for col in cat_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna('desconhecido')
    
    # 4. Colunas de texto livre (preencher com string vazia)
    text_cols = [
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
        '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
        'Déjame un mensaje',
        '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
    ]
    
    for col in text_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna('')
    
    # 5. Colunas de qualidade (tratamento específico)
    quality_cols = [col for col in df_result.columns if 'qualidade' in col.lower() or 'qualidad' in col.lower()]
    quality_numeric_cols = []
    
    # Verificar quais colunas de qualidade são numéricas
    for col in quality_cols:
        if col in df_result.columns:
            try:
                non_null_values = df_result[col].dropna()
                pd.to_numeric(non_null_values, errors='raise')
                quality_numeric_cols.append(col)
            except:
                # Se não for numérica, tratar como categórica
                if df_result[col].isna().sum() > 0:
                    df_result[col] = df_result[col].fillna('desconhecido')
    
    # Processar colunas de qualidade numéricas
    for col in quality_numeric_cols:
        if col in df_result.columns:
            # Converter para numérico forçando valores não numéricos para NaN
            df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
            
            if fit:
                # Calcular e armazenar a mediana
                median_value = df_result[col].median()
                params['missing_values'][f'median_{col}'] = median_value
            else:
                # Usar mediana armazenada ou calcular se não existir
                median_value = params['missing_values'].get(f'median_{col}')
                if median_value is None:
                    median_value = df_result[col].median()
            
            # Preencher valores ausentes
            df_result[col] = df_result[col].fillna(median_value)
    
    # 6. Outras colunas numéricas (preencher com mediana)
    numeric_cols = df_result.select_dtypes(include=['number']).columns.tolist()
    other_numeric_cols = [col for col in numeric_cols if col not in ['target'] + quality_numeric_cols]
    
    for col in other_numeric_cols:
        if col in df_result.columns and df_result[col].isna().sum() > 0:
            if fit:
                # Calcular e armazenar a mediana
                median_value = df_result[col].median()
                params['missing_values'][f'median_{col}'] = median_value
            else:
                # Usar mediana armazenada ou calcular se não existir
                median_value = params['missing_values'].get(f'median_{col}')
                if median_value is None:
                    median_value = df_result[col].median()
            
            # Preencher valores ausentes
            df_result[col] = df_result[col].fillna(median_value)
    
    return df_result, params

def handle_outliers(df, fit=True, params=None):
    """Trata outliers com estratégias diferentes para colunas de qualidade.
    
    Args:
        df: DataFrame pandas com dados
        fit: Se True, calcular limites, caso contrário usar limites existentes
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame com outliers tratados
        Dicionário com parâmetros atualizados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    if 'outliers' not in params:
        params['outliers'] = {}
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Identificar colunas numéricas
    numeric_cols = df_result.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['target']]
    
    # Identificar colunas de qualidade numéricas
    quality_numeric_cols = [col for col in numeric_cols 
                           if 'qualidade' in col.lower() or 'qualidad' in col.lower()]
    
    # Processar cada coluna numérica
    for col in numeric_cols:
        if col in df_result.columns and df_result[col].nunique() > 10:
            # Tratamento específico para colunas de qualidade
            if col in quality_numeric_cols:
                if fit:
                    # Calcular limites
                    lower_bound = df_result[col].quantile(0.01)
                    upper_bound = df_result[col].quantile(0.99)
                    params['outliers'][f'{col}_bounds'] = (lower_bound, upper_bound)
                else:
                    # Usar limites armazenados
                    if f'{col}_bounds' in params['outliers']:
                        lower_bound, upper_bound = params['outliers'][f'{col}_bounds']
                    else:
                        # Fallback
                        lower_bound = df_result[col].quantile(0.01)
                        upper_bound = df_result[col].quantile(0.99)
                
                # Aplicar capping
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
            
            else:
                # Tratamento padrão para outras colunas numéricas
                if fit:
                    q1 = df_result[col].quantile(0.25)
                    q3 = df_result[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    params['outliers'][f'{col}_bounds'] = (lower_bound, upper_bound)
                else:
                    # Usar limites armazenados
                    if f'{col}_bounds' in params['outliers']:
                        lower_bound, upper_bound = params['outliers'][f'{col}_bounds']
                    else:
                        # Fallback
                        q1 = df_result[col].quantile(0.25)
                        q3 = df_result[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                
                # Aplicar capping
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_result, params

def normalize_values(df, fit=True, params=None):
    """Normaliza valores numéricos usando StandardScaler.
    
    Args:
        df: DataFrame pandas com dados
        fit: Se True, ajustar scaler, caso contrário usar scaler existente
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame com valores normalizados
        Dicionário com parâmetros atualizados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    if 'normalization' not in params:
        params['normalization'] = {}
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Selecionar colunas numéricas que não são target
    numeric_cols = df_result.select_dtypes(include=['number']).columns.tolist()
    cols_to_normalize = [col for col in numeric_cols if col != 'target' and col in df_result.columns]
    
    # Identificar colunas com variabilidade suficiente
    cols_with_std = []
    for col in cols_to_normalize:
        if df_result[col].std() > 0:
            cols_with_std.append(col)
    
    # Armazenar as colunas com variabilidade
    if fit:
        params['normalization']['cols_with_std'] = cols_with_std
    else:
        # No modo transform, usar as colunas identificadas no fit
        cols_with_std = params['normalization'].get('cols_with_std', cols_with_std)
    
    # Se temos colunas para normalizar
    if cols_with_std:
        if fit:
            # Ajustar o scaler e armazenar estatísticas
            scaler = StandardScaler()
            df_normalized = scaler.fit_transform(df_result[cols_with_std])
            
            # Armazenar médias e desvios padrão para cada coluna
            params['normalization']['means'] = dict(zip(cols_with_std, scaler.mean_))
            params['normalization']['stds'] = dict(zip(cols_with_std, scaler.scale_))
        else:
            # Usar estatísticas armazenadas para normalizar
            if 'means' in params['normalization'] and 'stds' in params['normalization']:
                # Criar um array para os dados normalizados
                df_normalized = np.zeros((len(df_result), len(cols_with_std)))
                
                # Normalizar cada coluna separadamente
                for i, col in enumerate(cols_with_std):
                    if col in params['normalization']['means'] and col in params['normalization']['stds']:
                        mean = params['normalization']['means'][col]
                        std = params['normalization']['stds'][col]
                        df_normalized[:, i] = (df_result[col].values - mean) / std
                    else:
                        # Se não temos as estatísticas, usar os dados originais
                        df_normalized[:, i] = df_result[col].values
            else:
                # Fallback: normalizar sem estatísticas anteriores
                scaler = StandardScaler()
                df_normalized = scaler.fit_transform(df_result[cols_with_std])
        
        # Atualizar o DataFrame com os valores normalizados
        df_result[cols_with_std] = df_normalized
    
    return df_result, params

def convert_data_types(df, fit=True, params=None):
    """Converte tipos de dados para formatos apropriados.
    
    Args:
        df: DataFrame pandas com dados
        fit: Flag para indicar se estamos no modo fit
        params: Dicionário com parâmetros (não utilizado nesta função)
        
    Returns:
        DataFrame com tipos de dados convertidos
        Parâmetros inalterados
    """
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Data e hora
    if 'Marca temporal' in df_result.columns:
        df_result['Marca temporal'] = pd.to_datetime(df_result['Marca temporal'], errors='coerce')
    
    if 'DATA' in df_result.columns:
        df_result['DATA'] = pd.to_datetime(df_result['DATA'], errors='coerce')
    
    return df_result, params