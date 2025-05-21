import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def create_identity_features(df, fit=False, params=None):
    """
    Cria features relacionadas à identidade do usuário (telefone válido, email validado, etc.)
    """
    df_result = df.copy()
    
    # Se não existir um dicionário de parâmetros, criar um novo
    if params is None:
        params = {}
    
    # Garantir que a coluna de telefone seja string antes de aplicar operações de string
    if '¿Cual es tu telefono?' in df_result.columns:
        # Converter NaN/None para string vazia primeiro
        df_result['¿Cual es tu telefono?'] = df_result['¿Cual es tu telefono?'].fillna('')
        # Converter todos os valores para string
        df_result['¿Cual es tu telefono?'] = df_result['¿Cual es tu telefono?'].astype(str)
        # Agora aplicar a operação de string com segurança
        df_result['valid_phone'] = df_result['¿Cual es tu telefono?'].str.replace(r'\D', '', regex=True).str.len() >= 8
    else:
        df_result['valid_phone'] = False
        
    return df_result, params

def create_temporal_features(df, fit=True, params=None):
    """Cria features baseadas em informações temporais.
    
    Args:
        df: DataFrame pandas
        fit: Se True, realiza processo de fit, caso contrário utiliza params
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame com features adicionadas
        Dicionário com parâmetros atualizados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Features de tempo baseadas em 'Marca temporal'
    if 'Marca temporal' in df_result.columns:
        # Converter para datetime se ainda não for
        if not pd.api.types.is_datetime64_dtype(df_result['Marca temporal']):
            df_result['Marca temporal'] = pd.to_datetime(df_result['Marca temporal'], errors='coerce')
        
        # Extrair componentes básicos
        df_result['hour'] = df_result['Marca temporal'].dt.hour
        df_result['day_of_week'] = df_result['Marca temporal'].dt.dayofweek
        df_result['month'] = df_result['Marca temporal'].dt.month
        df_result['year'] = df_result['Marca temporal'].dt.year
        
        # Features cíclicas para hora e dia da semana
        df_result['hour_sin'] = np.sin(2 * np.pi * df_result['hour'] / 24)
        df_result['hour_cos'] = np.cos(2 * np.pi * df_result['hour'] / 24)
        df_result['day_sin'] = np.sin(2 * np.pi * df_result['day_of_week'] / 7)
        df_result['day_cos'] = np.cos(2 * np.pi * df_result['day_of_week'] / 7)
        
        # Período do dia
        df_result['period_of_day'] = pd.cut(
            df_result['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['madrugada', 'manha', 'tarde', 'noite']
        )
    
    # Repetir processo para DATA se existir
    if 'DATA' in df_result.columns:
        if not pd.api.types.is_datetime64_dtype(df_result['DATA']):
            df_result['DATA'] = pd.to_datetime(df_result['DATA'], errors='coerce')
        
        # Extrair componentes apenas se conversão foi bem-sucedida
        if not df_result['DATA'].isna().all():
            df_result['utm_hour'] = df_result['DATA'].dt.hour
            df_result['utm_day_of_week'] = df_result['DATA'].dt.dayofweek
            df_result['utm_month'] = df_result['DATA'].dt.month
            df_result['utm_year'] = df_result['DATA'].dt.year
    
    return df_result, params

def encode_categorical_features(df, fit=True, params=None):
    """Codifica variáveis categóricas usando diferentes estratégias.
    
    Args:
        df: DataFrame pandas
        fit: Se True, aprende mapeamentos, caso contrário usa mapeamentos existentes
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame com variáveis categóricas codificadas
        Dicionário com parâmetros atualizados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    if 'categorical_encoding' not in params:
        params['categorical_encoding'] = {}
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # 1. Mapas para variáveis ordinais
    age_map = {
        '18 años a 24 años': 1,
        '25 años a 34 años': 2,
        '35 años a 44 años': 3,
        '45 años a 54 años': 4,
        'Mas de 54': 5,
        'desconhecido': 0
    }
    
    time_map = {
        'Te acabo de conocer a través d...': 0,
        'Te sigo desde hace 1 mes': 1,
        'Te sigo desde hace 3 meses': 2,
        'Te sigo desde hace más de 5 me...': 3,
        'Te sigo desde hace 1 año': 4,
        'Te sigo hace más de 1 año': 5,
        'desconhecido': -1
    }
    
    availability_map = {
        'Menos de 1 hora al día': 0,
        '1 hora al día': 1,
        '2 horas al día': 2,
        '3 horas al día': 3,
        'Más de 3 horas al día': 4,
        'desconhecido': -1
    }
    
    salary_map = {
        'Menos de US$3000': 1,
        'US$3000 a US$5000': 2,
        'US$5000 o más': 3,
        'US$10000 o más': 4,
        'US$20000 o más': 5,
        'desconhecido': 0
    }
    
    desired_salary_map = {
        'Al menos US$ 3000 por año': 1,
        'Más de US$5000 por año': 2,
        'Más de US$10000 por año': 3,
        'Más de US$20000 por año': 4,
        'desconhecido': 0
    }
    
    belief_map = {
        'Creo que no...': 0,
        'Tal vez': 1,
        '¡Sí, sin duda!': 2,
        '¡Si por su puesto!': 2,  # Variação ortográfica do sim
        'desconhecido': -1
    }
    
    gender_map = {
        'Mujer': 1, 
        'Hombre': 0, 
        'desconhecido': -1
    }
    
    # Guardar mapas no params se estivermos no modo fit
    if fit:
        params['categorical_encoding']['age_map'] = age_map
        params['categorical_encoding']['time_map'] = time_map
        params['categorical_encoding']['availability_map'] = availability_map
        params['categorical_encoding']['salary_map'] = salary_map
        params['categorical_encoding']['desired_salary_map'] = desired_salary_map
        params['categorical_encoding']['belief_map'] = belief_map
        params['categorical_encoding']['gender_map'] = gender_map
    
    # 2. Aplicar mapas para variáveis ordinais
    if '¿Cuál es tu edad?' in df_result.columns:
        df_result['age_encoded'] = df_result['¿Cuál es tu edad?'].map(params['categorical_encoding']['age_map'])
    
    if '¿Hace quánto tiempo me conoces?' in df_result.columns:
        df_result['time_known_encoded'] = df_result['¿Hace quánto tiempo me conoces?'].map(params['categorical_encoding']['time_map'])
    
    if '¿Cuál es tu disponibilidad de tiempo para estudiar inglés?' in df_result.columns:
        df_result['availability_encoded'] = df_result['¿Cuál es tu disponibilidad de tiempo para estudiar inglés?'].map(params['categorical_encoding']['availability_map'])
    
    if '¿Cuál es tu sueldo anual? (en dólares)' in df_result.columns:
        df_result['current_salary_encoded'] = df_result['¿Cuál es tu sueldo anual? (en dólares)'].map(params['categorical_encoding']['salary_map'])
    
    if '¿Cuánto te gustaría ganar al año?' in df_result.columns:
        df_result['desired_salary_encoded'] = df_result['¿Cuánto te gustaría ganar al año?'].map(params['categorical_encoding']['desired_salary_map'])
    
    # Ambas colunas de crença
    for col in ['¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?', 
                '¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?']:
        if col in df_result.columns:
            new_col = 'belief_salary_encoded' if 'salario' in col else 'belief_work_encoded'
            df_result[new_col] = df_result[col].map(params['categorical_encoding']['belief_map'])
    
    # Gênero
    if '¿Cuál es tu género?' in df_result.columns:
        df_result['gender_encoded'] = df_result['¿Cuál es tu género?'].map(params['categorical_encoding']['gender_map'])
    
    # 3. Encoding para variáveis nominais de alta cardinalidade
    nominal_high_cardinality = ['¿Cual es tu país?', '¿Cuál es tu profesión?']
    
    for col in nominal_high_cardinality:
        if col in df_result.columns:
            # Frequency Encoding
            if fit:
                freq_map = df_result[col].value_counts(normalize=True).to_dict()
                params['categorical_encoding'][f'{col}_freq_map'] = freq_map
            else:
                # Use stored frequency map
                freq_map = params['categorical_encoding'].get(f'{col}_freq_map', {})
            
            col_name = 'country_freq' if 'país' in col else 'profession_freq'
            df_result[col_name] = df_result[col].map(freq_map)
            
            # Agrupar categorias raras
            threshold = 0.01  # 1% de frequência
            if fit:
                rare_categories = [cat for cat, freq in freq_map.items() if freq < threshold]
                params['categorical_encoding'][f'{col}_rare_categories'] = rare_categories
            else:
                rare_categories = params['categorical_encoding'].get(f'{col}_rare_categories', [])
            
            # Criar variável agrupada
            grouped_col = col + '_grouped'
            df_result[grouped_col] = df_result[col].apply(lambda x: 'Rare' if x in rare_categories else x)
            
            # Label Encoding para categoria agrupada
            encoded_col = 'country_encoded' if 'país' in col else 'profession_encoded'
            
            if fit:
                le = LabelEncoder()
                df_result[encoded_col] = le.fit_transform(df_result[grouped_col])
                params['categorical_encoding'][f'{col}_label_mapping'] = dict(zip(le.classes_, range(len(le.classes_))))
            else:
                # Converter usando o mapeamento aprendido anteriormente
                mapping = params['categorical_encoding'].get(f'{col}_label_mapping', {})
                df_result[encoded_col] = df_result[grouped_col].map(mapping).fillna(-1).astype(int)
    
    # 4. Tratamento de UTMs
    utm_cols = [col for col in df_result.columns if 'UTM_' in col or 'utm_' in col]
    
    for col in utm_cols:
        if col in df_result.columns:
            # Verificar cardinalidade
            cardinality = df_result[col].nunique()
            
            if cardinality <= 10:  # Baixa cardinalidade
                # Label Encoding
                if fit:
                    le = LabelEncoder()
                    df_result[f'{col}_encoded'] = le.fit_transform(df_result[col].fillna('unknown').astype(str))
                    params['categorical_encoding'][f'{col}_label_mapping'] = dict(zip(le.classes_, range(len(le.classes_))))
                else:
                    # Converter usando o mapeamento aprendido anteriormente
                    mapping = params['categorical_encoding'].get(f'{col}_label_mapping', {})
                    df_result[f'{col}_encoded'] = df_result[col].fillna('unknown').astype(str).map(mapping).fillna(-1).astype(int)
            else:  # Alta cardinalidade
                # Frequency Encoding
                if fit:
                    freq_map = df_result[col].value_counts(normalize=True).to_dict()
                    params['categorical_encoding'][f'{col}_freq_map'] = freq_map
                else:
                    freq_map = params['categorical_encoding'].get(f'{col}_freq_map', {})
                
                df_result[f'{col}_freq'] = df_result[col].map(freq_map)
    
    # 5. GCLID como indicador binário
    if 'GCLID' in df_result.columns:
        df_result['has_gclid'] = df_result['GCLID'].notna().astype(int)
    
    return df_result, params

def feature_engineering(df, fit=True, params=None):
    """Executa todo o pipeline de engenharia de features não-textuais.
    
    Args:
        df: DataFrame pandas
        fit: Se True, aprende parâmetros, caso contrário usa parâmetros existentes
        params: Dicionário com parâmetros aprendidos na fase de fit
        
    Returns:
        DataFrame com features criadas
        Dicionário com parâmetros atualizados
    """
    # Inicializar parâmetros
    if params is None:
        params = {}
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # 1. Colunas a remover (exceto texto)
    cols_to_remove = [
        '¿Cómo te llamas?',
        '¿Cual es tu telefono?',
        '¿Cuál es tu instagram?'
    ]
    
    # Verificar quais colunas existem no dataframe
    cols_to_remove = [col for col in cols_to_remove if col in df_result.columns]
    
    # 2. Criar features
    df_result, params = create_identity_features(df_result, fit, params)
    df_result, params = create_temporal_features(df_result, fit, params)
    df_result, params = encode_categorical_features(df_result, fit, params)
    
    # 3. Remover colunas originais após criação das features
    df_result = df_result.drop(columns=cols_to_remove, errors='ignore')
    
    return df_result, params