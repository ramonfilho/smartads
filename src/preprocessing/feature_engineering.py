import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from src.utils.feature_naming import standardize_feature_name
from src.utils.column_type_classifier import ColumnTypeClassifier

def create_identity_features(df, fit=True, params=None):
    """Cria features baseadas nos campos de identidade do usuário.
    
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
    
    # Feature de comprimento do nome
    if '¿Cómo te llamas?' in df_result.columns:
        df_result[standardize_feature_name('name_length')] = df_result['¿Cómo te llamas?'].str.len()
        df_result[standardize_feature_name('name_word_count')] = df_result['¿Cómo te llamas?'].str.split().str.len()
    
    # Feature de validade do telefone
    if '¿Cual es tu telefono?' in df_result.columns:
        df_result[standardize_feature_name('valid_phone')] = df_result['¿Cual es tu telefono?'].str.replace(r'\D', '', regex=True).str.len() >= 8
    
    # Feature de presença de instagram
    if '¿Cuál es tu instagram?' in df_result.columns:
        df_result[standardize_feature_name('has_instagram')] = df_result['¿Cuál es tu instagram?'].notna() & (df_result['¿Cuál es tu instagram?'] != '')
    
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
    
    # DEBUG inicial
    print("\n  DEBUG create_temporal_features:")
    print(f"    Total de colunas no início: {len(df_result.columns)}")
    
    # Procurar colunas temporais
    temporal_candidates = [col for col in df_result.columns if any(word in col.lower() for word in ['temporal', 'data', 'date', 'time', 'marca'])]
    print(f"    Colunas temporais candidatas: {temporal_candidates}")
    
    # Verificar tipos das colunas candidatas
    for col in temporal_candidates:
        print(f"      - {col}: tipo = {df_result[col].dtype}")
    
    # Features de tempo baseadas em 'Marca temporal'
    marca_temporal_col = None
    
    # Tentar diferentes variações do nome
    for possible_name in ['Marca temporal', 'marca_temporal', 'Marca_temporal', 'MARCA_TEMPORAL']:
        if possible_name in df_result.columns:
            marca_temporal_col = possible_name
            print(f"    ✓ Encontrada coluna marca temporal: '{marca_temporal_col}'")
            break
    
    if marca_temporal_col:
        # Verificar tipo e conteúdo
        print(f"    Tipo atual: {df_result[marca_temporal_col].dtype}")
        print(f"    Primeiros valores: {df_result[marca_temporal_col].head(3).tolist()}")
        
        # Converter para datetime se ainda não for
        if not pd.api.types.is_datetime64_dtype(df_result[marca_temporal_col]):
            print("    Convertendo para datetime...")
            df_result[marca_temporal_col] = pd.to_datetime(df_result[marca_temporal_col], errors='coerce')
            
            # Verificar resultado da conversão
            null_count = df_result[marca_temporal_col].isna().sum()
            total_count = len(df_result)
            print(f"    Valores nulos após conversão: {null_count} de {total_count} ({null_count/total_count*100:.1f}%)")
            
            if null_count == total_count:
                print("    ❌ ERRO: Todos os valores se tornaram nulos após conversão!")
                print(f"    Amostra dos valores originais: {df[marca_temporal_col].head(3).tolist()}")
        
        # Extrair componentes básicos apenas se temos dados válidos
        valid_dates = df_result[marca_temporal_col].notna().sum()
        if valid_dates > 0:
            df_result[standardize_feature_name('hour')] = df_result[marca_temporal_col].dt.hour
            df_result[standardize_feature_name('day_of_week')] = df_result[marca_temporal_col].dt.dayofweek
            df_result[standardize_feature_name('month')] = df_result[marca_temporal_col].dt.month
            df_result[standardize_feature_name('year')] = df_result[marca_temporal_col].dt.year
            
            # Features cíclicas para hora e dia da semana
            hour_col = standardize_feature_name('hour')
            day_col = standardize_feature_name('day_of_week')
            
            df_result[standardize_feature_name('hour_sin')] = np.sin(2 * np.pi * df_result[hour_col] / 24)
            df_result[standardize_feature_name('hour_cos')] = np.cos(2 * np.pi * df_result[hour_col] / 24)
            df_result[standardize_feature_name('day_sin')] = np.sin(2 * np.pi * df_result[day_col] / 7)
            df_result[standardize_feature_name('day_cos')] = np.cos(2 * np.pi * df_result[day_col] / 7)
            
            # Período do dia
            df_result[standardize_feature_name('period_of_day')] = pd.cut(
                df_result[hour_col], 
                bins=[0, 6, 12, 18, 24], 
                labels=['madrugada', 'manha', 'tarde', 'noite']
            )
            
            print(f"    ✓ Features temporais criadas a partir de '{marca_temporal_col}'")
        else:
            print(f"    ❌ Nenhuma data válida em '{marca_temporal_col}' para criar features")
    else:
        print("    ❌ Coluna 'marca_temporal' não encontrada em nenhuma variação!")
    
    # Repetir processo para DATA se existir
    data_col = None
    for possible_name in ['DATA', 'data', 'Data']:
        if possible_name in df_result.columns:
            data_col = possible_name
            print(f"\n    ✓ Encontrada coluna data: '{data_col}'")
            break
    
    if data_col:
        print(f"    Tipo atual: {df_result[data_col].dtype}")
        
        if not pd.api.types.is_datetime64_dtype(df_result[data_col]):
            df_result[data_col] = pd.to_datetime(df_result[data_col], errors='coerce')
        
        # Extrair componentes apenas se conversão foi bem-sucedida
        if not df_result[data_col].isna().all():
            df_result[standardize_feature_name('utm_hour')] = df_result[data_col].dt.hour
            df_result[standardize_feature_name('utm_day_of_week')] = df_result[data_col].dt.dayofweek
            df_result[standardize_feature_name('utm_month')] = df_result[data_col].dt.month
            df_result[standardize_feature_name('utm_year')] = df_result[data_col].dt.year
            
            print(f"    ✓ Features UTM temporais criadas a partir de '{data_col}'")
    
    # DEBUG final - verificar o que foi criado
    print("\n    RESUMO de features temporais criadas:")
    temporal_features_created = []
    for feature in ['hour', 'day_of_week', 'month', 'year', 'hour_sin', 'hour_cos', 
                   'day_sin', 'day_cos', 'period_of_day', 'utm_hour', 'utm_day_of_week', 
                   'utm_month', 'utm_year']:
        if feature in df_result.columns:
            temporal_features_created.append(feature)
    
    print(f"    Total de features temporais: {len(temporal_features_created)}")
    if temporal_features_created:
        print(f"    Features: {temporal_features_created}")
    else:
        print("    ❌ NENHUMA feature temporal foi criada!")
    
    print(f"    Total de colunas no final: {len(df_result.columns)}")
    return df_result, params

def encode_categorical_features(df, fit=True, params=None):
    """Codifica variáveis categóricas usando diferentes estratégias."""
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
        'Te acabo de conocer a través del anuncio.': 0,
        'Te sigo desde hace 1 mes': 1,
        'Te sigo desde hace 3 meses': 2,
        'Te sigo desde hace más de 5 meses': 3,
        'Te sigo desde hace 1 año': 4,
        'Te sigo hace más de 1 año': 5,
        'Te sigo hace más de 2 años': 6,
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
        '¡Si por su puesto!': 2,
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
    
    # DEBUG para diagnóstico
    print("\n  DEBUG encode_categorical_features:")
    print(f"    Total de colunas: {len(df_result.columns)}")
    
    # 2. Aplicar mapas para variáveis ordinais - USANDO STANDARDIZE!
    
    # Idade
    age_col = standardize_feature_name('¿Cuál es tu edad?')
    if age_col in df_result.columns:
        df_result[standardize_feature_name('age_encoded')] = df_result[age_col].map(params['categorical_encoding']['age_map'])
    else:
        print(f"    ⚠️ Coluna idade não encontrada: {age_col}")
    
    # Tempo conhecido
    time_col = standardize_feature_name('¿Hace quánto tiempo me conoces?')
    if time_col in df_result.columns:
        df_result[standardize_feature_name('time_known_encoded')] = df_result[time_col].map(params['categorical_encoding']['time_map'])
    
    # Disponibilidade
    avail_col = standardize_feature_name('¿Cuál es tu disponibilidad de tiempo para estudiar inglés?')
    if avail_col in df_result.columns:
        df_result[standardize_feature_name('availability_encoded')] = df_result[avail_col].map(params['categorical_encoding']['availability_map'])
    
    # Salário atual
    salary_col = standardize_feature_name('¿Cuál es tu sueldo anual? (en dólares)')
    if salary_col in df_result.columns:
        df_result[standardize_feature_name('current_salary_encoded')] = df_result[salary_col].map(params['categorical_encoding']['salary_map'])
    
    # Salário desejado
    desired_col = standardize_feature_name('¿Cuánto te gustaría ganar al año?')
    if desired_col in df_result.columns:
        df_result[standardize_feature_name('desired_salary_encoded')] = df_result[desired_col].map(params['categorical_encoding']['desired_salary_map'])
    
    # Crenças sobre salário
    belief_salary_col = standardize_feature_name('¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?')
    if belief_salary_col in df_result.columns:
        df_result[standardize_feature_name('belief_salary_encoded')] = df_result[belief_salary_col].map(params['categorical_encoding']['belief_map'])
    
    # Crenças sobre trabalho
    belief_work_col = standardize_feature_name('¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?')
    if belief_work_col in df_result.columns:
        df_result[standardize_feature_name('belief_work_encoded')] = df_result[belief_work_col].map(params['categorical_encoding']['belief_map'])
    
    # Gênero
    gender_col = standardize_feature_name('¿Cuál es tu género?')
    if gender_col in df_result.columns:
        df_result[standardize_feature_name('gender_encoded')] = df_result[gender_col].map(params['categorical_encoding']['gender_map'])
    
    # 3. Encoding para variáveis nominais de alta cardinalidade
    
    # DEBUG para encontrar coluna de país
    print("\n    DEBUG - Procurando coluna de país:")
    country_candidates = [col for col in df_result.columns if 'pais' in col.lower() or 'país' in col.lower()]
    print(f"    Candidatas para país: {country_candidates}")
    
    # Tentar múltiplas variações para país
    country_col = None
    country_variations = [
        standardize_feature_name('¿Cual es tu país?'),
        standardize_feature_name('¿Cuál es tu país?'),
        'cual_es_tu_pais'
    ]
    
    for variation in country_variations:
        if variation in df_result.columns:
            country_col = variation
            print(f"    ✓ Encontrada coluna país: {country_col}")
            break
    
    # Profissão
    profession_col = standardize_feature_name('¿Cuál es tu profesión?')
    if profession_col not in df_result.columns:
        # Já foi removida, pular
        profession_col = None
        print("    ℹ️ Coluna profissão já foi removida")
    
    # Processar apenas colunas existentes
    nominal_high_cardinality = [col for col in [country_col, profession_col] if col is not None]
    
    for col in nominal_high_cardinality:
        if col in df_result.columns:
            # Frequency Encoding
            if fit:
                freq_map = df_result[col].value_counts(normalize=True).to_dict()
                params['categorical_encoding'][f'{col}_freq_map'] = freq_map
            else:
                # Use stored frequency map
                freq_map = params['categorical_encoding'].get(f'{col}_freq_map', {})
            
            # Determinar nome da coluna de saída
            if 'pais' in col.lower() or 'país' in col.lower():
                col_name = standardize_feature_name('country_freq')
                encoded_col = standardize_feature_name('country_encoded')
            else:
                col_name = standardize_feature_name('profession_freq')
                encoded_col = standardize_feature_name('profession_encoded')
            
            df_result[col_name] = df_result[col].map(freq_map).fillna(0)
            
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
            if fit:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                # Garantir que não há NaN
                df_result[grouped_col] = df_result[grouped_col].fillna('Unknown')
                df_result[encoded_col] = le.fit_transform(df_result[grouped_col])
                params['categorical_encoding'][f'{col}_label_mapping'] = dict(zip(le.classes_, range(len(le.classes_))))
            else:
                # Converter usando o mapeamento aprendido anteriormente
                mapping = params['categorical_encoding'].get(f'{col}_label_mapping', {})
                df_result[grouped_col] = df_result[grouped_col].fillna('Unknown')
                df_result[encoded_col] = df_result[grouped_col].map(mapping).fillna(-1).astype(int)
            
            print(f"    ✓ Criadas features para {col}: {col_name} e {encoded_col}")
    
    # 4. Tratamento de UTMs
    utm_cols = [col for col in df_result.columns if 'UTM_' in col or 'utm_' in col]
    
    for col in utm_cols:
        if col in df_result.columns:
            # Verificar cardinalidade
            cardinality = df_result[col].nunique()
            
            if cardinality <= 10:  # Baixa cardinalidade
                # Label Encoding
                if fit:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df_result[standardize_feature_name(f'{col}_encoded')] = le.fit_transform(df_result[col].fillna('unknown').astype(str))
                    params['categorical_encoding'][f'{col}_label_mapping'] = dict(zip(le.classes_, range(len(le.classes_))))
                else:
                    # Converter usando o mapeamento aprendido anteriormente
                    mapping = params['categorical_encoding'].get(f'{col}_label_mapping', {})
                    df_result[standardize_feature_name(f'{col}_encoded')] = df_result[col].fillna('unknown').astype(str).map(mapping).fillna(-1).astype(int)
            else:  # Alta cardinalidade
                # Frequency Encoding
                if fit:
                    freq_map = df_result[col].value_counts(normalize=True).to_dict()
                    params['categorical_encoding'][f'{col}_freq_map'] = freq_map
                else:
                    freq_map = params['categorical_encoding'].get(f'{col}_freq_map', {})
                
                df_result[standardize_feature_name(f'{col}_freq')] = df_result[col].map(freq_map).fillna(0)
    
    # 5. GCLID como indicador binário
    if 'GCLID' in df_result.columns or 'gclid' in df_result.columns:
        gclid_col = 'GCLID' if 'GCLID' in df_result.columns else 'gclid'
        df_result[standardize_feature_name('has_gclid')] = df_result[gclid_col].notna().astype(int)
    
    # DEBUG final
    encoded_cols = [col for col in df_result.columns if 'encoded' in col or '_freq' in col]
    print(f"\n    Total de features criadas: {len(encoded_cols)}")
    
    return df_result, params

def feature_engineering(df, fit=True, params=None):
    if params is None:
        params = {}

    # ADICIONE ESTE DEBUG TEMPORÁRIO
    print("\n=== DEBUG: RASTREAMENTO DE COLUNAS EM feature_engineering ===")
    print(f"Colunas na entrada: {len(df.columns)}")
    initial_cols = set(df.columns)
    
    # Cria uma cópia para não modificar o original
    df_result = df.copy()
    
    # Verificar se temos classificações nos params
    if 'column_classifications' in params:
        print("\n✓ Usando classificações existentes para feature engineering")
        classifications = params['column_classifications']
    else:
        # Fallback: reclassificar se necessário
        classifier = ColumnTypeClassifier(
            use_llm=False,
            use_classification_cache=True,
            confidence_threshold=0.7
        )
        classifications = classifier.classify_dataframe(df_result)
    
    print("\n  Executando pipeline de feature engineering:")
    print(f"  Colunas iniciais: {df_result.shape[1]}")
    
    # ADICIONAR DEBUG AQUI
    temporal_cols = [col for col in df_result.columns 
                    if any(term in col.lower() for term in ['temporal', 'data', 'date', 'marca'])]
    print(f"  Colunas temporais disponíveis: {temporal_cols}")

    # 1. Colunas a remover (exceto texto) - MANTER ESTE CÓDIGO!
    # Identificar colunas que são texto mas serão removidas
    text_to_remove = ['como_te_llamas', 'cual_es_tu_telefono', 'cual_es_tu_instagram', 'cual_es_tu_profesion']
    
    cols_to_remove = [
        col for col in text_to_remove 
        if col in df_result.columns 
        and col in classifications
        and classifications[col].get('type') == 'text'
    ]
    
    # Verificar quais colunas existem no dataframe
    cols_to_remove = [col for col in cols_to_remove if col in df_result.columns]
    
    # 2. Criar features
    df_result, params = create_identity_features(df_result, fit, params)
    
    # ADICIONAR DEBUG PARA TEMPORAL
    temp_before = df_result.shape[1]
    df_result, params = create_temporal_features(df_result, fit, params)
    temp_after = df_result.shape[1]
    print(f"  Features temporais criadas: {temp_after - temp_before}")
    
    # Verificar se features temporais básicas foram criadas
    temporal_basics = ['hour', 'day_of_week', 'month', 'year']
    temporal_found = [f for f in temporal_basics if f in df_result.columns]
    if not temporal_found:
        print("  ⚠️ AVISO: Nenhuma feature temporal básica foi criada!")
    else:
        print(f"  ✓ Features temporais básicas criadas: {temporal_found}")
    
    df_result, params = encode_categorical_features(df_result, fit, params)

    # DEBUG: Verificar quais colunas foram criadas
    print("\n  DEBUG - Colunas após encoding:")
    encoded_cols = [col for col in df_result.columns if 'encoded' in col]
    print(f"    Colunas encoded: {encoded_cols}")
    salary_related = [col for col in df_result.columns if 'salary' in col]
    print(f"    Colunas com 'salary': {salary_related}")

    # 3. Remover colunas originais após criação das features - MANTER!
    df_result = df_result.drop(columns=cols_to_remove, errors='ignore')

    final_cols = set(df_result.columns)
    removed_cols = initial_cols - final_cols
    added_cols = final_cols - initial_cols
    
    print(f"\nColunas removidas em feature_engineering: {len(removed_cols)}")
    for col in sorted(removed_cols):
        print(f"  - {col}")
    
    print(f"\nColunas adicionadas: {len(added_cols)}")
    
    return df_result, params