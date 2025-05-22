"""
Funções para normalização de colunas entre diferentes lançamentos.
VERSÃO FINAL - baseada no alinhamento de comportamentos esperados.
"""

import pandas as pd
import numpy as np

# Dicionário de mapeamento - FINAL
COLUMN_MAPPINGS = {
    # === NORMALIZAÇÃO COLUNAS DE QUALIDADE ===
    'Qualidade (Nome) ': 'Qualidade (Nome)',
    'Qualidade (nome)': 'Qualidade (Nome)',
    'Qualidade (Número) ': 'Qualidade (Número)',
    'Qualidade (número)': 'Qualidade (Número)',
    'Qualidade (Numero)': 'Qualidade (Número)',
    
    # === PERGUNTA INMERSIÓN - UNIFICAR ===
    '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?': '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
    '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?': '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
}

# Colunas para remover - BASEADO NO ALINHAMENTO
COLUMNS_TO_REMOVE = [
    # Colunas de teste/residuais
    'teste', 'prediction', 'class', 'Prediction', 'Decil', 'Probabilidade',
    
    # Colunas não disponíveis na inferência
    'lançamento', 'lançamento_utm',
    
    # UTM_CAMPAIGN duplicatas (manter apenas UTM_CAMPAING)
    'UTM_CAMPAIGN (VINI)', 'UTM_CAMPAIGN2', 'UTM_CAMPAIGN(CORRECTA)',
    
    # Coluna genérica problemática
    'Campanhas ',  # Com espaço no final
]

# Features novas do L22
L22_NEW_FEATURES = [
    '¿Cuáles son tus principales razones para aprender inglés?',
    '¿Has comprado algún curso para aprender inglés antes?'
]

def consolidate_data_columns(df, launch_id=None):
    """
    Consolida colunas de DATA duplicadas (Data → DATA).
    
    Args:
        df: DataFrame para processar
        launch_id: ID do lançamento (para logs)
        
    Returns:
        DataFrame com DATA consolidada
    """
    df_result = df.copy()
    
    # Consolidar Data → DATA
    if 'Data' in df_result.columns and 'DATA' in df_result.columns:
        # Mesclar dados: preencher NaN em DATA com valores de Data
        mask = df_result['DATA'].isna() & df_result['Data'].notna()
        count_merged = mask.sum()
        df_result.loc[mask, 'DATA'] = df_result.loc[mask, 'Data']
        
        # Remover coluna Data
        df_result = df_result.drop(columns=['Data'])
        
        if launch_id and count_merged > 0:
            print(f"  🔗 {launch_id} - Consolidado {count_merged} registros: Data → DATA")
    
    elif 'Data' in df_result.columns and 'DATA' not in df_result.columns:
        # Renomear Data para DATA se DATA não existir
        df_result = df_result.rename(columns={'Data': 'DATA'})
        
        if launch_id:
            print(f"  🔄 {launch_id} - Renomeado: Data → DATA")
    
    return df_result

def consolidate_utm_campaign_columns(df, launch_id=None):
    """
    Consolida colunas UTM_CAMPAIGN duplicadas mantendo apenas UTM_CAMPAING.
    
    Args:
        df: DataFrame para processar
        launch_id: ID do lançamento (para logs)
        
    Returns:
        DataFrame com UTM_CAMPAIGN consolidado
    """
    df_result = df.copy()
    
    # Lista de colunas UTM_CAMPAIGN duplicadas
    duplicate_utm_cols = [
        'UTM_CAMPAIGN (VINI)',
        'UTM_CAMPAIGN2', 
        'UTM_CAMPAIGN(CORRECTA)'
    ]
    
    consolidated_count = 0
    removed_cols = []
    
    for dup_col in duplicate_utm_cols:
        if dup_col in df_result.columns:
            if 'UTM_CAMPAING' in df_result.columns:
                # Mesclar dados: preencher NaN em UTM_CAMPAING com valores da duplicata
                mask = df_result['UTM_CAMPAING'].isna() & df_result[dup_col].notna()
                count_merged = mask.sum()
                consolidated_count += count_merged
                df_result.loc[mask, 'UTM_CAMPAING'] = df_result.loc[mask, dup_col]
            
            # Remover coluna duplicada
            df_result = df_result.drop(columns=[dup_col])
            removed_cols.append(dup_col)
    
    if launch_id and (consolidated_count > 0 or removed_cols):
        print(f"  🔗 {launch_id} - UTM_CAMPAIGN consolidado:")
        if consolidated_count > 0:
            print(f"      Registros mesclados: {consolidated_count}")
        if removed_cols:
            print(f"      Colunas removidas: {removed_cols}")
    
    return df_result

def apply_column_mappings(df, launch_id=None):
    """Aplica mapeamentos de nome de colunas."""
    df_result = df.copy()
    
    mappings_applied = []
    for old_name, new_name in COLUMN_MAPPINGS.items():
        if old_name in df_result.columns:
            df_result = df_result.rename(columns={old_name: new_name})
            mappings_applied.append(f"{old_name} → {new_name}")
    
    if mappings_applied and launch_id:
        print(f"  📝 {launch_id} - Mapeamentos aplicados:")
        for mapping in mappings_applied:
            print(f"     {mapping}")
    
    return df_result

def handle_quality_column_special_case(df, launch_id=None):
    """Trata caso especial da coluna 'Qualidade' simples do L16 e L17."""
    df_result = df.copy()
    
    has_simple_quality = 'Qualidade' in df_result.columns
    has_quality_nome = 'Qualidade (Nome)' in df_result.columns
    has_quality_numero = 'Qualidade (Número)' in df_result.columns
    
    if has_simple_quality and not (has_quality_nome or has_quality_numero):
        sample_values = df_result['Qualidade'].dropna().head(10).tolist()
        
        text_values = [val for val in sample_values if isinstance(val, str)]
        numeric_values = [val for val in sample_values if isinstance(val, (int, float))]
        
        if len(text_values) >= len(numeric_values):
            df_result['Qualidade (Nome)'] = df_result['Qualidade']
            df_result['Qualidade (Número)'] = np.nan
            df_result = df_result.drop(columns=['Qualidade'])
            
            if launch_id:
                print(f"  🔧 {launch_id} - 'Qualidade' → 'Qualidade (Nome)'")
                print(f"      Valores exemplo: {text_values[:3]}")
        else:
            df_result['Qualidade (Número)'] = df_result['Qualidade']
            df_result['Qualidade (Nome)'] = None
            df_result = df_result.drop(columns=['Qualidade'])
            
            if launch_id:
                print(f"  🔧 {launch_id} - 'Qualidade' → 'Qualidade (Número)'")
                print(f"      Valores exemplo: {numeric_values[:3]}")
    
    return df_result

def ensure_l22_features_exist(df, launch_id=None):
    """Garante que as features novas do L22 existam em todos os datasets."""
    df_result = df.copy()
    features_added = []
    
    for feature in L22_NEW_FEATURES:
        if feature not in df_result.columns:
            df_result[feature] = np.nan
            features_added.append(feature)
    
    if features_added and launch_id:
        print(f"  ➕ {launch_id} - Features L22 adicionadas (NaN): {len(features_added)}")
    
    return df_result

def remove_unwanted_columns(df, launch_id=None):
    """Remove colunas indesejadas baseado no alinhamento."""
    df_result = df.copy()
    
    cols_to_remove = [col for col in COLUMNS_TO_REMOVE if col in df_result.columns]
    
    if cols_to_remove:
        df_result = df_result.drop(columns=cols_to_remove)
        
        if launch_id:
            print(f"  🗑️  {launch_id} - Colunas removidas: {cols_to_remove}")
    
    return df_result

def normalize_survey_columns(df, launch_id=None):
    """
    Função principal para normalizar colunas de um DataFrame de pesquisa.
    VERSÃO FINAL baseada no alinhamento de comportamentos.
    """
    if launch_id:
        print(f"\n🔄 Normalizando colunas do {launch_id}...")
        print(f"   Colunas originais: {df.shape[1]}")
    
    # 1. Consolidar colunas de DATA duplicadas
    df_normalized = consolidate_data_columns(df, launch_id)
    
    # 2. Consolidar colunas UTM_CAMPAIGN duplicadas
    df_normalized = consolidate_utm_campaign_columns(df_normalized, launch_id)
    
    # 3. Aplicar mapeamentos de nomes
    df_normalized = apply_column_mappings(df_normalized, launch_id)
    
    # 4. Tratar caso especial da coluna Qualidade
    df_normalized = handle_quality_column_special_case(df_normalized, launch_id)
    
    # 5. Garantir que features do L22 existam
    df_normalized = ensure_l22_features_exist(df_normalized, launch_id)
    
    # 6. Remover colunas indesejadas (EXCETO colunas vazias - script 02 trata)
    df_normalized = remove_unwanted_columns(df_normalized, launch_id)
    
    if launch_id:
        print(f"   Colunas finais: {df_normalized.shape[1]}")
        
        # Mostrar colunas importantes
        quality_cols = [col for col in df_normalized.columns if 'qualidade' in col.lower()]
        if quality_cols:
            print(f"   Colunas de qualidade: {quality_cols}")
        
        email_cols = [col for col in df_normalized.columns if 'email' in col.lower() or 'e-mail' in col.lower()]
        if email_cols:
            print(f"   Colunas de email: {email_cols}")
        
        data_cols = [col for col in df_normalized.columns if col in ['DATA', 'Marca temporal']]
        if data_cols:
            print(f"   Colunas de data: {data_cols}")
    
    return df_normalized

def validate_normalized_columns(df, launch_id=None):
    """Validação básica do resultado da normalização."""
    if launch_id:
        print(f"\n✅ Validação {launch_id}:")
        
        # Verificar colunas críticas
        critical_cols = ['target', 'email_norm']
        missing_critical = [col for col in critical_cols if col not in df.columns]
        
        if missing_critical:
            print(f"   ⚠️  Colunas críticas faltando: {missing_critical}")
        else:
            print(f"   ✅ Colunas críticas presentes")
        
        # Verificar duplicatas de DATA
        if 'Data' in df.columns and 'DATA' in df.columns:
            print(f"   ⚠️  Ainda existem duplicatas de DATA")
        
        # Verificar duplicatas de UTM_CAMPAIGN
        utm_duplicates = [col for col in df.columns if 'utm_campaign' in col.lower() and col != 'UTM_CAMPAING']
        if utm_duplicates:
            print(f"   ⚠️  Ainda existem duplicatas UTM_CAMPAIGN: {utm_duplicates}")
    
    return True