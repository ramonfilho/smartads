"""
Módulo para comparação de métodos de matching - PIPELINE DE TREINO.

Compara dois métodos de validação de telefone:
- Método 1: Validação original (variantes múltiplas)
- Método 2: Validação do feature engineering (robusto)

Produz dois datasets para testes futuros.
"""

import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)


def normalizar_telefone_completo(telefone):
    """Normaliza telefone brasileiro e cria todas as variantes possíveis"""
    if pd.isna(telefone):
        return set()

    # Extrair apenas dígitos
    digitos = ''.join(filter(str.isdigit, str(telefone)))

    if len(digitos) < 8:  # Muito curto para ser válido
        return set()

    variantes = set()

    # Remover código do país (55) se presente
    if digitos.startswith('55') and len(digitos) > 10:
        digitos_sem_pais = digitos[2:]
    else:
        digitos_sem_pais = digitos

    # Casos baseados no comprimento após remover código do país
    if len(digitos_sem_pais) == 11:  # Formato: DDD + 9 + 8 dígitos
        variantes.add(digitos_sem_pais)  # 37999610179
        variantes.add(digitos_sem_pais[3:])  # 99610179 (sem DDD)
        # Formato antigo (sem o 9)
        if digitos_sem_pais[2] == '9':
            formato_antigo = digitos_sem_pais[:2] + digitos_sem_pais[3:]  # 3799610179
            variantes.add(formato_antigo)
            variantes.add(formato_antigo[2:])  # 99610179 (sem DDD, sem 9)

    elif len(digitos_sem_pais) == 10:  # Formato: DDD + 8 dígitos (antigo)
        variantes.add(digitos_sem_pais)  # 3799610179
        variantes.add(digitos_sem_pais[2:])  # 99610179 (sem DDD)
        # Formato novo (com o 9)
        if digitos_sem_pais[2] in ['8', '9']:
            formato_novo = digitos_sem_pais[:2] + '9' + digitos_sem_pais[2:]  # 37999610179
            variantes.add(formato_novo)

    elif len(digitos_sem_pais) == 9:  # Formato: 9 + 8 dígitos (sem DDD)
        variantes.add(digitos_sem_pais)  # 999610179
        # Formato antigo (sem o 9)
        if digitos_sem_pais[0] == '9':
            formato_antigo = digitos_sem_pais[1:]  # 99610179
            variantes.add(formato_antigo)

    elif len(digitos_sem_pais) == 8:  # Formato: 8 dígitos (sem DDD, sem 9)
        variantes.add(digitos_sem_pais)  # 99610179
        # Formato novo (com o 9)
        if digitos_sem_pais[0] in ['8', '9']:
            formato_novo = '9' + digitos_sem_pais  # 999610179
            variantes.add(formato_novo)

    # Remover variantes muito curtas ou inválidas
    variantes_validas = {v for v in variantes if len(v) >= 8}

    return variantes_validas


def normalizar_telefone_robusto(telefone):
    """Normaliza telefone considerando notação científica e padrões brasileiros (do feature engineering)"""
    if pd.isna(telefone):
        return None

    # Converter para string e lidar com notação científica
    # CORREÇÃO: Se é float, converter diretamente para int para remover .0
    if isinstance(telefone, float):
        tel_str = str(int(telefone))
    else:
        tel_str = str(telefone)

    # Se está em notação científica, converter para número inteiro
    if 'e+' in tel_str.lower() or 'E+' in tel_str:
        try:
            tel_str = str(int(float(tel_str)))
        except:
            pass

    # Extrair apenas dígitos
    digitos = re.sub(r'\D', '', tel_str)

    if len(digitos) < 8:
        return None

    # Remover código do país (55) se presente
    if digitos.startswith('55') and len(digitos) > 10:
        digitos = digitos[2:]

    # Verificar se é um telefone válido brasileiro
    if len(digitos) in [10, 11]:  # DDD + 8 ou 9 dígitos
        return digitos
    elif len(digitos) in [8, 9]:  # Sem DDD
        return digitos

    return None


def normalizar_email(email):
    """Normaliza email para matching"""
    if pd.isna(email):
        return None

    email_str = str(email).strip().lower()

    # Verificar se é um email válido básico
    if '@' in email_str and email_str != 'nan' and len(email_str) > 5:
        return email_str

    return None


def fazer_matching_comparacao_metodos(df_pesquisa_v1: pd.DataFrame, df_vendas: pd.DataFrame):
    """
    Faz matching comparando dois métodos de validação de telefone.

    Método 1: Validação original (variantes múltiplas)
    Método 2: Validação do feature engineering (robusto)

    Args:
        df_pesquisa_v1: DataFrame de pesquisa (versão 1 pós-cutoff)
        df_vendas: DataFrame de vendas unificado

    Returns:
        Tupla (dataset_metodo1, dataset_metodo2)
    """
    print("MATCHING ROBUSTO POR EMAIL E TELEFONE")
    print("COMPARANDO MÉTODOS DE VALIDAÇÃO DE TELEFONE")
    print("=" * 80)

    df_pesquisa = df_pesquisa_v1.copy()
    df_vendas_copy = df_vendas.copy()

    print(f"Processando dataset...")

    # 1. NORMALIZAR EMAILS (igual para ambos métodos)
    emails_pesquisa = {}
    for idx, email in df_pesquisa['E-mail'].items():
        email_norm = normalizar_email(email)
        if email_norm:
            emails_pesquisa[idx] = email_norm

    emails_vendas = set()
    for email in df_vendas_copy['email']:
        email_norm = normalizar_email(email)
        if email_norm:
            emails_vendas.add(email_norm)

    print(f"\n" + "="*50)
    print("MÉTODO 1: VALIDAÇÃO ORIGINAL (VARIANTES MÚLTIPLAS)")
    print("="*50)

    print(f"  Emails únicos na pesquisa: {len(emails_pesquisa):,}")
    print(f"  Emails únicos nas vendas: {len(emails_vendas):,}")

    # 2. MÉTODO 1: NORMALIZAR TELEFONES (ORIGINAL)
    telefones_pesquisa_m1 = {}
    telefones_validos_count_m1 = 0
    for idx, telefone in df_pesquisa['Telefone'].items():
        variantes = normalizar_telefone_completo(telefone)
        if variantes:
            telefones_pesquisa_m1[idx] = variantes
            telefones_validos_count_m1 += 1

    telefones_vendas_m1 = set()
    for telefone in df_vendas_copy['telefone']:
        variantes = normalizar_telefone_completo(telefone)
        telefones_vendas_m1.update(variantes)

    # Calcular percentual de telefones válidos - Método 1
    total_telefones_pesquisa = len(df_pesquisa)
    pct_telefones_validos_m1 = (telefones_validos_count_m1 / total_telefones_pesquisa) * 100

    print(f"  Telefones válidos na pesquisa: {telefones_validos_count_m1:,}/{total_telefones_pesquisa:,} ({pct_telefones_validos_m1:.1f}%)")
    print(f"  Variantes de telefone nas vendas: {len(telefones_vendas_m1):,}")

    # 3. FAZER MATCHING - MÉTODO 1
    matches_email_m1 = set()
    matches_telefone_m1 = set()

    # Matching por email
    for idx, email in emails_pesquisa.items():
        if email in emails_vendas:
            matches_email_m1.add(idx)

    # Matching por telefone
    for idx, variantes in telefones_pesquisa_m1.items():
        if variantes & telefones_vendas_m1:  # Interseção de conjuntos
            matches_telefone_m1.add(idx)

    total_matches_m1 = len(matches_email_m1 | matches_telefone_m1)
    matches_apenas_email_m1 = len(matches_email_m1 - matches_telefone_m1)
    matches_apenas_telefone_m1 = len(matches_telefone_m1 - matches_email_m1)
    matches_ambos_m1 = len(matches_email_m1 & matches_telefone_m1)
    taxa_conversao_m1 = (total_matches_m1 / total_telefones_pesquisa) * 100

    print(f"  Total de matches: {total_matches_m1:,}")
    print(f"  Taxa de conversão: {taxa_conversao_m1:.2f}%")
    print(f"  Matches apenas por email: {matches_apenas_email_m1:,}")
    print(f"  Matches apenas por telefone: {matches_apenas_telefone_m1:,}")
    print(f"  Matches por ambos: {matches_ambos_m1:,}")

    print(f"\n" + "="*50)
    print("MÉTODO 2: VALIDAÇÃO DO FEATURE ENGINEERING (ROBUSTO)")
    print("="*50)

    # 4. MÉTODO 2: NORMALIZAR TELEFONES (DO FEATURE ENGINEERING)
    telefones_pesquisa_m2 = {}
    telefones_validos_count_m2 = 0
    for idx, telefone in df_pesquisa['Telefone'].items():
        telefone_norm = normalizar_telefone_robusto(telefone)
        if telefone_norm:
            telefones_pesquisa_m2[idx] = telefone_norm
            telefones_validos_count_m2 += 1

    telefones_vendas_m2 = set()
    for telefone in df_vendas_copy['telefone']:
        telefone_norm = normalizar_telefone_robusto(telefone)
        if telefone_norm:
            telefones_vendas_m2.add(telefone_norm)

    # Calcular percentual de telefones válidos - Método 2
    pct_telefones_validos_m2 = (telefones_validos_count_m2 / total_telefones_pesquisa) * 100

    print(f"  Telefones válidos na pesquisa: {telefones_validos_count_m2:,}/{total_telefones_pesquisa:,} ({pct_telefones_validos_m2:.1f}%)")
    print(f"  Telefones únicos nas vendas: {len(telefones_vendas_m2):,}")

    # 5. FAZER MATCHING - MÉTODO 2
    matches_email_m2 = matches_email_m1.copy()  # Email é igual nos dois métodos
    matches_telefone_m2 = set()

    # Matching por telefone - método 2 (comparação direta)
    for idx, telefone_norm in telefones_pesquisa_m2.items():
        if telefone_norm in telefones_vendas_m2:
            matches_telefone_m2.add(idx)

    total_matches_m2 = len(matches_email_m2 | matches_telefone_m2)
    matches_apenas_email_m2 = len(matches_email_m2 - matches_telefone_m2)
    matches_apenas_telefone_m2 = len(matches_telefone_m2 - matches_email_m2)
    matches_ambos_m2 = len(matches_email_m2 & matches_telefone_m2)
    taxa_conversao_m2 = (total_matches_m2 / total_telefones_pesquisa) * 100

    print(f"  Total de matches: {total_matches_m2:,}")
    print(f"  Taxa de conversão: {taxa_conversao_m2:.2f}%")
    print(f"  Matches apenas por email: {matches_apenas_email_m2:,}")
    print(f"  Matches apenas por telefone: {matches_apenas_telefone_m2:,}")
    print(f"  Matches por ambos: {matches_ambos_m2:,}")

    print(f"\n" + "="*50)
    print("COMPARAÇÃO DOS MÉTODOS")
    print("="*50)

    diferenca_telefones_validos = telefones_validos_count_m2 - telefones_validos_count_m1
    diferenca_matches_total = total_matches_m2 - total_matches_m1
    diferenca_matches_telefone = len(matches_telefone_m2) - len(matches_telefone_m1)

    print(f"Diferença em telefones válidos: {diferenca_telefones_validos:+,} ({pct_telefones_validos_m2:.1f}% vs {pct_telefones_validos_m1:.1f}%)")
    print(f"Diferença em matches totais: {diferenca_matches_total:+,} ({taxa_conversao_m2:.2f}% vs {taxa_conversao_m1:.2f}%)")
    print(f"Diferença em matches por telefone: {diferenca_matches_telefone:+,}")

    # 6. CRIAR DOIS DATASETS FINAIS
    print(f"\n" + "="*50)
    print("CRIANDO DOIS DATASETS FINAIS")
    print("="*50)

    # Dataset Método 1 (Original)
    df_resultado_m1 = df_pesquisa.copy()
    df_resultado_m1['target'] = 0
    for idx in matches_email_m1 | matches_telefone_m1:
        df_resultado_m1.loc[idx, 'target'] = 1

    # Dataset Método 2 (Feature Engineering)
    df_resultado_m2 = df_pesquisa.copy()
    df_resultado_m2['target'] = 0
    for idx in matches_email_m2 | matches_telefone_m2:
        df_resultado_m2.loc[idx, 'target'] = 1

    print(f"Dataset Método 1 (Original): {len(df_resultado_m1):,} registros, {len(df_resultado_m1.columns)} colunas")
    print(f"  Target positivo: {df_resultado_m1['target'].sum():,} ({df_resultado_m1['target'].mean()*100:.2f}%)")

    print(f"Dataset Método 2 (Feature Engineering): {len(df_resultado_m2):,} registros, {len(df_resultado_m2.columns)} colunas")
    print(f"  Target positivo: {df_resultado_m2['target'].sum():,} ({df_resultado_m2['target'].mean()*100:.2f}%)")

    print("Ambos os datasets contêm colunas originais + target")

    logger.info(f"✅ Comparação de métodos concluída")

    return df_resultado_m1, df_resultado_m2
