"""
Módulo para matching com validação cruzada usando alunos TODOS.xlsx

Este módulo implementa matching em duas etapas:
1. Matching primário: pesquisa ↔ vendas (email)
2. Matching secundário validado: pesquisa ↔ alunos TODOS ↔ vendas DevClub

Garante que apenas alunos DevClub confirmados sejam adicionados.
"""

import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


def normalizar_email(email):
    """Normaliza email para matching"""
    if pd.isna(email):
        return None

    email_str = str(email).strip().lower()

    # Verificar se é um email válido básico
    if '@' in email_str and email_str != 'nan' and len(email_str) > 5:
        return email_str

    return None


def fazer_matching_email_with_validation(
    df_pesquisa_v1: pd.DataFrame,
    df_vendas: pd.DataFrame,
    alunos_todos_path: str = '../data/devclub/alunos_empregados_e_pesquisa_alunos/alunos TODOS.xlsx'
) -> pd.DataFrame:
    """
    Faz matching com validação cruzada usando alunos TODOS.xlsx

    Estratégia:
    1. Matching primário: pesquisa ↔ vendas (por email)
    2. Identificar emails DevClub nas vendas
    3. Matching secundário: pesquisa ↔ alunos TODOS (por email)
    4. Validar secundário: só aceita se email está em vendas DevClub

    Args:
        df_pesquisa_v1: DataFrame de pesquisa (versão 1 pós-cutoff)
        df_vendas: DataFrame de vendas
        alunos_todos_path: Caminho para arquivo alunos TODOS.xlsx

    Returns:
        DataFrame com target adicionado (matches primários + secundários validados)
    """
    print("MATCHING COM VALIDAÇÃO CRUZADA - alunos TODOS.xlsx")
    print("=" * 70)

    df_pesquisa = df_pesquisa_v1.copy()
    df_vendas_copy = df_vendas.copy()

    # === ETAPA 1: MATCHING PRIMÁRIO (pesquisa ↔ vendas) ===
    print(f"\n📧 ETAPA 1: MATCHING PRIMÁRIO (pesquisa ↔ vendas)")
    print("-" * 70)

    # Normalizar emails da pesquisa
    emails_pesquisa = {}
    for idx, email in df_pesquisa['E-mail'].items():
        email_norm = normalizar_email(email)
        if email_norm:
            emails_pesquisa[idx] = email_norm

    # Normalizar emails das vendas
    emails_vendas_all = set()
    for email in df_vendas_copy['email']:
        email_norm = normalizar_email(email)
        if email_norm:
            emails_vendas_all.add(email_norm)

    # Matching primário
    matches_primarios = set()
    for idx, email in emails_pesquisa.items():
        if email in emails_vendas_all:
            matches_primarios.add(idx)

    print(f"  Emails únicos na pesquisa: {len(emails_pesquisa):,}")
    print(f"  Emails únicos nas vendas: {len(emails_vendas_all):,}")
    print(f"  Matches primários (pesquisa ↔ vendas): {len(matches_primarios):,}")

    # === ETAPA 2: IDENTIFICAR EMAILS DEVCLUB NAS VENDAS ===
    print(f"\n🎯 ETAPA 2: IDENTIFICANDO EMAILS DEVCLUB")
    print("-" * 70)

    produtos_devclub = [
        'DevClub - Full Stack 2025',
        'DevClub FullStack Pro - OFICIAL',
        'Formação DevClub FullStack Pro - OFICI',
        'DevClub - Full Stack 2025 - EV',
        'DevClub - FS - Vitalício',
        '[Vitalício] Formação DevClub FullStack',
        'Formação DevClub FullStack Pro - COMER',
        'DevClub Vitalício',
        'DevClub 3.0 - 2024',
        '(Desativado) DevClub 3.0 - 2024',
        '(Desativado) DevClub 3.0 - 2024 - Novo'
    ]

    # Filtrar vendas DevClub
    df_vendas_devclub = df_vendas_copy[df_vendas_copy['produto'].isin(produtos_devclub)].copy()

    # Emails DevClub (nossa lista de confiança)
    emails_devclub_vendas = set()
    for email in df_vendas_devclub['email']:
        email_norm = normalizar_email(email)
        if email_norm:
            emails_devclub_vendas.add(email_norm)

    print(f"  Produtos DevClub: {len(produtos_devclub)}")
    print(f"  Vendas DevClub: {len(df_vendas_devclub):,}")
    print(f"  Emails únicos DevClub nas vendas: {len(emails_devclub_vendas):,}")

    # === ETAPA 3: CARREGAR alunos TODOS.xlsx ===
    print(f"\n📚 ETAPA 3: CARREGANDO alunos TODOS.xlsx")
    print("-" * 70)

    if not os.path.exists(alunos_todos_path):
        print(f"  ⚠️  Arquivo não encontrado: {alunos_todos_path}")
        print(f"  Continuando apenas com matches primários...")
        emails_alunos_todos = set()
    else:
        df_alunos_todos = pd.read_excel(alunos_todos_path)

        # Normalizar emails
        emails_alunos_todos = set()
        for email in df_alunos_todos['Qual seu e-mail ?']:
            email_norm = normalizar_email(email)
            if email_norm:
                emails_alunos_todos.add(email_norm)

        print(f"  Total de registros: {len(df_alunos_todos):,}")
        print(f"  Emails únicos válidos: {len(emails_alunos_todos):,}")

    # === ETAPA 4: MATCHING SECUNDÁRIO VALIDADO ===
    print(f"\n✅ ETAPA 4: MATCHING SECUNDÁRIO VALIDADO")
    print("-" * 70)

    # Encontrar novos matches potenciais
    # São emails que:
    # 1. Estão em alunos TODOS (responderam pesquisa pós-compra)
    # 2. NÃO matchearam no primário (são novos)
    # 3. Estão confirmados como DevClub nas vendas (validação)

    matches_secundarios_validados = set()

    for idx, email in emails_pesquisa.items():
        # Pular se já matcheou no primário
        if idx in matches_primarios:
            continue

        # Verificar se está em alunos TODOS E em vendas DevClub
        if email in emails_alunos_todos and email in emails_devclub_vendas:
            matches_secundarios_validados.add(idx)

    print(f"  Emails em alunos TODOS: {len(emails_alunos_todos):,}")
    print(f"  Novos matches potenciais (em alunos TODOS mas não no primário): {len(emails_alunos_todos - set(emails_pesquisa[idx] for idx in matches_primarios)):,}")
    print(f"  Matches secundários VALIDADOS (confirmados em vendas DevClub): {len(matches_secundarios_validados):,}")

    # === ETAPA 5: CRIAR TARGET FINAL ===
    print(f"\n🎯 ETAPA 5: CONSOLIDANDO MATCHES")
    print("-" * 70)

    df_resultado = df_pesquisa.copy()
    df_resultado['target'] = 0

    # Marcar matches primários
    for idx in matches_primarios:
        df_resultado.loc[idx, 'target'] = 1

    # Marcar matches secundários validados
    for idx in matches_secundarios_validados:
        df_resultado.loc[idx, 'target'] = 1

    # === ESTATÍSTICAS FINAIS ===
    total_registros = len(df_resultado)
    total_matches = df_resultado['target'].sum()
    taxa_conversao = (total_matches / total_registros) * 100

    ganho_absoluto = len(matches_secundarios_validados)
    ganho_percentual = (ganho_absoluto / len(matches_primarios) * 100) if len(matches_primarios) > 0 else 0

    print(f"  Matches primários: {len(matches_primarios):,}")
    print(f"  Matches secundários validados: {len(matches_secundarios_validados):,}")
    print(f"  GANHO: +{ganho_absoluto:,} matches (+{ganho_percentual:.1f}%)")
    print(f"  Total de matches FINAL: {total_matches:,}")
    print(f"  Taxa de conversão: {taxa_conversao:.2f}%")

    print(f"\n" + "=" * 70)
    print("DATASET FINAL CRIADO!")
    print(f"dataset_v1_final: {len(df_resultado):,} registros, {len(df_resultado.columns)} colunas")
    print(f"Target baseado em: matching primário + validação cruzada alunos TODOS")
    print(f"✅ Todos os matches secundários foram VALIDADOS em vendas DevClub")

    logger.info(f"✅ Matching com validação concluído: {total_matches} matches (+{ganho_absoluto} via validação)")

    return df_resultado
