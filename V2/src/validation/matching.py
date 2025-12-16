"""
Módulo para matching (vinculação) entre leads e vendas.

Vincula leads captados com vendas realizadas usando:
1. Match primário: Email exato
2. Match secundário: Telefone exato
3. Validação temporal: Venda após captura do lead
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def match_leads_to_sales(
    leads_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    use_temporal_validation: bool = False
) -> pd.DataFrame:
    """
    Vincula leads com vendas usando email e telefone.

    Lógica de matching:
    1. Match por email (normalizado)
    2. Match por telefone (se email não bateu)
    3. Venda deve ocorrer APÓS captura do lead
    4. [Opcional] Validação temporal pode ser desabilitada para análise de resultados

    Args:
        leads_df: DataFrame de leads (data_loader.LeadDataLoader)
        sales_df: DataFrame de vendas (data_loader.SalesDataLoader)
        use_temporal_validation: Se True, aplica janela de 30 dias. Se False, sem restrição temporal.

    Returns:
        DataFrame de leads com colunas adicionadas:
        - converted: bool (True se vendeu)
        - sale_value: float (valor da venda, ou 0)
        - sale_date: datetime (data da venda, ou None)
        - sale_origin: str ('guru', 'tmb', ou None)
        - match_method: str ('email', 'telefone', ou None)
    """
    mode = "com validação temporal (30 dias)" if use_temporal_validation else "sem validação temporal"
    logger.info(f"🔗 Iniciando matching ({mode})")
    logger.info(f"   Leads: {len(leads_df)}")
    logger.info(f"   Vendas: {len(sales_df)}")

    # Criar cópias para não alterar DataFrames originais
    leads = leads_df.copy()
    sales = sales_df.copy()

    # Inicializar colunas de resultado
    leads['converted'] = False
    leads['sale_value'] = 0.0
    leads['sale_date'] = pd.NaT
    leads['sale_origin'] = None
    leads['match_method'] = None

    # Contadores de matching
    matched_by_email = 0
    matched_by_phone = 0

    # Criar dicionários de vendas por email e telefone para lookup rápido
    sales_by_email = {}
    sales_by_phone = {}

    for idx, sale in sales.iterrows():
        email = sale['email']
        phone = sale['telefone']
        sale_data = {
            'sale_value': sale['sale_value'],
            'sale_date': sale['sale_date'],
            'sale_origin': sale['origem']
        }

        # Indexar por email
        if pd.notna(email):
            if email not in sales_by_email:
                sales_by_email[email] = []
            sales_by_email[email].append(sale_data)

        # Indexar por telefone
        if pd.notna(phone):
            if phone not in sales_by_phone:
                sales_by_phone[phone] = []
            sales_by_phone[phone].append(sale_data)

    logger.info(f"   Índices criados: {len(sales_by_email)} emails, {len(sales_by_phone)} telefones")

    # Iterar sobre leads e buscar matches
    for idx, lead in leads.iterrows():
        lead_email = lead['email']
        lead_phone = lead['telefone']
        lead_date = lead['data_captura']

        if pd.isna(lead_date):
            continue  # Pular leads sem data de captura

        # Tentar match por email primeiro
        matched = False
        if pd.notna(lead_email) and lead_email in sales_by_email:
            for sale in sales_by_email[lead_email]:
                if not use_temporal_validation or _is_valid_match(lead_date, sale['sale_date']):
                    leads.at[idx, 'converted'] = True
                    leads.at[idx, 'sale_value'] = sale['sale_value']
                    leads.at[idx, 'sale_date'] = sale['sale_date']
                    leads.at[idx, 'sale_origin'] = sale['sale_origin']
                    leads.at[idx, 'match_method'] = 'email'
                    matched_by_email += 1
                    matched = True
                    break  # Primeira venda válida encontrada

        # Se não bateu por email, tentar por telefone
        if not matched and pd.notna(lead_phone) and lead_phone in sales_by_phone:
            for sale in sales_by_phone[lead_phone]:
                if not use_temporal_validation or _is_valid_match(lead_date, sale['sale_date']):
                    leads.at[idx, 'converted'] = True
                    leads.at[idx, 'sale_value'] = sale['sale_value']
                    leads.at[idx, 'sale_date'] = sale['sale_date']
                    leads.at[idx, 'sale_origin'] = sale['sale_origin']
                    leads.at[idx, 'match_method'] = 'telefone'
                    matched_by_phone += 1
                    matched = True
                    break  # Primeira venda válida encontrada

    total_matched = matched_by_email + matched_by_phone
    match_rate = (total_matched / len(leads) * 100) if len(leads) > 0 else 0

    logger.info(f"   ✅ Matching concluído:")
    logger.info(f"      Total conversões: {total_matched}")
    logger.info(f"      Por email: {matched_by_email}")
    logger.info(f"      Por telefone: {matched_by_phone}")
    logger.info(f"      Taxa de conversão geral: {match_rate:.2f}%")

    return leads


def _is_valid_match(lead_date: pd.Timestamp, sale_date: pd.Timestamp) -> bool:
    """
    Valida se uma venda é válida para um lead (apenas temporal básico).

    Critério:
    1. Venda deve ocorrer APÓS a captura do lead (ou no mesmo dia)

    Args:
        lead_date: Data de captura do lead
        sale_date: Data da venda

    Returns:
        True se é um match válido
    """
    if pd.isna(lead_date) or pd.isna(sale_date):
        return False

    # Venda deve ser no mesmo dia ou depois do lead
    return sale_date >= lead_date


def get_matching_stats(matched_df: pd.DataFrame, total_sales: int = None) -> Dict:
    """
    Calcula estatísticas sobre o matching realizado.

    Args:
        matched_df: DataFrame com resultados do matching
        total_sales: Total de vendas reais (do período), não apenas as identificadas

    Returns:
        Dicionário com estatísticas:
        - total_leads: Total de leads
        - total_sales: Total de vendas reais (do período)
        - total_conversions: Total de conversões identificadas (matched)
        - tracking_rate: Taxa de trackeamento (conversões identificadas / vendas reais * 100)
        - conversion_rate: Taxa de conversão (%)
        - matched_by_email: Conversões via email
        - matched_by_phone: Conversões via telefone
        - match_rate_email: Taxa de match por email (%)
        - match_rate_phone: Taxa de match por telefone (%)
        - total_revenue: Receita total
        - avg_ticket: Ticket médio
        - conversions_guru: Conversões da Guru
        - conversions_tmb: Conversões da TMB
    """
    total_leads = len(matched_df)
    converted = matched_df[matched_df['converted'] == True]
    total_conversions = len(converted)

    conversion_rate = (total_conversions / total_leads * 100) if total_leads > 0 else 0
    tracking_rate = (total_conversions / total_sales * 100) if (total_sales and total_sales > 0) else 100.0

    matched_by_email = len(converted[converted['match_method'] == 'email'])
    matched_by_phone = len(converted[converted['match_method'] == 'telefone'])

    match_rate_email = (matched_by_email / total_conversions * 100) if total_conversions > 0 else 0
    match_rate_phone = (matched_by_phone / total_conversions * 100) if total_conversions > 0 else 0

    total_revenue = converted['sale_value'].sum()
    avg_ticket = total_revenue / total_conversions if total_conversions > 0 else 0

    conversions_guru = len(converted[converted['sale_origin'] == 'guru'])
    conversions_tmb = len(converted[converted['sale_origin'] == 'tmb'])

    return {
        'total_leads': total_leads,
        'total_sales': total_sales if total_sales else total_conversions,
        'total_conversions': total_conversions,
        'tracking_rate': round(tracking_rate, 2),
        'conversion_rate': round(conversion_rate, 2),
        'matched_by_email': matched_by_email,
        'matched_by_phone': matched_by_phone,
        'match_rate_email': round(match_rate_email, 2),
        'match_rate_phone': round(match_rate_phone, 2),
        'total_revenue': round(total_revenue, 2),
        'avg_ticket': round(avg_ticket, 2),
        'conversions_guru': conversions_guru,
        'conversions_tmb': conversions_tmb,
    }


def print_matching_summary(stats: Dict):
    """
    Imprime resumo visual das estatísticas de matching.

    Args:
        stats: Dicionário retornado por get_matching_stats()
    """
    print("\n" + "=" * 80)
    print("📊 RESUMO DO MATCHING")
    print("=" * 80)
    print(f"\n📈 Leads e Conversões:")
    print(f"   Total de leads: {stats['total_leads']:,}")
    print(f"   Total de conversões: {stats['total_conversions']:,}")
    print(f"   Taxa de conversão: {stats['conversion_rate']:.2f}%")

    print(f"\n🔗 Método de Match:")
    print(f"   Por email: {stats['matched_by_email']:,} ({stats['match_rate_email']:.1f}%)")
    print(f"   Por telefone: {stats['matched_by_phone']:,} ({stats['match_rate_phone']:.1f}%)")

    print(f"\n💰 Receita:")
    print(f"   Total: R$ {stats['total_revenue']:,.2f}")
    print(f"   Ticket médio: R$ {stats['avg_ticket']:,.2f}")

    print(f"\n🏪 Origem das Vendas:")
    print(f"   Guru: {stats['conversions_guru']:,} conversões")
    print(f"   TMB: {stats['conversions_tmb']:,} conversões")
    print("=" * 80)


def filter_by_period(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    date_col: str = 'data_captura'
) -> pd.DataFrame:
    """
    Filtra DataFrame por período de datas.

    Args:
        df: DataFrame a filtrar
        start_date: Data início (formato: 'YYYY-MM-DD')
        end_date: Data fim (formato: 'YYYY-MM-DD')
        date_col: Nome da coluna de data

    Returns:
        DataFrame filtrado

    Examples:
        >>> df = pd.DataFrame({
        ...     'data_captura': pd.to_datetime(['2025-11-10', '2025-11-15', '2025-12-05'])
        ... })
        >>> filtered = filter_by_period(df, '2025-11-11', '2025-12-01')
        >>> len(filtered)
        1
    """
    if date_col not in df.columns:
        raise ValueError(f"Coluna '{date_col}' não encontrada no DataFrame")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Ajustar end para incluir o dia inteiro (até 23:59:59.999)
    # Quando passamos '2025-11-03', pd.to_datetime converte para '2025-11-03 00:00:00'
    # Precisamos incluir TODO o dia 03/11, então usamos < próximo dia
    end_inclusive = end + pd.Timedelta(days=1)

    logger.info(f"📅 Filtrando por período: {start_date} a {end_date} (dia inteiro)")

    before = len(df)
    df_filtered = df[
        (df[date_col] >= start) &
        (df[date_col] < end_inclusive)  # < próximo dia para incluir todo o end_date
    ].copy()
    after = len(df_filtered)

    logger.info(f"   {before} → {after} registros ({after/before*100:.1f}%)" if before > 0 else "   0 registros")

    return df_filtered


def filter_conversions_by_capture_period(
    matched_df: pd.DataFrame,
    period_start: str,
    period_end: str
) -> pd.DataFrame:
    """
    Remove conversões de leads captados FORA do período de análise.

    Quando um lead foi captado antes do período mas converteu durante o período,
    essa conversão NÃO deve ser atribuída às campanhas do período de análise.

    Args:
        matched_df: DataFrame com matching realizado
        period_start: Data início do período de captação (formato: 'YYYY-MM-DD')
        period_end: Data fim do período de captação (formato: 'YYYY-MM-DD')

    Returns:
        DataFrame com conversões apenas de leads captados no período

    Example:
        >>> # Lead captado 06/11, convertido 30/11, período 18-24/11 → excluído
        >>> filtered = filter_conversions_by_capture_period(matched_df, '2025-11-18', '2025-11-24')
    """
    logger.info(f"📅 Filtrando conversões por período de captura: {period_start} a {period_end}")

    start = pd.to_datetime(period_start)
    end = pd.to_datetime(period_end)
    end_inclusive = end + pd.Timedelta(days=1)  # Incluir o dia inteiro

    # Separar convertidos de não-convertidos
    conversions = matched_df[matched_df['converted'] == True].copy()
    non_conversions = matched_df[matched_df['converted'] == False].copy()

    before_filter = len(conversions)

    # Filtrar conversões por data_captura dentro do período
    conversions_filtered = conversions[
        (conversions['data_captura'] >= start) &
        (conversions['data_captura'] < end_inclusive)
    ].copy()

    after_filter = len(conversions_filtered)
    removed = before_filter - after_filter

    logger.info(f"   Conversões antes: {before_filter}")
    logger.info(f"   Conversões depois: {after_filter}")
    logger.info(f"   Conversões fora do período removidas: {removed}")

    if removed > 0:
        # Identificar quais foram removidas
        removed_conversions = conversions[~conversions.index.isin(conversions_filtered.index)]
        logger.info(f"   Emails removidos (capturados fora do período):")
        for email in removed_conversions['email'].unique():
            capture_date = removed_conversions[removed_conversions['email'] == email]['data_captura'].iloc[0]
            logger.info(f"      • {email} (capturado em {capture_date.date()})")

    # Recombinar conversões filtradas + não-convertidos (mantemos todos os leads do período)
    result = pd.concat([conversions_filtered, non_conversions], ignore_index=True)

    logger.info(f"   ✅ Filtragem por período concluída")

    return result


def deduplicate_conversions(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicatas artificiais de conversões.

    Quando o mesmo email é captado múltiplas vezes e corresponde à mesma venda,
    cria-se duplicatas artificiais. Esta função mantém apenas 1 conversão por venda real.

    Lógica:
    1. Agrupa conversões por (email, sale_date, sale_value)
    2. Para cada grupo, mantém apenas o lead capturado PRIMEIRO (data_captura mais antiga)
    3. Mantém todos os leads não convertidos

    Args:
        matched_df: DataFrame com matching realizado

    Returns:
        DataFrame sem duplicatas artificiais

    Example:
        >>> # Mesmo email, 5 capturas, 1 venda → mantém apenas 1 conversão
        >>> deduplicated = deduplicate_conversions(matched_df)
    """
    logger.info("🧹 Iniciando deduplicação de conversões...")

    # Separar convertidos de não-convertidos
    conversions = matched_df[matched_df['converted'] == True].copy()
    non_conversions = matched_df[matched_df['converted'] == False].copy()

    before_dedup = len(conversions)
    logger.info(f"   Conversões antes: {before_dedup}")

    if len(conversions) == 0:
        logger.info("   Sem conversões para deduplic")
        return matched_df

    # Criar chave de agrupamento: email + sale_date + sale_value
    conversions['_dedup_key'] = (
        conversions['email'].astype(str).str.strip().str.lower() + '|' +
        conversions['sale_date'].astype(str) + '|' +
        conversions['sale_value'].astype(str)
    )

    # Para cada grupo, manter apenas o lead capturado PRIMEIRO (data_captura mais antiga)
    # Ordenar por data_captura (mais antiga primeiro) e pegar o primeiro de cada grupo
    conversions_sorted = conversions.sort_values('data_captura')
    deduplicated = conversions_sorted.drop_duplicates(subset='_dedup_key', keep='first')

    # Remover coluna auxiliar
    deduplicated = deduplicated.drop('_dedup_key', axis=1)

    after_dedup = len(deduplicated)
    removed = before_dedup - after_dedup

    logger.info(f"   Conversões depois: {after_dedup}")
    logger.info(f"   Duplicatas removidas: {removed}")

    # Recombinar convertidos (sem duplicatas) + não-convertidos
    result = pd.concat([deduplicated, non_conversions], ignore_index=True)

    logger.info(f"   ✅ Deduplicação concluída")

    return result


def analyze_conversion_by_decile(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analisa conversão por decil (preview para metrics_calculator).

    Args:
        matched_df: DataFrame com matching realizado

    Returns:
        DataFrame com análise por decil
    """
    if 'decile' not in matched_df.columns:
        logger.warning("⚠️ Coluna 'decile' não encontrada")
        return pd.DataFrame()

    df_with_decile = matched_df[matched_df['decile'].notna()].copy()

    results = []
    for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
        decile_df = df_with_decile[df_with_decile['decile'] == decile]
        total = len(decile_df)
        converted = len(decile_df[decile_df['converted'] == True])
        conversion_rate = (converted / total * 100) if total > 0 else 0

        results.append({
            'decile': decile,
            'total_leads': total,
            'conversions': converted,
            'conversion_rate': round(conversion_rate, 2)
        })

    return pd.DataFrame(results)
