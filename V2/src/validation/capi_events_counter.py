"""
Contador de Eventos CAPI
Busca eventos LeadQualified, LeadQualifiedHighQuality dos logs do Cloud Run
e cruza com leads_capi (database ou CSV) para obter campaign IDs

Nota: 'Faixa A' é evento de outro sistema, não do nosso
"""

import subprocess
import logging
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Tentar importar database (pode falhar se não configurado localmente)
try:
    import sys
    import os
    # Adicionar path do projeto para importar api.database
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from api.database import SessionLocal, LeadCAPI
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    SessionLocal = None
    LeadCAPI = None
    logger.debug(f"Database import failed: {e}")


def extract_email_from_log_line(line: str) -> Optional[str]:
    """
    Extrai email de uma linha de log

    Formato: "✅ LeadQualified enviado: email@example.com (decil: D10, valor proj: R$ 69.10)"
    """
    match = re.search(r'enviado:\s*([^\s]+@[^\s]+)\s*\(', line)
    if match:
        return match.group(1).strip().lower()
    return None


def extract_event_type_from_log_line(line: str) -> Optional[str]:
    """
    Extrai tipo de evento de uma linha de log

    Returns: 'LeadQualified', 'LeadQualifiedHighQuality', ou 'Faixa A'
    """
    if 'LeadQualifiedHighQuality enviado:' in line:
        return 'LeadQualifiedHighQuality'
    elif 'LeadQualified enviado:' in line:
        return 'LeadQualified'
    elif 'Faixa A enviado:' in line:
        return 'Faixa A'
    return None


def get_capi_events_from_logs(
    start_date: str,
    end_date: str,
    project_id: str = 'smart-ads-451319',
    service_name: str = 'smart-ads-api'
) -> Dict[str, List[str]]:
    """
    Busca eventos CAPI dos logs do Cloud Run

    Args:
        start_date: Data início (YYYY-MM-DD)
        end_date: Data fim (YYYY-MM-DD)
        project_id: ID do projeto GCP
        service_name: Nome do serviço Cloud Run

    Returns:
        Dict com emails como chave e lista de eventos como valor
        Exemplo: {
            'email@example.com': ['LeadQualified', 'LeadQualifiedHighQuality'],
            'other@example.com': ['LeadQualified']
        }
    """
    logger.info(f"🔍 Buscando eventos CAPI nos logs do Cloud Run ({start_date} a {end_date})...")

    # Converter datas para formato ISO com timezone
    start_timestamp = f"{start_date}T00:00:00Z"
    end_timestamp = f"{end_date}T23:59:59Z"

    # Buscar logs com gcloud
    filter_query = (
        f'resource.type=cloud_run_revision AND '
        f'resource.labels.service_name={service_name} AND '
        f'(textPayload:"LeadQualified enviado:" OR textPayload:"LeadQualifiedHighQuality enviado:" OR textPayload:"Faixa A enviado:") AND '
        f'timestamp>="{start_timestamp}" AND timestamp<="{end_timestamp}"'
    )

    cmd = [
        'gcloud', 'logging', 'read',
        filter_query,
        '--limit=10000',
        '--format=value(textPayload)',
        f'--project={project_id}'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )

        log_lines = result.stdout.strip().split('\n')

        # Processar logs e agrupar por email
        events_by_email = defaultdict(list)

        for line in log_lines:
            if not line.strip():
                continue

            email = extract_email_from_log_line(line)
            event_type = extract_event_type_from_log_line(line)

            if email and event_type:
                events_by_email[email].append(event_type)

        logger.info(f"   ✅ {len(events_by_email)} emails únicos com eventos CAPI")

        # Log estatísticas
        lq_count = sum(1 for events in events_by_email.values() if 'LeadQualified' in events)
        lqhq_count = sum(1 for events in events_by_email.values() if 'LeadQualifiedHighQuality' in events)
        faixa_a_count = sum(1 for events in events_by_email.values() if 'Faixa A' in events)

        logger.info(f"   📊 LeadQualified: {lq_count} emails")
        logger.info(f"   📊 LeadQualifiedHighQuality: {lqhq_count} emails")
        logger.info(f"   📊 Faixa A: {faixa_a_count} emails")

        return dict(events_by_email)

    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout ao buscar logs do Cloud Run")
        return {}
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao buscar logs: {e}")
        logger.error(f"   stderr: {e.stderr}")
        return {}
    except Exception as e:
        logger.error(f"❌ Erro inesperado ao buscar logs: {e}")
        return {}


def get_campaign_ids_from_database(
    emails: List[str]
) -> Dict[str, str]:
    """
    Busca campaign IDs dos leads no database PostgreSQL

    Args:
        emails: Lista de emails para buscar

    Returns:
        Dict com email como chave e campaign_id como valor
        Exemplo: {'email@example.com': '120234062599950534'}
    """
    if not DB_AVAILABLE:
        logger.warning("⚠️ Database não disponível, use get_campaign_ids_from_csv()")
        return {}

    if not emails:
        return {}

    logger.info(f"🔍 Buscando campaign IDs no database para {len(emails)} emails...")

    try:
        db = SessionLocal()
        try:
            # Buscar emails e utm_campaign do database
            results = db.query(LeadCAPI.email, LeadCAPI.utm_campaign).filter(
                LeadCAPI.email.in_([e.lower() for e in emails])
            ).all()

            logger.info(f"   ✅ {len(results)} leads encontrados no database")

            # Mapear email -> campaign_id
            email_to_campaign = {}

            for email, utm_campaign in results:
                if utm_campaign:
                    # Extrair ID numérico (18 dígitos)
                    match = re.search(r'\d{18,}', str(utm_campaign))
                    if match:
                        campaign_id = match.group(0)
                        email_to_campaign[email.lower()] = campaign_id

            logger.info(f"   ✅ {len(email_to_campaign)} leads com campaign_id válido")

            return email_to_campaign

        finally:
            db.close()

    except Exception as e:
        logger.error(f"❌ Erro ao buscar campaign IDs do database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def get_campaign_ids_from_csv(
    emails: List[str],
    start_date: str,
    end_date: str,
    csv_path: str = None
) -> Dict[str, str]:
    """
    Busca campaign IDs dos leads na planilha CSV
    Filtra por período de captação

    Args:
        emails: Lista de emails para buscar
        start_date: Data início de captação (YYYY-MM-DD)
        end_date: Data fim de captação (YYYY-MM-DD)
        csv_path: Caminho para o CSV de leads (opcional)

    Returns:
        Dict com email como chave e campaign_id como valor
        Exemplo: {'email@example.com': '120234062599950534'}
    """
    if not emails:
        return {}

    import pandas as pd
    from datetime import datetime as dt

    logger.info(f"🔍 Buscando campaign IDs na planilha CSV para {len(emails)} emails...")

    # Se não passou o path, buscar o mais recente
    if csv_path is None:
        import glob
        csv_files = glob.glob('files/validation/leads/*.csv')
        if not csv_files:
            logger.error("❌ Nenhum arquivo CSV de leads encontrado")
            return {}
        csv_path = max(csv_files, key=lambda x: x)  # Mais recente
        logger.info(f"   Usando: {csv_path}")

    try:
        # Carregar CSV
        df = pd.read_csv(csv_path, low_memory=False)

        # Normalizar nomes de colunas
        df.columns = df.columns.str.strip()

        # Encontrar coluna de email
        email_col = None
        for col in df.columns:
            if 'email' in col.lower() or 'e-mail' in col.lower():
                email_col = col
                break

        if email_col is None:
            logger.error("❌ Coluna de email não encontrada no CSV")
            return {}

        # Encontrar coluna de data
        date_col = None
        for col in df.columns:
            if 'data' in col.lower() or 'created' in col.lower() or 'timestamp' in col.lower():
                date_col = col
                break

        # Encontrar coluna de campaign
        campaign_col = None
        for col in df.columns:
            if 'campaign' in col.lower() or 'campanha' in col.lower():
                campaign_col = col
                break

        if campaign_col is None:
            logger.warning("⚠️ Coluna de campanha não encontrada, tentando usar nome completo")
            # Tentar buscar em nome completo
            name_col = None
            for col in df.columns:
                if 'nome' in col.lower() and 'campanha' in col.lower():
                    name_col = col
                    campaign_col = name_col
                    break

        # Normalizar emails
        df[email_col] = df[email_col].str.lower().str.strip()

        # Filtrar por período se tiver coluna de data
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # Incluir todo o dia
            df = df[(df[date_col] >= start_dt) & (df[date_col] < end_dt)]
            logger.info(f"   Filtrado por período: {len(df)} leads")

        # Filtrar por emails
        df = df[df[email_col].isin([e.lower() for e in emails])]
        logger.info(f"   ✅ {len(df)} leads encontrados no CSV")

        # Mapear email -> campaign_id
        email_to_campaign = {}

        if campaign_col:
            for _, row in df.iterrows():
                email = row[email_col]
                campaign_value = str(row[campaign_col]) if pd.notna(row[campaign_col]) else ''

                # Extrair ID numérico (18 dígitos)
                match = re.search(r'\d{18,}', campaign_value)
                if match:
                    campaign_id = match.group(0)
                    email_to_campaign[email] = campaign_id

        logger.info(f"   ✅ {len(email_to_campaign)} leads com campaign_id válido")

        return email_to_campaign

    except Exception as e:
        logger.error(f"❌ Erro ao buscar campaign IDs do CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def count_capi_events_by_campaign(
    start_date: str,
    end_date: str,
    project_id: str = 'smart-ads-451319',
    service_name: str = 'smart-ads-api',
    csv_path: str = None
) -> Dict[str, Dict[str, int]]:
    """
    Conta eventos CAPI por campanha
    Busca logs do Cloud Run e cruza com database

    Args:
        start_date: Data início (YYYY-MM-DD)
        end_date: Data fim (YYYY-MM-DD)
        project_id: ID do projeto GCP
        service_name: Nome do serviço Cloud Run
        csv_path: Caminho para CSV de leads (opcional)

    Returns:
        Dict com campaign_id como chave e contagem de eventos como valor
        Exemplo: {
            '120234062599950534': {
                'lead': 162,
                'LeadQualified': 216,
                'LeadQualifiedHighQuality': 4
            }
        }
    """
    logger.info("=" * 80)
    logger.info("📊 CONTANDO EVENTOS CAPI DOS LOGS (abordagem alternativa)")
    logger.info("=" * 80)

    # 1. Buscar eventos dos logs
    events_by_email = get_capi_events_from_logs(start_date, end_date, project_id, service_name)

    if not events_by_email:
        logger.warning("⚠️ Nenhum evento CAPI encontrado nos logs")
        return {}

    # 2. Buscar campaign IDs (database ou CSV)
    emails = list(events_by_email.keys())

    # Tentar database primeiro
    if DB_AVAILABLE:
        logger.info("   Tentando buscar do database...")
        email_to_campaign = get_campaign_ids_from_database(emails)

        if not email_to_campaign:
            logger.warning("   ⚠️ Database vazio ou inacessível, tentando CSV...")
            email_to_campaign = get_campaign_ids_from_csv(emails, start_date, end_date, csv_path)
        else:
            logger.info(f"   ✅ Usando dados do database")
    else:
        logger.info("   Database não disponível, usando CSV...")
        email_to_campaign = get_campaign_ids_from_csv(emails, start_date, end_date, csv_path)

    if not email_to_campaign:
        logger.warning("⚠️ Nenhum lead encontrado com campaign_id (nem database nem CSV)")
        return {}

    # 3. Agregar eventos por campanha
    campaign_events = defaultdict(lambda: defaultdict(int))

    for email, events in events_by_email.items():
        campaign_id = email_to_campaign.get(email)

        if not campaign_id:
            continue

        # Contar cada tipo de evento
        for event_type in events:
            campaign_events[campaign_id][event_type] += 1

    # 4. Log resumo
    logger.info("")
    logger.info("📋 RESUMO - Eventos CAPI por Campanha:")
    for campaign_id, events in sorted(campaign_events.items()):
        logger.info(f"   • {campaign_id}:")
        for event_type, count in sorted(events.items()):
            logger.info(f"      - {event_type}: {count}")

    logger.info("")
    logger.info(f"✅ Total: {len(campaign_events)} campanhas com eventos CAPI")
    logger.info("=" * 80)

    return dict(campaign_events)
