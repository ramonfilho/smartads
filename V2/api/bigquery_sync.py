"""
Sync PostgreSQL → BigQuery
Sincroniza dados do PostgreSQL para BigQuery para analytics no Looker Studio
"""

import os
import logging
from google.cloud import bigquery
from sqlalchemy.orm import Session
from api.database import get_recent_leads, SessionLocal

logger = logging.getLogger(__name__)

# Configuração BigQuery
PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'smart-ads-451319')
DATASET_ID = 'devclub'
TABLE_ID = 'leads_capi'

def sync_postgres_to_bigquery(limit: int = 1000) -> dict:
    """
    Sincroniza dados do PostgreSQL para BigQuery

    Args:
        limit: Número máximo de registros a sincronizar (default: 1000 últimos)

    Returns:
        Dict com status e estatísticas do sync
    """
    try:
        # Inicializar cliente BigQuery
        client = bigquery.Client(project=PROJECT_ID)
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

        # Buscar leads do PostgreSQL
        db = SessionLocal()
        try:
            leads = get_recent_leads(db, limit=limit)

            if not leads:
                return {
                    "status": "success",
                    "message": "Nenhum lead para sincronizar",
                    "rows_synced": 0
                }

            # Converter para formato BigQuery
            rows_to_insert = []
            for lead in leads:
                row = {
                    "id": lead.id,
                    "email": lead.email,
                    "name": lead.name,
                    "phone": lead.phone,
                    "fbp": lead.fbp,
                    "fbc": lead.fbc,
                    "event_id": lead.event_id,
                    "user_agent": lead.user_agent,
                    "client_ip": lead.client_ip,
                    "event_source_url": lead.event_source_url,
                    "utm_source": lead.utm_source,
                    "utm_medium": lead.utm_medium,
                    "utm_campaign": lead.utm_campaign,
                    "utm_term": lead.utm_term,
                    "utm_content": lead.utm_content,
                    "tem_comp": lead.tem_comp,
                    "created_at": lead.created_at.isoformat() if lead.created_at else None,
                    "updated_at": lead.updated_at.isoformat() if lead.updated_at else None
                }
                rows_to_insert.append(row)

            # Inserir no BigQuery (replace = truncate + insert)
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            )

            job = client.load_table_from_json(
                rows_to_insert,
                table_ref,
                job_config=job_config
            )

            job.result()  # Wait for completion

            logger.info(f"✅ Sincronizados {len(rows_to_insert)} leads para BigQuery")

            return {
                "status": "success",
                "message": f"Sincronizados {len(rows_to_insert)} leads",
                "rows_synced": len(rows_to_insert),
                "table": table_ref
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"❌ Erro ao sincronizar com BigQuery: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "rows_synced": 0
        }


def get_bigquery_stats() -> dict:
    """
    Retorna estatísticas da tabela no BigQuery
    """
    try:
        client = bigquery.Client(project=PROJECT_ID)
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

        # Query para contar registros
        query = f"""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT email) as unique_emails,
            COUNT(fbp) as with_fbp,
            COUNT(fbc) as with_fbc,
            MAX(created_at) as last_updated
        FROM `{table_ref}`
        """

        query_job = client.query(query)
        results = list(query_job.result())

        if results:
            row = results[0]
            return {
                "status": "success",
                "total_rows": row.total_rows,
                "unique_emails": row.unique_emails,
                "with_fbp": row.with_fbp,
                "with_fbc": row.with_fbc,
                "last_updated": row.last_updated.isoformat() if row.last_updated else None
            }

        return {
            "status": "success",
            "total_rows": 0
        }

    except Exception as e:
        logger.error(f"❌ Erro ao buscar stats do BigQuery: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
