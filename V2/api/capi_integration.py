"""
Meta Conversions API (CAPI) Integration
Envio de eventos server-side para melhorar atribuição
"""

import os
import time
import hashlib
import logging
from typing import Dict, List, Optional
from facebook_business.adobjects.serverside.event import Event
from facebook_business.adobjects.serverside.event_request import EventRequest
from facebook_business.adobjects.serverside.user_data import UserData
from facebook_business.adobjects.serverside.custom_data import CustomData
from api.business_config import PRODUCT_VALUE, CONVERSION_RATES

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PIXEL_ID = os.getenv('META_PIXEL_ID', '1254837345551611')  # Default do cliente
ACCESS_TOKEN = os.getenv('META_ACCESS_TOKEN')  # Obrigatório via env var

def hash_data(data: str) -> str:
    """
    Hash SHA256 de dados pessoais (formato Meta CAPI)
    Remove espaços, lowercase, depois hash
    """
    if not data:
        return None
    normalized = str(data).lower().strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

# =============================================================================
# ENVIO DE EVENTOS
# =============================================================================

def send_lead_qualified_event(
    email: str,
    phone: Optional[str],
    lead_score: float,
    decil: str,
    event_id: str,
    fbp: Optional[str],
    fbc: Optional[str],
    user_agent: Optional[str],
    client_ip: Optional[str],
    event_source_url: Optional[str],
    event_timestamp: int
) -> Dict:
    """
    Envia evento LeadQualified para leads D10

    Args:
        email: Email do lead
        phone: Telefone do lead
        lead_score: Score do modelo ML
        decil: Decil (D1-D10)
        event_id: ID único do evento (deduplicação)
        fbp: Facebook Browser ID (_fbp cookie)
        fbc: Facebook Click ID (_fbc cookie)
        user_agent: User agent do navegador
        client_ip: IP do cliente
        event_source_url: URL da página de origem
        event_timestamp: Timestamp UNIX do lead original (não atual!)

    Returns:
        Dict com resultado do envio
    """
    if not ACCESS_TOKEN:
        logger.error("❌ META_ACCESS_TOKEN não configurado")
        return {"status": "error", "message": "ACCESS_TOKEN não configurado"}

    try:
        # UserData (dados do usuário hashados)
        user_data = UserData(
            emails=[hash_data(email)] if email else None,
            phones=[hash_data(phone)] if phone else None,
            client_ip_address=client_ip,
            client_user_agent=user_agent,
            fbp=fbp,
            fbc=fbc
        )

        # CustomData (valor projetado baseado em taxa de conversão)
        taxa_conversao = CONVERSION_RATES.get(decil, 0.0)
        valor_projetado = PRODUCT_VALUE * taxa_conversao

        custom_data = CustomData(
            value=valor_projetado,
            currency='BRL',
            custom_properties={
                'lead_score': lead_score,
                'decil': decil,
                'taxa_conversao': taxa_conversao
            }
        )

        # Event
        event = Event(
            event_name='LeadQualified',
            event_time=event_timestamp,
            event_id=f"qualified_{event_id}",  # Prefixo para diferenciar do Pixel
            user_data=user_data,
            custom_data=custom_data,
            event_source_url=event_source_url,
            action_source='website'
        )

        # EventRequest
        event_request = EventRequest(
            events=[event],
            pixel_id=PIXEL_ID,
            access_token=ACCESS_TOKEN
        )

        # Enviar
        response = event_request.execute()

        logger.info(f"✅ LeadQualified enviado: {email} (decil: {decil}, valor proj: R$ {valor_projetado:.2f})")

        return {
            "status": "success",
            "event_id": event_id,
            "email": email,
            "decil": decil,
            "valor_projetado": valor_projetado,
            "response": str(response)
        }

    except Exception as e:
        logger.error(f"❌ Erro ao enviar LeadQualified: {str(e)}")
        return {
            "status": "error",
            "event_id": event_id,
            "email": email,
            "message": str(e)
        }

def send_purchase_event(
    email: str,
    phone: Optional[str],
    valor_venda: float,
    original_event_id: str,
    fbp: Optional[str],
    fbc: Optional[str],
    user_agent: Optional[str],
    client_ip: Optional[str],
    event_source_url: Optional[str]
) -> Dict:
    """
    Envia evento Purchase quando lead vira venda

    Args:
        email: Email do lead
        phone: Telefone do lead
        valor_venda: Valor REAL da venda
        original_event_id: Event ID do lead original (para linking)
        fbp: Facebook Browser ID
        fbc: Facebook Click ID
        user_agent: User agent
        client_ip: IP do cliente
        event_source_url: URL de origem

    Returns:
        Dict com resultado do envio
    """
    if not ACCESS_TOKEN:
        logger.error("❌ META_ACCESS_TOKEN não configurado")
        return {"status": "error", "message": "ACCESS_TOKEN não configurado"}

    try:
        # UserData
        user_data = UserData(
            emails=[hash_data(email)] if email else None,
            phones=[hash_data(phone)] if phone else None,
            client_ip_address=client_ip,
            client_user_agent=user_agent,
            fbp=fbp,
            fbc=fbc
        )

        # CustomData (valor REAL da venda)
        custom_data = CustomData(
            value=valor_venda,
            currency='BRL'
        )

        # Event
        event = Event(
            event_name='Purchase',
            event_time=int(time.time()),
            event_id=f"purchase_{original_event_id}",
            user_data=user_data,
            custom_data=custom_data,
            event_source_url=event_source_url,
            action_source='system_generated'  # Conversão offline
        )

        # EventRequest
        event_request = EventRequest(
            events=[event],
            pixel_id=PIXEL_ID,
            access_token=ACCESS_TOKEN
        )

        # Enviar
        response = event_request.execute()

        logger.info(f"✅ Purchase enviado: {email} (valor: R$ {valor_venda:.2f})")

        return {
            "status": "success",
            "event_id": original_event_id,
            "email": email,
            "valor_venda": valor_venda,
            "response": str(response)
        }

    except Exception as e:
        logger.error(f"❌ Erro ao enviar Purchase: {str(e)}")
        return {
            "status": "error",
            "event_id": original_event_id,
            "email": email,
            "message": str(e)
        }

def send_batch_events(leads: List[Dict]) -> Dict:
    """
    Envia múltiplos eventos LeadQualified em batch
    Usado pelo processamento diário

    Args:
        leads: Lista de dicts com dados dos leads D10

    Returns:
        Dict com estatísticas do envio
    """
    if not ACCESS_TOKEN:
        logger.error("❌ META_ACCESS_TOKEN não configurado")
        return {
            "status": "error",
            "message": "ACCESS_TOKEN não configurado",
            "total": 0,
            "success": 0,
            "errors": 0
        }

    results = {
        "total": len(leads),
        "success": 0,
        "errors": 0,
        "details": []
    }

    for lead in leads:
        result = send_lead_qualified_event(
            email=lead['email'],
            phone=lead.get('phone'),
            lead_score=lead['lead_score'],
            decil=lead['decil'],
            event_id=lead['event_id'],
            fbp=lead.get('fbp'),
            fbc=lead.get('fbc'),
            user_agent=lead.get('user_agent'),
            client_ip=lead.get('client_ip'),
            event_source_url=lead.get('event_source_url'),
            event_timestamp=lead['event_timestamp']
        )

        if result['status'] == 'success':
            results['success'] += 1
        else:
            results['errors'] += 1

        results['details'].append(result)

    logger.info(f"📊 Batch CAPI: {results['success']}/{results['total']} enviados com sucesso")

    return results
