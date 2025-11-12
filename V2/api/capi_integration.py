"""
Meta Conversions API (CAPI) Integration
Envio de eventos server-side para melhorar atribuição
"""

import os
import time
import hashlib
import logging
from typing import Dict, List, Optional
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.serverside.event import Event
from facebook_business.adobjects.serverside.event_request import EventRequest
from facebook_business.adobjects.serverside.user_data import UserData
from facebook_business.adobjects.serverside.custom_data import CustomData
from facebook_business.adobjects.serverside.action_source import ActionSource
from api.business_config import PRODUCT_VALUE, CONVERSION_RATES

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PIXEL_ID = os.getenv('META_PIXEL_ID', '241752320666130')  # Pixel de BM - Campanhas
ACCESS_TOKEN = os.getenv('META_ACCESS_TOKEN')  # Obrigatório via env var

# Inicializar API do Facebook (se token disponível)
if ACCESS_TOKEN:
    FacebookAdsApi.init(access_token=ACCESS_TOKEN)

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

def send_lead_qualified_with_value(
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
    event_timestamp: int,
    test_event_code: Optional[str] = None,
    survey_data: Optional[Dict] = None
) -> Dict:
    """
    ESTRATÉGIA 1: Envia TODOS os leads (D1-D10) com VALOR DIFERENCIADO por decil

    Comportamento:
    - Envia todos os leads independente do decil
    - Cada decil tem um valor diferente baseado na taxa de conversão corrigida
    - D10 = R$ 69.10, D1 = R$ 7.67, etc.
    - Meta otimiza para VALOR (Expected Value = Probabilidade × Valor)

    Quando usar:
    - Quer que Meta priorize leads de alta qualidade através de valores mais altos
    - Tem dados suficientes para calibrar valores por decil
    - Prefere otimização por valor monetário

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

        # Preparar custom_properties com dados ML
        custom_props = {
            'lead_score': lead_score,
            'decil': decil,
            'taxa_conversao': taxa_conversao
        }

        # Adicionar dados da pesquisa se disponíveis (enriquecem targeting)
        if survey_data:
            # Filtrar valores None/vazios
            survey_clean = {k: v for k, v in survey_data.items() if v is not None and str(v).strip() != ''}
            custom_props.update(survey_clean)

        custom_data = CustomData(
            value=valor_projetado,
            currency='BRL',
            custom_properties=custom_props
        )

        # Event
        event = Event(
            event_name='LeadQualified',
            event_time=event_timestamp,
            event_id=f"qualified_{event_id}",  # Prefixo para diferenciar do Pixel
            user_data=user_data,
            custom_data=custom_data,
            event_source_url=event_source_url,
            action_source=ActionSource.WEBSITE
        )

        # EventRequest
        event_request_params = {
            'events': [event],
            'pixel_id': PIXEL_ID,
            'access_token': ACCESS_TOKEN
        }
        if test_event_code:
            event_request_params['test_event_code'] = test_event_code

        event_request = EventRequest(**event_request_params)

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
        logger.error(f"❌ Erro ao enviar LeadQualified com valor: {str(e)}")
        return {
            "status": "error",
            "event_id": event_id,
            "email": email,
            "message": str(e)
        }

def send_lead_qualified_high_quality(
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
    event_timestamp: int,
    test_event_code: Optional[str] = None,
    survey_data: Optional[Dict] = None
) -> Dict:
    """
    ESTRATÉGIA 2: Envia APENAS D8, D9 e D10 SEM VALOR

    Comportamento:
    - Filtra: só envia se decil in ['D8', 'D9', 'D10']
    - SEM valor monetário (Meta otimiza para volume de conversões)
    - Meta aprende com perfil de alta qualidade
    - Volume menor mas mais focado

    Quando usar (Gestor de Tráfego):
    - Criar campanha separada otimizando para "LeadQualifiedHighQuality"
    - Usar Cost Cap ou Lowest Cost (não Target ROAS)
    - Foco em volume de leads qualificados

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
        Dict com resultado do envio (ou skipped se não for D8-D10)
    """
    # Filtro: só envia D8, D9, D10
    if decil not in ['D8', 'D9', 'D10']:
        logger.debug(f"⏭️  Lead {decil} ignorado (estratégia D8-D10 only)")
        return {
            "status": "skipped",
            "event_id": event_id,
            "email": email,
            "decil": decil,
            "reason": "Decil abaixo de D8 (filtrado)"
        }

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

        # CustomData (SEM valor - Meta otimiza para volume)
        # Preparar custom_properties
        custom_props = {
            'lead_score': lead_score,
            'decil': decil,
            'estrategia': 'high_quality_only'
        }

        # Adicionar dados da pesquisa se disponíveis
        if survey_data:
            survey_clean = {k: v for k, v in survey_data.items() if v is not None and str(v).strip() != ''}
            custom_props.update(survey_clean)

        custom_data = CustomData(
            currency='BRL',
            custom_properties=custom_props
        )

        # Event
        event = Event(
            event_name='LeadQualifiedHighQuality',  # Nome diferente!
            event_time=event_timestamp,
            event_id=f"hq_{event_id}",  # Prefixo diferente para evitar dedup
            user_data=user_data,
            custom_data=custom_data,
            event_source_url=event_source_url,
            action_source=ActionSource.WEBSITE
        )

        # EventRequest
        event_request_params = {
            'events': [event],
            'pixel_id': PIXEL_ID,
            'access_token': ACCESS_TOKEN
        }
        if test_event_code:
            event_request_params['test_event_code'] = test_event_code

        event_request = EventRequest(**event_request_params)

        # Enviar
        response = event_request.execute()

        logger.info(f"✅ LeadQualifiedHighQuality enviado: {email} (decil: {decil})")

        return {
            "status": "success",
            "event_id": event_id,
            "email": email,
            "decil": decil,
            "estrategia": "high_quality_only",
            "response": str(response)
        }

    except Exception as e:
        logger.error(f"❌ Erro ao enviar LeadQualifiedHighQuality: {str(e)}")
        return {
            "status": "error",
            "event_id": event_id,
            "email": email,
            "message": str(e)
        }

def send_both_lead_events(
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
    event_timestamp: int,
    test_event_code: Optional[str] = None,
    survey_data: Optional[Dict] = None
) -> Dict:
    """
    TESTE A/B: Envia AMBOS os eventos para permitir teste de 2 estratégias

    Esta função envia:
    1. LeadQualified (com valor, D1-D10)
    2. LeadQualifiedHighQuality (sem valor, D8-D10 only)

    O gestor de tráfego cria 2 campanhas:
    - Campanha A (50% budget): Otimiza para "LeadQualified"
    - Campanha B (50% budget): Otimiza para "LeadQualifiedHighQuality"

    Após 4 semanas, compara:
    - CPL, Volume, Taxa conversão real, ROAS

    Args:
        Mesmos args das funções individuais

    Returns:
        Dict com resultado de ambos os envios
    """
    logger.info(f"📤 Enviando AMBOS eventos para teste A/B: {email} ({decil})")

    # Enviar evento 1: COM VALOR (D1-D10)
    result_with_value = send_lead_qualified_with_value(
        email=email,
        phone=phone,
        lead_score=lead_score,
        decil=decil,
        event_id=event_id,
        fbp=fbp,
        fbc=fbc,
        user_agent=user_agent,
        client_ip=client_ip,
        event_source_url=event_source_url,
        event_timestamp=event_timestamp,
        test_event_code=test_event_code,
        survey_data=survey_data
    )

    # Enviar evento 2: SEM VALOR (D8-D10 only)
    result_high_quality = send_lead_qualified_high_quality(
        email=email,
        phone=phone,
        lead_score=lead_score,
        decil=decil,
        event_id=event_id,
        fbp=fbp,
        fbc=fbc,
        user_agent=user_agent,
        client_ip=client_ip,
        event_source_url=event_source_url,
        event_timestamp=event_timestamp,
        test_event_code=test_event_code,
        survey_data=survey_data
    )

    return {
        "status": "success",
        "email": email,
        "decil": decil,
        "evento_com_valor": result_with_value,
        "evento_high_quality": result_high_quality
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
            action_source=ActionSource.SYSTEM_GENERATED  # Conversão offline
        )

        # EventRequest
        event_request_params = {
            'events': [event],
            'pixel_id': PIXEL_ID,
            'access_token': ACCESS_TOKEN
        }
        if test_event_code:
            event_request_params['test_event_code'] = test_event_code

        event_request = EventRequest(**event_request_params)

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
    Envia múltiplos eventos CAPI em batch (AMBAS AS ESTRATÉGIAS)
    Usado pelo processamento diário

    Para cada lead, envia:
    - LeadQualified (com valor, todos os decis)
    - LeadQualifiedHighQuality (sem valor, D8-D10 only)

    Args:
        leads: Lista de dicts com dados dos leads

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
        # Usar send_both_lead_events para enviar ambas as estratégias
        result = send_both_lead_events(
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
            event_timestamp=lead['event_timestamp'],
            survey_data=lead.get('survey_data')  # NOVO: Dados da pesquisa
            # test_event_code=None (padrão) -> vai para PRODUÇÃO
        )

        if result['status'] == 'success':
            results['success'] += 1
        else:
            results['errors'] += 1

        results['details'].append(result)

    logger.info(f"📊 Batch CAPI: {results['success']}/{results['total']} enviados com sucesso")

    return results
