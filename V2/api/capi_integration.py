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
from facebook_business.adobjects.serverside.gender import Gender
from api.business_config import PRODUCT_VALUE, CONVERSION_RATES

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PIXEL_ID = os.getenv('META_PIXEL_ID', '241752320666130')  # Pixel de BM - Campanhas
ACCESS_TOKEN = os.getenv('META_ACCESS_TOKEN')  # Obrigatório via env var

# =============================================================================
# MAPEAMENTO DDD → ESTADO (Brasil)
# =============================================================================

DDD_TO_STATE = {
    # São Paulo
    '11': 'SP', '12': 'SP', '13': 'SP', '14': 'SP', '15': 'SP', '16': 'SP', '17': 'SP', '18': 'SP', '19': 'SP',
    # Rio de Janeiro
    '21': 'RJ', '22': 'RJ', '24': 'RJ',
    # Espírito Santo
    '27': 'ES', '28': 'ES',
    # Minas Gerais
    '31': 'MG', '32': 'MG', '33': 'MG', '34': 'MG', '35': 'MG', '37': 'MG', '38': 'MG',
    # Paraná
    '41': 'PR', '42': 'PR', '43': 'PR', '44': 'PR', '45': 'PR', '46': 'PR',
    # Santa Catarina
    '47': 'SC', '48': 'SC', '49': 'SC',
    # Rio Grande do Sul
    '51': 'RS', '53': 'RS', '54': 'RS', '55': 'RS',
    # Distrito Federal
    '61': 'DF',
    # Goiás
    '62': 'GO', '64': 'GO',
    # Tocantins
    '63': 'TO',
    # Mato Grosso
    '65': 'MT', '66': 'MT',
    # Mato Grosso do Sul
    '67': 'MS',
    # Acre
    '68': 'AC',
    # Rondônia
    '69': 'RO',
    # Bahia
    '71': 'BA', '73': 'BA', '74': 'BA', '75': 'BA', '77': 'BA',
    # Sergipe
    '79': 'SE',
    # Pernambuco
    '81': 'PE', '87': 'PE',
    # Alagoas
    '82': 'AL',
    # Paraíba
    '83': 'PB',
    # Rio Grande do Norte
    '84': 'RN',
    # Ceará
    '85': 'CE', '88': 'CE',
    # Piauí
    '86': 'PI', '89': 'PI',
    # Maranhão
    '98': 'MA', '99': 'MA',
    # Pará
    '91': 'PA', '93': 'PA', '94': 'PA',
    # Amazonas
    '92': 'AM', '97': 'AM',
    # Roraima
    '95': 'RR',
    # Amapá
    '96': 'AP',
}

def get_state_from_phone(phone: str) -> Optional[str]:
    """
    Extrai o estado brasileiro a partir do DDD do telefone

    Args:
        phone: Telefone (pode ter +55, espaços, etc)

    Returns:
        Sigla do estado (SP, RJ, etc) ou None se não encontrar
    """
    if not phone:
        return None

    # Garantir que phone é string (Apps Script pode enviar como int)
    phone_str = str(phone)

    # Remove tudo que não é número
    digits = ''.join(filter(str.isdigit, phone_str))

    # Se começar com 55 (código Brasil), remove
    if digits.startswith('55') and len(digits) > 10:
        digits = digits[2:]

    # O DDD são os 2 primeiros dígitos
    if len(digits) >= 2:
        ddd = digits[:2]
        return DDD_TO_STATE.get(ddd)

    return None

def normalize_gender(gender_str) -> Optional[Gender]:
    """
    Normaliza o gênero para o formato Meta CAPI (enum Gender)

    Args:
        gender_str: Resposta do formulário ("Masculino", "Feminino", etc)

    Returns:
        Gender.MALE para masculino, Gender.FEMALE para feminino, None para outros
    """
    if not gender_str:
        return None

    # Converter para string e validar
    try:
        gender_lower = str(gender_str).lower().strip()

        # Ignorar valores numéricos ou muito curtos/longos
        if gender_lower.isdigit() or len(gender_lower) < 1 or len(gender_lower) > 20:
            return None

        if gender_lower in ['masculino', 'homem', 'male', 'm']:
            return Gender.MALE
        elif gender_lower in ['feminino', 'mulher', 'female', 'f']:
            return Gender.FEMALE
    except Exception:
        pass

    return None

# Inicializar API do Facebook (se token disponível)
if ACCESS_TOKEN:
    FacebookAdsApi.init(access_token=ACCESS_TOKEN)

def hash_data(data) -> Optional[str]:
    """
    Hash SHA256 de dados pessoais (formato Meta CAPI)
    Remove espaços, lowercase, depois hash
    """
    if data is None or data == '':
        return None
    try:
        normalized = str(data).lower().strip()
        if not normalized:
            return None
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    except Exception:
        return None

# =============================================================================
# ENVIO DE EVENTOS
# =============================================================================

def send_lead_qualified_with_value(
    email: str,
    phone: Optional[str],
    first_name: Optional[str],
    last_name: Optional[str],
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
        # Extrair dados adicionais para melhor matching
        # 1. Estado: inferir do DDD do telefone
        state = get_state_from_phone(phone)

        # 2. País: Brasil como padrão (telefones brasileiros)
        country = 'br' if phone else None

        # 3. Cidade, CEP e Gênero: do survey_data se disponível
        city = None
        zip_code = None
        gender = None

        if survey_data:
            city = survey_data.get('cidade')
            zip_code = survey_data.get('cep')
            # Gênero: normalizar para formato Meta (m/f)
            # Nota: app.py monta survey_data com chave 'genero' (não 'O seu gênero:')
            gender_raw = survey_data.get('genero')
            gender = normalize_gender(gender_raw)

        # UserData (dados do usuário hashados)
        # IMPORTANTE: Esses campos melhoram o Event Quality Score do Meta
        user_data = UserData(
            emails=[hash_data(email)] if email else None,
            phones=[hash_data(phone)] if phone else None,
            first_names=[hash_data(first_name)] if first_name else None,
            last_names=[hash_data(last_name)] if last_name else None,
            # Novos campos para melhorar matching:
            states=[hash_data(state)] if state else None,
            cities=[hash_data(city)] if city else None,
            country_codes=[hash_data(country)] if country else None,
            zip_codes=[hash_data(zip_code)] if zip_code else None,
            genders=[gender] if gender else None,
            # Campos de contexto (não hashados):
            client_ip_address=client_ip,
            client_user_agent=user_agent,
            fbp=fbp,
            fbc=fbc
        )

        # CustomData (valor projetado baseado em taxa de conversão)
        taxa_conversao = CONVERSION_RATES.get(decil, 0.0)
        valor_projetado = PRODUCT_VALUE * taxa_conversao

        # Preparar custom_properties com dados ML
        # IMPORTANTE: Converter valores para string para compatibilidade com Meta API
        custom_props = {
            'lead_score': str(lead_score),
            'decil': decil,  # já é string
            'taxa_conversao': str(taxa_conversao)
        }

        # Adicionar dados da pesquisa se disponíveis (enriquecem targeting)
        if survey_data:
            # Filtrar valores None/vazios e converter tudo para string
            survey_clean = {k: str(v) for k, v in survey_data.items() if v is not None and str(v).strip() != ''}
            custom_props.update(survey_clean)

        custom_data = CustomData(
            value=valor_projetado,
            currency='BRL',
            custom_properties=custom_props
        )

        # DEBUG: Log custom_properties e VALUE para verificar o que está sendo enviado
        logger.info(f"🔍 DEBUG LeadQualified para {email}:")
        logger.info(f"   VALUE enviado: R$ {valor_projetado:.2f} (decil: {decil}, taxa: {taxa_conversao})")
        logger.info(f"   Total de custom_properties: {len(custom_props)}")
        logger.info(f"   Keys: {list(custom_props.keys())}")
        logger.info(f"   Sample: {dict(list(custom_props.items())[:5])}")

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

        # DEBUG: Log do payload JSON EXATO que será enviado para Meta API
        import json
        try:
            # Usar o método export_value() do SDK para ver o JSON exato
            if hasattr(custom_data, 'export_value'):
                serialized = custom_data.export_value()
                logger.info(f"🔍 DEBUG JSON EXATO enviado para Meta API (custom_data):")
                logger.info(json.dumps(serialized, indent=2, ensure_ascii=False))
            else:
                # Fallback: extrair manualmente
                payload_debug = {
                    'value': custom_data.value if hasattr(custom_data, 'value') else None,
                    'currency': custom_data.currency if hasattr(custom_data, 'currency') else None,
                    'custom_properties': custom_data.custom_properties if hasattr(custom_data, 'custom_properties') else None
                }
                logger.info(f"🔍 DEBUG Payload custom_data enviado para Meta:")
                logger.info(json.dumps(payload_debug, indent=2, ensure_ascii=False))
        except Exception as debug_err:
            logger.warning(f"⚠️  Erro ao extrair payload debug: {debug_err}")

        # Enviar
        response = event_request.execute()

        # DEBUG: Log da resposta da Meta para confirmar recebimento
        import json
        try:
            # A resposta da Meta contém informações sobre eventos recebidos e processados
            logger.info(f"🔍 DEBUG Resposta da Meta API:")
            logger.info(f"   Response type: {type(response)}")
            logger.info(f"   Response: {response}")
            # Se for um objeto com atributos, tentar extrair
            if hasattr(response, '__dict__'):
                logger.info(f"   Response dict: {json.dumps(response.__dict__, default=str, indent=2)}")
        except Exception as resp_err:
            logger.warning(f"⚠️  Erro ao logar resposta: {resp_err}")

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
        import traceback
        logger.error(f"❌ Erro ao enviar LeadQualified com valor: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "event_id": event_id,
            "email": email,
            "message": str(e)
        }

def send_lead_qualified_high_quality(
    email: str,
    phone: Optional[str],
    first_name: Optional[str],
    last_name: Optional[str],
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
    ESTRATÉGIA 2: Envia APENAS D9 e D10 SEM VALOR

    Comportamento:
    - Filtra: só envia se decil in ['D9', 'D10']
    - SEM valor monetário (Meta otimiza para volume de conversões)
    - Meta aprende com perfil de alta qualidade (top 20% dos leads)
    - Volume menor mas mais focado

    Quando usar (Gestor de Tráfego):
    - Criar campanha separada otimizando para "LeadQualifiedHighQuality"
    - Usar Cost Cap ou Lowest Cost (não Target ROAS)
    - Foco em volume de leads qualificados (top 20%)

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
        Dict com resultado do envio (ou skipped se não for D9-D10)
    """
    # Filtro: só envia D09, D10 (formato normalizado com zeros)
    # Nota: thresholds foram normalizados em prediction.py para garantir formato D01-D10
    if decil not in ['D09', 'D10']:
        logger.debug(f"⏭️  Lead {decil} ignorado (estratégia D09-D10 only)")
        return {
            "status": "skipped",
            "event_id": event_id,
            "email": email,
            "decil": decil,
            "reason": "Decil abaixo de D09 (filtrado)"
        }

    if not ACCESS_TOKEN:
        logger.error("❌ META_ACCESS_TOKEN não configurado")
        return {"status": "error", "message": "ACCESS_TOKEN não configurado"}

    try:
        # Extrair dados adicionais para melhor matching
        # 1. Estado: inferir do DDD do telefone
        state = get_state_from_phone(phone)

        # 2. País: Brasil como padrão (telefones brasileiros)
        country = 'br' if phone else None

        # 3. Cidade, CEP e Gênero: do survey_data se disponível
        city = None
        zip_code = None
        gender = None

        if survey_data:
            city = survey_data.get('cidade')
            zip_code = survey_data.get('cep')
            # Gênero: normalizar para formato Meta (m/f)
            # Nota: app.py monta survey_data com chave 'genero' (não 'O seu gênero:')
            gender_raw = survey_data.get('genero')
            gender = normalize_gender(gender_raw)

        # UserData (dados do usuário hashados)
        # IMPORTANTE: Esses campos melhoram o Event Quality Score do Meta
        user_data = UserData(
            emails=[hash_data(email)] if email else None,
            phones=[hash_data(phone)] if phone else None,
            first_names=[hash_data(first_name)] if first_name else None,
            last_names=[hash_data(last_name)] if last_name else None,
            # Novos campos para melhorar matching:
            states=[hash_data(state)] if state else None,
            cities=[hash_data(city)] if city else None,
            country_codes=[hash_data(country)] if country else None,
            zip_codes=[hash_data(zip_code)] if zip_code else None,
            genders=[gender] if gender else None,
            # Campos de contexto (não hashados):
            client_ip_address=client_ip,
            client_user_agent=user_agent,
            fbp=fbp,
            fbc=fbc
        )

        # CustomData (SEM valor - Meta otimiza para volume)
        # Preparar custom_properties
        # IMPORTANTE: Converter valores para string para compatibilidade com Meta API
        custom_props = {
            'lead_score': str(lead_score),
            'decil': decil,  # já é string
            'estrategia': 'high_quality_only'
        }

        # Adicionar dados da pesquisa se disponíveis
        if survey_data:
            # Filtrar valores None/vazios e converter tudo para string
            survey_clean = {k: str(v) for k, v in survey_data.items() if v is not None and str(v).strip() != ''}
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
        import traceback
        logger.error(f"❌ Erro ao enviar LeadQualifiedHighQuality: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "event_id": event_id,
            "email": email,
            "message": str(e)
        }

def send_both_lead_events(
    email: str,
    phone: Optional[str],
    first_name: Optional[str],
    last_name: Optional[str],
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
    2. LeadQualifiedHighQuality (sem valor, D9-D10 only)

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
        first_name=first_name,
        last_name=last_name,
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

    # Enviar evento 2: SEM VALOR (D9-D10 only)
    result_high_quality = send_lead_qualified_high_quality(
        email=email,
        phone=phone,
        first_name=first_name,
        last_name=last_name,
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
    first_name: Optional[str],
    last_name: Optional[str],
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
            first_names=[hash_data(first_name)] if first_name else None,
            last_names=[hash_data(last_name)] if last_name else None,
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

def send_batch_events(leads: List[Dict], db=None) -> Dict:
    """
    Envia múltiplos eventos CAPI em batch (AMBAS AS ESTRATÉGIAS)
    Usado pelo processamento diário

    Para cada lead, envia:
    - LeadQualified (com valor, todos os decis)
    - LeadQualifiedHighQuality (sem valor, D9-D10 only)

    Args:
        leads: Lista de dicts com dados dos leads
        db: SQLAlchemy session para registrar envios (opcional)

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
        # DEBUG: Log do tipo de lead_score para identificar problema "'int' object is not iterable"
        lead_score_value = lead['lead_score']
        logger.info(f"DEBUG: lead_score type={type(lead_score_value)}, value={lead_score_value}")

        # Usar send_both_lead_events para enviar ambas as estratégias
        result = send_both_lead_events(
            email=lead['email'],
            phone=lead.get('phone'),
            first_name=lead.get('first_name'),
            last_name=lead.get('last_name'),
            lead_score=lead_score_value,
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

            # Registrar envio no banco (se db session disponível)
            if db:
                try:
                    from api.database import mark_lead_capi_sent
                    mark_lead_capi_sent(db, lead['email'])
                except Exception as mark_error:
                    logger.warning(f"⚠️ Não foi possível marcar CAPI sent para {lead['email']}: {mark_error}")
        else:
            results['errors'] += 1

        results['details'].append(result)

    logger.info(f"📊 Batch CAPI: {results['success']}/{results['total']} enviados com sucesso")

    return results
