# Captura de Dados para CAPI

## Objetivo
Recall de atribuição Meta: 50-60% → 90-100%

## Fluxo Completo do Sistema

### 1. Frontend captura dados do lead
- Usuário preenche formulário (nome, email, telefone)
- JavaScript captura: `_fbp`, `_fbc`, `event_id`, `user_agent`, `event_source_url`
- Envia para 2 lugares:
  - SellFlux (sistema legado) → salva na planilha Google Sheets
  - API webhook → salva no PostgreSQL

### 2. API salva dados CAPI no banco
- Endpoint: `POST /webhook/lead_capture`
- Recebe dados do frontend
- Captura `client_ip` do header
- Salva tudo no PostgreSQL (tabela `leads_capi`)

### 3. Google Sheets armazena lead (sistema legado)
- SellFlux salva: nome, email, telefone, UTMs, data
- Não salva dados CAPI (fbp, fbc ficam no PostgreSQL privado)

### 4. Apps Script classifica leads (1x/dia às 00:00)
- Lê leads dos últimos 21 dias da planilha
- Chama API: `POST /predict/batch`
- API retorna: `lead_score`, `decil` (D1-D10)
- Apps Script escreve scores na planilha

### 5. Apps Script envia batch CAPI (1x/dia às 00:00, após step 4)
- Filtra leads D10 do dia anterior (00:00-23:59)
- Chama API: `POST /capi/process_daily_batch`
- Envia: emails, scores, decis dos leads D10

### 6. API enriquece e envia para Meta CAPI
- Busca dados CAPI no PostgreSQL (por email)
- Enriquece leads D10 com: fbp, fbc, user_agent, client_ip
- Envia eventos `LeadQualified` para Meta Conversions API
- Meta faz matching com anúncios usando fbp/fbc
- Resultado: atribuição sobe de 50-60% para 80-90%

## Dados a Capturar (Frontend)

### 1. Cookies Meta
```javascript
function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return '';
}

const fbp = getCookie('_fbp');  // Browser ID (sempre existe)
const fbc = getCookie('_fbc');  // Click ID (só se veio de anúncio)
```

### 2. Event ID (Deduplicação)
```javascript
const eventID = `lead_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
```

### 3. User Agent
```javascript
const userAgent = navigator.userAgent;
```

### 4. URL Atual
```javascript
const eventSourceUrl = window.location.href;
```

## Alterações no Código

### sendToSellFlux()

**Assinatura:**
```javascript
function sendToSellFlux(name, email, phone, hasComputer, utm, fbp, fbc, eventID, userAgent, eventSourceUrl)
```

**Payload:**
```javascript
const payload = {
  name: name,
  email: email,
  phone: phone,
  tem_comp: hasComputer,
  fbp: fbp,
  fbc: fbc,
  event_id: eventID,
  user_agent: userAgent,
  event_source_url: eventSourceUrl,
  utm_source: utm.utm_source,
  utm_medium: utm.utm_medium,
  utm_campaign: utm.utm_campaign,
  utm_term: utm.utm_term,
  utm_content: utm.utm_content,
  data: dataHora
};
```

**Chamada:**
```javascript
await sendToSellFlux(fullname, email, phone, hasComputer, utm, fbp, fbc, eventID, userAgent, eventSourceUrl);
```

## Backend (Webhook)

### Capturar IP do Cliente
```python
client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
```

### Armazenar na Planilha
Colunas adicionais:
- `fbp`
- `fbc`
- `event_id`
- `user_agent`
- `client_ip`
- `event_source_url`

## CAPI - Envio de Eventos

**Timing:** Batch 1x/dia às 00:00 (junto com processamento ML)

### Evento 1: LeadQualified
Enviar para todos leads com `decil == "D10"` do dia anterior:

```python
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.serverside.event import Event
from facebook_business.adobjects.serverside.event_request import EventRequest
from facebook_business.adobjects.serverside.user_data import UserData
from facebook_business.adobjects.serverside.custom_data import CustomData
import hashlib

# Hash de dados pessoais
def hash_data(data):
    return hashlib.sha256(data.lower().strip().encode('utf-8')).hexdigest()

user_data = UserData(
    emails=[hash_data(email)],
    phones=[hash_data(phone)],
    client_ip_address=client_ip,
    client_user_agent=user_agent,
    fbp=fbp,
    fbc=fbc
)

custom_data = CustomData(
    value=PRODUCT_VALUE * CONVERSION_RATES['D10'],
    currency='BRL',
    custom_properties={
        'lead_score': lead_score,
        'decil': decil
    }
)

event = Event(
    event_name='LeadQualified',  # Evento customizado
    event_time=int(lead_timestamp),  # Timestamp original do lead (não atual)
    event_id=f"qualified_{event_id}",  # ID diferente do Pixel
    user_data=user_data,
    custom_data=custom_data,
    event_source_url=event_source_url,
    action_source='website'
)

event_request = EventRequest(
    events=[event],
    pixel_id=PIXEL_ID,
    access_token=ACCESS_TOKEN
)

event_request.execute()
```

### Evento 2: Purchase (X dias depois)
Enviar quando lead vira venda:

```python
custom_data = CustomData(
    value=2027.38,  # Valor real da venda
    currency='BRL'
)

event = Event(
    event_name='Purchase',
    event_time=int(time.time()),
    event_id=f"purchase_{original_event_id}",
    user_data=user_data,  # Mesmos dados
    custom_data=custom_data,
    event_source_url=event_source_url,
    action_source='system_generated'  # Conversão offline
)

event_request = EventRequest(
    events=[event],
    pixel_id=PIXEL_ID,
    access_token=ACCESS_TOKEN
)

event_request.execute()
```

## Deduplicação Pixel + CAPI

**Problema:** Pixel (client-side) e CAPI (server-side) enviam mesmo evento.

**Solução:** Usar mesmo `event_id` em ambos:

```javascript
// Frontend (Pixel)
fbq('track', 'Lead', {
  value: 2027.38,
  currency: 'BRL'
}, {
  eventID: eventID  // Mesmo ID!
});
```

Meta deduplica automaticamente se `event_id` for igual.

## Dependências Python

```bash
pip install facebook-business
```

## Variáveis de Ambiente

```python
PIXEL_ID = "seu_pixel_id"
ACCESS_TOKEN = "seu_token_capi"
```

## Fases de Implementação

### Fase 1: Básico (50% → 80-90%)
**Esforço:** Baixo (2h)
**Status:** Planejado

- [ ] Capturar `fbp`, `fbc` no frontend
- [ ] Gerar `event_id` único
- [ ] Capturar `user_agent`, `event_source_url`
- [ ] Adicionar campos no webhook
- [ ] Armazenar dados na planilha
- [ ] Capturar `client_ip` no backend
- [ ] Implementar envio CAPI para LeadQualified (D10)
- [ ] Implementar envio CAPI para Purchase
- [ ] Configurar deduplicação Pixel + CAPI
- [ ] Testar em Meta Event Manager

### Fase 2: Enriquecimento (80-90% → 90-95%)
**Esforço:** Médio (1-2 dias)
**Status:** Futuro

**Fingerprinting Básico:**
```javascript
// FingerprintJS (open-source)
import FingerprintJS from '@fingerprintjs/fingerprintjs';
const fp = await FingerprintJS.load();
const result = await fp.get();
const fingerprint = result.visitorId;
```

**Dados Adicionais:**
```javascript
const enrichedData = {
  device_type: /Mobile|Android|iPhone/.test(navigator.userAgent) ? 'mobile' : 'desktop',
  browser: navigator.userAgent.match(/(Chrome|Firefox|Safari|Edge)/)?.[1] || 'unknown',
  os: navigator.platform,
  screen_resolution: `${screen.width}x${screen.height}`,
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  language: navigator.language,
  referrer: document.referrer
};
```

**IP Real (não proxy):**
```javascript
// Usar serviço externo para garantir IP real
const ipResponse = await fetch('https://api.ipify.org?format=json');
const { ip } = await ipResponse.json();
```

### Fase 3: Avançado (90-95% → 95-98%)
**Esforço:** Alto (1-2 semanas)
**Status:** Futuro

- Processamento real-time (substituir batch)
- Probabilistic matching (similaridade de emails)
- Cross-device tracking

### Fase 4: Expert (95-98% → 98-100%)
**Esforço:** Muito Alto (meses)
**Status:** Não recomendado

- Fingerprinting avançado (canvas, WebGL, audio)
- ML para matching
- Infraestrutura dedicada
