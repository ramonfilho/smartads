# 🚀 GUIA COMPLETO: Duplicar Landing Page com Integração CAPI

**Objetivo:** Duplicar a LP `https://lp.devclub.com.br/inscricao-lf-v2-crt/` e integrar com seu sistema de captura de leads para Meta CAPI.

**Problema Identificado:** Frontend atual demora para responder e não implementou os campos necessários para o Event Quality Score 9+ da Meta.

---

## 📊 CAMPOS NECESSÁRIOS PARA META CAPI (Event Quality Score 9+)

| Campo | Status | Origem |
|-------|--------|--------|
| ✅ email | Implementado | Formulário |
| ✅ phone | Implementado | Formulário |
| ✅ first_name | **NOVO** | Split do nome completo |
| ✅ last_name | **NOVO** | Split do nome completo |
| ✅ fbp | Implementado | Cookie `_fbp` |
| ✅ fbc | Implementado | Cookie `_fbc` |
| ✅ event_id | Implementado | Gerado automaticamente |
| ✅ client_ip | Implementado | Capturado no backend |
| ✅ user_agent | Implementado | `navigator.userAgent` |
| ✅ event_source_url | Implementado | `window.location.href` |

---

## ✅ ALTERAÇÕES JÁ FEITAS NO SISTEMA

### 1. Backend (database.py)
- ✅ Adicionados campos `first_name` e `last_name` no modelo `LeadCAPI`
- ✅ Campo `name` mantido para compatibilidade

### 2. API (app.py)
- ✅ Schema `LeadCaptureRequest` atualizado para aceitar `first_name` e `last_name`
- ✅ Webhook salva os novos campos no banco

### 3. JavaScript (codigo_formulario_completo_com_capi.js)
- ✅ Função `splitName()` criada para separar nome completo
- ✅ Payload agora envia `first_name` e `last_name` separados

### 4. Migração SQL
- ✅ Arquivo `migration_add_first_last_name.sql` criado
- ✅ Popula campos para registros existentes

---

## 🛠️ PASSOS PARA DUPLICAÇÃO E DEPLOY

### **PASSO 1: Aplicar Migração no Banco de Dados**

```bash
# Conectar ao PostgreSQL do Cloud SQL
gcloud sql connect smart-ads-db --user=postgres --database=smart_ads

# OU usar cliente local (substitua credenciais)
psql -h SEU_IP -U postgres -d smart_ads

# Executar migração
\i /Users/ramonmoreira/Desktop/smart_ads/V2/api/migration_add_first_last_name.sql

# Verificar resultado
SELECT first_name, last_name, name FROM leads_capi LIMIT 10;
```

---

### **PASSO 2: Fazer Deploy do Backend Atualizado**

```bash
cd /Users/ramonmoreira/Desktop/smart_ads/V2/api

# Build e deploy no Cloud Run
gcloud run deploy smart-ads-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="$(cat .env | xargs)"

# Testar endpoint
curl -X POST https://smart-ads-api-gazrm25mda-uc.a.run.app/webhook/lead_capture \
  -H "Content-Type: application/json" \
  -d '{
    "name": "João Silva",
    "first_name": "João",
    "last_name": "Silva",
    "email": "joao@test.com",
    "event_id": "test_'$(date +%s)'"
  }'

# Deve retornar: {"status":"success",...}
```

---

### **PASSO 3: Baixar Landing Page Original**

#### Opção A: Manualmente (Navegador)
```
1. Abrir: https://lp.devclub.com.br/inscricao-lf-v2-crt/
2. Cmd+U (Mac) ou Ctrl+U (Windows) - View Source
3. Cmd+S - Salvar como "index.html"
4. Baixar também:
   - Logo: logo-11.png
   - Ícones SVG: agenda.svg, etc
```

#### Opção B: Via Terminal (Mais Rápido)
```bash
cd /Users/ramonmoreira/Desktop/smart_ads/V2
mkdir landing_page_capi
cd landing_page_capi

# Baixar HTML
curl -o index.html https://lp.devclub.com.br/inscricao-lf-v2-crt/

# Baixar assets (ajustar URLs conforme necessário)
curl -o logo.png https://lp.devclub.com.br/path-to-logo.png
```

---

### **PASSO 4: Integrar Código JavaScript Atualizado**

```bash
cd /Users/ramonmoreira/Desktop/smart_ads/V2/landing_page_capi

# Copiar código JavaScript atualizado
cp ../api/codigo_formulario_completo_com_capi.js ./
```

Agora edite o `index.html`:

1. **Localize** o `<script>` do formulário (geralmente antes de `</body>`)
2. **Remova** todo o JavaScript do formulário existente
3. **Adicione** ANTES do `</body>`:

```html
<!-- Código CAPI Atualizado -->
<script src="codigo_formulario_completo_com_capi.js"></script>
```

**OU** cole diretamente inline:

```html
<script>
  [cole todo o conteúdo de codigo_formulario_completo_com_capi.js aqui]
</script>
```

---

### **PASSO 5: Ajustar Configurações da Página**

No arquivo JavaScript, ajuste:

```javascript
// Linha 67 - URL do webhook (trocar para PRODUÇÃO)
const response = await fetch('https://smart-ads-api-gazrm25mda-uc.a.run.app/webhook/lead_capture', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
});

// Linha 100 - ActiveCampaign (OPCIONAL - remover se não usar)
tempForm.action = "https://rodolfomori.activehosted.com/proc.php";
// Se não quiser enviar para ActiveCampaign, comente linhas 59-130

// Linha 320 - URL de redirecionamento (ALTERAR!)
let redirectURL = "https://SUA-PAGINA-DE-OBRIGADO.com/";
```

---

### **PASSO 6: Testar Localmente**

```bash
cd /Users/ramonmoreira/Desktop/smart_ads/V2/landing_page_capi

# Iniciar servidor local
python3 -m http.server 8000

# Abrir: http://localhost:8000
```

#### Checklist de Testes:

1. **Abrir Console do Navegador** (Cmd+Option+I)

2. **Preencher formulário:**
   - Nome: João Silva Santos
   - Email: teste@example.com
   - Telefone: (11) 96123-4567
   - Tem computador: Sim

3. **Clicar em "Enviar"**

4. **Verificar Console:**
   ```
   ✅ Deve aparecer:
   📊 CAPI - FBP: fb.1.xxxxx | FBC: ⚠️ ausente
   ✅ CAPI enviado: {status: "success", lead_id: 123, event_id: "lead_..."}
   ```

5. **Verificar Network Tab (F12 > Network):**
   - Request para `/webhook/lead_capture`
   - Status: 200 OK
   - Response: `{"status":"success",...}`

6. **Verificar Payload (Request Body):**
   ```json
   {
     "name": "João Silva Santos",
     "first_name": "João",
     "last_name": "Silva Santos",
     "email": "teste@example.com",
     "phone": "+5511961234567",
     "fbp": "fb.1.xxxxx",
     "fbc": null,
     "event_id": "lead_1731600000_abc123",
     "user_agent": "Mozilla/5.0...",
     "event_source_url": "http://localhost:8000/"
   }
   ```

7. **Verificar redirecionamento:** Deve ir para página de obrigado

---

### **PASSO 7: Deploy da Landing Page**

#### Opção A: Vercel (Recomendado)

```bash
# Instalar Vercel CLI
npm install -g vercel

# Deploy
cd /Users/ramonmoreira/Desktop/smart_ads/V2/landing_page_capi
vercel --prod

# Seguir prompts:
# - Login com GitHub/Google
# - Project name: landing-page-capi
# - Deploy

# URL gerada: https://landing-page-capi.vercel.app
```

#### Opção B: Netlify Drag & Drop

1. Acessar: https://app.netlify.com/drop
2. Arrastar pasta `landing_page_capi`
3. Aguardar upload
4. URL gerada automaticamente

#### Opção C: GitHub Pages

```bash
cd /Users/ramonmoreira/Desktop/smart_ads/V2/landing_page_capi

git init
git add .
git commit -m "feat: landing page com integração CAPI completa"

# Criar repo no GitHub
gh repo create landing-page-capi --public --source=. --remote=origin --push

# Habilitar GitHub Pages:
# Settings > Pages > Source: main branch > Save
# URL: https://SEU-USUARIO.github.io/landing-page-capi/
```

---

### **PASSO 8: Configurar GTM Server (Stape) - OPCIONAL**

Se você usa GTM Server-Side (Stape), configure webhook adicional:

```javascript
// No GTM Server Container, adicionar tag:
Tag Type: HTTP Request
URL: https://smart-ads-api-gazrm25mda-uc.a.run.app/webhook/lead_capture
Method: POST
Content-Type: application/json

// Body (JSON):
{
  "name": {{DL - user_name}},
  "first_name": {{DL - first_name}},
  "last_name": {{DL - last_name}},
  "email": {{DL - user_email}},
  "phone": {{DL - user_phone}},
  "fbp": {{Cookie - _fbp}},
  "fbc": {{Cookie - _fbc}},
  "event_id": {{Event ID}},
  "user_agent": {{User Agent}},
  "client_ip": {{Client IP}},
  "event_source_url": {{Page URL}},
  "utm_source": {{DL - utm_source}},
  "utm_medium": {{DL - utm_medium}},
  "utm_campaign": {{DL - utm_campaign}}
}

// Trigger: Custom Event = "cadastro"
```

---

### **PASSO 9: Testar em Produção**

#### 9.1 - Teste Manual

```bash
# Abrir URL da página deployada
open https://landing-page-capi.vercel.app

# Preencher formulário real
# Verificar Console (F12)
# Confirmar envio bem-sucedido
```

#### 9.2 - Verificar Banco de Dados

```bash
# Chamar endpoint de stats
curl https://smart-ads-api-gazrm25mda-uc.a.run.app/webhook/lead_capture/stats

# Verificar últimos leads
curl https://smart-ads-api-gazrm25mda-uc.a.run.app/webhook/lead_capture/recent

# Ver lead específico no PostgreSQL
gcloud sql connect smart-ads-db --user=postgres
\c smart_ads
SELECT first_name, last_name, email, fbp, event_id
FROM leads_capi
ORDER BY created_at DESC
LIMIT 5;
```

#### 9.3 - Criar Anúncio de Teste na Meta

```
1. Ads Manager > Criar Campanha
2. Objetivo: Leads
3. Ad Set:
   - Público: Restrito (ex: só você via Custom Audience)
   - Orçamento: $5/dia
4. Ad:
   - Link: https://landing-page-capi.vercel.app/?fbclid=TEST
   - CTA: "Inscreva-se"
5. Publicar
```

#### 9.4 - Simular Clique em Anúncio (Gerar FBC)

```bash
# Adicionar fbclid manualmente na URL
open "https://landing-page-capi.vercel.app/?fbclid=IwAR0TESTE123"

# Preencher formulário
# Verificar se FBC foi capturado no Console:
# FBC: fb.1.1234567890123.IwAR0TESTE123
```

#### 9.5 - Verificar Event Quality Score

```
1. Meta Events Manager
2. Data Sources > [Seu Pixel]
3. Event Matching > Ver detalhes
4. Verificar score (meta: 9+)
5. Confirmar campos recebidos:
   ✅ em (email)
   ✅ ph (phone)
   ✅ fn (first_name) ← NOVO!
   ✅ ln (last_name) ← NOVO!
   ✅ fbp
   ✅ fbc (se clicou em ad)
   ✅ client_ip_address
   ✅ client_user_agent
   ✅ event_source_url
```

---

## 🎯 CHECKLIST FINAL DE VALIDAÇÃO

```
✅ Migração SQL aplicada no banco
✅ Backend deployado com novos campos
✅ Endpoint /webhook/lead_capture aceita first_name/last_name
✅ Landing page duplicada e editada
✅ JavaScript atualizado integrado
✅ Teste local passou (Console mostra ✅ CAPI enviado)
✅ Deploy em produção (Vercel/Netlify/GitHub Pages)
✅ Formulário envia dados corretamente
✅ Dados chegam no PostgreSQL com first_name/last_name
✅ FBP capturado corretamente
✅ FBC aparece quando clica em anúncio
✅ Event ID único gerado
✅ Redirecionamento funciona
✅ Event Quality Score >= 9 (verificar após 24h)
✅ UTMs capturados corretamente
✅ ActiveCampaign continua funcionando (se habilitado)
```

---

## 🚨 TROUBLESHOOTING COMUM

### Problema 1: Erro 422 - Unprocessable Entity

```
Causa: Backend recebeu campos faltando ou inválidos
Solução:
1. Verificar payload no Network tab
2. Conferir se event_id está sendo gerado
3. Confirmar email é válido
```

### Problema 2: CORS Error

```
❌ Access to fetch blocked by CORS policy

Solução: Adicionar CORS no backend (app.py):

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://landing-page-capi.vercel.app"],  # Ajustar domínio
    allow_methods=["POST"],
    allow_headers=["*"],
)
```

### Problema 3: FBP não capturado

```
❌ FBP: null no console

Causas:
- Bloqueador de ads ativo
- Cookie de terceiros bloqueado
- Facebook Pixel não carregou

Solução: Adicionar retry com delay
setTimeout(() => {
  const fbp = getCookie('_fbp');
  if (fbp) sendToCapiAPI(...);
}, 2000);
```

### Problema 4: first_name/last_name null

```
Causa: Função splitName() não está sendo chamada

Debug:
console.log('Nome original:', fullname);
const { firstName, lastName } = splitName(fullname);
console.log('Separado:', firstName, lastName);

Verificar se payload contém campos:
console.log('Payload:', JSON.stringify(payload, null, 2));
```

### Problema 5: Webhook retorna 500

```
❌ Error 500: Internal Server Error

Debug:
1. Ver logs do Cloud Run:
   gcloud run services logs read smart-ads-api --limit 50

2. Testar endpoint diretamente:
   curl -X POST https://smart-ads-api-gazrm25mda-uc.a.run.app/webhook/lead_capture \
     -H "Content-Type: application/json" \
     -d '{"name":"Test","first_name":"John","last_name":"Doe","email":"test@test.com","event_id":"test123"}'

3. Verificar se migração SQL foi aplicada:
   SELECT column_name FROM information_schema.columns
   WHERE table_name = 'leads_capi';
```

---

## 📞 RESUMO EXECUTIVO (TL;DR)

1. ✅ **Backend atualizado** - Aceita `first_name` e `last_name`
2. ✅ **JavaScript atualizado** - Separa nome automaticamente
3. ✅ **Migração criada** - Adiciona colunas no banco

**Próximos passos:**
1. Aplicar migração SQL (PASSO 1)
2. Deploy do backend (PASSO 2)
3. Baixar LP e integrar JS (PASSOS 3-5)
4. Testar local (PASSO 6)
5. Deploy da LP (PASSO 7)
6. Testar produção e Event Quality Score (PASSOS 8-9)

**Tempo estimado:** 45-60 minutos

**Event Quality Score esperado:** 9+ (com todos os campos)

---

## 📚 RECURSOS ADICIONAIS

**Arquivos importantes:**
- `codigo_formulario_completo_com_capi.js` - JavaScript com split de nome
- `migration_add_first_last_name.sql` - Migração do banco
- `database.py:28-33` - Modelo atualizado
- `app.py:358-365` - Schema da API

**Endpoints úteis:**
- `POST /webhook/lead_capture` - Recebe leads
- `GET /webhook/lead_capture/stats` - Estatísticas
- `GET /webhook/lead_capture/recent` - Últimos 10 leads

**Documentação Meta:**
- [CAPI Parameters](https://developers.facebook.com/docs/marketing-api/conversions-api/parameters)
- [Event Matching](https://www.facebook.com/business/help/611774685654668)

---

**Última atualização:** 2025-11-14
**Versão:** 2.0 (com first_name/last_name)
