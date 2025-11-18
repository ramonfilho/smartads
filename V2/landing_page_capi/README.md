# Landing Page com Integração CAPI (Meta Conversions API)

## 🚀 Demo em Produção
**URL:** https://endearing-chebakia-55eb90.netlify.app

## 📋 O que foi implementado

Esta landing page captura leads e envia **automaticamente** para:
1. ✅ ActiveCampaign (CRM)
2. ✅ Webhook próprio com dados CAPI para Meta

### Campos capturados e enviados:

```javascript
{
  "name": "Nome Completo",           // Nome completo
  "first_name": "Nome",              // ← NOVO - Primeiro nome separado
  "last_name": "Sobrenome",          // ← NOVO - Sobrenome separado
  "email": "email@example.com",
  "phone": "+5511999999999",
  "fbp": "fb.1.xxxxx",               // Cookie Facebook Browser ID
  "fbc": "fb.1.click_id",            // Cookie Facebook Click ID (quando clica em ad)
  "event_id": "lead_123456789_abc",  // ID único para deduplicação
  "user_agent": "Mozilla/5.0...",
  "event_source_url": "https://...",
  "client_ip": "1.2.3.4",           // Capturado no backend
  "utm_source": "facebook",
  "utm_medium": "cpc",
  "utm_campaign": "campaign_name",
  "utm_term": "term",
  "utm_content": "content",
  "tem_comp": "SIM"                  // Tem computador
}
```

## 🎯 Benefícios para Meta Ads

Com esses campos, o **Event Quality Score** da Meta sobe para **9-10**, melhorando:
- 📈 Performance das campanhas
- 🎯 Otimização do algoritmo
- 💰 Custo por lead (CPL)

## 🔧 Como Implementar em Outra Página

### Opção 1: Código Inline (usado nesta página)

No evento de submit do formulário, **antes** de enviar para ActiveCampaign:

```javascript
// 1. Funções auxiliares
function getCookie(name) {
    const value = \`; \${document.cookie}\`;
    const parts = value.split(\`; \${name}=\`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
}

function splitName(fullName) {
    if (!fullName) return { firstName: null, lastName: null };
    const trimmedName = fullName.trim();
    const spaceIndex = trimmedName.indexOf(' ');
    if (spaceIndex === -1) return { firstName: trimmedName, lastName: null };
    return {
        firstName: trimmedName.substring(0, spaceIndex),
        lastName: trimmedName.substring(spaceIndex + 1).trim()
    };
}

// 2. Capturar dados CAPI
const fbp = getCookie('_fbp');
const fbc = getCookie('_fbc');
const eventID = \`lead_\${Date.now()}_\${Math.random().toString(36).substr(2, 9)}\`;
const { firstName, lastName } = splitName(fullname); // fullname = nome capturado do form

console.log('📊 CAPI - FBP:', fbp || '❌ ausente', '| FBC:', fbc || '⚠️ ausente');

// 3. Enviar para webhook
fetch('https://smart-ads-api-12955519745.us-central1.run.app/webhook/lead_capture', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        name: fullname,
        first_name: firstName,
        last_name: lastName,
        email: email,
        phone: phone,
        tem_comp: hasComputer,
        fbp: fbp,
        fbc: fbc,
        event_id: eventID,
        user_agent: navigator.userAgent,
        event_source_url: window.location.href,
        utm_source: utmParams.utm_source || null,
        utm_medium: utmParams.utm_medium || null,
        utm_campaign: utmParams.utm_campaign || null,
        utm_term: utmParams.utm_term || null,
        utm_content: utmParams.utm_content || null
    })
})
.then(r => r.json())
.then(data => console.log('✅ CAPI enviado:', data))
.catch(err => console.error('❌ Erro CAPI:', err));

// 4. Continuar com envio normal para ActiveCampaign...
```

### Opção 2: Script Externo (mais organizado)

Use o arquivo `codigo_formulario_completo_com_capi.js` incluído neste repositório:

```html
<script src="codigo_formulario_completo_com_capi.js"></script>
```

⚠️ **IMPORTANTE:** Este arquivo assume que os campos do formulário têm os IDs:
- `#fullname`
- `#email`
- `#phone-input`
- `#field_144SIM` (radio "tem computador")
- `#field_144Não` (radio "não tem computador")

Se sua página usa IDs diferentes, ajuste no arquivo JS.

## 🔍 Como Testar

1. **Abra o Console** do navegador (Cmd+Option+I no Mac / F12 no Windows)
2. **Preencha o formulário**
3. **Clique em Enviar**
4. **Verifique as mensagens:**
   ```
   📊 CAPI - FBP: fb.1.xxxxx | FBC: ⚠️ ausente
   ✅ CAPI enviado: {status: "success", lead_id: 123, ...}
   ```

5. **Confirme no banco:**
   ```bash
   curl https://smart-ads-api-12955519745.us-central1.run.app/webhook/lead_capture/recent
   ```

## 📊 Verificar Event Quality Score na Meta

1. Acesse: **Meta Events Manager**
2. Vá em: **Data Sources > [Seu Pixel]**
3. Clique em: **Event Matching**
4. Verifique o score (meta: **9+**)

Você deve ver todos esses parâmetros chegando:
- ✅ em (email)
- ✅ ph (phone)
- ✅ fn (first_name) ← **NOVO**
- ✅ ln (last_name) ← **NOVO**
- ✅ fbp
- ✅ fbc (se clicou em anúncio)
- ✅ client_ip_address
- ✅ client_user_agent
- ✅ event_source_url

## 🆘 Troubleshooting

### Erro: CORS policy blocked
**Solução:** Backend já está configurado com CORS. Se usar outro domínio, adicionar em `app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://seu-dominio.com", "*"],
    ...
)
```

### FBP não capturado
**Causa:** Bloqueador de ads ou cookie bloqueado
**Solução:** Testar em navegador anônimo sem extensões

### first_name/last_name null
**Causa:** Função `splitName()` não executou
**Solução:** Adicionar console.log para debug:
```javascript
console.log('🔍 DEBUG - fullname:', fullname);
const { firstName, lastName } = splitName(fullname);
console.log('🔍 DEBUG - firstName:', firstName, '| lastName:', lastName);
```

## 📁 Arquivos Importantes

```
landing_page_capi/
├── index.html                                 # Página com código CAPI inline
├── codigo_formulario_completo_com_capi.js     # Script isolado (alternativa)
├── css/                                       # Estilos da página
├── images/                                    # Imagens da página
└── README.md                                  # Este arquivo
```

## 🔗 Endpoints da API

- **POST** `/webhook/lead_capture` - Recebe leads
- **GET** `/webhook/lead_capture/stats` - Estatísticas
- **GET** `/webhook/lead_capture/recent` - Últimos 10 leads

## 📞 Contato

Em caso de dúvidas, verificar:
- Documentação completa: `V2/api/GUIA_COMPLETO_DUPLICACAO_LP_CAPI.md`
- Instruções de deploy: `V2/api/documentacao_deploy_gcp.md`

---

**Última atualização:** 2025-11-14
**Status:** ✅ Em produção - Funcionando
**URL:** https://endearing-chebakia-55eb90.netlify.app
