# Captura de Dados para CAPI

## Objetivo
Recall de atribuição Meta: 50-60% → 90-100%

## Fluxo Completo do Sistema

### 1. Frontend captura dados do lead
- Usuário preenche formulário (nome, email, telefone)
- JavaScript captura: `_fbp`, `_fbc`, `event_id`, `user_agent`, `event_source_url`
- Envia para 2 lugares **em paralelo**:
  - **SellFlux** (sistema legado) → salva dados básicos na planilha (sem fbp/fbc)
  - **API CAPI** (novo) → salva dados CAPI no PostgreSQL (com fbp/fbc)

### 2. API salva dados CAPI no banco
- Endpoint: `POST /webhook/lead_capture`
- Recebe dados do frontend
- Captura `client_ip` do header
- Salva tudo no PostgreSQL (tabela `leads_capi`)

### 3. Google Sheets armazena lead (sistema legado)
- SellFlux salva: nome, email, telefone, UTMs, data
- Não salva dados CAPI (fbp, fbc ficam no PostgreSQL)

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

## 📝 Implementação Frontend

### Código Pronto

**Usar o arquivo:** `codigo_formulario_completo_com_capi.js`

Substituir o código JavaScript existente da página por este arquivo completo (já tem tudo integrado).

---

## Backend (Já Implementado)

O backend **já está pronto** nos arquivos:
- `app.py` - Endpoint `/webhook/lead_capture` (recebe dados do formulário e salva no PostgreSQL)
- `capi_integration.py` - Envia eventos `LeadQualified` para Meta CAPI
- `database.py` - Funções de banco de dados
- `apps-script-code.js` - Batch diário de leads D10 (1x/dia às 00:00)

---

## 🧪 Como Testar

### Passo 1: Adicionar Código no Formulário

Substituir código JavaScript da página pelo arquivo `codigo_formulario_completo_com_capi.js`

---

### Passo 2: Testar no Navegador

1. Abrir Console: `Cmd + Option + I` → aba "Console"
2. Preencher formulário de teste
3. Clicar em Enviar
4. Verificar Console - deve aparecer:
```
📊 CAPI - FBP: fb.1.1234567890... | FBC: ⚠️ ausente (normal)
✅ CAPI enviado: {status: "success", message: "Lead capturado com sucesso", ...}
```

---

### Passo 3: Confirmar Dados no Banco

```bash
curl https://smart-ads-api-12955519745.us-central1.run.app/webhook/lead_capture/stats
```

**Deve retornar:**
```json
{"total_leads":1,"leads_with_fbp":1,"leads_with_fbc":0,"fbp_fill_rate":100.0,"fbc_fill_rate":0.0}
```

**✅ Sucesso:** Se `total_leads > 0`, os dados estão chegando no banco!

---

## 🔮 Futuras Melhorias (Opcional)

**Fase 2 (80-90% → 90-95%):** Adicionar fingerprinting básico, dados de dispositivo, timezone, referrer

**Fase 3 (90-95% → 95-98%):** Processamento real-time, probabilistic matching de emails, cross-device tracking

**Fase 4 (95-98% → 98-100%):** Fingerprinting avançado (canvas, WebGL), ML para matching (não recomendado - custo muito alto)
