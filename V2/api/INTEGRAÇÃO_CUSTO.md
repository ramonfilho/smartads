# 🎯 Objetivo

Expandir o sistema Smart Ads para incluir **análise econômica das campanhas**, integrando dados de custo da API do Meta Ads com as predições de qualidade do modelo ML.

## 📊 Métricas Calculadas

Para cada UTM (Campaign, Medium, Content, Term), calcular em **4 janelas temporais** (1D, 3D, 7D, Total):

### 1. CPL (Custo por Lead)
```
CPL = Custo Total / Número de Leads
```

### 2. Taxa de Conversão Projetada
```
Taxa Projetada = Σ(% leads no decil × taxa conversão do decil)
```

**Taxas de conversão por decil** (do `model_metadata_v1_devclub_rf_temporal_single-3.json`):

| Decil | Taxa de Conversão |
|-------|-------------------|
| D10   | 2.13%            |
| D9    | 1.37%            |
| D8    | 1.19%            |
| D7    | 1.19%            |
| D6    | 1.11%            |
| D5    | 1.02%            |
| D4    | 0.94%            |
| D3    | 0.85%            |
| D2    | 0.26%            |
| D1    | 0.26%            |

### 3. ROAS Projetado
```
ROAS = (Valor Produto × Taxa Projetada) / CPL
```

### 4. CPL Máximo Aceitável
```
CPL Max = (Valor Produto × Taxa Projetada) / ROAS Mínimo
```
Onde **ROAS Mínimo = 2.0x** (hardcoded)

### 5. Margem para Manobra
```
Margem = ((CPL Max - CPL Atual) / CPL Max) × 100%
```

---

## 🏗️ Decisões Arquiteturais

### Localização da Integração API Meta
**Cloud Run** (não Apps Script)

**Fluxo:**
```
Apps Script
  ↓ POST /analyze_utms_with_costs
  ↓ { leads, meta_credentials, business_config }

Cloud Run
  ↓ 1. Predições (lead_score, decile)
  ↓ 2. API Meta (custos por UTM/período)
  ↓ 3. Calcula 5 métricas
  ↓ 4. Retorna consolidado

Apps Script
  ↓ Escreve abas: "Análise UTM - 7D/3D/1D/Total"
```

### Interface
**Abas separadas por período** (não dropdown)

---

## ⚙️ Configuração

### Business Config
```python
# V2/api/app.py ou módulo de config
BUSINESS_CONFIG = {
    "product_value": X,
    "min_roas": 2.0,
    "conversion_rates": {  # Do model_metadata JSON
        "D1": 0.0026, "D2": 0.0026, "D3": 0.0085, "D4": 0.0094,
        "D5": 0.0102, "D6": 0.0111, "D7": 0.0119, "D8": 0.0119,
        "D9": 0.0137, "D10": 0.0213
    }
}
```

### Meta Ads Config
```python
META_CONFIG = {
    "account_id": "act_SEU_ID",      # ← Trocar
    "access_token": "SEU_TOKEN",     # ← Trocar
    "api_version": "v18.0"
}
```

---

## 🎯 Decisões de Implementação

### Estrutura das Abas
- **4 abas separadas:** "Análise UTM - 1D", "3D", "7D", "Total"
- **5 dimensões por aba:** Campaign, Medium, Term, Adset, Ad
- **Removido:** Content (substituído por Ad do Meta)

### Colunas Adicionadas
| Gasto | CPL | Taxa Proj. | ROAS Proj. | CPL Max | Margem |

### Níveis da API Meta
- **Campaign** (campanha)
- **Adset** (conjunto de anúncios)
- **Ad** (anúncio individual)

### Cruzamento de Dados
**Produção (Cliente):**
- UTMs já configurados nas URLs dos anúncios
- Cruzamento direto por nome (campaign_name → Campaign do Sheet)

**Teste (Sua Conta):**
- Mapeamento manual em `meta_campaign_mapping.py`
- Suas campanhas → UTMs fictícios do cliente

### Matching de Nomes
- Campaign do Meta pode ter sufixo: `| 2025-04-15|120220370119870390`
- Usar **matching parcial** (remover sufixo antes de cruzar)

### Abordagem de Trabalho
- **Granular:** Uma decisão/pergunta por vez
- **Validação incremental:** Testar cada etapa antes de avançar

---

## 🧪 Estratégia de Teste

### Fase 1: Conta Pessoal
- Usar sua conta Meta Ads para desenvolvimento
- Implementar lógica completa com seus dados
- Validar cálculos end-to-end

### Fase 2: Cliente
- Obter credenciais da conta do cliente
- Trocar apenas `account_id` e `access_token`
- Testar com dados reais do cliente


---

## 📝 Implementação - Passo a Passo

### 1. Setup API Meta (Sua Conta)

#### Obter Token de Acesso:
1. Ir para https://developers.facebook.com/tools/explorer/
2. Selecionar seu app (ou criar um)
3. Permissões: `ads_read`, `ads_management`
4. Gerar token (válido por 60 dias inicialmente)
5. Copiar token

#### Obter Account ID:
1. Ir para https://business.facebook.com/settings/ad-accounts
2. Copiar ID da conta (formato: `act_XXXXXXXXX`)

#### Testar manualmente:
```bash
curl "https://graph.facebook.com/v18.0/act_SEU_ID/insights?access_token=SEU_TOKEN&fields=campaign_name,spend,actions"
```

---

### 2. Cloud Run - Novo Endpoint

#### Criar arquivo: `V2/api/meta_integration.py`
```python
import requests
from datetime import datetime, timedelta

def get_campaign_costs(account_id, access_token, days=7):
    """Busca custos por campanha nos últimos X dias."""
    url = f"https://graph.facebook.com/v18.0/{account_id}/insights"

    since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    until = datetime.now().strftime('%Y-%m-%d')

    params = {
        'access_token': access_token,
        'time_range': f'{{"since":"{since}","until":"{until}"}}',
        'level': 'campaign',
        'fields': 'campaign_name,spend,actions',
        'limit': 1000
    }

    response = requests.get(url, params=params)
    return response.json()
```

#### Adicionar em: `V2/api/app.py`
```python
from .meta_integration import get_campaign_costs

@app.post("/analyze_utms_with_costs")
async def analyze_with_costs(request: AnalysisRequest):
    # 1. Fazer predições (já existe)
    predictions = pipeline.predict(request.leads)

    # 2. Buscar custos da API Meta
    costs_data = get_campaign_costs(
        request.meta_config.account_id,
        request.meta_config.access_token,
        days=7
    )

    # 3. Calcular métricas
    metrics = calculate_utm_metrics(predictions, costs_data)

    return metrics
```

---

### 3. Apps Script - Atualizar Chamada

#### Modificar: `V2/api/apps-script-code.js`
```javascript
function generateUTMAnalysisWithCosts() {
  // Configuração Meta (temporário - hardcoded)
  const META_CONFIG = {
    account_id: 'act_SEU_ID',
    access_token: 'SEU_TOKEN'
  };

  // Chamada para novo endpoint
  const response = UrlFetchApp.fetch(
    'https://smart-ads-api-12955519745.us-central1.run.app/analyze_utms_with_costs',
    {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify({
        leads: leads,
        meta_config: META_CONFIG,
        business_config: {
          product_value: 497,
          min_roas: 2.0
        }
      })
    }
  );

  // Escrever abas por período
  const data = JSON.parse(response.getContentText());
  writeAnalysisSheet('Análise UTM - 7D', data['7D']);
  writeAnalysisSheet('Análise UTM - 3D', data['3D']);
  writeAnalysisSheet('Análise UTM - 1D', data['1D']);
}
```

---

### 4. Estrutura das Abas

Cada aba (**Análise UTM - 7D**, etc.) terá:

| Dimensão | Valor | Leads | Gasto | CPL | %D10 | %D8-10 | Taxa Proj. | ROAS Proj. | CPL Max | Margem | Tier | Ação |
|----------|-------|-------|-------|-----|------|--------|------------|------------|---------|--------|------|------|

**Formatação condicional:**
- Margem > +50%: 🟢 Verde
- Margem 0 a +50%: 🟡 Amarelo
- Margem < 0%: 🔴 Vermelho

**Critérios de Ação:**
- **Campaign/Medium/Term/Adset** (controle de orçamento):
  - Margem > +50%: "Escalar"
  - Margem 0% a +50%: "Manter"
  - Margem < 0%: "Reduzir"
- **Ad/Content** (sem controle de orçamento):
  - Margem ≥ 0%: "Manter"
  - Margem < 0%: "Pausar"

---

## 🗺️ Roadmap - Próximos Módulos

### Módulos Completados:
1. ✅ **data/ingestion.py** - Ingestão de dados
2. ✅ **data/preprocessing.py** - Limpeza e preparação
3. ✅ **features/utm_unification.py** - Unificação UTM
4. ✅ **features/engineering.py** - Feature engineering
5. ✅ **features/encoding.py** - Encoding categórico
6. ✅ **serving/model_serving.py** - Predição

### 🎯 ATUAL: 7. api/meta_integration.py
**Objetivo:** Enriquecer análise UTM com dados econômicos (CPL, ROAS projetado).

**Sub-etapas:**
- [ ] 7.1 `get_campaign_costs()` - Buscar custos via API Meta
- [ ] 7.2 `calculate_projected_conversion()` - Taxa projetada por decil
- [ ] 7.3 `calculate_roas_metrics()` - ROAS, CPL Max, Margem
- [ ] 7.4 Novo endpoint `/analyze_utms_with_costs` no Cloud Run
- [ ] 7.5 Atualizar Apps Script para usar novo endpoint
- [ ] 7.6 Gerar abas por período (1D, 3D, 7D, Total)

**Configuração:**
- Hardcoded por enquanto (`product_value: XXXXX (À DECIDIR, CONSULTAR USUÁRIO)`, `min_roas: 2.0`)
- **API Meta:** Testado primeiro em conta pessoal, depois migrar para cliente
- **Interface:** Abas separadas (não dropdown) para simplicidade

### Próximos Módulos:
8. **investigation/** - Módulos de investigação (geram configs)
9. **Configuração cliente-específica** - Baseada nas investigações