# 🔐 Permissões Necessárias para Integração Meta Ads API

**Última atualização**: Outubro 2025
**Documento**: Guia completo de permissões para acesso à conta Meta Ads do cliente

---

## 📋 Resumo Executivo

Para que o sistema Smart Ads funcione com a conta real do cliente, você precisa de **acesso à conta de anúncios com permissão "Analyst" (Analista)**. Esta é a permissão **MAIS RESTRITA** e **SEGURA** que permite ler dados de campanhas sem qualquer capacidade de modificação ou acesso a informações financeiras sensíveis.

---

## 🎯 Opções de Acesso (da mais segura à menos segura)

### ✅ **OPÇÃO 1: Partner com Analyst (RECOMENDADO)**

**Como funciona:**
- Cliente adiciona você como **Partner (Parceiro)** no Business Manager dele
- Cliente atribui permissão **"Analyst"** na conta de anúncios específica
- Você cria um **System User** no seu próprio Business Manager
- Cliente concede acesso ao seu System User com permissão Analyst

**O que você PODE fazer:**
- ✅ Ler dados de campanhas (nomes, IDs, estrutura)
- ✅ Acessar insights e métricas (impressions, clicks, spend, CTR, etc.)
- ✅ Gerar relatórios de performance
- ✅ Visualizar dados históricos de gastos

**O que você NÃO PODE fazer:**
- ❌ Criar, editar ou deletar campanhas/adsets/ads
- ❌ Ver métodos de pagamento
- ❌ Ver informações de billing (faturas, transações bancárias)
- ❌ Modificar orçamentos
- ❌ Adicionar ou remover usuários
- ❌ Alterar configurações da conta
- ❌ Acessar dados financeiros sensíveis (cartões de crédito, etc.)

**Permissões de API necessárias:**
- `ads_read` (leitura de insights)
- `read_insights` (leitura de relatórios)

**Nível de Risco:** 🟢 **MUITO BAIXO** - Acesso totalmente read-only

---

### ⚠️ **OPÇÃO 2: Partner com Advertiser**

**Como funciona:**
- Similar à Opção 1, mas com permissão **"Advertiser"**

**O que você PODE fazer (adicional ao Analyst):**
- ✅ Criar e gerenciar campanhas/adsets/ads
- ✅ Editar orçamentos de campanhas
- ✅ **Ver informações de billing** (gasto total, faturas)

**O que você NÃO PODE fazer:**
- ❌ Adicionar/remover métodos de pagamento
- ❌ Fazer alterações em cartões de crédito
- ❌ Gerenciar usuários
- ❌ Alterar configurações da conta

**Permissões de API necessárias:**
- `ads_management` (leitura + escrita)
- `read_insights`

**Nível de Risco:** 🟡 **MÉDIO** - Pode criar/editar campanhas e gastar dinheiro do cliente

---

### 🔴 **OPÇÃO 3: Partner com Admin (NÃO RECOMENDADO)**

**O que você PODE fazer:**
- ✅ Controle total da conta de anúncios
- ✅ Gerenciar billing e pagamentos
- ✅ Adicionar/remover usuários
- ✅ Alterar todas as configurações

**Nível de Risco:** 🔴 **MUITO ALTO** - Controle total da conta

---

## 🏗️ Arquitetura Recomendada para Produção

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENTE (Business Manager)               │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Ad Account: act_XXXXXXXXX                          │   │
│  │  - Campanhas Reais                                  │   │
│  │  - Dados de Spend Reais                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            │ Permissão: Analyst (read-only) │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Partners > Seu Business Manager                    │   │
│  │  - Acesso: Analyst em Ad Account                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              VOCÊ (Seu Business Manager)                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  System User (para API)                             │   │
│  │  - Access Token de longa duração                    │   │
│  │  - Permissões: ads_read, read_insights              │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Smart Ads API (Cloud Run)                          │   │
│  │  - Token armazenado como Secret                     │   │
│  │  - Busca dados via Meta Ads API                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Passo a Passo: Como Cliente Deve Conceder Acesso

### **Para o Cliente Executar:**

#### 1. Adicionar Você como Partner

1. Acessar **Business Settings** no Meta Business Manager
2. Ir em **Users** > **Partners**
3. Clicar em **Add Partners**
4. Inserir o **Business ID** do seu Business Manager
5. Clicar em **Add**

#### 2. Conceder Permissão na Conta de Anúncios

1. Ainda em **Business Settings**, ir em **Accounts** > **Ad Accounts**
2. Selecionar a conta de anúncios específica
3. Clicar em **Add Partners**
4. Selecionar seu Business Manager (adicionado no passo 1)
5. **IMPORTANTE**: Selecionar permissão **"Analyst"** (NÃO "Advertiser" ou "Admin")
6. Clicar em **Save Changes**

---

### **Para Você Executar:**

#### 3. Criar System User no Seu Business Manager

1. Acessar **seu** Business Settings
2. Ir em **Users** > **System Users**
3. Clicar em **Add**
4. Nomear: `smart-ads-api-system-user`
5. Selecionar role: **Admin** (no SEU Business Manager)
6. Clicar em **Create System User**

#### 4. Gerar Access Token

1. Clicar no System User criado
2. Clicar em **Generate New Token**
3. Selecionar o App (criar um app se necessário)
4. Selecionar permissões:
   - ✅ `ads_read`
   - ✅ `read_insights`
5. Copiar e salvar o token de forma segura

#### 5. Atribuir Acesso do System User à Conta do Cliente

1. Ainda na tela do System User
2. Clicar em **Assign Assets**
3. Selecionar **Ad Accounts**
4. Localizar a conta do cliente (act_XXXXXXXXX)
5. Verificar que aparece com permissão "Analyst"
6. Salvar

---

## 🔒 Segurança e Boas Práticas

### ✅ **O Cliente DEVE Saber:**

1. **Você terá acesso apenas a dados de leitura** (campanhas, gastos, métricas)
2. **Você NÃO pode modificar nada** (criar/editar/deletar campanhas)
3. **Você NÃO verá informações financeiras sensíveis** (cartões, métodos de pagamento)
4. **O cliente pode revogar seu acesso a qualquer momento** (em 30 segundos)
5. **O token de acesso pode ser regenerado** se houver suspeita de comprometimento

### ✅ **Você DEVE Garantir:**

1. **Armazenar o token de forma segura** (Google Secret Manager, não no código)
2. **Usar HTTPS** para todas as comunicações
3. **Não compartilhar o token** com terceiros
4. **Registrar (log) todos os acessos à API** para auditoria
5. **Renovar tokens periodicamente** (tokens de System User expiram em 60 dias por padrão)

---

## 🧪 Diferença: Sandbox vs Produção

| Aspecto | Sandbox (Atual) | Produção (Com Cliente) |
|---------|-----------------|------------------------|
| **Dados** | Fictícios (criados manualmente) | Reais (campanhas ativas) |
| **Spend** | Sempre $0.00 (sem histórico) | Valores reais de gasto |
| **Campanhas** | Criadas manualmente, pausadas | Campanhas reais, ativas |
| **Insights** | Vazios (sem impressions) | Dados completos de performance |
| **Cobrança** | Nunca cobrado | Gastos reais do cliente |
| **Objetivo** | Testar estrutura de API | Análise real de performance |

---

## ❓ FAQ - Perguntas Frequentes

### **1. O cliente será cobrado por me dar acesso?**
❌ **NÃO**. Adicionar parceiros/usuários no Business Manager é 100% gratuito.

### **2. Posso gastar dinheiro do cliente acidentalmente com acesso Analyst?**
❌ **IMPOSSÍVEL**. Analyst é read-only. Você não pode criar/editar campanhas ou orçamentos.

### **3. Verei os dados do cartão de crédito/método de pagamento do cliente?**
❌ **NÃO**. Analyst não tem acesso a métodos de pagamento ou informações bancárias.

### **4. Verei quanto o cliente gastou no total?**
✅ **SIM**. Você verá o spend (gasto) agregado por campanha/adset/ad, mas não os detalhes financeiros como forma de pagamento.

### **5. O cliente pode remover meu acesso a qualquer momento?**
✅ **SIM**. Em Business Settings > Partners, ele pode remover seu acesso instantaneamente.

### **6. Preciso de um Business Manager próprio?**
✅ **SIM**. Para criar System Users e gerar tokens de API de longa duração, você precisa ter seu próprio Business Manager.

### **7. Posso usar minha conta pessoal do Facebook?**
⚠️ **NÃO RECOMENDADO**. Tokens de usuários pessoais expiram em 60-90 dias e dependem de você estar logado. System Users são permanentes e não dependem de login.

### **8. O que acontece se eu perder o access token?**
✅ **SEM PROBLEMA**. Você pode gerar um novo token no Business Manager a qualquer momento. O token antigo é revogado automaticamente.

---

## 📊 Dados Que Você TERÁ Acesso (com Analyst)

### ✅ **Estrutura de Campanhas**
- Nome da campanha (ex: "DEVLF | CAP | FRIO | FASE 04...")
- Nome do adset (ex: "ADV | Linguagem de programação")
- Nome do ad (ex: "DEV-AD0027-vid-captação-V0-DEV")
- Status (ativo, pausado, etc.)
- Datas de criação e modificação

### ✅ **Métricas de Performance**
- Impressions (impressões)
- Clicks (cliques)
- CTR (taxa de cliques)
- CPC (custo por clique)
- CPM (custo por mil impressões)
- **Spend (gasto total por período)**
- Conversões (se configuradas)
- Reach (alcance)

### ✅ **Dados Agregados**
- Gasto por dia/semana/mês
- Gasto por campanha/adset/ad
- Performance por dimensão UTM

### ❌ **Dados Que Você NÃO TERÁ Acesso**
- Métodos de pagamento (cartões, PayPal, etc.)
- Limites de crédito
- Faturas detalhadas (PDFs de billing)
- Informações bancárias
- Outros ativos do cliente não compartilhados (Pixels, Pages, etc.)

---

## 🚀 Próximos Passos Recomendados

1. **Criar seu próprio Business Manager** (se ainda não tem)
   - https://business.facebook.com/create

2. **Criar um App no Facebook Developers** (para gerar tokens)
   - https://developers.facebook.com/apps/create/

3. **Enviar para o cliente:**
   - Este documento (PERMISSOES_META_ADS.md)
   - Seu Business ID
   - Instruções de como adicionar você como Partner com permissão Analyst

4. **Após receber acesso:**
   - Criar System User
   - Gerar Access Token
   - Testar conexão com a API
   - Atualizar `meta_config.py` com novo token e account_id
   - Deploy no Cloud Run

---

## 📞 Suporte e Referências

- **Meta Business Help Center**: https://www.facebook.com/business/help
- **System Users Documentation**: https://developers.facebook.com/docs/marketing-api/system-users/
- **Marketing API Permissions**: https://developers.facebook.com/docs/marketing-api/overview/authorization/

---

**Última revisão**: 2025-10-13
**Versão**: 1.0
**Status**: ✅ Validado com fontes oficiais Meta
