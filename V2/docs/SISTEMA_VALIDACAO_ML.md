# Sistema de Validação ML - Documentação Técnica

**Atualizado:** 2025-12-15

---

## 1. CLASSIFICAÇÃO DE CAMPANHAS (3 TIPOS)

### 1.1 "Eventos ML"
- **Critério:** Tem "MACHINE LEARNING" no nome **E** usa eventos customizados
- **Optimization Goal:** `LeadQualified` ou `LeadQualifiedHighQuality`
- **Quando:** Criadas a partir de 25/11 na conta principal
- **Importância:** ⭐⭐⭐ **FOCO PRINCIPAL DA VALIDAÇÃO**

### 1.2 "Otimização ML"
- **Critério:** Tem "MACHINE LEARNING" no nome **MAS** usa eventos padrão
- **Optimization Goal:** `LEAD`, `OFFSITE_CONVERSIONS`, etc.
- **Quando:** Criadas em 18/11 na conta principal
- **Importância:** ⭐ Teste inicial, não é o foco

### 1.3 "Controle"
- **Critério:** NÃO tem "MACHINE LEARNING" no nome
- **Exemplos:** "ESCALA SCORE", "FAIXA A", "FAIXA B", etc.
- **Importância:** ⭐⭐ Base de comparação para validar performance ML

---

## 2. TIMELINE DE IMPLEMENTAÇÃO

| Data | Evento | Nota Score | Conta | Observação |
|------|--------|------------|-------|------------|
| 27/10 | Primeira campanha (teste painel) | - | - | Início dos testes |
| 10/11 | Eventos criados | 4.5 | 7867 | Primeira versão |
| 16/11 | Implementado fbp e fbc | 7.4 | - | Melhoria tracking |
| **18/11** | **Campanhas Eventos ML** | - | **7867** | Alimentar IA |
| **18/11** | **Campanhas Otimização ML** | - | **1880** | Sem eventos custom |
| 21/11 | Implementado sendBeacon | 9.1 | - | Tracking otimizado |
| **25/11** | **Campanhas Eventos ML** | - | **1880** | **3 campanhas criadas** |
| 02/12 | Todos os 2 eventos | - | - | Sistema completo |

---

## 3. INFORMAÇÕES TÉCNICAS

### 3.1 Contas Meta Ads
```yaml
Conta Principal (final 1880): act_188005769808959  # Ads - Rodolfo Mori
Conta Teste (final 7867):     act_786790755803474  # Ads - Gestor de IA

Meta Access Token:
EAAS9hlWC7lkBPmTFNOvHZBVZAW6ESTsmVCStlrcslFvNLxr2xBkKrI0kTmI6dou1aB5UOJLFwQo9gwAg1NZBCSWZCZAkxflALfnFeZC8nYRJJO5TZAfy1vswWFs0nCsZBpOanId4ULYCJMzPqt7UuhfuNBablHZAIchs1T7vEGWXgk6Sq2t8YirZBIPldNDVtyp7DxYQZDZD
```

### 3.2 Eventos Customizados CAPI
```yaml
LeadQualified:           # Decis D1-10 com valor de conversão
LeadQualifiedHighQuality: # Decis D8-D10 no começo, e D9-D10 depois
```

### 3.3 Janela de Matching
```yaml
max_match_days: 30      # Janela máxima leads → vendas
product_value: 2000.00  # Valor do produto (R$)
```

### 3.4 Períodos de Validação
```yaml
Estrutura:
  - Captação: 7 dias (geração de leads)
  - Vendas:   7 dias (matching leads → vendas)

Validação Atual (15/12):
  - Captação: 2025-11-25 a 2025-12-01
  - Vendas:   2025-12-01 a 2025-12-15
```

---

## 4. LÓGICA DE COMPARAÇÃO JUSTA

### 4.1 Princípio: "Maçãs com Maçãs"

O script **sempre roda com TODAS as campanhas**, mas a **comparação justa** é feita apenas entre:
- **Campanhas "Eventos ML"** (criadas em 25/11)
- **Campanhas "Controle"** que são estruturalmente similares

### 4.2 Critérios de Fair Control

Para cada campanha "Eventos ML", buscar campanhas "Controle" com:

| Critério | Tolerância | Prioridade |
|----------|------------|------------|
| **Budget** | ±30% | Obrigatório |
| **Anúncios** | 80%+ overlap | Obrigatório |
| **Targeting** | - | ❌ Removido (rate limits) |

### 4.3 Matched Pairs de Anúncios

**Conceito:** Anúncio (ad_code: AD0XXX) que aparece em **ambas**:
- Campanhas "Eventos ML"
- Campanhas "Controle"

**Comparação:** Performance do mesmo anúncio em diferentes contextos (ML vs Controle)

---

## 5. CAMPANHAS COM SPEND NO PERÍODO (27/11 a 01/12)

### 5.1 Conta Principal (act_188005769808959)

#### 📊 EVENTOS ML (3 campanhas)

1. **DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2 | 2025-11-27**
   - ID: 120236428684090390
   - Custom Event: LeadQualifiedHighQuality
   - Spend: R$ 2,344.43
   - Adsets: 10

2. **DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2 | 2025-11-27**
   - ID: 120236428684840390
   - Custom Event: LeadQualifiedHighQuality
   - Spend: R$ 459.45
   - Adsets: 4

3. **DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2 | 2025-11-27**
   - ID: 120236428684850390
   - Custom Event: LeadQualifiedHighQuality
   - Spend: R$ 1,441.66
   - Adsets: 4

#### 📊 OTIMIZAÇÃO ML (2 campanhas)

1. **DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2 | 2025-05-28**
   - ID: 120234748179990390
   - Optimization Goal: LEAD (padrão)
   - Spend: R$ 1,985.48

2. **DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2 | 2025-05-30**
   - ID: 120234898385570390
   - Optimization Goal: LEAD (padrão)
   - Spend: R$ 445.73

#### 📊 CONTROLE (7 campanhas)

1. **DEVLF | CAP | FRIO | FASE 01 | ABERTO ADV+ | PG2 | SCORE | 2025-04-15**
   - ID: 120220370119870390
   - Spend: R$ 1,220.32

2. **DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2 | 2025-04-13**
   - ID: 120224064762630390
   - Spend: R$ 1,982.06

3. **DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2 | 2025-05-13**
   - ID: 120224064761980390
   - Spend: R$ 1,989.91

4. **DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2 | 2025-05-13**
   - ID: 120224064762010390
   - Spend: R$ 444.81

5. **DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2 | 2025-05-13**
   - ID: 120224064762600390
   - Spend: R$ 1,983.28

6. **DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2 | 2025-07-08**
   - ID: 120228073033890390
   - Spend: R$ 427.98

7. **DEVLF | CAP | FRIO | FASE 04 | ADV | FAIXA A | PG2 | 2025-08-13**
   - ID: 120230454190910390
   - Spend: R$ 251.35

### 5.2 Conta Teste/Gestor IA (act_786790755803474)

**⚠️ IMPORTANTE:** Estas 4 campanhas devem ser analisadas **separadamente** da conta principal.

#### 📊 CONTROLE - Conta 7867 (4 campanhas)

1. **DEVLF | CAP | FRIO | FASE 04 | ADV | FAIXA A | S/ ABERTO | PG2 | 2025-10-14**
   - ID: 120232220702050534
   - Spend: R$ 504.32

2. **DEVLF | CAP | FRIO | FASE 04 | ADV | FAIXA A | S/ ABERTO | PG2 | 2025-10-22**
   - ID: 120232666823120534
   - Spend: R$ 522.68

3. **DEVLF | CAP | FRIO | FASE 04 | ADV | FAIXA A | S/ ABERTO | PG2 | 2025-10-22**
   - ID: 120232666823150534
   - Spend: R$ 504.59

4. **DEVLF | CAP | FRIO | FASE 04 | ADV | ML | S/ ABERTO | PG2 | 2025-11-11**
   - ID: 120234062599950534
   - Spend: R$ 2,107.45

---

## 6. ANÁLISE COMPLETA: MATCHED PAIRS

### 6.1 Anúncios nas Campanhas "Eventos ML" (27/11) COM SPEND

**Período analisado:** 27/11 a 01/12
**Total:** 9 ad_codes únicos com spend > 0

**Distribuição por campanha:**
- Campanha 120236428684090390: AD0022, AD0027, AD0043
- Campanha 120236428684840390: AD0004, AD0017, AD0027
- Campanha 120236428684850390: AD0013, AD0014, AD0017, AD0018, AD0022, AD0033

**Lista completa:**
```
AD0004, AD0013, AD0014, AD0017, AD0018, AD0022, AD0027, AD0033, AD0043
```

### 6.2 Matched Pairs ✅ VALIDADO VIA META API

**Total:** 8 matched pairs (aparecem em AMBOS: ML e Controle com spend > 0)

**Lista:**
```
AD0004, AD0013, AD0014, AD0017, AD0018, AD0022, AD0027, AD0033
```

**Status:** ✅ Sistema funcionando corretamente - identificou todos os 8 matched pairs

### 6.3 Anúncios Exclusivos

#### Exclusivo ML (1 anúncio)
**AD0043** - Aparece apenas em campanhas ML, não em Controle
- Campanha: 120236428684090390

#### Exclusivos Controle (exemplos verificados)
**AD0046** - Aparece apenas em campanhas Controle
- Campanha: 120220370119870390
- Spend: R$ 126.88

**AD0065** - Aparece apenas em campanhas Controle
- Campanha: 120220370119870390
- Spend: R$ 129.21

### 6.4 Conclusão da Investigação

**Status:** ✅ INVESTIGAÇÃO CONCLUÍDA

**Resultado:** O sistema está funcionando **PERFEITAMENTE**:
1. Identificou corretamente os 8 matched pairs
2. Não incluiu AD0043 (exclusivo ML) como matched pair
3. Não incluiu AD0046 e AD0065 (exclusivos Controle) como matched pairs

**Metodologia:** Validação manual via Meta Ads API comparando anúncios com spend > 0 no período 27/11-01/12 em campanhas ML vs Controle

---

## 7. SISTEMA DE COMPARAÇÃO: EVENTO ML

### 7.1 Conceito: "Maçãs com Maçãs"

Uma comparação justa exige condições estruturalmente **idênticas ou similares**. Implementamos **dois níveis** de comparação para campanhas com **Eventos ML** (LeadQualifiedHighQuality):

---

### 7.2 NÍVEL 1: Evento ML (adsets iguais)

**Objetivo:** Validação rigorosa do impacto ML em condições **perfeitamente controladas**

#### Campanhas Comparadas

**Eventos ML - ADV (2 campanhas):**
```yaml
120236428684840390:
  Nome: "DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2"
  Criação: 2025-11-27
  Budget: CBO R$ 300/dia
  Evento: LeadQualifiedHighQuality (CAPI)
  Adsets: 4 adsets ADV
  Spend_Período: R$ 459.45

120236428684850390:
  Nome: "DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2"
  Criação: 2025-11-27
  Budget: CBO R$ 300/dia
  Evento: LeadQualifiedHighQuality (CAPI)
  Adsets: 4 adsets ADV
  Spend_Período: R$ 1,441.66
```

**Controle - ADV (2 campanhas):**
```yaml
120224064762630390:
  Nome: "DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2"
  Budget: CBO R$ 390/dia (30% maior - ACEITÁVEL)
  Evento: Sem ML (OFFSITE_CONVERSIONS padrão)
  Adsets: 4 adsets ADV (mesma estrutura)
  Spend_Período: R$ 1,982.06

120224064761980390:
  Nome: "DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2"
  Budget: CBO R$ 390/dia (30% maior - ACEITÁVEL)
  Evento: Sem ML (OFFSITE_CONVERSIONS padrão)
  Adsets: 4 adsets ADV (mesma estrutura)
  Spend_Período: R$ 1,989.91
```

#### Adsets Idênticos

Todas as 4 campanhas têm **exatamente os mesmos adsets**:

```yaml
Adset_1:
  Nome: "ADV | Linguagem de programação"
  Targeting: Interesse específico
  Budget: R$ 0 (CBO distribui)
  Matched_Ads: AD0013, AD0014, AD0017, AD0018, AD0033

Adset_2:
  Nome: "ADV | Lookalike 1% Cadastrados - DEV 2.0 + Interesse Ciência da Computação"
  Targeting: Lookalike 1% + interesse
  Budget: R$ 0 (CBO distribui)
  Matched_Ads: AD0014, AD0017, AD0022, AD0033

Adset_3:
  Nome: "ADV | Lookalike 2% Cadastrados - DEV 2.0 + Interesses"
  Targeting: Lookalike 2% + interesses
  Budget: R$ 0 (CBO distribui)
  Matched_Ads: AD0013, AD0014, AD0018, AD0033

Adset_4:
  Nome: "ADV | Lookalike 2% Alunos + Interesse Linguagem de Programação"
  Targeting: Lookalike 2% alunos + interesse
  Budget: R$ 0 (CBO distribui)
  Matched_Ads: AD0018, AD0022, AD0033 (apenas em Controle)
```

#### Matched Ads Super Justos

**Total:** 6 anúncios que aparecem nos **mesmos adsets** em ambos lados

```yaml
Códigos: [AD0013, AD0014, AD0017, AD0018, AD0022, AD0033]

Distribuição:
  - AD0013: 2 adsets (Linguagem, Lookalike 2% Cadastrados)
  - AD0014: 3 adsets (Linguagem, Lookalike 1%, Lookalike 2% Cadastrados)
  - AD0017: 2 adsets (Linguagem, Lookalike 1%)
  - AD0018: 3 adsets (Linguagem, Lookalike 1%, Lookalike 2% Cadastrados)
  - AD0022: 1 adset  (Lookalike 1%)
  - AD0033: 3 adsets (Linguagem, Lookalike 1%, Lookalike 2% Cadastrados)
```

#### Critérios de Validade

✅ **Estrutura Idêntica:**
- Mesmos 4 adsets
- Mesmo targeting em cada adset
- Mesma configuração CBO

✅ **Budget Comparável:**
- ML: R$ 300/dia por campanha
- Controle: R$ 390/dia por campanha
- Diferença: 30% (dentro da tolerância)

✅ **Optimization Goal:**
- ML: OFFSITE_CONVERSIONS (otimizado por LeadQualifiedHighQuality via CAPI)
- Controle: OFFSITE_CONVERSIONS (sem eventos customizados)

✅ **Período Idêntico:**
- 27/11 a 01/12 (5 dias)

#### Exclusões

❌ **AD0004 e AD0027:** Aparecem em ML mas não nos mesmos adsets em Controle
❌ **Campanha ABERTO ML (120236428684090390):** Estrutura incomparável com Controle ABERTO

---

### 7.3 NÍVEL 2: Evento ML (todos)

**Objetivo:** Análise exploratória de todas as campanhas Evento ML, independente de estrutura

#### Campanhas Comparadas

**Eventos ML - TODAS (3 campanhas):**
```yaml
120236428684090390:  # ABERTO
  Budget: CBO R$ 550/dia
  Evento: LeadQualifiedHighQuality
  Spend: R$ 2,344.43
  Matched_Ads: AD0022, AD0027, AD0043

120236428684840390:  # ADV
  Budget: CBO R$ 300/dia
  Evento: LeadQualifiedHighQuality
  Spend: R$ 459.45
  Matched_Ads: AD0004, AD0017, AD0027

120236428684850390:  # ADV
  Budget: CBO R$ 300/dia
  Evento: LeadQualifiedHighQuality
  Spend: R$ 1,441.66
  Matched_Ads: AD0013, AD0014, AD0017, AD0018, AD0022, AD0033
```

**Controle - TODAS com spend (7 campanhas):**
```yaml
120220370119870390:  # ABERTO (ABO multi-adsets)
120224064762630390:  # ADV
120224064761980390:  # ADV
120224064762010390:  # ADV
120224064762600390:  # ADV
120228073033890390:  # ADV
120230454190910390:  # ADV
```

#### Matched Ads Gerais

**Total:** 8 anúncios (todos os matched pairs)

```yaml
Códigos: [AD0004, AD0013, AD0014, AD0017, AD0018, AD0022, AD0027, AD0033]

Origem_ML:
  - ABERTO: AD0022, AD0027, AD0043
  - ADV: AD0004, AD0013, AD0014, AD0017, AD0018, AD0022, AD0027, AD0033

Origem_Controle:
  - Múltiplas campanhas e estruturas
```

#### Critérios de Validade

✅ **Matched Ads:**
- Anúncios aparecem em AMBOS: ML e Controle
- Com spend > 0 no período

⚠️ **Estrutura Variada:**
- Inclui CBO e ABO
- Diferentes targets
- Diferentes budgets

⚠️ **Comparação Menos Rigorosa:**
- Não controla por estrutura de adset
- Não controla por budget exato
- Útil para visão geral, não para validação rigorosa

---

### 7.4 Resumo Comparativo

| Aspecto | Adsets Iguais | Todos |
|---------|---------------|-------|
| **Campanhas ML** | 2 (ADV) | 3 (todas) |
| **Campanhas Controle** | 2 (ADV) | 7 (todas) |
| **Matched Ads** | 6 | 8 |
| **Estrutura** | Idêntica | Variada |
| **Budget** | ±30% | Variado |
| **Adsets** | Mesmos | Podem ser iguais ou diferentes |
| **Uso** | Validação rigorosa ML | Análise exploratória |
| **Confiança** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

### 7.5 Implementação no Código

#### Evento ML (adsets iguais)

```python
# Campanhas específicas
ml_campaigns = ['120236428684840390', '120236428684850390']
control_campaigns = ['120224064762630390', '120224064761980390']

# Matched ads
adsets_iguais_ads = ['AD0013', 'AD0014', 'AD0017', 'AD0018', 'AD0022', 'AD0033']

# Filtro: mesmo adset em ambos
filter_by_adset_match = True
```

#### Evento ML (todos)

```python
# Todas as campanhas Eventos ML
ml_campaigns = ['120236428684090390', '120236428684840390', '120236428684850390']

# Todas as campanhas Controle com spend
control_campaigns = [
    '120220370119870390', '120224064762630390', '120224064761980390',
    '120224064762010390', '120224064762600390', '120228073033890390',
    '120230454190910390'
]

# Todos matched ads
todos_ads = ['AD0004', 'AD0013', 'AD0014', 'AD0017', 'AD0018', 'AD0022', 'AD0027', 'AD0033']

# Sem filtro de adset
filter_by_adset_match = False
```

---

## 8. COMANDOS DE VALIDAÇÃO

### 8.1 Evento ML (adsets iguais)

```bash
python src/validation/validate_ml_performance.py \
  --start-date 2025-11-27 \
  --end-date 2025-12-01 \
  --sales-start-date 2025-12-02 \
  --sales-end-date 2025-12-15 \
  --account-id act_188005769808959 \
  --comparison-level adsets_iguais
```

**Resultado:**
- 2 campanhas ML ADV vs 2 campanhas Controle ADV
- 6 matched ads (mesmos adsets)
- Comparação rigorosa - validação do impacto ML

**Abas Excel:**
- Agregação Matched Pairs
- Detalhamento Anúncios
- Resumo Todos Anúncios

### 8.2 Evento ML (todos)

```bash
python src/validation/validate_ml_performance.py \
  --start-date 2025-11-27 \
  --end-date 2025-12-01 \
  --sales-start-date 2025-12-02 \
  --sales-end-date 2025-12-15 \
  --account-id act_188005769808959 \
  --comparison-level todos
```

**Resultado:**
- 3 campanhas ML (todas) vs 7 campanhas Controle
- 8 matched ads
- Comparação exploratória - visão geral

**Abas Excel:**
- Agregação Matched Pairs
- Detalhamento Anúncios
- Resumo Todos Anúncios

### 8.3 Ambos Níveis (Padrão - Recomendado)

```bash
python src/validation/validate_ml_performance.py \
  --start-date 2025-11-27 \
  --end-date 2025-12-01 \
  --sales-start-date 2025-12-02 \
  --sales-end-date 2025-12-15 \
  --account-id act_188005769808959 \
  --comparison-level both
```

**Resultado:**
- Gera AMBAS comparações em abas separadas do Excel
- Melhor para análise completa

**Abas Excel:**
- 📊 Adsets Iguais - Agregação
- 📋 Adsets Iguais - Detalhes
- 📝 Adsets Iguais - Resumo
- 📊 Todos - Agregação
- 📋 Todos - Detalhes
- 📝 Todos - Resumo

---

## 9. ARQUIVOS RELEVANTES

```yaml
Classificação:
  - src/validation/campaign_classifier.py
  - configs/validation_config.yaml

Matching:
  - src/validation/matching.py
  - src/validation/fair_campaign_comparison.py

Validação:
  - src/validation/validate_ml_performance.py

CAPI:
  - api/capi_integration.py
  - api/business_config.py

Métricas:
  - src/validation/metrics_calculator.py
```

---

## 8. COMANDO DE VALIDAÇÃO

```bash
python src/validation/validate_ml_performance.py \
  --start-date 2025-11-25 \
  --end-date 2025-12-01 \
  --sales-start-date 2025-12-01 \
  --sales-end-date 2025-12-15 \
  --account-id act_188005769808959 act_786790755803474
```

---

## 9. INVESTIGAÇÕES E PROBLEMAS CONHECIDOS

### 9.1 Discrepância de Vendas Entre Abas (2025-12-16)

**Problema Identificado:**
Inconsistência no número de vendas reportadas entre diferentes abas do relatório Excel:

| Aba | Vendas Reportadas | Status |
|-----|-------------------|--------|
| Detalhes das Conversões | 22 trackeadas | ✅ Correto (dado bruto) |
| Performance Geral | 16 identificadas | ⚠️ Após deduplicação |
| Performance por Campanha | 4 vendas | ❌ Faltam 7 vendas |

**Fluxo de Perda de Vendas:**
```
22 vendas trackeadas (Detalhes das Conversões)
  ↓ [-6] Deduplicação de vendas duplicadas
16 vendas no matched_df
  ↓ [-7] Perdidas durante agregação por campanha (causa: investigando)
9 vendas em campaign_stats
  ↓ [-5] Campanhas removidas (spend=0 E leads=0)
4 vendas na aba Performance por Campanha
```

**Detalhamento das Perdas:**

1. **6 vendas duplicadas removidas** (22 → 16)
   - Deduplicação intencional de vendas que aparecem múltiplas vezes
   - ✅ Comportamento correto

2. **7 vendas perdidas na agregação** (16 → 9)
   - ❌ **PROBLEMA PRINCIPAL:** Vendas presentes em `matched_df` mas não agregadas em `campaign_stats`
   - **Causa em investigação:** Possíveis razões:
     - Vendas sem `campaign_name` válido
     - Erro no `groupby` por campanha
     - Vendas em campanhas que não passam pelo filtro inicial

3. **5 vendas removidas com campanhas inativas** (9 → 4)
   - Campanhas com `spend=0` E `leads=0` são removidas
   - Vendas afetadas:
     - 3 vendas: "DEVLF | CAP | FRIO | FASE 04 | ADV | ML | S/ ABERTO"
     - 1 venda: "DEVLF | CAP | FRIO | FASE 04 | ADV | FAIXA A | S/ ABERTO"
     - 1 venda: "DEVLF | CAP | FRIO | FASE 04 | ADV | FAIXA A | S/ ABERTO"
   - ⚠️ **Atenção:** Essas campanhas têm vendas mas não aparecem na Meta API

**Diagnóstico Atual:**
- ✅ Taxa de resposta corrigida (eventos LQHQ não mais somados ao denominador)
- ✅ Campanhas EXCLUIR removidas da aba Performance por Campanha
- ✅ Período de vendas corrigido (01/12 a 15/12)
- ❌ **Pendente:** Investigar perda de 7 vendas na agregação

**Próximos Passos:**
1. Adicionar logs para identificar quais vendas estão sendo perdidas no `groupby`
2. Verificar se há vendas em `matched_df` sem `campaign_name` válido
3. Corrigir lógica de agregação para preservar todas as vendas válidas

---

**Última atualização:** 2025-12-16

**Histórico de atualizações:**
- **2025-12-16:** Investigação de discrepância de vendas entre abas documentada
- Taxa de resposta corrigida (eventos LQHQ separados do denominador)
- Campanhas EXCLUIR filtradas da aba Performance por Campanha
- Período de vendas corrigido para 01/12 a 15/12
- Validação completa via Meta API: 8 matched pairs confirmados
- Sistema de comparação em 2 níveis implementado: "Evento ML (adsets iguais)" e "Evento ML (todos)"
- Parâmetro `--comparison-level {adsets_iguais|todos|both}` adicionado ao CLI
- Nomenclatura uniformizada para refletir foco em campanhas com Eventos ML customizados