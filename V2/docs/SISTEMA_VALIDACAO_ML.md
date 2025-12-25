# Sistema de Validação ML - Documentação Técnica

**Atualizado:** 2025-12-18

---

## 1. CLASSIFICAÇÃO DE CAMPANHAS

### 1.1 "Eventos ML"
- **Critério:** Tem "MACHINE LEARNING" ou "| ML |" no nome **E** usa eventos customizados
- **Optimization Goal:** `LeadQualified` ou `LeadQualifiedHighQuality`
- **Importância:** ⭐⭐⭐ **FOCO PRINCIPAL DA VALIDAÇÃO**

### 1.2 "Otimização ML"
- **Critério:** Tem "MACHINE LEARNING" no nome **MAS** usa eventos padrão
- **Optimization Goal:** `LEAD`, `OFFSITE_CONVERSIONS`, etc.
- **Importância:** ⭐ Teste inicial, **NÃO INCLUÍDO NA VALIDAÇÃO**

### 1.3 "Controle"
- **Critério:** NÃO tem "MACHINE LEARNING" ou "| ML |" no nome
- **Exemplos:** "ESCALA SCORE", "FAIXA A", etc.
- **Importância:** ⭐⭐ Base de comparação para validar performance ML

---

## 2. CONTAS E CONFIGURAÇÕES

### 2.1 Contas Meta Ads
```yaml
Conta Principal (final 1880): act_188005769808959  # Ads - Rodolfo Mori
Conta Teste (final 7867):     act_786790755803474  # Ads - Gestor de IA
```

### 2.2 Eventos Customizados CAPI
```yaml
LeadQualified:            # Decisão D1-10 com valor de conversão
LeadQualifiedHighQuality: # Decisão D8-D10 no começo, D9-D10 depois
```

### 2.3 Janela de Matching
```yaml
max_match_days: 30      # Janela máxima leads → vendas
product_value: 2000.00  # Valor do produto (R$)
```

### 2.4 Períodos de Validação
```yaml
Estrutura:
  - Captação: 18/11/2025 a 01/12/2025 (14 dias)
  - Vendas:   02/12/2025 a 22/12/2025 (até finalização)
```

---

## 3. ESTRUTURA DE COMPARAÇÃO FAIR CONTROL

### 3.1 Princípio: "Maçãs com Maçãs"

A comparação é feita **NO NÍVEL DE ADSETS E ADS**, não campanhas.

**Por quê?**
- ✅ Compara **exatamente o mesmo público/targeting** (adsets matched)
- ✅ Compara **exatamente o mesmo criativo** (ads matched)
- ✅ Aproveita **TODAS as 4 campanhas ML** (sem perder dados)
- ✅ Mais granular e defensável estatisticamente

---

### 3.2 CAMPANHAS EVENTOS ML (4 campanhas)

#### Conta 1880 (act_188005769808959)

**1. ML #090 - ABERTO**
```yaml
ID: 120236428684090000
Budget: R$ 550/dia
Evento: LeadQualifiedHighQuality
Gasto: R$ 2,344.43
Conversões LQHQ: 183
Adsets: 3 (ABERTO)
AD codes: AD0022, AD0027, AD0043
```

**2. ML #840 - ADV**
```yaml
ID: 120236428684840000
Budget: R$ 300/dia
Evento: LeadQualifiedHighQuality
Gasto: R$ 459.45
Conversões LQHQ: 24
Adsets: 1 (ADV)
AD codes: AD0004, AD0017, AD0027
```

**3. ML #850 - ADV**
```yaml
ID: 120236428684850000
Budget: R$ 300/dia
Evento: LeadQualifiedHighQuality
Gasto: R$ 1,441.66
Conversões LQHQ: 39
Adsets: 3 (ADV)
AD codes: AD0013, AD0014, AD0017, AD0018, AD0022, AD0033
```

#### Conta 7867 (act_786790755803474)

**4. ML S/ ABERTO**
```yaml
ID: 120234062599950000
Budget: R$ 400/dia
Evento: LeadQualified
Gasto: R$ 5,459.02
Conversões LQ: 899
Adsets: 2 (ADV)
AD codes: AD0013, AD0014, AD0017, AD0027, AD0033
```

---

### 3.3 NÍVEL 1: COMPARAÇÃO POR ADSETS (5 adsets matched)

**Matched Adsets:** Conjuntos de anúncios que aparecem **EM PELO MENOS** uma campanha ML **E EM PELO MENOS** uma campanha Controle.

#### Adset 1: ABERTO | AD0022
```yaml
Aparece em:
  ML: ML #090
  Controle: ABERTO ADV+ 2025-04-15
```

#### Adset 2: ABERTO | AD0027
```yaml
Aparece em:
  ML: ML #090
  Controle: ABERTO ADV+ 2025-04-15
```

#### Adset 3: ADV | Linguagem de programação
```yaml
Aparece em:
  ML: ML #850, ML S/ ABERTO (7867)
  Controle: ESCALA SCORE 05-13, ESCALA SCORE 04-13, FAIXA A 10-14
```

#### Adset 4: ADV | Lookalike 1% Cadastrados - DEV 2.0 + Interesse Ciência da Computação
```yaml
Aparece em:
  ML: ML #850, ML S/ ABERTO (7867)
  Controle: ESCALA SCORE 05-13, ESCALA SCORE 04-13, FAIXA A 10-14, FAIXA A 10-22
```

#### Adset 5: ADV | Lookalike 2% Cadastrados - DEV 2.0 + Interesses
```yaml
Aparece em:
  ML: ML #840, ML #850
  Controle: ESCALA SCORE 05-13, ESCALA SCORE 04-13, ESCALA SCORE 07-08
```

---

### 3.4 NÍVEL 2: COMPARAÇÃO POR ADS (7 ads matched)

**Matched Ads:** Anúncios (AD codes) que aparecem **EM PELO MENOS** uma campanha ML **E EM PELO MENOS** uma campanha Controle.

```yaml
AD codes matched: AD0013, AD0014, AD0017, AD0018, AD0022, AD0027, AD0033

Exclusivos ML: AD0004, AD0043
Exclusivos Controle: Nenhum (todos os ads de controle também aparecem em ML)
```

---

### 3.5 CRITÉRIOS DE INCLUSÃO

#### Para Adsets:
```python
# Adset incluído na comparação se:
1. Nome do adset aparece em >= 1 campanha ML
2. Nome do adset aparece em >= 1 campanha Controle
3. Gasto do adset >= R$ 200 (em cada lado)
```

#### Para Ads:
```python
# Ad incluído na comparação se:
1. AD code aparece em >= 1 campanha ML
2. AD code aparece em >= 1 campanha Controle
3. Gasto do ad >= R$ 200 (em cada lado)
```

**Justificativa do filtro R$ 200:**
- Garante dados maduros (passou fase de aprendizado)
- Significância estatística mínima
- Elimina ruído de testes pequenos

---

## 4. MÉTRICAS DE NEGÓCIO

### 4.1 Métricas Calculadas

#### Por Adset/Ad:
```python
CPL = Gasto / Leads
CPA = Gasto / Vendas
ROAS = (Vendas × Valor_Produto) / Gasto
Margem = (Receita - Gasto) / Receita
Taxa_Conversão = Vendas / Leads

Onde:
  Receita = Vendas × R$ 2.000
```

#### Diferença ML vs Controle:
```python
Diff_% = ((ML - Controle) / Controle) × 100

Exemplo:
  CPL ML: R$ 50
  CPL Controle: R$ 100
  Diff: -50% (ML é 50% melhor)
```

---

## 5. ESTRUTURA DO RELATÓRIO EXCEL

### 5.1 Aba: Comparação Adsets
```
Colunas:
  - Nome do Adset
  - Gasto ML | Gasto Controle
  - Leads ML | Leads Controle
  - CPL ML | CPL Controle | Diff%
  - Vendas ML | Vendas Controle
  - CPA ML | CPA Controle | Diff%
  - ROAS ML | ROAS Controle | Diff%
  - Margem ML | Margem Controle | Diff%
  - Taxa Conversão ML | Taxa Conversão Controle | Diff%
```

### 5.2 Aba: Comparação Ads
```
Colunas:
  - AD Code
  - Nome do Anúncio
  - Gasto ML | Gasto Controle
  - Leads ML | Leads Controle
  - CPL ML | CPL Controle | Diff%
  - Vendas ML | Vendas Controle
  - CPA ML | CPA Controle | Diff%
  - ROAS ML | ROAS Controle | Diff%
  - Margem ML | Margem Controle | Diff%
  - Taxa Conversão ML | Taxa Conversão Controle | Diff%
```

### 5.3 Aba: Detalhes das Conversões
```
Colunas:
  - Lead ID
  - Campanha
  - Adset
  - AD Code
  - Data Lead
  - Data Venda (se houver)
  - Dias até Venda
  - Valor Venda
```

---

## 6. ARQUIVOS DO SISTEMA

### 6.1 Fonte de Dados
```yaml
Relatórios Excel da Meta (por conta):
  - meta_reports/Ads---[Conta]-Campanhas-[período].xlsx
  - meta_reports/Ads---[Conta]-Conjuntos-de-anúncios-[período].xlsx
  - meta_reports/Ads---[Conta]-Anúncios-[período].xlsx

Banco de Dados (leads e vendas):
  - Sistema de matching automático leads → vendas
```

### 6.2 Arquivos Python
```yaml
Comparação:
  - src/validation/adset_comparison.py  # Nova comparação por adsets
  - src/validation/ad_comparison.py     # Nova comparação por ads

Matching:
  - src/validation/matching.py           # Matching leads → vendas

Métricas:
  - src/validation/metrics_calculator.py # Cálculo ROAS, CPA, Margem

Validação Principal:
  - src/validation/validate_ml_performance.py # Script principal

CAPI:
  - api/capi_integration.py
  - api/business_config.py
```

---

## 7. COMANDO DE VALIDAÇÃO

```bash
python src/validation/validate_ml_performance.py \
  --start-date 2025-11-18 \
  --end-date 2025-12-01 \
  --sales-start-date 2025-12-02 \
  --sales-end-date 2025-12-22 \
  --account-id act_188005769808959 act_786790755803474 \
  --min-spend 200
```

**Parâmetros:**
- `--start-date`: Início período de captação de leads
- `--end-date`: Fim período de captação de leads
- `--sales-start-date`: Início janela de matching de vendas
- `--sales-end-date`: Fim janela de matching de vendas (22/12)
- `--account-id`: IDs das contas a analisar
- `--min-spend`: Gasto mínimo para incluir adset/ad (padrão: R$ 200)

---

## 8. VALIDAÇÃO VIA EXCEL

### 8.1 Descoberta: Ad Name como Proxy para Creative ID

**Validação realizada (2025-12-18):**
- Ad Names únicos identificam criativos únicos
- Nomenclatura consistente: `DEV-AD0033-vid-captação-V0-NATIVO`
- Excel produz resultados idênticos à Meta API
- ✅ **Recomendado para uso rotineiro**

### 8.2 Campos Obrigatórios dos Relatórios Excel

**Em Anúncios (Ads):**
- Nome do anúncio (proxy para creative)
- Nome da campanha
- Nome do conjunto de anúncios
- Valor usado (BRL)
- Resultados
- Indicador de resultados

**Em Conjuntos de Anúncios (Adsets):**
- Nome do conjunto de anúncios
- Nome da campanha
- Valor usado (BRL)
- Resultados
- Indicador de resultados

---

## 9. RESUMO DA VALIDAÇÃO

### 9.1 O que validamos

**Hipótese:**
> Campanhas com Eventos ML (LeadQualified/LQHQ) geram **melhores resultados de negócio** (ROAS, CPA, Margem) que campanhas Controle tradicionais.

**Método:**
- Comparação **no nível de adsets e ads** (não campanhas)
- 5 adsets matched
- 7 ads matched
- 4 campanhas ML vs múltiplas campanhas Controle
- Métricas normalizadas: CPL, CPA, ROAS, Margem, Taxa Conversão
- Filtro de gasto mínimo: R$ 200

**Período:**
- Captação: 18/11 a 01/12/2025
- Vendas: 02/12 a 22/12/2025

---

**Última atualização:** 2025-12-18

**Histórico de mudanças:**
- **2025-12-18:** REESTRUTURAÇÃO COMPLETA - Removida comparação por campanhas, foco total em Adsets e Ads matched. Métricas de negócio: ROAS, CPA, Margem. Filtro gasto mínimo R$ 200.
