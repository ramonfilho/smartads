# CUSTOS UTM - PROBLEMAS E SOLUÇÕES

## PROBLEMAS

### 1. Bare IDs sem custo (867 leads - 8.6% do total)
Valores com apenas ID/nome genérico aparecem sem matching de custo:
- Campaign: `120220370119870390`, `devlf`
- Medium: `dgen`, `paid`
- Term: `fb`, `ig`, `120230637730870390`, IDs compostos `22527413657--179264823996--750940275529`
- Content: `750940275538`, `120230637730630390`

**Causa**: Matching por nome falha quando há apenas ID

### 2. Campaign ID duplicado com custo replicado
Campaign `120220370119870390` aparece 4x (560 leads):
- `120220370119870390` → 4 leads, R$ 0
- `...ABERTO ADV+...120220370119870390` → 300 leads, R$ 5674.52
- `...ABERTO ADV+ | SCORE...120220370119870390` → 141 leads, R$ 5674.52
- `...ABERTO...120220370119870390` → 119 leads, R$ 5674.52

**Causa**: Custo total da campaign sendo atribuído a cada adset

### 3. Código atual
`meta_integration.py`:
- `get_costs_by_utm()` (L77-114): Busca nos 3 níveis, agrupa por NOME
- `enrich_utm_analysis_with_costs()` (L179-241): Matching por nome/fuzzy
- `match_campaign_name()` (L144-176): Remove Campaign ID do final, fuzzy 85%

`app.py` (L829-834): Chamada do enrichment

## SOLUÇÕES

### A) Campaign ID + Adset Name (matching)
1. Extrair Campaign ID do UTM
2. Buscar adsets daquela campaign via API
3. Fazer matching do nome do adset
4. Pegar custo exato do adset

**Vantagem**: Sem mudança em UTMs
**Desvantagem**: Depende de matching de nome

### B) Adset ID direto no UTM
Incluir Adset ID no UTM → busca direta via API

**Vantagem**: Match exato, confiável
**Desvantagem**: Requer mudança na configuração de UTMs

### C) Hierarquia combinada (ID + fallback)
1. Tentar extrair Adset ID/Ad ID se disponível
2. Fallback para Campaign ID + nome se não houver

## INVESTIGAR

- [ ] Estrutura dos IDs compostos em Term (`XX--YY--ZZ`)
- [ ] Se UTMs já contêm Adset ID/Ad ID em algum campo
- [ ] Qual campo do Meta retorna Adset ID nos insights
- [ ] Viabilidade de adicionar Adset ID nos UTMs futuros
- [ ] Nível de análise desejado: Campaign, Adset ou Ad?

## MAPEAMENTO UTM → META

```
Campaign: [Nome] | [Data] | [Campaign ID]
Medium:   [Targeting] | [Audiência]  → ADSET
Term:     fb/ig ou IDs compostos
Content:  [Código Criativo]          → AD
```

**Hierarquia Meta**:
```
Campaign (120220370119870390) - Custo: soma dos adsets
  ├─ Adset (ABERTO ADV+) - Custo próprio
  │   ├─ Ad (Criativo 1)
  │   └─ Ad (Criativo 2)
  └─ Adset (ABERTO) - Custo próprio
```
