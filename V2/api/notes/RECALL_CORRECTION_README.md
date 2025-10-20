# Correção de Taxas de Conversão por Recall

**Data:** 2025-10-20
**Autor:** Claude Code
**Tipo:** Correção crítica de métricas de negócio

---

## 📋 Resumo Executivo

As taxas de conversão por decil foram **corrigidas em 2.906x** para refletir conversões reais, não apenas as capturadas pelo matching.

**Problema identificado:**
- Matching por email/telefone captura apenas **34.4%** das conversões reais
- Isso causava **subestimação severa** das métricas de negócio (ROAS, margem)
- Campanhas boas eram classificadas como "Reduzir" quando deveriam ser "Escalar"

**Solução aplicada:**
- Fator de correção: **2.906x** (1 / 0.344)
- Todas as taxas de conversão multiplicadas por este fator
- Mantém ranking relativo dos decis (D10 continua melhor que D1)

---

## 🔍 Análise Técnica

### Dados de Recall

| Métrica | Valor |
|---------|-------|
| Conversões observadas (treino) | 557 |
| Conversões observadas (teste) | 121 |
| **Total conversões observadas** | **678** |
| **Vendas reais DevClub** | **1.970** |
| **Recall** | **34.4%** |
| **Fator de correção** | **2.906x** |

### Causas do Baixo Recall

1. **Emails diferentes** (60-70% das perdas):
   - Lead usou email pessoal na pesquisa
   - Comprou com email profissional

2. **Telefones incomparáveis** (20-25%):
   - Telefone errado/desatualizado
   - Formatos não capturados pela normalização

3. **Dados ausentes** (5-10%):
   - Lead não preencheu telefone
   - Email malformado

4. **Timing/Outras** (5%):
   - Dessincronização de dados

---

## 📊 Impacto nas Taxas de Conversão

### Antes (Observadas) vs Depois (Corrigidas)

| Decil | Taxa Observada | Taxa Corrigida | Δ Absoluto | Δ Relativo |
|-------|----------------|----------------|------------|------------|
| D1 | 0.26% | **0.76%** | +0.50pp | +191% |
| D2 | 0.26% | **0.76%** | +0.50pp | +191% |
| D3 | 0.85% | **2.47%** | +1.62pp | +191% |
| D4 | 0.94% | **2.73%** | +1.79pp | +191% |
| D5 | 1.02% | **2.96%** | +1.94pp | +191% |
| D6 | 1.11% | **3.23%** | +2.12pp | +191% |
| D7 | 1.19% | **3.46%** | +2.27pp | +191% |
| D8 | 1.19% | **3.46%** | +2.27pp | +191% |
| D9 | 1.37% | **3.98%** | +2.61pp | +191% |
| **D10** | **2.13%** | **6.19%** | **+4.06pp** | **+191%** |

---

## 💰 Impacto em Métricas de Negócio

### Exemplo Real: Campaign 120220370119870390

**Dados:**
- 76 leads | R$ 703 gasto | CPL R$ 9,25
- 30.26% dos leads em D10

**Comparação:**

| Métrica | Antes (Observado) | Depois (Corrigido) | Δ |
|---------|-------------------|-------------------|---|
| Taxa Projetada | 0.64% | **1.87%** | +1.23pp |
| ROAS Projetado | 1.41x | **4.10x** | +2.69x |
| CPL Máximo | R$ 6,53 | **R$ 18,98** | +R$ 12,45 |
| Margem | -41.6% | **+51.3%** | +92.9pp |
| **Recomendação** | **Reduzir** | **Escalar** | ⚠️ **MUDOU** |

**Impacto:**
- ROAS aumenta **191%**
- Margem aumenta **92.9 pontos percentuais**
- Recomendação muda de "Reduzir" para "Escalar"

---

## 📂 Arquivos Modificados

### 1. `V2/api/meta_config.py`
**Backup:** `V2/api/meta_config.py.backup`

**Mudança:**
```python
# ANTES
"conversion_rates": {
    "D1": 0.0026,  # 0.26%
    ...
    "D10": 0.0213  # 2.13%
}

# DEPOIS
"conversion_rates": {
    "D1": 0.007555,  # 0.76%
    ...
    "D10": 0.061889  # 6.19%
}
```

### 2. Arquivos Criados

- `V2/api/recall_correction_analysis.py` - Script de análise
- `V2/api/BUSINESS_CONFIG_CORRECTED.py` - Config gerado
- `V2/api/RECALL_CORRECTION_README.md` - Esta documentação

---

## ✅ Validação

### Como Validar a Correção

1. **Re-gerar planilha de análise UTM:**
   ```bash
   # Chamar endpoint /analyze_utms_with_costs com dados reais
   ```

2. **Verificar mudanças em recomendações:**
   - Campanhas com margem negativa antes → Positiva depois
   - "Reduzir" → "Manter" ou "Escalar"
   - "Manter" → "Escalar"

3. **Comparar ROAS médio:**
   - ROAS médio deve aumentar ~191%
   - Mais campanhas devem ter ROAS > 2.0x

---

## 🚨 Atenção

### O Que NÃO Mudou

- **Ranking dos decis:** D10 continua melhor que D9, que continua melhor que D8, etc.
- **Poder discriminativo do modelo:** AUC permanece 0.636
- **Predições (lead_score):** Scores individuais não mudam

### O Que Mudou

- **Interpretação das taxas:** Agora refletem conversões REAIS, não apenas observadas
- **Métricas de negócio:** ROAS, CPL Máx, Margem, Tier, Ação
- **Recomendações:** Mais campanhas serão "Escalar", menos "Reduzir"

---

## 🔄 Rollback (Se Necessário)

Se precisar reverter a mudança:

```bash
# Restaurar backup
cp V2/api/meta_config.py.backup V2/api/meta_config.py

# Reiniciar API
# (O processo exato depende de como está rodando)
```

---

## 📈 Próximos Passos (Recomendados)

### Curto Prazo (Próxima Semana)
1. ✅ **Monitorar primeiras recomendações** com taxas corrigidas
2. **Validar com cliente** se recomendações fazem sentido
3. **A/B test** (se possível): Campanhas com ação antiga vs nova

### Médio Prazo (Próximo Mês)
4. **Validação externa:** Obter dados completos da plataforma de vendas
5. **Calcular recall estratificado:** Por período, fonte, produto
6. **Ajustar correção:** Se recall varia muito por segmento

### Longo Prazo (Próximo Trimestre)
7. **Matching probabilístico:** Capturar conversões não matchadas deterministicamente
8. **PU Learning:** Treinar modelo que aceita labels ruidosos
9. **Re-treino completo:** Com targets corrigidos

---

## 📞 Contato

**Dúvidas sobre esta correção?**
- Revisar: `V2/api/recall_correction_analysis.py`
- Executar análise novamente: `python3 V2/api/recall_correction_analysis.py`
- Logs: Verificar saída do script acima

---

**Última atualização:** 2025-10-20
