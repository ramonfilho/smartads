# Métricas Personalizáveis do Sistema

Este documento descreve as métricas configuráveis do sistema de análise e recomendação de budget.

---

## 1. ROAS Mínimo

**O que é:**
ROAS (Return on Ad Spend) mínimo aceitável. Define o retorno esperado para cada R$ 1,00 investido em anúncios. Exemplo: ROAS 2.0x = para cada R$ 1,00 gasto, espera-se R$ 2,00 de retorno.

**Como alterar:**
Arquivo: `V2/api/meta_config.py`
Variável: `BUSINESS_CONFIG['min_roas']` (linha 17)

**Valor atual:** `2.0` (2.0x)

---

## 2. Thresholds de Confiança Estatística

**O que é:**
Número mínimo de leads necessários para atingir cada nível de confiança nas recomendações. Ajustado automaticamente pelo período analisado (1 dia, 3 dias, 7 dias, etc.).

**Como alterar:**
Arquivo: `V2/api/economic_metrics.py`
Função: `calculate_confidence_level()` (linhas 142-144)

**Valores base (por dia):**
- Insuficiente: `< 3 leads/dia`
- Baixa: `3-9 leads/dia`
- Média: `10-19 leads/dia`
- Alta: `≥ 20 leads/dia`

**Exemplos por período:**

| Período | Insuficiente | Baixa | Média | Alta |
|---------|--------------|-------|-------|------|
| 1 dia | < 3 | 3-9 | 10-19 | ≥ 20 |
| 3 dias | < 9 | 9-29 | 30-59 | ≥ 60 |
| 7 dias | < 21 | 21-69 | 70-139 | ≥ 140 |

---

## 3. Thresholds de Margem (Ações)

**O que é:**
Percentual de margem de manobra que determina a ação recomendada. A variação sempre usa `margem × fator_confiança`.

**Como alterar:**
Arquivo: `V2/api/economic_metrics.py`
Função: `calculate_budget_variation()` (linhas 173-204)

**Lógica atual:**
- Margem > 0%: Aumentar (margem × fator confiança)
  - Ex: 30% margem + 15 leads (média 0.5) = Aumentar 15%
  - Ex: 80% margem + 25 leads (alta 0.8) = Aumentar 64%
- Margem **< 0%**: Reduzir conforme necessário (CPL Max / CPL Atual) × ajuste por confiança

---

## 4. Fatores de Confiança

**O que é:**
Percentual da margem disponível que será usado na recomendação, baseado no volume de leads.

**Como alterar:**
Arquivo: `V2/api/economic_metrics.py`
Função: `calculate_budget_variation()` (variável `confidence_factors`, linhas 167-171)

**Valores atuais:**
- Baixa confiança (3-9 leads): `0.3` (usa 30% da margem)
- Média confiança (10-19 leads): `0.5` (usa 50% da margem)
- Alta confiança (≥20 leads): `0.8` (usa 80% da margem)

---

## 5. Threshold de Gasto Sem Leads

**O que é:**
Valor mínimo de gasto que, com 0 leads, indica que o item está comprovadamente ruim e deve ser removido/pausado imediatamente.

**Como alterar:**
Arquivo: `V2/api/economic_metrics.py`
Função: `determine_action()` (variável `SPEND_THRESHOLD`, linha 234)

**Valor atual:** `R$ 50,00`

**Lógica:**
- Se `leads = 0` AND `spend > R$ 50`:
  - Anúncios/Criativos: **"Remover"**
  - Campanhas/Adsets: **"Reduzir 100%"** (pausar)

---

## 6. Cores da Coluna Ação

**O que é:**
Lógica de cores aplicada na coluna Ação do Google Sheets, baseada no percentual de variação recomendado.

**Como alterar:**
Arquivo: `V2/api/apps-script-code.js`
Linhas 663-706

**Valores atuais:**
- **Verde**: Aumentar > 30%
- **Amarelo**: Aumentar 1-30%
- **Vermelho**: Reduzir (qualquer %) ou Remover
- **Cinza**: Manter, Aguardar dados, ABO, CBO

---

## Observações

1. **ROAS Mínimo** pode ser alterado via API (parâmetro `min_roas` no request) sem modificar o código
2. **Thresholds de Confiança** baseados em requisitos da Learning Phase do Meta (reduzido de 50 para 10 conversões em 2024)
3. **Fatores de Confiança** aplicam mais cautela quando há poucos dados (3-9 leads) e mais agressividade com dados robustos (≥20 leads)
4. Todas as ações consideram o volume de leads: < 3 leads sempre retorna "Aguardar dados"

---

**Última atualização:** 2025-10-20
