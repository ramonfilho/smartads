# Sistema de Validação de Performance ML - Guia de Uso

## 📋 Visão Geral

Sistema completo para validar a performance do modelo de ML de lead scoring, comparando:
1. **Campanhas COM ML vs SEM ML** (taxa conversão, ROAS, margem)
2. **Performance por Decil D1-D10** (conversão real vs esperada, separando Guru vs Guru+TMB)

## 🏗️ Arquitetura

```
V2/
├── src/validation/
│   ├── data_loader.py          # Carrega leads CSV + vendas Excel
│   ├── campaign_classifier.py  # Classifica COM/SEM ML
│   ├── matching.py             # Match leads ↔ vendas
│   ├── metrics_calculator.py   # Calcula métricas (CPL, ROAS, margem)
│   ├── report_generator.py     # Gera Excel com 6 abas
│   └── visualization.py        # Gera 5 gráficos PNG
├── scripts/
│   └── validate_ml_performance.py  # CLI principal
├── configs/
│   └── validation_config.yaml      # Configurações globais
└── validation/                     # Dados (gitignore)
    ├── leads/                      # CSV do Google Sheets aqui
    ├── vendas/                     # Guru + TMB xlsx aqui
    └── resultados/                 # Outputs gerados aqui
```

## 📝 Preparação dos Dados

### 1. Leads (Google Sheets CSV)

Baixe o CSV do Google Sheets e salve em `validation/leads/leads_completo.csv`

**Colunas necessárias:**
- `Data` - Timestamp da captura
- `E-mail` - Email do lead
- `Campaign` - Nome da campanha UTM
- `lead_score` - Score do modelo (0-1)
- `Telefone` - Telefone (opcional, para matching)
- `Nome Completo` - Nome (opcional)

### 2. Vendas

Salve arquivos Excel em `validation/vendas/`:
- **Guru:** `guru_*.xlsx` ou `GURU*.xlsx`
- **TMB:** `tmb_*.xlsx` ou `TMB*.xlsx`

**Colunas Guru:**
- `email contato`, `valor venda`, `data aprovacao` ou `data pedido`

**Colunas TMB:**
- `Cliente Email`, `Ticket (R$)`, `Data Pedido`, `Status` (deve ser 'Efetivado')

### 3. Meta Access Token

Configure o token no arquivo `configs/validation_config.yaml`:

```yaml
meta_access_token: "EAAV..."  # Substitua pelo token real
```

## 🚀 Uso

### Opção 1: Usar Período Pré-Configurado

```bash
python scripts/validate_ml_performance.py \
  --periodo periodo_1 \
  --account-id act_123456789
```

Períodos disponíveis (em `validation_config.yaml`):
- `periodo_1`: Lançamento 11/11 (2025-11-11 a 2025-12-01)
- `periodo_2`: Lançamento 18/11 (2025-11-18 a 2025-12-08)
- `periodo_3`: Lançamento 25/11 (2025-11-25 a 2025-12-15)

### Opção 2: Usar Datas Customizadas

```bash
python scripts/validate_ml_performance.py \
  --start-date 2025-11-11 \
  --end-date 2025-12-01 \
  --account-id act_123456789
```

### Opção 3: Sobrescrever Parâmetros

```bash
python scripts/validate_ml_performance.py \
  --periodo periodo_1 \
  --account-id act_123456789 \
  --product-value 2500 \
  --max-match-days 45 \
  --leads-path custom/path/leads.csv \
  --vendas-path custom/path/vendas \
  --output-dir custom/path/resultados
```

### Parâmetros Disponíveis

| Parâmetro | Descrição | Obrigatório |
|-----------|-----------|-------------|
| `--periodo` | Período pré-configurado (periodo_1, periodo_2, periodo_3) | Sim* |
| `--start-date` | Data início (YYYY-MM-DD) | Sim* |
| `--end-date` | Data fim (YYYY-MM-DD) | Sim* |
| `--account-id` | ID da conta Meta (act_XXXXXXXXX) | **Sim** |
| `--leads-path` | Caminho para CSV de leads | Não |
| `--vendas-path` | Caminho para pasta de vendas | Não |
| `--output-dir` | Diretório de saída | Não |
| `--config` | Caminho para config YAML | Não |
| `--product-value` | Valor do produto (R$) | Não |
| `--max-match-days` | Janela de matching (dias) | Não |
| `--meta-token` | Token Meta API | Não |

\* Deve usar `--periodo` **OU** `--start-date/--end-date`

## 📊 Outputs Gerados

### 1. Terminal (Tempo Real)

Durante a execução, você verá:

```
================================================================================
🚀 SISTEMA DE VALIDAÇÃO DE PERFORMANCE ML - LEAD SCORING
================================================================================

📂 CARREGANDO DADOS...
   ✅ 8450 leads carregados
   ✅ 245 vendas carregadas (Guru + TMB)

🏷️ CLASSIFICANDO CAMPANHAS...
   ✅ COM ML: 3500 leads (48.5%)
   ✅ SEM ML: 3716 leads (51.5%)

🔗 VINCULANDO LEADS COM VENDAS...
   ✅ Conversões: 180
   ✅ Taxa de conversão geral: 2.49%

================================================================================
📊 RESUMO EXECUTIVO - COMPARAÇÃO ML vs NÃO-ML
================================================================================
+------------------+----------+----------+
| Métrica          | COM ML   | SEM ML   |
+==================+==========+==========+
| Total de Leads   | 3,500    | 3,716    |
| Conversões       | 105      | 75       |
| Taxa Conversão   | 3.00%    | 2.02%    |
| ROAS             | 2.47x    | 1.63x    |
+------------------+----------+----------+

🏆 VENCEDOR: COM ML (ROAS 51.5% maior)

================================================================================
📈 PERFORMANCE POR DECIL (Real vs Esperado)
================================================================================
[Tabela detalhada com Guru vs Guru+TMB]
```

### 2. Excel (6 Abas)

Arquivo: `validation/resultados/validation_report_YYYYMMDD_HHMMSS.xlsx`

**Aba 1 - Resumo Executivo:**
- Estatísticas gerais (leads, conversões, receita)
- Comparação COM_ML vs SEM_ML
- Vencedor destacado

**Aba 2 - Métricas por Campanha:**
- Tabela detalhada por campanha
- Colunas: ml_type, campaign, leads, conversions, conversion_rate, spend, cpl, roas, margin

**Aba 3 - Performance por Decil:**
- **IMPORTANTE:** Separação Guru vs Guru+TMB
- Colunas: decile, leads, conversions_guru, conversions_total, conversion_rate_guru, conversion_rate_total, expected_conversion_rate, performance_ratio_guru, performance_ratio_total, revenue_guru, revenue_total

**Aba 4 - Comparação ML:**
- Tabela agregada COM_ML vs SEM_ML
- Diferenças absolutas e percentuais

**Aba 5 - Matching Stats:**
- Estatísticas de vinculação leads-vendas
- Match por email vs telefone

**Aba 6 - Configuração:**
- Parâmetros utilizados na análise

### 3. Gráficos PNG (5 arquivos)

Salvos em `validation/resultados/`:

1. **`conversion_rate_comparison.png`**
   - Barras: Taxa conversão COM ML vs SEM ML

2. **`roas_comparison.png`**
   - Barras: ROAS COM ML vs SEM ML
   - Linha horizontal: breakeven (ROAS = 1.0)

3. **`decile_performance.png`**
   - Barras agrupadas: Taxa real vs esperada (D1-D10)

4. **`cumulative_revenue_by_decile.png`**
   - Linha: Receita acumulada por decil

5. **`contribution_margin_by_campaign.png`**
   - Barras: Margem por campanha (verde positivo, vermelho negativo)

## 🔍 Lógica de Classificação

### Campanhas

**Filtro Base:**
Deve conter: `"DEVLF | CAP | FRIO"` (campanhas de captação)

**COM ML:**
Contém: `"MACHINE LEARNING"`
Exemplo: `"DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2"`

**SEM ML:**
Contém outros padrões como `"ESCALA SCORE"`, `"FAIXA A"`, etc.
Exemplo: `"DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2"`

**EXCLUIR:**
Não contém `"DEVLF | CAP | FRIO"` (não é campanha de captação)

### Matching Leads ↔ Vendas

1. **Match primário:** Email exato (normalizado)
2. **Match secundário:** Telefone exato (se email não bateu)
3. **Validação temporal:** Venda deve ser APÓS captura do lead
4. **Janela máxima:** 30 dias (configurável)

## ⚙️ Configuração

Edite `configs/validation_config.yaml` para ajustar:

```yaml
# Token Meta API
meta_access_token: "EAAV..."

# Configurações globais
product_value: 2000.00
max_match_days: 30

# Padrões de campanha
campaign_filters:
  base_pattern: "DEVLF | CAP | FRIO"
  ml_pattern: "MACHINE LEARNING"

# Períodos
periodos:
  periodo_1:
    name: "Lançamento 11/11"
    start_date: "2025-11-11"
    end_date: "2025-12-01"

# Taxas esperadas (do modelo)
expected_conversion_rates:
  D1: 0.003836
  D2: 0.004933
  # ... D3-D10
```

## ⚠️ Troubleshooting

### Erro: "Arquivo de leads não encontrado"
- Verifique se o CSV está em `validation/leads/`
- Use `--leads-path` para especificar caminho customizado

### Erro: "Nenhuma venda carregada"
- Verifique se os arquivos Excel estão em `validation/vendas/`
- Arquivos devem começar com `guru_` ou `tmb_`
- Use `--vendas-path` para especificar caminho customizado

### Aviso: "Meta access token não configurado"
- Configure o token em `validation_config.yaml`
- Ou use `--meta-token` na linha de comando
- Sem token, spend será 0 para todas as campanhas (ROAS não será calculado corretamente)

### Erro: "Período não encontrado"
- Verifique se o período existe em `validation_config.yaml`
- Use `--start-date/--end-date` para período customizado

## 📌 Notas Importantes

### Guru vs Guru+TMB

**Por que separamos?**

O modelo foi treinado **APENAS** com vendas da Guru. Por isso:

- **Métricas Guru:** Mostram performance nos dados de treinamento (ground truth)
- **Métricas Total (Guru+TMB):** Mostram generalização do modelo para novos dados

Se o modelo estiver bem calibrado:
- Performance Guru ≈ Performance Total
- Se Total >> Guru: Modelo está sub-predizendo (conservador)
- Se Total << Guru: Modelo está sobre-predizendo (otimista)

### Taxa de Conversão Esperada

As taxas esperadas vêm do arquivo de configuração do modelo (`api/business_config.py`):

```python
CONVERSION_RATES = {
    'D1': 0.003836,  # 0.38%
    'D2': 0.004933,  # 0.49%
    # ...
    'D10': 0.034551  # 3.46%
}
```

## 📚 Dependências

O sistema usa bibliotecas já instaladas no projeto:
- `pandas` - Manipulação de dados
- `numpy` - Cálculos numéricos
- `matplotlib` - Gráficos
- `seaborn` - Visualizações
- `xlsxwriter` - Geração de Excel
- `pyyaml` - Leitura de configuração
- `tabulate` - Tabelas no terminal

Se `tabulate` não estiver instalada:
```bash
pip install tabulate
```

## 🎯 Exemplo Completo

```bash
# 1. Preparar dados
cp ~/Downloads/leads_google_sheets.csv validation/leads/leads_completo.csv
cp ~/Downloads/guru_vendas_*.xlsx validation/vendas/
cp ~/Downloads/tmb_vendas_*.xlsx validation/vendas/

# 2. Configurar token Meta
nano configs/validation_config.yaml
# (editar meta_access_token)

# 3. Executar validação
python scripts/validate_ml_performance.py \
  --periodo periodo_1 \
  --account-id act_123456789

# 4. Verificar outputs
ls -lh validation/resultados/
```

## 🔗 Próximos Passos

Após implementar o sistema, você pode:

1. **Testar com dados históricos:**
   ```bash
   python scripts/validate_ml_performance.py \
     --start-date 2025-10-01 \
     --end-date 2025-10-31 \
     --account-id act_123456789
   ```

2. **Automatizar validações periódicas:**
   - Criar cron job ou GitHub Action
   - Gerar relatórios automaticamente após cada lançamento

3. **Adicionar alertas:**
   - Enviar email se ROAS COM_ML < SEM_ML
   - Alertar se performance real << esperada

4. **Integrar com dashboard:**
   - Usar Excel/PNG em apresentações
   - Publicar métricas em Google Data Studio

---

**Sistema implementado em:** 2025-11-26
**Versão:** 1.0
**Status:** ✅ Pronto para produção
