# 📚 DOCUMENTAÇÃO - Pasta `src/validation/`

**Atualizado:** 2025-12-25 | **Versão:** 1.0.0

## 🎯 OBJETIVO ATUAL

**Processo de compatibilização:**
O script `validate_ml_performance.py` foi sofreu um git restore para uma versão antiga, mas todos os outros arquivos com as funções auxiliares que ele usa não sofreram o mesmo restore. 

Numa tentativa de "consertar" o erro, foram adicionadas mais de mil linhas no `validate_ml_performance.py`, mas essas linhas também contém parte da lógica antiga. 

O objetivo é re-adaptar a versão antiga do `validate_ml_performance.py` para usar os módulos e funções atuais e já existentes dos arquivos na pasta validation. Aqui estão alguns detalhes:

## 📋 INVENTÁRIO DE FUNÇÕES

---

## 1. **`__init__.py`**
**Propósito:** Módulo de inicialização do pacote de validação

**Conteúdo:**
- Apenas docstring explicativa
- Define `__version__ = "1.0.0"`

---

## 2. **`campaign_classifier.py`** (462 linhas)
**Propósito:** Classificação de campanhas em COM_ML vs SEM_ML e COM_CAPI vs SEM_CAPI

### Funções:

1. **`_check_campaign_ids_in_meta(excluded_df, campaign_col)`**
   - Verifica IDs numéricos de campanha na Meta API
   - Busca campanhas ativas para IDs fornecidos

2. **`is_captacao_campaign(campaign_name)`**
   - Verifica se é campanha de captação (contém "DEVLF | CAP | FRIO")

3. **`classify_campaign(campaign_name)`**
   - Classifica campanha em 'COM_ML', 'SEM_ML' ou 'EXCLUIR'
   - Lógica: MACHINE LEARNING → COM_ML, outras → SEM_ML

4. **`add_ml_classification(df, campaign_col='campaign')`**
   - Adiciona coluna 'ml_type' ao DataFrame
   - Filtra e remove campanhas 'EXCLUIR'
   - Retorna (df_filtrado, excluded_count)

5. **`get_classification_stats(df)`**
   - Retorna estatísticas sobre classificação ML
   - Retorna: total, com_ml_count, sem_ml_count, percentuais

6. **`list_unique_campaigns(df, ml_type=None)`**
   - Lista campanhas únicas, opcionalmente filtradas por tipo

7. **`classify_campaign_capi(optimization_goal)`**
   - Classifica por eventos CAPI: 'COM_CAPI' ou 'SEM_CAPI'
   - COM_CAPI: LeadQualified, LeadQualifiedHighQuality

8. **`add_capi_classification(df, optimization_goal_col='optimization_goal')`**
   - Adiciona coluna 'capi_type' baseado em optimization_goal

9. **`get_capi_classification_stats(df)`**
   - Retorna estatísticas sobre classificação CAPI

---

## 3. **`capi_events_counter.py`** (419 linhas)
**Propósito:** Contador de eventos CAPI dos logs do Cloud Run

### Funções:

1. **`extract_email_from_log_line(line)`**
   - Extrai email de linha de log do Cloud Run
   - Padrão: "✅ LeadQualified enviado: email@..."

2. **`extract_event_type_from_log_line(line)`**
   - Extrai tipo de evento: LeadQualified, LeadQualifiedHighQuality, Faixa A

3. **`get_capi_events_from_logs(start_date, end_date, project_id, service_name)`**
   - Busca eventos CAPI dos logs via `gcloud logging read`
   - Retorna: Dict {email: [eventos]}

4. **`get_campaign_ids_from_database(emails)`**
   - Busca campaign IDs do PostgreSQL (LeadCAPI table)
   - Retorna: Dict {email: campaign_id}

5. **`get_campaign_ids_from_csv(emails, start_date, end_date, csv_path=None)`**
   - Busca campaign IDs de CSV de leads
   - Filtra por período de captação

6. **`count_capi_events_by_campaign(start_date, end_date, project_id, service_name, csv_path=None)`**
   - Conta eventos CAPI por campanha
   - Combina logs + database/CSV
   - Retorna: Dict {campaign_id: {LeadQualified: N, ...}}

---

## 4. **`data_loader.py`** (734 linhas)
**Propósito:** Carregamento e normalização de leads (Google Sheets) e vendas (Guru/TMB)

### Classes:

#### **`LeadDataLoader`**
- **`load_leads_csv(csv_path)`**: Carrega CSV de leads do Google Sheets
  - Normaliza emails, telefones, datas
  - Atribui decis via lead_score
  - Retorna DataFrame normalizado

- **`_get_thresholds()`**: Carrega thresholds do modelo (lazy loading com cache)

- **`_assign_decile_from_score(score)`**: Atribui decil (D1-D10) baseado no score

#### **`SalesDataLoader`**
- **`load_guru_sales(guru_paths)`**: Carrega vendas da Guru (Excel)
  - Normaliza emails, telefones, valores, datas

- **`load_tmb_sales(tmb_paths)`**: Carrega vendas da TMB (Excel)
  - Filtra apenas "Efetivado"

- **`combine_sales(guru_df, tmb_df, guru_paths, tmb_paths)`**: Combina Guru + TMB
  - Deduplicação: prioriza Guru em conflitos

#### **`CAPILeadDataLoader`**
- **`load_capi_leads(start_date, end_date, emails_filter=None)`**: Carrega leads do banco CAPI via API

- **`load_combined_leads(csv_path, start_date, end_date)`**: Combina Pesquisa + CAPI
  - Prioriza pesquisa (tem lead_score)
  - Adiciona leads CAPI extras (sem pesquisa)
  - Retorna (DataFrame, stats)

---

## 5. **`matching.py`** (461 linhas)
**Propósito:** Matching (vinculação) entre leads e vendas

### Funções:

1. **`match_leads_to_sales(leads_df, sales_df, use_temporal_validation=False)`**
   - Vincula leads com vendas por email/telefone
   - Adiciona colunas: converted, sale_value, sale_date, sale_origin, match_method

2. **`_is_valid_match(lead_date, sale_date)`**
   - Valida se venda ocorreu APÓS captura do lead

3. **`get_matching_stats(matched_df, total_sales=None)`**
   - Calcula estatísticas de matching
   - Retorna: total_leads, conversions, conversion_rate, tracking_rate, revenue, etc.

4. **`print_matching_summary(stats)`**
   - Imprime resumo visual das estatísticas

5. **`filter_by_period(df, start_date, end_date, date_col='data_captura')`**
   - Filtra DataFrame por período de datas

6. **`filter_conversions_by_capture_period(matched_df, period_start, period_end)`**
   - Remove conversões de leads captados FORA do período

7. **`deduplicate_conversions(matched_df)`**
   - Remove duplicatas artificiais (mesmo email, 5 capturas, 1 venda → mantém 1)
   - Mantém lead capturado PRIMEIRO

8. **`analyze_conversion_by_decile(matched_df)`**
   - Preview de análise por decil (D1-D10)

---

## 6. **`period_calculator.py`** (270 linhas)
**Propósito:** Cálculo automático de períodos (captação, CPL, vendas)

### Classe:

#### **`PeriodCalculator`**
- **`calculate_periods(lead_capture_start)`**: Calcula os 3 períodos
  - Semana 1: Captação (Terça a Segunda - 7 dias)
  - Semana 2: CPL (Terça a Domingo - 6 dias)
  - Semana 3: Vendas (Segunda a Domingo - 7 dias)

- **`get_sales_period(lead_capture_start, lead_capture_end)`**: Retorna apenas período de vendas

- **`validate_period_logic(lead_start, lead_end, sales_start, sales_end)`**: Valida se períodos seguem lógica

### Função:

1. **`calculate_periods_from_start(lead_capture_start)`**: Wrapper de conveniência

---

## 7. **`visualization.py`** (564 linhas)
**Propósito:** Geração de gráficos PNG para validação

### Classe:

#### **`ValidationVisualizer`**
- **`generate_all_charts(campaign_metrics, decile_metrics, ml_comparison, output_dir)`**: Gera os 5 gráficos

- **`plot_conversion_rate_comparison(ml_comparison, output_dir)`**: Taxa de conversão COM vs SEM ML

- **`plot_roas_comparison(ml_comparison, output_dir)`**: ROAS COM vs SEM ML (com linha breakeven)

- **`plot_decile_performance(decile_metrics, output_dir)`**: Real vs Esperado por Decil

- **`plot_cumulative_revenue(decile_metrics, output_dir)`**: Receita acumulada D1→D10

- **`plot_contribution_margin(campaign_metrics, output_dir, top_n=15)`**: Margem por campanha (top 15)

---

## 8. **`meta_reports_loader.py`** (580 linhas)
**Propósito:** Carrega relatórios Excel exportados do Meta Ads (substitui Meta API)

### Funções:

1. **`normalize_unicode(text)`**: Normaliza texto Unicode para NFC

### Classe:

#### **`MetaReportsLoader`**
- **`load_all_reports(start_date, end_date)`**: Carrega todos os relatórios
  - Retorna: {'campaigns': df, 'adsets': df, 'ads': df}

- **`_load_and_consolidate(file_paths, report_type)`**: Consolida múltiplos Excel

- **`_extract_account_name(filename)`**: Extrai nome da conta do arquivo

- **`_normalize_column_names(df, report_type)`**: Normaliza colunas para padrão

- **`build_costs_hierarchy(start_date, end_date)`**: Constrói estrutura costs_hierarchy
  - Formato esperado pelo CampaignMetricsCalculator
  - Agrega campaigns, adsets, spend, budget, events

- **`load_adsets_for_comparison(ml_campaign_ids, control_campaign_ids)`**: Carrega adsets para comparação

- **`load_ads_for_comparison(ml_campaign_ids, control_campaign_ids)`**: Carrega ads para comparação

---

## 9. **`metrics_calculator.py`** (~2000 linhas)
**Propósito:** Cálculo de métricas de performance (campanhas, decis, ML vs Non-ML)

### Classes:

#### **`CampaignMetricsCalculator`** (linha 28)
- **`calculate_campaign_metrics(...)`**: Calcula métricas completas por campanha
  - Suporta costs_hierarchy_consolidated pré-carregado
  - Retorna: DataFrame com ml_type, leads, conversions, conversion_rate, spend, ROAS, margin, etc.

#### **`DecileMetricsCalculator`** (linha 1260)
- **`calculate_decile_metrics(...)`**: Métricas por decil (D1-D10)
  - Retorna: DataFrame com conversion_rate_total, expected_conversion_rate, revenue, etc.

### Funções:

1. **`compare_ml_vs_non_ml(campaign_metrics)`** (linha 1381)
   - Compara métricas agregadas: COM_ML vs SEM_ML
   - Retorna: Dict com comparison, com_ml, sem_ml

2. **`calculate_overall_stats(matched_df, costs_hierarchy, product_value)`** (linha 1481)
   - Estatísticas gerais do período
   - Retorna: total_leads, conversions, revenue, total_spend, ROAS, etc.

3. **`calculate_comparison_group_metrics(matched_df, costs_hierarchy, product_value, groups)`** (linha 1637)
   - Métricas por grupos de comparação (adsets_iguais, todos)

---

## 10. **`report_generator.py`** (~2000+ linhas)
**Propósito:** Geração de relatórios Excel com múltiplas abas

### Classe:

#### **`ValidationReportGenerator`** (linha 20)

Métodos principais (todos privados `_write_*`):
- **`generate_report(...)`**: Método principal que gera o Excel completo

**Abas geradas:**
- **Resumo**: `_write_summary_tab()`
- **Métricas Campanha**: `_write_campaign_metrics_tab()`
- **Métricas Decil**: `_write_decile_metrics_tab()`
- **Comparação ML vs Non-ML**: `_write_ml_comparison_tab()`
- **Fair Comparison**: `_write_fair_comparison_tab()` (adsets_iguais, todos)
- **Análise Ads**: `_write_ad_level_analysis_tab()`
- **Comparação CAPI**: `_write_capi_comparison_tab()` (se disponível)

**Métodos auxiliares:**
- Formatação de células, cores, estilos
- Criação de tabelas e gráficos
- Cálculo de estatísticas agregadas

---

## 11. **`fair_campaign_comparison.py`** (2535 linhas)
**Propósito:** Comparação justa entre campanhas ML e Controle (adsets, ads)

### Funções Principais:

1. **`create_refined_campaign_map(...)`** (linha 41): Cria mapeamento refinado de campanhas

2. **`identify_matched_adset_pairs(...)`** (linha 145): Identifica pares de adsets correspondentes

3. **`identify_matched_ad_pairs(...)`** (linha 208): Identifica pares de ads correspondentes

4. **`compare_all_adsets_performance(...)`** (linha 271): Compara performance de todos adsets

5. **`compare_adset_performance(...)`** (linha 425): Compara 1 par de adsets

6. **`compare_ad_performance(...)`** (linha 907): Compara performance de ads

7. **`compare_ads_in_matched_adsets(...)`** (linha 1164): Compara ads dentro de adsets matched

8. **`compare_matched_ads_in_matched_adsets(...)`** (linha 1227): Compara ads correspondentes

9. **`prepare_adset_comparison_for_excel(...)`** (linha 1339): Prepara dados de adsets para Excel

10. **`prepare_ad_comparison_for_excel(...)`** (linha 1425): Prepara dados de ads para Excel

11. **`get_comparison_config(comparison_level)`** (linha 1549): Retorna config de comparação

12. **`filter_campaigns_by_level(...)`** (linha 1565): Filtra campanhas por nível

13. **`filter_ads_by_level(...)`** (linha 1601): Filtra ads por nível

14. **`filter_ads_by_adset(...)`** (linha 1636): Filtra ads por adset

### Classe (adicionada para compatibilidade):

#### **`FairCampaignMatcher`** (linha 1665)
- Classe legada para compatibilidade com validate_ml_performance.py (versão 16/12)
- Métodos para matching de campanhas ML vs Controle

### Funções Duplicadas (final do arquivo - compatibilidade):
- **`identify_matched_ad_pairs(...)`** (linha 2232)
- **`get_ad_level_metrics(...)`** (linha 2278)
- **`compare_ad_performance(...)`** (linha 2359)
- **`prepare_ad_comparison_for_excel(...)`** (linha 2457)

---

## 12. **`validate_ml_performance.py`** (1125 linhas)
**Propósito:** Script principal CLI que orquestra todo o pipeline de validação

**Estrutura:**
- Imports de todos os módulos acima
- Configuração de argparse para CLI
- Pipeline completo:
  1. Carregamento de dados (leads + vendas)
  2. Classificação (COM_ML vs SEM_ML)
  3. Matching (leads → vendas)
  4. Cálculo de métricas (campanhas, decis)
  5. Comparação ML vs Non-ML
  6. Fair comparison (opcional)
  7. Geração de relatório Excel
  8. Geração de gráficos PNG

**Parâmetros principais:**
- `--periodo`: Período pré-configurado ou custom
- `--account-id`: ID(s) da conta Meta
- `--disable-fair-comparison`: Desabilita comparação justa
- `--no-cache`: Desabilita cache da Meta API

---

## 📊 RESUMO QUANTITATIVO

- **Total de arquivos:** 12
- **Total de classes:** 8
- **Total de funções:** ~70+
- **Linhas de código:** ~9,000+

**Principais capacidades:**
✅ Carregamento de dados (CSV, Excel, PostgreSQL, API)
✅ Classificação automática de campanhas
✅ Matching leads→vendas
✅ Cálculo de métricas (campanhas, decis, ML vs Non-ML)
✅ Comparação justa (adsets, ads)
✅ Geração de relatórios Excel completos
✅ Geração de gráficos PNG
✅ Suporte a múltiplas contas Meta
✅ Uso de relatórios locais (sem Meta API)
✅ Contador de eventos CAPI

---

## 🔄 MUDANÇAS IMPLEMENTADAS

### 25/12/2025 - Compatibilização Script Principal
- **Meta API → MetaReportsLoader**: Script usa relatórios Excel locais
- **sys.path corrigido**: `parent.parent.parent` para imports corretos
- **Funções legadas adicionadas**: FairCampaignMatcher, get_ad_level_metrics, etc. em fair_campaign_comparison.py

### Próximos Passos
- [ ] Executar script com dados reais
- [ ] Identificar incompatibilidades no relatório gerado
- [ ] Implementar funções dos módulos atualizados no script principal

---

**Última atualização:** 2025-12-25
