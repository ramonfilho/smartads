# Plano de Implementação: Sistema de Validação de Performance ML

## Objetivo

Criar sistema de validação para medir a performance real do modelo de ML de lead scoring, comparando:
1. Campanhas COM ML vs SEM ML (taxa conversão, ROAS, margem)
2. Performance real por decil D1-D10 (conversão e ROI)

## Contexto do Negócio

**Funil de Vendas:**
- Terça: Início captação → Segunda: Fim captação
- Terça a Domingo: Período CPL
- Segunda: Abertura carrinho → Domingo: Fechamento carrinho

**Períodos de Análise:**
1. Captação: 11/11/2025, Validação: 01/12/2025
2. Captação: 18/11/2025, Validação: 08/12/2025
3. Captação: 25/11/2025, Validação: 15/12/2025

**Identificação de Campanhas:**
- **Filtro base:** Deve conter "DEVLF | CAP | FRIO" (campanhas de captação para o lançamento)
- **COM ML:** Contém "MACHINE LEARNING" (ex: "DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2 | 2025-05-30")
- **SEM ML:** Contém "ESCALA SCORE" ou outros padrões (ex: "DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2 | 2025-07-08")
- **Excluir:** Campanhas que não contém "DEVLF | CAP | FRIO" (não são de captação para lançamento)

## Arquitetura do Sistema

### Estrutura de Pastas

```
V2/
├── src/
│   └── validation/
│       ├── __init__.py
│       ├── data_loader.py          # Carrega leads CSV + vendas Excel
│       ├── campaign_classifier.py  # Classifica COM/SEM ML
│       ├── matching.py             # Match leads ↔ vendas
│       ├── metrics_calculator.py   # Calcula todas as métricas
│       ├── report_generator.py     # Gera Excel multi-abas
│       └── visualization.py        # Gera gráficos PNG
├── scripts/
│   └── validate_ml_performance.py  # CLI principal
├── validation/                     # Dados de validação (gitignore)
│   ├── leads/                      # Google Sheets CSV aqui (todo o período)
│   ├── vendas/                     # Guru + TMB xlsx aqui (todo o período)
│   └── resultados/                 # Output: Excel + gráficos
└── configs/
    └── validation_config.yaml      # Configurações globais
```

### Módulos e Responsabilidades

#### 1. `data_loader.py` (Base do Pipeline)
**Função:** Carregar e normalizar dados de entrada

**Classes:**
```python
class LeadDataLoader:
    def load_leads_csv(csv_path: str) -> pd.DataFrame
        # Lê CSV do Google Sheets
        # Colunas esperadas: Data, E-mail, Nome Completo, Telefone,
        #                    Campaign, lead_score, Source, Medium, etc.
        # Normaliza emails/telefones
        # Retorna DataFrame padronizado

class SalesDataLoader:
    def load_guru_sales(guru_paths: List[str]) -> pd.DataFrame
        # Lê arquivos Excel da Guru
        # Colunas: email contato, nome contato, valor venda,
        #          utm_campaign, data pedido
        # Normaliza emails/telefones
        # Retorna DataFrame padronizado com origem='guru'

    def load_tmb_sales(tmb_paths: List[str]) -> pd.DataFrame
        # Lê arquivos Excel da TMB
        # Colunas: Cliente Email, Cliente Nome, Ticket (R$),
        #          utm_campaign, Status
        # Filtra apenas Status='Efetivado'
        # Retorna DataFrame padronizado com origem='tmb'

    def combine_sales(guru_df, tmb_df) -> pd.DataFrame
        # Combina e deduplica vendas (prioriza Guru se conflito)
```

**Normalização:**
- Email: lowercase, strip spaces
- Telefone: remove caracteres não numéricos, valida DDD brasileiro
- Datas: parse para datetime com tratamento de erros

---

#### 2. `campaign_classifier.py` (Classificação ML)
**Função:** Identificar campanhas COM vs SEM ML

**Funções:**
```python
def is_captacao_campaign(campaign_name: str) -> bool:
    """
    Verifica se é campanha de captação para lançamento.
    Retorna: True se contém "DEVLF | CAP | FRIO"
    """
    if not campaign_name or pd.isna(campaign_name):
        return False
    return 'devlf | cap | frio' in campaign_name.lower()

def classify_campaign(campaign_name: str) -> str:
    """
    Classifica campanha de captação em COM_ML, SEM_ML ou EXCLUIR.

    Lógica:
    1. Se não contém "DEVLF | CAP | FRIO" → 'EXCLUIR' (não é de captação)
    2. Se contém "MACHINE LEARNING" → 'COM_ML'
    3. Senão (ex: "ESCALA SCORE") → 'SEM_ML'

    Exemplos:
    - "DEVLF | CAP | FRIO | FASE 04 | ADV | MACHINE LEARNING | PG2" → COM_ML
    - "DEVLF | CAP | FRIO | FASE 04 | ADV | ESCALA SCORE | PG2" → SEM_ML
    - "DEVLF | AQUECIMENTO | FASE 01 | ..." → EXCLUIR
    """
    if not campaign_name or pd.isna(campaign_name):
        return 'EXCLUIR'

    campaign_lower = campaign_name.lower()

    # 1. Verificar se é campanha de captação
    if 'devlf | cap | frio' not in campaign_lower:
        return 'EXCLUIR'

    # 2. Classificar COM_ML vs SEM_ML
    if 'machine learning' in campaign_lower:
        return 'COM_ML'
    else:
        return 'SEM_ML'

def add_ml_classification(df: pd.DataFrame, campaign_col: str = 'Campaign') -> pd.DataFrame:
    """Adiciona coluna 'ml_type' ao DataFrame e filtra campanhas excluídas"""
    df['ml_type'] = df[campaign_col].apply(classify_campaign)

    # Filtrar apenas campanhas de captação (COM_ML ou SEM_ML)
    before_count = len(df)
    df = df[df['ml_type'] != 'EXCLUIR'].copy()
    after_count = len(df)

    excluded_count = before_count - after_count
    if excluded_count > 0:
        print(f"⚠️ {excluded_count} leads de campanhas não-captação foram excluídos")

    return df
```

---

#### 3. `matching.py` (Match Leads ↔ Vendas)
**Função:** Vincular leads captados com vendas realizadas

**Lógica de Matching:**
1. **Match primário:** Email exato (normalizado)
2. **Match secundário:** Telefone exato (normalizado) quando email não bate
3. **Validação temporal:** Venda deve ocorrer APÓS captura do lead
4. **Janela máxima:** 30 dias entre captura e venda (configurável)

**Funções:**
```python
def match_leads_to_sales(
    leads_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    max_days_window: int = 30
) -> pd.DataFrame:
    """
    Retorna leads_df com colunas adicionadas:
    - converted: bool (se vendeu)
    - sale_value: float (valor da venda, ou 0)
    - sale_date: datetime (data da venda, ou None)
    - sale_origin: str ('guru', 'tmb', ou None)
    - match_method: str ('email', 'telefone', ou None)
    """
    # Reutiliza funções de V2/src/matching/matching_email_telefone.py:
    # - normalizar_email()
    # - normalizar_telefone_robusto()

def get_matching_stats(matched_df: pd.DataFrame) -> Dict:
    """Retorna estatísticas de matching para validação"""
    return {
        'total_leads': len(matched_df),
        'matched_by_email': ...,
        'matched_by_phone': ...,
        'total_conversions': ...,
        'match_rate': ...
    }
```

**Casos Edge:**
- Lead com múltiplas vendas: considera primeira venda (janela de conversão)
- Venda sem lead correspondente: ignorada (não estava na campanha)
- Email/telefone inválido: tenta ambos os métodos

---

#### 4. `metrics_calculator.py` (Core - Cálculo de Métricas)
**Função:** Calcular todas as métricas de performance

**Classes:**
```python
class CampaignMetricsCalculator:
    def __init__(self, meta_api_integration: MetaAdsIntegration, product_value: float):
        self.meta_api = meta_api_integration
        self.product_value = product_value

    def calculate_campaign_metrics(
        self,
        matched_df: pd.DataFrame,
        account_id: str,
        period_start: str,
        period_end: str
    ) -> pd.DataFrame:
        """
        Calcula métricas por campanha:
        - leads: count total de leads
        - conversions: count de converted=True
        - conversion_rate: conversions / leads
        - total_revenue: sum(sale_value) OU conversions * product_value
        - spend: buscado via Meta API
        - cpl: spend / leads
        - roas: total_revenue / spend
        - contribution_margin: total_revenue - spend
        - margin_percent: (contribution_margin / spend) * 100

        Agrupa por: ml_type (COM_ML, SEM_ML) e Campaign
        """
        # 1. Agregar dados de conversão
        campaign_stats = matched_df.groupby(['ml_type', 'Campaign']).agg({
            'email': 'count',  # total leads
            'converted': 'sum',  # conversions
            'sale_value': 'sum'  # revenue (se disponível)
        })

        # 2. Buscar custos via Meta API
        costs = self.meta_api.get_costs_hierarchy(
            account_id=account_id,
            since_date=period_start,
            until_date=period_end
        )
        # Mapear custos para campanhas (por nome)

        # 3. Calcular métricas usando funções de V2/api/economic_metrics.py:
        # - calculate_cpl(spend, leads)
        # - calculate_contribution_margin(product_value, conversion_rate, leads, spend)

        return campaign_metrics_df

class DecileMetricsCalculator:
    def calculate_decile_performance(
        self,
        matched_df: pd.DataFrame,
        product_value: float
    ) -> pd.DataFrame:
        """
        Calcula métricas reais por decil (D1-D10) separando Guru vs Guru+TMB.

        IMPORTANTE: Modelo foi treinado APENAS com vendas Guru, então precisamos
        avaliar performance separadamente:
        - Guru: Dados do treinamento (ground truth)
        - Guru+TMB: Dados completos de validação

        Métricas calculadas:
        - leads: count por decil
        - conversions_guru: count converted=True WHERE sale_origin='guru'
        - conversions_total: count converted=True (guru + tmb)
        - conversion_rate_guru: conversions_guru / leads
        - conversion_rate_total: conversions_total / leads
        - expected_conversion_rate: taxa esperada do modelo (CONVERSION_RATES)
        - performance_ratio_guru: conversion_rate_guru / expected_conversion_rate
        - performance_ratio_total: conversion_rate_total / expected_conversion_rate
        - revenue_guru: conversions_guru * product_value
        - revenue_total: conversions_total * product_value

        Agrupa por: decile (D1-D10)
        """
        # Thresholds já estão nos dados (coluna 'decile' do lead_score)
        # CONVERSION_RATES esperadas em V2/api/business_config.py

        decile_metrics = []
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_df = matched_df[matched_df['decile'] == decile]

            # Total de leads
            leads = len(decile_df)

            # Conversões separadas por origem
            conversions_guru = len(decile_df[
                (decile_df['converted'] == True) &
                (decile_df['sale_origin'] == 'guru')
            ])
            conversions_total = len(decile_df[decile_df['converted'] == True])

            # Taxas de conversão
            conversion_rate_guru = (conversions_guru / leads * 100) if leads > 0 else 0
            conversion_rate_total = (conversions_total / leads * 100) if leads > 0 else 0

            # Taxa esperada do modelo
            expected_rate = CONVERSION_RATES[decile] * 100  # Converter para %

            # Performance ratios
            performance_ratio_guru = (conversion_rate_guru / expected_rate) if expected_rate > 0 else 0
            performance_ratio_total = (conversion_rate_total / expected_rate) if expected_rate > 0 else 0

            # Receitas
            revenue_guru = conversions_guru * product_value
            revenue_total = conversions_total * product_value

            decile_metrics.append({
                'decile': decile,
                'leads': leads,
                'conversions_guru': conversions_guru,
                'conversions_total': conversions_total,
                'conversion_rate_guru': conversion_rate_guru,
                'conversion_rate_total': conversion_rate_total,
                'expected_conversion_rate': expected_rate,
                'performance_ratio_guru': performance_ratio_guru,
                'performance_ratio_total': performance_ratio_total,
                'revenue_guru': revenue_guru,
                'revenue_total': revenue_total
            })

        return pd.DataFrame(decile_metrics)
```

**Reutilização:**
- `V2/api/meta_integration.py`: `MetaAdsIntegration.get_costs_hierarchy()`
- `V2/api/economic_metrics.py`: `calculate_cpl()`, `calculate_contribution_margin()`
- `V2/api/business_config.py`: `CONVERSION_RATES`, `PRODUCT_VALUE`
- `V2/files/20251111_212345/model_metadata_v1_devclub_rf_temporal_single.json`: thresholds

---

#### 5. `report_generator.py` (Output Excel)
**Função:** Gerar relatório Excel multi-abas com formatação

**Estrutura do Excel:**

```python
class ValidationReportGenerator:
    def generate_excel_report(
        self,
        campaign_metrics: pd.DataFrame,
        decile_metrics: pd.DataFrame,
        ml_comparison: Dict,
        matching_stats: Dict,
        output_path: str
    ):
        """
        Gera Excel com 6 abas:
        1. 'Resumo Executivo' - KPIs principais, comparação COM vs SEM ML
        2. 'Métricas por Campanha' - Tabela detalhada campaign_metrics
        3. 'Performance por Decil' - Tabela decile_metrics com esperado vs real
        4. 'Comparação ML vs Não-ML' - Tabela agregada COM_ML vs SEM_ML
        5. 'Matching Stats' - Estatísticas de vinculação leads-vendas
        6. 'Configuração' - Parâmetros usados na análise
        """
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        workbook = writer.book

        # Formatos
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#4472C4', 'font_color': 'white'
        })
        percent_format = workbook.add_format({'num_format': '0.00%'})
        currency_format = workbook.add_format({'num_format': 'R$ #,##0.00'})

        # Aba 1: Resumo Executivo
        summary_data = {
            'Métrica': ['Total de Leads', 'Conversões', 'Taxa Conversão',
                       'Receita Total', 'Gasto Total', 'ROAS', 'Margem'],
            'COM ML': [...],
            'SEM ML': [...],
            'Diferença %': [...]
        }
        pd.DataFrame(summary_data).to_excel(writer, 'Resumo Executivo')

        # Aba 2-6: restante dos dados
        # Aplicar formatação condicional em colunas específicas
```

---

#### 6. `visualization.py` (Gráficos)
**Função:** Gerar visualizações em PNG

**Gráficos a Gerar:**

```python
class ValidationVisualizer:
    def generate_all_charts(
        self,
        campaign_metrics: pd.DataFrame,
        decile_metrics: pd.DataFrame,
        output_dir: str
    ):
        """Gera 5 gráficos PNG"""

    def plot_conversion_rate_comparison(self):
        # Gráfico de barras: Taxa conversão COM ML vs SEM ML
        # Eixo Y: Taxa (%)
        # Barras lado a lado

    def plot_roas_comparison(self):
        # Gráfico de barras: ROAS COM ML vs SEM ML
        # Linha horizontal em ROAS = 1.0 (breakeven)

    def plot_decile_performance(self):
        # Gráfico de barras: Conversão Real vs Esperada por Decil
        # Eixo X: D1-D10
        # Barras agrupadas (Real, Esperado)

    def plot_cumulative_revenue(self):
        # Gráfico de linha: Receita acumulada por Decil
        # Mostra que D9-D10 geram maior receita

    def plot_contribution_margin(self):
        # Gráfico de barras: Margem de Contribuição por Campanha
        # Ordenado do maior para o menor
        # Cores: verde (positivo), vermelho (negativo)
```

**Biblioteca:** matplotlib + seaborn (já estão instaladas)

---

#### 7. `validate_ml_performance.py` (CLI Principal)
**Função:** Script de linha de comando para executar análise

**Interface:**

```bash
# Uso básico (período será filtrado por datas do config)
python scripts/validate_ml_performance.py \
  --periodo periodo_1 \
  --account-id act_XXXXXXXXX

# Uso com parâmetros customizados
python scripts/validate_ml_performance.py \
  --periodo periodo_1 \
  --account-id act_XXXXXXXXX \
  --product-value 2000 \
  --max-match-days 30 \
  --leads-path validation/leads/leads_completo.csv \
  --vendas-path validation/vendas \
  --config configs/validation_config.yaml

# Uso com datas customizadas (sobrescreve config)
python scripts/validate_ml_performance.py \
  --account-id act_XXXXXXXXX \
  --start-date 2025-11-11 \
  --end-date 2025-12-01 \
  --product-value 2000
```

**Fluxo de Execução:**

```python
def main():
    print("=" * 80)
    print("🚀 SISTEMA DE VALIDAÇÃO DE PERFORMANCE ML - LEAD SCORING")
    print("=" * 80)

    # 1. Parse argumentos
    args = parse_args()

    # 2. Carregar configuração
    config = load_config(args.config)

    # 3. Carregar dados
    print("\n📂 CARREGANDO DADOS...")
    leads_df = LeadDataLoader().load_leads_csv(args.leads_path)
    print(f"   ✅ {len(leads_df)} leads carregados")

    sales_df = SalesDataLoader().combine_sales(
        guru_paths=glob(f"{args.vendas_path}/guru_*.xlsx"),
        tmb_paths=glob(f"{args.vendas_path}/tmb_*.xlsx")
    )
    print(f"   ✅ {len(sales_df)} vendas carregadas (Guru + TMB)")

    # 4. Classificar campanhas
    print("\n🏷️ CLASSIFICANDO CAMPANHAS...")
    leads_df = add_ml_classification(leads_df, campaign_col='Campaign')
    com_ml = len(leads_df[leads_df['ml_type'] == 'COM_ML'])
    sem_ml = len(leads_df[leads_df['ml_type'] == 'SEM_ML'])
    print(f"   ✅ COM ML: {com_ml} leads ({com_ml/len(leads_df)*100:.1f}%)")
    print(f"   ✅ SEM ML: {sem_ml} leads ({sem_ml/len(leads_df)*100:.1f}%)")

    # 5. Matching
    print("\n🔗 VINCULANDO LEADS COM VENDAS...")
    matched_df = match_leads_to_sales(leads_df, sales_df, max_days_window=args.max_match_days)
    matching_stats = get_matching_stats(matched_df)
    print(f"   ✅ Conversões: {matching_stats['total_conversions']}")
    print(f"   ✅ Taxa de conversão geral: {matching_stats['conversion_rate']:.2f}%")
    print(f"   ✅ Match por email: {matching_stats['matched_by_email']}")
    print(f"   ✅ Match por telefone: {matching_stats['matched_by_phone']}")

    # 6. Buscar custos Meta
    print("\n💰 BUSCANDO CUSTOS DAS CAMPANHAS (META API)...")
    meta_api = MetaAdsIntegration(access_token=config['meta_access_token'])

    # 7. Calcular métricas
    print("\n📊 CALCULANDO MÉTRICAS...")
    campaign_calc = CampaignMetricsCalculator(meta_api, args.product_value)
    campaign_metrics = campaign_calc.calculate_campaign_metrics(
        matched_df, args.account_id, args.start_date, args.end_date
    )
    print(f"   ✅ Métricas calculadas para {len(campaign_metrics)} campanhas")

    decile_calc = DecileMetricsCalculator()
    decile_metrics = decile_calc.calculate_decile_performance(
        matched_df, args.product_value
    )
    print(f"   ✅ Performance calculada para todos os decis (D1-D10)")

    # 8. Comparação ML
    ml_comparison = compare_ml_vs_non_ml(campaign_metrics)

    # 9. EXIBIR RESUMO NO TERMINAL
    print("\n" + "=" * 80)
    print("📊 RESUMO EXECUTIVO - COMPARAÇÃO ML vs NÃO-ML")
    print("=" * 80)
    print_summary_table(ml_comparison)

    print("\n" + "=" * 80)
    print("📈 PERFORMANCE POR DECIL (Real vs Esperado)")
    print("=" * 80)
    print_decile_table(decile_metrics)

    # 10. Gerar relatório Excel
    print("\n📄 Gerando relatório Excel...")
    output_dir = "validation/resultados"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"{output_dir}/validation_report_{timestamp}.xlsx"
    report_gen = ValidationReportGenerator()
    report_gen.generate_excel_report(
        campaign_metrics, decile_metrics, ml_comparison,
        matching_stats, excel_path
    )
    print(f"   ✅ Excel salvo: {excel_path}")

    # 11. Gerar gráficos
    print("\n📈 Gerando visualizações...")
    viz = ValidationVisualizer()
    viz.generate_all_charts(campaign_metrics, decile_metrics, output_dir)
    print(f"   ✅ 5 gráficos PNG salvos em: {output_dir}/")

    # 12. Finalização
    print("\n" + "=" * 80)
    print("✅ VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 80)
    print(f"\n📁 Arquivos gerados:")
    print(f"   - {excel_path}")
    print(f"   - {output_dir}/conversion_rate_comparison.png")
    print(f"   - {output_dir}/roas_comparison.png")
    print(f"   - {output_dir}/decile_performance.png")
    print(f"   - {output_dir}/cumulative_revenue_by_decile.png")
    print(f"   - {output_dir}/contribution_margin_by_campaign.png")

def print_summary_table(ml_comparison: Dict):
    """Exibe tabela de comparação ML vs Não-ML no terminal"""
    from tabulate import tabulate

    data = [
        ['Total de Leads', ml_comparison['com_ml']['leads'], ml_comparison['sem_ml']['leads']],
        ['Conversões', ml_comparison['com_ml']['conversions'], ml_comparison['sem_ml']['conversions']],
        ['Taxa Conversão', f"{ml_comparison['com_ml']['conversion_rate']:.2f}%", f"{ml_comparison['sem_ml']['conversion_rate']:.2f}%"],
        ['Gasto Total', f"R$ {ml_comparison['com_ml']['spend']:,.2f}", f"R$ {ml_comparison['sem_ml']['spend']:,.2f}"],
        ['CPL', f"R$ {ml_comparison['com_ml']['cpl']:,.2f}", f"R$ {ml_comparison['sem_ml']['cpl']:,.2f}"],
        ['ROAS', f"{ml_comparison['com_ml']['roas']:.2f}x", f"{ml_comparison['sem_ml']['roas']:.2f}x"],
        ['Margem Contrib.', f"R$ {ml_comparison['com_ml']['margin']:,.2f}", f"R$ {ml_comparison['sem_ml']['margin']:,.2f}"],
    ]

    headers = ['Métrica', 'COM ML', 'SEM ML']
    print(tabulate(data, headers=headers, tablefmt='grid'))

    # Mostrar vencedor
    if ml_comparison['com_ml']['roas'] > ml_comparison['sem_ml']['roas']:
        improvement = (ml_comparison['com_ml']['roas'] / ml_comparison['sem_ml']['roas'] - 1) * 100
        print(f"\n🏆 VENCEDOR: COM ML (ROAS {improvement:.1f}% maior)")
    else:
        print(f"\n⚠️ SEM ML performou melhor")

def print_decile_table(decile_metrics: pd.DataFrame):
    """Exibe tabela de performance por decil no terminal (Guru vs Guru+TMB)"""
    from tabulate import tabulate

    # Formatar dados para exibição
    table_data = []
    for _, row in decile_metrics.iterrows():
        table_data.append([
            row['decile'],
            row['leads'],
            row['conversions_guru'],
            row['conversions_total'],
            f"{row['conversion_rate_guru']:.2f}%",
            f"{row['conversion_rate_total']:.2f}%",
            f"{row['expected_conversion_rate']:.2f}%",
            f"{row['performance_ratio_guru']:.2f}x",
            f"{row['performance_ratio_total']:.2f}x",
            f"R$ {row['revenue_guru']:,.0f}",
            f"R$ {row['revenue_total']:,.0f}"
        ])

    headers = [
        'Decil', 'Leads',
        'Conv\nGuru', 'Conv\nTotal',
        'Taxa\nGuru', 'Taxa\nTotal',
        'Taxa\nEsperada',
        'Perf\nGuru', 'Perf\nTotal',
        'Receita\nGuru', 'Receita\nTotal'
    ]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Resumo de performance
    total_guru = decile_metrics['revenue_guru'].sum()
    total_tmb_only = decile_metrics['revenue_total'].sum() - total_guru
    print(f"\n💰 Receita Total Guru: R$ {total_guru:,.2f}")
    print(f"💰 Receita Total TMB: R$ {total_tmb_only:,.2f}")
    print(f"💰 Receita Total (Guru+TMB): R$ {decile_metrics['revenue_total'].sum():,.2f}")
```

---

### Arquivo de Configuração (`validation_config.yaml`)

```yaml
# Token de acesso da Meta Ads API
meta_access_token: "EAAV..."

# Configurações globais
product_value: 2000.00
max_match_days: 30

# Padrões de campanha
campaign_filters:
  base_pattern: "DEVLF | CAP | FRIO"  # Filtro base (campanhas de captação)
  ml_pattern: "MACHINE LEARNING"       # Padrão para COM ML
  non_ml_patterns:                     # Padrões para SEM ML
    - "ESCALA SCORE"
    - "FAIXA A"
    - "FAIXA B"

# Caminhos dos arquivos (únicos, contêm todo o período)
paths:
  leads: "validation/leads"      # CSV do Google Sheets
  vendas: "validation/vendas"    # Excel Guru + TMB

# Períodos de análise (para filtrar dados por data)
periodos:
  periodo_1:
    name: "Lançamento 11/11"
    start_date: "2025-11-11"
    end_date: "2025-12-01"

  periodo_2:
    name: "Lançamento 18/11"
    start_date: "2025-11-18"
    end_date: "2025-12-08"

  periodo_3:
    name: "Lançamento 25/11"
    start_date: "2025-11-25"
    end_date: "2025-12-15"

# Taxas de conversão esperadas (do modelo)
expected_conversion_rates:
  D1: 0.003836
  D2: 0.004933
  D3: 0.006421
  D4: 0.008366
  D5: 0.010896
  D6: 0.014197
  D7: 0.018499
  D8: 0.024105
  D9: 0.031412
  D10: 0.034551
```

---

## Fluxo de Dados

```
1. Google Sheets CSV → LeadDataLoader → leads_df (normalizado)
   ├─ Colunas: email, nome, telefone, Data, Campaign, lead_score, decile, UTMs

2. Guru + TMB Excel → SalesDataLoader → sales_df (normalizado)
   ├─ Colunas: email, nome, telefone, sale_date, sale_value, utm_campaign, origem

3. leads_df → CampaignClassifier → leads_df + ml_type
   ├─ ml_type: 'COM_ML' (MACHINE LEARNING) ou 'SEM_ML'

4. leads_df + sales_df → Matching → matched_df
   ├─ Novas colunas: converted, sale_value, sale_date, match_method

5. matched_df + Meta API → CampaignMetricsCalculator → campaign_metrics
   ├─ Por campanha: leads, conversions, conversion_rate, spend, cpl, roas, margin

6. matched_df → DecileMetricsCalculator → decile_metrics
   ├─ Por decil: leads, conversions, real_rate, expected_rate, performance_ratio

7. campaign_metrics + decile_metrics → ReportGenerator → Excel (6 abas)

8. campaign_metrics + decile_metrics → Visualizer → 5 gráficos PNG
```

---

## Tratamento de Casos Edge

### 1. Dados Ausentes
- **Lead sem lead_score:** Excluir da análise de decis (log warning)
- **Lead sem Campaign:** Classificar como 'SEM_ML' por padrão
- **Venda sem email/telefone:** Tentar ambos métodos de matching
- **Campanha sem custo na Meta API:** Usar spend=0 (log warning)

### 2. Múltiplas Vendas
- **Lead com 2+ vendas:** Considerar apenas a primeira venda na janela
- **Email em múltiplas campanhas:** Atribuir à campanha mais recente antes da venda

### 3. Validação de Dados
- **Email inválido:** Normalizar lowercase, validar formato básico (@)
- **Telefone inválido:** Validar DDD brasileiro (11-99), remover não-numéricos
- **Data inválida:** Parse com pd.to_datetime(errors='coerce'), excluir se NaT
- **Valor negativo:** Log error, excluir da análise

### 4. Matching
- **Janela temporal:** Venda > 30 dias após lead → não considerar conversão
- **Lead antes da campanha:** Verificar data_lead >= data_inicio_campanha
- **Deduplicação:** Email ou telefone duplicado → priorizar lead mais recente

---

## Validação dos Resultados

### Checks Automáticos
1. **Total de vendas:** Sum(conversions) == len(sales_df matched)
2. **Receita total:** Sum(sale_value) == Sum(conversions × product_value)
3. **ROAS mínimo:** Nenhuma campanha com spend > 0 e leads = 0
4. **Match rate:** > 70% (se menor, investigar qualidade dos dados)

### Verificações Manuais
- Comparar total de leads com Google Sheets original
- Validar custos Meta API contra Facebook Ads Manager
- Conferir amostra de matchings (10-20 leads) manualmente

---

## Output Final

### 1. Output no Terminal

Ao executar o script, o usuário verá:

```
================================================================================
🚀 SISTEMA DE VALIDAÇÃO DE PERFORMANCE ML - LEAD SCORING
================================================================================

📂 CARREGANDO DADOS...
   ✅ 8450 leads carregados
   ✅ 245 vendas carregadas (Guru + TMB)

🏷️ CLASSIFICANDO CAMPANHAS...
   ⚠️ 1234 leads de campanhas não-captação foram excluídos
   ✅ COM ML: 3500 leads (48.5%)
   ✅ SEM ML: 3716 leads (51.5%)

🔗 VINCULANDO LEADS COM VENDAS...
   ✅ Conversões: 180
   ✅ Taxa de conversão geral: 2.49%
   ✅ Match por email: 165
   ✅ Match por telefone: 15

💰 BUSCANDO CUSTOS DAS CAMPANHAS (META API)...

📊 CALCULANDO MÉTRICAS...
   ✅ Métricas calculadas para 12 campanhas
   ✅ Performance calculada para todos os decis (D1-D10)

================================================================================
📊 RESUMO EXECUTIVO - COMPARAÇÃO ML vs NÃO-ML
================================================================================
+------------------+----------+----------+
| Métrica          | COM ML   | SEM ML   |
+==================+==========+==========+
| Total de Leads   | 3,500    | 3,716    |
+------------------+----------+----------+
| Conversões       | 105      | 75       |
+------------------+----------+----------+
| Taxa Conversão   | 3.00%    | 2.02%    |
+------------------+----------+----------+
| Gasto Total      | R$ 85k   | R$ 92k   |
+------------------+----------+----------+
| CPL              | R$ 24.29 | R$ 24.76 |
+------------------+----------+----------+
| ROAS             | 2.47x    | 1.63x    |
+------------------+----------+----------+
| Margem Contrib.  | R$ 125k  | R$ 58k   |
+------------------+----------+----------+

🏆 VENCEDOR: COM ML (ROAS 51.5% maior)

================================================================================
📈 PERFORMANCE POR DECIL (Guru vs Guru+TMB)
================================================================================
+--------+-------+------+-------+-------+-------+---------+------+-------+-----------+-----------+
| Decil  | Leads | Conv | Conv  | Taxa  | Taxa  | Taxa    | Perf | Perf  | Receita   | Receita   |
|        |       | Guru | Total | Guru  | Total | Esperada| Guru | Total | Guru      | Total     |
+========+=======+======+=======+=======+=======+=========+======+=======+===========+===========+
| D1     | 721   | 2    | 3     | 0.28% | 0.42% | 0.38%   |0.73x | 1.11x | R$ 4k     | R$ 6k     |
+--------+-------+------+-------+-------+-------+---------+------+-------+-----------+-----------+
| D2     | 722   | 3    | 5     | 0.42% | 0.69% | 0.49%   |0.86x | 1.41x | R$ 6k     | R$ 10k    |
+--------+-------+------+-------+-------+-------+---------+------+-------+-----------+-----------+
| ...    | ...   | ...  | ...   | ...   | ...   | ...     | ...  | ...   | ...       | ...       |
+--------+-------+------+-------+-------+-------+---------+------+-------+-----------+-----------+
| D10    | 720   | 25   | 30    | 3.47% | 4.17% | 3.46%   |1.00x | 1.20x | R$ 50k    | R$ 60k    |
+--------+-------+------+-------+-------+-------+---------+------+-------+-----------+-----------+

💰 Receita Total Guru: R$ 180,000.00
💰 Receita Total TMB: R$ 45,000.00
💰 Receita Total (Guru+TMB): R$ 225,000.00

📄 Gerando relatório Excel...
   ✅ Excel salvo: validation/resultados/validation_report_20251126_153045.xlsx

📈 Gerando visualizações...
   ✅ 5 gráficos PNG salvos em: validation/resultados/

================================================================================
✅ VALIDAÇÃO CONCLUÍDA COM SUCESSO!
================================================================================

📁 Arquivos gerados:
   - validation/resultados/validation_report_20251126_153045.xlsx
   - validation/resultados/conversion_rate_comparison.png
   - validation/resultados/roas_comparison.png
   - validation/resultados/decile_performance.png
   - validation/resultados/cumulative_revenue_by_decile.png
   - validation/resultados/contribution_margin_by_campaign.png
```

### 2. Arquivos Gerados

```
validation/resultados/
├── validation_report_20251126_153045.xlsx  # Excel com 6 abas (timestamp no nome)
├── conversion_rate_comparison.png          # Gráfico 1: Barras COM vs SEM ML
├── roas_comparison.png                     # Gráfico 2: ROAS COM vs SEM ML
├── decile_performance.png                  # Gráfico 3: Real vs Esperado D1-D10
├── cumulative_revenue_by_decile.png        # Gráfico 4: Receita acumulada
└── contribution_margin_by_campaign.png     # Gráfico 5: Margem por campanha
```

### Estrutura do Excel

**Aba 1 - Resumo Executivo:**
| Métrica | COM ML | SEM ML | Diferença % |
|---------|--------|--------|-------------|
| Total Leads | 5,000 | 3,000 | +66.7% |
| Conversões | 150 | 60 | +150% |
| Taxa Conversão | 3.0% | 2.0% | +50% |
| ROAS | 3.5x | 2.1x | +66.7% |
| Margem | R$ 105k | R$ 33k | +218% |

**Aba 2 - Métricas por Campanha:**
| Campaign | ml_type | leads | conversions | rate | spend | cpl | roas | margin |
|----------|---------|-------|-------------|------|-------|-----|------|--------|

**Aba 3 - Performance por Decil (Guru vs Guru+TMB):**

IMPORTANTE: Modelo treinado apenas com Guru, então precisamos avaliar ambos separadamente.

| Decil | Leads | Conv Guru | Conv Total | Taxa Guru | Taxa Total | Taxa Esperada | Perf Guru | Perf Total | Receita Guru | Receita Total |
|-------|-------|-----------|------------|-----------|------------|---------------|-----------|------------|--------------|---------------|
| D1 | 500 | 2 | 3 | 0.40% | 0.60% | 0.38% | 1.05x | 1.58x | R$ 4k | R$ 6k |
| D2 | 500 | 3 | 5 | 0.60% | 1.00% | 0.49% | 1.22x | 2.04x | R$ 6k | R$ 10k |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| D10 | 500 | 18 | 22 | 3.60% | 4.40% | 3.46% | 1.04x | 1.27x | R$ 36k | R$ 44k |

**Totais:**
- Receita Total Guru: R$ XXX
- Receita Total TMB: R$ YYY
- Receita Total (Guru+TMB): R$ ZZZ

**Insights:**
- Performance Guru mostra precisão no dado de treinamento
- Performance Total mostra generalização para novos dados (TMB)

---

## Cronograma de Implementação

**Fase 1 - Setup (2h):**
- Criar estrutura de pastas
- Configurar validation_config.yaml
- Instalar dependências

**Fase 2 - Data Loading (4h):**
- Implementar LeadDataLoader
- Implementar SalesDataLoader
- Testar com dados de treino existentes

**Fase 3 - Classificação e Matching (4h):**
- Implementar CampaignClassifier
- Implementar lógica de matching
- Validar qualidade do matching

**Fase 4 - Métricas (6h):**
- Implementar CampaignMetricsCalculator
- Implementar DecileMetricsCalculator
- Integrar com Meta API
- Testes unitários

**Fase 5 - Outputs (4h):**
- Implementar ReportGenerator
- Implementar Visualizer
- Formatação do Excel

**Fase 6 - CLI e Testes (2h):**
- Implementar validate_ml_performance.py
- Testar fluxo end-to-end
- Documentação

**Total: ~22 horas (3 dias úteis)**

---

## Arquivos Críticos para Leitura Antes da Implementação

1. `V2/api/meta_integration.py` - Entender API Meta e estrutura de custos
2. `V2/api/economic_metrics.py` - Reutilizar funções de ROAS/CPL
3. `V2/src/matching/matching_email_telefone.py` - Normalização de contatos
4. `V2/files/20251111_212345/model_metadata_v1_devclub_rf_temporal_single.json` - Thresholds
5. `data/devclub/treino/GURU VENDAS 2025.xlsx` - Estrutura vendas Guru
6. `data/devclub/TMB/vendas total.xlsx` - Estrutura vendas TMB

---

## Próximos Passos

1. ✅ Plano aprovado
2. Criar estrutura de pastas e arquivos vazios
3. Implementar módulos na ordem: data_loader → matching → metrics → report → viz → cli
4. Testar com dados históricos (data/devclub/treino)
5. Validar com período real quando arquivos estiverem disponíveis
