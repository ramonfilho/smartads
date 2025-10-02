# Guia do Projeto - Pipeline Configurável de Lead Scoring

## Contexto do Projeto
Este projeto visa criar um sistema de pipelines configuráveis para lead scoring, baseado em modelos de Machine Learning validados. O objetivo é evoluir de um sistema single-client (DevClub) para uma arquitetura multi-client reutilizável, mantendo toda a lógica de negócio já validada.

## Fase Atual: Profissionalização (FASE 2)
**Pré-requisito:** Cliente DevClub validado e pagando pela solução

### Objetivos da FASE 2
1. **Extrair componentes reutilizáveis** do notebook/pipeline atual
2. **Criar arquitetura configurável** por cliente
3. **Separar pipelines de treino e produção** mantendo componentes compartilhados
4. **Implementar MLflow** para tracking de experimentos
5. **Preparar base** para escala futura (FASE 3: MLOps completo)

## Nova Estrutura de Arquivos Proposta
src/
├── investigation/
│   ├── data_profiling.py      # Análise exploratória de dados
│   ├── column_analysis.py     # Análise de colunas (tipos, valores, missing)
│   ├── category_discovery.py  # Descoberta de categorias únicas
│   └── config_generator.py    # Gera configs/cliente.yaml baseado nas investigações
├── data_processing/
│   ├── ingestion.py           # Consolidação de datasets
│   ├── cleaning.py            # Limpeza e tratamento
│   ├── matching.py            # Matching email/telefone
│   └── feature_engineering.py # Criação de features
├── training/
│   ├── train_pipeline.py      # Pipeline completo de treino
│   └── model_training.py      # Lógica de treino/avaliação
├── serving/
│   ├── predict_pipeline.py    # Pipeline de predição
│   └── model_serving.py       # Carregamento e predição
└── utils/
    ├── config_loader.py       # Carregamento de configurações
    └── mlflow_utils.py        # Tracking e logging
configs/
├── devclub.yaml               # Configuração específica DevClub
└── template.yaml              # Template para novos clientes
data/
├── devclub/                   # Dados específicos DevClub
└── [cliente2]/                # Dados do próximo cliente

## 🔴 METODOLOGIA DE REFATORAÇÃO

### Estratégia: Migração Incremental - ZERO QUEBRA DE FUNCIONALIDADE

**PRINCÍPIO FUNDAMENTAL:** A funcionalidade atual (DevClub) deve continuar 100% operacional durante toda a refatoração.

**FLUXO DE TRABALHO:**
1. **Validar entendimento** da estrutura atual
2. **Extrair UM componente** por vez
3. **Testar compatibilidade** com pipeline atual
4. **Confirmar resultados idênticos** antes de prosseguir
5. **Avançar para próximo componente**

### 🔧 PRINCÍPIOS FUNDAMENTAIS DE IMPLEMENTAÇÃO

#### 1. **CONFIGURAÇÃO PRIMEIRO**
**TODO código deve ser configurável desde o início.**

- ❌ **NUNCA** hardcodar valores específicos do DevClub no código
- ✅ **SEMPRE** receber parâmetros via `configs/cliente.yaml`
- ✅ Funções recebem argumentos explícitos, não assumem defaults específicos
- ✅ Código deve funcionar para **qualquer cliente** apenas trocando a config

**Exemplo:**
```python
# ❌ ERRADO - Hardcoded
def filter_sheets(df):
    termos_manter = ["Pesquisa", "Vendas", "tmb"]  # Específico DevClub!

# ✅ CORRETO - Configurável
def filter_sheets(df, termos_manter: List[str]):
    # termos_manter vem de configs/devclub.yaml
```

#### 2. **GRANULARIDADE EXTREMA**
**TODA funcionalidade deve ser dividida em sub-etapas mínimas verificáveis.**

- Não só `ingestion.py`, mas **TUDO**: `cleaning`, `matching`, `feature_engineering`, `encoding`
- **Uma função pequena por vez**
- Cada função deve ser:
  - ✅ Testável isoladamente
  - ✅ Validável contra o notebook (outputs idênticos)
  - ✅ Aprovável antes de prosseguir para a próxima

**Exemplo de granularidade:**
```
❌ ERRADO: Criar ingestion.py completo (400 linhas) de uma vez

✅ CORRETO:
  1.1 read_excel_files() → validar → aprovar
  1.2 remove_duplicates_per_sheet() → validar → aprovar
  1.3 filter_sheets() → validar → aprovar
  1.4 consolidate_sheets() → validar → aprovar
  1.5 filter_by_date() → validar → aprovar
```

### Regras Críticas
- **FUNCIONALIDADE PRIMEIRO** - Pipeline DevClub deve continuar funcionando
- **UM COMPONENTE POR VEZ** - Extrair/refatorar apenas um módulo por sessão
- **SUB-ETAPAS GRANULARES** - Dividir cada módulo em funções mínimas verificáveis
- **TESTES DE REGRESSÃO** - Comparar outputs antes/depois de cada mudança
- **CONFIGURAÇÃO DESDE O INÍCIO** - Nada hardcoded, tudo vem de configs
- **VALIDAÇÃO CONTÍNUA** - Aprovar cada etapa antes de prosseguir

### Componentes a Extrair (Ordem Correta)

**Nota:** Sub-etapas granulares são descobertas e documentadas à medida que avançamos sessão por sessão.

#### **ATUAL: 1. data_processing/ingestion.py** - Consolidação de datasets
Sub-etapas granulares identificadas:
- 1.1 `read_excel_files()` - Leitura de múltiplos arquivos Excel
- 1.2 `remove_duplicates_per_sheet()` - Remoção de duplicatas por aba
- 1.3 `filter_sheets()` - Filtragem de abas por critérios configuráveis
- 1.4 `consolidate_sheets()` - Concatenação em DataFrame único
- 1.5 `filter_by_date()` - Filtragem temporal opcional

#### Próximos módulos (sub-etapas a descobrir):
2. **data_processing/cleaning.py** - Limpeza e tratamento básico
3. **data_processing/matching.py** - Matching e criação de targets
4. **data_processing/feature_engineering.py** - Criação de features
5. **training/model_training.py** - Treino e avaliação
6. **serving/model_serving.py** - Predição
7. **investigation/** - Módulos de investigação (geram configs)
8. **Configuração cliente-específica** - Baseada nas investigações

### Protocolo de Validação (CRÍTICO)

**Fluxo obrigatório para cada sub-etapa:**

1. **Criar função** configurável (sem valores hardcoded)
2. **Adicionar parâmetros necessários** ao `configs/devclub.yaml`
3. **Teste unitário** - Verificar que a função executa sem erros isoladamente
4. **Integração ao pipeline** - Adicionar a função ao pipeline de treino (`train_pipeline.py`)
5. **Teste integrado** - Executar pipeline completo e gerar output
6. **PARAR E AGUARDAR** - Compartilhar output com usuário
7. **Usuário valida** - Comparar output com notebook original (única fonte da verdade)
8. **Aprovação explícita** - Só avançar após confirmação do usuário

**Detalhamento da Integração ao Pipeline:**
- Criar/atualizar script que reproduz o notebook célula por célula
- Cada nova função aprovada é adicionada ao script de integração
- Script deve usar `configs/devclub.yaml` para todos os parâmetros
- Output do script deve ser comparável ao output do notebook

**REGRA CRÍTICA:**
- ❌ **NUNCA** avançar para a próxima função sem aprovação do usuário
- ❌ **NUNCA** assumir que o output está correto
- ✅ **SEMPRE** testar isolado (unitário) E integrado (pipeline completo)
- ✅ **SEMPRE** esperar confirmação: "está correto, pode prosseguir"

### Validações Obrigatórias (Feitas pelo Usuário)
- **Shapes de dataframes** idênticos em cada etapa
- **Valores** idênticos ou equivalentes
- **Estrutura de dados** mantida
- **Lógica de negócio** preservada

## Diferenças entre Pipelines

### Pipeline de Treino
- **Input:** Múltiplos arquivos históricos
- **Processamento:** Consolidação + Matching + Feature Engineering + Treino
- **Output:** Modelo salvo + Métricas + Artefatos MLflow

### Pipeline de Produção  
- **Input:** Arquivo único de leads
- **Processamento:** Feature Engineering + Predição
- **Output:** Scores de leads

### Componentes Compartilhados
- Limpeza de dados
- Feature engineering
- Validações de entrada
- Tratamento de categorias não vistas

## 🔍 Módulos de Investigação → Configuração

### Conceito Fundamental
**As configurações de um cliente são DESCOBERTAS, não inventadas.**

O fluxo para onboarding de um novo cliente é:
```
dados_cliente → investigation/ → configs/cliente.yaml → pipeline usa essa config
```

### Responsabilidade dos Módulos de Investigação
Extrair do notebook DevClub as **análises exploratórias** que foram necessárias para descobrir:
- Quais colunas usar/descartar
- Quais tipos de limpeza aplicar (encoding, missing values, outliers)
- Quais categorias únicas existem em cada coluna categórica
- Quais features criar
- Quais thresholds e parâmetros usar
- Distribuições e estatísticas relevantes

### Módulos a Criar
1. **data_profiling.py** - Overview geral dos dados (shape, tipos, missing %)
2. **column_analysis.py** - Análise detalhada coluna por coluna
3. **category_discovery.py** - Mapeamento de todas as categorias únicas
4. **config_generator.py** - Converte outputs das investigações em `configs/cliente.yaml`

### Objetivo
Quando um novo cliente chegar, rodar:
```bash
python src/investigation/run_investigation.py --client novo_cliente
```
E obter automaticamente (ou semi-automaticamente) o arquivo `configs/novo_cliente.yaml` pronto para uso nos pipelines.