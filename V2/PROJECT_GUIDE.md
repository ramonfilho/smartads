# Guia do Projeto - Pipeline de Lead Scoring DevClub

## Contexto do Projeto

Este projeto visa criar um pipeline de produção para lead scoring, baseado em um modelo de Machine Learning desenvolvido e validado em ambiente de notebook (Google Colab). O modelo foi treinado com dados históricos de leads e alunos da DevClub, alcançando métricas promissoras para priorização de budget em campanhas de marketing.

## Estrutura de Arquivos

### Arquivos do Modelo na pasta arquivos_modelo
- `modelo_lead_scoring_v1_devclub.pkl`: Modelo RandomForest treinado
- `model_metadata.json`: Metadados do modelo (hiperparâmetros, métricas de performance)
- `features_ordenadas.json`: Lista ordenada das 65 features esperadas pelo modelo
- `feature_registry.json`: Registro detalhado das features (se existir)
- `smart_ads_devclub_eda_v3.py`: Script Python extraído do notebook original (mais legível para análise)

### Dados de Treinamento (Referência)
- Pasta: `data/devclub/LF + ALUNOS/`
- 30 arquivos utilizados no treinamento
- data/devclub/LF + ALUNOS/Lead Score LF24.xlsx (eTemplate de entrada em produção)

## Especificações Técnicas

### Modelos em Produção

O sistema utiliza 4 modelos treinados que atendem critérios rigorosos de performance:

**Modelos Implementados:**
1. **V1 DEVCLUB RF Sem_UTM**: AUC 0.629, Top3 45.4%, Lift 2.3x, Monotonia 77.8%
2. **V1 DEVCLUB LGBM Cutoff_10_08**: AUC 0.629, Top3 43.8%, Lift 2.1x, Monotonia 77.8%
3. **V2 DEVCLUB RF Cutoff_10_08**: AUC 0.622, Top3 44.2%, Lift 2.0x, Monotonia 77.8%
4. **V2 TODOS RF Cutoff_10_08**: AUC 0.615, Top3 45.6%, Lift 1.9x, Monotonia 77.8%

**Critérios de Seleção:**
- Monotonia > 70%
- Lift máximo > 1.7x
- Top 3 decis > 43% das conversões

## Diferenças entre o Arquivo Python resultante do download do Notebook (smart_ads_devclub_eda_v3.py) vs Produção

### No arquivo Python baixado do Notebook (Desenvolvimento)
- Merge de múltiplos arquivos (leads + alunos)
- Análise exploratória extensiva (células / sessões com o título "Investigação")
- Experimentação com diferentes modelos
- Validação temporal com dados históricos

### Em Produção
- Input único: arquivo de leads no formato padrão
- Modelo fixo (RandomForest)
- Processamento em batch

## Metodologia de Migração - Para cada sessão / célula:
  1. Se a sessão começar com Investigação: ignorar
  2. Se contiver código de desenvolvimento: confirmar necessidade
  3. Se contiver código de produção:
     3.1. Confirmar necessidade linha a linha
     3.2. Se necessário:
        3.2.1. Criar componente em src/
        3.2.2. Apresentar comparação original vs adaptado
        3.2.3. **PARAR e aguardar aprovação do componente**
        3.2.4. **Se aprovado:**
           - Integrar no pipeline principal
           - Testar pipeline com dados de exemplo
           - Mostrar resultado do teste
        3.2.5. **SOMENTE após teste bem-sucedido:** continuar para próxima seção

  ## Fluxo de Trabalho:
  - Análise Seção → Criar Componente → Integrar → Testar → Próxima Seção
  - NÃO acumular múltiplos componentes antes de integrar
  - Cada integração deve ser validada antes de prosseguir

  ## Estrutura de Diretórios do Projeto V2

  V2/
  ├── src/
  │   ├── data/
  │   │   ├── preprocessing.py    # Componentes de pré-processamento
  │   │   └── loader.py           # Carregamento de dados
  │   ├── features/
  │   │   └── engineering.py      # Engenharia de features
  │   ├── model/
  │   │   └── scoring.py          # Aplicação do modelo
  │   └── pipeline.py             # Pipeline principal (integra todos os componentes)
  ├── tests/
  │   ├── test_components.py      # Testes unitários dos componentes
  │   └── test_pipeline.py        # Teste de integração do pipeline completo
  └── main.py                     # Script de execução em produção

  ## Regras de Integração:
  - **pipeline.py**: APENAS importa e orquestra componentes, sem lógica própria
  - **Componentes em src/**: Contêm a lógica migrada do notebook
  - **Testes**: Sempre em diretório separado `tests/` com os dados reais:
    - Arquivo: `data/devclub/LF + ALUNOS/Lead score LF 24.xlsx`
    - Aba: `LF Pesquisa`
    - Caminho relativo dos testes: `../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx`
  - **main.py**: Entry point para produção (não misturar com testes)

**PRINCÍPIO BASE:** Manter a **LÓGICA DE PROCESSAMENTO IDÊNTICA** ao ambiente de treinamento, permitindo apenas adaptações necessárias para produção.

**Por que esta regra é CRÍTICA:**
- ❌ **Qualquer discrepância na lógica pode causar perda de features**
- ❌ **Categorias podem ser processadas incorretamente**
- ❌ **O modelo pode quebrar silenciosamente**
- ❌ **Debugar pipelines de ML é um pesadelo real**
- ❌ **Diferenças sutis podem causar data drift artificial**

### 📋 CATEGORIAS DE ALTERAÇÕES

#### ✅ ALTERAÇÕES SEGURAS (Permitidas sem consulta)
1. **Estrutura de dados:** Múltiplos arquivos/abas → DataFrame único
2. **Interface:** Remover código do Colab (upload, files.upload())
3. **Visualização:** Remover plots, prints de análise exploratória, pular células ou sessões com o título "Investigação"
4. **Modularização:** Dividir código em funções (mantendo lógica idêntica)
5. **Infraestrutura:** Adicionar logging, tratamento de erros, validações

#### ⚠️ ALTERAÇÕES CRÍTICAS (Requerem aprovação explícita)
- Número, quantidade, abreviação o ordem dos itens de alguma lista
- Ordem de operações
- Condições, filtros ou critérios
- Nomes de colunas ou mapeamentos
- Tratamento de valores nulos/missing
- Tipos de dados ou conversões
- Fórmulas ou cálculos

## Pipeline de Produção - Requisitos

### Entrada
- Arquivo Excel no formato do Lead Score LF24.xlsx
- Apenas dados de leads (sem necessidade de merge com alunos)

- **Manter compatibilidade** com o formato Lead Score LF24.xlsx
- **Preservar a ordem das features** conforme features_ordenadas.json
## Métricas de Sucesso
- Pipeline processa novos leads sem erros