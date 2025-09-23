# Guia do Projeto - Pipeline de Lead Scoring DevClub

## Contexto do Projeto
Este projeto visa replicar um pipeline de treinamento para lead scoring, baseado em um modelo de Machine Learning desenvolvido e validado em ambiente de notebook (Google Colab). O modelo foi treinado com dados históricos de leads e alunos da DevClub, para priorização de budget em campanhas de marketing.

## Estrutura de Arquivos

### Arquivos do Modelo na pasta arquivos_modelo
- Arquivos .pkl : Modelos treinados
- `model_metadata*.json`: Metadados dos modelos (hiperparâmetros, métricas de performance)
- `features_ordenadas*.json`: Lista ordenada das features esperadas pelos modelos
- `feature_registry*.json`: Registro detalhado das features
- `smart_ads_devclub_eda_v3.py`: Script Python extraído do notebook original

### Dados de Treinamento
- Pasta: `data/devclub/LF + ALUNOS/`
- 30 arquivos utilizados no treinamento
- `Lead Score LF24.xlsx`: Template de entrada em produção (aba: LF Pesquisa)

## 📁 MAPA DE LOCALIZAÇÃO DOS ARQUIVOS

### Estrutura Completa de Diretórios
```
smart_ads/
├── data/                              # Dados de treinamento
│   └── devclub/
│       └── LF + ALUNOS/
│           └── Lead score LF 24.xlsx  # ← ARQUIVO DE TESTE PRINCIPAL
├── V2/                                # Pipeline de produção (diretório atual)
│   ├── src/                           # Código fonte
│   ├── tests/                         # Testes (você está aqui quando roda testes)
│   ├── arquivos_modelo/               # Modelos e features esperadas
│   └── PROJECT_GUIDE.md              # Este arquivo
└── smart_ads_devclub_eda_v3-3.py     # Notebook original (fonte da verdade)
```

### Caminhos Absolutos Definitivos
- **Arquivo de teste:** `/Users/ramonmoreira/Desktop/smart_ads/data/devclub/LF + ALUNOS/Lead score LF 24.xlsx`
- **Notebook original:** `/Users/ramonmoreira/Desktop/smart_ads/V2/smart_ads_devclub_eda_v3-3.py`
- **Pipeline V2:** `/Users/ramonmoreira/Desktop/smart_ads/V2/`
- **Testes:** `/Users/ramonmoreira/Desktop/smart_ads/V2/tests/`

### Caminhos Relativos por Contexto
**Se você está em `/Users/ramonmoreira/Desktop/smart_ads/V2/tests/`:**
- Arquivo de teste: `../../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx`
- Notebook: `../smart_ads_devclub_eda_v3-3.py`

**Se você está em `/Users/ramonmoreira/Desktop/smart_ads/V2/`:**
- Arquivo de teste: `../data/devclub/LF + ALUNOS/Lead score LF 24.xlsx`
- Notebook: `./smart_ads_devclub_eda_v3-3.py`

V2/
├── src/
│   ├── data/          # Componentes de pré-processamento
│   ├── features/      # Engenharia de features
│   ├── model/         # Aplicação do modelo
│   └── pipeline.py    # Pipeline principal
├── tests/
│   └── test_pipeline.py  # Testes de integração
└── main.py            # Script de execução
```

## 🔴 METODOLOGIA DE DEBUGGING

### Estratégia: Comparação via Prints - GRANULARIDADE MÁXIMA

#### ETAPA 1: IDENTIFICAÇÃO DETALHADA DA SESSÃO
1. **PULAR AUTOMATICAMENTE** sessões que:
   - Iniciam com "Investigação"
   - Processam múltiplos arquivos (consolidação/união)
   - São específicas de treino e não aplicáveis à produção
2. **IDENTIFICAR A CÉLULA/SESSÃO** do notebook com precisão:
   - Número exato das linhas no arquivo .py
   - Título/comentário da sessão
   - Descrição completa do que a sessão faz
3. **LISTAR TODAS AS FUNÇÕES** executadas naquela sessão:
   - Cada operação realizada em ordem
   - Transformações aplicadas aos dados
   - Prints esperados com formato exato
4. **MAPEAR PARA O PIPELINE V2**:
   - Arquivo(s) específico(s) do pipeline
   - Método(s)/função(ões) correspondente(s)
   - Linha(s) exata(s) no código de produção
5. **CONFIRMAR COM USUÁRIO** antes de prosseguir:
   - "Identifiquei a sessão X (linhas Y-Z)"
   - "Esta sessão faz: [lista detalhada]"
   - "No pipeline V2, isso corresponde a: [arquivo:método:linhas]"
   - "Posso prosseguir com a comparação?"

#### ETAPA 2: EXECUÇÃO E COMPARAÇÃO
1. **USAR ARQUIVO DE TESTE EXISTENTE**: `/Users/ramonmoreira/Desktop/smart_ads/V2/tests/test_pipeline.py`
2. **EXECUTAR SEMPRE O PIPELINE COMPLETO** - NUNCA executar funções isoladas ou simulações de dados
3. **COLETAR PRINTS** do pipeline real executando com dados reais
4. **COMPARAR** linha por linha com prints do notebook
5. **VERIFICAR NÚMEROS EXATOS** - qualquer diferença numérica (colunas, registros, percentuais) é divergência crítica
6. **NENHUMA MICRO DIVERGÊNCIA É ACEITA** - todos os números devem ser idênticos
7. **AO ENCONTRAR DIVERGÊNCIA - PARAR IMEDIATAMENTE**

**IMPORTANTE:** Todos os testes e comparações devem ser feitos executando o pipeline verdadeiro (`python tests/test_pipeline.py`), não funções isoladas ou dados simulados. Somente assim garantimos que o estado dos dados está correto em cada etapa.

#### ETAPA 3: CORREÇÃO GRANULAR
1. **IDENTIFICAR A DIVERGÊNCIA ESPECÍFICA**:
   - Qual valor esperado vs obtido
   - Em qual linha/operação ocorreu
   - Qual a causa provável
2. **SUGERIR CORREÇÃO MÍNIMA**:
   - Apenas UMA linha/operação por vez
   - Explicar exatamente o que será alterado
   - Mostrar antes/depois do código
3. **AGUARDAR APROVAÇÃO** antes de implementar

#### ETAPA 4: VALIDAÇÃO
1. **SE APROVADO**: Implementar e re-executar
2. **SE REPROVADO**: Continuar na mesma sessão com nova abordagem
3. **APROVAR RESULTADO** com usuário antes de avançar

**Fluxo:** Identificar com precisão → Confirmar entendimento → Executar → Comparar → Se divergir: PARAR → Sugerir UMA correção mínima → Aprovar → Implementar → Re-executar → Aprovar resultado → Próxima sessão

### Regras Críticas
- **VALIDAÇÃO PRÉVIA OBRIGATÓRIA** - Antes de criar tarefas, análises ou executar qualquer código, validar com o usuário O QUE pretende fazer e COMO
- **UMA SESSÃO DO NOTEBOOK POR VEZ** - Processar sessão por sessão, nunca múltiplas de uma vez
- **PARAR EM CADA DIVERGÊNCIA** - Ao encontrar divergência, PARAR e aguardar aprovação antes de corrigir
- **UMA CORREÇÃO POR VEZ** - Corrigir apenas UMA divergência específica e re-executar
- **Lógica IDÊNTICA** ao ambiente de treinamento
- **SEM soluções handcoded** - apenas debug das diferenças
- **Notebook Colab** é a fonte da verdade (não editar .py local)

### Alterações Permitidas
- Remover código do Colab (upload, visualizações)
- Modularizar mantendo lógica idêntica
- Adicionar logging e tratamento de erros

### Alterações PROIBIDAS
- **CRIAÇÃO DE DADOS SINTÉTICOS PARA TESTE** - É proibida a criação de dados sintéticos ou fictícios para teste. Sempre usar os dados reais do projeto.

### Alterações que Requerem Aprovação
- Ordem de operações ou listas
- Nomes de colunas ou mapeamentos
- Tratamento de valores nulos
- Fórmulas ou cálculos

## Objetivo Imediato
Debugar sistematicamente o pipeline via prints para identificar e corrigir as divergências entre treino e produção, garantindo que as features sejam geradas identicamente.