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
- `smart_ads_devclub_eda_v4.py`: Script Python extraído do notebook mais atualizado

### Dados de Treinamento
- Pasta: `data/devclub/LF + ALUNOS/`
- 29 arquivos utilizados no treinamento
- `Lead Score LF24.xlsx`: Template de entrada em produção (aba: LF Pesquisa)

## 🔴 METODOLOGIA DE DEBUGGING

### Estratégia: Comparação via Prints - GRANULARIDADE MÁXIMA

**IMPORTANTE:** Todos os testes e comparações devem ser feitos executando o pipeline verdadeiro (`python tests/test_pipeline.py`), não funções isoladas ou dados simulados. Somente assim garantimos que o estado dos dados está correto em cada etapa.

**AGUARDAR APROVAÇÃO** antes de implementar

**APROVAR RESULTADO** com usuário antes de avançar

**Fluxo:** Identificar com precisão → Confirmar entendimento → Executar → Comparar → Se divergir: PARAR → Sugerir UMA correção mínima → Aprovar → Implementar → Re-executar → Aprovar resultado → Próxima sessão

### Regras Críticas
- **VALIDAÇÃO PRÉVIA OBRIGATÓRIA** - Antes de criar tarefas, análises ou executar qualquer código, validar com o usuário O QUE pretende fazer e COMO
- **UMA SESSÃO POR VEZ** - Processar sessão por sessão, nunca múltiplas de uma vez
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