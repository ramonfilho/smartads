# Guia do Projeto - Pipeline de Lead Scoring DevClub

## Contexto do Projeto

Este projeto visa criar um pipeline de produção para lead scoring, baseado em um modelo de Machine Learning desenvolvido e validado em ambiente de notebook (Google Colab). O modelo foi treinado com dados históricos de leads e alunos da DevClub, alcançando métricas promissoras para priorização de budget em campanhas de marketing.

## Estrutura de Arquivos

### Arquivos do Modelo
- `modelo_lead_scoring_v1_devclub.pkl`: Modelo RandomForest treinado
- `model_metadata.json`: Metadados do modelo (hiperparâmetros, métricas de performance)
- `features_ordenadas.json`: Lista ordenada das 65 features esperadas pelo modelo
- `feature_registry.json`: Registro detalhado das features (se existir)
- `smart_ads_devclub_eda_v3.ipynb`: Notebook com EDA e desenvolvimento do modelo

### Dados de Treinamento (Referência)
- Pasta: `data/devclub/LF + ALUNOS/`
- 30 arquivos utilizados no treinamento
- Lead Score LF24.xlsx: Template de entrada em produção

## Especificações Técnicas

### Modelo
- **Tipo**: RandomForestClassifier (scikit-learn 1.6.1)
- **Features**: 65 variáveis categóricas e numéricas
- **Performance**:
  - AUC: 0.649
  - Lift máximo (decil 10): 2.27x
  - Concentração top 3 decis: 48.74%

### Features Principais
1. **Demográficas**: idade, faixa salarial, gênero
2. **Comportamentais**: situação atual, posse de cartão de crédito
3. **Interesse**: experiência com programação, interesse no curso
4. **UTM/Marketing**: source, medium, term
5. **Validação**: qualidade dos dados (nome, email, telefone)

## Pipeline de Produção - Requisitos

### Entrada
- Arquivo Excel no formato do Lead Score LF24.xlsx
- Apenas dados de leads (sem necessidade de merge com alunos)

### Processamento Necessário
1. **Leitura e validação** do arquivo de entrada
2. **Engenharia de features**:
   - Criação de variáveis derivadas (comprimento do nome, validações)
   - Codificação de variáveis categóricas
   - Tratamento de valores ausentes
3. **Alinhamento de features** com as 65 esperadas pelo modelo
4. **Predição** usando o modelo pkl
5. **Saída** com scores e probabilidades

### Estrutura Proposta
```
V2/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # Carregamento de dados
│   │   └── validator.py      # Validação de dados
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py    # Criação de features
│   │   └── processor.py      # Processamento final
│   ├── models/
│   │   ├── __init__.py
│   │   └── predictor.py      # Interface do modelo
│   └── utils/
│       ├── __init__.py
│       └── config.py         # Configurações
├── scripts/
│   └── run_pipeline.py       # Script principal
├── tests/
│   └── test_pipeline.py      # Testes unitários
├── config/
│   └── pipeline_config.yaml  # Configurações
└── requirements.txt           # Dependências
```

## Diferenças: Notebook vs Produção

### No Notebook (Desenvolvimento)
- Merge de múltiplos arquivos (leads + alunos)
- Análise exploratória extensiva
- Experimentação com diferentes modelos
- Validação temporal com dados históricos

### Em Produção
- Input único: arquivo de leads no formato padrão
- Apenas transformações essenciais
- Modelo fixo (RandomForest)
- Processamento em batch ou real-time

## Metodologia de Migração - Análise Célula por Célula

### Processo de Análise
Para cada célula do notebook, seguiremos este protocolo:

1. **Análise de Necessidade**
   - ✅ Necessária: Transformação essencial para produção
   - ⚠️ Adaptação: Requer modificação para produção
   - ❌ Desnecessária: Apenas exploratória/desenvolvimento

2. **Replicação do Código**
   - Identificação do módulo de destino (data/, features/, etc.)
   - Apresentação do código original vs código adaptado
   - Localização exata no projeto (arquivo:linha)

3. **Validação**
   - Teste comparativo: resultado notebook vs pipeline
   - Verificação de compatibilidade de output
   - Asserções específicas para garantir equivalência

### Template de Documentação por Célula

```
CÉLULA #X: [Descrição]
Status: ✅Necessária/⚠️Adaptação/❌Desnecessária
Tipo: [Importação/Transformação/Visualização/Modelagem]

Código Original:
[código do notebook]

Código Produção:
[código adaptado]
Localização: src/módulo/arquivo.py:linha

Teste de Validação:
[teste unitário que garante equivalência]
```

## Próximos Passos

1. **Análise do notebook** célula por célula com documentação detalhada
2. **Criação incremental do pipeline** com aprovação por etapa
3. **Implementação das transformações** com testes de equivalência
4. **Validação contínua** comparando outputs notebook vs pipeline
5. **Documentação de uso** do pipeline final

## Considerações Importantes

- **Omitir completamente a pasta V1** (versão desatualizada)
- **Manter compatibilidade** com o formato Lead Score LF24.xlsx
- **Preservar a ordem das features** conforme features_ordenadas.json
- **Implementar logging** para rastreabilidade
- **Adicionar tratamento de erros** robusto

## Métricas de Sucesso

- Pipeline processa novos leads sem erros
- Scores gerados são consistentes com o modelo original
- Performance mantida (AUC ~0.65, Lift ~2.3x no decil superior)
- Código modular e testável