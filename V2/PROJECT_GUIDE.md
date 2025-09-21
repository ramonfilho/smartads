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

### Estrutura do Código V2
```
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

### Estratégia: Comparação via Prints
1. **Identificar** prints no script de treino (smart_ads_devclub_eda_v3.py)
2. **Replicar** prints idênticos no pipeline produção
3. **Comparar** outputs treino vs produção
4. **Corrigir** divergências uma etapa por vez
5. **Validar** outputs idênticos antes de prosseguir

**Fluxo:** Print → Comparar → Corrigir → Validar → Próxima Etapa

## Status Atual

### ✅ Implementado
- Pipeline modular completo em V2/src/
- Testes de validação em tests/
- 4 modelos treinados com métricas validadas

### 🚨 Problemas Identificados
1. **Features UTM:** valores diferentes do esperado
2. **Ordem features:** `pd.get_dummies()` produz ordem alfabética vs esperada
3. **Features artificiais:** handcoded sem correspondência no treino
4. **Validação falhando:** 0/4 modelos passam nos testes de compatibilidade

## Princípios Fundamentais

### Regras Críticas
- **Lógica IDÊNTICA** ao ambiente de treinamento
- **SEM soluções handcoded** - apenas debug das diferenças
- **Uma correção por vez** - sempre com validação
- **Notebook Colab** é a fonte da verdade (não editar .py local)

### Alterações Permitidas
- Remover código do Colab (upload, visualizações)
- Modularizar mantendo lógica idêntica
- Adicionar logging e tratamento de erros

### Alterações que Requerem Aprovação
- Ordem de operações ou listas
- Nomes de colunas ou mapeamentos
- Tratamento de valores nulos
- Fórmulas ou cálculos

## Objetivo Imediato
Debugar sistematicamente o pipeline via prints para identificar e corrigir as divergências entre treino e produção, garantindo que as features sejam geradas identicamente.