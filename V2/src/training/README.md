# Pipeline de Treino - DevClub

## O que estamos criando

Este é o **pipeline de TREINO**, completamente separado do pipeline de produção.

### Estrutura:

```
V2/
├── src/
│   ├── training/
│   │   ├── train_pipeline.py          # Pipeline de treino (célula por célula do notebook)
│   │   └── README.md                   # Este arquivo
│   ├── data_processing/
│   │   ├── ingestion.py                # Usado por AMBOS (treino e produção)
│   │   ├── category_unification.py     # Usado por AMBOS
│   │   ├── column_unification.py       # Usado por AMBOS
│   │   ├── feature_removal.py          # Usado por AMBOS
│   │   ├── utm_training.py             # APENAS treino (célula 10)
│   │   └── utm_unification.py          # APENAS produção (NÃO TOCAR!)
│   └── production_pipeline.py          # Pipeline de produção (NÃO TOCAR!)
```

### Regras importantes:

1. **Criar novos módulos em `src/` para cada célula do notebook**
   - Nomear como `*_training.py` se for específico de treino
   - Se o módulo já existe (ex: `utm_unification.py`), NÃO editar - criar novo (ex: `utm_training.py`)

2. **NÃO adicionar código direto no `train_pipeline.py`**
   - Sempre criar função em módulo separado
   - Importar e usar no pipeline

3. **NÃO tocar em arquivos de produção**
   - `production_pipeline.py`
   - Módulos sem sufixo `_training`

## Como verificar cada nova célula

Após adicionar uma célula ao pipeline:

1. **Executar o pipeline completo:**
   ```bash
   python V2/src/training/train_pipeline.py 2>&1
   ```

2. **O output COMPLETO da nova célula deve ser mostrado como texto**
   - Cada caractere visível
   - Sem resumos ou comparações
   - Para comparação manual linha por linha

3. **Exemplo de extração de uma célula específica:**
   ```bash
   python V2/src/training/train_pipeline.py 2>&1 | sed -n '/^TÍTULO DA CÉLULA$/,/última linha$/p'
   ```

## Células implementadas

- ✅ Célula 1: Leitura de arquivos
- ✅ Célula 2: Filtragem + Remoção de duplicatas
- ✅ Célula 3: Remoção de colunas desnecessárias
- ✅ Célula 4: Consolidação de datasets
- ✅ Célula 5: Unificação de colunas duplicadas
- ✅ Célula 7: Unificação de categorias
- ✅ Célula 8: Remoção de features desnecessárias
- ✅ Célula 10: Unificação de UTM Source e Term
- ✅ Célula 11: Extração de públicos do Medium
- ✅ Célula 11.1: Unificação de Medium para produção
- ✅ Célula 13: Criação de dataset pós-cutoff
- ✅ Célula 15: Matching robusto (método original)

## Estratégia de Matching

O pipeline suporta múltiplos métodos de matching para testes comparativos:

### Método 1: Validação Original (Variantes Múltiplas)
- **Módulo:** `matching_training.py`
- **Estratégia:** Cria múltiplas variantes de telefone (com/sem DDD, com/sem 9, etc.)
- **Matching:** Usa interseção de conjuntos
- **Taxa de validação:** ~27% dos telefones são válidos
- **Vantagem:** Captura mais formatos diferentes
- **Desvantagem:** Taxa de validação baixa

### Método 2: Validação Feature Engineering (Robusto)
- **Módulo:** `matching_comparison_training.py`
- **Estratégia:** Normalização robusta com tratamento de notação científica e floats
- **Matching:** Comparação direta após normalização
- **Taxa de validação:** ~99% dos telefones são válidos
- **Vantagem:** Valida quase todos os telefones, mais matches
- **Desvantagem:** Pode gerar falsos positivos se não for cuidadoso

### Método 3: [A IMPLEMENTAR]
- Estratégia a ser definida
- Objetivo: Encontrar o melhor balanço entre precisão e recall

### Como usar

Por padrão, o pipeline usa o **Método 1** (célula 15).

Para testar outros métodos futuramente, você pode:
1. Comentar a célula 15 atual
2. Descomentar/adicionar a célula com o método desejado
3. Executar o pipeline completo
4. Comparar resultados (taxa de conversão, precisão, recall, etc.)

**No final, ficaremos apenas com o método vencedor para produção.**
