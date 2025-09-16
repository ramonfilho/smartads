# Análise Célula por Célula - Notebook smart_ads_devclub_eda_v3

## Seção 1: Importação e Upload de Arquivos

### CÉLULA #1: Imports e Upload
**Status:** ❌ Desnecessária
**Tipo:** Importação/Upload
**Linhas:** 12-18

**Código Original:**
```python
from google.colab import files
import pandas as pd
import os
# Upload dos arquivos
print("📤 FAÇA UPLOAD DOS ARQUIVOS .XLSX")
uploaded = files.upload()
```

**Justificativa:** Em produção, os arquivos serão fornecidos via path direto, não via upload do Colab.

**Código Produção:** Será substituído por leitura direta de arquivo
```python
# src/data/loader.py
import pandas as pd
import os
from typing import Dict, Any

def load_lead_file(filepath: str) -> pd.DataFrame:
    """Carrega arquivo de leads no formato Excel"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    return pd.read_excel(filepath)
```

---

## Seção 2: Filtragem de Abas e Remoção de Duplicatas

### CÉLULA #2: Filtro de Abas + Remoção de Duplicatas
**Status:** ⚠️ Adaptação
**Tipo:** Transformação
**Linhas:** 26-131

**Código Original:**
```python
# FILTRAR ABAS + REMOVER DUPLICATAS
termos_manter = ["Pesquisa", "Vendas", "tmb", "Guru", "Sheet"]
termos_remover = ["Pontuação", "Lead Score", "DEBUG_LOG", "Tabela Dinâmica 1", "Detalhe1", "Alunos", "Guru", "TMB", "LEADS"]
min_linhas = 230

for filename in uploaded.keys():
    xl_file = pd.ExcelFile(filename)
    for sheet_name in xl_file.sheet_names:
        df = pd.read_excel(xl_file, sheet_name=sheet_name)
        # Aplicar critérios de filtragem...
        df_sem_duplicatas = df.drop_duplicates(keep='first')
```

**Justificativa:** Em produção, trabalharemos com arquivo único (Lead Score LF24.xlsx) sem múltiplas abas. Apenas a remoção de duplicatas é necessária.

**Código Produção:**
```python
# src/data/validator.py
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove registros duplicados do DataFrame"""
    initial_count = len(df)
    df_clean = df.drop_duplicates(keep='first')
    removed_count = initial_count - len(df_clean)

    if removed_count > 0:
        print(f"Removidas {removed_count} linhas duplicadas")

    return df_clean
```
**Localização:** src/data/validator.py:5-14

---

## Seção 3: Limpeza de Colunas Desnecessárias

### CÉLULA #3: Remoção de Colunas
**Status:** ⚠️ Adaptação
**Tipo:** Transformação
**Linhas:** 135-234

**Código Original:**
```python
def obter_colunas_remover_exatas():
    return [
        'Faixa 3.0', 'Faixa A 3.0', 'Faixa B 3.0', 'Faixa C 3.0',
        # ... lista extensa de colunas
    ]

def aplicar_limpeza_colunas():
    colunas_remover = obter_colunas_remover_exatas()
    # Remove colunas desnecessárias e Unnamed
```

**Justificativa:** Em produção, precisamos remover apenas as colunas que não fazem parte das 65 features do modelo.

**Código Produção:**
```python
# src/data/cleaner.py
from typing import List
import pandas as pd

COLUNAS_DESNECESSARIAS = [
    'Faixa 3.0', 'Faixa A 3.0', 'Faixa B 3.0', 'CEP', 'cep',
    'Bairro', 'Cidade', 'Estado', 'Logradouro', 'Complemento',
    'Cliente CPF', 'Data Cancelado', 'Data Efetivado',
    'data cancelamento', 'data garantia', 'Endereço completo',
    # ... apenas colunas confirmadas como desnecessárias
]

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas desnecessárias e Unnamed"""
    # Remove colunas Unnamed
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]

    # Remove colunas da lista de desnecessárias
    cols_to_remove = []
    for col in df.columns:
        if col in COLUNAS_DESNECESSARIAS or col in unnamed_cols:
            cols_to_remove.append(col)

    df_clean = df.drop(columns=cols_to_remove, errors='ignore')

    print(f"Removidas {len(cols_to_remove)} colunas desnecessárias")
    return df_clean
```
**Localização:** src/data/cleaner.py:1-28

---

## Seção 4: Consolidação de Datasets

### CÉLULA #4: União de Datasets de Pesquisa e Vendas
**Status:** ❌ Desnecessária
**Tipo:** Transformação/Merge
**Linhas:** 238-299

**Código Original:**
```python
def consolidar_datasets():
    # Identificar e consolidar datasets de pesquisa
    dados_pesquisa = []
    dados_vendas = []
    # ... código de consolidação
    df_pesquisa_consolidado = pd.concat(dados_pesquisa, ignore_index=True)
    df_vendas_consolidado = pd.concat(dados_vendas, ignore_index=True)
```

**Justificativa:** Em produção não há necessidade de consolidar múltiplos arquivos. Receberemos apenas um arquivo de leads por vez.

---

## Resumo da Análise (Células 1-4)

### Transformações Necessárias para Produção:
1. ✅ **Remoção de duplicatas** - Manter integridade dos dados
2. ✅ **Limpeza de colunas** - Remover colunas não utilizadas pelo modelo
3. ❌ **Upload via Colab** - Substituir por leitura direta
4. ❌ **Consolidação de múltiplos arquivos** - Não aplicável em produção

### Próximas Células a Analisar:
- Engenharia de features
- Codificação de variáveis categóricas
- Tratamento de valores ausentes
- Preparação final para o modelo

---

## Seção 5: Unificação de Colunas Duplicadas (CORRIGIDA)

### CÉLULA #5: Unificação de Colunas - Pesquisa e Vendas
**Status:** ⚠️ Adaptação
**Tipo:** Transformação
**Linhas:** 321-467

**Análise Detalhada do Código Original:**

1. **PESQUISA** - Unifica colunas duplicadas/truncadas:
   - `investiu_curso_online`: unifica variações com/sem espaço no final
   - `interesse_programacao`: unifica variações com/sem espaço no final
   - USA FILLNA para combinar valores (linhas 361-367, 377-383)

2. **VENDAS** - Unifica e depois REMOVE:
   - Unifica UTMs: `utm_last_*` com `utm_*` usando fillna (linhas 429-440)
   - **IMPORTANTE:** REMOVE todas as colunas UTM unificadas (linhas 444-450)
   - Unifica outras colunas: valor, produto, nome, email, telefone usando fillna

**Observação Crítica:** No notebook, as UTMs são processadas apenas em VENDAS e depois REMOVIDAS. No Lead Score LF24 de produção, as UTMs já vêm padronizadas como: SOURCE, MEDIUM, CAMPAIGN, TERM, CONTENT.

**Justificativa:** Em produção, as colunas já virão padronizadas no formato Lead Score. Precisamos mapear possíveis variações incluindo UTMs.

**Código Produção:**
```python
# src/features/column_mapper.py
from typing import Dict, List
import pandas as pd

COLUMN_MAPPINGS = {
    # Colunas de pesquisa
    'investiu_curso_online': [
        'Já investiu em algum curso online para aprender uma nova forma de ganhar dinheiro?',
        'Já investiu em algum curso online para aprender uma nova forma de ganhar dinheiro? ',
        'ja_investiu_curso_online'
    ],
    'interesse_programacao': [
        'O que mais te chama atenção na profissão de Programador?',
        'O que mais te chama atenção na profissão de Programador? ',
        'interesse_programador'
    ],
    # Colunas básicas
    'nome': ['NOME', 'Cliente Nome', 'nome_completo', 'nome contato'],
    'email': ['E-MAIL', 'Cliente Email', 'email_contato'],
    'telefone': ['TELEFONE', 'Telefone', 'telefone_contato'],
    # UTMs - padronizadas no Lead Score
    'Source': ['SOURCE', 'utm_source', 'utm_last_source'],
    'Medium': ['MEDIUM', 'utm_medium', 'utm_last_medium'],
    'Term': ['TERM', 'utm_term', 'utm_last_term']
}

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas usando fillna como no notebook"""
    df_copy = df.copy()

    for standard_name, variations in COLUMN_MAPPINGS.items():
        existing_cols = [col for col in variations if col in df_copy.columns]

        if len(existing_cols) > 1:
            # Replica estratégia do notebook: fillna iterativo
            df_copy[standard_name] = df_copy[existing_cols[0]]
            for col in existing_cols[1:]:
                df_copy[standard_name] = df_copy[standard_name].fillna(df_copy[col])
            df_copy = df_copy.drop(columns=existing_cols)
        elif len(existing_cols) == 1:
            df_copy = df_copy.rename(columns={existing_cols[0]: standard_name})

    return df_copy
```
**Localização:** src/features/column_mapper.py:1-43

---

## Seção 6: Investigação de Colunas de Programação

### CÉLULA #6: Análise Temporal
**Status:** ❌ Desnecessária
**Tipo:** Investigação
**Linhas:** 474-499

**Justificativa:** Análise exploratória. Conclusão: colunas NÃO devem ser unificadas (refutada).

---

## Seção 8: Remoção de Features Desnecessárias

### CÉLULA #8: Remoção de Campaign e Content
**Status:** ✅ Necessária
**Tipo:** Transformação
**Linhas:** 1462-1588

**Análise:** O modelo foi treinado SEM as colunas Campaign e Content, portanto elas DEVEM ser removidas em produção também.

**Colunas removidas:**
1. `Campaign`: Específica de cada lançamento
2. `Content`: Anúncios individuais de cada lançamento
3. Colunas vazias ou com nomes problemáticos

**Código Produção:**
```python
# src/features/feature_remover.py
import pandas as pd

def remove_unnecessary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove features que não foram usadas no treinamento do modelo"""
    df_copy = df.copy()

    # Features que devem ser removidas (modelo foi treinado sem elas)
    features_to_remove = ['Campaign', 'Content', 'CAMPAIGN', 'CONTENT']

    # Remover também colunas vazias ou problemáticas
    for col in df_copy.columns:
        if col == '' or pd.isna(col) or col is None or str(col).strip() == '':
            features_to_remove.append(col)

    # Remover apenas as que existem
    existing_to_remove = [col for col in features_to_remove if col in df_copy.columns]

    if existing_to_remove:
        df_copy = df_copy.drop(columns=existing_to_remove)
        print(f"Removidas {len(existing_to_remove)} colunas: {existing_to_remove}")

    return df_copy
```
**Localização:** src/features/feature_remover.py:1-23

---

## Seção 10: Unificação de UTM Source e Term

### CÉLULA #10: Categorização de Source e Term
**Status:** ✅ Necessária
**Tipo:** Transformação
**Linhas:** 1912-2060

**Transformações:**

1. **Source** - Agrupa categorias menores em "outros":
   - Mantém: `facebook-ads` (89.6%), `google-ads` (8.3%)
   - Agrupa em "outros": `fb`, `teste`, `[field id="utm_source"]`, etc.

2. **Term** - Padroniza plataformas:
   - `ig` → `instagram`
   - `fb` → `facebook`
   - IDs numéricos (com `--`) → `outros`
   - Parâmetros dinâmicos (com `{`) → `outros`

**Código Produção:**
```python
# src/features/utm_normalizer.py
import pandas as pd
import re

def normalize_utm_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza campos UTM conforme padrão do modelo"""
    df_copy = df.copy()

    # Normalizar Source
    if 'Source' in df_copy.columns or 'SOURCE' in df_copy.columns:
        col_name = 'Source' if 'Source' in df_copy.columns else 'SOURCE'

        # Manter principais, agrupar resto em "outros"
        main_sources = ['facebook-ads', 'google-ads', 'facebook_ads', 'google_ads']
        df_copy[col_name] = df_copy[col_name].apply(
            lambda x: x if pd.isna(x) or x in main_sources else 'outros'
        )

    # Normalizar Term
    if 'Term' in df_copy.columns or 'TERM' in df_copy.columns:
        col_name = 'Term' if 'Term' in df_copy.columns else 'TERM'

        # Mapear valores conhecidos
        term_mapping = {'ig': 'instagram', 'fb': 'facebook'}

        def normalize_term(value):
            if pd.isna(value):
                return value
            value_str = str(value).lower()

            # Aplicar mapeamento direto
            if value_str in term_mapping:
                return term_mapping[value_str]

            # IDs numéricos ou parâmetros dinâmicos
            if '--' in value_str or '{' in value_str:
                return 'outros'

            # Manter instagram/facebook, outros vira "outros"
            if value_str in ['instagram', 'facebook']:
                return value_str

            return 'outros'

        df_copy[col_name] = df_copy[col_name].apply(normalize_term)

    return df_copy
```
**Localização:** src/features/utm_normalizer.py:1-45

---

## Seção 11: Unificação de UTM Medium

### CÉLULA #11: Extração de Públicos do Medium
**Status:** ✅ Necessária
**Tipo:** Transformação
**Linhas:** 2061-2200

**Análise:** Extrai o tipo de público dos valores complexos de Medium (ex: "ADV | Interesse Python" → "Interesse Python")

Continuar com análise das próximas células? (S/N)