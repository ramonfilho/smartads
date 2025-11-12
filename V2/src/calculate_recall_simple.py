"""
Cálculo simples de recall e fator de correção
Usa dados diretos dos arquivos e do modelo
"""

import pandas as pd
import glob
import os

print("="*80)
print("CÁLCULO DE RECALL E FATOR DE CORREÇÃO")
print("="*80)

# Período do modelo (do metadata)
PERIODO_INICIO = pd.to_datetime('2025-03-01')
PERIODO_FIM = pd.to_datetime('2025-11-04')

print(f"\nPeríodo do modelo: {PERIODO_INICIO.date()} a {PERIODO_FIM.date()}")

# === 1. DADOS DO MODELO (já treinado) ===
print(f"\n{'='*80}")
print("DADOS DO MODELO (metadata)")
print("="*80)

# Do model_metadata.json
TOTAL_LEADS = 108_700
TRAINING_LEADS = 75_950
TEST_LEADS = 32_750
CONVERSOES_OBSERVADAS_TRAIN = 530
CONVERSOES_OBSERVADAS_TEST = 239
CONVERSOES_OBSERVADAS_TOTAL = CONVERSOES_OBSERVADAS_TRAIN + CONVERSOES_OBSERVADAS_TEST

print(f"\nTotal de leads: {TOTAL_LEADS:,}")
print(f"  Training: {TRAINING_LEADS:,} ({CONVERSOES_OBSERVADAS_TRAIN} conversões)")
print(f"  Test: {TEST_LEADS:,} ({CONVERSOES_OBSERVADAS_TEST} conversões)")
print(f"\nConversões OBSERVADAS (matches): {CONVERSOES_OBSERVADAS_TOTAL}")
print(f"Taxa observada: {CONVERSOES_OBSERVADAS_TOTAL/TOTAL_LEADS*100:.4f}%")

# === 2. CARREGAR VENDAS REAIS DOS ARQUIVOS ===
print(f"\n{'='*80}")
print("CARREGANDO VENDAS REAIS DOS ARQUIVOS")
print("="*80)

data_dir = "/Users/ramonmoreira/Desktop/smart_ads/data/devclub/treino"
vendas_files = [
    "vendas_guru_1.1.xlsx",
    "vendas_guru_2.1.xlsx",
    "vendas_guru_2.2.xlsx",
    "GURU VENDAS 2025.xlsx"
]

vendas_all = []

for filename in vendas_files:
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"⚠️  Arquivo não encontrado: {filename}")
        continue

    print(f"\nProcessando: {filename}")

    # Ler arquivo
    excel_file = pd.ExcelFile(filepath)

    # Processar cada aba
    for sheet_name in excel_file.sheet_names:
        # Pular abas que não são vendas
        if any(skip in sheet_name.lower() for skip in ['pesquisa', 'lead', 'pontuação', 'debug', 'tabela dinâmica']):
            continue

        print(f"  Aba: {sheet_name}")
        df = pd.read_excel(filepath, sheet_name=sheet_name)

        # Filtrar colunas importantes
        # Tentar encontrar colunas de data, email, telefone, valor, produto
        colunas_relevantes = []

        # Data (priorizar data pedido ou data aprovacao)
        data_col = None
        for col in df.columns:
            if 'data pedido' in col.lower() or 'data aprovacao' in col.lower() or 'data aprovação' in col.lower():
                data_col = col
                break
        if not data_col:
            for col in df.columns:
                if col.lower() in ['data', 'date'] or ('data' in col.lower() and col.lower() not in ['data 1ª captura', 'data última captura', 'data cancelamento']):
                    data_col = col
                    break

        if data_col:
            df['data_venda'] = pd.to_datetime(df[data_col], errors='coerce')
        else:
            print(f"    ⚠️  Coluna de data não encontrada")
            continue

        # Email
        email_col = None
        for col in df.columns:
            if 'email' in col.lower() and 'contato' in col.lower():
                email_col = col
                break
        if not email_col:
            for col in df.columns:
                if 'email' in col.lower():
                    email_col = col
                    break
        if email_col:
            df['email'] = df[email_col]
        else:
            df['email'] = None

        # Telefone
        tel_col = None
        for col in df.columns:
            if 'telefone' in col.lower() and 'contato' in col.lower():
                tel_col = col
                break
        if not tel_col:
            for col in df.columns:
                if 'telefone' in col.lower() or 'phone' in col.lower() or 'celular' in col.lower():
                    tel_col = col
                    break
        if tel_col:
            df['telefone'] = df[tel_col]
        else:
            df['telefone'] = None

        # Valor
        valor_col = None
        for col in df.columns:
            if 'valor' in col.lower() and 'venda' not in col.lower():
                valor_col = col
                break
        if valor_col:
            df['valor'] = df[valor_col]

        # Produto
        produto_col = None
        for col in df.columns:
            if 'produto' in col.lower():
                produto_col = col
                break
        if produto_col:
            df['produto'] = df[produto_col]
        else:
            df['produto'] = 'unknown'

        print(f"    Linhas: {len(df):,}")

        vendas_all.append(df[['data_venda', 'email', 'telefone', 'valor', 'produto']].copy())

# Consolidar todas as vendas
print(f"\n{'='*80}")
print("CONSOLIDANDO VENDAS")
print("="*80)

vendas_consolidadas = pd.concat(vendas_all, ignore_index=True)
print(f"Total de registros: {len(vendas_consolidadas):,}")

# Filtrar por período (com janela de 30 dias)
print(f"\nFiltrando por período (com janela de conversão de 30 dias)...")
vendas_periodo = vendas_consolidadas[
    (vendas_consolidadas['data_venda'] >= PERIODO_INICIO - pd.Timedelta(days=30)) &
    (vendas_consolidadas['data_venda'] <= PERIODO_FIM + pd.Timedelta(days=30))
].copy()

print(f"Vendas no período: {len(vendas_periodo):,}")

# Remover duplicatas
print(f"\nRemovendo duplicatas...")

# Criar chave de deduplicação (email + produto + data + valor)
vendas_periodo['email_lower'] = vendas_periodo['email'].fillna('').astype(str).str.lower().str.strip()
vendas_periodo['telefone_clean'] = vendas_periodo['telefone'].fillna('').astype(str).str.strip()
vendas_periodo['produto_clean'] = vendas_periodo['produto'].fillna('').astype(str).str.strip()

# Remover linhas sem email E sem telefone
vendas_validas = vendas_periodo[
    (vendas_periodo['email_lower'] != '') | (vendas_periodo['telefone_clean'] != '')
].copy()

print(f"Vendas com email ou telefone: {len(vendas_validas):,}")

# Remover duplicatas exatas (mesmo email + produto + data + valor)
vendas_periodo['chave_dedup'] = (
    vendas_periodo['email_lower'] + '|' +
    vendas_periodo['produto_clean'] + '|' +
    vendas_periodo['data_venda'].astype(str) + '|' +
    vendas_periodo['valor'].astype(str)
)

vendas_unicas = vendas_periodo.drop_duplicates(subset='chave_dedup', keep='first')

print(f"Vendas após remover duplicatas: {len(vendas_unicas):,}")
print(f"Duplicatas removidas: {len(vendas_periodo) - len(vendas_unicas):,} ({(len(vendas_periodo) - len(vendas_unicas))/len(vendas_periodo)*100:.1f}%)")

# === 3. CALCULAR RECALL E FATOR DE CORREÇÃO ===
print(f"\n{'='*80}")
print("CÁLCULO FINAL")
print("="*80)

VENDAS_REAIS = len(vendas_unicas)
recall = CONVERSOES_OBSERVADAS_TOTAL / VENDAS_REAIS if VENDAS_REAIS > 0 else 0
fator_correcao = 1 / recall if recall > 0 else 0

taxa_observada = CONVERSOES_OBSERVADAS_TOTAL / TOTAL_LEADS
taxa_real = VENDAS_REAIS / TOTAL_LEADS

print(f"\n📊 RESUMO:")
print(f"  Leads no período: {TOTAL_LEADS:,}")
print(f"  Conversões OBSERVADAS (matches): {CONVERSOES_OBSERVADAS_TOTAL}")
print(f"  Vendas REAIS (sem duplicatas): {VENDAS_REAIS:,}")
print(f"\n📈 TAXAS:")
print(f"  Taxa OBSERVADA: {taxa_observada*100:.4f}%")
print(f"  Taxa REAL: {taxa_real*100:.4f}%")
print(f"\n🔧 MÉTRICAS:")
print(f"  Recall: {recall*100:.1f}%")
print(f"  Fator de correção: {fator_correcao:.3f}x")

# === 4. IMPACTO NOS VALORES CAPI ===
print(f"\n{'='*80}")
print("IMPACTO NOS VALORES CAPI")
print("="*80)

CONVERSION_RATES_ATUAL = {
    "D1": 0.002137,
    "D2": 0.002748,
    "D3": 0.002138,
    "D4": 0.005802,
    "D5": 0.003053,
    "D6": 0.006721,
    "D7": 0.006718,
    "D8": 0.010382,
    "D9": 0.014043,
    "D10": 0.019243,
}

PRODUCT_VALUE = 2000.0

print(f"\nVALORES ATUAIS (baseados em matches):")
for decil, taxa in CONVERSION_RATES_ATUAL.items():
    valor_capi = taxa * PRODUCT_VALUE
    print(f"  {decil}: {taxa*100:.2f}% → R$ {valor_capi:.2f}")

print(f"\nVALORES CORRIGIDOS (aplicando fator {fator_correcao:.3f}x):")
for decil, taxa in CONVERSION_RATES_ATUAL.items():
    taxa_corrigida = taxa * fator_correcao
    valor_capi_corrigido = taxa_corrigida * PRODUCT_VALUE
    print(f"  {decil}: {taxa_corrigida*100:.2f}% → R$ {valor_capi_corrigido:.2f}")

# === 5. RECOMENDAÇÕES ===
print(f"\n{'='*80}")
print("RECOMENDAÇÕES")
print("="*80)

if fator_correcao > 1:
    print(f"\n💡 SITUAÇÃO:")
    print(f"  Estamos SUBESTIMANDO em {fator_correcao:.3f}x")
    print(f"  Valores CAPI são {(1-1/fator_correcao)*100:.0f}% menores que deveriam")

    print(f"\n💡 OPÇÕES DE CORREÇÃO:")

    print(f"\n  1️⃣ CONSERVADOR (1.5x):")
    print(f"     D10: R$ {CONVERSION_RATES_ATUAL['D10']*1.5*PRODUCT_VALUE:.2f}")
    print(f"     Risco: Baixo | Ganho: Moderado")

    print(f"\n  2️⃣ MODERADO (2.0x):")
    print(f"     D10: R$ {CONVERSION_RATES_ATUAL['D10']*2.0*PRODUCT_VALUE:.2f}")
    print(f"     Risco: Médio | Ganho: Alto")

    print(f"\n  3️⃣ AGRESSIVO ({fator_correcao:.2f}x - full recall):")
    print(f"     D10: R$ {CONVERSION_RATES_ATUAL['D10']*fator_correcao*PRODUCT_VALUE:.2f}")
    print(f"     Risco: Alto | Ganho: Máximo")

    print(f"\n  4️⃣ TESTE A/B:")
    print(f"     Testar 1.5x vs 2.0x vs {fator_correcao:.2f}x")
    print(f"     Comparar ROAS em 2-4 semanas")

print("\n" + "="*80)
