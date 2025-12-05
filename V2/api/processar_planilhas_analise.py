#!/usr/bin/env python3
"""
Script para gerar predições (lead_score e decil) em planilhas de análise
Mantém os arquivos originais intactos, criando cópias com as predições

IMPORTANTE: Usa pipeline ML local (sem chamar API) para economizar custos de Cloud Run
"""
import pandas as pd
import sys
from pathlib import Path

# Adicionar caminho do projeto ao PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_pipeline import LeadScoringPipeline
from src.model.decil_thresholds import atribuir_decis_batch

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

BASE_PATH = Path('/Users/ramonmoreira/Desktop/smart_ads/data/devclub')
TREINO_PATH = BASE_PATH / 'treino'
ANALISE_PATH = BASE_PATH / 'analise'
SRC_PATH = Path(__file__).parent.parent / 'src'

ARQUIVOS = [
    'LEAD SCORE LF 14.xlsx',
    'LEAD SCORE LF 15.xlsx',
    'LF10 Lead Score.xlsx',
    '[LF07] Leads_Pesquisa.xlsx'
]

# Colunas necessárias para a API
COLUNAS_API = [
    'Nome Completo',
    'E-mail',
    'Telefone',
    'O seu gênero:',
    'Qual a sua idade?',
    'O que você faz atualmente?',
    'Atualmente, qual a sua faixa salarial?',
    'Você possui cartão de crédito?',
    'Já estudou programação?',
    'Você já fez/faz/pretende fazer faculdade?',
    'Já investiu em algum curso online para aprender uma nova forma de ganhar dinheiro?',
    'O que mais te chama atenção na profissão de Programador?',
    'O que mais você quer ver no evento?',
    'Source'
]

# =============================================================================
# FUNÇÕES
# =============================================================================

def preparar_leads_para_api(df):
    """Prepara DataFrame para envio à API"""
    # Filtrar apenas leads válidos (com nome e email)
    leads_validos = df[
        (df['Nome Completo'].notna()) &
        (df['Nome Completo'].astype(str).str.strip() != '') &
        (df['E-mail'].notna()) &
        (df['E-mail'].astype(str).str.strip() != '')
    ].copy()

    # Adicionar row_id para mapeamento
    leads_validos['row_id'] = leads_validos.index

    # Selecionar apenas colunas disponíveis
    colunas_disponiveis = [col for col in COLUNAS_API if col in leads_validos.columns]
    colunas_envio = colunas_disponiveis + ['row_id']

    return leads_validos[colunas_envio]

def gerar_predicoes_local(leads_df, pipeline):
    """Gera predições usando pipeline ML local (sem chamar API)"""
    print(f"   🤖 Gerando predições localmente com modelo ML...")

    # Salvar mapeamento row_id -> índice ANTES do processamento
    # porque o pipeline pode remover/reordenar linhas
    row_id_mapping = {i: row['row_id'] for i, row in leads_df.iterrows()}

    # Salvar como CSV temporário (pipeline espera filepath)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_csv = f.name
        leads_df.to_csv(temp_csv, index=False)

    try:
        # Executar pipeline de predição (with_predictions=True para gerar scores)
        result_df = pipeline.run(temp_csv, with_predictions=True)
    finally:
        # Remover arquivo temporário
        import os
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

    # Extrair scores
    scores = result_df['lead_score'].values

    # Carregar thresholds do modelo
    thresholds = pipeline.predictor.metadata.get('decil_thresholds', {}).get('thresholds')

    if not thresholds:
        raise Exception("Thresholds não encontrados no metadata do modelo!")

    # Calcular decis
    decis = atribuir_decis_batch(scores, thresholds)

    # Formatar decis com zero à esquerda (D01, D02, ..., D10)
    decis_formatados = []
    for decil in decis:
        # Extrair número do decil (D1 -> 1, D10 -> 10)
        num = int(decil.replace('D', ''))
        # Formatar com zero à esquerda
        decis_formatados.append(f'D{num:02d}')

    # Criar DataFrame com predições
    # O pipeline mantém a ordem das linhas, então podemos mapear por índice
    pred_list = []
    for i, (score, decil) in enumerate(zip(scores, decis_formatados)):
        if i in row_id_mapping:
            pred_list.append({
                'row_id': row_id_mapping[i],
                'lead_score': score,
                'decil': decil
            })

    pred_df = pd.DataFrame(pred_list)

    print(f"   ✅ {len(pred_df)} predições geradas")

    return pred_df

def processar_arquivo(arquivo, pipeline):
    """Processa um arquivo Excel e gera versão com predições"""
    print(f"\n{'='*80}")
    print(f"📁 PROCESSANDO: {arquivo}")
    print(f"{'='*80}")

    # Ler arquivo original
    filepath_original = TREINO_PATH / arquivo
    print(f"   📖 Lendo arquivo original...")

    df_original = pd.read_excel(filepath_original, sheet_name='LF Pesquisa')

    print(f"   Total de linhas: {len(df_original)}")

    # Preparar leads para predição
    print(f"   🔧 Preparando leads...")
    leads_preparados = preparar_leads_para_api(df_original)

    if len(leads_preparados) == 0:
        print(f"   ⚠️  Nenhum lead válido encontrado")
        return

    print(f"   Leads válidos: {len(leads_preparados)}")

    # Gerar predições localmente
    predicoes = gerar_predicoes_local(leads_preparados, pipeline)

    # Criar cópia do DataFrame original
    df_com_predicoes = df_original.copy()

    # Adicionar colunas lead_score e decil se não existirem
    if 'lead_score' not in df_com_predicoes.columns:
        df_com_predicoes['lead_score'] = None
    if 'decil' not in df_com_predicoes.columns:
        df_com_predicoes['decil'] = None

    # Mapear predições de volta ao DataFrame original usando row_id
    for _, pred in predicoes.iterrows():
        row_id = pred['row_id']
        df_com_predicoes.loc[row_id, 'lead_score'] = pred['lead_score']
        df_com_predicoes.loc[row_id, 'decil'] = pred['decil']

    # Salvar arquivo com predições
    nome_saida = arquivo.replace('.xlsx', '_com_predicoes.xlsx')
    filepath_saida = ANALISE_PATH / nome_saida

    print(f"   💾 Salvando arquivo com predições...")
    df_com_predicoes.to_excel(filepath_saida, sheet_name='LF Pesquisa', index=False)

    print(f"   ✅ Arquivo salvo: {filepath_saida}")

    # Estatísticas
    total_com_score = df_com_predicoes['lead_score'].notna().sum()
    print(f"\n   📊 ESTATÍSTICAS:")
    print(f"      Total de leads: {len(df_com_predicoes)}")
    print(f"      Com predição: {total_com_score}")
    print(f"      Sem predição: {len(df_com_predicoes) - total_com_score}")

    if total_com_score > 0:
        print(f"\n   📈 DISTRIBUIÇÃO DE DECIS:")
        decis_count = df_com_predicoes['decil'].value_counts().sort_index()
        for decil, count in decis_count.items():
            print(f"      {decil}: {count} leads")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("🚀 PROCESSAMENTO DE PLANILHAS PARA ANÁLISE (LOCAL)")
    print("=" * 80)
    print(f"Diretório de entrada: {TREINO_PATH}")
    print(f"Diretório de saída: {ANALISE_PATH}")
    print(f"Modo: Pipeline ML Local (sem custos de Cloud Run)")
    print()

    # Criar diretório de saída se não existir
    ANALISE_PATH.mkdir(parents=True, exist_ok=True)

    # Carregar pipeline ML
    print("🔧 Carregando pipeline ML e modelo...")
    try:
        pipeline = LeadScoringPipeline()
        print(f"   ✅ Pipeline carregado com sucesso")
        print(f"   Modelo: {pipeline.predictor.metadata.get('model_name', 'N/A')}")
        print(f"   Acurácia: {pipeline.predictor.metadata.get('accuracy', 'N/A')}")
    except Exception as e:
        print(f"   ❌ ERRO ao carregar pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # Processar cada arquivo
    for arquivo in ARQUIVOS:
        try:
            processar_arquivo(arquivo, pipeline)
        except Exception as e:
            print(f"\n   ❌ ERRO ao processar {arquivo}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("✅ PROCESSAMENTO COMPLETO!")
    print("=" * 80)
    print(f"\nArquivos salvos em: {ANALISE_PATH}")
    print("💰 Economia: Processamento local - zero custos de Cloud Run!")

if __name__ == '__main__':
    main()
