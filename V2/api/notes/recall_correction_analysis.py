"""
Análise de Recall e Correção de Taxas de Conversão
Calcula o fator de correção baseado em conversões observadas vs reais
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


# ==============================================================================
# DADOS DO MODELO
# ==============================================================================

# Caminho para os metadados do modelo
MODEL_METADATA_PATH = Path(__file__).parent.parent / "arquivos_modelo" / "model_metadata_v1_devclub_rf_temporal_single-3.json"

# Carregar metadados
with open(MODEL_METADATA_PATH, 'r') as f:
    model_metadata = json.load(f)

# Extrair conversões observadas
training_conversions = model_metadata['training_data']['target_distribution']['training_positive_count']
test_conversions = model_metadata['training_data']['target_distribution']['test_positive_count']
total_observed_conversions = training_conversions + test_conversions

# Taxas de conversão observadas por decil (do treino)
observed_rates = {
    "D1": model_metadata['decil_analysis']['decil_1']['conversion_rate'],
    "D2": model_metadata['decil_analysis']['decil_2']['conversion_rate'],
    "D3": model_metadata['decil_analysis']['decil_3']['conversion_rate'],
    "D4": model_metadata['decil_analysis']['decil_4']['conversion_rate'],
    "D5": model_metadata['decil_analysis']['decil_5']['conversion_rate'],
    "D6": model_metadata['decil_analysis']['decil_6']['conversion_rate'],
    "D7": model_metadata['decil_analysis']['decil_7']['conversion_rate'],
    "D8": model_metadata['decil_analysis']['decil_8']['conversion_rate'],
    "D9": model_metadata['decil_analysis']['decil_9']['conversion_rate'],
    "D10": model_metadata['decil_analysis']['decil_10']['conversion_rate'],
}


# ==============================================================================
# DADOS REAIS DE VENDAS
# ==============================================================================

# Vendas reais DevClub (fonte: meta_config.py, baseado em análise manual)
REAL_DEVCLUB_SALES = 1970  # Vendas desde 01/03/2025


# ==============================================================================
# CÁLCULO DE RECALL
# ==============================================================================

def calculate_recall() -> Tuple[float, float]:
    """
    Calcula recall do matching e fator de correção

    Returns:
        (recall, correction_factor)
    """
    recall = total_observed_conversions / REAL_DEVCLUB_SALES
    correction_factor = 1 / recall

    return recall, correction_factor


def calculate_corrected_rates(observed_rates: Dict[str, float], correction_factor: float) -> Dict[str, float]:
    """
    Aplica fator de correção às taxas observadas

    Args:
        observed_rates: Taxas de conversão observadas por decil
        correction_factor: Fator de correção (1/recall)

    Returns:
        Taxas de conversão corrigidas
    """
    corrected_rates = {}
    for decil, rate in observed_rates.items():
        corrected_rates[decil] = rate * correction_factor

    return corrected_rates


def analyze_impact(observed_rates: Dict[str, float], corrected_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Analisa impacto da correção nas taxas

    Returns:
        DataFrame com comparação
    """
    data = []

    for decil in observed_rates.keys():
        obs_rate = observed_rates[decil]
        corr_rate = corrected_rates[decil]
        delta_abs = corr_rate - obs_rate
        delta_rel = (delta_abs / obs_rate * 100) if obs_rate > 0 else 0

        data.append({
            'Decil': decil,
            'Taxa Observada (%)': obs_rate * 100,
            'Taxa Corrigida (%)': corr_rate * 100,
            'Δ Absoluto (pp)': delta_abs * 100,
            'Δ Relativo (%)': delta_rel
        })

    return pd.DataFrame(data)


def simulate_utm_impact(correction_factor: float):
    """
    Simula impacto em métricas UTM usando exemplo real da planilha

    Campaign X: 76 leads, R$ 703 gasto, 30.26% em D10
    """
    print("\n" + "="*80)
    print("SIMULAÇÃO DE IMPACTO EM MÉTRICAS UTM")
    print("="*80)
    print("\nExemplo: Campaign 120220370119870390 (da planilha 15/10/2025)")
    print("  - Leads: 76")
    print("  - Gasto: R$ 703,09")
    print("  - CPL: R$ 9,25")
    print("  - % em D10: 30.26%")
    print("  - Product Value: R$ 2.027,38")
    print("  - ROAS Mínimo: 2.0x")

    # Distribuição de decis (simplificada - só D10 para exemplo)
    pct_d10 = 0.3026

    # Taxas observadas vs corrigidas
    taxa_obs_d10 = observed_rates['D10']
    taxa_corr_d10 = taxa_obs_d10 * correction_factor

    # Taxa projetada (simplificada - só D10)
    taxa_proj_obs = pct_d10 * taxa_obs_d10
    taxa_proj_corr = pct_d10 * taxa_corr_d10

    # Métricas de negócio
    product_value = 2027.38
    cpl = 9.25
    min_roas = 2.0

    # ROAS projetado
    roas_obs = (product_value * taxa_proj_obs) / cpl
    roas_corr = (product_value * taxa_proj_corr) / cpl

    # CPL máximo
    cpl_max_obs = (product_value * taxa_proj_obs) / min_roas
    cpl_max_corr = (product_value * taxa_proj_corr) / min_roas

    # Margem
    margem_obs = ((cpl_max_obs - cpl) / cpl_max_obs) * 100
    margem_corr = ((cpl_max_corr - cpl) / cpl_max_corr) * 100

    # Ação recomendada
    acao_obs = "Escalar" if margem_obs > 50 else ("Manter" if margem_obs >= 0 else "Reduzir")
    acao_corr = "Escalar" if margem_corr > 50 else ("Manter" if margem_corr >= 0 else "Reduzir")

    print("\n" + "-"*80)
    print("COMPARAÇÃO DE MÉTRICAS:")
    print("-"*80)
    print(f"{'Métrica':<30} {'Observado':<20} {'Corrigido':<20} {'Δ':<15}")
    print("-"*80)
    print(f"{'Taxa Proj. (%)':<30} {taxa_proj_obs*100:<20.2f} {taxa_proj_corr*100:<20.2f} {(taxa_proj_corr-taxa_proj_obs)*100:+.2f}pp")
    print(f"{'ROAS Proj. (x)':<30} {roas_obs:<20.2f} {roas_corr:<20.2f} {roas_corr-roas_obs:+.2f}x")
    print(f"{'CPL Máx (R$)':<30} {cpl_max_obs:<20.2f} {cpl_max_corr:<20.2f} {cpl_max_corr-cpl_max_obs:+.2f}")
    print(f"{'Margem (%)':<30} {margem_obs:<20.1f} {margem_corr:<20.1f} {margem_corr-margem_obs:+.1f}pp")
    print(f"{'Ação':<30} {acao_obs:<20} {acao_corr:<20} {'⚠️ MUDOU' if acao_obs != acao_corr else '✓ Igual'}")
    print("-"*80)

    if acao_obs != acao_corr:
        print("\n🚨 ALERTA: Recomendação mudou de '{}' para '{}'!".format(acao_obs, acao_corr))

    print(f"\n💡 INSIGHT: Com taxas corrigidas, ROAS aumenta {((roas_corr/roas_obs - 1) * 100):.1f}%")
    print(f"             Margem aumenta {margem_corr - margem_obs:.1f} pontos percentuais")


def generate_corrected_config(corrected_rates: Dict[str, float], recall: float, correction_factor: float) -> str:
    """
    Gera código Python para novo BUSINESS_CONFIG

    Returns:
        String com código Python formatado
    """
    config_code = f'''# Configuração de negócio - TAXAS CORRIGIDAS POR RECALL
BUSINESS_CONFIG = {{
    "product_value": 2027.38,  # Baseado em análise de 1.970 vendas DevClub desde 01/03/2025
    "min_roas": 2.0,

    # TAXAS DE CONVERSÃO CORRIGIDAS
    # Recall do matching: {recall:.1%} ({total_observed_conversions} conversões observadas / {REAL_DEVCLUB_SALES} vendas reais)
    # Fator de correção aplicado: {correction_factor:.3f}x
    #
    # IMPORTANTE: Estas taxas refletem conversões REAIS estimadas, não apenas as capturadas pelo matching.
    # O matching por email/telefone captura apenas ~{recall:.1%} das conversões devido a:
    #   - Emails diferentes entre pesquisa e compra
    #   - Telefones inválidos/incomparáveis
    #   - Dados ausentes
    #
    # Método de correção: Taxa corrigida = Taxa observada / Recall
    #
    "conversion_rates": {{
'''

    for decil, rate in corrected_rates.items():
        obs_rate = observed_rates[decil]
        config_code += f'        "{decil}": {rate:.6f},  # {rate*100:.2f}% (era {obs_rate*100:.2f}% observado, +{(rate-obs_rate)*100:.2f}pp)\n'

    config_code += '''    }
}
'''

    return config_code


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ANÁLISE DE RECALL E CORREÇÃO DE TAXAS DE CONVERSÃO")
    print("="*80)

    # Calcular recall
    recall, correction_factor = calculate_recall()

    print(f"\n📊 DADOS DO MATCHING:")
    print(f"  Conversões observadas (treino): {training_conversions:,}")
    print(f"  Conversões observadas (teste): {test_conversions:,}")
    print(f"  Total conversões observadas: {total_observed_conversions:,}")
    print(f"  Vendas reais DevClub: {REAL_DEVCLUB_SALES:,}")

    print(f"\n🎯 RECALL DO MATCHING:")
    print(f"  Recall: {recall:.1%} ({total_observed_conversions}/{REAL_DEVCLUB_SALES})")
    print(f"  Fator de correção: {correction_factor:.3f}x")

    print(f"\n⚠️  INTERPRETAÇÃO:")
    print(f"  O matching captura apenas {recall:.1%} das conversões reais!")
    print(f"  Para cada conversão observada, existem ~{correction_factor:.1f} conversões reais.")

    # Calcular taxas corrigidas
    corrected_rates = calculate_corrected_rates(observed_rates, correction_factor)

    # Análise de impacto
    print(f"\n📈 IMPACTO DA CORREÇÃO NAS TAXAS POR DECIL:")
    print("="*80)

    impact_df = analyze_impact(observed_rates, corrected_rates)
    print(impact_df.to_string(index=False))

    # Simulação de impacto em métricas UTM
    simulate_utm_impact(correction_factor)

    # Gerar novo config
    print("\n" + "="*80)
    print("NOVO BUSINESS_CONFIG (COPIAR PARA meta_config.py)")
    print("="*80)
    new_config = generate_corrected_config(corrected_rates, recall, correction_factor)
    print(new_config)

    # Salvar em arquivo
    output_file = Path(__file__).parent / "BUSINESS_CONFIG_CORRECTED.py"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# " + "="*78 + "\n")
        f.write("# BUSINESS_CONFIG CORRIGIDO POR RECALL\n")
        f.write("# Gerado automaticamente por recall_correction_analysis.py\n")
        f.write("# " + "="*78 + "\n\n")
        f.write(new_config)

    print(f"\n✅ Novo config salvo em: {output_file}")

    print("\n" + "="*80)
    print("PRÓXIMOS PASSOS:")
    print("="*80)
    print("1. Revisar as taxas corrigidas acima")
    print("2. Fazer backup do meta_config.py atual:")
    print("   cp V2/api/meta_config.py V2/api/meta_config.py.backup")
    print("3. Substituir BUSINESS_CONFIG em meta_config.py pelo código acima")
    print("4. Re-gerar planilha de análise UTM para ver impacto nas recomendações")
    print("5. Validar que campanhas 'Manter' que deveriam ser 'Escalar' mudaram")
    print("="*80)
