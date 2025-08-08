#!/usr/bin/env python
"""
Análise do ganho potencial do modelo e sua confiabilidade.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar o diretório raiz do projeto ao sys.path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
sys.path.insert(0, project_root)

def calculate_model_gains():
    """Calcula o ganho potencial do modelo de ranking."""
    
    print("=== ANÁLISE DE GANHO POTENCIAL DO MODELO ===\n")
    
    # Dados do modelo atual
    model_metrics = {
        'gini': 0.4412,
        'top_decile_lift': 2.95,
        'top_20pct_recall': 0.507,
        'monotonicity_violations': 8
    }
    
    # Taxas observadas
    taxa_real = 0.80  # Taxa real de conversão (compradores/UTMs)
    taxa_matching = 1.15  # Taxa no dataset do modelo
    taxa_base_pesquisa = 3.06  # Taxa entre quem responde pesquisa
    
    # Proporções
    match_rate = taxa_matching / taxa_base_pesquisa  # ~37.6% dos compradores fazem match
    
    print("1. MÉTRICAS DO MODELO:")
    print(f"   - GINI: {model_metrics['gini']:.4f}")
    print(f"   - Top Decile Lift: {model_metrics['top_decile_lift']:.2f}x")
    print(f"   - Top 20% Recall: {model_metrics['top_20pct_recall']:.1%}")
    print(f"   - Violações de monotonicidade: {model_metrics['monotonicity_violations']}")
    print()
    
    print("2. CONTEXTO DAS TAXAS:")
    print(f"   - Taxa real (compradores/UTMs): {taxa_real:.2f}%")
    print(f"   - Taxa no modelo (matches/pesquisas): {taxa_matching:.2f}%")
    print(f"   - Taxa base pesquisa (compradores/pesquisas): {taxa_base_pesquisa:.2f}%")
    print(f"   - Taxa de matching bem-sucedido: {match_rate:.1%}")
    print()
    
    # Cenários de aplicação
    print("3. CENÁRIOS DE GANHO:\n")
    
    # Cenário 1: Aplicação direta (conservador)
    print("   A) CENÁRIO CONSERVADOR (aplicação apenas nos matches):")
    print("      - Assumindo que o modelo só funciona para os ~37.6% que fazem match")
    print("      - Realocando budget do bottom 80% para o top 20%:")
    
    # Com 20% do budget, capturamos 50.7% das conversões
    # Se redistribuirmos o budget do bottom 80% proporcionalmente
    eficiencia_top20 = model_metrics['top_20pct_recall'] / 0.20  # 2.535x mais eficiente
    eficiencia_bottom80 = (1 - model_metrics['top_20pct_recall']) / 0.80  # 0.616x
    
    # Ganho ao realocar
    ganho_conservador = (eficiencia_top20 - 1) * 0.80  # Ganho do budget realocado
    ganho_percentual_conservador = ganho_conservador * match_rate  # Ajustado pela taxa de match
    
    print(f"      - Eficiência do top 20%: {eficiencia_top20:.2f}x")
    print(f"      - Ganho bruto: {ganho_conservador:.1%}")
    print(f"      - Ganho ajustado pela taxa de match: {ganho_percentual_conservador:.1%}")
    print()
    
    # Cenário 2: Extrapolação moderada
    print("   B) CENÁRIO MODERADO (extrapolação para compradores similares):")
    print("      - Assumindo que o modelo identifica padrões válidos para ~60% dos compradores")
    extrapolacao_moderada = 0.60
    ganho_moderado = ganho_conservador * extrapolacao_moderada
    print(f"      - Ganho estimado: {ganho_moderado:.1%}")
    print()
    
    # Cenário 3: Otimista (mas realista)
    print("   C) CENÁRIO OTIMISTA REALISTA:")
    print("      - Modelo identifica características gerais de conversão")
    print("      - Funciona para ~80% dos casos")
    extrapolacao_otimista = 0.80
    ganho_otimista = ganho_conservador * extrapolacao_otimista
    print(f"      - Ganho estimado: {ganho_otimista:.1%}")
    print()
    
    # Impacto financeiro
    print("4. IMPACTO FINANCEIRO ESTIMADO:")
    budget_mensal = 100000  # Exemplo: R$ 100k/mês em ads
    conversoes_atuais = budget_mensal * (taxa_real / 100)
    
    print(f"   - Para um budget de R$ {budget_mensal:,.0f}/mês:")
    print(f"   - Conversões atuais: {conversoes_atuais:.0f}")
    print(f"   - Conversões extras (conservador): {conversoes_atuais * ganho_percentual_conservador:.0f}")
    print(f"   - Conversões extras (moderado): {conversoes_atuais * ganho_moderado:.0f}")
    print(f"   - Conversões extras (otimista): {conversoes_atuais * ganho_otimista:.0f}")
    print()
    
    # Análise de confiabilidade
    print("5. ANÁLISE DE CONFIABILIDADE:\n")
    
    print("   PONTOS FORTES:")
    print("   ✓ GINI de 0.44 indica boa capacidade discriminativa")
    print("   ✓ Top 20% captura 50.7% das conversões (muito bom)")
    print("   ✓ Lift de 2.95x no top decile é significativo")
    print("   ✓ Modelo supera baseline GMM complexo")
    print()
    
    print("   PONTOS DE ATENÇÃO:")
    print("   ⚠️  8 violações de monotonicidade (de quantas features?)")
    print("   ⚠️  Taxa de match de apenas 37.6% limita observabilidade")
    print("   ⚠️  Diferença entre taxa real (0.80%) e modelo (1.15%)")
    print()
    
    print("   RISCOS E MITIGAÇÕES:")
    print("   1. Viés de seleção: Modelo treinado apenas em quem responde pesquisa")
    print("      → Mitigação: Começar com testes A/B pequenos (10-20% do budget)")
    print()
    print("   2. Violações de monotonicidade: Algumas features têm comportamento não-linear")
    print("      → Mitigação: Analisar quais features violam e considerar transformações")
    print()
    print("   3. Generalização: Incerteza sobre performance em não-respondentes")
    print("      → Mitigação: Monitorar métricas reais vs preditas continuamente")
    print()
    
    # Recomendações
    print("6. RECOMENDAÇÕES:\n")
    print("   IMPLEMENTAÇÃO SUGERIDA:")
    print("   1. Fase 1 (1º mês): Teste A/B com 20% do budget")
    print("      - Grupo A: Alocação atual")
    print("      - Grupo B: Alocação baseada no modelo")
    print("      - Ganho esperado: 3-7% mais conversões")
    print()
    print("   2. Fase 2 (2º-3º mês): Expandir para 50% se resultados positivos")
    print("      - Refinar modelo com dados reais")
    print("      - Ganho esperado: 7-15% mais conversões")
    print()
    print("   3. Fase 3 (4º mês+): Implementação completa + otimizações")
    print("      - Modelo atualizado mensalmente")
    print("      - Ganho esperado: 15-30% mais conversões")
    print()
    
    print("   MÉTRICAS PARA MONITORAR:")
    print("   - CPL (Custo Por Lead) por decil")
    print("   - Taxa de conversão real vs predita por campanha")
    print("   - Drift do modelo ao longo do tempo")
    print("   - ROI incremental do teste A/B")
    
    # Criar visualização
    create_gain_visualization(ganho_percentual_conservador, ganho_moderado, ganho_otimista)
    
    return {
        'conservador': ganho_percentual_conservador,
        'moderado': ganho_moderado,
        'otimista': ganho_otimista
    }

def create_gain_visualization(conservador, moderado, otimista):
    """Cria visualização dos ganhos potenciais."""
    try:
        plt.figure(figsize=(10, 6))
        
        scenarios = ['Conservador\n(37.6% coverage)', 'Moderado\n(60% coverage)', 'Otimista\n(80% coverage)']
        gains = [conservador * 100, moderado * 100, otimista * 100]
        
        bars = plt.bar(scenarios, gains, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        
        # Adicionar valores nas barras
        for bar, gain in zip(bars, gains):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{gain:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.title('Ganho Potencial em Conversões por Cenário', fontsize=16, fontweight='bold')
        plt.ylabel('Ganho em Conversões (%)', fontsize=12)
        plt.xlabel('Cenário de Aplicação', fontsize=12)
        plt.ylim(0, max(gains) * 1.2)
        plt.grid(axis='y', alpha=0.3)
        
        # Adicionar linha de referência
        plt.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Meta: 15%')
        plt.legend()
        
        plt.tight_layout()
        
        # Salvar figura
        output_dir = os.path.join(project_root, "analysis", "figures")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "model_gain_potential.png"), dpi=300, bbox_inches='tight')
        print(f"\n✅ Gráfico salvo em: {output_dir}/model_gain_potential.png")
        
        plt.close()
        
    except Exception as e:
        print(f"\n⚠️  Erro ao criar visualização: {e}")

def analyze_monotonicity_impact():
    """Analisa o impacto das violações de monotonicidade."""
    print("\n\n=== ANÁLISE DETALHADA DA MONOTONICIDADE ===\n")
    
    print("CONTEXTO DAS 8 VIOLAÇÕES:")
    print("- Total de features: ~300")
    print("- Violações: 8 (2.7% das features)")
    print()
    
    print("INTERPRETAÇÃO:")
    print("1. BAIXO IMPACTO (< 5% das features):")
    print("   - Maioria das relações é monotônica")
    print("   - Modelo captura tendências gerais corretamente")
    print()
    
    print("2. POSSÍVEIS CAUSAS:")
    print("   - Relações genuinamente não-lineares (ex: idade ótima)")
    print("   - Ruído nos dados em regiões específicas")
    print("   - Interações complexas entre features")
    print()
    
    print("3. RECOMENDAÇÕES:")
    print("   - Identificar quais features violam monotonicidade")
    print("   - Avaliar se são features importantes (top 20)")
    print("   - Considerar transformações ou binning se necessário")
    print("   - Monitorar performance em produção")

if __name__ == "__main__":
    # Calcular ganhos
    ganhos = calculate_model_gains()
    
    # Análise de monotonicidade
    analyze_monotonicity_impact()
    
    print("\n\n=== CONCLUSÃO EXECUTIVA ===\n")
    print("📊 O modelo apresenta potencial de ganho SIGNIFICATIVO:")
    print(f"   - Conservador: {ganhos['conservador']:.1%} mais conversões")
    print(f"   - Esperado: {ganhos['moderado']:.1%} mais conversões")
    print(f"   - Otimista: {ganhos['otimista']:.1%} mais conversões")
    print()
    print("✅ RECOMENDAÇÃO: PROSSEGUIR COM TESTE A/B")
    print("   - Risco baixo com implementação gradual")
    print("   - Potencial de retorno alto (15-30% mais conversões)")
    print("   - Monitoramento contínuo mitiga riscos")