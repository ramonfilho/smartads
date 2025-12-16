import pandas as pd

excel_path = 'files/validation/resultados/validation_report_2025-11-27_to_2025-12-01.xlsx'

# Ler Performance por Campanha
df_perf = pd.read_excel(excel_path, sheet_name='Performance por Campanha', header=3)

# Ler Comparação Justa
df_comp = pd.read_excel(excel_path, sheet_name='Comparação Justa', header=3)

# Filtrar a campanha com LeadQualified
print('='*100)
print('CAMPANHA COM LeadQualified - COMPARAÇÃO ENTRE ABAS')
print('='*100)

# Performance por Campanha
perf_row = df_perf[df_perf['Campanha'].str.contains('S/ ABERTO', na=False)]
if len(perf_row) > 0:
    perf_row = perf_row.iloc[0]
    print('\n📊 ABA: Performance por Campanha')
    print(f"  Campanha: {perf_row['Campanha']}")
    print(f"  Tipo: {perf_row['Tipo de campanha']}")
    print(f"  Evento: {perf_row['Evento de conversão']}")
    print(f"  Leads: {perf_row['Leads']}")
    print(f"  LeadQualified: {perf_row.get('LeadQualified', 'N/A')}")
    print(f"  LeadQualifiedHighQuality: {perf_row.get('LeadQualifiedHighQuality', 'N/A')}")
    print(f"  Respostas pesquisa: {perf_row['Respostas pesquisa']}")
    print(f"  % de resposta: {perf_row['% de resposta']:.2f}%")
    print(f"  Vendas: {perf_row['Vendas']}")

# Comparação Justa
comp_row = df_comp[df_comp['Campanha'].str.contains('S/ ABERTO', na=False)]
if len(comp_row) > 0:
    comp_row = comp_row.iloc[0]
    print('\n📊 ABA: Comparação Justa')
    print(f"  Campanha: {comp_row['Campanha']}")
    print(f"  Grupo: {comp_row['Grupo']}")
    print(f"  Evento: {comp_row['Evento de conversão']}")
    print(f"  Leads: {comp_row['Leads']}")
    print(f"  LeadQualified: {comp_row.get('LeadQualified', 'N/A')}")
    print(f"  LeadQualifiedHighQuality: {comp_row.get('LeadQualifiedHighQuality', 'N/A')}")
    print(f"  Respostas pesquisa: {comp_row['Respostas pesquisa']}")
    print(f"  % de resposta: {comp_row['% de resposta']:.2f}%")
    print(f"  Vendas: {comp_row['Vendas']}")

print()
print('='*100)
print('ANÁLISE:')
print('='*100)
if len(perf_row) > 0 and len(comp_row) > 0:
    if perf_row['Leads'] != comp_row['Leads']:
        print(f"⚠️  DIVERGÊNCIA nos Leads: Performance={perf_row['Leads']}, Comparação={comp_row['Leads']}")
    pct_diff = abs(perf_row['% de resposta'] - comp_row['% de resposta'])
    if pct_diff > 0.01:
        print(f"⚠️  DIVERGÊNCIA na % resposta: Performance={perf_row['% de resposta']:.2f}%, Comparação={comp_row['% de resposta']:.2f}%")
    if perf_row['Leads'] == comp_row['Leads'] and pct_diff <= 0.01:
        print("✅ Dados consistentes entre as abas!")
