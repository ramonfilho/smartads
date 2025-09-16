import pandas as pd

# Verificar colunas UTM no arquivo de referência
filepath = '../data/devclub/[LF24] Leads.xlsx'
xl = pd.ExcelFile(filepath)
print('Abas disponíveis:', xl.sheet_names)

df = pd.read_excel(xl, sheet_name=xl.sheet_names[0])
print(f'\nTotal de colunas: {len(df.columns)}')
print('\nColunas relacionadas a UTM/Source/Medium/Campaign/Term:')
utm_cols = [col for col in df.columns if any(term in col.lower() for term in ['utm', 'source', 'medium', 'campaign', 'content', 'term'])]
for col in utm_cols:
    print(f'  - {col}')

print('\nTodas as colunas:')
for i, col in enumerate(df.columns, 1):
    print(f'{i:3}. {col}')