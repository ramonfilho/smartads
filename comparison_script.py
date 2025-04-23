#!/usr/bin/env python
"""
Script para comparar os resultados do processamento unificado com os resultados
do processamento original (script 6) e identificar incompatibilidades.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Caminhos específicos para comparação
NEW_DIR = "/Users/ramonmoreira/desktop/smart_ads/data/verification_test"
REF_DIR = "/Users/ramonmoreira/Desktop/smart_ads/data/02_processed_text_code6"
SUBSET = "validation"
OUTPUT_FILE = "comparison_report.md"

def load_dataframes():
    """
    Carrega os dataframes das duas fontes para comparação.
    
    Returns:
        Tuple com (df_new, df_ref)
    """
    # Carregar arquivo do novo processamento
    new_path = os.path.join(NEW_DIR, f"{SUBSET}.csv")
    if not os.path.exists(new_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {new_path}")
    
    df_new = pd.read_csv(new_path)
    print(f"Arquivo novo carregado: {new_path}, shape: {df_new.shape}")
    
    # Carregar arquivo de referência
    ref_path = os.path.join(REF_DIR, f"{SUBSET}.csv")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Arquivo de referência não encontrado: {ref_path}")
    
    df_ref = pd.read_csv(ref_path)
    print(f"Arquivo de referência carregado: {ref_path}, shape: {df_ref.shape}")
    
    return df_new, df_ref

def analyze_column_sets(df_new, df_ref):
    """
    Analisa os conjuntos de colunas entre os dois dataframes.
    
    Args:
        df_new: DataFrame do novo processamento
        df_ref: DataFrame de referência
        
    Returns:
        Dict com análise de colunas
    """
    cols_new = set(df_new.columns)
    cols_ref = set(df_ref.columns)
    
    cols_common = cols_new.intersection(cols_ref)
    cols_only_new = cols_new - cols_ref
    cols_only_ref = cols_ref - cols_new
    
    print(f"Análise de colunas:")
    print(f"- Total de colunas no novo: {len(cols_new)}")
    print(f"- Total de colunas na referência: {len(cols_ref)}")
    print(f"- Colunas em comum: {len(cols_common)}")
    print(f"- Colunas apenas no novo: {len(cols_only_new)}")
    print(f"- Colunas apenas na referência: {len(cols_only_ref)}")
    
    # Categorizar as colunas exclusivas
    ref_only_categories = categorize_columns(cols_only_ref)
    new_only_categories = categorize_columns(cols_only_new)
    
    print("\nCategorias de colunas exclusivas da referência:")
    for category, columns in ref_only_categories.items():
        print(f"- {category}: {len(columns)} colunas")
        if len(columns) <= 5:
            print(f"  Exemplos: {', '.join(list(columns))}")
        else:
            print(f"  Exemplos: {', '.join(list(columns)[:5])}...")
    
    print("\nCategorias de colunas exclusivas do novo:")
    for category, columns in new_only_categories.items():
        print(f"- {category}: {len(columns)} colunas")
        if len(columns) <= 5:
            print(f"  Exemplos: {', '.join(list(columns))}")
        else:
            print(f"  Exemplos: {', '.join(list(columns)[:5])}...")
    
    return {
        'common': cols_common,
        'only_new': cols_only_new,
        'only_ref': cols_only_ref,
        'ref_categories': ref_only_categories,
        'new_categories': new_only_categories
    }

def categorize_columns(column_set):
    """
    Categoriza colunas baseado em padrões comuns.
    
    Args:
        column_set: Conjunto de nomes de colunas
        
    Returns:
        Dict com categorias e suas colunas
    """
    categories = {
        'tfidf': set(),
        'sentiment': set(),
        'commitment': set(),
        'career': set(),
        'aspiration': set(),
        'motivation': set(),
        'temporal': set(),
        'original_text': set(),
        'other': set()
    }
    
    for col in column_set:
        col_lower = col.lower()
        if 'tfidf' in col_lower:
            categories['tfidf'].add(col)
        elif any(term in col_lower for term in ['sentiment', 'pos', 'neg', 'compound']):
            categories['sentiment'].add(col)
        elif any(term in col_lower for term in ['commitment', 'has_commitment']):
            categories['commitment'].add(col)
        elif any(term in col_lower for term in ['career', 'professional']):
            categories['career'].add(col)
        elif 'aspiration' in col_lower:
            categories['aspiration'].add(col)
        elif 'motivation' in col_lower:
            categories['motivation'].add(col)
        elif any(term in col_lower for term in ['hour', 'day', 'time', 'date']):
            categories['temporal'].add(col)
        elif 'original' in col_lower:
            categories['original_text'].add(col)
        else:
            categories['other'].add(col)
    
    # Remover categorias vazias
    return {k: v for k, v in categories.items() if v}

def compare_common_columns(df_new, df_ref, common_cols):
    """
    Compara os valores das colunas em comum entre os dataframes.
    
    Args:
        df_new: DataFrame do novo processamento
        df_ref: DataFrame de referência
        common_cols: Lista de colunas em comum
        
    Returns:
        Dict com resultados da comparação
    """
    print("\nComparando valores das colunas em comum...")
    
    # Garantir que os dataframes estejam alinhados pelos emails
    if 'email_norm' in common_cols:
        print("Alinhando dataframes pelo email_norm...")
        
        # Verificar duplicatas em email_norm
        dups_new = df_new['email_norm'].duplicated().sum()
        dups_ref = df_ref['email_norm'].duplicated().sum()
        
        if dups_new > 0 or dups_ref > 0:
            print(f"Atenção: Encontradas duplicatas em email_norm:")
            print(f"- Novo: {dups_new} duplicatas")
            print(f"- Referência: {dups_ref} duplicatas")
            
            # Remover duplicatas para viabilizar o merge
            if dups_new > 0:
                df_new = df_new.drop_duplicates(subset=['email_norm'])
            if dups_ref > 0:
                df_ref = df_ref.drop_duplicates(subset=['email_norm'])
        
        # Encontrar emails em comum
        emails_new = set(df_new['email_norm'].dropna())
        emails_ref = set(df_ref['email_norm'].dropna())
        emails_common = emails_new.intersection(emails_ref)
        
        print(f"Emails em comum: {len(emails_common)}")
        
        # Filtrar para manter apenas emails em comum
        df_new_filtered = df_new[df_new['email_norm'].isin(emails_common)]
        df_ref_filtered = df_ref[df_ref['email_norm'].isin(emails_common)]
        
        print(f"Registros após filtragem:")
        print(f"- Novo: {len(df_new_filtered)}")
        print(f"- Referência: {len(df_ref_filtered)}")
        
        # Verificar tipos de dados das colunas em comum
        dtype_diffs = []
        for col in common_cols:
            if col in df_new_filtered.columns and col in df_ref_filtered.columns:
                dtype_new = df_new_filtered[col].dtype
                dtype_ref = df_ref_filtered[col].dtype
                if dtype_new != dtype_ref:
                    dtype_diffs.append((col, dtype_new, dtype_ref))
        
        if dtype_diffs:
            print("\nDiferenças de tipos de dados:")
            for col, dtype_new, dtype_ref in dtype_diffs:
                print(f"- {col}: Novo={dtype_new}, Ref={dtype_ref}")
        
        # Analisar diferenças numéricas
        numeric_diffs = []
        for col in tqdm(common_cols, desc="Analisando colunas numéricas"):
            if col in df_new_filtered.columns and col in df_ref_filtered.columns:
                # Verificar se ambas são numéricas
                try:
                    vals_new = pd.to_numeric(df_new_filtered[col], errors='coerce')
                    vals_ref = pd.to_numeric(df_ref_filtered[col], errors='coerce')
                    
                    # Calcular diferenças
                    if vals_new.notna().sum() > 0 and vals_ref.notna().sum() > 0:
                        mean_new = vals_new.mean()
                        mean_ref = vals_ref.mean()
                        std_new = vals_new.std()
                        std_ref = vals_ref.std()
                        
                        # Calcular correlação
                        # Usar merge para alinhar os valores
                        temp_df = pd.DataFrame({
                            'email': df_new_filtered['email_norm'],
                            'new': vals_new
                        })
                        temp_df = temp_df.merge(
                            pd.DataFrame({
                                'email': df_ref_filtered['email_norm'],
                                'ref': vals_ref
                            }),
                            on='email',
                            how='inner'
                        )
                        
                        # Calcular correlação apenas para valores não-nulos
                        mask = temp_df['new'].notna() & temp_df['ref'].notna()
                        if mask.sum() >= 10:  # Pelo menos 10 valores para correlação
                            corr = temp_df.loc[mask, ['new', 'ref']].corr().iloc[0, 1]
                        else:
                            corr = np.nan
                        
                        # Determinar consistência baseada na correlação
                        if np.isnan(corr):
                            consistency = "Indeterminada"
                        elif corr > 0.9:
                            consistency = "Alta"
                        elif corr > 0.7:
                            consistency = "Média"
                        elif corr > 0.5:
                            consistency = "Baixa"
                        else:
                            consistency = "Muito Baixa"
                        
                        numeric_diffs.append({
                            'column': col,
                            'mean_new': mean_new,
                            'mean_ref': mean_ref,
                            'std_new': std_new,
                            'std_ref': std_ref,
                            'correlation': corr,
                            'consistency': consistency
                        })
                except Exception as e:
                    continue
        
        # Criar DataFrame com as diferenças
        diff_df = pd.DataFrame(numeric_diffs)
        
        # Categorizar resultados
        if len(diff_df) > 0:
            consistent_cols = diff_df[diff_df['consistency'].isin(['Alta', 'Média'])]['column'].tolist()
            inconsistent_cols = diff_df[diff_df['consistency'].isin(['Baixa', 'Muito Baixa'])]['column'].tolist()
            
            print(f"\nColunas consistentes: {len(consistent_cols)}")
            print(f"Colunas inconsistentes: {len(inconsistent_cols)}")
            
            if inconsistent_cols:
                print("\nDetalhes das colunas inconsistentes:")
                inconsistent_diff = diff_df[diff_df['column'].isin(inconsistent_cols)]
                for _, row in inconsistent_diff.iterrows():
                    print(f"- {row['column']}: Corr={row['correlation']:.3f}, Consistência={row['consistency']}")
                    print(f"  Novo: Média={row['mean_new']:.4f}, Std={row['std_new']:.4f}")
                    print(f"  Ref: Média={row['mean_ref']:.4f}, Std={row['std_ref']:.4f}")
            
            return {
                'dtype_diffs': dtype_diffs,
                'numeric_diffs': diff_df,
                'consistent_cols': consistent_cols,
                'inconsistent_cols': inconsistent_cols
            }
    
    print("Coluna email_norm não encontrada em ambos os dataframes. Análise detalhada não é possível.")
    return None

def write_comparison_report(results, output_file):
    """
    Escreve um relatório detalhado de comparação.
    
    Args:
        results: Dicionário com resultados da comparação
        output_file: Caminho para salvar o relatório
    """
    with open(output_file, 'w') as f:
        f.write("# RELATÓRIO DE COMPARAÇÃO DE DATASETS\n\n")
        
        f.write("## Análise de Colunas\n\n")
        f.write(f"- Colunas em comum: {len(results['column_analysis']['common'])}\n")
        f.write(f"- Colunas apenas no novo: {len(results['column_analysis']['only_new'])}\n")
        f.write(f"- Colunas apenas na referência: {len(results['column_analysis']['only_ref'])}\n\n")
        
        f.write("### Categorias de colunas exclusivas da referência:\n")
        for category, columns in results['column_analysis']['ref_categories'].items():
            f.write(f"- {category}: {len(columns)} colunas\n")
        
        f.write("\n### Categorias de colunas exclusivas do novo:\n")
        for category, columns in results['column_analysis']['new_categories'].items():
            f.write(f"- {category}: {len(columns)} colunas\n")
        
        f.write("\n## Análise de Valores\n\n")
        
        if results.get('value_analysis'):
            f.write(f"### Tipos de Dados\n")
            for col, dtype_new, dtype_ref in results['value_analysis'].get('dtype_diffs', []):
                f.write(f"- {col}: Novo={dtype_new}, Ref={dtype_ref}\n")
            
            f.write("\n### Consistência de Colunas\n")
            f.write(f"- Colunas consistentes: {len(results['value_analysis'].get('consistent_cols', []))}\n")
            f.write(f"- Colunas inconsistentes: {len(results['value_analysis'].get('inconsistent_cols', []))}\n\n")
            
            if results['value_analysis'].get('inconsistent_cols'):
                f.write("### Detalhes das Colunas Inconsistentes\n")
                inconsistent_cols = results['value_analysis'].get('inconsistent_cols', [])
                inconsistent_diff = results['value_analysis']['numeric_diffs'][
                    results['value_analysis']['numeric_diffs']['column'].isin(inconsistent_cols)
                ]
                
                for _, row in inconsistent_diff.iterrows():
                    f.write(f"- {row['column']}:\n")
                    f.write(f"  - Correlação: {row['correlation']:.3f}\n")
                    f.write(f"  - Consistência: {row['consistency']}\n")
                    f.write(f"  - Novo: Média={row['mean_new']:.4f}, Std={row['std_new']:.4f}\n")
                    f.write(f"  - Ref: Média={row['mean_ref']:.4f}, Std={row['std_ref']:.4f}\n\n")
        
        f.write("\n## Recomendações\n\n")
        
        # Adicionar recomendações baseadas na análise
        if results.get('value_analysis') and results['value_analysis'].get('inconsistent_cols'):
            f.write("1. Verificar as implementações das seguintes features inconsistentes:\n")
            for col in results['value_analysis'].get('inconsistent_cols', [])[:5]:
                f.write(f"   - {col}\n")
            
            f.write("\n2. Comparar os parâmetros utilizados em ambas as abordagens\n")
            f.write("3. Verificar se as inconsistências impactam o desempenho do modelo\n")
        else:
            f.write("1. As diferenças principais estão nas colunas presentes, não nos valores\n")
            f.write("2. Avaliar se as colunas ausentes em um dos processamentos são necessárias\n")
        
        f.write("\n## Conclusão\n\n")
        
        total_problems = len(results.get('value_analysis', {}).get('inconsistent_cols', []))
        if total_problems > 10:
            f.write("Existe um número significativo de inconsistências entre os datasets. Recomenda-se uma revisão detalhada das implementações.\n")
        elif total_problems > 0:
            f.write("Existem algumas inconsistências que precisam ser corrigidas, mas a maioria das colunas em comum está consistente.\n")
        else:
            f.write("Os datasets estão razoavelmente consistentes em termos de valores. As principais diferenças estão nas colunas presentes em cada um.\n")

def main():
    print(f"=== COMPARAÇÃO DE RESULTADOS: NOVO vs REFERÊNCIA ===")
    print(f"Novo diretório: {NEW_DIR}")
    print(f"Diretório de referência: {REF_DIR}")
    print(f"Conjunto de dados: {SUBSET}")
    
    # Carregar dataframes
    df_new, df_ref = load_dataframes()
    
    # Analisar conjuntos de colunas
    column_analysis = analyze_column_sets(df_new, df_ref)
    
    # Comparar valores nas colunas em comum
    value_analysis = compare_common_columns(df_new, df_ref, column_analysis['common'])
    
    # Compilar resultados
    results = {
        'column_analysis': column_analysis,
        'value_analysis': value_analysis
    }
    
    # Escrever relatório
    write_comparison_report(results, OUTPUT_FILE)
    print(f"\nRelatório de comparação salvo em {OUTPUT_FILE}")

if __name__ == "__main__":
    main()