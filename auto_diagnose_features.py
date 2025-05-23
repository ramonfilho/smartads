#!/usr/bin/env python
"""
Script de diagnóstico automático que detecta a estrutura correta dos diretórios.
"""

import os
import pandas as pd
import glob

# Configuração base
PROJECT_ROOT = "/Users/ramonmoreira/Desktop/smart_ads/data"

def find_train_files(base_path, version):
    """Encontra todos os arquivos train.csv em uma versão."""
    pattern = os.path.join(base_path, version, "**/train.csv")
    files = glob.glob(pattern, recursive=True)
    
    # Organizar por ordem de processamento
    stages_order = ['01_split', '02_preprocessed', '03_feature', '04_feature', '05_feature']
    
    def get_stage_priority(filepath):
        for i, stage in enumerate(stages_order):
            if stage in filepath:
                return i
        return 999
    
    files_sorted = sorted(files, key=get_stage_priority)
    return files_sorted

def analyze_file(filepath):
    """Analisa um arquivo e retorna estatísticas."""
    try:
        df = pd.read_csv(filepath)
        
        # Contar tipos de features
        text_cols = [col for col in df.columns if '¿' in col or 'Cuando' in col or 'Déjame' in col]
        nlp_cols = [col for col in df.columns if any(x in col.lower() for x in ['tfidf', 'bow', 'sentiment', 'length', 'count'])]
        numeric_cols = len(df.select_dtypes(include=['float64', 'int64']).columns)
        
        stats = {
            'path': filepath.replace(PROJECT_ROOT + '/', ''),
            'rows': len(df),
            'cols': len(df.columns),
            'text_cols': len(text_cols),
            'nlp_features': len(nlp_cols),
            'numeric_cols': numeric_cols,
            'has_target': 'target' in df.columns,
            'has_lancamento': 'lançamento' in df.columns
        }
        
        if 'target' in df.columns:
            stats['conversion_rate'] = df['target'].mean()
            
        return stats
    except Exception as e:
        return {'path': filepath, 'error': str(e)}

def main():
    print("="*80)
    print("DIAGNÓSTICO AUTOMÁTICO DE FEATURES - V4 vs V5")
    print("="*80)
    
    # Encontrar arquivos V4
    print("\n🔍 Procurando arquivos V4...")
    v4_files = find_train_files(PROJECT_ROOT, "V4")
    print(f"   Encontrados: {len(v4_files)} arquivos")
    
    # Encontrar arquivos V5
    print("\n🔍 Procurando arquivos V5...")
    v5_files = find_train_files(PROJECT_ROOT, "V5")
    print(f"   Encontrados: {len(v5_files)} arquivos")
    
    # Analisar V4
    print("\n📊 ANÁLISE V4 (Pipeline Antigo):")
    print("-"*80)
    v4_stats = []
    for filepath in v4_files:
        stats = analyze_file(filepath)
        v4_stats.append(stats)
        if 'error' not in stats:
            print(f"\n📁 {stats['path']}")
            print(f"   Linhas: {stats['rows']:,} | Colunas: {stats['cols']}")
            print(f"   Features NLP: {stats['nlp_features']} | Taxa conversão: {stats.get('conversion_rate', 'N/A')}")
            print(f"   Tem 'lançamento': {'✅' if stats['has_lancamento'] else '❌'}")
    
    # Analisar V5
    print("\n\n📊 ANÁLISE V5 (Pipeline Novo):")
    print("-"*80)
    v5_stats = []
    for filepath in v5_files:
        stats = analyze_file(filepath)
        v5_stats.append(stats)
        if 'error' not in stats:
            print(f"\n📁 {stats['path']}")
            print(f"   Linhas: {stats['rows']:,} | Colunas: {stats['cols']}")
            print(f"   Features NLP: {stats['nlp_features']} | Taxa conversão: {stats.get('conversion_rate', 'N/A')}")
            print(f"   Tem 'lançamento': {'✅' if stats['has_lancamento'] else '❌'}")
    
    # Comparação final
    print("\n\n🔄 COMPARAÇÃO FINAL V4 vs V5:")
    print("-"*80)
    
    # Encontrar os datasets finais (com mais colunas)
    v4_final = max([s for s in v4_stats if 'error' not in s], key=lambda x: x['cols'])
    v5_final = max([s for s in v5_stats if 'error' not in s], key=lambda x: x['cols'])
    
    print(f"\n📊 Dataset Final V4: {v4_final['path']}")
    print(f"   Features: {v4_final['cols']}")
    print(f"   Features NLP: {v4_final['nlp_features']}")
    
    print(f"\n📊 Dataset Final V5: {v5_final['path']}")
    print(f"   Features: {v5_final['cols']}")
    print(f"   Features NLP: {v5_final['nlp_features']}")
    
    print(f"\n📉 DIFERENÇA:")
    print(f"   Features perdidas: {v4_final['cols'] - v5_final['cols']}")
    print(f"   Features NLP perdidas: {v4_final['nlp_features'] - v5_final['nlp_features']}")
    
    # Analisar perda entre etapas V5
    print("\n\n📍 ONDE AS FEATURES SÃO PERDIDAS (V5):")
    print("-"*80)
    for i in range(len(v5_stats)-1):
        if 'error' not in v5_stats[i] and 'error' not in v5_stats[i+1]:
            diff = v5_stats[i+1]['cols'] - v5_stats[i]['cols']
            if diff != 0:
                print(f"\n{v5_stats[i]['path'].split('/')[-2]} → {v5_stats[i+1]['path'].split('/')[-2]}")
                print(f"   Mudança: {diff:+d} features ({v5_stats[i]['cols']} → {v5_stats[i+1]['cols']})")
                print(f"   NLP: {v5_stats[i]['nlp_features']} → {v5_stats[i+1]['nlp_features']}")

if __name__ == "__main__":
    main()