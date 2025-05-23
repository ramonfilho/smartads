#!/usr/bin/env python
"""
Script de diagnóstico para identificar onde e quais features foram perdidas
na transição do pipeline antigo para o novo.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Configuração de caminhos
PROJECT_ROOT = "/Users/ramonmoreira/Desktop/smart_ads"
OLD_DATA_PATH = "/Users/ramonmoreira/Desktop/smart_ads/data/V4"  # Pipeline antigo
NEW_DATA_PATH = "/Users/ramonmoreira/Desktop/smart_ads/data/V5"  # Pipeline novo

def load_dataset_safely(file_path, description):
    """Carrega dataset com tratamento de erro."""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✅ Carregado: {description}")
            print(f"   Arquivo: {file_path}")
            print(f"   Shape: {df.shape}")
            return df
        else:
            print(f"❌ Não encontrado: {description}")
            print(f"   Arquivo: {file_path}")
            return None
    except Exception as e:
        print(f"❌ Erro ao carregar {description}: {str(e)}")
        return None

def analyze_columns(df, stage_name):
    """Analisa as colunas de um DataFrame."""
    if df is None:
        return None
    
    analysis = {
        'stage': stage_name,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'column_types': df.dtypes.value_counts().to_dict(),
        'has_target': 'target' in df.columns
    }
    
    if 'target' in df.columns:
        analysis['conversion_rate'] = df['target'].mean()
        analysis['n_positive'] = df['target'].sum()
    
    # Categorizar colunas
    analysis['text_columns'] = []
    analysis['numeric_columns'] = []
    analysis['categorical_columns'] = []
    analysis['binary_columns'] = []
    analysis['nlp_features'] = []
    
    for col in df.columns:
        # Identificar colunas de texto originais
        if any(keyword in col.lower() for keyword in ['mensaje', 'esperas', 'cambiará', '¿qué', '¿cuál']):
            analysis['text_columns'].append(col)
        
        # Identificar features NLP geradas
        elif any(prefix in col for prefix in ['tfidf_', 'bow_', 'sentiment_', 'length_', 'word_count_']):
            analysis['nlp_features'].append(col)
        
        # Identificar colunas numéricas
        elif df[col].dtype in ['int64', 'float64']:
            unique_values = df[col].nunique()
            if unique_values == 2:
                analysis['binary_columns'].append(col)
            else:
                analysis['numeric_columns'].append(col)
        
        # Identificar colunas categóricas
        else:
            analysis['categorical_columns'].append(col)
    
    return analysis

def compare_datasets(old_analysis, new_analysis):
    """Compara dois conjuntos de dados e identifica diferenças."""
    if old_analysis is None or new_analysis is None:
        return None
    
    old_cols = set(old_analysis['columns'])
    new_cols = set(new_analysis['columns'])
    
    comparison = {
        'columns_lost': sorted(list(old_cols - new_cols)),
        'columns_added': sorted(list(new_cols - old_cols)),
        'columns_kept': sorted(list(old_cols & new_cols)),
        'n_columns_lost': len(old_cols - new_cols),
        'n_columns_added': len(new_cols - old_cols),
        'n_columns_kept': len(old_cols & new_cols),
        'total_change': len(old_cols) - len(new_cols)
    }
    
    # Análise por tipo de feature
    comparison['nlp_features_lost'] = [col for col in comparison['columns_lost'] if any(prefix in col for prefix in ['tfidf_', 'bow_', 'sentiment_'])]
    comparison['categorical_features_lost'] = [col for col in comparison['columns_lost'] if col in old_analysis['categorical_columns']]
    comparison['numeric_features_lost'] = [col for col in comparison['columns_lost'] if col in old_analysis['numeric_columns']]
    
    return comparison

def diagnose_pipeline():
    """Executa diagnóstico completo do pipeline."""
    print("="*80)
    print("DIAGNÓSTICO DE PERDA DE FEATURES - SMART ADS")
    print("="*80)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Definir estágios do pipeline
    stages = {
        '01_split': '01_split/train.csv',
        '02_preprocessed': '02_preprocessed/train.csv',
        '03_feature_eng_1': '03_feature_engineering_1/train.csv',
        '04_feature_eng_2': '04_feature_engineering_2/train.csv'
    }
    
    # Carregar e analisar cada estágio (V5 - novo pipeline)
    analyses = {}
    dataframes = {}
    
    print("\n📊 ANÁLISE POR ESTÁGIO DO PIPELINE (V5 - NOVO)")
    print("-"*80)
    
    for stage_name, file_path in stages.items():
        full_path = os.path.join(NEW_DATA_PATH, file_path)
        df = load_dataset_safely(full_path, f"{stage_name} (V5)")
        if df is not None:
            dataframes[stage_name] = df
            analyses[stage_name] = analyze_columns(df, stage_name)
            
            print(f"\n📌 {stage_name}:")
            print(f"   Linhas: {analyses[stage_name]['n_rows']:,}")
            print(f"   Colunas: {analyses[stage_name]['n_columns']}")
            if analyses[stage_name].get('conversion_rate') is not None:
                print(f"   Taxa conversão: {analyses[stage_name]['conversion_rate']:.4f}")
            else:
                print(f"   Taxa conversão: N/A")
            print(f"   Colunas de texto: {len(analyses[stage_name]['text_columns'])}")
            print(f"   Features NLP: {len(analyses[stage_name]['nlp_features'])}")
            print(f"   Features numéricas: {len(analyses[stage_name]['numeric_columns'])}")
            print(f"   Features categóricas: {len(analyses[stage_name]['categorical_columns'])}")
    
    # Comparar estágios consecutivos
    print("\n\n🔍 ANÁLISE DE MUDANÇAS ENTRE ESTÁGIOS")
    print("-"*80)
    
    stage_pairs = [
        ('01_split', '02_preprocessed'),
        ('02_preprocessed', '03_feature_eng_1'),
        ('03_feature_eng_1', '04_feature_eng_2')
    ]
    
    for stage1, stage2 in stage_pairs:
        if stage1 in analyses and stage2 in analyses:
            print(f"\n📉 {stage1} → {stage2}:")
            comparison = compare_datasets(analyses[stage1], analyses[stage2])
            
            print(f"   Colunas perdidas: {comparison['n_columns_lost']}")
            print(f"   Colunas adicionadas: {comparison['n_columns_added']}")
            print(f"   Mudança líquida: {comparison['total_change']:+d}")
            
            if comparison['columns_lost']:
                print(f"\n   🗑️  Principais colunas perdidas:")
                for col in comparison['columns_lost'][:10]:  # Mostrar apenas as 10 primeiras
                    print(f"      - {col}")
                if len(comparison['columns_lost']) > 10:
                    print(f"      ... e mais {len(comparison['columns_lost']) - 10} colunas")
            
            if comparison['nlp_features_lost']:
                print(f"\n   📝 Features NLP perdidas: {len(comparison['nlp_features_lost'])}")
    
    # Análise específica de features críticas
    print("\n\n🎯 ANÁLISE DE FEATURES CRÍTICAS")
    print("-"*80)
    
    critical_features = [
        'lançamento',
        'DATA',
        'E-MAIL',
        '¿Cuál es tu e-mail?',
        'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida?',
        '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
        'Déjame un mensaje'
    ]
    
    for stage_name, analysis in analyses.items():
        if analysis:
            print(f"\n{stage_name}:")
            for feature in critical_features:
                if feature in analysis['columns']:
                    print(f"   ✅ {feature}")
                else:
                    print(f"   ❌ {feature}")
    
    # Comparação com dados antigos (V4)
    print("\n\n📈 COMPARAÇÃO COM PIPELINE ANTERIOR (V4 vs V5)")
    print("-"*80)
    
    # Tentar múltiplos caminhos possíveis para V4
    possible_v4_paths = [
        os.path.join(OLD_DATA_PATH, "04_feature_engineering_2_2/train.csv"),  # Parece haver _2_2
        os.path.join(OLD_DATA_PATH, "04_feature_engineering_2/train.csv"),
        os.path.join(OLD_DATA_PATH, "05_feature_selection/train.csv")  # Talvez o final seja este
    ]
    
    old_df = None
    for old_path in possible_v4_paths:
        old_df = load_dataset_safely(old_path, f"Pipeline V4 - {os.path.basename(os.path.dirname(old_path))}")
        if old_df is not None:
            break
    
    if old_df is not None and '04_feature_eng_2' in dataframes:
        old_analysis = analyze_columns(old_df, "V4_pipeline")
        new_df = dataframes['04_feature_eng_2']
        
        print(f"\n🔄 Pipeline V4 (Anterior) vs V5 (Atual):")
        print(f"   Features V4: {len(old_df.columns)}")
        print(f"   Features V5: {len(new_df.columns)}")
        print(f"   Diferença: {len(old_df.columns) - len(new_df.columns)} features perdidas")
        
        # Identificar features específicas perdidas
        old_cols = set(old_df.columns)
        new_cols = set(new_df.columns)
        lost_features = old_cols - new_cols
        
        # Categorizar features perdidas
        lost_by_type = {
            'tfidf': [f for f in lost_features if 'tfidf' in f.lower()],
            'bow': [f for f in lost_features if 'bow' in f.lower()],
            'sentiment': [f for f in lost_features if 'sentiment' in f.lower()],
            'categorical': [f for f in lost_features if any(cat in f for cat in ['_encoded', '_dummy'])],
            'interaction': [f for f in lost_features if '_x_' in f or '_interaction_' in f],
            'other': [f for f in lost_features if not any(x in f.lower() for x in ['tfidf', 'bow', 'sentiment', '_encoded', '_x_'])]
        }
        
        print(f"\n📊 Features perdidas por tipo:")
        for feature_type, features in lost_by_type.items():
            if features:
                print(f"   {feature_type}: {len(features)} features")
                for f in features[:5]:
                    print(f"      - {f}")
                if len(features) > 5:
                    print(f"      ... e mais {len(features) - 5}")
    
    # Salvar relatório
    report_path = os.path.join(PROJECT_ROOT, f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    print(f"\n\n💾 Salvando relatório detalhado em: {report_path}")
    
    with open(report_path, 'w') as f:
        f.write("RELATÓRIO DE DIAGNÓSTICO - PERDA DE FEATURES\n")
        f.write("="*80 + "\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for stage_name, analysis in analyses.items():
            if analysis:
                f.write(f"\n{stage_name}:\n")
                f.write(f"  Colunas: {analysis['n_columns']}\n")
                f.write(f"  Features NLP: {len(analysis['nlp_features'])}\n")
                f.write(f"  Todas as colunas:\n")
                for col in sorted(analysis['columns']):
                    f.write(f"    - {col}\n")
    
    return analyses, dataframes

def suggest_fixes(analyses):
    """Sugere correções baseadas no diagnóstico."""
    print("\n\n💡 SUGESTÕES DE CORREÇÃO")
    print("-"*80)
    
    # Verificar se temos perda significativa de features NLP
    if '02_preprocessed' in analyses and '04_feature_eng_2' in analyses:
        nlp_before = len([col for col in analyses['02_preprocessed']['columns'] if any(prefix in col for prefix in ['tfidf_', 'bow_'])])
        nlp_after = len(analyses['04_feature_eng_2']['nlp_features'])
        
        if nlp_after < nlp_before * 0.5:
            print("\n⚠️  PROBLEMA: Perda significativa de features NLP")
            print("   SOLUÇÃO: Verificar scripts 03 e 04 - possível problema com:")
            print("   - Parâmetros do TF-IDF (min_df, max_features)")
            print("   - Processamento de texto (normalização, limpeza)")
            print("   - Colunas de texto não sendo processadas")
    
    # Verificar se lançamento foi perdido
    if '01_split' in analyses and 'lançamento' in analyses['01_split']['columns']:
        if '04_feature_eng_2' in analyses and 'lançamento' not in analyses['04_feature_eng_2']['columns']:
            print("\n⚠️  PROBLEMA: Coluna 'lançamento' perdida")
            print("   IMPACTO: Impossível criar features por campanha")
            print("   SOLUÇÃO: Preservar 'lançamento' como feature auxiliar no script 01")
    
    print("\n\n✅ PRÓXIMOS PASSOS RECOMENDADOS:")
    print("1. Revisar o output deste diagnóstico")
    print("2. Identificar as features mais importantes perdidas")
    print("3. Modificar scripts para preservar features críticas")
    print("4. Considerar pipeline dupla: treino (V4) vs produção (V5)")
    print("5. Ajustar parâmetros de feature engineering para recuperar features NLP")

if __name__ == "__main__":
    # Executar diagnóstico
    analyses, dataframes = diagnose_pipeline()
    
    # Sugerir correções
    suggest_fixes(analyses)
    
    print("\n\n✅ Diagnóstico concluído!")