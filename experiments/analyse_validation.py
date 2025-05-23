#!/usr/bin/env python
"""
Script para analisar o dataset de validação e responder perguntas específicas:
1. Quais colunas de email existem?
2. A variável target está presente?
3. Quais colunas esperadas estão faltando?
"""

import pandas as pd
import os
import sys

# Adicionar path do projeto
project_root = "/Users/ramonmoreira/Desktop/smart_ads"
sys.path.insert(0, project_root)

from src.utils.local_storage import (
    connect_to_gcs, list_files_by_extension, categorize_files,
    load_csv_or_excel, extract_launch_id
)

def get_all_expected_columns():
    """
    Coleta todas as colunas de todos os lançamentos + L24 para criar
    conjunto completo de colunas esperadas.
    """
    print("   📂 Carregando dados de todos os lançamentos...")
    
    expected_columns = set()
    
    try:
        # 1. Carregar colunas dos arquivos de lançamento (L16-L22)
        bucket = connect_to_gcs("local_bucket")
        file_paths = list_files_by_extension(bucket, prefix="")
        survey_files, _, _, _ = categorize_files(file_paths)
        
        launch_columns = {}
        
        for file_path in survey_files:
            launch_id = extract_launch_id(file_path)
            if not launch_id:
                continue
                
            df = load_csv_or_excel(bucket, file_path)
            if df is not None:
                launch_columns[launch_id] = list(df.columns)
                expected_columns.update(df.columns)
                print(f"      {launch_id}: {df.shape[1]} colunas")
        
        # 2. Carregar colunas do L24 (produção)
        l24_path = os.path.join(project_root, "data/l24_data.csv")
        if os.path.exists(l24_path):
            l24_df = pd.read_csv(l24_path)
            # Remover colunas de predição
            l24_cols = [col for col in l24_df.columns 
                       if col not in ['Prediction', 'Decil', 'Probabilidade', 'prediction', 'decil']]
            expected_columns.update(l24_cols)
            print(f"      L24: {len(l24_cols)} colunas (sem predições)")
        
        # 3. Adicionar colunas criadas pela pipeline
        pipeline_columns = [
            'email_norm',  # Criada pela normalização de email
            'target',      # Criada pelo matching
        ]
        expected_columns.update(pipeline_columns)
        print(f"      Pipeline: {len(pipeline_columns)} colunas adicionais")
        
        # 4. Mostrar estatísticas por lançamento
        print(f"\n   📊 Distribuição de colunas por lançamento:")
        for launch_id, cols in launch_columns.items():
            print(f"      {launch_id}: {len(cols)} colunas")
        
        print(f"\n   📈 Total de colunas únicas encontradas: {len(expected_columns)}")
        
        return sorted(expected_columns)
        
    except Exception as e:
        print(f"   ❌ Erro ao carregar colunas de lançamentos: {e}")
        
        # Fallback: usar colunas baseadas no diagnóstico anterior
        fallback_columns = [
            # Dados UTM
            'DATA', 'E-MAIL', 'UTM_CAMPAING', 'UTM_SOURCE', 'UTM_MEDIUM',
            'UTM_CONTENT', 'UTM_TERM', 'GCLID',
            
            # Dados da pesquisa (comuns)
            'Marca temporal',
            '¿Cómo te llamas?',
            '¿Cuál es tu género?',
            '¿Cuál es tu edad?',
            '¿Cual es tu país?',
            '¿Cuál es tu e-mail?',
            'email',  # Versão original
            '¿Cual es tu telefono?',
            '¿Cuál es tu instagram?',
            '¿Hace quánto tiempo me conoces?',
            '¿Cuál es tu disponibilidad de tiempo para estudiar inglés?',
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
            '¿Cuál es tu profesión?',
            '¿Cuál es tu sueldo anual? (en dólares)',
            '¿Cuánto te gustaría ganar al año?',
            '¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?',
            '¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?',
            'Déjame un mensaje',
            
            # Variações da pergunta Inmersión
            '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
            '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?',
            '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
            
            # Features L22
            '¿Cuáles son tus principales razones para aprender inglés?',
            '¿Has comprado algún curso para aprender inglés antes?',
            
            # Qualidade (todas as variações)
            'Qualidade',
            'Qualidade (Nome)',
            'Qualidade (Nome) ',
            'Qualidade (nome)',
            'Qualidade (Número)',
            'Qualidade (Número) ',
            'Qualidade (número)',
            'Qualidade (Numero)',
            
            # Sistema
            'status',
            'lançamento',
            
            # Pipeline
            'email_norm',
            'target',
        ]
        
        print(f"   🔄 Usando conjunto fallback: {len(fallback_columns)} colunas")
        return sorted(fallback_columns)

def analyze_validation_dataset():
    """Analisa o dataset de validação de forma detalhada."""
    
    # Caminho do arquivo
    validation_path = "/Users/ramonmoreira/Desktop/smart_ads/data/split/validation.csv"
    
    print("=== ANÁLISE DO DATASET DE VALIDAÇÃO ===")
    print(f"Arquivo: {validation_path}")
    
    # Verificar se arquivo existe
    if not os.path.exists(validation_path):
        print(f"❌ ERRO: Arquivo não encontrado!")
        return
    
    try:
        # Carregar dataset
        df = pd.read_csv(validation_path)
        print(f"✅ Arquivo carregado com sucesso")
        print(f"📊 Shape: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # 1. ANÁLISE DAS COLUNAS DE EMAIL
        print("\n" + "="*60)
        print("1️⃣ ANÁLISE DAS COLUNAS DE EMAIL")
        print("="*60)
        
        email_columns = []
        for col in df.columns:
            if any(term in col.lower() for term in ['email', 'e-mail', 'mail']):
                email_columns.append(col)
        
        print(f"📧 Total de colunas de email encontradas: {len(email_columns)}")
        
        for i, col in enumerate(email_columns, 1):
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            percentage = (non_null_count / total_count) * 100
            
            # Amostra de valores
            sample_values = df[col].dropna().head(3).tolist()
            
            print(f"\n{i}. 📬 {col}")
            print(f"   Preenchimento: {non_null_count:,}/{total_count:,} ({percentage:.1f}%)")
            print(f"   Tipo: {df[col].dtype}")
            print(f"   Amostra: {sample_values}")
            print(f"   Valores únicos: {df[col].nunique():,}")
        
        # 2. ANÁLISE DA VARIÁVEL TARGET
        print("\n" + "="*60)
        print("2️⃣ ANÁLISE DA VARIÁVEL TARGET")
        print("="*60)
        
        # Procurar coluna target
        target_columns = []
        for col in df.columns:
            if any(term in col.lower() for term in ['target', 'alvo', 'label']):
                target_columns.append(col)
        
        if target_columns:
            print(f"🎯 Colunas de target encontradas: {target_columns}")
            
            for col in target_columns:
                print(f"\n📋 {col}:")
                print(f"   Tipo: {df[col].dtype}")
                print(f"   Valores únicos: {sorted(df[col].dropna().unique())}")
                print(f"   Distribuição:")
                value_counts = df[col].value_counts()
                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"      {value}: {count:,} ({percentage:.1f}%)")
        else:
            print("❌ NENHUMA coluna de target encontrada!")
            
            # Verificar colunas binárias que podem ser target
            print("\n🔍 Procurando possíveis colunas target (binárias/categóricas):")
            possible_targets = []
            
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    unique_count = df[col].nunique()
                    if 2 <= unique_count <= 10:
                        unique_values = sorted(df[col].dropna().unique())
                        possible_targets.append({
                            'column': col,
                            'unique_count': unique_count,
                            'values': unique_values
                        })
            
            if possible_targets:
                for target_info in possible_targets:
                    print(f"   🤔 {target_info['column']}: {target_info['values']} ({target_info['unique_count']} únicos)")
            else:
                print("   Nenhuma coluna binária/categórica encontrada.")
        
        # 3. ANÁLISE DE COLUNAS FALTANDO
        print("\n" + "="*60)
        print("3️⃣ ANÁLISE DE COLUNAS FALTANDO")
        print("="*60)
        
        print("🔍 Carregando colunas esperadas de todos os lançamentos + L24...")
        expected_columns = get_all_expected_columns()
        
        print(f"📊 Total de colunas esperadas: {len(expected_columns)}")
        
        current_columns = set(df.columns)
        expected_columns_set = set(expected_columns)
        
        missing_columns = expected_columns_set - current_columns
        extra_columns = current_columns - expected_columns_set
        
        print(f"📊 Resumo de compatibilidade:")
        print(f"   Colunas presentes: {len(current_columns)}")
        print(f"   Colunas esperadas: {len(expected_columns_set)}")
        print(f"   Colunas em comum: {len(current_columns & expected_columns_set)}")
        print(f"   Colunas faltando: {len(missing_columns)}")
        print(f"   Colunas extras: {len(extra_columns)}")
        
        if missing_columns:
            print(f"\n❌ COLUNAS FALTANDO ({len(missing_columns)}):")
            
            # Categorizar colunas faltando
            critical_missing = set()
            utm_missing = set()
            quality_missing = set()
            other_missing = set()
            
            for col in missing_columns:
                if col in ['target', 'email_norm']:
                    critical_missing.add(col)
                elif any(term in col for term in ['UTM', 'DATA', 'GCLID', 'E-MAIL']):
                    utm_missing.add(col)
                elif 'qualidade' in col.lower():
                    quality_missing.add(col)
                else:
                    other_missing.add(col)
            
            if critical_missing:
                print(f"\n   🚨 CRÍTICAS ({len(critical_missing)}):")
                for col in sorted(critical_missing):
                    print(f"      - {col}")
            
            if utm_missing:
                print(f"\n   📡 UTM/Sistema ({len(utm_missing)}):")
                for col in sorted(utm_missing):
                    print(f"      - {col}")
            
            if quality_missing:
                print(f"\n   ⭐ Qualidade ({len(quality_missing)}):")
                for col in sorted(quality_missing):
                    print(f"      - {col}")
            
            if other_missing:
                print(f"\n   📋 Outras ({len(other_missing)}):")
                for col in sorted(other_missing):
                    print(f"      - {col}")
        
        if extra_columns:
            print(f"\n➕ COLUNAS EXTRAS ({len(extra_columns)}):")
            for col in sorted(extra_columns):
                print(f"   - {col}")
        
        # 4. LISTA COMPLETA DE COLUNAS
        print("\n" + "="*60)
        print("4️⃣ LISTA COMPLETA DE COLUNAS PRESENTES")
        print("="*60)
        
        print(f"Total: {len(df.columns)} colunas")
        for i, col in enumerate(df.columns, 1):
            data_type = df[col].dtype
            non_null = df[col].notna().sum()
            null_pct = (df[col].isna().sum() / len(df)) * 100
            
            print(f"{i:2d}. {col}")
            print(f"     Tipo: {data_type} | Preenchimento: {non_null:,} ({100-null_pct:.1f}%)")
        
        # 5. RECOMENDAÇÕES
        print("\n" + "="*60)
        print("5️⃣ RECOMENDAÇÕES")
        print("="*60)
        
        print("📧 EMAILS:")
        if len(email_columns) > 1:
            print("   ✅ Use 'email_norm' como principal (se existir)")
            print("   ✅ Fallback para 'E-MAIL' se email_norm vazio")
        
        print("\n🎯 TARGET:")
        if not target_columns:
            print("   ❌ CRÍTICO: Coluna target ausente!")
            print("   🔧 Verificar script 01_data_collection_and_integration.py")
            print("   🔧 Verificar função create_target_variable")
        
        print("\n📋 COLUNAS FALTANDO:")
        if missing_columns:
            critical_missing = {'target', 'email_norm'}
            critical_found = critical_missing & missing_columns
            
            if critical_found:
                print(f"   🚨 CRÍTICAS: {critical_found}")
            
            utm_missing = {col for col in missing_columns if any(term in col for term in ['UTM', 'DATA', 'GCLID'])}
            if utm_missing:
                print(f"   ⚠️  UTM: {len(utm_missing)} colunas (verificar merge com dados UTM)")
        
    except Exception as e:
        print(f"❌ ERRO ao analisar arquivo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_validation_dataset()