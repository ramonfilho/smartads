#!/usr/bin/env python
"""
Script para comparar parâmetros joblib entre versões.
Compara os arquivos de parâmetros das pastas src e V4_API.
"""

import joblib
import os
import json
import glob
import numpy as np
from deepdiff import DeepDiff
from datetime import datetime
import warnings

# Suprimir warnings sobre NaN
warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_joblib_file(filepath):
    """Carrega arquivo joblib com tratamento de erro."""
    try:
        return joblib.load(filepath)
    except Exception as e:
        print(f"Erro ao carregar {filepath}: {e}")
        return None

def find_joblib_files(directory):
    """Encontra todos os arquivos .joblib em um diretório."""
    if not os.path.exists(directory):
        return []
    
    # Buscar arquivos .joblib
    pattern = os.path.join(directory, "*.joblib")
    files = glob.glob(pattern)
    
    # Se não encontrar, tentar buscar recursivamente
    if not files:
        pattern = os.path.join(directory, "**/*.joblib")
        files = glob.glob(pattern, recursive=True)
    
    return sorted(files)

def compare_params(params1, params2, name1="Versão 1", name2="Versão 2"):
    """Compara dois conjuntos de parâmetros e retorna as diferenças."""
    # Função para converter NaN para string antes da comparação
    def convert_nan_to_str(obj):
        if isinstance(obj, dict):
            return {k: convert_nan_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan_to_str(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return "NaN"
        else:
            return obj
    
    # Converter NaN antes de comparar
    params1_converted = convert_nan_to_str(params1)
    params2_converted = convert_nan_to_str(params2)
    
    # Configurar DeepDiff
    try:
        diff = DeepDiff(
            params1_converted, 
            params2_converted, 
            ignore_order=True, 
            verbose_level=2
        )
    except Exception as e:
        print(f"Erro durante comparação: {e}")
        # Tentar comparação mais simples
        diff = DeepDiff(params1, params2, ignore_order=True)
    
    results = {
        "values_changed": {},
        "items_added": {},
        "items_removed": {},
        "type_changes": {}
    }
    
    # Valores alterados
    if 'values_changed' in diff:
        for key, changes in diff['values_changed'].items():
            clean_key = key.replace("root", "").replace("[", ".").replace("]", "").replace("'", "")
            try:
                old_val = changes.get('old_value', 'N/A')
                new_val = changes.get('new_value', 'N/A')
                
                # Converter valores para string de forma segura
                old_str = str(old_val) if not isinstance(old_val, float) or not np.isnan(old_val) else "NaN"
                new_str = str(new_val) if not isinstance(new_val, float) or not np.isnan(new_val) else "NaN"
                
                results['values_changed'][clean_key] = {
                    name1: old_str,
                    name2: new_str
                }
            except Exception as e:
                results['values_changed'][clean_key] = {
                    name1: "Error converting value",
                    name2: "Error converting value"
                }
    
    # Itens adicionados
    if 'dictionary_item_added' in diff:
        for item in diff['dictionary_item_added']:
            clean_key = str(item).replace("root", "").replace("[", ".").replace("]", "").replace("'", "")
            results['items_added'][clean_key] = f"Adicionado em {name2}"
    
    # Itens removidos
    if 'dictionary_item_removed' in diff:
        for item in diff['dictionary_item_removed']:
            clean_key = str(item).replace("root", "").replace("[", ".").replace("]", "").replace("'", "")
            results['items_removed'][clean_key] = f"Removido em {name2}"
    
    # Mudanças de tipo
    if 'type_changes' in diff:
        for key, changes in diff['type_changes'].items():
            clean_key = key.replace("root", "").replace("[", ".").replace("]", "").replace("'", "")
            old_type = str(type(changes.get('old_value', 'N/A')))
            new_type = str(type(changes.get('new_value', 'N/A')))
            results['type_changes'][clean_key] = {
                name1: old_type,
                name2: new_type
            }
    
    return results

def save_comparison_report(comparisons, output_dir):
    """Salva relatório de comparação em arquivo texto."""
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Arquivo de texto
    txt_file = os.path.join(output_dir, "params_comparison_report.txt")
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RELATÓRIO DE COMPARAÇÃO DE PARÂMETROS\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for script_num, comparison in comparisons.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"SCRIPT {script_num}: {comparison['file']}\n")
            f.write(f"{'='*60}\n")
            
            if comparison['error']:
                f.write(f"\nERRO: {comparison['error']}\n")
                continue
            
            f.write(f"\nArquivos comparados:\n")
            f.write(f"  Original: {os.path.basename(comparison.get('original_path', 'N/A'))}\n")
            f.write(f"  V4_API:   {os.path.basename(comparison.get('v4_api_path', 'N/A'))}\n")
            
            if comparison['no_differences']:
                f.write("\n✅ NENHUMA DIFERENÇA ENCONTRADA\n")
                continue
            
            # Valores alterados
            if comparison['differences']['values_changed']:
                f.write(f"\n📝 VALORES ALTERADOS ({len(comparison['differences']['values_changed'])}):\n")
                for key, values in comparison['differences']['values_changed'].items():
                    f.write(f"\n  {key}:\n")
                    f.write(f"    Original: {values['Original']}\n")
                    f.write(f"    V4_API:   {values['V4_API']}\n")
            
            # Itens adicionados
            if comparison['differences']['items_added']:
                f.write(f"\n➕ ITENS ADICIONADOS EM V4_API ({len(comparison['differences']['items_added'])}):\n")
                for key, desc in comparison['differences']['items_added'].items():
                    f.write(f"  - {key}\n")
            
            # Itens removidos
            if comparison['differences']['items_removed']:
                f.write(f"\n➖ ITENS REMOVIDOS EM V4_API ({len(comparison['differences']['items_removed'])}):\n")
                for key, desc in comparison['differences']['items_removed'].items():
                    f.write(f"  - {key}\n")
            
            # Mudanças de tipo
            if comparison['differences']['type_changes']:
                f.write(f"\n🔄 MUDANÇAS DE TIPO ({len(comparison['differences']['type_changes'])}):\n")
                for key, types in comparison['differences']['type_changes'].items():
                    f.write(f"\n  {key}:\n")
                    f.write(f"    Original: {types['Original']}\n")
                    f.write(f"    V4_API:   {types['V4_API']}\n")
    
    print(f"📄 Relatório TXT salvo em: {txt_file}")
    
    # Arquivo JSON
    json_file = os.path.join(output_dir, "params_comparison.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Dados JSON salvos em: {json_file}")

def main():
    """Função principal para comparar os parâmetros."""
    
    # Definir caminhos base
    base_path = "/Users/ramonmoreira/Desktop/smart_ads"
    original_base = os.path.join(base_path, "src/preprocessing")
    v4_api_base = os.path.join(base_path, "V4_API/src/preprocessing")
    
    # Diretório de saída
    output_dir = os.path.join(base_path, "reports/params_comparison")
    
    # Scripts para comparar
    scripts = ["02", "03", "04"]
    
    # Armazenar comparações
    comparisons = {}
    
    print("🔍 Iniciando comparação de parâmetros...\n")
    print(f"📁 Diretório de saída: {output_dir}\n")
    
    for script in scripts:
        print(f"\n{'='*60}")
        print(f"Comparando parâmetros do Script {script}...")
        print(f"{'='*60}")
        
        # Diretórios dos parâmetros
        original_dir = os.path.join(original_base, f"{script}_params")
        v4_api_dir = os.path.join(v4_api_base, f"{script}_params")
        
        # Buscar arquivos joblib em ambos os diretórios
        original_files = find_joblib_files(original_dir)
        v4_api_files = find_joblib_files(v4_api_dir)
        
        print(f"\n📂 Arquivos encontrados em original: {len(original_files)}")
        if original_files:
            for f in original_files:
                print(f"   - {os.path.basename(f)}")
                
        print(f"📂 Arquivos encontrados em V4_API: {len(v4_api_files)}")
        if v4_api_files:
            for f in v4_api_files:
                print(f"   - {os.path.basename(f)}")
        
        # Verificar se encontrou arquivos
        if not original_files:
            print(f"❌ Nenhum arquivo joblib encontrado em: {original_dir}")
            comparisons[script] = {
                'file': f'{script}_params',
                'error': 'Nenhum arquivo joblib encontrado no diretório original',
                'no_differences': False,
                'differences': {}
            }
            continue
            
        if not v4_api_files:
            print(f"❌ Nenhum arquivo joblib encontrado em: {v4_api_dir}")
            comparisons[script] = {
                'file': f'{script}_params',
                'error': 'Nenhum arquivo joblib encontrado no diretório V4_API',
                'no_differences': False,
                'differences': {}
            }
            continue
        
        # Se houver múltiplos arquivos, pegar o primeiro ou tentar encontrar o principal
        original_file = original_files[0]
        v4_api_file = v4_api_files[0]
        
        # Tentar encontrar arquivo com nome padrão se houver múltiplos
        if len(original_files) > 1:
            for f in original_files:
                if f"{script}_params.joblib" in f:
                    original_file = f
                    break
                    
        if len(v4_api_files) > 1:
            for f in v4_api_files:
                if f"{script}_params.joblib" in f:
                    v4_api_file = f
                    break
        
        print(f"\n📄 Comparando:")
        print(f"   Original: {os.path.basename(original_file)}")
        print(f"   V4_API:   {os.path.basename(v4_api_file)}")
        
        # Carregar parâmetros
        print("\n📂 Carregando arquivos...")
        params_original = load_joblib_file(original_file)
        params_v4_api = load_joblib_file(v4_api_file)
        
        if params_original is None or params_v4_api is None:
            comparisons[script] = {
                'file': f'{script}_params',
                'original_path': original_file,
                'v4_api_path': v4_api_file,
                'error': 'Erro ao carregar um ou ambos arquivos',
                'no_differences': False,
                'differences': {}
            }
            continue
        
        # Comparar parâmetros
        print("🔄 Comparando parâmetros...")
        try:
            differences = compare_params(params_original, params_v4_api, "Original", "V4_API")
        except Exception as e:
            print(f"❌ Erro durante comparação: {e}")
            comparisons[script] = {
                'file': f'{script}_params',
                'original_path': original_file,
                'v4_api_path': v4_api_file,
                'error': f'Erro durante comparação: {str(e)}',
                'no_differences': False,
                'differences': {}
            }
            continue
        
        # Verificar se há diferenças
        has_differences = any([
            differences['values_changed'],
            differences['items_added'],
            differences['items_removed'],
            differences['type_changes']
        ])
        
        if not has_differences:
            print("✅ Nenhuma diferença encontrada!")
        else:
            print("⚠️  Diferenças encontradas:")
            if differences['values_changed']:
                print(f"   - {len(differences['values_changed'])} valores alterados")
            if differences['items_added']:
                print(f"   - {len(differences['items_added'])} itens adicionados")
            if differences['items_removed']:
                print(f"   - {len(differences['items_removed'])} itens removidos")
            if differences['type_changes']:
                print(f"   - {len(differences['type_changes'])} mudanças de tipo")
        
        comparisons[script] = {
            'file': f'{script}_params',
            'original_path': original_file,
            'v4_api_path': v4_api_file,
            'error': None,
            'no_differences': not has_differences,
            'differences': differences
        }
    
    # Salvar relatórios
    print(f"\n\n📁 Salvando relatórios em: {output_dir}")
    save_comparison_report(comparisons, output_dir)
    
    # Criar sumário
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("SUMÁRIO DA COMPARAÇÃO DE PARÂMETROS\n")
        f.write("="*50 + "\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        for script, data in comparisons.items():
            f.write(f"Script {script}: ")
            if data['error']:
                f.write(f"❌ ERRO - {data['error']}\n")
            elif data['no_differences']:
                f.write("✅ SEM DIFERENÇAS\n")
            else:
                f.write("⚠️  COM DIFERENÇAS\n")
                if data['differences']['values_changed']:
                    f.write(f"     - {len(data['differences']['values_changed'])} valores alterados\n")
                if data['differences']['items_added']:
                    f.write(f"     - {len(data['differences']['items_added'])} itens adicionados\n")
                if data['differences']['items_removed']:
                    f.write(f"     - {len(data['differences']['items_removed'])} itens removidos\n")
                if data['differences']['type_changes']:
                    f.write(f"     - {len(data['differences']['type_changes'])} mudanças de tipo\n")
    
    print(f"📄 Sumário salvo em: {summary_file}")
    
    print("\n✨ Comparação concluída!")

if __name__ == "__main__":
    main()