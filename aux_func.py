#!/usr/bin/env python
"""
map_feature_names_v5.py - Versão final que detecta todas as variações de nomes
"""
import pandas as pd
import numpy as np
import os
import json
import joblib
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

PROJECT_ROOT = "/Users/ramonmoreira/desktop/smart_ads"

def remove_accents(text):
    """Remove acentos de um texto"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def create_base_variations(base_name):
    """
    Cria todas as variações possíveis de um nome base
    """
    variations = set()
    
    # Original
    variations.add(base_name)
    
    # Lowercase
    lower = base_name.lower()
    variations.add(lower)
    
    # Sem acentos
    no_accents = remove_accents(base_name)
    variations.add(no_accents)
    variations.add(no_accents.lower())
    
    # Com underscores
    with_underscores = base_name.replace(' ', '_')
    variations.add(with_underscores)
    variations.add(with_underscores.lower())
    
    # Sem acentos e com underscores
    no_accents_underscores = remove_accents(with_underscores)
    variations.add(no_accents_underscores)
    variations.add(no_accents_underscores.lower())
    
    # Versões truncadas (primeiros 30 caracteres)
    for var in list(variations):
        if len(var) > 30:
            variations.add(var[:30])
            variations.add(var[:31])  # Por segurança
            variations.add(var[:32])
    
    # Remover caracteres especiais
    for var in list(variations):
        cleaned = re.sub(r'[¿?¡!,.:;()\-]', '', var)
        variations.add(cleaned)
        variations.add(cleaned.replace(' ', '_'))
    
    return variations

def find_all_feature_mappings():
    """
    Encontra TODOS os mapeamentos possíveis entre features
    """
    print("=== MAPEAMENTO COMPLETO DE FEATURES V5 ===\n")
    
    # 1. Carregar dataset
    train_path = os.path.join(PROJECT_ROOT, "data/new/04_feature_selection/train.csv")
    train_df = pd.read_csv(train_path)
    print(f"Dataset carregado: {train_df.shape}")
    
    features = [col for col in train_df.columns if col != 'target']
    
    # 2. Definir colunas de texto conhecidas e suas variações
    known_text_bases = {
        'cuando_hables': [
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
            'Cuando_hables_inglés_con_fluidez,_¿qué_cambiará_en_tu_vida?_¿Qué_oportunidades_se_abrirán_para_ti?',
            'Cuando_hables_inglés_con_fluid',
            'cuando_hables_inglés_con_fluid',
            'cuando_hables_ingles_con_fluid'
        ],
        'que_esperas': [
            '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
            '¿Qué_esperas_aprender_en_el_evento_Cero_a_Inglés_Fluido?',
            'Qué_esperas_aprender_en_el_eve',
            'que_esperas_aprender_en_el_eve',
            'que_esperas_aprender_en_el'
        ],
        'dejame': [
            'Déjame un mensaje',
            'Déjame_un_mensaje',
            'déjame_un_mensaje',
            'dejame_un_mensaje'
        ]
    }
    
    # Criar todas as variações possíveis
    all_variations = {}
    for base_key, base_list in known_text_bases.items():
        all_variations[base_key] = set()
        for base in base_list:
            all_variations[base_key].update(create_base_variations(base))
    
    # 3. Classificar features por tipo e sufixo
    feature_components = {}
    for feat in features:
        # Identificar tipo
        feat_type = 'other'
        suffix = ''
        base = feat
        
        for type_marker in ['_tfidf_', '_topic_', '_motiv_', '_sentiment', '_embedding_']:
            if type_marker in feat:
                parts = feat.split(type_marker, 1)
                base = parts[0]
                feat_type = type_marker.strip('_')
                suffix = parts[1] if len(parts) > 1 else ''
                break
        
        # Identificar a qual texto base pertence
        base_category = None
        for category, variations in all_variations.items():
            if base in variations or any(base.startswith(var[:20]) for var in variations if len(var) >= 20):
                base_category = category
                break
        
        feature_components[feat] = {
            'base': base,
            'type': feat_type,
            'suffix': suffix,
            'category': base_category,
            'full_name': feat
        }
    
    # 4. Criar mapeamentos
    mappings = {}
    
    # Agrupar por categoria + tipo + sufixo
    groups = defaultdict(list)
    for feat, comp in feature_components.items():
        if comp['category'] and comp['type'] != 'other':
            key = f"{comp['category']}_{comp['type']}_{comp['suffix']}"
            groups[key].append(feat)
    
    # Mapear dentro de cada grupo
    print(f"\n📊 Processando {len(groups)} grupos de features...")
    total_pairs = 0
    
    for group_key, group_features in sorted(groups.items()):
        if len(group_features) < 2:
            continue
        
        # Ordenar por tamanho (menor primeiro)
        group_features.sort(key=len)
        
        # O menor é o canônico
        canonical = group_features[0]
        
        # Todos os outros mapeiam para ele
        for feat in group_features[1:]:
            mappings[feat] = canonical
            total_pairs += 1
        
        if total_pairs <= 20:  # Mostrar primeiros exemplos
            print(f"\n✅ Grupo {group_key}: {len(group_features)-1} mapeamentos")
            print(f"   Canônico: {canonical}")
            for feat in group_features[1:2]:  # Mostrar 1 exemplo
                print(f"   Mapeado: {feat[:60]}...")
    
    # 5. Estatísticas
    print(f"\n📈 RESUMO FINAL:")
    print(f"   Total de features: {len(features)}")
    print(f"   Features mapeadas: {len(mappings)}")
    print(f"   Taxa de mapeamento: {len(mappings)/len(features)*100:.1f}%")
    
    # Features não mapeadas
    unmapped = []
    for feat in features:
        if feat not in mappings and not any(feat == v for v in mappings.values()):
            unmapped.append(feat)
    
    unmapped_by_type = defaultdict(int)
    unmapped_examples = defaultdict(list)
    
    for feat in unmapped:
        comp = feature_components[feat]
        unmapped_by_type[comp['type']] += 1
        if len(unmapped_examples[comp['type']]) < 3:
            unmapped_examples[comp['type']].append(feat)
    
    print(f"\n⚠️ Features não mapeadas: {len(unmapped)}")
    for feat_type, count in sorted(unmapped_by_type.items()):
        print(f"\n   {feat_type}: {count} features")
        for example in unmapped_examples[feat_type]:
            print(f"      Ex: {example[:60]}...")
    
    # 6. Salvar resultados
    reverse_mappings = {v: k for k, v in mappings.items()}
    
    # Adicionar mapeamento de colunas de texto base
    text_column_mapping = {}
    for feat, comp in feature_components.items():
        if comp['category'] and comp['type'] == 'other':
            # É uma coluna de texto base
            if comp['category'] == 'cuando_hables':
                text_column_mapping[feat] = 'cuando_hables_ingles_fluido'
            elif comp['category'] == 'que_esperas':
                text_column_mapping[feat] = 'que_esperas_aprender_evento'
            elif comp['category'] == 'dejame':
                text_column_mapping[feat] = 'dejame_mensaje'
    
    mapping_data = {
        'canonical_mapping': mappings,
        'reverse_mapping': reverse_mappings,
        'text_column_mapping': text_column_mapping,
        'statistics': {
            'total_features': len(features),
            'mapped_features': len(mappings),
            'unmapped_features': len(unmapped),
            'mapping_rate': f"{len(mappings)/len(features)*100:.1f}%",
            'unmapped_by_type': dict(unmapped_by_type)
        }
    }
    
    output_path = os.path.join(PROJECT_ROOT, "src/preprocessing/params/feature_name_mapping_final.json")
    with open(output_path, 'w') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Mapeamento final salvo em: {output_path}")
    
    # Parâmetros de produção
    prod_params = {
        'feature_mapping': mapping_data
    }
    
    prod_path = os.path.join(PROJECT_ROOT, "src/preprocessing/params/production_feature_mapping_final.joblib")
    joblib.dump(prod_params, prod_path)
    print(f"📁 Parâmetros de produção salvos em: {prod_path}")
    
    return mapping_data

if __name__ == "__main__":
    mapping = find_all_feature_mappings()
    
    # Teste
    print("\n=== TESTE DO MAPEAMENTO ===")
    examples = [
        "Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?_tfidf_trabajo",
        "cuando_hables_inglés_con_fluid_tfidf_trabajo",
        "Déjame un mensaje_topic_3",
        "déjame_un_mensaje_topic_3"
    ]
    
    for ex in examples:
        if ex in mapping['canonical_mapping']:
            print(f"\n✅ {ex[:50]}...")
            print(f"   → {mapping['canonical_mapping'][ex]}")
        else:
            print(f"\n❓ {ex[:50]}... (verificar se é canônico)")