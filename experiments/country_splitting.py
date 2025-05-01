#!/usr/bin/env python
"""
Script melhorado para agrupar países com base em comportamentos e características semelhantes.

Principais melhorias:
1. Melhor tratamento para categoria "desconhecido"
2. Agrupamento mais inteligente de países pequenos
3. Agrupamento baseado em características similares
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Caminhos específicos para os diretórios de entrada e saída
INPUT_DIR = "data/02_3_processed_text_code6"
OUTPUT_DIR = "data/02_3_processed_text_selection_per_country"

def identify_columns(df):
    """Identifica colunas importantes no DataFrame."""
    column_dict = {
        'country': None,
        'target': None,
        'age': [],
        'text_features': []
    }
    
    # Identificar coluna de país
    country_patterns = ['país', 'pais', 'country', '¿Cual es tu país?']
    for col in df.columns:
        if any(pattern.lower() in col.lower() for pattern in country_patterns):
            column_dict['country'] = col
            break
    
    # Identificar coluna target
    if 'target' in df.columns:
        column_dict['target'] = 'target'
    
    # Identificar colunas de idade
    age_patterns = ['age', 'edad', 'idade', '¿Cuál es tu edad?']
    for col in df.columns:
        if any(pattern.lower() in col.lower() for pattern in age_patterns):
            column_dict['age'].append(col)
    
    # Identificar features relacionadas a texto e motivação
    text_patterns = [
        'professional_motivation_score', 
        'sentiment', 
        'aspiration_score',
        'commitment_score', 
        'career_term_score',
        'motivation',
        'text_embedding',
        'motiv_'
    ]
    
    for col in df.columns:
        if any(pattern in col.lower() for pattern in text_patterns):
            column_dict['text_features'].append(col)
    
    return column_dict

def load_datasets(input_dir=INPUT_DIR):
    """Carrega os datasets de treino, validação e teste."""
    print(f"\n=== Carregando datasets de {input_dir} ===")
    
    # Definir caminhos
    train_path = os.path.join(input_dir, "train.csv")
    val_path = os.path.join(input_dir, "validation.csv")
    test_path = os.path.join(input_dir, "test.csv")
    
    # Verificar existência dos arquivos
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_path}")
    
    # Carregar treino (obrigatório)
    df_train = pd.read_csv(train_path)
    print(f"Dataset de treino carregado: {df_train.shape[0]} linhas, {df_train.shape[1]} colunas")
    
    # Carregar validação (opcional)
    df_val = None
    if os.path.exists(val_path):
        df_val = pd.read_csv(val_path)
        print(f"Dataset de validação carregado: {df_val.shape[0]} linhas, {df_val.shape[1]} colunas")
    
    # Carregar teste (opcional)
    df_test = None
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        print(f"Dataset de teste carregado: {df_test.shape[0]} linhas, {df_test.shape[1]} colunas")
    
    return df_train, df_val, df_test

def print_sample_values(df, col_name, n=10):
    """Imprime valores de amostra de uma coluna para debug."""
    print(f"\nAmostra de valores da coluna '{col_name}':")
    sample_values = df[col_name].dropna().unique()[:n]
    for value in sample_values:
        print(f"  - {value} (tipo: {type(value)})")
    print(f"Número de valores únicos: {df[col_name].nunique()}")

def identify_countries_with_sufficient_data(df_train, column_dict, min_samples, min_conversions):
    """Identifica países com dados suficientes para modelos individuais."""
    print("\n=== Identificando países com dados suficientes ===")
    
    country_col = column_dict['country']
    target_col = column_dict['target']
    
    if not country_col or not target_col:
        raise ValueError("Colunas de país ou target não identificadas!")
    
    # Verificar e imprimir amostras da coluna de país
    print_sample_values(df_train, country_col)
    
    # Calcular estatísticas por país
    country_stats = df_train.groupby(country_col).agg(
        count=(target_col, 'count'),
        conversions=(target_col, 'sum'),
        conversion_rate=(target_col, 'mean')
    ).reset_index()
    
    # Classificar países com base nos critérios
    sufficient_countries = []
    insufficient_countries = []
    special_cases = []  # Para "desconhecido" e "Otro país"
    
    for _, row in country_stats.iterrows():
        country = row[country_col]
        count = row['count']
        conversions = row['conversions']
        conv_rate = row['conversion_rate']
        
        # Tratar casos especiais
        if country == 'desconhecido' or country is None or pd.isna(country):
            special_cases.append(('desconhecido', count, conversions, conv_rate))
            print(f"'desconhecido': CASO ESPECIAL - {count} amostras, {conversions} conversões, taxa {conv_rate:.2%}")
            continue
            
        if country == 'Otro país':
            special_cases.append(('Otro país', count, conversions, conv_rate))
            print(f"'Otro país': CASO ESPECIAL - {count} amostras, {conversions} conversões, taxa {conv_rate:.2%}")
            continue
        
        # Classificar países normais
        if count >= min_samples and conversions >= min_conversions:
            sufficient_countries.append(country)
            print(f"{country}: SUFICIENTE - {count} amostras, {conversions} conversões, taxa {conv_rate:.2%}")
        else:
            insufficient_countries.append(country)
            reasons = []
            if count < min_samples:
                reasons.append(f"poucas amostras ({count}/{min_samples})")
            if conversions < min_conversions:
                reasons.append(f"poucas conversões ({conversions}/{min_conversions})")
            
            print(f"{country}: INSUFICIENTE - {', '.join(reasons)}, taxa {conv_rate:.2%}")
    
    return sufficient_countries, insufficient_countries, special_cases, country_stats

def extract_clustering_features(df_train, countries_to_cluster, column_dict):
    """Extrai features para clustering de países."""
    print("\n=== Extraindo features para clustering ===")
    
    country_col = column_dict['country']
    target_col = column_dict['target']
    age_cols = column_dict['age'][:1]  # Limitar a 1 coluna de idade para simplificar
    text_feature_cols = column_dict['text_features']
    
    # Filtrar apenas países para agrupar
    df_countries = df_train[df_train[country_col].isin(countries_to_cluster)]
    
    # Inicializar DataFrame para features de clustering
    clustering_data = []
    
    # Verificar se temos países para processar
    if len(countries_to_cluster) == 0:
        print("Nenhum país para agrupar.")
        return None, None
    
    # Processar cada país
    for country in countries_to_cluster:
        country_data = df_countries[df_countries[country_col] == country]
        
        if len(country_data) == 0:
            print(f"Aviso: Nenhum dado encontrado para o país {country}")
            continue
        
        # Feature 1: Taxa de conversão
        conversion_rate = country_data[target_col].mean()
        
        # Features de cluster
        country_features = {
            country_col: country,
            'conversion_rate': conversion_rate,
            'sample_count': len(country_data)
        }
        
        # Feature 2: Extrair features textuais mais importantes
        # Selecionar um subconjunto relevante de features textuais para melhorar o clustering
        important_text_features = [
            col for col in text_feature_cols 
            if any(key in col.lower() for key in [
                'motivation_score', 'sentiment', 'motiv_work', 
                'oportunidades', 'trabajo', 'career', 'tfidf_trabajo'
            ])
        ]
        
        for col in important_text_features[:20]:  # Limitar a 20 features para não sobrecarregar
            if col in country_data.columns:
                try:
                    country_features[col] = country_data[col].mean()
                except:
                    pass  # Ignorar features que não podem ser agregadas
        
        # Feature 3: Distribuição etária (simplificada)
        for col in age_cols:
            if col in country_data.columns:
                if country_data[col].dtype == 'object' or country_data[col].nunique() < 10:
                    value_counts = country_data[col].value_counts(normalize=True)
                    # Limitar a 3 categorias mais frequentes
                    top_values = value_counts.nlargest(3)
                    for value, count in top_values.items():
                        # Garantir que o valor é uma string válida para nome de coluna
                        value_str = str(value).replace(' ', '_').replace('.', '_')
                        feature_name = f"{col}_{value_str}"
                        country_features[feature_name] = count
                else:
                    country_features[f"{col}_mean"] = country_data[col].mean()
        
        clustering_data.append(country_features)
    
    # Verificar se temos dados suficientes para clustering
    if len(clustering_data) < 2:
        print("Dados insuficientes para clustering. Pelo menos 2 países são necessários.")
        return None, None
    
    # Criar DataFrame
    cluster_df = pd.DataFrame(clustering_data)
    
    # Definir features para clustering
    feature_cols = [col for col in cluster_df.columns 
                   if col != country_col and col != 'sample_count']
    
    return cluster_df, feature_cols

def find_optimal_clusters(X, max_clusters=5):
    """
    Encontra o número ótimo de clusters usando silhouette score.
    
    Args:
        X: Matriz de features
        max_clusters: Número máximo de clusters a testar
        
    Returns:
        Número ótimo de clusters
    """
    # Se tivermos poucos dados, limitar o número máximo de clusters
    n_samples = X.shape[0]
    max_clusters = min(max_clusters, n_samples - 1)
    
    if max_clusters < 2:
        return 2  # Mínimo de 2 clusters
    
    # Avaliar diferentes números de clusters
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calcular silhouette score
        try:
            score = silhouette_score(X, labels)
            silhouette_scores.append((n_clusters, score))
            print(f"  {n_clusters} clusters: silhouette score = {score:.4f}")
        except:
            print(f"  Erro ao calcular silhouette score para {n_clusters} clusters")
    
    # Encontrar número ótimo de clusters
    if silhouette_scores:
        optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    else:
        optimal_clusters = 2  # Default
    
    print(f"Número ótimo de clusters: {optimal_clusters}")
    return optimal_clusters

def cluster_countries(cluster_df, feature_cols, country_col, n_clusters=None):
    """Agrupa países em clusters baseado em suas características."""
    print("\n=== Agrupando países em clusters ===")
    
    # Verificar se temos dados suficientes
    if cluster_df is None or len(cluster_df) < 2:
        print("Dados insuficientes para realizar clustering.")
        return None
    
    # Preparar features para clustering
    X = cluster_df[feature_cols].copy()
    
    # Lidar com valores ausentes
    X = X.fillna(X.mean())
    
    # Verificar se temos features válidas
    if X.shape[1] == 0:
        print("Erro: Nenhuma feature válida para clustering")
        return None
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encontrar número ótimo de clusters se não fornecido
    if n_clusters is None or n_clusters > len(cluster_df) - 1:
        print("Determinando número ótimo de clusters...")
        n_clusters = find_optimal_clusters(X_scaled, min(5, len(cluster_df) - 1))
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Adicionar cluster ao DataFrame
    cluster_df['cluster'] = cluster_labels
    
    # Mostrar agrupamentos
    for cluster_id in range(n_clusters):
        countries_in_cluster = cluster_df[cluster_df['cluster'] == cluster_id][country_col].tolist()
        # Garantir que todos os países são strings
        countries_str = [str(c) for c in countries_in_cluster]
        print(f"Cluster {cluster_id}: {len(countries_str)} países - {', '.join(countries_str)}")
    
    return cluster_df

def manual_country_grouping(insufficient_countries, special_cases):
    """
    Agrupa manualmente países pequenos com base em região geográfica e características.
    
    Args:
        insufficient_countries: Lista de países com dados insuficientes
        special_cases: Lista de casos especiais (desconhecido, otro país)
        
    Returns:
        Dict com agrupamentos manuais
    """
    print("\n=== Agrupando manualmente países pequenos por região ===")
    
    # Classificar países por região
    central_america = ['Costa Rica', 'Panamá', 'Guatemala', 'Honduras', 'El Salvador', 'Nicaragua']
    south_america_andean = ['Perú', 'Bolivia', 'Paraguay', 'Uruguay', 'Venezuela']
    caribbean = ['Cuba', 'Puerto Rico', 'Haití', 'Jamaica', 'Trinidad y Tobago', 'Bahamas']
    
    # Inicializar grupos
    manual_groups = {
        'América Central': [],
        'Países Andinos': [],
        'Caribe': [],
        'Outros Países Pequenos': []
    }
    
    # Classificar países insuficientes
    for country in insufficient_countries:
        if country in central_america:
            manual_groups['América Central'].append(country)
        elif country in south_america_andean:
            manual_groups['Países Andinos'].append(country)
        elif country in caribbean:
            manual_groups['Caribe'].append(country)
        else:
            manual_groups['Outros Países Pequenos'].append(country)
    
    # Tratar casos especiais
    desconhecido_group = []
    otro_pais_group = []
    
    for name, _, _, _ in special_cases:
        if name == 'desconhecido':
            desconhecido_group.append(name)
        elif name == 'Otro país':
            otro_pais_group.append(name)
    
    # Adicionar casos especiais aos grupos
    if desconhecido_group:
        manual_groups['Desconhecido'] = desconhecido_group
    
    if otro_pais_group:
        manual_groups['Outros Países'] = otro_pais_group
    
    # Filtrar grupos vazios
    manual_groups = {k: v for k, v in manual_groups.items() if v}
    
    # Mostrar grupos
    for group_name, countries in manual_groups.items():
        print(f"Grupo manual '{group_name}': {', '.join(countries)}")
    
    return manual_groups

def combine_clustering_with_manual_groups(df_train, cluster_df, country_col, manual_groups):
    """
    Combina resultados de clustering com grupos manuais.
    
    Args:
        df_train: DataFrame com dados de treino
        cluster_df: DataFrame com resultados de clustering
        country_col: Nome da coluna de país
        manual_groups: Dict com grupos manuais
        
    Returns:
        Dict com todos os grupos finais
    """
    print("\n=== Combinando clustering com grupos manuais ===")
    
    all_groups = {}
    
    # Adicionar grupos de clustering se disponíveis
    if cluster_df is not None:
        for cluster_id in cluster_df['cluster'].unique():
            countries = cluster_df[cluster_df['cluster'] == cluster_id][country_col].tolist()
            # Verificar se o cluster tem países suficientes
            if countries:
                all_groups[f'Cluster {cluster_id}'] = countries
    
    # Adicionar grupos manuais
    for group_name, countries in manual_groups.items():
        # Verificar se os países existem no DataFrame
        existing_countries = [c for c in countries if c in df_train[country_col].values]
        if existing_countries:
            all_groups[group_name] = existing_countries
    
    # Mostrar grupos finais
    for group_name, countries in all_groups.items():
        country_count = len(countries)
        sample_count = df_train[df_train[country_col].isin(countries)].shape[0]
        print(f"Grupo final '{group_name}': {country_count} países, {sample_count} amostras")
    
    return all_groups

def split_data_by_country_groups(df_train, df_val, df_test, sufficient_countries, all_groups, column_dict):
    """Divide os dados por países e grupos de países."""
    print("\n=== Dividindo dados por grupos de países ===")
    
    country_col = column_dict['country']
    
    # Inicializar dicionário para armazenar dados por grupo
    country_groups = {}
    
    # 1. Países com dados suficientes (individuais)
    for country in sufficient_countries:
        # Filtar dados de treino
        train_country = df_train[df_train[country_col] == country].copy()
        
        # Filtrar dados de validação (se disponíveis)
        val_country = None
        if df_val is not None:
            if country_col in df_val.columns:  # Verificar se a coluna existe
                val_country = df_val[df_val[country_col] == country].copy()
        
        # Filtrar dados de teste (se disponíveis)
        test_country = None
        if df_test is not None:
            if country_col in df_test.columns:  # Verificar se a coluna existe
                test_country = df_test[df_test[country_col] == country].copy()
        
        # Usar uma versão limpa do nome do país para o nome do grupo
        country_str = str(country).replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Armazenar no dicionário
        country_groups[country_str] = {
            'train': train_country,
            'validation': val_country,
            'test': test_country,
            'type': 'individual'
        }
        
        print(f"Grupo individual para {country}: {len(train_country)} amostras de treino")
    
    # 2. Grupos de países com dados insuficientes
    for group_name, countries in all_groups.items():
        # Filtrar dados
        train_group = df_train[df_train[country_col].isin(countries)].copy()
        
        val_group = None
        if df_val is not None:
            if country_col in df_val.columns:  # Verificar se a coluna existe
                val_group = df_val[df_val[country_col].isin(countries)].copy()
        
        test_group = None
        if df_test is not None:
            if country_col in df_test.columns:  # Verificar se a coluna existe
                test_group = df_test[df_test[country_col].isin(countries)].copy()
        
        # Sanitizar nome do grupo
        group_str = group_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Armazenar no dicionário
        country_groups[group_str] = {
            'train': train_group,
            'validation': val_group,
            'test': test_group,
            'type': 'group',
            'countries': countries
        }
        
        print(f"Grupo '{group_name}': {len(countries)} países, {len(train_group)} amostras de treino")
    
    # 3. Modelo global (todos os dados)
    country_groups['global'] = {
        'train': df_train.copy(),
        'validation': df_val.copy() if df_val is not None else None,
        'test': df_test.copy() if df_test is not None else None,
        'type': 'global'
    }
    
    print(f"Grupo global: {len(df_train)} amostras de treino")
    
    return country_groups

def save_country_groups(country_groups, output_dir=OUTPUT_DIR):
    """Salva os grupos de países em arquivos separados."""
    print(f"\n=== Salvando dados por grupo de países em {output_dir} ===")
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar cada grupo
    for group_name, group_data in country_groups.items():
        # Criar diretório para o grupo - sanitizar nome
        safe_group_name = str(group_name).replace(" ", "_").replace("/", "_").replace("\\", "_")
        group_dir = os.path.join(output_dir, safe_group_name)
        os.makedirs(group_dir, exist_ok=True)
        
        # Salvar dados de treino
        if group_data['train'] is not None:
            train_path = os.path.join(group_dir, "train.csv")
            group_data['train'].to_csv(train_path, index=False)
        
        # Salvar dados de validação (se disponíveis)
        if group_data['validation'] is not None:
            val_path = os.path.join(group_dir, "validation.csv")
            group_data['validation'].to_csv(val_path, index=False)
        
        # Salvar dados de teste (se disponíveis)
        if group_data['test'] is not None:
            test_path = os.path.join(group_dir, "test.csv")
            group_data['test'].to_csv(test_path, index=False)
        
        # Salvar informações do grupo
        info = {
            'type': group_data['type'],
            'train_samples': len(group_data['train']) if group_data['train'] is not None else 0,
            'validation_samples': len(group_data['validation']) if group_data['validation'] is not None else 0,
            'test_samples': len(group_data['test']) if group_data['test'] is not None else 0
        }
        
        # Adicionar lista de países para grupos
        if 'countries' in group_data:
            info['countries'] = group_data['countries']
            
        # Salvar informações em formato JSON
        import json
        info_path = os.path.join(group_dir, "group_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Grupo {group_name} salvo em {group_dir}")

def main():
    # Configurar parser de argumentos para permitir override dos caminhos padrão
    parser = argparse.ArgumentParser(description='Agrupa países e divide dados para modelagem')
    parser.add_argument('--input-dir', default=INPUT_DIR, 
                      help=f'Diretório contendo os arquivos de entrada (padrão: {INPUT_DIR})')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, 
                      help=f'Diretório para salvar os resultados (padrão: {OUTPUT_DIR})')
    parser.add_argument('--min-samples', type=int, default=500, 
                      help='Mínimo de amostras para modelo individual')
    parser.add_argument('--min-conversions', type=int, default=20, 
                      help='Mínimo de conversões para modelo individual')
    
    args = parser.parse_args()
    
    try:
        print(f"Processando dados de {args.input_dir} para {args.output_dir}")
        
        # 1. Importar os datasets
        df_train, df_val, df_test = load_datasets(args.input_dir)
        
        # 2. Identificar colunas importantes
        column_dict = identify_columns(df_train)
        
        # Verificar se encontrou a coluna de país
        if not column_dict['country']:
            # Exibir as primeiras colunas do dataframe para ajudar na identificação manual
            print("\nColunas disponíveis no dataframe (primeiras 20):")
            for i, col in enumerate(df_train.columns[:20]):
                print(f"  {i}: {col}")
            
            # Solicitar ao usuário a coluna de país correta
            country_col_idx = input("\nInforme o índice da coluna de país: ")
            try:
                country_col = df_train.columns[int(country_col_idx)]
                column_dict['country'] = country_col
                print(f"Utilizando coluna '{country_col}' como coluna de país")
            except (ValueError, IndexError):
                raise ValueError("Índice de coluna inválido.")
        
        print(f"\nColunas identificadas:")
        print(f"  País: {column_dict['country']}")
        print(f"  Target: {column_dict['target']}")
        print(f"  Idade: {column_dict['age']}")
        print(f"  Features textuais: {len(column_dict['text_features'])} colunas")
        
        # 3. Filtrar países com dados suficientes e identificar casos especiais
        sufficient_countries, insufficient_countries, special_cases, country_stats = identify_countries_with_sufficient_data(
            df_train, column_dict, args.min_samples, args.min_conversions
        )
        
        # 4. Criar agrupamentos manuais para países específicos
        manual_groups = manual_country_grouping(insufficient_countries, special_cases)
        
        # 5. Extrair features e fazer clustering para países restantes
        # Identificar países que não estão em grupos manuais
        countries_to_cluster = []
        for country in insufficient_countries:
            in_manual_group = False
            for group_countries in manual_groups.values():
                if country in group_countries:
                    in_manual_group = True
                    break
            
            if not in_manual_group:
                countries_to_cluster.append(country)
        
        cluster_df = None
        if countries_to_cluster:
            # Extrair features
            cluster_df, feature_cols = extract_clustering_features(
                df_train, countries_to_cluster, column_dict
            )
            
            # Fazer clustering
            if cluster_df is not None and len(cluster_df) > 1:
                cluster_df = cluster_countries(
                    cluster_df, feature_cols, column_dict['country']
                )
        
        # 6. Combinar resultados de clustering com grupos manuais
        all_groups = combine_clustering_with_manual_groups(
            df_train, cluster_df, column_dict['country'], manual_groups
        )
        
        # 7. Dividir dados por grupos de países
        country_groups = split_data_by_country_groups(
            df_train, df_val, df_test, sufficient_countries, all_groups, column_dict
        )
        
        # 8. Salvar os dados por grupo
        save_country_groups(country_groups, args.output_dir)
        
        print(f"\n=== Processo concluído com sucesso ===")
        print(f"Dados divididos por país salvos em {args.output_dir}")
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())