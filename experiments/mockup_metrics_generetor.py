#!/usr/bin/env python
"""
Script para geração de mockup de dashboard para reunião.
Analisa UTMs, qualidade de leads e recomendação de investimento.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
from datetime import datetime, timedelta

# Configuração de paths absolutos
BASE_DIR = "/Users/ramonmoreira/desktop/smart_ads"
RAW_TEST_PATH = os.path.join(BASE_DIR, "data/01_split/test.csv")
PROCESSED_TEST_PATH = os.path.join(BASE_DIR, "data/02_3_processed/test.csv")
GMM_CALIB_DIR = os.path.join(BASE_DIR, "models/calibrated/gmm_calibrated_20250508_130725")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports/utm_analysis_mockup")

# Orçamento diário real
DAILY_BUDGET = 15000  # $15,000 por dia

# Valor médio do lead convertido em dólares
LEAD_VALUE = 497

# UTMs realistas
REALISTIC_UTMS = [
    'facebook_ads_remarketing', 
    'facebook_ads_lookalike',
    'facebook_ads_interest',
    'google_search_brand', 
    'google_search_competitors',
    'google_search_generic',
    'google_display_remarketing',
    'instagram_stories',
    'instagram_feed',
    'youtube_pre_roll'
]

# Criar diretório de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classe GMM_Wrapper necessária para carregar o modelo GMM
class GMM_Wrapper:
    """
    Classe wrapper para o GMM que implementa a API sklearn para calibração.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pca_model = pipeline['pca_model']
        self.gmm_model = pipeline['gmm_model']
        self.scaler_model = pipeline['scaler_model']
        self.cluster_models = pipeline['cluster_models']
        self.n_clusters = pipeline['n_clusters']
        self.threshold = pipeline.get('threshold', 0.15)
        
        # Adicionar atributos necessários para a API sklearn
        self.classes_ = np.array([0, 1])  # Classes binárias
        self._fitted = True  # Marcar como já ajustado
        self._estimator_type = "classifier"  # Indicar explicitamente que é um classificador
        
    def fit(self, X, y):
        # Como o modelo já está treinado, apenas verificamos as classes
        self.classes_ = np.unique(y)
        self._fitted = True
        return self
        
    def predict_proba(self, X):
        # Preparar os dados para o modelo GMM
        X_numeric = X.select_dtypes(include=['number'])
        
        # Substituir valores NaN por 0
        X_numeric = X_numeric.fillna(0)
        
        # Aplicar o scaler
        if hasattr(self.scaler_model, 'feature_names_in_'):
            # Garantir que temos exatamente as features esperadas pelo scaler
            scaler_features = self.scaler_model.feature_names_in_
            
            # Identificar features em X_numeric que não estão no scaler
            unseen_features = [col for col in X_numeric.columns if col not in scaler_features]
            if unseen_features:
                print(f"Removendo {len(unseen_features)} features não vistas durante treinamento")
                X_numeric = X_numeric.drop(columns=unseen_features)
            
            # Identificar features que faltam em X_numeric mas estão no scaler
            missing_features = [col for col in scaler_features if col not in X_numeric.columns]
            if missing_features:
                print(f"Adicionando {len(missing_features)} features ausentes vistas durante treinamento")
                for col in missing_features:
                    X_numeric[col] = 0.0
            
            # Garantir a ordem correta das colunas
            X_numeric = X_numeric[scaler_features]
        
        # Verificar novamente por NaNs após o ajuste de colunas
        X_numeric = X_numeric.fillna(0)
        
        X_scaled = self.scaler_model.transform(X_numeric)
        
        # Verificar por NaNs no array após scaling
        if np.isnan(X_scaled).any():
            # Se ainda houver NaNs, substitua-os por zeros
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # Aplicar PCA
        X_pca = self.pca_model.transform(X_scaled)
        
        # Aplicar GMM para obter cluster labels e probabilidades
        cluster_labels = self.gmm_model.predict(X_pca)
        cluster_probs = self.gmm_model.predict_proba(X_pca)
        
        # Inicializar array de probabilidades
        n_samples = len(X)
        y_pred_proba = np.zeros((n_samples, 2), dtype=float)
        
        # Para cada cluster, fazer previsões
        for cluster_id, model_info in self.cluster_models.items():
            # Converter cluster_id para inteiro se for string
            cluster_id_int = int(cluster_id) if isinstance(cluster_id, str) else cluster_id
            
            # Selecionar amostras deste cluster
            cluster_mask = (cluster_labels == cluster_id_int)
            
            if not any(cluster_mask):
                continue
            
            # Obter modelo específico do cluster
            model = model_info['model']
            
            # Detectar quais features o modelo espera
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                
                # Criar um DataFrame temporário com as features corretas
                X_temp = X.copy()
                
                # Lidar com features ausentes ou extras
                missing_features = [col for col in expected_features if col not in X.columns]
                for col in missing_features:
                    X_temp[col] = 0.0
                
                # Garantir a ordem correta das colunas
                features_to_use = [col for col in expected_features if col in X_temp.columns]
                X_cluster = X_temp.loc[cluster_mask, features_to_use].astype(float)
                
                # Substituir NaNs por zeros
                X_cluster = X_cluster.fillna(0)
            else:
                # Usar todas as features numéricas disponíveis
                X_cluster = X.loc[cluster_mask].select_dtypes(include=['number']).fillna(0)
            
            if len(X_cluster) > 0:
                # Fazer previsões
                try:
                    proba = model.predict_proba(X_cluster)
                    
                    # Armazenar resultados
                    y_pred_proba[cluster_mask] = proba
                except Exception as e:
                    print(f"ERRO ao fazer previsões para o cluster {cluster_id_int}: {e}")
                    # Em caso de erro, usar probabilidades default
                    y_pred_proba[cluster_mask, 0] = 0.9  # classe negativa (majoritária)
                    y_pred_proba[cluster_mask, 1] = 0.1  # classe positiva (minoritária)
        
        return y_pred_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

def load_gmm_model(model_dir):
    """
    Carrega um modelo GMM calibrado.
    
    Args:
        model_dir: Diretório do modelo
        
    Returns:
        Dict com modelo e threshold ou None se houver erro
    """
    model_path = os.path.join(model_dir, "gmm_calibrated.joblib")
    threshold_path = os.path.join(model_dir, "threshold.txt")
    
    try:
        gmm_model = joblib.load(model_path)
        with open(threshold_path, 'r') as f:
            gmm_threshold = float(f.read().strip())
        print(f"GMM calibrado carregado com threshold: {gmm_threshold:.4f}")
        return {'model': gmm_model, 'threshold': gmm_threshold}
    except Exception as e:
        print(f"Erro ao carregar GMM: {e}")
        return None

# Função para garantir que a coluna de UTM existe e é processada corretamente
def ensure_utm_columns(df):
    """Verifica e padroniza colunas de UTM no dataframe."""
    # Verificar coluna de UTM
    utm_columns = [col for col in df.columns if 'utm_' in col.lower() or 'UTM_' in col]
    
    if not utm_columns:
        print("AVISO: Nenhuma coluna de UTM encontrada. Criando coluna UTM_SOURCE com valores realistas.")
        # Criar UTM artificial com valores realistas
        df['UTM_SOURCE'] = np.random.choice(REALISTIC_UTMS, size=len(df))
        utm_columns = ['UTM_SOURCE']
    
    # Identificar a coluna principal de UTM (geralmente UTM_SOURCE)
    utm_main = [col for col in utm_columns if 'source' in col.lower()]
    utm_main = utm_main[0] if utm_main else utm_columns[0]
    
    # Garantir que não há valores ausentes
    if df[utm_main].isna().any():
        df[utm_main] = df[utm_main].fillna('unknown')
    
    # Verificar se temos UTMs realistas, caso contrário substituir
    unique_utms = df[utm_main].unique()
    if len(unique_utms) < 5 or all(len(utm) < 10 for utm in unique_utms):
        print("Substituindo UTMs por valores mais realistas...")
        
        # Criar mapeamento: UTMs antigas -> UTMs realistas
        mapping = {}
        for i, old_utm in enumerate(unique_utms):
            mapping[old_utm] = REALISTIC_UTMS[i % len(REALISTIC_UTMS)]
        
        # Aplicar mapeamento
        df[utm_main] = df[utm_main].map(mapping)
    
    return df, utm_main

def prepare_data_and_predict():
    """Carrega dados, prepara e faz predições."""
    print("Carregando datasets...")
    
    # Tentar carregar dados brutos
    try:
        raw_df = pd.read_csv(RAW_TEST_PATH)
        print(f"Dataset bruto carregado: {raw_df.shape}")
    except Exception as e:
        print(f"Erro ao carregar dataset bruto: {e}")
        print("Criando dataset simulado para demonstração...")
        # Criar dataset simulado se o original não estiver disponível
        raw_df = pd.DataFrame({
            'email': [f'user{i}@example.com' for i in range(1000)],
            'UTM_SOURCE': np.random.choice(REALISTIC_UTMS, size=1000),
            'UTM_MEDIUM': np.random.choice(['cpc', 'organic', 'social', 'email'], size=1000),
            'UTM_CAMPAIGN': np.random.choice(['campaign_a', 'campaign_b', 'campaign_c'], size=1000),
            'DATA': [datetime.now() - timedelta(days=random.randint(1, 30)) for _ in range(1000)]
        })
    
    # Garantir e processar colunas de UTM
    raw_df, utm_column = ensure_utm_columns(raw_df)
    
    # Obter lista de UTMs e contagem de leads
    utm_counts = raw_df[utm_column].value_counts().reset_index()
    utm_counts.columns = ['UTM', 'Total_Leads']
    
    # Carregar dados processados
    try:
        processed_df = pd.read_csv(PROCESSED_TEST_PATH)
        print(f"Dataset processado carregado: {processed_df.shape}")
    except Exception as e:
        print(f"Erro ao carregar dataset processado: {e}")
        print("Usando dataset bruto para demonstração...")
        processed_df = raw_df.copy()
        # Adicionar target artificial se não existir
        if 'target' not in processed_df.columns:
            processed_df['target'] = np.random.choice([0, 1], size=len(processed_df), p=[0.98, 0.02])
    
    # Verificar se o dataset processado também tem UTM
    processed_df, processed_utm_column = ensure_utm_columns(processed_df)
    
    # Carregar modelo GMM usando a mesma lógica do código anterior
    gmm_info = load_gmm_model(GMM_CALIB_DIR)
    
    # Fazer predições com o modelo GMM
    try:
        # Verificar se temos o modelo
        if gmm_info is not None:
            gmm_model = gmm_info['model']
            threshold = gmm_info['threshold']
            
            # Remover target se existir para predição
            X = processed_df.drop(columns=['target']) if 'target' in processed_df.columns else processed_df
            
            # Fazer predições
            print("Gerando predições com modelo GMM...")
            probs = gmm_model.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)
            
            # Adicionar predições ao dataframe
            processed_df['probability'] = probs
            processed_df['prediction'] = preds
        else:
            # Gerar probabilidades aleatórias para demonstração
            print("Usando probabilidades simuladas para demonstração...")
            processed_df['probability'] = np.random.beta(0.5, 5, size=len(processed_df))  # Distribuição desbalanceada
            threshold = 0.1  # Valor padrão
            processed_df['prediction'] = (processed_df['probability'] >= threshold).astype(int)
    except Exception as e:
        print(f"Erro durante predição: {e}")
        # Gerar probabilidades aleatórias para demonstração
        print("Usando probabilidades simuladas para demonstração...")
        processed_df['probability'] = np.random.beta(0.5, 5, size=len(processed_df))  # Distribuição desbalanceada
        threshold = 0.1  # Valor padrão
        processed_df['prediction'] = (processed_df['probability'] >= threshold).astype(int)
    
    # Criar decis de probabilidade
    processed_df['probability_decile'] = pd.qcut(processed_df['probability'], 10, labels=False, duplicates='drop') + 1
    
    # Adicionar datas simuladas se não existirem para tendência temporal
    if 'DATA' not in processed_df.columns:
        start_date = datetime.now() - timedelta(days=30)
        dates = [start_date + timedelta(days=i % 30) for i in range(len(processed_df))]
        processed_df['DATA'] = dates
    else:
        # Converter para datetime se já existir
        processed_df['DATA'] = pd.to_datetime(processed_df['DATA'], errors='coerce')
        # Preencher datas inválidas
        if processed_df['DATA'].isna().any():
            invalid_dates = processed_df['DATA'].isna()
            start_date = datetime.now() - timedelta(days=30)
            processed_df.loc[invalid_dates, 'DATA'] = [start_date + timedelta(days=i % 30) for i in range(sum(invalid_dates))]
    
    # Simular custos por lead realistas para cada UTM
    unique_utms = processed_df[processed_utm_column].unique()
    
    # Custos mais realistas baseados na plataforma
    utm_costs = {}
    for utm in unique_utms:
        utm_lower = utm.lower()
        if 'google_search' in utm_lower:
            # CPCs mais altos para Google Search
            utm_costs[utm] = round(random.uniform(3.0, 6.0), 2)
        elif 'google_display' in utm_lower:
            # CPCs médios para Google Display
            utm_costs[utm] = round(random.uniform(2.0, 4.0), 2)
        elif 'facebook' in utm_lower:
            # CPCs variados para Facebook
            if 'remarketing' in utm_lower:
                utm_costs[utm] = round(random.uniform(1.5, 3.0), 2)
            else:
                utm_costs[utm] = round(random.uniform(2.0, 4.5), 2)
        elif 'instagram' in utm_lower:
            # CPCs para Instagram
            utm_costs[utm] = round(random.uniform(2.5, 5.0), 2)
        elif 'youtube' in utm_lower:
            # CPCs para YouTube
            utm_costs[utm] = round(random.uniform(2.0, 4.0), 2)
        else:
            # Outros
            utm_costs[utm] = round(random.uniform(1.0, 6.0), 2)
    
    # Retornar todos os dados necessários
    return raw_df, processed_df, utm_counts, utm_column, processed_utm_column, utm_costs

def analyze_utm_performance(processed_df, utm_column, utm_costs):
    """Analisa performance por UTM e gera recomendações de investimento."""
    print("Analisando performance por UTM...")
    
    # Agrupar por UTM para análise
    utm_analysis = processed_df.groupby(utm_column).agg(
        total_leads=('probability', 'count'),
        qualified_leads=('prediction', 'sum'),
        avg_score=('probability', 'mean'),
        qualified_rate=('prediction', 'mean')
    ).reset_index()
    
    # Adicionar custo por lead para cada UTM
    utm_analysis['cost_per_lead'] = utm_analysis[utm_column].map(utm_costs)
    
    # Adicionar valor estimado (usando LEAD_VALUE e probabilidade média)
    utm_analysis['estimated_value_per_lead'] = utm_analysis['avg_score'] * LEAD_VALUE
    
    # Calcular ROI para cada UTM
    utm_analysis['roi'] = utm_analysis['estimated_value_per_lead'] / utm_analysis['cost_per_lead']
    
    # Calcular score de eficiência (ROI normalizado)
    max_roi = utm_analysis['roi'].max()
    utm_analysis['efficiency_score'] = utm_analysis['roi'] / max_roi if max_roi > 0 else 0
    
    # Analisar distribuição por decil para cada UTM
    decile_distribution = {}
    for utm in processed_df[utm_column].unique():
        utm_data = processed_df[processed_df[utm_column] == utm]
        decile_counts = utm_data['probability_decile'].value_counts().sort_index()
        decile_distribution[utm] = {
            'counts': decile_counts.to_dict(),
            'total': len(utm_data)
        }
    
    # Calcular tendência de qualidade ao longo do tempo
    processed_df['date'] = processed_df['DATA'].dt.date
    time_trend = processed_df.groupby([utm_column, 'date']).agg(
        avg_daily_score=('probability', 'mean'),
        total_daily_leads=('probability', 'count'),
        qualified_daily_leads=('prediction', 'sum')
    ).reset_index()
    
    # Calcular recomendação de investimento
    total_budget = DAILY_BUDGET  # $15,000 por dia
    
    # Método baseado no efficiency_score
    utm_analysis['recommended_budget'] = total_budget * utm_analysis['efficiency_score'] / utm_analysis['efficiency_score'].sum()
    
    # Arredondar para 2 casas decimais
    utm_analysis['recommended_budget'] = utm_analysis['recommended_budget'].round(2)
    
    # Ordenar por ROI (melhor estratégia de investimento)
    utm_analysis = utm_analysis.sort_values('roi', ascending=False)
    
    return utm_analysis, decile_distribution, time_trend

def create_visualizations(utm_analysis, decile_distribution, time_trend, utm_column):
    """Cria visualizações para as métricas."""
    print("Criando visualizações...")
    
    # Definir estilo
    sns.set(style="whitegrid")
    
    # 1. Gráfico de barras para o número total de leads por UTM
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x=utm_column, y='total_leads', data=utm_analysis, palette='Blues_d')
    
    # Adicionar valores nas barras
    for i, v in enumerate(utm_analysis['total_leads']):
        ax.text(i, v + 10, f"{int(v)}", ha='center', fontsize=10)
    
    plt.title('Número Total de Leads por UTM', fontsize=16)
    plt.xlabel('UTM', fontsize=12)
    plt.ylabel('Quantidade de Leads', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'total_leads_by_utm.png'), dpi=300)
    plt.close()
    
    # 2. Gráfico de barras para o score médio por UTM
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x=utm_column, y='avg_score', data=utm_analysis, palette='Greens_d')
    
    # Adicionar labels com valores
    for i, v in enumerate(utm_analysis['avg_score']):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
    
    plt.title('Score Médio por UTM', fontsize=16)
    plt.xlabel('UTM', fontsize=12)
    plt.ylabel('Score Médio (Probabilidade)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'avg_score_by_utm.png'), dpi=300)
    plt.close()
    
    # 3. Gráfico de distribuição por decil para algumas UTMs principais (top 4)
    top_utms = utm_analysis.head(min(4, len(utm_analysis)))[utm_column].tolist()
    
    for utm in top_utms:
        if utm in decile_distribution:
            plt.figure(figsize=(12, 7))
            
            deciles = list(range(1, 11))
            counts = [decile_distribution[utm]['counts'].get(d, 0) for d in deciles]
            total = decile_distribution[utm]['total']
            
            # Converter para percentagens
            percentages = [count / total * 100 if total > 0 else 0 for count in counts]
            
            ax = sns.barplot(x=deciles, y=percentages, palette='Blues')
            
            # Adicionar labels com contagens
            for i, (p, c) in enumerate(zip(percentages, counts)):
                ax.text(i, p + 0.5, f"{c} leads", ha='center', fontsize=9)
            
            plt.title(f'Distribuição por Decil - UTM: {utm}', fontsize=16)
            plt.xlabel('Decil de Probabilidade', fontsize=12)
            plt.ylabel('Porcentagem de Leads (%)', fontsize=12)
            plt.xticks(range(10), deciles)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'decile_distribution_{utm}.png'), dpi=300)
            plt.close()
    
    # 4. Gráfico de tendência temporal para algumas UTMs principais
    for utm in top_utms:
        utm_trend = time_trend[time_trend[utm_column] == utm].sort_values('date')
        
        if len(utm_trend) > 1:  # Precisamos de pelo menos 2 pontos para uma linha
            plt.figure(figsize=(14, 7))
            
            # Linha para score médio
            plt.plot(utm_trend['date'], utm_trend['avg_daily_score'], 'b-', marker='o', label='Score Médio')
            
            # Barras para leads qualificados
            plt.bar(utm_trend['date'], utm_trend['qualified_daily_leads'], alpha=0.3, color='g', label='Leads Qualificados')
            
            plt.title(f'Tendência de Qualidade ao Longo do Tempo - UTM: {utm}', fontsize=16)
            plt.xlabel('Data', fontsize=12)
            plt.ylabel('Score / Leads Qualificados', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'time_trend_{utm}.png'), dpi=300)
            plt.close()
    
    # 5. Gráfico de recomendação de investimento
    plt.figure(figsize=(14, 8))
    
    # Criar barras para recomendação
    ax = sns.barplot(x=utm_column, y='recommended_budget', data=utm_analysis, palette='Reds_d')
    
    # Adicionar labels com valores recomendados e ROI
    for i, (budget, roi) in enumerate(zip(utm_analysis['recommended_budget'], utm_analysis['roi'])):
        ax.text(i, budget + (DAILY_BUDGET * 0.02), f"${budget:.2f}\nROI: {roi:.2f}", ha='center', fontsize=9)
    
    plt.title(f'Recomendação de Investimento por UTM (Orçamento Total: ${DAILY_BUDGET:,})', fontsize=16)
    plt.xlabel('UTM', fontsize=12)
    plt.ylabel('Orçamento Recomendado ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'investment_recommendation.png'), dpi=300)
    plt.close()
    
    # 6. Tabela de recomendação de investimento (versão texto)
    fig, ax = plt.subplots(figsize=(14, len(utm_analysis) * 0.5 + 2))
    ax.axis('off')
    
    table_data = utm_analysis[[utm_column, 'total_leads', 'qualified_leads', 'avg_score', 
                            'cost_per_lead', 'roi', 'recommended_budget']]
    
    # Renomear colunas para exibição
    table_data.columns = ['UTM', 'Total Leads', 'Leads Qualificados', 'Score Médio', 
                        'Custo por Lead ($)', 'ROI', 'Orçamento Recomendado ($)']
    
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.1, 0.12, 0.12, 0.12, 0.1, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title(f'Tabela de Recomendação de Investimento (Orçamento Diário: ${DAILY_BUDGET:,})', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'investment_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizações criadas e salvas em: {OUTPUT_DIR}")

def main():
    print("=== Iniciando Geração de Mockup para Dashboard ===")
    
    # Carregar dados e fazer predições
    raw_df, processed_df, utm_counts, utm_column, processed_utm_column, utm_costs = prepare_data_and_predict()
    
    # Analisar performance por UTM
    utm_analysis, decile_distribution, time_trend = analyze_utm_performance(
        processed_df, processed_utm_column, utm_costs
    )
    
    # Criar visualizações
    create_visualizations(utm_analysis, decile_distribution, time_trend, processed_utm_column)
    
    # Salvar dados para uso posterior
    utm_analysis.to_csv(os.path.join(OUTPUT_DIR, 'utm_analysis.csv'), index=False)
    processed_df.to_csv(os.path.join(OUTPUT_DIR, 'processed_data_with_predictions.csv'), index=False)
    
    # Exibir tabela final com recomendação de investimento
    print(f"\n=== RECOMENDAÇÃO DE INVESTIMENTO (Orçamento Diário: ${DAILY_BUDGET:,}) ===")
    display_data = utm_analysis[[processed_utm_column, 'total_leads', 'qualified_leads', 
                              'avg_score', 'cost_per_lead', 'roi', 'recommended_budget']]
    display_data.columns = ['UTM', 'Total Leads', 'Leads Qualificados', 'Score Médio', 
                           'Custo/Lead ($)', 'ROI', 'Orçamento ($)']
    print(display_data.to_string(index=False))
    
    print(f"\nMockup completo! Todos os dados e visualizações foram salvos em: {OUTPUT_DIR}")
    print("Use estas visualizações para a sua apresentação aos gestores de tráfego.")

if __name__ == "__main__":
    main()