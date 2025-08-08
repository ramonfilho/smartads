"""
Módulo de treinamento para Gaussian Mixture Model (GMM).
Contém toda a lógica de treinamento separada do contexto __main__.
"""

import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import time
import json
import mlflow
import joblib
import contextlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Importar a classe GMM_Wrapper
from src.modeling.gmm_wrapper import GMM_Wrapper
from src.evaluation.mlflow_utils import setup_mlflow_tracking, find_optimal_threshold


class GMMTrainer:
    """
    Classe responsável pelo treinamento completo do modelo GMM.
    Encapsula toda a lógica para evitar problemas de serialização.
    """
    
    def __init__(self, config):
        """
        Inicializa o trainer com as configurações.
        
        Args:
            config: Dicionário com todas as configurações necessárias
        """
        self.config = config
        self.gmm_dir = None
        self.experiment_id = None
        
    @contextlib.contextmanager
    def safe_mlflow_run(self, experiment_id=None, run_name=None, nested=False):
        """Context manager para garantir que runs do MLflow sejam encerrados corretamente."""
        active_run = mlflow.active_run()
        if active_run and not nested:
            yield active_run
        else:
            run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=nested)
            try:
                yield run
            finally:
                mlflow.end_run()
    
    def create_directory_structure(self, use_optimized=True):
        """
        Cria a estrutura de diretórios para o experimento.
        
        Args:
            use_optimized: Se True, cria diretório para versão otimizada
        """
        # Diretórios base
        os.makedirs(self.config['mlflow_dir'], exist_ok=True)
        os.makedirs(self.config['artifact_dir'], exist_ok=True)
        
        # Diretório para GMM (diferenciado se otimizado)
        if use_optimized:
            self.gmm_dir = os.path.join(self.config['artifact_dir'], 'gmm_optimized_v2')
        else:
            self.gmm_dir = os.path.join(self.config['artifact_dir'], 'gmm_optimized')
        
        os.makedirs(self.gmm_dir, exist_ok=True)
        
        return {
            'gmm_dir': self.gmm_dir,
        }
    
    def load_data(self):
        """Carrega e prepara o dataset."""
        print("Carregando datasets...")
        
        # Caminhos dos arquivos
        train_path = os.path.join(self.config['data_dir'], "train.csv")
        val_path = os.path.join(self.config['data_dir'], "validation.csv")
        
        # Verificar existência
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(f"Datasets não encontrados em {self.config['data_dir']}")
        
        # Carregar dados
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        print(f"Dataset de treino: {train_df.shape}")
        print(f"Dataset de validação: {val_df.shape}")
        
        # Separar features e target
        target_col = 'target'
        
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        
        X_train = train_df.drop(columns=[target_col])
        X_val = val_df.drop(columns=[target_col])
        
        # Garantir colunas iguais
        common_cols = list(set(X_train.columns).intersection(set(X_val.columns)))
        X_train = X_train[common_cols]
        X_val = X_val[common_cols]
        
        print(f"Features comuns: {len(common_cols)}")
        print(f"Taxa de conversão (treino): {y_train.mean():.4f}")
        print(f"Taxa de conversão (validação): {y_val.mean():.4f}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_names': common_cols
        }
    
    def prepare_numeric_features(self, X_train, X_val):
        """Identifica e prepara features numéricas para clustering."""
        print("Identificando features numéricas...")
        
        # Identificar colunas numéricas
        numeric_cols = []
        for col in X_train.columns:
            try:
                X_train[col].astype(float)
                X_val[col].astype(float)
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        # Extrair e processar colunas numéricas
        X_train_numeric = X_train[numeric_cols].copy().astype(float)
        X_val_numeric = X_val[numeric_cols].copy().astype(float)
        
        # Substituir valores ausentes
        X_train_numeric.fillna(0, inplace=True)
        X_val_numeric.fillna(0, inplace=True)
        
        print(f"Total de features numéricas: {len(numeric_cols)}")
        
        return X_train_numeric, X_val_numeric, numeric_cols
    
    def apply_pca(self, X_train_numeric, X_val_numeric, variance_threshold=0.8, max_components=100):
        """Aplica PCA para redução de dimensionalidade."""
        print("Aplicando PCA...")
        random_state = self.config['model_params']['random_state']
        
        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_numeric)
        X_val_scaled = scaler.transform(X_val_numeric)
        
        # Determinar componentes
        max_possible = min(X_train_scaled.shape[0], X_train_scaled.shape[1]) - 1
        max_components = min(max_components, max_possible)
        
        # Análise de variância
        pca_analysis = PCA(n_components=max_components, random_state=random_state)
        pca_analysis.fit(X_train_scaled)
        
        # Calcular variância explicada
        cumulative_variance = np.cumsum(pca_analysis.explained_variance_ratio_)
        
        # Encontrar número ideal de componentes
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        n_components = max(n_components, 10)  # Garantir mínimo de 10 componentes
        
        print(f"Número ideal de componentes: {n_components} (variância explicada: {cumulative_variance[n_components-1]:.4f})")
        
        # Aplicar PCA final
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        
        print(f"Dimensão final após PCA: {X_train_pca.shape}")
        
        return X_train_pca, X_val_pca, pca, scaler
    
    def analyze_clusters(self, cluster_labels, y, prefix="treino"):
        """Analisa clusters quanto à taxa de conversão e distribuição."""
        # Verificar se os arrays têm o mesmo tamanho
        if len(cluster_labels) != len(y):
            print(f"AVISO: Tamanhos diferentes - cluster_labels: {len(cluster_labels)}, y: {len(y)}")
            # Ajustar para o menor tamanho
            min_size = min(len(cluster_labels), len(y))
            cluster_labels = cluster_labels[:min_size]
            if isinstance(y, pd.Series):
                y = y.iloc[:min_size]
            else:
                y = y[:min_size]
            print(f"  Ajustados para tamanho comum: {min_size}")
        
        # Adicionar labels de cluster e target a um DataFrame
        clusters_df = pd.DataFrame({
            'cluster': cluster_labels,
            'target': y.values if isinstance(y, pd.Series) else y
        })
        
        # Análise dos clusters
        print(f"\nAnálise dos clusters no conjunto de {prefix}:")
        cluster_stats = clusters_df.groupby('cluster').agg({
            'target': ['count', 'sum', 'mean']
        })
        cluster_stats.columns = ['samples', 'conversions', 'conversion_rate']
        print(cluster_stats)
        
        # Identificar cluster com menor taxa de conversão
        if not cluster_stats.empty:
            try:
                min_conv_cluster_id = cluster_stats['conversion_rate'].idxmin()
                min_conv_rate = cluster_stats.loc[min_conv_cluster_id, 'conversion_rate']
                print(f"\nCluster com menor taxa de conversão: {min_conv_cluster_id} ({min_conv_rate:.4f})")
            except:
                min_conv_cluster_id = None
                print("\nNão foi possível identificar o cluster com menor taxa de conversão.")
        else:
            min_conv_cluster_id = None
            print("\nNão foi possível identificar clusters com estatísticas válidas.")
        
        return cluster_stats, min_conv_cluster_id
    
    def evaluate_gmm_configuration(self, X_train_pca, y_train, X_val_pca, y_val, 
                                  n_components, covariance_type):
        """
        Avalia uma configuração específica do GMM.
        
        Returns:
            dict com métricas de avaliação
        """
        random_state = self.config['model_params']['random_state']
        
        # Treinar GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=200,
            n_init=5
        )
        
        try:
            gmm.fit(X_train_pca)
            
            # Obter clusters
            train_labels = gmm.predict(X_train_pca)
            val_labels = gmm.predict(X_val_pca)
            
            # Analisar clusters
            train_stats, _ = self.analyze_clusters(train_labels, y_train, prefix="treino")
            val_stats, _ = self.analyze_clusters(val_labels, y_val, prefix="validação")
            
            # Calcular métricas
            # BIC e AIC
            bic = gmm.bic(X_train_pca)
            aic = gmm.aic(X_train_pca)
            
            # Variância nas taxas de conversão (queremos alta variância = clusters bem separados)
            if not train_stats.empty:
                conversion_variance = train_stats['conversion_rate'].var()
            else:
                conversion_variance = 0
            
            # Log-likelihood
            log_likelihood = gmm.score(X_train_pca)
            
            return {
                'n_components': n_components,
                'covariance_type': covariance_type,
                'bic': bic,
                'aic': aic,
                'log_likelihood': log_likelihood,
                'conversion_variance': conversion_variance,
                'train_stats': train_stats,
                'val_stats': val_stats,
                'success': True
            }
            
        except Exception as e:
            print(f"Erro ao avaliar {n_components} componentes com {covariance_type}: {e}")
            return {
                'n_components': n_components,
                'covariance_type': covariance_type,
                'success': False,
                'error': str(e)
            }
    
    def find_optimal_gmm_params(self, X_train_pca, y_train, X_val_pca, y_val):
        """
        Busca os parâmetros ótimos do GMM para o dataset atual.
        
        Returns:
            dict com os melhores parâmetros encontrados
        """
        print("\n" + "="*80)
        print("INICIANDO BUSCA DE PARÂMETROS ÓTIMOS DO GMM")
        print("="*80)
        
        results = []
        
        # Grid search
        for n_comp in self.config['param_search']['n_components_range']:
            for cov_type in self.config['param_search']['covariance_types']:
                print(f"\nTestando: n_components={n_comp}, covariance_type={cov_type}")
                
                result = self.evaluate_gmm_configuration(
                    X_train_pca, y_train, X_val_pca, y_val,
                    n_comp, cov_type
                )
                
                if result['success']:
                    results.append(result)
                    print(f"  BIC: {result['bic']:.2f}")
                    print(f"  AIC: {result['aic']:.2f}")
                    print(f"  Conversion variance: {result['conversion_variance']:.6f}")
        
        # Converter para DataFrame para análise
        results_df = pd.DataFrame([r for r in results if r['success']])
        
        if results_df.empty:
            print("\nNENHUMA configuração foi bem-sucedida! Usando parâmetros originais.")
            return self.config['gmm_params_original']
        
        # Salvar resultados da busca
        results_df.to_csv(os.path.join(self.gmm_dir, 'param_search_results.csv'), index=False)
        
        # Visualizar resultados
        self._plot_param_search_results(results_df)
        
        # Selecionar melhor configuração
        # Estratégia: balancear BIC baixo com boa separação de clusters
        # NOTA: Ajustando peso da variância devido aos valores muito pequenos observados
        results_df['score'] = -results_df['bic'] + 1e8 * results_df['conversion_variance']  # Aumentado de 1e6 para 1e8
        best_idx = results_df['score'].idxmax()
        best_params = results_df.loc[best_idx]
        
        optimal_params = {
            'n_components': int(best_params['n_components']),
            'covariance_type': best_params['covariance_type']
        }
        
        print("\n" + "="*80)
        print("RESULTADOS DA BUSCA DE PARÂMETROS")
        print("="*80)
        print(f"Melhor configuração encontrada:")
        print(f"  n_components: {optimal_params['n_components']}")
        print(f"  covariance_type: {optimal_params['covariance_type']}")
        print(f"  BIC: {best_params['bic']:.2f}")
        print(f"  AIC: {best_params['aic']:.2f}")
        print(f"  Variância de conversão: {best_params['conversion_variance']:.6f}")
        
        # Comparar com parâmetros originais
        print(f"\nParâmetros originais:")
        print(f"  n_components: {self.config['gmm_params_original']['n_components']}")
        print(f"  covariance_type: {self.config['gmm_params_original']['covariance_type']}")
        
        # Salvar comparação (removendo DataFrames que não são serializáveis)
        # Converter results_df removendo colunas problemáticas
        results_for_json = results_df.copy()
        # Remover colunas que contêm DataFrames
        if 'train_stats' in results_for_json.columns:
            results_for_json = results_for_json.drop(columns=['train_stats'])
        if 'val_stats' in results_for_json.columns:
            results_for_json = results_for_json.drop(columns=['val_stats'])
        
        comparison = {
            'optimal_params': optimal_params,
            'optimal_metrics': {
                'bic': float(best_params['bic']),
                'aic': float(best_params['aic']),
                'conversion_variance': float(best_params['conversion_variance'])
            },
            'original_params': self.config['gmm_params_original'],
            'all_results': results_for_json.to_dict('records')
        }
        
        with open(os.path.join(self.gmm_dir, 'param_optimization_report.json'), 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return optimal_params
    
    def _plot_param_search_results(self, results_df):
        """Gera visualizações dos resultados da busca de parâmetros."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: BIC por configuração
        ax = axes[0, 0]
        for cov_type in self.config['param_search']['covariance_types']:
            data = results_df[results_df['covariance_type'] == cov_type]
            if not data.empty:
                ax.plot(data['n_components'], data['bic'], marker='o', label=cov_type)
        ax.set_xlabel('Número de Componentes')
        ax.set_ylabel('BIC (menor é melhor)')
        ax.set_title('BIC por Configuração')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: AIC por configuração
        ax = axes[0, 1]
        for cov_type in self.config['param_search']['covariance_types']:
            data = results_df[results_df['covariance_type'] == cov_type]
            if not data.empty:
                ax.plot(data['n_components'], data['aic'], marker='o', label=cov_type)
        ax.set_xlabel('Número de Componentes')
        ax.set_ylabel('AIC (menor é melhor)')
        ax.set_title('AIC por Configuração')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Variância de conversão
        ax = axes[1, 0]
        for cov_type in self.config['param_search']['covariance_types']:
            data = results_df[results_df['covariance_type'] == cov_type]
            if not data.empty:
                ax.plot(data['n_components'], data['conversion_variance'], marker='o', label=cov_type)
        ax.set_xlabel('Número de Componentes')
        ax.set_ylabel('Variância nas Taxas de Conversão')
        ax.set_title('Separação dos Clusters')
        ax.legend()
        ax.grid(True)
        
        # Plot 4: Heatmap resumo
        ax = axes[1, 1]
        pivot_bic = results_df.pivot(index='n_components', columns='covariance_type', values='bic')
        sns.heatmap(pivot_bic, annot=True, fmt='.0f', cmap='YlOrRd_r', ax=ax)
        ax.set_title('Heatmap BIC (menor é melhor)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.gmm_dir, 'param_search_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_cluster_models(self, X_train, y_train, cluster_labels_train, gmm_params):
        """
        Treina modelos específicos para cada cluster.
        """
        max_depth = self.config['model_params']['max_depth']
        n_estimators = self.config['model_params']['n_estimators']
        random_state = self.config['model_params']['random_state']
        
        # Estrutura para armazenar modelos
        cluster_models = []
        
        # Filtrar features numéricas
        numeric_cols = []
        for col in X_train.columns:
            try:
                X_train[col].astype(float)
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        X_train_numeric = X_train[numeric_cols].copy().astype(float)
        print(f"  Usando {len(numeric_cols)} features numéricas para treinamento")
        
        # Treinar modelos para cada cluster
        with self.safe_mlflow_run(experiment_id=self.experiment_id, 
                                 run_name=f"gmm_cluster_models_{gmm_params['n_components']}_{gmm_params['covariance_type']}") as parent_run:
            
            mlflow.log_params({
                "algorithm": "GaussianMixture",
                "n_components": gmm_params['n_components'],
                "covariance_type": gmm_params['covariance_type'],
                "max_depth": str(max_depth),
                "n_estimators": n_estimators,
                "random_state": random_state
            })
            
            # Identificar clusters únicos
            unique_clusters = np.unique(cluster_labels_train)
            
            for cluster_id in unique_clusters:
                print(f"\n{'='*50}")
                print(f"Treinando modelo para cluster {cluster_id}...")
                
                # Selecionar dados do cluster atual
                cluster_mask = (cluster_labels_train == cluster_id)
                X_train_cluster = X_train_numeric[cluster_mask]
                y_train_cluster = y_train[cluster_mask]
                
                # Verificar se há dados suficientes
                print(f"  Amostras de treino no cluster: {len(X_train_cluster)}")
                print(f"  Taxa de conversão (treino): {y_train_cluster.mean():.4f}")
                
                # Pular clusters com poucos dados
                min_samples = 100
                if len(X_train_cluster) < min_samples:
                    print(f"  Cluster {cluster_id} tem poucos dados (<{min_samples}), pulando treinamento.")
                    continue
                
                # Calcular peso da classe minoritária para balanceamento
                conversion_rate = y_train_cluster.mean()
                if conversion_rate > 0 and conversion_rate < 0.5:
                    # Scale negativo para positivo
                    scale_pos_weight = (1 - conversion_rate) / conversion_rate
                    
                    # Se taxa for muito baixa, limitar o peso
                    if scale_pos_weight > 100:
                        scale_pos_weight = 100
                        
                    class_weight = {0: 1, 1: float(scale_pos_weight)}
                    print(f"  Aplicando class_weight: {class_weight}")
                else:
                    class_weight = None
                
                # Criar e treinar modelo
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    class_weight=class_weight,
                    random_state=random_state,
                    n_jobs=-1
                )
                
                try:
                    model.fit(X_train_cluster, y_train_cluster)
                    
                    # Armazenar informações
                    model_info = {
                        "cluster_id": int(cluster_id),
                        "model": model,
                        "threshold": 0.5,  # Será otimizado depois
                        "features": numeric_cols,
                        "n_samples": len(X_train_cluster),
                        "conversion_rate": float(conversion_rate)
                    }
                    
                    # Salvar modelo localmente
                    model_path = os.path.join(self.gmm_dir, f"cluster_{cluster_id}_model.joblib")
                    joblib.dump(model, model_path)
                    
                    # Adicionar à lista
                    cluster_models.append(model_info)
                    
                    # Registrar metadados
                    mlflow.log_params({
                        f"cluster_{cluster_id}_samples": len(X_train_cluster),
                        f"cluster_{cluster_id}_conversion_rate": float(conversion_rate),
                        f"cluster_{cluster_id}_class_weight": str(class_weight)
                    })
                    
                    # Registrar modelo no MLflow
                    mlflow.sklearn.log_model(model, f"gmm_cluster_{cluster_id}_model")
                    
                except Exception as e:
                    print(f"  Erro ao treinar modelo para cluster {cluster_id}: {e}")
        
        return cluster_models
    
    def evaluate_ensemble(self, X_val, y_val, cluster_labels_val, cluster_models, gmm_params):
        """
        Avalia o ensemble de modelos usando threshold global.
        """
        # Criar dicionário cluster -> modelo
        cluster_model_map = {model["cluster_id"]: model for model in cluster_models}
        
        # Arrays para previsões
        y_pred_proba = np.zeros_like(y_val, dtype=float)
        
        # Fazer previsões por cluster
        for cluster_id, model_info in cluster_model_map.items():
            # Selecionar amostras
            cluster_mask = (cluster_labels_val == cluster_id)
            
            if not any(cluster_mask):
                continue
                
            # Obter modelo e features
            model = model_info["model"]
            features = model_info["features"]
            
            # Extrair features relevantes
            X_cluster = X_val[features][cluster_mask].astype(float)
            
            if len(X_cluster) > 0:
                # Prever probabilidades
                proba = model.predict_proba(X_cluster)[:, 1]
                y_pred_proba[cluster_mask] = proba
        
        # Encontrar threshold ótimo
        threshold_results = find_optimal_threshold(y_val, y_pred_proba)
        best_threshold = threshold_results['best_threshold']
        
        # Aplicar threshold
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        # Calcular métricas
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        
        # Registrar métricas
        with self.safe_mlflow_run(experiment_id=self.experiment_id, 
                                 run_name=f"gmm_evaluation_{gmm_params['n_components']}_{gmm_params['covariance_type']}"):
            mlflow.log_metrics({
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "threshold": best_threshold,
                "n_components": gmm_params['n_components']
            })
            
            # Salvar matriz de confusão no MLflow como JSON
            mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")
        
        # Imprimir resultados
        print(f"\nResultados do Ensemble com GMM:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Threshold: {best_threshold:.4f}")
        print("\nMatrix de Confusão:")
        print(cm)
        
        # Salvar resultados localmente
        results = {
            "algorithm": "GaussianMixture",
            "gmm_params": gmm_params,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(best_threshold),
            "confusion_matrix": cm.tolist()
        }
        
        with open(os.path.join(self.gmm_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run(self, use_param_search=True):
        """
        Executa o pipeline completo de treinamento do GMM.
        
        Args:
            use_param_search: Se True, busca parâmetros ótimos. Se False, usa originais.
            
        Returns:
            dict com resultados do treinamento
        """
        print("Iniciando clustering com GMM...")
        print(f"Busca de parâmetros: {'HABILITADA' if use_param_search else 'DESABILITADA'}")
        
        # Criar estrutura de diretórios
        self.create_directory_structure(use_optimized=use_param_search)
        
        # Limpar runs MLflow ativos
        active_run = mlflow.active_run()
        if active_run:
            print(f"Encerrando run MLflow ativo: {active_run.info.run_id}")
            mlflow.end_run()
        
        # Configurar MLflow
        self.experiment_id = setup_mlflow_tracking(
            tracking_dir=self.config['mlflow_dir'],
            experiment_name=self.config['experiment_name'],
            clean_previous=False
        )
        
        # Carregar dados
        data = self.load_data()
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        
        # Preparar features numéricas
        X_train_numeric, X_val_numeric, numeric_cols = self.prepare_numeric_features(X_train, X_val)
        
        # Aplicar PCA
        X_train_pca, X_val_pca, pca_model, scaler = self.apply_pca(X_train_numeric, X_val_numeric)
        
        # Determinar parâmetros do GMM
        if use_param_search and self.config['param_search']['enable_search']:
            # Buscar parâmetros ótimos
            gmm_params = self.find_optimal_gmm_params(X_train_pca, y_train, X_val_pca, y_val)
        else:
            # Usar parâmetros originais
            gmm_params = self.config['gmm_params_original']
            print(f"\nUsando parâmetros originais:")
            print(f"  n_components: {gmm_params['n_components']}")
            print(f"  covariance_type: {gmm_params['covariance_type']}")
        
        # Criar GMM com parâmetros escolhidos
        print(f"\nCriando modelo GMM com parâmetros finais:")
        print(f"  n_components: {gmm_params['n_components']}")
        print(f"  covariance_type: {gmm_params['covariance_type']}")
        
        gmm = GaussianMixture(
            n_components=gmm_params['n_components'],
            covariance_type=gmm_params['covariance_type'],
            random_state=self.config['model_params']['random_state'],
            max_iter=200,
            n_init=5
        )
        
        # Treinar GMM
        print("\nTreinando modelo GMM...")
        start_time = time.time()
        gmm.fit(X_train_pca)
        training_time = time.time() - start_time
        print(f"  Tempo de treinamento: {training_time:.2f} segundos")
        
        # Obter clusters
        print("\nAtribuindo clusters aos dados...")
        train_labels = gmm.predict(X_train_pca)
        val_labels = gmm.predict(X_val_pca)
        
        # Analisar clusters
        train_stats, _ = self.analyze_clusters(train_labels, y_train, prefix="treino")
        val_stats, _ = self.analyze_clusters(val_labels, y_val, prefix="validação")
        
        # Treinar modelos por cluster
        print("\nTreinando modelos específicos por cluster...")
        cluster_models = self.train_cluster_models(
            X_train, y_train, train_labels, gmm_params
        )
        
        # Avaliar ensemble
        print("\nAvaliando ensemble de modelos...")
        if cluster_models:
            results = self.evaluate_ensemble(X_val, y_val, val_labels, cluster_models, gmm_params)
            
            # Criar pipeline para o GMM_Wrapper
            pipeline = {
                'pca_model': pca_model,
                'gmm_model': gmm,
                'scaler_model': scaler,
                'cluster_models': {model["cluster_id"]: {"model": model["model"], "threshold": model["threshold"]} 
                                  for model in cluster_models},
                'n_clusters': gmm_params['n_components'],
                'threshold': results['threshold']
            }
            
            # Criar instância do GMM_Wrapper
            gmm_wrapper = GMM_Wrapper(pipeline)
            
            # Salvar o wrapper usando joblib
            wrapper_path = os.path.join(self.gmm_dir, 'gmm_wrapper.joblib')
            joblib.dump(gmm_wrapper, wrapper_path)
            print(f"GMM_Wrapper salvo em: {wrapper_path}")
            
            # Salvar componentes individuais para compatibilidade
            joblib.dump(gmm, os.path.join(self.gmm_dir, 'gmm_model.joblib'))
            joblib.dump(pca_model, os.path.join(self.gmm_dir, 'pca_model.joblib'))
            joblib.dump(scaler, os.path.join(self.gmm_dir, 'scaler_model.joblib'))
            
            # Salvar estatísticas dos clusters
            train_stats.to_json(os.path.join(self.gmm_dir, 'train_cluster_stats.json'))
            val_stats.to_json(os.path.join(self.gmm_dir, 'val_cluster_stats.json'))
        else:
            print("Nenhum modelo de cluster criado. Não é possível avaliar o ensemble.")
            results = None
        
        # Baseline para comparação
        baseline_precision = 0.94
        baseline_recall = 0.27
        baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall)
        baseline_f1 = round(baseline_f1, 4)
        
        # Comparar com baseline
        if results:
            print("\nComparação com o Baseline:")
            print(f"  Baseline: Precision={baseline_precision:.4f}, Recall={baseline_recall:.4f}, F1={baseline_f1:.4f}")
            print(f"  GMM:      Precision={results['precision']:.4f}, Recall={results['recall']:.4f}, F1={results['f1']:.4f}")
            
            if results['f1'] > baseline_f1:
                print("  GMM tem desempenho superior ao baseline!")
            else:
                print("  GMM tem desempenho inferior ao baseline.")
        
        # Salvar informações de configuração
        with open(os.path.join(self.gmm_dir, 'experiment_config.json'), 'w') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'n_samples_train': len(X_train),
                'n_samples_val': len(X_val),
                'n_features': len(numeric_cols),
                'n_pca_components': X_train_pca.shape[1],
                'conversion_rate_train': float(y_train.mean()),
                'conversion_rate_val': float(y_val.mean()),
                'gmm_params': gmm_params,
                'param_search_enabled': use_param_search,
                'model_params': self.config['model_params'],
                'training_time': training_time,
                'baseline': {
                    'precision': baseline_precision,
                    'recall': baseline_recall,
                    'f1': baseline_f1
                }
            }, f, indent=2)
        
        print("\nExperimento GMM concluído com sucesso!")
        print(f"Resultados salvos em: {self.gmm_dir}")
        print(f"Logs MLflow em: {self.config['mlflow_dir']}")
        
        return results