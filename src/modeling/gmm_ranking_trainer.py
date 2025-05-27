"""
Módulo de treinamento GMM otimizado para ranking.
Separa toda lógica de treinamento para evitar problemas de serialização.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import lightgbm as lgb
import time
import json
import mlflow
import joblib
import contextlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Importar a classe GMM_Wrapper
from src.modeling.gmm_wrapper import GMM_Wrapper
from src.utils.mlflow_utils import setup_mlflow_tracking


class GMMRankingTrainer:
    """
    Classe responsável pelo treinamento do GMM otimizado para ranking.
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
    
    def create_directory_structure(self):
        """
        Cria a estrutura de diretórios para o experimento.
        """
        # Diretórios base
        os.makedirs(self.config['mlflow_dir'], exist_ok=True)
        os.makedirs(self.config['artifact_dir'], exist_ok=True)
        
        # Diretório para GMM
        self.gmm_dir = os.path.join(self.config['artifact_dir'], 'gmm_ranking_optimized')
        os.makedirs(self.gmm_dir, exist_ok=True)
        
        return {'gmm_dir': self.gmm_dir}
    
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
    
    def prepare_features_for_ranking(self, X_train, X_val, y_train):
        """
        Preparação melhorada de features com foco em ranking.
        """
        print("\nPreparando features para ranking...")
        
        # Identificar colunas numéricas
        numeric_cols = []
        for col in X_train.columns:
            try:
                X_train[col].astype(float)
                X_val[col].astype(float)
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        print(f"Features numéricas identificadas: {len(numeric_cols)}")
        
        # Converter para numérico
        X_train_numeric = X_train[numeric_cols].astype(float).fillna(0)
        X_val_numeric = X_val[numeric_cols].astype(float).fillna(0)
        
        # 1. Remover baixa variância
        print("Removendo features de baixa variância...")
        selector = VarianceThreshold(threshold=0.01)
        X_train_var = selector.fit_transform(X_train_numeric)
        X_val_var = selector.transform(X_val_numeric)
        cols_after_var = np.array(numeric_cols)[selector.get_support()].tolist()
        print(f"  Features após filtro de variância: {X_train_var.shape[1]}")
        
        # 2. Remover alta correlação
        print("Removendo features altamente correlacionadas...")
        corr_matrix = pd.DataFrame(X_train_var, columns=cols_after_var).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        cols_after_corr = [col for col in cols_after_var if col not in to_drop]
        X_train_corr = pd.DataFrame(X_train_var, columns=cols_after_var)[cols_after_corr].values
        X_val_corr = pd.DataFrame(X_val_var, columns=cols_after_var)[cols_after_corr].values
        print(f"  Features após remover correlações: {X_train_corr.shape[1]}")
        
        # 3. SelectKBest para reduzir ainda mais
        print("Aplicando SelectKBest...")
        k_features = min(500, X_train_corr.shape[1])
        selector_k = SelectKBest(f_classif, k=k_features)
        X_train_selected = selector_k.fit_transform(X_train_corr, y_train)
        X_val_selected = selector_k.transform(X_val_corr)
        
        # Manter nomes das features selecionadas
        selected_features = np.array(cols_after_corr)[selector_k.get_support()].tolist()
        print(f"  Features finais selecionadas: {X_train_selected.shape[1]}")
        
        return X_train_selected, X_val_selected, selected_features
    
    def optimize_pca_for_ranking(self, X_train_scaled, y_train, variance_thresholds=[0.90, 0.95, 0.98, 0.99]):
        """
        Otimiza número de componentes PCA focando em métricas de ranking.
        """
        print("\nOtimizando PCA para ranking...")
        
        best_score = -np.inf
        best_n_components = None
        best_variance = None
        
        max_components = min(100, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1] - 1)
        
        for var_threshold in variance_thresholds:
            # PCA com threshold de variância
            pca_temp = PCA(n_components=max_components, random_state=self.config['random_state'])
            pca_temp.fit(X_train_scaled)
            
            cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= var_threshold) + 1
            n_components = max(n_components, 150)
            
            # Transformar dados
            X_pca = pca_temp.transform(X_train_scaled)[:, :n_components]
            
            # Avaliar com modelo simples
            rf = RandomForestClassifier(
                n_estimators=50, 
                max_depth=5, 
                random_state=self.config['random_state'],
                n_jobs=-1
            )
            rf.fit(X_pca, y_train)
            
            # Calcular GINI (proxy para AUC)
            y_pred_proba = rf.predict_proba(X_pca)[:, 1]
            gini = 2 * self._calculate_auc_fast(y_train, y_pred_proba) - 1
            
            print(f"  Variância {var_threshold:.2f} → {n_components} componentes → GINI: {gini:.4f}")
            
            if gini > best_score:
                best_score = gini
                best_n_components = n_components
                best_variance = var_threshold
        
        print(f"\nMelhor configuração PCA: {best_n_components} componentes (var={best_variance:.2f}, GINI={best_score:.4f})")
        
        return best_n_components
    
    def _calculate_auc_fast(self, y_true, y_scores):
        """Cálculo rápido de AUC sem sklearn."""
        # Ordenar por score
        order = np.argsort(y_scores)[::-1]
        y_true_ordered = y_true.iloc[order] if hasattr(y_true, 'iloc') else y_true[order]
        
        # Calcular AUC usando método trapezoidal
        tpr = np.cumsum(y_true_ordered) / np.sum(y_true_ordered)
        fpr = np.cumsum(~y_true_ordered) / np.sum(~y_true_ordered)
        
        # Adicionar ponto (0,0)
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        
        # Calcular área
        auc = np.trapz(tpr, fpr)
        return auc
    
    def evaluate_ranking_metrics(self, y_true, y_pred_proba):
        """
        Calcula métricas específicas para ranking.
        """
        # Criar DataFrame para análise
        df = pd.DataFrame({
            'y_true': y_true,
            'y_proba': y_pred_proba
        })
        
        # 1. Análise por decil
        df['decile'] = pd.qcut(df['y_proba'].rank(method='first'), 
                               q=10, labels=range(1, 11))
        
        decile_stats = df.groupby('decile').agg({
            'y_true': ['sum', 'count', 'mean']
        })
        decile_stats.columns = ['conversions', 'total', 'rate']
        
        # Calcular lift
        overall_rate = df['y_true'].mean()
        decile_stats['lift'] = decile_stats['rate'] / overall_rate
        
        # 2. KS Statistic
        df_sorted = df.sort_values('y_proba', ascending=False)
        
        # Proporções acumuladas
        cum_pos = np.cumsum(df_sorted['y_true']) / df['y_true'].sum()
        cum_neg = np.cumsum(~df_sorted['y_true']) / (~df['y_true']).sum()
        
        ks_statistic = np.max(np.abs(cum_pos - cum_neg))
        
        # 3. Gini coefficient
        auc = self._calculate_auc_fast(y_true, y_pred_proba)
        gini = 2 * auc - 1
        
        # 4. Top-K metrics
        n_samples = len(df)
        top_10_pct = int(n_samples * 0.1)
        top_20_pct = int(n_samples * 0.2)
        
        df_sorted_desc = df.sort_values('y_proba', ascending=False)
        
        metrics = {
            'gini': gini,
            'ks_statistic': ks_statistic,
            'top_decile_lift': decile_stats.loc[10, 'lift'],
            'top_2deciles_lift': decile_stats.loc[[9, 10], 'lift'].mean(),
            'top_10pct_recall': df_sorted_desc.head(top_10_pct)['y_true'].sum() / df['y_true'].sum(),
            'top_20pct_recall': df_sorted_desc.head(top_20_pct)['y_true'].sum() / df['y_true'].sum(),
            'monotonicity_violations': self._check_monotonicity(decile_stats['rate'])
        }
        
        return metrics, decile_stats
    
    def _check_monotonicity(self, rates):
        """Verifica violações de monotonicidade (taxa deve decrescer)."""
        violations = 0
        rates_list = rates.tolist()
        for i in range(len(rates_list) - 1):
            if rates_list[i] < rates_list[i + 1]:  # Taxa aumentou
                violations += 1
        return violations
    
    def find_optimal_gmm_for_ranking(self, X_train_pca, y_train, X_val_pca, y_val):
        """
        Busca parâmetros ótimos do GMM focando em métricas de ranking.
        """
        print("\n" + "="*80)
        print("BUSCANDO PARÂMETROS ÓTIMOS DO GMM PARA RANKING")
        print("="*80)
        
        results = []
        
        for n_comp in self.config['param_search']['n_components_range']:
            for cov_type in self.config['param_search']['covariance_types']:
                print(f"\nTestando: n_components={n_comp}, covariance_type={cov_type}")
                
                try:
                    # Treinar GMM
                    gmm = GaussianMixture(
                        n_components=n_comp,
                        covariance_type=cov_type,
                        reg_covar=1e-6,
                        random_state=self.config['random_state'],
                        max_iter=200,
                        n_init=5
                    )
                    gmm.fit(X_train_pca)
                    
                    # Obter clusters
                    train_labels = gmm.predict(X_train_pca)
                    val_labels = gmm.predict(X_val_pca)
                    
                    # Treinar modelo simples por cluster para avaliação
                    cluster_proba = np.zeros_like(y_val, dtype=float)
                    valid_clusters = 0
                    
                    for cluster_id in np.unique(train_labels):
                        mask_train = train_labels == cluster_id
                        mask_val = val_labels == cluster_id
                        
                        if mask_train.sum() < 50 or mask_val.sum() < 10:
                            continue
                        
                        # Verificar se há ambas as classes no cluster de treino
                        y_cluster_train = y_train.iloc[mask_train]
                        n_classes = len(np.unique(y_cluster_train))
                        
                        if n_classes < 2:
                            # Cluster tem apenas uma classe
                            print(f"  Cluster {cluster_id}: apenas classe {y_cluster_train.iloc[0]} - pulando")
                            # Atribuir probabilidade baseada na classe predominante
                            if y_cluster_train.iloc[0] == 0:
                                cluster_proba[mask_val] = 0.01  # Baixa probabilidade
                            else:
                                cluster_proba[mask_val] = 0.99  # Alta probabilidade
                            continue
                        
                        # Modelo simples para avaliação
                        rf = RandomForestClassifier(
                            n_estimators=50,
                            max_depth=5,
                            random_state=self.config['random_state'],
                            class_weight='balanced'  # Importante para dados desbalanceados
                        )
                        
                        try:
                            rf.fit(X_train_pca[mask_train], y_cluster_train)
                            
                            if mask_val.sum() > 0:
                                # Verificar formato da saída
                                proba = rf.predict_proba(X_val_pca[mask_val])
                                if proba.shape[1] == 2:
                                    cluster_proba[mask_val] = proba[:, 1]
                                else:
                                    # Fallback se apenas uma classe
                                    cluster_proba[mask_val] = proba[:, 0]
                                
                                valid_clusters += 1
                        except Exception as e:
                            print(f"  Erro ao treinar modelo para cluster {cluster_id}: {e}")
                            # Atribuir probabilidade padrão
                            cluster_proba[mask_val] = y_train.mean()  # Taxa base
                    
                    # Verificar se temos clusters válidos suficientes
                    if valid_clusters < 2:
                        print(f"  Apenas {valid_clusters} clusters válidos - configuração ruim")
                        continue
                    
                    # Avaliar métricas de ranking
                    metrics, decile_stats = self.evaluate_ranking_metrics(y_val, cluster_proba)
                    
                    # Score composto para ranking
                    ranking_score = (
                    metrics['gini'] * 100 +  
                    min(metrics['top_decile_lift'], 5) * 20 +  # Aumentar peso do lift
                    metrics['top_20pct_recall'] * 50 -  
                    metrics['monotonicity_violations'] * 5   # Reduzir penalização
                    )
                    
                    result = {
                        'n_components': n_comp,
                        'covariance_type': cov_type,
                        'valid_clusters': valid_clusters,
                        'gini': metrics['gini'],
                        'ks_statistic': metrics['ks_statistic'],
                        'top_decile_lift': metrics['top_decile_lift'],
                        'top_20pct_recall': metrics['top_20pct_recall'],
                        'monotonicity_violations': metrics['monotonicity_violations'],
                        'ranking_score': ranking_score
                    }
                    
                    results.append(result)
                    
                    print(f"  Clusters válidos: {valid_clusters}")
                    print(f"  GINI: {metrics['gini']:.4f}")
                    print(f"  Top Decile Lift: {metrics['top_decile_lift']:.2f}")
                    print(f"  Top 20% Recall: {metrics['top_20pct_recall']:.2%}")
                    print(f"  Ranking Score: {ranking_score:.2f}")
                    
                except Exception as e:
                    print(f"  Erro geral: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Resto do método permanece igual...
        
        # Selecionar melhor configuração
        if not results:
            print("\nNENHUMA configuração válida! Usando padrão.")
            return self.config['gmm_params_default']
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.gmm_dir, 'ranking_param_search.csv'), index=False)
        
        # Melhor por ranking score
        best_idx = results_df['ranking_score'].idxmax()
        best_params = results_df.loc[best_idx]
        
        optimal_params = {
            'n_components': int(best_params['n_components']),
            'covariance_type': best_params['covariance_type']
        }
        
        print("\n" + "="*80)
        print("MELHOR CONFIGURAÇÃO PARA RANKING:")
        print(f"  n_components: {optimal_params['n_components']}")
        print(f"  covariance_type: {optimal_params['covariance_type']}")
        print(f"  GINI: {best_params['gini']:.4f}")
        print(f"  Top Decile Lift: {best_params['top_decile_lift']:.2f}")
        print(f"  Top 20% Recall: {best_params['top_20pct_recall']:.2%}")
        print("="*80)
        
        return optimal_params
    
    def train_ranking_optimized_cluster_models(self, X_train, y_train, cluster_labels_train, gmm_params):
        """
        Treina modelos otimizados para ranking em cada cluster.
        """
        print("\nTreinando modelos otimizados para ranking por cluster...")
        
        cluster_models = []
        
        # Features numéricas
        numeric_cols = [col for col in X_train.columns 
                    if X_train[col].dtype in ['int64', 'float64']]
        X_train_numeric = X_train[numeric_cols].astype(float).fillna(0)
        
        with self.safe_mlflow_run(experiment_id=self.experiment_id,
                                run_name=f"gmm_ranking_cluster_models"):
            
            for cluster_id in np.unique(cluster_labels_train):
                print(f"\nCluster {cluster_id}:")
                
                # Selecionar dados do cluster
                mask = cluster_labels_train == cluster_id
                X_cluster = X_train_numeric[mask]
                y_cluster = y_train[mask]
                
                print(f"  Amostras: {len(X_cluster)}")
                print(f"  Taxa de conversão: {y_cluster.mean():.4f}")
                print(f"  Classes únicas: {np.unique(y_cluster)}")
                
                if len(X_cluster) < 100:
                    print(f"  Pulando - poucos dados")
                    continue
                
                # Verificar se há ambas as classes
                n_classes = len(np.unique(y_cluster))
                if n_classes < 2:
                    print(f"  Pulando - apenas uma classe presente")
                    # Criar modelo dummy que sempre prediz a probabilidade base
                    from sklearn.dummy import DummyClassifier
                    model = DummyClassifier(strategy='constant', 
                                        constant=1 if y_cluster.iloc[0] == 1 else 0)
                    model.fit(X_cluster, y_cluster)
                    model_type = 'dummy'
                    
                    # Para DummyClassifier, definir GINI como 0
                    gini = 0.0
                else:
                    # Escolher modelo baseado no tamanho do cluster
                    if len(X_cluster) > 5000 and n_classes == 2:
                        # Limpar nomes de features para LightGBM
                        X_cluster_clean = X_cluster.copy()
                        feature_mapping = {}
                        
                        # Renomear colunas problemáticas
                        import re
                        rename_dict = {}
                        for col in X_cluster_clean.columns:
                            # Substituir caracteres problemáticos
                            clean_name = re.sub(r'[^\w\s]', '_', str(col))  # Remove caracteres especiais
                            clean_name = re.sub(r'\s+', '_', clean_name)    # Substitui espaços
                            clean_name = re.sub(r'__+', '_', clean_name)    # Remove underscores múltiplos
                            clean_name = clean_name.strip('_')               # Remove underscores nas extremidades
                            
                            if clean_name != col:
                                rename_dict[col] = clean_name
                        
                        if rename_dict:
                            print(f"  Limpando {len(rename_dict)} nomes de features para LightGBM")
                            X_cluster_clean = X_cluster_clean.rename(columns=rename_dict)
                            feature_mapping = {v: k for k, v in rename_dict.items()}  # Mapeamento reverso
                        
                        # LightGBM para clusters grandes
                        try:
                            model = lgb.LGBMClassifier(
                                objective='binary',
                                metric='auc',
                                n_estimators=200,
                                max_depth=5,
                                num_leaves=31,
                                learning_rate=0.05,
                                min_child_samples=20,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                is_unbalance=True,
                                random_state=self.config['random_state'],
                                n_jobs=-1,
                                verbosity=-1
                            )
                            model.fit(X_cluster_clean, y_cluster)
                            model_type = 'lightgbm'
                            
                            # Avaliar GINI
                            y_pred_proba = model.predict_proba(X_cluster_clean)[:, 1]
                            gini = 2 * self._calculate_auc_fast(y_cluster, y_pred_proba) - 1
                            print(f"  GINI (treino): {gini:.4f}")
                            
                        except Exception as e:
                            print(f"  LightGBM falhou: {e}, usando RandomForest")
                            # Fallback para RandomForest
                            model = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=8,
                                min_samples_split=20,
                                min_samples_leaf=10,
                                class_weight='balanced',
                                random_state=self.config['random_state'],
                                n_jobs=-1
                            )
                            model.fit(X_cluster, y_cluster)
                            model_type = 'randomforest'
                            feature_mapping = {}
                            
                            # Avaliar GINI
                            y_pred_proba = model.predict_proba(X_cluster)[:, 1]
                            gini = 2 * self._calculate_auc_fast(y_cluster, y_pred_proba) - 1
                            print(f"  GINI (treino): {gini:.4f}")
                            
                    else:
                        # RandomForest para clusters menores
                        model = RandomForestClassifier(
                            n_estimators=100,
                            max_depth=8,
                            min_samples_split=20,
                            min_samples_leaf=10,
                            max_features='sqrt',
                            class_weight='balanced_subsample',
                            random_state=self.config['random_state'],
                            n_jobs=-1
                        )
                        model.fit(X_cluster, y_cluster)
                        model_type = 'randomforest'
                        feature_mapping = {}
                        
                        # Avaliar GINI
                        try:
                            y_pred_proba = model.predict_proba(X_cluster)[:, 1]
                            gini = 2 * self._calculate_auc_fast(y_cluster, y_pred_proba) - 1
                            print(f"  GINI (treino): {gini:.4f}")
                        except:
                            gini = 0.0
                            print(f"  Não foi possível calcular GINI")
                
                print(f"  Usando modelo: {model_type}")
                
                # Salvar informações
                model_info = {
                    "cluster_id": int(cluster_id),
                    "model": model,
                    "model_type": model_type,
                    "features": numeric_cols,
                    "feature_mapping": feature_mapping if 'feature_mapping' in locals() else {},
                    "n_samples": len(X_cluster),
                    "conversion_rate": float(y_cluster.mean()),
                    "n_classes": n_classes,
                    "gini_train": float(gini) if not np.isnan(gini) else 0.0
                }
                
                cluster_models.append(model_info)
                
                # Salvar modelo
                model_path = os.path.join(self.gmm_dir, f"cluster_{cluster_id}_model.joblib")
                joblib.dump(model, model_path)
                
                # Log no MLflow
                mlflow.log_params({
                    f"cluster_{cluster_id}_type": model_type,
                    f"cluster_{cluster_id}_samples": len(X_cluster),
                    f"cluster_{cluster_id}_n_classes": n_classes,
                    f"cluster_{cluster_id}_gini": float(gini) if not np.isnan(gini) else 0.0
                })
        
        return cluster_models
    
    def evaluate_ranking_ensemble(self, X_val, y_val, cluster_labels_val, cluster_models, gmm_params):
        """
        Avalia o ensemble com foco em métricas de ranking.
        """
        print("\n" + "="*80)
        print("AVALIANDO ENSEMBLE PARA RANKING")
        print("="*80)
        
        # Criar mapa cluster -> modelo
        cluster_model_map = {model["cluster_id"]: model for model in cluster_models}
        
        # Fazer predições
        y_pred_proba = np.zeros_like(y_val, dtype=float)
        
        for cluster_id, model_info in cluster_model_map.items():
            mask = cluster_labels_val == cluster_id
            
            if not any(mask):
                continue
            
            model = model_info["model"]
            features = model_info["features"]
            model_type = model_info.get("model_type", "unknown")
            n_classes = model_info.get("n_classes", 2)
            
            X_cluster = X_val[features][mask].astype(float).fillna(0)
            
            if len(X_cluster) > 0:
                try:
                    if model_type == 'dummy' or n_classes < 2:
                        # Para modelos dummy ou clusters com uma classe
                        # Usar a taxa de conversão do cluster como probabilidade
                        cluster_rate = model_info.get("conversion_rate", y_val.mean())
                        y_pred_proba[mask] = cluster_rate
                        print(f"  Cluster {cluster_id}: usando taxa fixa {cluster_rate:.4f}")
                    else:
                        # Para modelos normais
                        proba = model.predict_proba(X_cluster)
                        if proba.shape[1] == 2:
                            y_pred_proba[mask] = proba[:, 1]
                        else:
                            # Fallback se apenas uma coluna
                            y_pred_proba[mask] = proba[:, 0]
                        print(f"  Cluster {cluster_id}: {mask.sum()} amostras processadas")
                except Exception as e:
                    print(f"  Erro no cluster {cluster_id}: {e}")
                    # Usar taxa base como fallback
                    y_pred_proba[mask] = y_val.mean()
        
        # Verificar se temos predições válidas
        if np.all(y_pred_proba == 0):
            print("AVISO: Todas as probabilidades são zero!")
            y_pred_proba = np.full_like(y_val, y_val.mean(), dtype=float)
        
        # Avaliar métricas de ranking
        metrics, decile_stats = self.evaluate_ranking_metrics(y_val, y_pred_proba)
        
        # Registrar no MLflow
        with self.safe_mlflow_run(experiment_id=self.experiment_id,
                                run_name=f"gmm_ranking_evaluation"):
            # Métricas principais
            mlflow.log_metrics({
                "gini": metrics['gini'],
                "ks_statistic": metrics['ks_statistic'],
                "top_decile_lift": metrics['top_decile_lift'],
                "top_2deciles_lift": metrics['top_2deciles_lift'],
                "top_10pct_recall": metrics['top_10pct_recall'],
                "top_20pct_recall": metrics['top_20pct_recall'],
                "monotonicity_violations": metrics['monotonicity_violations']
            })
            
            # Salvar análise por decil
            decile_path = os.path.join(self.gmm_dir, "decile_analysis.csv")
            decile_stats.to_csv(decile_path)
            mlflow.log_artifact(decile_path)
        
        # Imprimir resultados
        print("\nMÉTRICAS DE RANKING:")
        print(f"  GINI: {metrics['gini']:.4f}")
        print(f"  KS Statistic: {metrics['ks_statistic']:.4f}")
        print(f"  Top Decile Lift: {metrics['top_decile_lift']:.2f}x")
        print(f"  Top 2 Deciles Lift: {metrics['top_2deciles_lift']:.2f}x")
        print(f"  Top 10% Recall: {metrics['top_10pct_recall']:.2%}")
        print(f"  Top 20% Recall: {metrics['top_20pct_recall']:.2%}")
        print(f"  Violações de Monotonicidade: {metrics['monotonicity_violations']}")
        
        print("\nANÁLISE POR DECIL:")
        print(decile_stats.round(4))
        
        # Verificar critérios de sucesso
        success_criteria = {
            'top_decile_lift_3x': metrics['top_decile_lift'] > 3.0,
            'top_20pct_recall_50pct': metrics['top_20pct_recall'] > 0.5,
            'monotonic': metrics['monotonicity_violations'] == 0
        }
        
        print("\nCRITÉRIOS DE SUCESSO:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASSOU" if passed else "❌ FALHOU"
            print(f"  {criterion}: {status}")
        
        # Salvar resultados
        results = {
            "algorithm": "GaussianMixture_Ranking",
            "gmm_params": gmm_params,
            "metrics": metrics,
            "success_criteria": success_criteria,
            "decile_analysis": decile_stats.to_dict()
        }
        
        with open(os.path.join(self.gmm_dir, "ranking_evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        return results, metrics
    
    def clean_feature_names(self, df):
        """
        Limpa nomes de features para compatibilidade com LightGBM.
        Remove caracteres especiais que causam problemas.
        """
        import re
        
        # Criar mapeamento de nomes
        rename_dict = {}
        for col in df.columns:
            # Substituir caracteres problemáticos
            clean_name = re.sub(r'[^\w\s-]', '_', col)  # Remove caracteres especiais
            clean_name = re.sub(r'[-\s]+', '_', clean_name)  # Substitui espaços e hífens
            clean_name = re.sub(r'__+', '_', clean_name)  # Remove underscores múltiplos
            clean_name = clean_name.strip('_')  # Remove underscores nas extremidades
            
            if clean_name != col:
                rename_dict[col] = clean_name
        
        if rename_dict:
            print(f"  Limpando {len(rename_dict)} nomes de features para LightGBM")
            df = df.rename(columns=rename_dict)
        
        return df, rename_dict

    def plot_ranking_diagnostics(self, y_true, y_pred_proba):
        """
        Gera visualizações para avaliar qualidade do ranking.
        """
        print("\nGerando visualizações de diagnóstico...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Lift por decil
        ax = axes[0, 0]
        _, decile_stats = self.evaluate_ranking_metrics(y_true, y_pred_proba)
        decile_stats = decile_stats.sort_index(ascending=False)  # Decil 10 primeiro
        ax.bar(range(10, 0, -1), decile_stats['lift'], color='steelblue')
        ax.axhline(y=1, color='red', linestyle='--', label='Baseline')
        ax.axhline(y=3, color='green', linestyle='--', label='Target 3x')
        ax.set_xlabel('Decil')
        ax.set_ylabel('Lift')
        ax.set_title('Lift por Decil')
        ax.set_xticks(range(10, 0, -1))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Curva de ganho cumulativo
        ax = axes[0, 1]
        df_sorted = pd.DataFrame({
            'y': y_true, 
            'prob': y_pred_proba
        }).sort_values('prob', ascending=False)
        
        cumsum = df_sorted['y'].cumsum()
        cum_pct = cumsum / cumsum.iloc[-1]
        pop_pct = np.arange(1, len(cumsum) + 1) / len(cumsum)
        
        # Calcular área sob a curva de ganho
        gain_area = np.trapz(cum_pct, pop_pct)
        
        ax.plot(pop_pct * 100, cum_pct * 100, 'b-', linewidth=2, label='Modelo')
        ax.plot([0, 100], [0, 100], 'r--', label='Aleatório')
        ax.fill_between(pop_pct * 100, cum_pct * 100, pop_pct * 100, alpha=0.3)
        ax.set_xlabel('% População (ordenada por score)')
        ax.set_ylabel('% Conversões Capturadas')
        ax.set_title(f'Curva de Ganho Cumulativo (Área: {gain_area:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar linhas de referência
        ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=20, color='gray', linestyle=':', alpha=0.5)
        
        # 3. Distribuição de scores por classe
        ax = axes[1, 0]
        scores_neg = y_pred_proba[y_true == 0]
        scores_pos = y_pred_proba[y_true == 1]
        
        ax.hist(scores_neg, bins=50, alpha=0.5, label='Não converteu', 
                density=True, color='red')
        ax.hist(scores_pos, bins=50, alpha=0.5, label='Converteu', 
                density=True, color='green')
        
        # KS statistic visual
        neg_hist, bins = np.histogram(scores_neg, bins=50, density=True)
        pos_hist, _ = np.histogram(scores_pos, bins=bins, density=True)
        
        ax.set_xlabel('Score de Probabilidade')
        ax.set_ylabel('Densidade')
        ax.set_title('Distribuição de Scores por Classe')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Taxa de conversão por vigésimo (mais granular que decil)
        ax = axes[1, 1]
        n_bins = 20
        df_analysis = pd.DataFrame({'y': y_true, 'prob': y_pred_proba})
        df_analysis['bin'] = pd.qcut(df_analysis['prob'].rank(method='first'), 
                                     q=n_bins, labels=range(1, n_bins+1))
        
        bin_stats = df_analysis.groupby('bin')['y'].agg(['mean', 'count'])
        
        ax.plot(range(1, n_bins+1), bin_stats['mean'] * 100, 'o-', 
                markersize=8, linewidth=2, color='darkblue')
        
        # Adicionar barras de volume
        ax2 = ax.twinx()
        ax2.bar(range(1, n_bins+1), bin_stats['count'], alpha=0.3, color='gray')
        ax2.set_ylabel('Volume de Amostras', color='gray')
        
        ax.set_xlabel('Vigésimo (5% bins)')
        ax.set_ylabel('Taxa de Conversão (%)', color='darkblue')
        ax.set_title('Taxa de Conversão por Percentil')
        ax.grid(True, alpha=0.3)
        
        # Linha de taxa média
        avg_rate = y_true.mean() * 100
        ax.axhline(y=avg_rate, color='red', linestyle='--', alpha=0.5, 
                  label=f'Média: {avg_rate:.2f}%')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.gmm_dir, 'ranking_diagnostics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualizações salvas em: {os.path.join(self.gmm_dir, 'ranking_diagnostics.png')}")
    
    def run(self):
        """
        Executa o pipeline completo de treinamento GMM otimizado para ranking.
        """
        print("\n" + "="*80)
        print("INICIANDO GMM OTIMIZADO PARA RANKING")
        print("="*80)
        
        # Criar estrutura de diretórios
        self.create_directory_structure()
        
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
        
        # Preparar features com foco em ranking
        X_train_prepared, X_val_prepared, selected_features = \
            self.prepare_features_for_ranking(X_train, X_val, y_train)
        
        # Normalizar dados
        print("\nNormalizando dados...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_prepared)
        X_val_scaled = scaler.transform(X_val_prepared)
        
        # Otimizar PCA para ranking
        n_components = self.optimize_pca_for_ranking(X_train_scaled, y_train)
        
        # Aplicar PCA final
        print(f"\nAplicando PCA com {n_components} componentes...")
        pca = PCA(n_components=n_components, random_state=self.config['random_state'])
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        
        # Buscar parâmetros ótimos do GMM para ranking
        gmm_params = self.find_optimal_gmm_for_ranking(
            X_train_pca, y_train, X_val_pca, y_val
        )
        
        # Treinar GMM final
        print(f"\nTreinando GMM final com parâmetros otimizados...")
        gmm = GaussianMixture(
            n_components=gmm_params['n_components'],
            covariance_type=gmm_params['covariance_type'],
            reg_covar=1e-6,
            random_state=self.config['random_state'],
            max_iter=200,
            n_init=10
        )
        
        start_time = time.time()
        gmm.fit(X_train_pca)
        training_time = time.time() - start_time
        print(f"  Tempo de treinamento GMM: {training_time:.2f} segundos")
        
        # Obter clusters
        train_labels = gmm.predict(X_train_pca)
        val_labels = gmm.predict(X_val_pca)
        
        # Analisar clusters
        self._analyze_clusters(train_labels, y_train, "treino")
        self._analyze_clusters(val_labels, y_val, "validação")
        
        # Treinar modelos otimizados para ranking
        cluster_models = self.train_ranking_optimized_cluster_models(
            X_train, y_train, train_labels, gmm_params
        )
        
        # Avaliar ensemble
        results, metrics = self.evaluate_ranking_ensemble(
            X_val, y_val, val_labels, cluster_models, gmm_params
        )
        
        # Gerar visualizações
        y_pred_proba = np.zeros_like(y_val, dtype=float)
        cluster_model_map = {model["cluster_id"]: model for model in cluster_models}

        for cluster_id, model_info in cluster_model_map.items():
            mask = val_labels == cluster_id
            if any(mask):
                model = model_info["model"]
                features = model_info["features"]
                model_type = model_info.get("model_type", "unknown")
                n_classes = model_info.get("n_classes", 2)
                
                X_cluster = X_val[features][mask].astype(float).fillna(0)
                
                if len(X_cluster) > 0:
                    try:
                        if model_type == 'dummy' or n_classes < 2:
                            # Para modelos dummy
                            cluster_rate = model_info.get("conversion_rate", 0.0)
                            y_pred_proba[mask] = cluster_rate
                        elif model_type == 'lightgbm' and 'feature_mapping' in model_info:
                            # Para LightGBM com features renomeadas
                            X_cluster_renamed = X_cluster.copy()
                            
                            # Aplicar renomeação se necessário
                            reverse_mapping = {v: k for k, v in model_info['feature_mapping'].items()}
                            if reverse_mapping:
                                rename_dict = {}
                                for col in X_cluster_renamed.columns:
                                    if col in reverse_mapping:
                                        rename_dict[col] = reverse_mapping[col]
                                if rename_dict:
                                    X_cluster_renamed = X_cluster_renamed.rename(columns=rename_dict)
                            
                            proba = model.predict_proba(X_cluster_renamed)
                            if proba.ndim == 2 and proba.shape[1] == 2:
                                y_pred_proba[mask] = proba[:, 1]
                            else:
                                y_pred_proba[mask] = proba if proba.ndim == 1 else proba[:, 0]
                        else:
                            # Para outros modelos
                            proba = model.predict_proba(X_cluster)
                            if proba.ndim == 2 and proba.shape[1] == 2:
                                y_pred_proba[mask] = proba[:, 1]
                            else:
                                y_pred_proba[mask] = proba if proba.ndim == 1 else proba[:, 0]
                                
                    except Exception as e:
                        print(f"Erro ao gerar predições para cluster {cluster_id}: {e}")
                        y_pred_proba[mask] = y_val.mean()

        self.plot_ranking_diagnostics(y_val, y_pred_proba)
        
        # Criar pipeline para GMM_Wrapper
        pipeline = {
            'pca_model': pca,
            'gmm_model': gmm,
            'scaler_model': scaler,
            'cluster_models': {
                model["cluster_id"]: {
                    "model": model["model"], 
                    "threshold": 0.5  # Não usado para ranking
                } for model in cluster_models
            },
            'n_clusters': gmm_params['n_components'],
            'selected_features': selected_features,
            'feature_preparation': {
                'variance_threshold': 0.01,
                'correlation_threshold': 0.95,
                'n_features_selected': len(selected_features)
            }
        }
        
        # Criar e salvar GMM_Wrapper
        gmm_wrapper = GMM_Wrapper(pipeline)
        wrapper_path = os.path.join(self.gmm_dir, 'gmm_ranking_wrapper.joblib')
        joblib.dump(gmm_wrapper, wrapper_path)
        print(f"\nGMM Wrapper salvo em: {wrapper_path}")
        
        # Salvar componentes individuais
        joblib.dump(gmm, os.path.join(self.gmm_dir, 'gmm_model.joblib'))
        joblib.dump(pca, os.path.join(self.gmm_dir, 'pca_model.joblib'))
        joblib.dump(scaler, os.path.join(self.gmm_dir, 'scaler_model.joblib'))
        
        # Salvar configuração completa
        config_summary = {
            'date': datetime.now().isoformat(),
            'data': {
                'n_samples_train': len(X_train),
                'n_samples_val': len(X_val),
                'n_features_original': len(data['feature_names']),
                'n_features_selected': len(selected_features),
                'n_pca_components': n_components
            },
            'gmm_params': gmm_params,
            'metrics': metrics,
            'training_time': training_time,
            'success': all(results['success_criteria'].values())
        }
        
        with open(os.path.join(self.gmm_dir, 'experiment_config.json'), 'w') as f:
            json.dump(config_summary, f, indent=2)
        
        print("\n" + "="*80)
        print("GMM PARA RANKING CONCLUÍDO COM SUCESSO!")
        print(f"Resultados salvos em: {self.gmm_dir}")
        print("="*80)
        
        return results
    
    def _analyze_clusters(self, cluster_labels, y, prefix):
        """Análise básica dos clusters."""
        print(f"\nAnálise de clusters - {prefix}:")
        
        df = pd.DataFrame({
            'cluster': cluster_labels,
            'target': y
        })
        
        stats = df.groupby('cluster').agg({
            'target': ['count', 'sum', 'mean']
        })
        stats.columns = ['n_samples', 'n_conversions', 'conversion_rate']
        
        print(stats)
        
        return stats