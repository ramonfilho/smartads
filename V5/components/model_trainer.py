# smart_ads_pipeline/components/model_trainer.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import pickle
import json
import re
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Componente para treinar modelo LightGBM com lógica de ranking e decis.
    Replica exatamente a lógica dos scripts 05_direct_ranking.py até 08_ranking_production.py
    """
    
    def __init__(self):
        """Inicializa o ModelTrainer."""
        self.model = None
        self.decile_thresholds = None
        self.feature_importance = None
        self.training_metrics = {}
        self.is_trained = False
        
        # Parâmetros do LightGBM (exatamente como no script 05)
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'max_depth': -1,
            'learning_rate': 0.03,
            'n_estimators': 500,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'is_unbalance': True,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        }
        
        logger.info("ModelTrainer inicializado com parâmetros LightGBM")
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa nomes de colunas para compatibilidade com LightGBM.
        Usado apenas se standardize_feature_name falhar.
        """
        rename_dict = {}
        for col in df.columns:
            # Remover/substituir caracteres problemáticos
            clean_name = re.sub(r'[^\w\s]', '_', str(col))  # Remove caracteres especiais
            clean_name = re.sub(r'\s+', '_', clean_name)    # Substitui espaços
            clean_name = re.sub(r'__+', '_', clean_name)    # Remove underscores múltiplos
            clean_name = clean_name.strip('_')               # Remove underscores nas extremidades
            
            if clean_name != col:
                rename_dict[col] = clean_name
        
        if rename_dict:
            logger.info(f"Limpando {len(rename_dict)} nomes de colunas para LightGBM")
            df = df.rename(columns=rename_dict)
            # Salvar mapeamento para uso posterior
            self.column_name_mapping = rename_dict
        
        return df
    
    def train_with_ranking(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          check_column_names: bool = True) -> Dict[str, Any]:
        """
        Treina modelo LightGBM com validação e cálculo de decis.
        Replica lógica exata do script 05_direct_ranking.py.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação
            y_val: Target de validação
            check_column_names: Se True, verifica e limpa nomes se necessário
            
        Returns:
            Dicionário com métricas de treino
        """
        logger.info("Iniciando treinamento com LightGBM...")
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        logger.info(f"Taxa de conversão - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}")
        
        # Verificar e limpar nomes de colunas se necessário
        if check_column_names:
            try:
                # Tentar treinar com nomes originais primeiro
                test_model = lgb.LGBMClassifier(**self.lgb_params)
                test_model.fit(X_train.head(100), y_train.head(100), 
                             eval_set=[(X_val.head(100), y_val.head(100))],
                             eval_metric='auc',
                             callbacks=[lgb.log_evaluation(0)])
                logger.info("Nomes de colunas compatíveis com LightGBM")
            except Exception as e:
                if "Do not support" in str(e) or "cannot contain" in str(e):
                    logger.warning(f"Nomes de colunas incompatíveis: {e}")
                    logger.info("Aplicando limpeza de nomes de colunas...")
                    X_train = self.clean_column_names(X_train)
                    X_val = self.clean_column_names(X_val)
                else:
                    raise e
        
        # Criar e treinar modelo
        self.model = lgb.LGBMClassifier(**self.lgb_params)
        
        # Treinar com early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        logger.info(f"Modelo treinado com {self.model.n_estimators_} árvores")
        
        # Fazer predições
        y_train_pred = self.model.predict_proba(X_train)[:, 1]
        y_val_pred = self.model.predict_proba(X_val)[:, 1]
        
        # Calcular métricas
        train_auc = roc_auc_score(y_train, y_train_pred)
        val_auc = roc_auc_score(y_val, y_val_pred)
        
        logger.info(f"AUC - Train: {train_auc:.4f}, Val: {val_auc:.4f}")
        
        # Calcular decis baseado no conjunto de TREINO (como no script 06)
        self._calculate_decile_thresholds(y_train_pred, y_train)
        
        # Avaliar modelo
        metrics = self._evaluate_ranking(X_val, y_val)
        
        # Adicionar métricas básicas
        metrics['train_auc'] = train_auc
        metrics['val_auc'] = val_auc
        metrics['n_estimators'] = self.model.n_estimators_
        
        # Salvar feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.training_metrics = metrics
        self.is_trained = True
        
        return metrics
    
    def _calculate_decile_thresholds(self, probabilities: np.ndarray, 
                                   true_labels: np.ndarray) -> None:
        """
        Calcula thresholds dos decis baseado nas probabilidades do TREINO.
        Replica lógica exata do script 06_decile_saving.py.
        """
        logger.info("Calculando thresholds de decis...")
        
        # Estatísticas das probabilidades
        logger.info(f"Estatísticas das probabilidades:")
        logger.info(f"  - Mínima: {probabilities.min():.6f}")
        logger.info(f"  - Máxima: {probabilities.max():.6f}")
        logger.info(f"  - Média: {probabilities.mean():.6f}")
        logger.info(f"  - Mediana: {np.median(probabilities):.6f}")
        logger.info(f"  - Desvio padrão: {probabilities.std():.6f}")
        
        # Calcular percentis (10, 20, ..., 90)
        percentiles = np.arange(10, 100, 10)
        self.decile_thresholds = np.percentile(probabilities, percentiles)
        
        logger.info("LIMIARES DE DECIS:")
        logger.info("-" * 50)
        for i, (perc, thresh) in enumerate(zip(percentiles, self.decile_thresholds)):
            logger.info(f"Decil {i+1} → {i+2}: probabilidade > {thresh:.6f} (percentil {perc})")
        logger.info(f"Decil 10: probabilidade > {self.decile_thresholds[-1]:.6f}")
        logger.info("-" * 50)
        
        # Validar distribuição
        deciles = self._assign_deciles(probabilities)
        decile_counts = pd.Series(deciles).value_counts().sort_index()
        
        logger.info("\nContagem por decil:")
        for decil, count in decile_counts.items():
            pct = count / len(probabilities) * 100
            logger.info(f"  Decil {decil}: {count:,} ({pct:.1f}%)")
        
        # Calcular performance por decil
        df_analysis = pd.DataFrame({
            'probability': probabilities,
            'target': true_labels,
            'decile': deciles
        })
        
        self.decile_stats = df_analysis.groupby('decile').agg({
            'target': ['sum', 'count', 'mean'],
            'probability': ['mean', 'min', 'max']
        }).round(6)
        
        self.decile_stats.columns = ['conversions', 'total', 'conv_rate', 
                                    'avg_prob', 'min_prob', 'max_prob']
        
        # Calcular lift
        overall_rate = true_labels.mean()
        self.decile_stats['lift'] = self.decile_stats['conv_rate'] / overall_rate
        
        logger.info("\nPerformance por decil:")
        logger.info(self.decile_stats)
    
    def _assign_deciles(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Atribui decil para cada probabilidade usando os thresholds calculados.
        """
        deciles = np.zeros(len(probabilities), dtype=int)
        
        for i, prob in enumerate(probabilities):
            # Encontrar o decil correto
            decile = 1
            for j, threshold in enumerate(self.decile_thresholds):
                if prob > threshold:
                    decile = j + 2
            deciles[i] = decile
        
        return deciles
    
    def _evaluate_ranking(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Avalia métricas de ranking.
        Replica lógica do script 05_direct_ranking.py.
        """
        logger.info("Avaliando métricas de ranking...")
        
        # Fazer predições
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Criar DataFrame para análise
        df = pd.DataFrame({
            'y_true': y,
            'y_proba': y_pred_proba
        })
        
        # Atribuir decis
        df['decile'] = self._assign_deciles(y_pred_proba)
        
        # Análise por decil
        decile_stats = df.groupby('decile').agg({
            'y_true': ['sum', 'count', 'mean']
        })
        decile_stats.columns = ['conversions', 'total', 'rate']
        
        # Calcular lift
        overall_rate = df['y_true'].mean()
        decile_stats['lift'] = decile_stats['rate'] / overall_rate
        
        # KS Statistic
        df_sorted = df.sort_values('y_proba', ascending=False)
        cum_pos = np.cumsum(df_sorted['y_true']) / df['y_true'].sum()
        cum_neg = np.cumsum(~df_sorted['y_true']) / (~df['y_true']).sum()
        ks_statistic = np.max(np.abs(cum_pos - cum_neg))
        
        # GINI
        auc = roc_auc_score(y, y_pred_proba)
        gini = 2 * auc - 1
        
        # Top-K metrics
        n_samples = len(df)
        top_10_pct = int(n_samples * 0.1)
        top_20_pct = int(n_samples * 0.2)
        
        df_sorted_desc = df.sort_values('y_proba', ascending=False)
        
        metrics = {
            'gini': gini,
            'auc': auc,
            'ks_statistic': ks_statistic,
            'top_decile_lift': decile_stats.loc[10, 'lift'] if 10 in decile_stats.index else 0,
            'top_2deciles_lift': decile_stats.loc[[9, 10], 'lift'].mean() if 9 in decile_stats.index and 10 in decile_stats.index else 0,
            'top_10pct_recall': df_sorted_desc.head(top_10_pct)['y_true'].sum() / df['y_true'].sum(),
            'top_20pct_recall': df_sorted_desc.head(top_20_pct)['y_true'].sum() / df['y_true'].sum(),
        }
        
        logger.info("\n" + "="*60)
        logger.info("RESULTADOS DO MODELO")
        logger.info("="*60)
        logger.info(f"GINI: {metrics['gini']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"KS Statistic: {metrics['ks_statistic']:.4f}")
        logger.info(f"Top Decile Lift: {metrics['top_decile_lift']:.2f}x")
        logger.info(f"Top 2 Deciles Lift: {metrics['top_2deciles_lift']:.2f}x")
        logger.info(f"Top 10% Recall: {metrics['top_10pct_recall']:.2%}")
        logger.info(f"Top 20% Recall: {metrics['top_20pct_recall']:.2%}")
        
        return metrics
    
    def predict_with_ranking(self, X: pd.DataFrame,
                           return_proba: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Faz predições com ranking baseado em decis.
        
        Args:
            X: Features para predição
            return_proba: Se True, retorna probabilidades; se False, retorna classes
            
        Returns:
            Tupla (probabilidades/classes, decis)
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Execute train_with_ranking primeiro.")
        
        # Limpar nomes se necessário
        if hasattr(self, 'column_name_mapping'):
            X = X.rename(columns=self.column_name_mapping)
        
        # Predições
        if return_proba:
            predictions = self.model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)
        
        # Atribuir decis
        deciles = self._assign_deciles(predictions if return_proba else 
                                     self.model.predict_proba(X)[:, 1])
        
        return predictions, deciles
    
    def save_artifacts(self, output_dir: str) -> None:
        """
        Salva todos os artefatos do modelo.
        Replica lógica do script 06_decile_saving.py.
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Salvar modelo
        model_path = output_path / "lightgbm_direct_ranking.joblib"
        joblib.dump(self.model, model_path)
        logger.info(f"✅ Modelo salvo em: {model_path}")
        
        # 2. Salvar thresholds como pickle
        thresholds_pkl = output_path / "decile_thresholds.pkl"
        with open(thresholds_pkl, 'wb') as f:
            pickle.dump(self.decile_thresholds, f)
        logger.info(f"✅ Thresholds salvos em: {thresholds_pkl}")
        
        # 3. Salvar thresholds como JSON (para inspeção)
        thresholds_dict = {
            'thresholds': self.decile_thresholds.tolist(),
            'percentiles': list(range(10, 100, 10)),
            'statistics': {
                'min_probability': float(self.decile_stats['min_prob'].min()),
                'max_probability': float(self.decile_stats['max_prob'].max()),
                'mean_probability': float(self.decile_stats['avg_prob'].mean())
            },
            'performance_by_decile': self.decile_stats.to_dict()
        }
        
        thresholds_json = output_path / "decile_thresholds.json"
        with open(thresholds_json, 'w') as f:
            json.dump(thresholds_dict, f, indent=2)
        logger.info(f"✅ Thresholds também salvos em JSON: {thresholds_json}")
        
        # 4. Salvar configuração do modelo
        config = {
            'model_type': 'LightGBM',
            'model_params': self.lgb_params,
            'training_metrics': self.training_metrics,
            'feature_importance_top10': self.feature_importance.head(10).to_dict('records'),
            'n_features': len(self.feature_importance),
            'top_decile_lift': float(self.decile_stats.loc[10, 'lift']) if 10 in self.decile_stats.index else None,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_path = output_path / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ Configuração do modelo salva em: {config_path}")
        
        # 5. Salvar feature importance completa
        importance_path = output_path / "feature_importance.csv"
        self.feature_importance.to_csv(importance_path, index=False)
        logger.info(f"✅ Feature importance salva em: {importance_path}")
        
        logger.info("\n✅ Todos os artefatos salvos com sucesso!")
    
    def load_artifacts(self, artifacts_dir: str) -> None:
        """
        Carrega artefatos salvos do modelo.
        """
        artifacts_path = Path(artifacts_dir)
        
        # Carregar modelo
        model_path = artifacts_path / "lightgbm_direct_ranking.joblib"
        self.model = joblib.load(model_path)
        
        # Carregar thresholds
        thresholds_path = artifacts_path / "decile_thresholds.pkl"
        with open(thresholds_path, 'rb') as f:
            self.decile_thresholds = pickle.load(f)
        
        # Carregar configuração
        config_path = artifacts_path / "model_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.lgb_params = config['model_params']
            self.training_metrics = config['training_metrics']
        
        # Carregar feature importance
        importance_path = artifacts_path / "feature_importance.csv"
        if importance_path.exists():
            self.feature_importance = pd.read_csv(importance_path)
        
        self.is_trained = True
        logger.info("✅ Artefatos carregados com sucesso!")
    
    def validate_on_test_set(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Valida modelo no conjunto de teste.
        Replica lógica do script 07_test_set_validation.py.
        """
        logger.info("=== VALIDAÇÃO NO CONJUNTO DE TESTE ===")
        logger.info(f"Shape do teste: {X_test.shape}")
        logger.info(f"Taxa de conversão no teste: {y_test.mean():.4f} ({y_test.sum()} conversões)")
        
        # Avaliar no teste
        test_metrics = self._evaluate_ranking(X_test, y_test)
        
        # Comparar com métricas de validação
        logger.info("\nCOMPARAÇÃO TREINO vs TESTE:")
        logger.info(f"AUC - Val: {self.training_metrics.get('val_auc', 0):.4f}, "
                   f"Test: {test_metrics['auc']:.4f}")
        logger.info(f"GINI - Val: {self.training_metrics.get('gini', 0):.4f}, "
                   f"Test: {test_metrics['gini']:.4f}")
        
        # Verificar critérios de sucesso
        logger.info("\nCRITÉRIOS DE SUCESSO:")
        success_criteria = {
            'top_decile_lift_gt_3': test_metrics.get('top_decile_lift', 0) > 3.0,
            'top_20pct_recall_gt_50': test_metrics.get('top_20pct_recall', 0) > 0.5,
            'auc_gt_0_7': test_metrics['auc'] > 0.7
        }
        
        for criterion, passed in success_criteria.items():
            logger.info(f"{criterion}: {'✅ PASSOU' if passed else '❌ FALHOU'}")
        
        test_metrics['success_criteria'] = success_criteria
        test_metrics['all_criteria_passed'] = all(success_criteria.values())
        
        return test_metrics