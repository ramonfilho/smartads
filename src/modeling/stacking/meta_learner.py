"""
Implementação de meta-modelos para combinar previsões de modelos especialistas.

Este módulo contém implementações de diferentes abordagens para combinar
as previsões de modelos especialistas em um sistema de stacking.
"""

import numpy as np
import pandas as pd
import os
import joblib
import mlflow
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

class MetaLearner(BaseEstimator, ClassifierMixin):
    """
    Classe base para meta-learners em sistemas de stacking.
    """
    
    def __init__(self, model_type="lightgbm", params=None, name=None):
        """
        Inicializa um meta-learner.
        
        Args:
            model_type: Tipo de modelo ('lightgbm', 'xgboost', 'random_forest', 'logistic')
            params: Parâmetros específicos para o modelo
            name: Nome para identificação do modelo
        """
        self.model_type = model_type
        self.params = params if params is not None else {}
        self.name = name if name is not None else f"meta_{model_type}"
        self.model = None
    
    def _create_model(self):
        """
        Cria uma instância do modelo com base no tipo e parâmetros especificados.
        
        Returns:
            Instância do modelo
        """
        if self.model_type == "lightgbm":
            import lightgbm as lgb
            
            # Parâmetros padrão para meta-modelo
            default_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1,
                'scale_pos_weight': 50
            }
            
            # Atualizar com parâmetros fornecidos
            default_params.update(self.params)
            
            return lgb.LGBMClassifier(**default_params)
            
        elif self.model_type == "xgboost":
            import xgboost as xgb
            
            # Parâmetros padrão
            default_params = {
                'objective': 'binary:logistic',
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'random_state': 42,
                'n_jobs': -1,
                'scale_pos_weight': 50
            }
            
            # Atualizar com parâmetros fornecidos
            default_params.update(self.params)
            
            return xgb.XGBClassifier(**default_params)
            
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            
            # Parâmetros padrão
            default_params = {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
            
            # Atualizar com parâmetros fornecidos
            default_params.update(self.params)
            
            return RandomForestClassifier(**default_params)
            
        elif self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            
            # Parâmetros padrão
            default_params = {
                'C': 1.0,
                'max_iter': 1000,
                'solver': 'liblinear',
                'random_state': 42,
                'class_weight': 'balanced'
            }
            
            # Atualizar com parâmetros fornecidos
            default_params.update(self.params)
            
            return LogisticRegression(**default_params)
        
        else:
            raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
    
    def fit(self, X, y):
        """
        Treina o meta-modelo.
        
        Args:
            X: Meta-features (previsões dos modelos especialistas)
            y: Target
            
        Returns:
            self
        """
        self.model = self._create_model()
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """
        Gera probabilidades de previsão.
        
        Args:
            X: Meta-features (previsões dos modelos especialistas)
            
        Returns:
            Array com probabilidades
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        return self.model.predict_proba(X)
    
    def predict(self, X, threshold=0.5):
        """
        Gera previsões binárias.
        
        Args:
            X: Meta-features (previsões dos modelos especialistas)
            threshold: Threshold para classificação binária
            
        Returns:
            Array com previsões binárias
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)
    
    def find_optimal_threshold(self, X, y):
        """
        Encontra o threshold ótimo para maximizar o F1 score.
        
        Args:
            X: Meta-features (previsões dos modelos especialistas)
            y: Target
            
        Returns:
            Threshold ótimo
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        probas = self.predict_proba(X)[:, 1]
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.01, 0.5, 0.01):
            y_pred = (probas >= threshold).astype(int)
            f1 = f1_score(y, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def save(self, filepath):
        """
        Salva o modelo em disco.
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Salvar modelo e configuração
        config = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'name': self.name
        }
        
        joblib.dump(config, filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Carrega um modelo salvo.
        
        Args:
            filepath: Caminho do modelo salvo
            
        Returns:
            Instância de MetaLearner com o modelo carregado
        """
        config = joblib.load(filepath)
        
        # Criar instância
        instance = cls(
            model_type=config['model_type'],
            params=config['params'],
            name=config['name']
        )
        
        instance.model = config['model']
        
        return instance
    
    def log_to_mlflow(self, X, y, run_id=None):
        """
        Registra métricas e o modelo no MLflow.
        
        Args:
            X: Meta-features de validação
            y: Target de validação
            run_id: ID do run MLflow (opcional)
            
        Returns:
            Dicionário com métricas
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
        
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        # Encontrar threshold ótimo
        best_threshold = self.find_optimal_threshold(X, y)
        
        # Gerar previsões
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        # Calcular métricas
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        pr_auc = average_precision_score(y, y_pred_proba)
        
        # Calcular positivos previstos
        positive_pct = y_pred.mean() * 100
        
        # Preparar métricas
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pr_auc': pr_auc,
            'threshold': best_threshold,
            'positive_pct': positive_pct
        }
        
        # Registrar no MLflow
        with mlflow.start_run(run_id=run_id):
            # Registrar métricas
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"meta_{metric_name}", metric_value)
            
            # Registrar parâmetros do modelo
            model_params = self.model.get_params()
            for param_name, param_value in model_params.items():
                # Converter para string se não for um tipo primitivo
                if not isinstance(param_value, (int, float, str, bool, type(None))):
                    param_value = str(param_value)
                
                mlflow.log_param(f"meta_{param_name}", param_value)
            
            # Registrar tipo do modelo
            mlflow.set_tag("meta_model_type", self.model_type)
            
            # Salvar e registrar modelo
            if self.model_type == "lightgbm":
                mlflow.lightgbm.log_model(self.model, artifact_path="meta_model")
            elif self.model_type == "xgboost":
                mlflow.xgboost.log_model(self.model, artifact_path="meta_model")
            else:
                mlflow.sklearn.log_model(self.model, artifact_path="meta_model")
        
        return metrics


class WeightedAverageMetaLearner(BaseEstimator, ClassifierMixin):
    """
    Meta-learner que usa média ponderada das previsões dos modelos especialistas.
    Útil quando queremos uma abordagem mais interpretável.
    """
    
    def __init__(self, weights=None, threshold=0.5):
        """
        Inicializa o meta-learner de média ponderada.
        
        Args:
            weights: Lista de pesos para cada modelo especialista
                    Se None, serão aprendidos durante o treinamento
            threshold: Threshold para classificação binária
        """
        self.weights = weights
        self.threshold = threshold
        self.fitted_weights_ = None
    
    def fit(self, X, y):
        """
        Aprende os pesos ótimos para cada modelo especialista.
        
        Args:
            X: Meta-features (previsões dos modelos especialistas)
            y: Target
            
        Returns:
            self
        """
        if self.weights is not None:
            self.fitted_weights_ = np.array(self.weights)
            return self
        
        # Número de modelos especialistas
        n_models = X.shape[1]
        
        # Inicializar com pesos iguais
        best_weights = np.ones(n_models) / n_models
        best_f1 = 0
        
        # Grid search simples para encontrar pesos ótimos
        # Para cada modelo, testar diferentes pesos de 0 a 1
        for i in range(n_models):
            for weight in np.linspace(0, 1, 11):
                weights = best_weights.copy()
                weights[i] = weight
                
                # Normalizar para soma 1
                weights = weights / weights.sum()
                
                # Calcular média ponderada
                y_pred_proba = np.sum(X * weights.reshape(1, -1), axis=1)
                
                # Converter para classes
                y_pred = (y_pred_proba >= self.threshold).astype(int)
                
                # Calcular F1
                f1 = f1_score(y, y_pred)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = weights
        
        self.fitted_weights_ = best_weights
        return self
    
    def predict_proba(self, X):
        """
        Gera probabilidades de previsão usando média ponderada.
        
        Args:
            X: Meta-features (previsões dos modelos especialistas)
            
        Returns:
            Array com probabilidades
        """
        if self.fitted_weights_ is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        # Calcular média ponderada
        y_pred_proba = np.sum(X * self.fitted_weights_.reshape(1, -1), axis=1)
        
        # Retornar no formato que o scikit-learn espera (classe 0 e classe 1)
        return np.vstack((1 - y_pred_proba, y_pred_proba)).T
    
    def predict(self, X, threshold=None):
        """
        Gera previsões binárias.
        
        Args:
            X: Meta-features (previsões dos modelos especialistas)
            threshold: Threshold para classificação binária
            
        Returns:
            Array com previsões binárias
        """
        if threshold is None:
            threshold = self.threshold
            
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)
    
    def get_weights(self):
        """
        Retorna os pesos aprendidos.
        
        Returns:
            Array com pesos para cada modelo especialista
        """
        return self.fitted_weights_
    
    def save(self, filepath):
        """
        Salva o modelo em disco.
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        if self.fitted_weights_ is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Salvar configuração
        config = {
            'fitted_weights_': self.fitted_weights_,
            'threshold': self.threshold
        }
        
        joblib.dump(config, filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Carrega um modelo salvo.
        
        Args:
            filepath: Caminho do modelo salvo
            
        Returns:
            Instância de WeightedAverageMetaLearner com configuração carregada
        """
        config = joblib.load(filepath)
        
        # Criar instância
        instance = cls(threshold=config['threshold'])
        instance.fitted_weights_ = config['fitted_weights_']
        
        return instance
    
    def log_to_mlflow(self, X, y, specialist_names=None, run_id=None):
        """
        Registra métricas e configuração no MLflow.
        
        Args:
            X: Meta-features de validação
            y: Target de validação
            specialist_names: Nomes dos modelos especialistas
            run_id: ID do run MLflow (opcional)
            
        Returns:
            Dicionário com métricas
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
        
        if self.fitted_weights_ is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        # Gerar previsões
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Calcular métricas
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        pr_auc = average_precision_score(y, y_pred_proba)
        
        # Calcular positivos previstos
        positive_pct = y_pred.mean() * 100
        
        # Preparar métricas
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pr_auc': pr_auc,
            'threshold': self.threshold,
            'positive_pct': positive_pct
        }
        
        # Registrar no MLflow
        with mlflow.start_run(run_id=run_id):
            # Registrar métricas
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"weighted_avg_{metric_name}", metric_value)
            
            # Registrar pesos
            for i, weight in enumerate(self.fitted_weights_):
                if specialist_names and i < len(specialist_names):
                    mlflow.log_param(f"weight_{specialist_names[i]}", weight)
                else:
                    mlflow.log_param(f"weight_specialist_{i+1}", weight)
            
            # Registrar tipo do modelo
            mlflow.set_tag("meta_model_type", "weighted_average")
            
            # Criar e registrar artefato de pesos
            weight_info = ""
            for i, weight in enumerate(self.fitted_weights_):
                if specialist_names and i < len(specialist_names):
                    weight_info += f"{specialist_names[i]}: {weight:.4f}\n"
                else:
                    weight_info += f"Specialist {i+1}: {weight:.4f}\n"
            
            # Salvar em arquivo temporário
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Specialist Weights:\n")
                f.write(weight_info)
                weights_file = f.name
            
            # Registrar arquivo
            mlflow.log_artifact(weights_file)
            
            # Remover arquivo temporário
            os.unlink(weights_file)
        
        return metrics