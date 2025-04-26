#treinamento dos modelos especialistas
"""
Implementação de modelos especialistas para diferentes tipos de dados.

Este módulo contém classes e funções para treinar e avaliar modelos
especializados em diferentes tipos de dados (demográficos, temporais, textuais).
"""

import numpy as np
import pandas as pd
import os
import joblib
import mlflow
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

class SpecialistModel:
    """
    Classe base para modelos especialistas.
    """
    
    def __init__(self, model_type="lightgbm", feature_type=None, params=None, name=None):
        """
        Inicializa um modelo especialista.
        
        Args:
            model_type: Tipo de modelo a ser usado ('lightgbm', 'xgboost', 'random_forest')
            feature_type: Tipo de feature que o modelo processa
            params: Parâmetros específicos para o modelo
            name: Nome para identificação do modelo
        """
        self.model_type = model_type
        self.feature_type = feature_type
        self.params = params if params is not None else {}
        self.name = name if name is not None else f"{feature_type}_{model_type}"
        self.model = None
    
    def _create_model(self):
        """
        Cria uma instância do modelo com base no tipo e parâmetros especificados.
        
        Returns:
            Instância do modelo
        """
        if self.model_type == "lightgbm":
            import lightgbm as lgb
            
            # Parâmetros base para todos os tipos
            base_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Parâmetros específicos por tipo de feature
            if self.feature_type == "text":
                specific_params = {
                    'num_leaves': 32,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'scale_pos_weight': 50
                }
            elif self.feature_type == "temporal":
                specific_params = {
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'scale_pos_weight': 50
                }
            else:  # demographic ou outro
                specific_params = {
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'scale_pos_weight': 50
                }
            
            # Combinar parâmetros
            params = {**base_params, **specific_params, **self.params}
            
            return lgb.LGBMClassifier(**params)
            
        elif self.model_type == "xgboost":
            import xgboost as xgb
            
            # Parâmetros base
            base_params = {
                'objective': 'binary:logistic',
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Parâmetros específicos
            if self.feature_type == "text":
                specific_params = {
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'min_child_weight': 1,
                    'scale_pos_weight': 50
                }
            elif self.feature_type == "temporal":
                specific_params = {
                    'max_depth': 7,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 2,
                    'scale_pos_weight': 50
                }
            else:
                specific_params = {
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'min_child_weight': 1,
                    'scale_pos_weight': 50
                }
            
            # Combinar parâmetros
            params = {**base_params, **specific_params, **self.params}
            
            return xgb.XGBClassifier(**params)
            
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            
            # Parâmetros base
            base_params = {
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
            
            # Parâmetros específicos
            if self.feature_type == "text":
                specific_params = {
                    'n_estimators': 200,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 2
                }
            elif self.feature_type == "temporal":
                specific_params = {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            else:
                specific_params = {
                    'n_estimators': 200,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            
            # Combinar parâmetros
            params = {**base_params, **specific_params, **self.params}
            
            return RandomForestClassifier(**params)
        
        else:
            raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
    
    def fit(self, X, y):
        """
        Treina o modelo especialista.
        
        Args:
            X: Features de treinamento
            y: Target de treinamento
            
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
            X: Features para previsão
            
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
            X: Features para previsão
            threshold: Threshold para classificação binária
            
        Returns:
            Array com previsões binárias
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)
    
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
        
        # Salvar modelo
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load(cls, filepath, feature_type=None, name=None):
        """
        Carrega um modelo salvo.
        
        Args:
            filepath: Caminho do modelo salvo
            feature_type: Tipo de feature do modelo
            name: Nome para identificação
            
        Returns:
            Instância de SpecialistModel com o modelo carregado
        """
        model = joblib.load(filepath)
        
        # Identificar o tipo de modelo
        if 'LGBMClassifier' in str(type(model)):
            model_type = "lightgbm"
        elif 'XGBClassifier' in str(type(model)):
            model_type = "xgboost"
        elif 'RandomForestClassifier' in str(type(model)):
            model_type = "random_forest"
        else:
            model_type = "unknown"
        
        # Criar instância
        instance = cls(model_type=model_type, feature_type=feature_type, name=name)
        instance.model = model
        
        return instance
    
    def log_to_mlflow(self, X_val, y_val, run_id=None, prefix=None):
        """
        Registra métricas e o modelo no MLflow.
        
        Args:
            X_val: Features de validação
            y_val: Target de validação
            run_id: ID do run MLflow (opcional)
            prefix: Prefixo para nomes de métricas (opcional)
            
        Returns:
            Dicionário com métricas
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        # Gerar previsões
        y_pred_proba = self.predict_proba(X_val)[:, 1]
        
        # Calcular métricas para diferentes thresholds
        thresholds = np.arange(0.01, 0.5, 0.01)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Calcular métricas com threshold ótimo
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        
        # Calcular positivos previstos
        positive_pct = y_pred.mean() * 100
        
        # Preparar métricas
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': best_f1,
            'pr_auc': pr_auc,
            'threshold': best_threshold,
            'positive_pct': positive_pct
        }
        
        # Adicionar prefixo se fornecido
        if prefix is None:
            prefix = f"specialist_{self.feature_type}_"
        
        # Registrar no MLflow
        with mlflow.start_run(run_id=run_id):
            # Registrar métricas
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{prefix}{metric_name}", metric_value)
            
            # Registrar parâmetros do modelo
            model_params = self.model.get_params()
            for param_name, param_value in model_params.items():
                # Converter para string se não for um tipo primitivo
                if not isinstance(param_value, (int, float, str, bool, type(None))):
                    param_value = str(param_value)
                
                mlflow.log_param(f"{prefix}{param_name}", param_value)
            
            # Registrar tipo do modelo
            mlflow.set_tag(f"{prefix}model_type", self.model_type)
            mlflow.set_tag(f"{prefix}feature_type", self.feature_type)
            
            # Salvar e registrar modelo
            if self.model_type == "lightgbm":
                mlflow.lightgbm.log_model(self.model, artifact_path=f"{prefix}model")
            elif self.model_type == "xgboost":
                mlflow.xgboost.log_model(self.model, artifact_path=f"{prefix}model")
            else:
                mlflow.sklearn.log_model(self.model, artifact_path=f"{prefix}model")
        
        return metrics


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Implementa um ensemble de stacking com modelos especialistas.
    """
    
    def __init__(self, specialist_models, meta_model=None, use_proba=True, 
                 cv=5, random_state=42, threshold=0.5):
        """
        Inicializa o ensemble de stacking.
        
        Args:
            specialist_models: Lista de modelos especialistas
            meta_model: Modelo para combinar previsões dos especialistas
            use_proba: Se True, usa probabilidades dos especialistas; caso contrário, usa classes
            cv: Número de folds para cross-validation ao gerar meta-features
            random_state: Seed para reprodutibilidade
            threshold: Threshold para classificação binária
        """
        self.specialist_models = specialist_models
        self.meta_model = meta_model
        self.use_proba = use_proba
        self.cv = cv
        self.random_state = random_state
        self.threshold = threshold
        self.trained_specialists = None
        self.feature_types = [model.feature_type for model in specialist_models]
    
    def fit(self, X, y):
        """
        Treina o ensemble de stacking.
        
        Args:
            X: Dicionário com features para cada tipo {'demographic': X_demographic, ...}
            y: Target
            
        Returns:
            self
        """
        # Validar entrada
        if not isinstance(X, dict):
            raise ValueError("X deve ser um dicionário com chaves para cada tipo de feature")
        
        # Verificar se os tipos de feature estão presentes
        for feature_type in self.feature_types:
            if feature_type not in X:
                raise ValueError(f"Tipo de feature '{feature_type}' não encontrado em X")
        
        # Gerar meta-features via cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Treinar meta-modelo
        if self.meta_model is None:
            import lightgbm as lgb
            self.meta_model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=5,
                verbose=-1,
                random_state=self.random_state,
                n_jobs=-1,
                scale_pos_weight=50
            )
        
        self.meta_model.fit(meta_features, y)
        
        # Treinar modelos especialistas com todos os dados
        self.trained_specialists = []
        
        for i, model in enumerate(self.specialist_models):
            feature_type = model.feature_type
            model.fit(X[feature_type], y)
            self.trained_specialists.append(model)
        
        return self
    
    def _generate_meta_features(self, X, y):
        """
        Gera meta-features para treinar o meta-modelo.
        
        Args:
            X: Dicionário com features para cada tipo
            y: Target
            
        Returns:
            Array de meta-features
        """
        n_models = len(self.specialist_models)
        n_samples = len(y)
        
        # Inicializar array para meta-features
        if self.use_proba:
            meta_features = np.zeros((n_samples, n_models))
        else:
            meta_features = np.zeros((n_samples, n_models), dtype=int)
        
        # Usar StratifiedKFold para manter proporção das classes
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Para cada fold
        for train_idx, val_idx in kf.split(np.zeros(n_samples), y):
            # Para cada modelo especialista
            for i, model in enumerate(self.specialist_models):
                feature_type = model.feature_type
                
                # Treinar modelo no conjunto de treino
                model_clone = SpecialistModel(
                    model_type=model.model_type,
                    feature_type=model.feature_type,
                    params=model.params,
                    name=model.name
                )
                
                model_clone.fit(X[feature_type].iloc[train_idx], y.iloc[train_idx])
                
                # Gerar previsões para o conjunto de validação
                if self.use_proba:
                    preds = model_clone.predict_proba(X[feature_type].iloc[val_idx])[:, 1]
                else:
                    preds = model_clone.predict(X[feature_type].iloc[val_idx], threshold=self.threshold)
                
                # Armazenar previsões
                meta_features[val_idx, i] = preds
        
        return meta_features
    
    def predict_proba(self, X):
        """
        Gera probabilidades de previsão.
        
        Args:
            X: Dicionário com features para cada tipo
            
        Returns:
            Array com probabilidades
        """
        if self.trained_specialists is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        # Gerar meta-features
        meta_features = np.zeros((len(next(iter(X.values()))), len(self.trained_specialists)))
        
        for i, model in enumerate(self.trained_specialists):
            feature_type = model.feature_type
            
            if self.use_proba:
                preds = model.predict_proba(X[feature_type])[:, 1]
            else:
                preds = model.predict(X[feature_type], threshold=self.threshold)
            
            meta_features[:, i] = preds
        
        # Gerar previsões com meta-modelo
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X, threshold=None):
        """
        Gera previsões binárias.
        
        Args:
            X: Dicionário com features para cada tipo
            threshold: Threshold para classificação binária
            
        Returns:
            Array com previsões binárias
        """
        if threshold is None:
            threshold = self.threshold
            
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)
    
    def save(self, dirpath):
        """
        Salva o ensemble em disco.
        
        Args:
            dirpath: Diretório para salvar os modelos
        """
        if self.trained_specialists is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        # Criar diretório se não existir
        os.makedirs(dirpath, exist_ok=True)
        
        # Salvar meta-modelo
        meta_model_path = os.path.join(dirpath, "meta_model.joblib")
        joblib.dump(self.meta_model, meta_model_path)
        
        # Salvar configuração
        config = {
            'use_proba': self.use_proba,
            'cv': self.cv,
            'random_state': self.random_state,
            'threshold': self.threshold,
            'feature_types': self.feature_types
        }
        
        config_path = os.path.join(dirpath, "config.joblib")
        joblib.dump(config, config_path)
        
        # Salvar especialistas
        specialists_dir = os.path.join(dirpath, "specialists")
        os.makedirs(specialists_dir, exist_ok=True)
        
        for i, model in enumerate(self.trained_specialists):
            model_path = os.path.join(specialists_dir, f"specialist_{model.feature_type}.joblib")
            model.save(model_path)
    
    @classmethod
    def load(cls, dirpath):
        """
        Carrega um ensemble salvo.
        
        Args:
            dirpath: Diretório com os modelos salvos
            
        Returns:
            Instância de StackingEnsemble com os modelos carregados
        """
        # Carregar configuração
        config_path = os.path.join(dirpath, "config.joblib")
        config = joblib.load(config_path)
        
        # Carregar meta-modelo
        meta_model_path = os.path.join(dirpath, "meta_model.joblib")
        meta_model = joblib.load(meta_model_path)
        
        # Carregar especialistas
        specialists_dir = os.path.join(dirpath, "specialists")
        
        specialist_models = []
        for feature_type in config['feature_types']:
            model_path = os.path.join(specialists_dir, f"specialist_{feature_type}.joblib")
            model = SpecialistModel.load(model_path, feature_type=feature_type)
            specialist_models.append(model)
        
        # Criar instância
        instance = cls(
            specialist_models=specialist_models,
            meta_model=meta_model,
            use_proba=config['use_proba'],
            cv=config['cv'],
            random_state=config['random_state'],
            threshold=config['threshold']
        )
        
        instance.trained_specialists = specialist_models
        instance.feature_types = config['feature_types']
        
        return instance
    
    def log_to_mlflow(self, X, y, run_id=None, log_specialists=True):
        """
        Registra métricas e modelos no MLflow.
        
        Args:
            X: Dicionário com features para cada tipo
            y: Target
            run_id: ID do run MLflow (opcional)
            log_specialists: Se True, registra também os modelos especialistas
            
        Returns:
            Dicionário com métricas
        """
        if self.trained_specialists is None:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        # Gerar previsões
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        # Calcular métricas para diferentes thresholds
        thresholds = np.arange(0.01, 0.5, 0.01)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Calcular métricas com threshold ótimo
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        pr_auc = average_precision_score(y, y_pred_proba)
        
        # Calcular positivos previstos
        positive_pct = y_pred.mean() * 100
        
        # Preparar métricas
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': best_f1,
            'pr_auc': pr_auc,
            'threshold': best_threshold,
            'positive_pct': positive_pct
        }
        
        # Registrar no MLflow
        with mlflow.start_run(run_id=run_id) as run:
            # Registrar métricas
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"ensemble_{metric_name}", metric_value)
            
            # Registrar configuração
            mlflow.log_param("ensemble_threshold", best_threshold)
            mlflow.log_param("ensemble_cv", self.cv)
            mlflow.log_param("ensemble_use_proba", self.use_proba)
            
            # Registrar tipo do modelo
            mlflow.set_tag("model_type", "stacking_ensemble")
            mlflow.set_tag("specialist_count", len(self.trained_specialists))
            
            # Salvar e registrar meta-modelo
            # Verificar tipo do meta-modelo
            if 'LGBMClassifier' in str(type(self.meta_model)):
                mlflow.lightgbm.log_model(self.meta_model, artifact_path="meta_model")
            elif 'XGBClassifier' in str(type(self.meta_model)):
                mlflow.xgboost.log_model(self.meta_model, artifact_path="meta_model")
            else:
                mlflow.sklearn.log_model(self.meta_model, artifact_path="meta_model")
            
            # Registrar especialistas
            if log_specialists:
                for model in self.trained_specialists:
                    model.log_to_mlflow(X[model.feature_type], y, run_id=run.info.run_id)
        
        return metrics


def prepare_specialist_data(df, feature_groups, target_col="target"):
    """
    Prepara os dados para os modelos especialistas.
    
    Args:
        df: DataFrame com todas as features
        feature_groups: Dicionário com grupos de features
        target_col: Nome da coluna target
        
    Returns:
        Tupla (X, y) onde X é um dicionário com DataFrames para cada tipo de feature
    """
    # Extrair target
    y = df[target_col] if target_col in df.columns else None
    
    # Criar dicionário de features
    X = {}
    
    for group_name, features in feature_groups.items():
        if features:  # Verificar se a lista não está vazia
            X[group_name] = df[features].copy()
    
    return X, y