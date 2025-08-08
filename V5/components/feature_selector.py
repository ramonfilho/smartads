# smart_ads_pipeline/components/feature_selector.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
import sys

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smart_ads_pipeline.core import BaseComponent, ExtendedParameterManager

logger = logging.getLogger(__name__)


class FeatureSelector(BaseComponent):
    """
    Componente responsável pela seleção de features.
    
    Este componente encapsula apply_feature_selection_pipeline() do pipeline original,
    mantendo a mesma lógica mas com interface OOP.
    
    Responsabilidades:
    - Remover features altamente correlacionadas
    - Calcular importância usando RandomForest (modo rápido)
    - Calcular importância usando RF + LGB + XGB (modo completo)
    - Selecionar top N features
    - Salvar lista de features selecionadas
    """
    
    def __init__(self, max_features: int = 300, 
                 importance_threshold: float = 0.1,
                 correlation_threshold: float = 0.95,
                 fast_mode: bool = True,
                 n_folds: int = 3):
        """
        Inicializa o FeatureSelector.
        
        Args:
            max_features: Número máximo de features a selecionar
            importance_threshold: Threshold mínimo de importância (%)
            correlation_threshold: Threshold para remover correlações altas
            fast_mode: Se True, usa apenas RandomForest (mais rápido)
            n_folds: Número de folds para cross-validation
        """
        super().__init__(name="feature_selector")
        
        self.max_features = max_features
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.fast_mode = fast_mode
        self.n_folds = n_folds
        
        # Parâmetros aprendidos
        self.selected_features = []
        self.feature_importance = None
        self.removed_by_correlation = []
        self.high_corr_pairs = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        """
        Aprende quais features selecionar.
        
        Args:
            X: DataFrame de treino
            y: Target (obrigatório para este componente)
            
        Returns:
            self
        """
        self._validate_input(X)
        if y is None:
            raise ValueError(f"{self.name}: Target (y) é obrigatório para seleção de features")
        
        logger.info(f"{self.name}: Iniciando fit com shape {X.shape}")
        logger.info(f"Configurações: max_features={self.max_features}, "
                   f"fast_mode={self.fast_mode}, correlation_threshold={self.correlation_threshold}")
        
        # Importar funções do pipeline original
        from src.evaluation.feature_importance import (
            identify_text_derived_columns,
            analyze_rf_importance,
            analyze_lgb_importance,
            analyze_xgb_importance,
            combine_importance_results
        )
        
        # Identificar features numéricas (excluindo target se estiver em X)
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        initial_n_features = len(numeric_cols)
        logger.info(f"Features numéricas iniciais: {initial_n_features}")
        
        # Preparar dados
        X_numeric = X[numeric_cols].fillna(0)
        
        # PASSO 1: Remover correlações muito altas
        logger.info("Removendo features altamente correlacionadas...")
        X_numeric, removed_cols, high_corr_pairs = self._remove_high_correlations(
            X_numeric, y, self.correlation_threshold
        )
        self.removed_by_correlation = removed_cols
        self.high_corr_pairs = high_corr_pairs
        
        # Atualizar lista de colunas
        numeric_cols = list(X_numeric.columns)
        logger.info(f"Features após remoção de correlações: {len(numeric_cols)}")
        
        # PASSO 2: Análise de importância
        logger.info("Analisando importância das features...")
        
        if self.fast_mode:
            logger.info("Modo rápido - usando apenas RandomForest")
            
            # Usar RandomForest
            rf_importance, rf_metrics = analyze_rf_importance(
                X_numeric, y, numeric_cols, n_folds=self.n_folds
            )
            
            # Criar estrutura compatível
            self.feature_importance = pd.DataFrame({
                'Feature': rf_importance['Feature'],
                'Mean_Importance': rf_importance['Importance_RF'],
                'Importance_RF': rf_importance['Importance_RF'],
                'Importance_LGB': 0,
                'Importance_XGB': 0,
                'Std_Importance': 0,
                'CV': 0
            })
            
        else:
            logger.info("Modo completo - usando múltiplos modelos")
            
            # RandomForest
            rf_importance, rf_metrics = analyze_rf_importance(
                X_numeric, y, numeric_cols, n_folds=self.n_folds
            )
            
            # LightGBM
            lgb_importance, lgb_metrics = analyze_lgb_importance(
                X_numeric, y, numeric_cols, n_folds=self.n_folds
            )
            
            # XGBoost
            xgb_importance, xgb_metrics = analyze_xgb_importance(
                X_numeric, y, numeric_cols, n_folds=self.n_folds
            )
            
            # Combinar resultados
            self.feature_importance = combine_importance_results(
                rf_importance, lgb_importance, xgb_importance
            )
        
        # PASSO 3: Selecionar top features
        logger.info(f"Selecionando top {self.max_features} features...")
        
        # Filtrar por importância mínima
        min_importance = self.feature_importance['Mean_Importance'].sum() * (self.importance_threshold / 100)
        important_features = self.feature_importance[
            self.feature_importance['Mean_Importance'] >= min_importance
        ]
        
        logger.info(f"Features com importância >= {min_importance:.4f}: {len(important_features)}")
        
        # Selecionar top N
        if len(important_features) > self.max_features:
            top_features = important_features.nlargest(self.max_features, 'Mean_Importance')
        else:
            top_features = important_features
        
        # Salvar features selecionadas
        self.selected_features = top_features['Feature'].tolist()
        
        logger.info(f"✅ {len(self.selected_features)} features selecionadas")
        logger.info(f"   Redução: {initial_n_features} → {len(self.selected_features)} "
                   f"({(1 - len(self.selected_features)/initial_n_features)*100:.1f}% removidas)")
        
        # Log top 10 features
        logger.info("Top 10 features mais importantes:")
        for i, row in top_features.head(10).iterrows():
            logger.info(f"  {i+1}. {row['Feature']}: {row['Mean_Importance']:.4f}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra apenas as features selecionadas.
        
        Args:
            X: DataFrame para transformar
            
        Returns:
            DataFrame com apenas features selecionadas (+ target se existir)
        """
        self._check_is_fitted()
        self._validate_input(X)
        
        logger.info(f"{self.name}: Filtrando para {len(self.selected_features)} features selecionadas")
        
        # Colunas a manter: features selecionadas + target (se existir)
        columns_to_keep = self.selected_features.copy()
        if 'target' in X.columns:
            columns_to_keep.append('target')
        
        # Verificar quais colunas existem
        existing_columns = [col for col in columns_to_keep if col in X.columns]
        missing_columns = [col for col in columns_to_keep if col not in X.columns]
        
        if missing_columns:
            logger.warning(f"{len(missing_columns)} colunas selecionadas não encontradas no DataFrame")
            if len(missing_columns) <= 10:
                logger.warning(f"Colunas faltantes: {missing_columns}")
        
        # Filtrar DataFrame
        X_filtered = X[existing_columns].copy()
        
        logger.info(f"{self.name}: Transform concluído. Shape: {X.shape} → {X_filtered.shape}")
        
        return X_filtered
    
    def _remove_high_correlations(self, X: pd.DataFrame, y: pd.Series, 
                                 threshold: float) -> Tuple[pd.DataFrame, List[str], List[Dict]]:
        """
        Remove features com alta correlação entre si.
        
        Args:
            X: DataFrame com features numéricas
            y: Target
            threshold: Threshold de correlação
            
        Returns:
            X_filtered: DataFrame sem features correlacionadas
            removed: Lista de features removidas
            pairs: Lista de pares correlacionados
        """
        # Calcular matriz de correlação
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Encontrar features para remover
        to_drop = []
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if upper.iloc[i, j] >= threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    
                    # Correlação com target
                    corr_i_target = abs(X[col_i].astype(float).corr(y.astype(float)))
                    corr_j_target = abs(X[col_j].astype(float).corr(y.astype(float)))
                    
                    # Remover a que tem menor correlação com target
                    if corr_i_target < corr_j_target:
                        to_drop.append(col_i)
                    else:
                        to_drop.append(col_j)
                    
                    high_corr_pairs.append({
                        'feature1': col_i,
                        'feature2': col_j,
                        'correlation': upper.iloc[i, j]
                    })
        
        # Remover duplicatas
        to_drop = list(set(to_drop))
        
        if to_drop:
            logger.info(f"Removendo {len(to_drop)} features com alta correlação")
            X_filtered = X.drop(columns=to_drop)
        else:
            X_filtered = X.copy()
        
        return X_filtered, to_drop, high_corr_pairs
    
    def _save_component_params(self, param_manager: ExtendedParameterManager) -> None:
        """Salva parâmetros do componente."""
        params = {
            'selected_features': self.selected_features,
            'n_features_selected': len(self.selected_features),
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
            'removed_by_correlation': self.removed_by_correlation,
            'high_corr_pairs': self.high_corr_pairs,
            'max_features': self.max_features,
            'importance_threshold': self.importance_threshold,
            'correlation_threshold': self.correlation_threshold,
            'fast_mode': self.fast_mode
        }
        
        # Salvar no param_manager
        param_manager.save_component_params(self.name, params)
        param_manager.save_selected_features(self.selected_features)
        
        # Também salvar em feature_selection para compatibilidade
        param_manager.params['feature_selection'].update(params)
    
    def _load_component_params(self, param_manager: ExtendedParameterManager) -> None:
        """Carrega parâmetros do componente."""
        params = param_manager.get_component_params(self.name)
        
        # Se não encontrar em components, tentar em feature_selection
        if not params:
            params = param_manager.params.get('feature_selection', {})
        
        self.selected_features = params.get('selected_features', [])
        self.removed_by_correlation = params.get('removed_by_correlation', [])
        self.high_corr_pairs = params.get('high_corr_pairs', [])
        
        # Recriar DataFrame de importância se disponível
        if 'feature_importance' in params and params['feature_importance']:
            self.feature_importance = pd.DataFrame(params['feature_importance'])
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre a seleção de features.
        
        Returns:
            Dicionário com estatísticas da seleção
        """
        info = {
            'n_features_selected': len(self.selected_features),
            'n_removed_by_correlation': len(self.removed_by_correlation),
            'n_high_corr_pairs': len(self.high_corr_pairs),
            'has_importance_scores': self.feature_importance is not None
        }
        
        if self.feature_importance is not None:
            info['top_5_features'] = self.feature_importance.head(5)['Feature'].tolist()
            info['importance_range'] = {
                'min': self.feature_importance['Mean_Importance'].min(),
                'max': self.feature_importance['Mean_Importance'].max(),
                'mean': self.feature_importance['Mean_Importance'].mean()
            }
        
        return info