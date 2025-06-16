# smart_ads_pipeline/pipelines/training_pipeline.py

import os
import sys
import pandas as pd
import numpy as np
import logging
import joblib
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split

# Adicionar o diretório do projeto ao path
project_root = "/Users/ramonmoreira/desktop/smart_ads"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smart_ads_pipeline.core import PipelineState, ExtendedParameterManager
from smart_ads_pipeline.data_handlers import DataLoader, DataMatcher
from smart_ads_pipeline.components import (
    DataPreprocessor,
    FeatureEngineer,
    TextProcessor,
    ProfessionalFeatures,
    FeatureSelector
)

# Importar funções de checkpoint do pipeline original
from unified_pipeline import (
    save_checkpoint,
    load_checkpoint,
    clear_checkpoints
)

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Pipeline completo de treinamento do Smart Ads.
    
    Orquestra todo o processo desde o carregamento dos dados até
    o salvamento do modelo e parâmetros.
    """
    
    def __init__(self):
        """Inicializa o pipeline de treino."""
        self.state = PipelineState()
        self.param_manager = ExtendedParameterManager()
        
        # Componentes
        self.data_loader = None
        self.data_matcher = None
        self.preprocessor = None
        self.feature_engineer = None
        self.text_processor = None
        self.professional_features = None
        self.feature_selector = None
        
        logger.info("TrainingPipeline inicializado")
    
    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa o pipeline completo de treino.
        
        Args:
            config: Dicionário de configuração com:
                - data_path: Caminho para dados brutos
                - output_dir: Diretório para salvar resultados
                - test_size: Proporção para teste (default: 0.3)
                - val_size: Proporção de val sobre test+val (default: 0.5)
                - random_state: Seed para reprodutibilidade (default: 42)
                - max_features: Número máximo de features (default: 300)
                - fast_mode: Modo rápido para feature selection (default: True)
                - use_checkpoints: Usar checkpoints (default: True)
                - clear_cache: Limpar cache antes de começar (default: False)
                - train_model: Se deve treinar um modelo (default: False)
                
        Returns:
            Dicionário com resultados do processo
        """
        logger.info("Iniciando pipeline de treino")
        logger.info(f"Configuração: {config}")
        
        # Configurações
        data_path = config['data_path']
        output_dir = config['output_dir']
        test_size = config.get('test_size', 0.3)
        val_size = config.get('val_size', 0.5)
        random_state = config.get('random_state', 42)
        max_features = config.get('max_features', 300)
        fast_mode = config.get('fast_mode', True)
        use_checkpoints = config.get('use_checkpoints', True)
        clear_cache = config.get('clear_cache', False)
        train_model = config.get('train_model', False)
        
        # Armazenar config no state
        self.state.config = config
        
        # Limpar cache se solicitado
        if clear_cache and use_checkpoints:
            clear_checkpoints()
        
        try:
            # ========================================================================
            # PARTE 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS
            # ========================================================================
            
            # Verificar checkpoint
            checkpoint_data_prep = None
            if use_checkpoints:
                checkpoint_data_prep = load_checkpoint('data_preparation')
            
            if checkpoint_data_prep:
                logger.info("Usando checkpoint de preparação de dados")
                train_df = checkpoint_data_prep['train']
                val_df = checkpoint_data_prep['val'] 
                test_df = checkpoint_data_prep['test']
            else:
                logger.info("\n=== PARTE 1: Carregamento e Preparação dos Dados ===")
                
                # 1.1 Carregar dados
                self.data_loader = DataLoader(data_path)
                data_dict = self.data_loader.load_training_data()
                self.state.log_step("data_loading", {
                    'surveys': data_dict['surveys'].shape,
                    'buyers': data_dict['buyers'].shape,
                    'utms': data_dict['utms'].shape
                })
                
                # 1.2 Matching e criação de target
                self.data_matcher = DataMatcher()
                final_df = self.data_matcher.match_and_create_target(data_dict)
                
                # NOVO: Aplicar amostragem se configurado
                sample_fraction = config.get('sample_fraction', None)
                if sample_fraction and sample_fraction < 1.0:
                    logger.info(f"Aplicando amostragem: {sample_fraction*100:.0f}% dos dados")
                    n_before = len(final_df)
                    final_df = final_df.sample(frac=sample_fraction, random_state=random_state)
                    n_after = len(final_df)
                    logger.info(f"Amostragem aplicada: {n_before} → {n_after} linhas")
                
                self.state.log_step("data_matching", {
                    'final_shape': final_df.shape,
                    'target_rate': (final_df['target'] == 1).mean() if 'target' in final_df.columns else 0
                })
                
                # 1.3 CLASSIFICAR COLUNAS ANTES DE QUALQUER PROCESSAMENTO
                logger.info("Classificando tipos de colunas...")
                from src.utils.column_type_classifier import ColumnTypeClassifier
                
                classifier = ColumnTypeClassifier(
                    use_llm=False,
                    use_classification_cache=True,
                    confidence_threshold=0.7
                )
                
                # Classificar o DataFrame completo
                classifications = classifier.classify_dataframe(final_df)
                
                # Salvar classificações no param_manager
                self.param_manager.save_preprocessing_params('column_classifications', classifications)
                logger.info(f"Classificações salvas para {len(classifications)} colunas")
                
                # 1.4 Split train/val/test
                logger.info("Dividindo dados em train/val/test...")
                
                # Estratificar se possível
                stratify_col = final_df['target'] if 'target' in final_df.columns else None
                
                # Primeira divisão: train vs (val + test)
                train_df, temp_df = train_test_split(
                    final_df,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify_col
                )
                
                # Segunda divisão: val vs test
                stratify_temp = temp_df['target'] if 'target' in temp_df.columns else None
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=val_size,
                    random_state=random_state,
                    stratify=stratify_temp
                )
                
                logger.info(f"Shapes: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")
                
                # Salvar checkpoint
                if use_checkpoints:
                    save_checkpoint({
                        'train': train_df,
                        'val': val_df,
                        'test': test_df
                    }, 'data_preparation')
            
            # Atualizar state
            self.state.update_dataframes(train=train_df, val=val_df, test=test_df)
            
            # ========================================================================
            # PARTE 2: PROCESSAMENTO DE FEATURES
            # ========================================================================
            
            # Verificar checkpoint
            checkpoint_features = None
            if use_checkpoints:
                checkpoint_features = load_checkpoint('feature_processing')
            
            if checkpoint_features:
                logger.info("Usando checkpoint de processamento de features")
                train_df = checkpoint_features['train']
                val_df = checkpoint_features['val']
                test_df = checkpoint_features['test']
                # Carregar param_manager
                if 'param_manager_path' in checkpoint_features:
                    self.param_manager.load(checkpoint_features['param_manager_path'])
            else:
                logger.info("\n=== PARTE 2: Processamento de Features ===")
                
                # 2.1 DataPreprocessor
                logger.info("Aplicando DataPreprocessor...")
                self.preprocessor = DataPreprocessor()
                train_df = self.preprocessor.fit_transform(train_df)
                val_df = self.preprocessor.transform(val_df)
                test_df = self.preprocessor.transform(test_df)
                self.preprocessor.save_params(self.param_manager)
                self.state.log_step("data_preprocessing", {"shape_after": train_df.shape})
                
                # 2.2 FeatureEngineer
                logger.info("Aplicando FeatureEngineer...")
                self.feature_engineer = FeatureEngineer()
                train_df = self.feature_engineer.fit_transform(train_df)
                val_df = self.feature_engineer.transform(val_df)
                test_df = self.feature_engineer.transform(test_df)
                self.feature_engineer.save_params(self.param_manager)
                self.state.log_step("feature_engineering", {"shape_after": train_df.shape})
                
                # 2.3 TextProcessor
                logger.info("Aplicando TextProcessor...")
                self.text_processor = TextProcessor()
                # IMPORTANTE: Passar o param_manager que já tem as classificações
                self.text_processor._param_manager = self.param_manager._internal_param_manager
                train_df = self.text_processor.fit_transform(train_df)
                val_df = self.text_processor.transform(val_df)
                test_df = self.text_processor.transform(test_df)
                self.text_processor.save_params(self.param_manager)
                self.state.log_step("text_processing", {"shape_after": train_df.shape})
                
                # 2.4 ProfessionalFeatures
                logger.info("Aplicando ProfessionalFeatures...")
                self.professional_features = ProfessionalFeatures(n_topics=5)
                # IMPORTANTE: Garantir que tem acesso ao param_manager com classificações
                self.professional_features._param_manager = self.param_manager._internal_param_manager
                train_df = self.professional_features.fit_transform(train_df)
                val_df = self.professional_features.transform(val_df)
                test_df = self.professional_features.transform(test_df)
                self.professional_features.save_params(self.param_manager)
                self.state.log_step("professional_features", {"shape_after": train_df.shape})
                logger.info(f"Val após ProfessionalFeatures: {val_df.shape}")
                logger.info(f"Test após ProfessionalFeatures: {test_df.shape}")
                train_df, val_df, test_df = self._align_datasets(train_df, val_df, test_df)

                # Atualizar state
                self.state.update_dataframes(train=train_df, val=val_df, test=test_df)
                
                # Salvar checkpoint
                if use_checkpoints:
                    # Salvar param_manager temporariamente
                    temp_param_path = os.path.join(output_dir, "temp_params.joblib")
                    os.makedirs(output_dir, exist_ok=True)
                    self.param_manager.save(temp_param_path)
                    
                    save_checkpoint({
                        'train': train_df,
                        'val': val_df,
                        'test': test_df,
                        'param_manager_path': temp_param_path
                    }, 'feature_processing')
            
            # Atualizar state
            self.state.update_dataframes(train=train_df, val=val_df, test=test_df)
            
            # ========================================================================
            # PARTE 3: SELEÇÃO DE FEATURES
            # ========================================================================
            
            logger.info("\n=== PARTE 3: Seleção de Features ===")
            
            # Separar target
            X_train = train_df.drop('target', axis=1)
            y_train = train_df['target']
            
            # Aplicar FeatureSelector
            self.feature_selector = FeatureSelector(
                max_features=max_features,
                fast_mode=fast_mode,
                n_folds=3
            )
            
            train_df = self.feature_selector.fit_transform(train_df, y_train)
            val_df = self.feature_selector.transform(val_df)
            test_df = self.feature_selector.transform(test_df)
            
            # Salvar parâmetros
            self.feature_selector.save_params(self.param_manager)
            
            # Atualizar state
            self.state.selected_features = self.feature_selector.selected_features
            self.state.feature_importance = self.feature_selector.feature_importance
            self.state.log_step("feature_selection", {
                "n_features_selected": len(self.feature_selector.selected_features),
                "shape_after": train_df.shape
            })
            
            # ========================================================================
            # PARTE 4: SALVAR RESULTADOS
            # ========================================================================
            
            logger.info("\n=== PARTE 4: Salvando Resultados ===")
            
            # Criar diretório de saída
            os.makedirs(output_dir, exist_ok=True)
            
            # 4.1 Salvar parâmetros
            params_path = os.path.join(output_dir, "pipeline_params.joblib")
            self.param_manager.save(params_path)
            logger.info(f"Parâmetros salvos em: {params_path}")
            
            # 4.2 Salvar datasets processados
            train_path = os.path.join(output_dir, "train.csv")
            val_path = os.path.join(output_dir, "val.csv")
            test_path = os.path.join(output_dir, "test.csv")
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Datasets salvos em: {output_dir}")
            
            # 4.3 Salvar importância das features
            if self.state.feature_importance is not None:
                importance_path = os.path.join(output_dir, "feature_importance.csv")
                self.state.feature_importance.to_csv(importance_path, index=False)
                logger.info(f"Importância das features salva em: {importance_path}")
            
            # ========================================================================
            # PARTE 5: TREINAR MODELO (OPCIONAL)
            # ========================================================================
            
            if train_model:
                logger.info("\n=== PARTE 5: Treinamento de Modelo ===")
                # Atualizar state com os dataframes finais
                self.state.update_dataframes(train=train_df, val=val_df, test=test_df)                
                self._train_model(train_df, val_df, test_df, output_dir)
                # Adicionar informações do modelo ao resultado
                if hasattr(self, 'model_trainer'):
                    results['model_metrics'] = self.state.metrics
                    results['model_artifacts_dir'] = os.path.join(output_dir, "model_artifacts")
                    results['top_decile_lift'] = self.state.metrics.get('test_top_decile_lift', 0)
                    
                    logger.info(f"\n📊 Métricas do Modelo:")
                    logger.info(f"   AUC Test: {self.state.metrics.get('test_auc', 0):.4f}")
                    logger.info(f"   GINI: {self.state.metrics.get('test_gini', 0):.4f}")
                    logger.info(f"   Top Decile Lift: {self.state.metrics.get('test_top_decile_lift', 0):.2f}x")
            
            # ========================================================================
            # RESUMO FINAL
            # ========================================================================
            
            summary = self.state.get_summary()
            logger.info("\n=== RESUMO DO PIPELINE ===")
            logger.info(f"Tempo total: {summary['execution_time_seconds']:.1f} segundos")
            logger.info(f"Features selecionadas: {summary['n_selected_features']}")
            logger.info(f"Shapes finais: {summary['data_shapes']}")
            
            # Retornar resultados
            results = {
                'success': True,
                'params_path': params_path,
                'output_dir': output_dir,
                'summary': summary,
                'train_shape': train_df.shape,
                'val_shape': val_df.shape,
                'test_shape': test_df.shape,
                'selected_features': self.state.selected_features
            }
            
            if train_model:
                results['metrics'] = self.state.metrics
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no pipeline: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'summary': self.state.get_summary()
            }
    def _align_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Garante que todos os datasets tenham exatamente as mesmas colunas.
        
        Args:
            train_df: DataFrame de treino
            val_df: DataFrame de validação  
            test_df: DataFrame de teste
            
        Returns:
            Tupla com os três DataFrames alinhados
        """
        logger.info("Alinhando colunas entre datasets...")
        
        # Usar colunas do treino como referência
        train_columns = train_df.columns.tolist()
        numeric_columns = set(train_df.select_dtypes(include=['number']).columns)
        
        # Função auxiliar para alinhar um dataset com o treino
        def align_to_train(df: pd.DataFrame, reference_columns: list, 
                        numeric_cols: set) -> pd.DataFrame:
            """Alinha um DataFrame para ter as mesmas colunas que o treino"""
            
            # Colunas faltantes
            missing_cols = set(reference_columns) - set(df.columns)
            
            # Adicionar colunas faltantes com valores padrão
            for col in missing_cols:
                if col in numeric_cols:
                    df[col] = 0
                else:
                    df[col] = None
            
            # Remover colunas extras
            extra_cols = set(df.columns) - set(reference_columns)
            if extra_cols:
                df = df.drop(columns=list(extra_cols))
            
            # Garantir mesma ordem
            df = df[reference_columns]
            
            return df
        
        # Alinhar validação e teste
        val_aligned = align_to_train(val_df, train_columns, numeric_columns)
        test_aligned = align_to_train(test_df, train_columns, numeric_columns)
        
        # Verificar alinhamento
        if list(train_df.columns) == list(val_aligned.columns) == list(test_aligned.columns):
            logger.info("✓ Datasets alinhados com sucesso")
        else:
            logger.warning("⚠️ Possível problema no alinhamento de colunas")
        
        # Log estatísticas
        logger.debug(f"Train: {train_df.shape}")
        logger.debug(f"Val: {val_aligned.shape}")
        logger.debug(f"Test: {test_aligned.shape}")
        
        return train_df, val_aligned, test_aligned

    def _train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Any:
        """
        Treina um modelo LightGBM (placeholder).
        
        Args:
            train_df: DataFrame de treino
            val_df: DataFrame de validação
            
        Returns:
            Modelo treinado
        """
        logger.info("Treinando modelo LightGBM...")
        
        # Separar features e target
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_val = val_df.drop('target', axis=1)
        y_val = val_df['target']
        
        # Importar LightGBM
        import lightgbm as lgb
        
        # Parâmetros básicos
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'random_state': 42,
            'verbose': -1
        }
        
        # Criar datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Treinar
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Avaliar
        from sklearn.metrics import roc_auc_score
        
        y_pred_val = model.predict(X_val)
        auc_val = roc_auc_score(y_val, y_pred_val)
        
        logger.info(f"AUC Validação: {auc_val:.4f}")
        
        # Salvar métricas
        self.state.metrics = {
            'auc_val': auc_val,
            'n_estimators': model.num_trees()
        }
        
        return model