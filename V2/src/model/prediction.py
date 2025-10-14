"""
Módulo de predição para lead scoring.
Carrega modelos treinados e realiza predições.
"""

import pickle
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LeadScoringPredictor:
    """Classe para realizar predições de lead scoring."""

    def __init__(self, model_name: str = "v1_devclub_rf_temporal"):
        """
        Inicializa o preditor com um modelo específico.

        Args:
            model_name: Nome do modelo a ser carregado (sem extensão)
        """
        self.model_name = model_name
        self.model = None
        self.feature_names = None
        self.metadata = None
        self.model_path = Path(__file__).parent.parent.parent / "arquivos_modelo"

    def load_model(self):
        """Carrega o modelo pickle e seus metadados."""
        model_file = self.model_path / f"modelo_lead_scoring_{self.model_name}.pkl"
        features_file = self.model_path / f"features_ordenadas_{self.model_name}-3.json"
        metadata_file = self.model_path / f"model_metadata_{self.model_name}-3.json"

        logger.info(f"Carregando modelo: {model_file}")
        try:
            # Tentar primeiro com joblib (mais robusto)
            self.model = joblib.load(model_file)
            logger.info("Modelo carregado com joblib")
        except Exception as e:
            logger.info(f"Joblib falhou ({e}), tentando com pickle...")
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Modelo carregado com pickle")

        logger.info(f"Carregando features esperadas: {features_file}")
        with open(features_file, 'r') as f:
            features_data = json.load(f)
            self.feature_names = features_data['feature_names']

        logger.info(f"Carregando metadados: {metadata_file}")
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        logger.info(f"Modelo carregado: {self.model_name}")
        logger.info(f"Features esperadas: {len(self.feature_names)}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara o DataFrame para ter exatamente as features esperadas pelo modelo.

        Args:
            df: DataFrame processado pelo pipeline

        Returns:
            DataFrame com features alinhadas ao modelo
        """
        logger.info(f"Preparando features para predição...")
        logger.info(f"Colunas recebidas: {len(df.columns)}")

        # Criar DataFrame com as features esperadas
        X = pd.DataFrame()

        missing_features = []
        for feature in self.feature_names:
            if feature in df.columns:
                X[feature] = df[feature]
            else:
                # Feature ausente - preencher com 0
                missing_features.append(feature)
                X[feature] = 0

        if missing_features:
            logger.warning(f"Features ausentes (preenchidas com 0): {len(missing_features)}")
            logger.debug(f"Features ausentes: {missing_features[:10]}...")  # Mostrar primeiras 10

        # Garantir ordem correta das colunas
        X = X[self.feature_names]

        logger.info(f"Features preparadas: {X.shape}")
        return X

    def predict(self, df: pd.DataFrame, original_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Realiza predições no DataFrame processado.

        Args:
            df: DataFrame processado pelo pipeline
            original_df: DataFrame original com todas as colunas (opcional)

        Returns:
            DataFrame com colunas de score adicionadas (original se fornecido, processado caso contrário)
        """
        if self.model is None:
            self.load_model()

        # Preparar features
        X = self.prepare_features(df)

        # Fazer predições
        logger.info("Realizando predições...")

        # Probabilidades para cada classe
        probabilities = self.model.predict_proba(X)

        # Score é a probabilidade da classe positiva (1)
        scores = probabilities[:, 1]

        # SEMPRE usar DataFrame processado (sem duplicatas)
        # O original pode ter mais linhas se houve remoção de duplicatas no pipeline
        result_df = df.copy()
        logger.info(f"Usando DataFrame processado ({len(df)} registros únicos)")

        if original_df is not None and len(original_df) != len(df):
            duplicatas_removidas = len(original_df) - len(df)
            logger.warning(f"⚠️  {duplicatas_removidas} duplicatas foram removidas no pipeline")
            logger.info("Retornando apenas registros únicos com scores")

        # Adicionar apenas score e decil
        result_df['lead_score'] = scores

        # Calcular decil (adaptado para lotes pequenos)
        if len(scores) >= 10:
            # Para lotes grandes: usar quantis exatos como no Colab
            result_df['decil'] = pd.qcut(
                scores,
                q=10,
                labels=[f'D{i}' for i in range(1, 11)],
                duplicates='drop'
            )
        else:
            # Para lotes pequenos: calcular decil baseado na probabilidade
            # Usar os mesmos thresholds que seriam usados em um conjunto maior
            # D10 = top 10% (score >= 0.9), D9 = 80-90%, etc.
            decil_values = []
            for score in scores:
                if score >= 0.9:
                    decil = 'D10'
                elif score >= 0.8:
                    decil = 'D9'
                elif score >= 0.7:
                    decil = 'D8'
                elif score >= 0.6:
                    decil = 'D7'
                elif score >= 0.5:
                    decil = 'D6'
                elif score >= 0.4:
                    decil = 'D5'
                elif score >= 0.3:
                    decil = 'D4'
                elif score >= 0.2:
                    decil = 'D3'
                elif score >= 0.1:
                    decil = 'D2'
                else:
                    decil = 'D1'
                decil_values.append(decil)
            result_df['decil'] = decil_values

        logger.info(f"Predições concluídas para {len(result_df)} registros")
        logger.info(f"Score médio: {scores.mean():.4f}")
        logger.info(f"Score min/max: {scores.min():.4f} / {scores.max():.4f}")

        # Estatísticas por decil
        decil_stats = result_df.groupby('decil')['lead_score'].agg(['count', 'mean']).round(4)
        logger.info(f"Distribuição por decil:\n{decil_stats}")

        return result_df

    def predict_single(self, lead_data: dict) -> float:
        """
        Realiza predição para um único lead.

        Args:
            lead_data: Dicionário com os dados do lead

        Returns:
            Score de probabilidade (0-1)
        """
        if self.model is None:
            self.load_model()

        # Converter para DataFrame
        df = pd.DataFrame([lead_data])

        # Preparar features
        X = self.prepare_features(df)

        # Fazer predição
        probability = self.model.predict_proba(X)[0, 1]

        return probability

    def get_feature_importances(self, top_n: int = 20):
        """
        Retorna as feature importances do modelo RandomForest.

        Args:
            top_n: Número de features mais importantes a retornar (padrão: 20)

        Returns:
            Lista de dicts com 'feature' e 'importance', ordenados por importância decrescente
        """
        if self.model is None:
            self.load_model()

        # Verificar se o modelo tem feature_importances_
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Modelo não possui feature_importances_")
            return []

        # Obter importâncias
        importances = self.model.feature_importances_

        # Criar lista de dicts com feature name e importance
        feature_importance_list = [
            {'feature': name, 'importance': float(imp)}
            for name, imp in zip(self.feature_names, importances)
        ]

        # Ordenar por importância decrescente
        feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)

        # Retornar top N
        return feature_importance_list[:top_n]