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
        features_file = self.model_path / f"features_ordenadas_{self.model_name}.json"
        metadata_file = self.model_path / f"model_metadata_{self.model_name}.json"

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

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza predições no DataFrame processado.

        Args:
            df: DataFrame processado pelo pipeline

        Returns:
            DataFrame original com colunas de score adicionadas
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

        # Adicionar scores ao DataFrame original
        result_df = df.copy()
        result_df['lead_score'] = scores
        result_df['lead_score_percentual'] = scores * 100

        # Calcular decil (1 = melhor, 10 = pior)
        result_df['decil'] = pd.qcut(
            scores,
            q=10,
            labels=range(10, 0, -1),  # 10 = melhor score, 1 = pior score
            duplicates='drop'
        )

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