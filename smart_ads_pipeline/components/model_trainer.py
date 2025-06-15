# smart_ads_pipeline/components/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Componente para treinar modelo com lógica de ranking e decis.
    Baseado nos scripts 05_direct_ranking.py até 08_ranking_production.py
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.calibrated_model = None
        self.decile_thresholds = None
        
    def train_with_ranking(self, X_train, y_train, X_val, y_val):
        """
        Treina modelo com calibração e cálculo de decis.
        Reproduz lógica dos scripts originais.
        """
        logger.info("Treinando modelo com ranking...")
        
        # 1. Treinar RandomForest base (como em 05_direct_ranking.py)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # 2. Calibrar modelo (como em 06_decile_saving.py)
        logger.info("Calibrando modelo...")
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method='isotonic',
            cv=3
        )
        self.calibrated_model.fit(X_train, y_train)
        
        # 3. Calcular probabilidades no conjunto de validação
        val_probs = self.calibrated_model.predict_proba(X_val)[:, 1]
        
        # 4. Calcular decis (como em 06_decile_saving.py)
        self.decile_thresholds = self._calculate_deciles(val_probs, y_val)
        
        # 5. Avaliar no conjunto de validação
        metrics = self._evaluate_model(X_val, y_val)
        
        return metrics
    
    def _calculate_deciles(self, probabilities, true_labels):
        """
        Calcula os thresholds dos decis baseado nas probabilidades.
        """
        # Criar DataFrame para análise
        df_analysis = pd.DataFrame({
            'probability': probabilities,
            'true_label': true_labels
        })
        
        # Ordenar por probabilidade decrescente
        df_analysis = df_analysis.sort_values('probability', ascending=False)
        
        # Calcular decis
        df_analysis['decile'] = pd.qcut(
            df_analysis['probability'].rank(method='first'),
            q=10,
            labels=False,
            duplicates='drop'
        )
        
        # Calcular thresholds para cada decil
        decile_info = []
        for decile in range(10):
            decile_data = df_analysis[df_analysis['decile'] == decile]
            
            if len(decile_data) > 0:
                info = {
                    'decile': decile + 1,
                    'min_prob': decile_data['probability'].min(),
                    'max_prob': decile_data['probability'].max(),
                    'count': len(decile_data),
                    'positives': decile_data['true_label'].sum(),
                    'conversion_rate': decile_data['true_label'].mean()
                }
                decile_info.append(info)
        
        logger.info("Decis calculados:")
        for info in decile_info:
            logger.info(f"  Decil {info['decile']}: "
                       f"Conv={info['conversion_rate']:.2%}, "
                       f"N={info['count']}")
        
        return pd.DataFrame(decile_info)
    
    def predict_with_ranking(self, X):
        """
        Faz predições com ranking baseado em decis.
        """
        if self.calibrated_model is None:
            raise ValueError("Modelo não treinado")
        
        # Obter probabilidades calibradas
        probabilities = self.calibrated_model.predict_proba(X)[:, 1]
        
        # Atribuir decis
        rankings = self._assign_deciles(probabilities)
        
        return probabilities, rankings
    
    def _assign_deciles(self, probabilities):
        """
        Atribui decil para cada probabilidade.
        """
        rankings = np.zeros(len(probabilities), dtype=int)
        
        for i, prob in enumerate(probabilities):
            # Encontrar decil correspondente
            for _, row in self.decile_thresholds.iterrows():
                if row['min_prob'] <= prob <= row['max_prob']:
                    rankings[i] = row['decile']
                    break
        
        return rankings