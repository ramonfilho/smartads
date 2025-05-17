"""
Módulo que define as classes usadas para calibração do modelo GMM.
Este módulo é compartilhado entre o script de recalibração e o módulo de inferência.
"""
import numpy as np

class IdentityCalibratedModel:
    """
    Classe que emula o CalibratedClassifierCV original.
    """
    def __init__(self, base_estimator, threshold=0.1):
        self.base_estimator = base_estimator
        self.threshold = threshold
        
    def predict_proba(self, X):
        """
        Retorna as probabilidades do estimador base.
        """
        return self.base_estimator.predict_proba(X)
    
    def predict(self, X):
        """
        Aplica o threshold às probabilidades.
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)