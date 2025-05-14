# src/inference/__init__.py
"""
Módulo de inferência para o projeto Smart Ads.
"""

# Importar componentes da pipeline de inferência
from .EmailNormalizationTransformer import EmailNormalizationTransformer
from .CompletePreprocessingTransformer import CompletePreprocessingTransformer
from .TextFeatureEngineeringTransformer import TextFeatureEngineeringTransformer
from .GMM_InferenceWrapper import GMM_InferenceWrapper

# Expor componentes para facilitar importação
__all__ = [
    'EmailNormalizationTransformer',
    'CompletePreprocessingTransformer',
    'TextFeatureEngineeringTransformer',
    'GMM_InferenceWrapper'
]