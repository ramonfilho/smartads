# smart_ads_pipeline/__init__.py

# Core
from .core.base_component import BaseComponent
from .core.pipeline_state import PipelineState
from .core.extended_parameter_manager import ExtendedParameterManager

# Data Handlers
from .data_handlers.data_loader import DataLoader
from .data_handlers.data_matcher import DataMatcher

# Components
from .components.data_preprocessor import DataPreprocessor
from .components.feature_engineer import FeatureEngineer
from .components.text_processor import TextProcessor
from .components.professional_features import ProfessionalFeatures
from .components.feature_selector import FeatureSelector

# Pipelines
from .pipelines.training_pipeline import TrainingPipeline

__all__ = [
    # Core
    'BaseComponent',
    'PipelineState',
    'ExtendedParameterManager',
    
    # Data Handlers
    'DataLoader',
    'DataMatcher',
    
    # Components
    'DataPreprocessor',
    'FeatureEngineer',
    'TextProcessor',
    'ProfessionalFeatures',
    'FeatureSelector',
    
    # Pipelines
    'TrainingPipeline'
]