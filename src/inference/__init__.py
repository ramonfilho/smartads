"""
Módulo de inferência para o projeto Smart Ads.
"""

from .gmm_inference_pipeline import GMMInferencePipeline
from .gmm_inference_service import start_service

__all__ = ['GMMInferencePipeline', 'start_service']