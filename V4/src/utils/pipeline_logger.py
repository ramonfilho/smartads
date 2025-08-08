# src/utils/pipeline_logger.py
import logging
from datetime import datetime

class PipelineLogger:
    def __init__(self, log_file='pipeline_execution.log'):
        self.logger = logging.getLogger('pipeline')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_model_saved(self, model_type, model_name, details=None):
        self.logger.info(f"✓ Model saved: {model_type} - {model_name}")
        if details:
            self.logger.debug(f"  Details: {details}")
    
    def log_model_loaded(self, model_type, model_name, success=True):
        if success:
            self.logger.info(f"✓ Model loaded: {model_type} - {model_name}")
        else:
            self.logger.warning(f"✗ Model NOT found: {model_type} - {model_name}")
    
    def log_features_created(self, feature_type, count):
        self.logger.info(f"✓ Features created: {feature_type} - {count} features")