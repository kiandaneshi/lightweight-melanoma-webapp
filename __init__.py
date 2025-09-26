"""
DermAI-Melanoma Training Pipeline

A comprehensive machine learning pipeline for melanoma classification
featuring both benchmark and mobile-optimized models.

Modules:
- config: Configuration management
- download_dataset: Kaggle dataset acquisition
- preprocess_data: Data preprocessing and patient-level splitting
- train_benchmark: High-performance model training
- train_mobile: Mobile-optimized model training
- evaluate_models: Comprehensive model evaluation
- export_models: Model export to TensorFlow Lite and TensorFlow.js
- grad_cam_utils: Explainability through Grad-CAM visualization

Usage:
    python -m training.run_full_pipeline
"""

__version__ = "1.0.0"
__author__ = "DermAI Research Team"
__description__ = "Machine learning pipeline for melanoma classification"

# Import main modules for easy access
from .config import Config
from .download_dataset import main as download_dataset
from .preprocess_data import main as preprocess_data
from .train_benchmark import main as train_benchmark
from .train_mobile import main as train_mobile
from .evaluate_models import main as evaluate_models
from .export_models import main as export_models
from .grad_cam_utils import main as generate_grad_cam

__all__ = [
    'Config',
    'download_dataset',
    'preprocess_data', 
    'train_benchmark',
    'train_mobile',
    'evaluate_models',
    'export_models',
    'generate_grad_cam'
]
