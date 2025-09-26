"""
Configuration settings for the DermAI training pipeline.
Centralized configuration management for all training scripts.
"""

import os
from pathlib import Path

class Config:
    """Configuration class for DermAI training pipeline."""
    
    # Base directories
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    EXPORT_DIR = PROJECT_ROOT / "exported_models"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    EXPORT_DIR.mkdir(exist_ok=True)
    
    # Dataset configuration
    DATASET_CONFIG = {
        'siim_isic': {
            'kaggle_dataset': 'cdeotte/melanoma-384x384',
            'competition': 'siim-isic-melanoma-classification',
            'image_size': (384, 384),
            'channels': 3
        },
        'ph2': {
            'url': 'https://www.fc.up.pt/addi/ph2%20database.html',
            'image_size': (224, 224),
            'channels': 3
        }
    }
    
    # Model configurations
    MODEL_CONFIGS = {
        'benchmark': {
            'architecture': 'efficientnet',
            'backbone': 'EfficientNetB3',
            'input_size': (384, 384),
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 1e-4,
            'optimizer': 'adam',
            'loss': 'focal_loss',
            'metrics': ['accuracy', 'auc', 'precision', 'recall']
        },
        'mobile': {
            'architecture': 'mobilenet',
            'backbone': 'MobileNetV3Small',
            'input_size': (224, 224),
            'batch_size': 32,
            'epochs': 80,
            'learning_rate': 2e-4,
            'optimizer': 'adam',
            'loss': 'focal_loss',
            'metrics': ['accuracy', 'auc', 'precision', 'recall'],
            'quantize': True
        }
    }
    
    # Training configuration
    TRAINING_CONFIG = {
        'patient_split': True,  # Critical for preventing data leakage
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42,
        'stratify': True,
        'class_weight': 'balanced',
        'early_stopping': {
            'patience': 15,
            'monitor': 'val_auc',
            'mode': 'max'
        },
        'reduce_lr': {
            'factor': 0.5,
            'patience': 7,
            'min_lr': 1e-7,
            'monitor': 'val_loss'
        }
    }
    
    # Data augmentation configuration
    AUGMENTATION_CONFIG = {
        'train': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.5,
            'rotate90': 0.5,
            'rotate_limit': 30,
            'rotate_prob': 0.7,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'color_jitter_prob': 0.8,
            'gauss_noise_prob': 0.3,
            'blur_prob': 0.3,
            'cutmix_prob': 0.5,
            'mixup_prob': 0.5
        },
        'validation': {
            'resize_only': True
        }
    }
    
    # Evaluation configuration
    EVALUATION_CONFIG = {
        'metrics': [
            'auroc',
            'auprc', 
            'sensitivity',
            'specificity',
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'brier_score'
        ],
        'plots': [
            'roc_curve',
            'precision_recall_curve',
            'confusion_matrix',
            'calibration_curve',
            'prediction_distribution'
        ],
        'thresholds': {
            'method': 'youden_j',  # or 'roc_optimal', 'pr_optimal'
            'custom': None
        }
    }
    
    # Grad-CAM configuration
    GRADCAM_CONFIG = {
        'target_layers': {
            'efficientnet': 'top_conv',
            'mobilenet': 'conv_pw_13',
            'resnet': 'conv5_block3_out'
        },
        'colormap': 'jet',
        'alpha': 0.6,
        'batch_size': 10
    }
    
    # Export configuration
    EXPORT_CONFIG = {
        'tflite': {
            'quantize': True,
            'optimization': 'default',
            'representative_dataset_size': 100
        },
        'tfjs': {
            'quantization_bytes': 1,  # uint8
            'optimize': True,
            'weight_shard_size_bytes': 4 * 1024 * 1024  # 4MB
        },
        'saved_model': {
            'include_optimizer': False
        }
    }
    
    # Environment configuration
    ENV_CONFIG = {
        'gpu_memory_growth': True,
        'mixed_precision': True,  # Enable for better performance on newer GPUs
        'seed': 42,
        'deterministic': True
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'save_logs': True,
        'log_dir': PROJECT_ROOT / 'logs'
    }
    
    # Medical and ethical guidelines
    MEDICAL_CONFIG = {
        'disclaimer': {
            'text': "This is an educational research demonstration only. "
                   "Not a medical device. Do not use for diagnostic purposes. "
                   "Always consult qualified medical professionals for health concerns.",
            'required': True,
            'prominent': True
        },
        'data_handling': {
            'patient_privacy': True,
            'data_anonymization': True,
            'local_processing': True,
            'no_cloud_upload': True
        },
        'model_limitations': {
            'dataset_bias': "Model trained on specific demographic populations",
            'generalization': "Performance may vary across different populations",
            'scope': "Limited to dermoscopic image analysis only"
        }
    }
    
    @classmethod
    def get_model_config(cls, model_type):
        """Get configuration for specific model type."""
        if model_type not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls.MODEL_CONFIGS[model_type]
    
    @classmethod
    def get_dataset_config(cls, dataset_name):
        """Get configuration for specific dataset."""
        if dataset_name not in cls.DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return cls.DATASET_CONFIG[dataset_name]
    
    @classmethod
    def setup_tensorflow(cls):
        """Setup TensorFlow with recommended settings."""
        import tensorflow as tf
        import logging
        
        # Set random seeds for reproducibility
        if cls.ENV_CONFIG['deterministic']:
            import random
            import numpy as np
            random.seed(cls.ENV_CONFIG['seed'])
            np.random.seed(cls.ENV_CONFIG['seed'])
            tf.random.set_seed(cls.ENV_CONFIG['seed'])
        
        # GPU configuration
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    if cls.ENV_CONFIG['gpu_memory_growth']:
                        tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        
        # Mixed precision (for performance on newer GPUs)
        if cls.ENV_CONFIG['mixed_precision']:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✓ Mixed precision enabled - expect ~1.5-2x training speedup")
            except Exception as e:
                print(f"⚠ Failed to enable mixed precision: {e}")
                print("Falling back to float32 training")
        
        # Reduce TF logging verbosity
        tf.get_logger().setLevel('ERROR')
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration."""
        import logging
        
        # Create logs directory
        cls.LOGGING_CONFIG['log_dir'].mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, cls.LOGGING_CONFIG['level']),
            format=cls.LOGGING_CONFIG['format']
        )
        
        if cls.LOGGING_CONFIG['save_logs']:
            # Add file handler
            file_handler = logging.FileHandler(
                cls.LOGGING_CONFIG['log_dir'] / 'training.log'
            )
            file_handler.setFormatter(
                logging.Formatter(cls.LOGGING_CONFIG['format'])
            )
            logging.getLogger().addHandler(file_handler)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings."""
        errors = []
        
        # Check required directories exist
        required_dirs = [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR]
        for directory in required_dirs:
            if not directory.exists():
                errors.append(f"Required directory does not exist: {directory}")
        
        # Validate model configurations
        for model_name, config in cls.MODEL_CONFIGS.items():
            if 'input_size' not in config:
                errors.append(f"Missing input_size for model: {model_name}")
            if 'batch_size' not in config:
                errors.append(f"Missing batch_size for model: {model_name}")
        
        # Check for Kaggle credentials
        kaggle_config_exists = (
            os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')) or
            ('KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ)
        )
        
        if not kaggle_config_exists:
            errors.append("Kaggle API credentials not found")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        print("Configuration validation passed ✓")
    
    @classmethod
    def print_config_summary(cls):
        """Print summary of current configuration."""
        print("DermAI Configuration Summary")
        print("=" * 40)
        print(f"Project root: {cls.PROJECT_ROOT}")
        print(f"Data directory: {cls.DATA_DIR}")
        print(f"Models directory: {cls.MODELS_DIR}")
        print(f"Results directory: {cls.RESULTS_DIR}")
        print(f"Export directory: {cls.EXPORT_DIR}")
        print()
        print("Model configurations:")
        for name, config in cls.MODEL_CONFIGS.items():
            print(f"  {name}: {config['architecture']} ({config['input_size']})")
        print()
        print("Training configuration:")
        print(f"  Patient-level splitting: {cls.TRAINING_CONFIG['patient_split']}")
        print(f"  Test size: {cls.TRAINING_CONFIG['test_size']}")
        print(f"  Validation size: {cls.TRAINING_CONFIG['val_size']}")
        print()
        print("Medical disclaimer required: ✓")

# Initialize configuration on import
if __name__ == "__main__":
    Config.setup_tensorflow()
    Config.setup_logging()
    Config.validate_config()
    Config.print_config_summary()
