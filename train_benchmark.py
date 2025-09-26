"""
Train benchmark model (EfficientNetV2/ResNet) for melanoma classification.
This model prioritizes performance over deployment constraints.
Uses EfficientNetV2-S for improved accuracy and faster training.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S, ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from config import Config
from preprocess_data import DataPreprocessor
from advanced_augmentation import create_advanced_augmentation_dataset, get_augmentation_config
from advanced_schedulers import create_advanced_callbacks

class BenchmarkModelTrainer:
    def __init__(self, model_name='efficientnet', input_size=(384, 384)):
        self.model_name = model_name
        self.input_size = input_size
        self.data_dir = Path(Config.DATA_DIR)
        self.models_dir = Path(Config.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.batch_size = 16  # Larger batch size for benchmark model
        self.epochs = 100
        self.learning_rate = 1e-4
        self.preprocessor = DataPreprocessor(input_size=input_size)

    def create_model(self, num_classes=1):
        """Create benchmark model architecture with EfficientNetV2-S."""
        if self.model_name == 'efficientnet':
            print("✓ Using EfficientNetV2-S architecture for improved performance")
            base_model = EfficientNetV2S(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        elif self.model_name == 'resnet':
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Unfreeze the last few layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Add custom head
        inputs = keras.Input(shape=(*self.input_size, 3))
        x = base_model(inputs, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dropout for regularization
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer (cast to float32 for mixed precision stability)
        if num_classes == 1:
            outputs = layers.Dense(1, name='predictions_logits')(x)
            outputs = layers.Activation('sigmoid', dtype='float32', name='predictions')(outputs)
        else:
            outputs = layers.Dense(num_classes, name='predictions_logits')(x)
            outputs = layers.Activation('softmax', dtype='float32', name='predictions')(outputs)
        
        model = Model(inputs, outputs, name=f'benchmark_{self.model_name}')
        return model

    def focal_loss(self, alpha=0.25, gamma=2.0):
        """
        Focal loss for handling class imbalance.
        Focuses training on hard examples.
        """
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Calculate focal loss
            alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
            
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_fixed

    def load_data(self):
        """Load training and validation data."""
        splits_dir = self.data_dir / "processed" / "splits"
        
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Training class balance: {train_df['target'].mean():.3f}")
        
        return train_df, val_df

    def calculate_class_weights(self, train_df):
        """Calculate class weights for handling imbalance."""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['target']),
            y=train_df['target']
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"Class weights: {class_weight_dict}")
        
        return class_weight_dict

    def setup_callbacks(self, model_name):
        """Setup advanced training callbacks with cosine annealing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"{model_name}_{timestamp}"
        model_path.mkdir(exist_ok=True)
        
        # Use advanced callbacks with cosine annealing scheduler
        callbacks = create_advanced_callbacks(
            initial_lr=self.learning_rate,
            total_epochs=self.epochs,
            model_save_path=model_path,
            patience=15,
            use_cosine_annealing=True,
            warmup_epochs=5,
            gradient_clip_norm=1.0  # Add gradient clipping for stability
        )
        
        print("✓ Using cosine annealing learning rate scheduler with warmup")
        print("✓ Gradient clipping enabled for training stability")
        
        return callbacks, model_path

    def create_datasets(self, train_df, val_df):
        """Create TensorFlow datasets with augmentation."""
        print("Creating training dataset...")
        train_dataset = self.preprocessor.create_tf_dataset(
            train_df,
            batch_size=self.batch_size,
            shuffle=True,
            transforms=self.preprocessor.train_transforms
        )
        
        # Apply advanced augmentation (CutMix and MixUp) to training data
        print("Applying advanced augmentation (CutMix & MixUp) to training dataset...")
        aug_config = get_augmentation_config()
        train_dataset = create_advanced_augmentation_dataset(train_dataset, aug_config)
        
        print("Creating validation dataset...")
        val_dataset = self.preprocessor.create_tf_dataset(
            val_df,
            batch_size=self.batch_size,
            shuffle=False,
            transforms=self.preprocessor.val_transforms
        )
        
        return train_dataset, val_dataset

    def train_model(self):
        """Main training loop."""
        print(f"Training benchmark model: {self.model_name}")
        print("=" * 60)
        
        # Load data
        train_df, val_df = self.load_data()
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_df)
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets(train_df, val_df)
        
        # Create model
        model = self.create_model()
        
        # Create optimizer with mixed precision support and gradient clipping
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        
        # Wrap with loss scaling for mixed precision training
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("✓ Loss scaling enabled for mixed precision training")
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=self.focal_loss(alpha=0.75, gamma=2.0),  # Use focal loss for imbalance
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        print(f"Model architecture:")
        model.summary()
        
        # Setup callbacks
        callbacks, model_path = self.setup_callbacks(f"benchmark_{self.model_name}")
        
        # Train model
        print(f"\nStarting training...")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Max epochs: {self.epochs}")
        
        history = model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save final model
        final_model_path = model_path / f"final_{self.model_name}.h5"
        model.save(str(final_model_path))
        print(f"Final model saved to: {final_model_path}")
        
        # Plot training history
        self.plot_training_history(history, model_path)
        
        return model, history, model_path

    def plot_training_history(self, history, save_dir):
        """Plot and save training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # AUC
        axes[1, 0].plot(history.history['auc'], label='Training AUC')
        axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {save_dir}")

def main():
    """Main training function."""
    print("DermAI Benchmark Model Training")
    print("=" * 50)
    
    # Setup TensorFlow and mixed precision
    Config.setup_tensorflow()
    print(f"TensorFlow mixed precision policy: {tf.keras.mixed_precision.global_policy()}")
    
    # Train EfficientNet benchmark model
    print("Training EfficientNet benchmark model...")
    efficientnet_trainer = BenchmarkModelTrainer(
        model_name='efficientnet',
        input_size=(384, 384)
    )
    
    try:
        model, history, model_path = efficientnet_trainer.train_model()
        
        print(f"\n✓ Benchmark model training completed!")
        print(f"Best model saved to: {model_path}")
        
        # Get final metrics
        val_loss = min(history.history['val_loss'])
        val_auc = max(history.history['val_auc'])
        val_accuracy = max(history.history['val_accuracy'])
        
        print(f"\nFinal validation metrics:")
        print(f"Loss: {val_loss:.4f}")
        print(f"AUC: {val_auc:.4f}")
        print(f"Accuracy: {val_accuracy:.4f}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

if __name__ == "__main__":
    main()
