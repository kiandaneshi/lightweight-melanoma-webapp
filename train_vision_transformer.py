"""
Vision Transformer (ViT) implementation for melanoma classification.
Extends the training pipeline with transformer architectures for comparison with CNNs.
"""

import os
from pathlib import Path
import json
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    TF_AVAILABLE = True
except ImportError as e:
    print(f"ML dependencies not available: {e}")
    TF_AVAILABLE = False

from config import Config

class VisionTransformer:
    """Vision Transformer implementation for medical image classification."""
    
    def __init__(self, 
                 image_size=224,
                 patch_size=16,
                 num_patches=None,
                 projection_dim=768,
                 num_heads=12,
                 transformer_units=[3072, 768],
                 transformer_layers=12,
                 mlp_head_units=[3072, 2048],
                 num_classes=1):
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches or (image_size // patch_size) ** 2
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_head_units
        self.num_classes = num_classes
        
        self.model = None

    def patches(self, images):
        """Extract patches from input images."""
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def patch_encoder(self):
        """Encode patches with position embeddings."""
        
        class PatchEncoder(keras.layers.Layer):
            def __init__(self, num_patches, projection_dim):
                super(PatchEncoder, self).__init__()
                self.num_patches = num_patches
                self.projection = keras.layers.Dense(units=projection_dim)
                self.position_embedding = keras.layers.Embedding(
                    input_dim=num_patches, output_dim=projection_dim
                )

            def call(self, patch):
                positions = tf.range(start=0, limit=self.num_patches, delta=1)
                encoded = self.projection(patch) + self.position_embedding(positions)
                return encoded
        
        return PatchEncoder(self.num_patches, self.projection_dim)

    def mlp(self, x, hidden_units, dropout_rate):
        """Multi-layer perceptron."""
        for units in hidden_units:
            x = keras.layers.Dense(units, activation=tf.nn.gelu)(x)
            x = keras.layers.Dropout(dropout_rate)(x)
        return x

    def build_model(self):
        """Build the Vision Transformer model."""
        
        inputs = keras.layers.Input(shape=(self.image_size, self.image_size, 3))
        
        # Create patches
        patches_layer = keras.layers.Lambda(self.patches)(inputs)
        
        # Encode patches
        encoded_patches = self.patch_encoder()(patches_layer)

        # Create multiple transformer blocks
        for _ in range(self.transformer_layers):
            # Layer normalization 1
            x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            
            # Multi-head attention
            attention_output = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            
            # Skip connection 1
            x2 = keras.layers.Add()([attention_output, encoded_patches])
            
            # Layer normalization 2
            x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP block
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            
            # Skip connection 2
            encoded_patches = keras.layers.Add()([x3, x2])

        # Create classifier head
        representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = keras.layers.GlobalAveragePooling1D()(representation)
        representation = keras.layers.Dropout(0.5)(representation)
        
        # Add MLP head
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        
        # Final classification layer
        if self.num_classes == 1:
            # Binary classification
            logits = keras.layers.Dense(1, activation='sigmoid', name='predictions')(features)
        else:
            # Multi-class classification
            logits = keras.layers.Dense(self.num_classes, activation='softmax', name='predictions')(features)
        
        self.model = keras.Model(inputs=inputs, outputs=logits)
        return self.model

    def compile_model(self, learning_rate=1e-4):
        """Compile the model with appropriate loss and metrics."""
        if self.num_classes == 1:
            loss = 'binary_crossentropy'
            metrics = ['binary_accuracy', tf.keras.metrics.AUC(name='auc')]
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()]
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return self.model

class ViTTrainer:
    """Trainer for Vision Transformer melanoma classification."""
    
    def __init__(self, 
                 data_dir=None,
                 models_dir=None,
                 image_size=224,
                 batch_size=32,
                 epochs=100):
        
        self.data_dir = Path(data_dir) if data_dir else Path(Config.DATA_DIR)
        self.models_dir = Path(models_dir) if models_dir else Path(Config.MODELS_DIR)
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Create output directory
        self.output_dir = self.models_dir / f"vit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"ViT training output directory: {self.output_dir}")

    def create_datasets(self):
        """Create train, validation, and test datasets."""
        
        # Load the data splits
        splits_dir = self.data_dir / "processed" / "splits"
        
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and test
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            directory=self.data_dir / "processed" / "images",
            x_col='image_name',
            y_col='target',
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='binary'
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            directory=self.data_dir / "processed" / "images",
            x_col='image_name',
            y_col='target',
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='binary'
        )
        
        test_generator = val_datagen.flow_from_dataframe(
            test_df,
            directory=self.data_dir / "processed" / "images",
            x_col='image_name',
            y_col='target',
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='binary'
        )
        
        return train_generator, val_generator, test_generator

    def train_model(self):
        """Train the Vision Transformer model."""
        
        if not TF_AVAILABLE:
            print("TensorFlow not available. Skipping ViT training.")
            return None
        
        print("Creating Vision Transformer model...")
        
        # Create ViT model
        vit = VisionTransformer(
            image_size=self.image_size,
            patch_size=16,
            projection_dim=768,
            num_heads=12,
            transformer_layers=12,
            num_classes=1  # Binary classification
        )
        
        model = vit.build_model()
        model = vit.compile_model(learning_rate=1e-4)
        
        # Print model summary
        model.summary()
        
        # Create datasets
        train_gen, val_gen, test_gen = self.create_datasets()
        
        # Calculate steps per epoch
        train_steps = len(train_gen)
        val_steps = len(val_gen)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(self.output_dir / "best_vit_melanoma.h5"),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=20,
                restore_best_weights=True
            ),
            keras.callbacks.CSVLogger(
                str(self.output_dir / "training_log.csv"),
                append=True
            )
        ]
        
        print(f"Starting ViT training for {self.epochs} epochs...")
        
        # Train the model
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(str(self.output_dir / "final_vit_melanoma.h5"))
        
        # Save training history
        with open(self.output_dir / "history.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
            json.dump(history_dict, f, indent=2)
        
        print("ViT training completed!")
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_results = model.evaluate(test_gen, verbose=1)
        
        # Save test results
        test_metrics = dict(zip(model.metrics_names, test_results))
        with open(self.output_dir / "test_results.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        
        print(f"Test Results: {test_metrics}")
        
        return model, history

if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("TensorFlow dependencies not available. Install with:")
        print("pip install tensorflow pandas scikit-learn")
        exit(1)
    
    trainer = ViTTrainer(
        image_size=224,
        batch_size=16,  # Smaller batch size for ViT due to memory requirements
        epochs=50
    )
    
    model, history = trainer.train_model()
    
    if model:
        print("Vision Transformer training completed successfully!")
        print(f"Model saved in: {trainer.output_dir}")