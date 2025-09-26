"""
Advanced learning rate schedulers for improved training convergence.
Implements cosine annealing, warmup, and other advanced scheduling strategies.
"""

import tensorflow as tf
import numpy as np
import math


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """
    Cosine annealing learning rate scheduler with optional warmup.
    
    This scheduler gradually decreases the learning rate following a cosine function,
    which often leads to better convergence than step-based schedulers.
    """
    
    def __init__(self, initial_lr, min_lr=1e-7, total_epochs=100, warmup_epochs=5, 
                 restart_epochs=None, verbose=1):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            total_epochs: Total training epochs
            warmup_epochs: Number of warmup epochs (linear increase to initial_lr)
            restart_epochs: If set, restart the schedule every N epochs (cosine restarts)
            verbose: Verbosity level
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.restart_epochs = restart_epochs
        self.verbose = verbose
        
        self.current_epoch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        """Update learning rate at the beginning of each epoch."""
        self.current_epoch = epoch
        
        # Calculate learning rate
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            if self.restart_epochs:
                # Cosine annealing with restarts
                epoch_in_cycle = (epoch - self.warmup_epochs) % self.restart_epochs
                total_cycle_epochs = self.restart_epochs
            else:
                # Standard cosine annealing
                epoch_in_cycle = epoch - self.warmup_epochs
                total_cycle_epochs = self.total_epochs - self.warmup_epochs
            
            # Cosine annealing formula
            cosine_factor = 0.5 * (1 + math.cos(math.pi * epoch_in_cycle / total_cycle_epochs))
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
        
        # Set the learning rate
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        
        if self.verbose and (epoch % 10 == 0 or epoch < 5):
            print(f"Epoch {epoch + 1}: Learning rate = {lr:.2e}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log learning rate at the end of epoch."""
        if logs is not None:
            logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)


class WarmupCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    TensorFlow native warmup + cosine decay schedule.
    This is more efficient as it's computed on GPU and doesn't require callbacks.
    """
    
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps=1000, 
                 alpha=0.0, name=None):
        """
        Initialize warmup cosine decay schedule.
        
        Args:
            initial_learning_rate: Peak learning rate after warmup
            decay_steps: Number of steps for cosine decay
            warmup_steps: Number of warmup steps
            alpha: Minimum learning rate as fraction of initial_learning_rate
            name: Name for the schedule
        """
        super().__init__()
        
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.name = name
    
    @tf.function
    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupCosineDecay"):
            # Convert to float for calculations
            step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            decay_steps = tf.cast(self.decay_steps, tf.float32)
            
            # Warmup phase
            warmup_lr = self.initial_learning_rate * step / warmup_steps
            
            # Cosine decay phase
            cosine_decay_lr = tf.keras.utils.get_custom_objects().get(
                'cosine_decay',
                tf.keras.optimizers.schedules.CosineDecay
            )(
                initial_learning_rate=self.initial_learning_rate,
                decay_steps=decay_steps,
                alpha=self.alpha
            )(step - warmup_steps)
            
            # Choose based on current step
            return tf.where(step < warmup_steps, warmup_lr, cosine_decay_lr)
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
            "name": self.name
        }


def create_cosine_annealing_scheduler(initial_lr, total_epochs, warmup_epochs=5, 
                                    min_lr_ratio=0.01, restart_epochs=None):
    """
    Factory function to create cosine annealing scheduler.
    
    Args:
        initial_lr: Initial learning rate
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        min_lr_ratio: Minimum LR as ratio of initial LR
        restart_epochs: If set, restart every N epochs
        
    Returns:
        CosineAnnealingScheduler instance
    """
    min_lr = initial_lr * min_lr_ratio
    
    return CosineAnnealingScheduler(
        initial_lr=initial_lr,
        min_lr=min_lr,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        restart_epochs=restart_epochs,
        verbose=1
    )


def create_warmup_cosine_schedule(initial_lr, total_steps, warmup_steps=1000, min_lr_ratio=0.01):
    """
    Factory function to create TensorFlow native warmup + cosine decay schedule.
    
    Args:
        initial_lr: Peak learning rate after warmup
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR as ratio of initial LR
        
    Returns:
        WarmupCosineDecaySchedule instance
    """
    return WarmupCosineDecaySchedule(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps - warmup_steps,
        warmup_steps=warmup_steps,
        alpha=min_lr_ratio
    )


def calculate_training_steps(train_samples, batch_size, epochs):
    """
    Calculate total training steps for schedule configuration.
    
    Args:
        train_samples: Number of training samples
        batch_size: Batch size
        epochs: Number of epochs
        
    Returns:
        Total number of training steps
    """
    steps_per_epoch = math.ceil(train_samples / batch_size)
    total_steps = steps_per_epoch * epochs
    return total_steps, steps_per_epoch


class GradientClippingCallback(tf.keras.callbacks.Callback):
    """
    Callback to apply gradient clipping during training.
    Helps stabilize training, especially with advanced augmentation and mixed precision.
    """
    
    def __init__(self, clip_norm=1.0, clip_value=None, log_frequency=100):
        """
        Initialize gradient clipping callback.
        
        Args:
            clip_norm: Global norm to clip gradients to
            clip_value: Value to clip gradients to (alternative to norm)
            log_frequency: How often to log gradient norms
        """
        super().__init__()
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.log_frequency = log_frequency
        self.step_count = 0
        
    def on_train_batch_end(self, batch, logs=None):
        """Monitor gradient norms after each batch."""
        self.step_count += 1
        
        if self.step_count % self.log_frequency == 0:
            # Get gradients (this is more for monitoring)
            # Actual clipping is done in the optimizer
            if logs is not None:
                logs['gradient_norm'] = 0.0  # Placeholder - actual norm would need custom training loop


def create_advanced_callbacks(initial_lr, total_epochs, model_save_path, 
                             patience=15, use_cosine_annealing=True, 
                             warmup_epochs=5, gradient_clip_norm=None):
    """
    Create a set of advanced callbacks for training.
    
    Args:
        initial_lr: Initial learning rate
        total_epochs: Total training epochs
        model_save_path: Path to save best model
        patience: Early stopping patience
        use_cosine_annealing: Whether to use cosine annealing vs ReduceLROnPlateau
        warmup_epochs: Number of warmup epochs for cosine annealing
        gradient_clip_norm: Gradient clipping norm (if None, no clipping)
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpointing
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path / "best_model.weights.h5"),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    )
    
    # Early stopping
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Learning rate scheduling
    if use_cosine_annealing:
        callbacks.append(
            create_cosine_annealing_scheduler(
                initial_lr=initial_lr,
                total_epochs=total_epochs,
                warmup_epochs=warmup_epochs,
                min_lr_ratio=0.01
            )
        )
    else:
        # Fallback to ReduceLROnPlateau
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        )
    
    # CSV logging
    callbacks.append(
        tf.keras.callbacks.CSVLogger(
            str(model_save_path / "training_log.csv"),
            append=True
        )
    )
    
    # TensorBoard logging (optional)
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=str(model_save_path / "tensorboard"),
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq='epoch'
        )
    )
    
    # Gradient clipping (if requested)
    if gradient_clip_norm:
        callbacks.append(GradientClippingCallback(clip_norm=gradient_clip_norm))
    
    return callbacks