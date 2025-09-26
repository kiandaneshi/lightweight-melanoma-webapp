"""
Advanced data augmentation techniques for melanoma classification.
Implements CutMix and MixUp for improved model performance.
"""

import tensorflow as tf
import numpy as np
from config import Config


class AdvancedAugmentation:
    """Advanced augmentation techniques for medical image classification."""
    
    def __init__(self, cutmix_prob=0.5, mixup_prob=0.5, cutmix_alpha=1.0, mixup_alpha=0.2):
        """
        Initialize advanced augmentation parameters.
        
        Args:
            cutmix_prob: Probability of applying CutMix
            mixup_prob: Probability of applying MixUp  
            cutmix_alpha: Beta distribution parameter for CutMix
            mixup_alpha: Beta distribution parameter for MixUp
        """
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
    
    @tf.function
    def get_random_bbox(self, image_shape, lam):
        """
        Generate random bounding box for CutMix.
        
        Args:
            image_shape: Shape of the image [H, W]
            lam: Lambda parameter from beta distribution
            
        Returns:
            Bounding box coordinates [y1, x1, y2, x2]
        """
        H, W = image_shape[0], image_shape[1]
        
        # Calculate cut ratio
        cut_rat = tf.sqrt(1.0 - lam)
        cut_w = tf.cast(W * cut_rat, tf.int32)
        cut_h = tf.cast(H * cut_rat, tf.int32)
        
        # Uniform random position for the cut
        cx = tf.random.uniform([], 0, W, dtype=tf.int32)
        cy = tf.random.uniform([], 0, H, dtype=tf.int32)
        
        # Calculate bounding box
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, W)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, H)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, W)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, H)
        
        return y1, x1, y2, x2
    
    @tf.function
    def cutmix(self, images, labels):
        """
        Apply CutMix augmentation to a batch of images.
        
        Args:
            images: Batch of images [B, H, W, C]
            labels: Batch of labels [B]
            
        Returns:
            Augmented images and mixed labels
        """
        batch_size = tf.shape(images)[0]
        image_shape = tf.shape(images)[1:3]
        
        # Generate lambda from beta distribution
        lam = tf.random.uniform([])
        if self.cutmix_alpha > 0:
            lam = tf.py_function(
                lambda: np.random.beta(self.cutmix_alpha, self.cutmix_alpha),
                [],
                tf.float32
            )
        
        # Random permutation for mixing
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Get random bounding box
        y1, x1, y2, x2 = self.get_random_bbox(image_shape, lam)
        
        # Apply CutMix
        mixed_images = images
        
        # Create a mask for the region to be replaced
        height, width = image_shape[0], image_shape[1]
        y_indices = tf.range(height)
        x_indices = tf.range(width)
        
        mask_y = (y_indices >= y1) & (y_indices < y2)
        mask_x = (x_indices >= x1) & (x_indices < x2)
        
        # Expand dimensions to match image shape
        mask_y = tf.expand_dims(mask_y, axis=1)  # [H, 1]
        mask_x = tf.expand_dims(mask_x, axis=0)  # [1, W]
        mask = tf.cast(mask_y & mask_x, tf.float32)  # [H, W]
        mask = tf.expand_dims(mask, axis=-1)  # [H, W, 1]
        mask = tf.expand_dims(mask, axis=0)  # [1, H, W, 1]
        
        # Apply the mask
        mixed_images = images * (1 - mask) + shuffled_images * mask
        
        # Adjust lambda based on actual cut area
        cut_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
        total_area = tf.cast(height * width, tf.float32)
        lam_adjusted = 1.0 - cut_area / total_area
        
        # Mix labels
        mixed_labels = lam_adjusted * labels + (1 - lam_adjusted) * shuffled_labels
        
        return mixed_images, mixed_labels
    
    @tf.function
    def mixup(self, images, labels):
        """
        Apply MixUp augmentation to a batch of images.
        
        Args:
            images: Batch of images [B, H, W, C]
            labels: Batch of labels [B]
            
        Returns:
            Augmented images and mixed labels
        """
        batch_size = tf.shape(images)[0]
        
        # Generate lambda from beta distribution
        lam = tf.random.uniform([])
        if self.mixup_alpha > 0:
            lam = tf.py_function(
                lambda: np.random.beta(self.mixup_alpha, self.mixup_alpha),
                [],
                tf.float32
            )
        
        # Random permutation for mixing
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * shuffled_images
        mixed_labels = lam * labels + (1 - lam) * shuffled_labels
        
        return mixed_images, mixed_labels
    
    @tf.function
    def apply_augmentation(self, images, labels):
        """
        Apply advanced augmentation based on probabilities.
        
        Args:
            images: Batch of images [B, H, W, C]
            labels: Batch of labels [B]
            
        Returns:
            Augmented images and labels
        """
        # Convert labels to float for mixing
        labels = tf.cast(labels, tf.float32)
        
        # Randomly choose augmentation
        rand_val = tf.random.uniform([])
        
        # Apply CutMix
        if rand_val < self.cutmix_prob:
            return self.cutmix(images, labels)
        # Apply MixUp
        elif rand_val < self.cutmix_prob + self.mixup_prob:
            return self.mixup(images, labels)
        # No augmentation
        else:
            return images, labels


def create_advanced_augmentation_dataset(dataset, augmentation_config):
    """
    Apply advanced augmentation to a TensorFlow dataset.
    
    Args:
        dataset: TensorFlow dataset containing (images, labels)
        augmentation_config: Configuration dict with augmentation parameters
        
    Returns:
        Augmented dataset
    """
    augmenter = AdvancedAugmentation(
        cutmix_prob=augmentation_config.get('cutmix_prob', 0.5),
        mixup_prob=augmentation_config.get('mixup_prob', 0.5),
        cutmix_alpha=1.0,
        mixup_alpha=0.2
    )
    
    def apply_advanced_aug(images, labels):
        return augmenter.apply_augmentation(images, labels)
    
    # Apply augmentation to batched dataset
    dataset = dataset.map(
        apply_advanced_aug,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset


def create_focal_loss_with_mixed_labels():
    """
    Create focal loss function that works with mixed labels from CutMix/MixUp.
    
    Returns:
        Focal loss function compatible with label mixing
    """
    def focal_loss_mixed(alpha=0.25, gamma=2.0):
        def focal_loss_fn(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Handle mixed labels (float values between 0 and 1)
            alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
            
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_fn
    
    return focal_loss_mixed


# Configuration from config.py integration
def get_augmentation_config():
    """Get augmentation configuration from main config."""
    aug_config = Config.AUGMENTATION_CONFIG['train']
    return {
        'cutmix_prob': aug_config.get('cutmix_prob', 0.5),
        'mixup_prob': aug_config.get('mixup_prob', 0.5),
    }