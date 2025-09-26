"""
Test-Time Augmentation (TTA) for improved inference accuracy.
Applies multiple augmentations during inference and averages predictions.
"""

import tensorflow as tf
import numpy as np
import albumentations as A
from pathlib import Path
import cv2
from typing import List, Tuple, Optional, Union
from config import Config


class TestTimeAugmentation:
    """
    Test-Time Augmentation class for melanoma classification.
    
    Applies multiple augmentations to input images during inference
    and averages the predictions for improved accuracy and robustness.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224), 
                 tta_steps: int = 8):
        """
        Initialize TTA with augmentation strategies.
        
        Args:
            input_size: Target image size (height, width)
            tta_steps: Number of augmented versions to generate
        """
        self.input_size = input_size
        self.tta_steps = tta_steps
        
        # Define TTA augmentation strategies
        self.tta_transforms = self._create_tta_transforms()
        
        # Base transform for non-augmented version
        self.base_transform = A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_tta_transforms(self) -> List[A.Compose]:
        """
        Create list of TTA augmentation transforms.
        
        Returns:
            List of albumentations transforms for TTA
        """
        transforms = []
        
        # Original image (no augmentation)
        transforms.append(A.Compose([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # Horizontal flip
        transforms.append(A.Compose([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # Vertical flip
        transforms.append(A.Compose([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # Both flips
        transforms.append(A.Compose([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # 90-degree rotations
        for angle in [90, 180, 270]:
            transforms.append(A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.Rotate(limit=(angle, angle), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
        
        # Slight rotations with flips
        transforms.append(A.Compose([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Rotate(limit=(15, 15), p=1.0),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # Ensure we have the requested number of transforms
        # Repeat some transforms if needed
        while len(transforms) < self.tta_steps:
            transforms.append(transforms[len(transforms) % 4])
        
        return transforms[:self.tta_steps]
    
    def augment_image(self, image: np.ndarray, transform_idx: int) -> np.ndarray:
        """
        Apply specific augmentation to an image.
        
        Args:
            image: Input image as numpy array
            transform_idx: Index of transform to apply
            
        Returns:
            Augmented image
        """
        if transform_idx >= len(self.tta_transforms):
            transform_idx = transform_idx % len(self.tta_transforms)
        
        transform = self.tta_transforms[transform_idx]
        augmented = transform(image=image)
        return augmented['image']
    
    def predict_with_tta(self, model: tf.keras.Model, 
                        image: Union[np.ndarray, str, Path],
                        batch_size: int = 8) -> Tuple[float, np.ndarray]:
        """
        Make prediction using test-time augmentation.
        
        Args:
            model: Trained Keras model
            image: Input image (numpy array or path to image)
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (averaged_prediction, all_predictions)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        # Generate all augmented versions
        augmented_images = []
        for i in range(self.tta_steps):
            aug_image = self.augment_image(image, i)
            augmented_images.append(aug_image)
        
        # Convert to batch
        batch = np.array(augmented_images)
        
        # Make predictions in batches
        all_predictions = []
        for i in range(0, len(batch), batch_size):
            batch_slice = batch[i:i + batch_size]
            preds = model.predict(batch_slice, verbose=0)
            all_predictions.append(preds)
        
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Average predictions
        avg_prediction = np.mean(all_predictions, axis=0)
        
        # Return scalar prediction for binary classification
        if avg_prediction.shape == (1,):
            return float(avg_prediction[0]), all_predictions.flatten()
        else:
            return avg_prediction, all_predictions
    
    def predict_batch_with_tta(self, model: tf.keras.Model,
                              images: List[Union[np.ndarray, str, Path]],
                              batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for a batch of images using TTA.
        
        Args:
            model: Trained Keras model
            images: List of images (numpy arrays or paths)
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (averaged_predictions, all_predictions)
        """
        avg_predictions = []
        all_predictions_list = []
        
        for image in images:
            avg_pred, all_preds = self.predict_with_tta(model, image, batch_size)
            avg_predictions.append(avg_pred)
            all_predictions_list.append(all_preds)
        
        return np.array(avg_predictions), np.array(all_predictions_list)
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array
        """
        image_path = str(image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def evaluate_with_tta(self, model: tf.keras.Model,
                         test_df: 'pd.DataFrame',
                         image_dir: Union[str, Path],
                         batch_size: int = 8) -> dict:
        """
        Evaluate model performance using TTA on test dataset.
        
        Args:
            model: Trained Keras model
            test_df: Test dataframe with image_name and target columns
            image_dir: Directory containing test images
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
        
        image_dir = Path(image_dir)
        predictions = []
        targets = []
        
        print(f"Evaluating {len(test_df)} images with TTA...")
        
        for idx, row in test_df.iterrows():
            image_name = row['image_name']
            target = row['target']
            
            # Find image file
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_path = image_dir / f"{image_name}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if image_path is None:
                print(f"Warning: Could not find image {image_name}")
                continue
            
            # Make TTA prediction
            try:
                pred, _ = self.predict_with_tta(model, image_path, batch_size)
                predictions.append(pred)
                targets.append(target)
                
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(test_df)} images")
                    
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        binary_predictions = (predictions > 0.5).astype(int)
        
        metrics = {
            'auc': roc_auc_score(targets, predictions),
            'accuracy': accuracy_score(targets, binary_predictions),
            'precision': precision_score(targets, binary_predictions),
            'recall': recall_score(targets, binary_predictions),
            'num_samples': len(predictions)
        }
        
        return metrics


class TTAEnsemble:
    """
    Ensemble multiple models with test-time augmentation.
    Combines the power of model ensembling with TTA.
    """
    
    def __init__(self, model_paths: List[Union[str, Path]], 
                 input_size: Tuple[int, int] = (224, 224),
                 tta_steps: int = 8):
        """
        Initialize TTA ensemble.
        
        Args:
            model_paths: List of paths to trained models
            input_size: Target image size
            tta_steps: Number of TTA steps per model
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.input_size = input_size
        self.tta_steps = tta_steps
        
        # Load models
        self.models = []
        for model_path in self.model_paths:
            try:
                model = tf.keras.models.load_model(str(model_path))
                self.models.append(model)
                print(f"Loaded model: {model_path.name}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
        
        if len(self.models) == 0:
            raise ValueError("No models could be loaded")
        
        # Initialize TTA for each model
        self.tta_augmenters = [
            TestTimeAugmentation(input_size, tta_steps) 
            for _ in self.models
        ]
    
    def predict_ensemble_tta(self, image: Union[np.ndarray, str, Path],
                           batch_size: int = 8) -> Tuple[float, np.ndarray]:
        """
        Make ensemble prediction with TTA.
        
        Args:
            image: Input image
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (ensemble_prediction, model_predictions)
        """
        model_predictions = []
        
        for model, tta_augmenter in zip(self.models, self.tta_augmenters):
            pred, _ = tta_augmenter.predict_with_tta(model, image, batch_size)
            model_predictions.append(pred)
        
        # Average across models
        ensemble_pred = np.mean(model_predictions)
        
        return float(ensemble_pred), np.array(model_predictions)


def create_tta_predictor(model_path: Union[str, Path], 
                        input_size: Tuple[int, int] = (224, 224),
                        tta_steps: int = 8) -> TestTimeAugmentation:
    """
    Factory function to create TTA predictor.
    
    Args:
        model_path: Path to trained model
        input_size: Input image size
        tta_steps: Number of TTA steps
        
    Returns:
        TestTimeAugmentation instance
    """
    return TestTimeAugmentation(input_size, tta_steps)


def compare_tta_vs_single(model: tf.keras.Model,
                         test_images: List[Union[np.ndarray, str, Path]],
                         test_targets: List[int],
                         input_size: Tuple[int, int] = (224, 224),
                         tta_steps: int = 8) -> dict:
    """
    Compare TTA predictions vs single image predictions.
    
    Args:
        model: Trained model
        test_images: List of test images
        test_targets: List of test targets
        input_size: Input image size
        tta_steps: Number of TTA steps
        
    Returns:
        Comparison metrics
    """
    from sklearn.metrics import roc_auc_score
    
    tta = TestTimeAugmentation(input_size, tta_steps)
    
    # Single predictions
    single_preds = []
    for image in test_images:
        if isinstance(image, (str, Path)):
            image = tta._load_image(image)
        
        # Apply base transform
        aug_image = tta.base_transform(image=image)['image']
        aug_image = np.expand_dims(aug_image, axis=0)
        
        pred = model.predict(aug_image, verbose=0)[0]
        single_preds.append(float(pred[0]) if pred.shape == (1,) else float(pred))
    
    # TTA predictions
    tta_preds, _ = tta.predict_batch_with_tta(model, test_images)
    
    # Calculate metrics
    single_auc = roc_auc_score(test_targets, single_preds)
    tta_auc = roc_auc_score(test_targets, tta_preds)
    
    return {
        'single_auc': single_auc,
        'tta_auc': tta_auc,
        'improvement': tta_auc - single_auc,
        'relative_improvement': (tta_auc - single_auc) / single_auc * 100
    }