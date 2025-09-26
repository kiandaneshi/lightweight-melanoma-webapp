"""
Grad-CAM implementation for generating attention heatmaps.
Provides explainability for melanoma classification predictions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import pandas as pd
from config import Config
from preprocess_data import DataPreprocessor

class GradCAMGenerator:
    def __init__(self, model_path, layer_name=None):
        """
        Initialize Grad-CAM generator.
        
        Args:
            model_path: Path to trained model
            layer_name: Name of layer to generate CAM from (auto-detect if None)
        """
        self.model = keras.models.load_model(model_path)
        self.layer_name = layer_name or self.find_target_layer()
        self.grad_model = self.create_grad_model()
        self.preprocessor = DataPreprocessor()

    def find_target_layer(self):
        """
        Automatically find the best layer for Grad-CAM.
        Usually the last convolutional layer.
        """
        # Look for common conv layer patterns
        layer_patterns = [
            'conv_pw_13',  # MobileNetV3
            'top_conv',    # EfficientNet
            'conv5_block3_out',  # ResNet
            'mixed10',     # Inception
        ]
        
        layer_names = [layer.name for layer in self.model.layers]
        
        # Try specific patterns first
        for pattern in layer_patterns:
            for layer_name in layer_names:
                if pattern in layer_name:
                    print(f"Found target layer: {layer_name}")
                    return layer_name
        
        # Fall back to last conv layer
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                print(f"Using last conv layer: {layer.name}")
                return layer.name
        
        # Ultimate fallback
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:  # 4D tensor (batch, h, w, channels)
                print(f"Using fallback layer: {layer.name}")
                return layer.name
        
        raise ValueError("Could not find suitable layer for Grad-CAM")

    def create_grad_model(self):
        """Create model that outputs both predictions and target layer activations."""
        try:
            target_layer = self.model.get_layer(self.layer_name)
        except ValueError:
            print(f"Layer {self.layer_name} not found. Available layers:")
            for layer in self.model.layers:
                print(f"  - {layer.name}")
            raise
        
        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[self.model.output, target_layer.output]
        )
        
        return grad_model

    def generate_gradcam(self, image, class_index=None):
        """
        Generate Grad-CAM heatmap for given image.
        
        Args:
            image: Preprocessed input image (224x224x3)
            class_index: Target class index (0 for binary classification)
            
        Returns:
            heatmap: Grad-CAM heatmap as numpy array
            prediction: Model prediction
        """
        # Add batch dimension
        img_array = np.expand_dims(image, axis=0)
        
        # Get predictions and activations with gradients
        with tf.GradientTape() as tape:
            predictions, activations = self.grad_model(img_array)
            
            if class_index is None:
                # For binary classification, use the single output
                class_channel = predictions[:, 0] if len(predictions.shape) > 1 else predictions
            else:
                class_channel = predictions[:, class_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, activations)
        
        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight activations by importance
        activations = activations[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, activations), axis=-1)
        
        # ReLU to keep only positive influences
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize heatmap
        if tf.reduce_max(heatmap) > 0:
            heatmap /= tf.reduce_max(heatmap)
        
        return heatmap.numpy(), predictions.numpy()[0]

    def overlay_heatmap(self, original_image, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            original_image: Original image (HxWx3)
            heatmap: Grad-CAM heatmap
            alpha: Overlay transparency
            colormap: OpenCV colormap for heatmap
            
        Returns:
            Overlayed image
        """
        # Resize heatmap to match image dimensions
        if original_image.shape[:2] != heatmap.shape:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to 8-bit
        heatmap_uint8 = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is in correct format
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Overlay
        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed

    def generate_gradcam_for_image_path(self, image_path, save_path=None):
        """
        Generate Grad-CAM for image file.
        
        Args:
            image_path: Path to image file
            save_path: Path to save visualization (optional)
            
        Returns:
            Dictionary with heatmap, prediction, and visualization
        """
        # Load and preprocess image
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        preprocessed = self.preprocessor.load_and_preprocess_image(
            image_path, 
            transforms=self.preprocessor.val_transforms
        )
        
        if preprocessed is None:
            raise ValueError(f"Could not preprocess image: {image_path}")
        
        # Generate Grad-CAM
        heatmap, prediction = self.generate_gradcam(preprocessed)
        
        # Create overlay
        overlay = self.overlay_heatmap(original_image, heatmap)
        
        # Create visualization
        fig = plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='hot', alpha=0.8)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title(f'Overlay (Prediction: {prediction[0]:.3f})')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to: {save_path}")
        
        plt.close()
        
        return {
            'heatmap': heatmap,
            'prediction': prediction,
            'overlay': overlay,
            'original': original_image
        }

    def generate_gradcam_batch(self, image_paths, output_dir, max_images=50):
        """
        Generate Grad-CAM visualizations for batch of images.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save visualizations
            max_images: Maximum number of images to process
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Limit number of images
        if len(image_paths) > max_images:
            image_paths = image_paths[:max_images]
            print(f"Processing first {max_images} images")
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
                
                # Generate filename
                image_name = Path(image_path).stem
                save_path = output_dir / f"gradcam_{image_name}.png"
                
                # Generate Grad-CAM
                result = self.generate_gradcam_for_image_path(image_path, save_path)
                
                results.append({
                    'image_path': str(image_path),
                    'prediction': float(result['prediction'][0]),
                    'gradcam_path': str(save_path)
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Save results summary
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "gradcam_results.csv", index=False)
        
        print(f"Batch Grad-CAM generation completed!")
        print(f"Processed: {len(results)} images")
        print(f"Results saved to: {output_dir}")
        
        return results_df

class GradCAMAnalyzer:
    """Analyze Grad-CAM results to understand model behavior."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()

    def analyze_attention_patterns(self, gradcam_results_csv, metadata_csv):
        """
        Analyze attention patterns across different cases.
        
        Args:
            gradcam_results_csv: CSV with Grad-CAM results
            metadata_csv: CSV with image metadata and labels
        """
        gradcam_df = pd.read_csv(gradcam_results_csv)
        metadata_df = pd.read_csv(metadata_csv)
        
        # Merge datasets
        gradcam_df['image_name'] = gradcam_df['image_path'].apply(
            lambda x: Path(x).stem
        )
        
        merged_df = gradcam_df.merge(
            metadata_df, 
            on='image_name', 
            how='inner'
        )
        
        print(f"Analyzing {len(merged_df)} images with Grad-CAM results")
        
        # Analysis by true label
        benign_df = merged_df[merged_df['target'] == 0]
        melanoma_df = merged_df[merged_df['target'] == 1]
        
        print(f"\nPrediction accuracy:")
        print(f"Benign cases: {len(benign_df)} images")
        print(f"  - Correct (pred < 0.5): {(benign_df['prediction'] < 0.5).sum()}")
        print(f"  - Incorrect (pred >= 0.5): {(benign_df['prediction'] >= 0.5).sum()}")
        
        print(f"Melanoma cases: {len(melanoma_df)} images")
        print(f"  - Correct (pred >= 0.5): {(melanoma_df['prediction'] >= 0.5).sum()}")
        print(f"  - Incorrect (pred < 0.5): {(melanoma_df['prediction'] < 0.5).sum()}")
        
        # Identify interesting cases for further analysis
        false_positives = benign_df[benign_df['prediction'] >= 0.5]
        false_negatives = melanoma_df[melanoma_df['prediction'] < 0.5]
        high_confidence_correct = merged_df[
            ((merged_df['target'] == 0) & (merged_df['prediction'] < 0.1)) |
            ((merged_df['target'] == 1) & (merged_df['prediction'] > 0.9))
        ]
        
        print(f"\nInteresting cases for review:")
        print(f"False positives: {len(false_positives)}")
        print(f"False negatives: {len(false_negatives)}")
        print(f"High confidence correct: {len(high_confidence_correct)}")
        
        return {
            'merged_df': merged_df,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'high_confidence': high_confidence_correct
        }

def main():
    """Main Grad-CAM generation function."""
    print("DermAI Grad-CAM Generation")
    print("=" * 50)
    
    # Configuration
    models_dir = Path(Config.MODELS_DIR)
    data_dir = Path(Config.DATA_DIR)
    results_dir = Path(Config.RESULTS_DIR)
    
    # Find trained mobile model (primary target for Grad-CAM)
    mobile_models = list(models_dir.glob("mobile_*/best_mobile_melanoma.h5"))
    
    if not mobile_models:
        print("No trained mobile models found. Please run training first.")
        return
    
    model_path = mobile_models[0]  # Use the first (most recent) model
    print(f"Using model: {model_path}")
    
    # Initialize Grad-CAM generator
    try:
        gradcam = GradCAMGenerator(str(model_path))
        print(f"Grad-CAM initialized with layer: {gradcam.layer_name}")
    except Exception as e:
        print(f"Error initializing Grad-CAM: {e}")
        return
    
    # Load test data
    splits_dir = data_dir / "processed" / "splits"
    test_df = pd.read_csv(splits_dir / "test.csv")
    
    print(f"Loaded {len(test_df)} test images")
    
    # Select interesting cases for visualization
    # High confidence melanoma cases
    melanoma_cases = test_df[test_df['target'] == 1].sample(n=min(10, test_df['target'].sum()), random_state=42)
    # High confidence benign cases  
    benign_cases = test_df[test_df['target'] == 0].sample(n=min(10, (test_df['target'] == 0).sum()), random_state=42)
    
    selected_cases = pd.concat([melanoma_cases, benign_cases])
    
    # Get full image paths
    preprocessor = DataPreprocessor()
    image_paths = []
    
    for _, row in selected_cases.iterrows():
        image_path = preprocessor.find_image_path(row['image_name'])
        if image_path:
            image_paths.append(str(image_path))
    
    print(f"Found {len(image_paths)} images for Grad-CAM generation")
    
    # Generate Grad-CAM visualizations
    gradcam_dir = results_dir / "gradcam_analysis"
    gradcam_dir.mkdir(exist_ok=True)
    
    try:
        results_df = gradcam.generate_gradcam_batch(
            image_paths, 
            gradcam_dir,
            max_images=20
        )
        
        print(f"✓ Grad-CAM generation completed!")
        
        # Analyze results
        analyzer = GradCAMAnalyzer()
        analysis = analyzer.analyze_attention_patterns(
            gradcam_dir / "gradcam_results.csv",
            splits_dir / "test.csv"
        )
        
        print(f"✓ Grad-CAM analysis completed!")
        print(f"Results available in: {gradcam_dir}")
        
    except Exception as e:
        print(f"Error during Grad-CAM generation: {e}")
        raise

if __name__ == "__main__":
    main()
