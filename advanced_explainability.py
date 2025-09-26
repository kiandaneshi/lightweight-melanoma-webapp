"""
Advanced explainability features using LIME and SHAP for melanoma classification.
Provides multiple explanation methods beyond Grad-CAM for comprehensive model interpretability.
"""

import os
import json
from pathlib import Path
from datetime import datetime

try:
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    TF_AVAILABLE = True
except ImportError as e:
    print(f"ML dependencies not available: {e}")
    TF_AVAILABLE = False

try:
    import lime
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    LIME_AVAILABLE = True
except ImportError:
    print("LIME not available. Install with: pip install lime")
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

from config import Config

class AdvancedExplainer:
    """Advanced explainability using LIME and SHAP for melanoma classification."""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.explanations = {}
        
        # Results directory
        self.results_dir = Path(Config.MODELS_DIR) / "explainability"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self, model_path=None):
        """Load trained melanoma classification model."""
        if model_path:
            self.model_path = model_path
            
        if not self.model_path:
            # Find the most recent best model
            models_dir = Path(Config.MODELS_DIR)
            model_files = list(models_dir.glob("**/best_*_melanoma.h5"))
            
            if not model_files:
                raise FileNotFoundError("No trained models found. Train a model first.")
            
            # Sort by modification time and get the most recent
            self.model_path = sorted(model_files, key=lambda x: x.stat().st_mtime)[-1]
        
        print(f"Loading model for explainability: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        return self.model

    def create_prediction_function(self):
        """Create prediction function compatible with explainability tools."""
        
        def predict_fn(images):
            """Prediction function for explainability tools."""
            # Ensure images are in correct format
            if images.ndim == 3:
                images = np.expand_dims(images, axis=0)
            
            # Normalize images to [0,1] if needed
            if images.max() > 1.0:
                images = images.astype(np.float32) / 255.0
            
            # Get predictions
            predictions = self.model.predict(images, verbose=0)
            
            # Handle different output formats
            if predictions.shape[1] == 1:
                # Binary classification - return both classes
                pred_proba = predictions.flatten()
                return np.column_stack([1 - pred_proba, pred_proba])
            else:
                # Multi-class classification
                return predictions
        
        return predict_fn

    def lime_explanation(self, image, num_samples=1000, top_labels=2):
        """Generate LIME explanation for an image."""
        
        if not LIME_AVAILABLE:
            print("LIME not available. Install with: pip install lime")
            return None
        
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Generating LIME explanation...")
        
        # Ensure image is uint8 format for LIME
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Initialize LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Create prediction function
        predict_fn = self.create_prediction_function()
        
        # Generate explanation
        explanation = explainer.explain_instance(
            image,
            predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
        )
        
        return explanation

    def shap_explanation(self, images, background_samples=100):
        """Generate SHAP explanation for images using DeepExplainer."""
        
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Generating SHAP explanation...")
        
        # Prepare background dataset for SHAP
        # In practice, use representative samples from training set
        background = np.random.random((background_samples, 224, 224, 3)).astype(np.float32)
        
        # Ensure images are in correct format
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        
        if images.max() > 1.0:
            images = images.astype(np.float32) / 255.0
        
        try:
            # Initialize SHAP DeepExplainer
            explainer = shap.DeepExplainer(self.model, background)
            
            # Generate SHAP values
            shap_values = explainer.shap_values(images)
            
            return shap_values, explainer
            
        except Exception as e:
            print(f"SHAP DeepExplainer failed: {e}")
            print("Trying GradientExplainer as fallback...")
            
            try:
                # Fallback to GradientExplainer
                explainer = shap.GradientExplainer(self.model, background)
                shap_values = explainer.shap_values(images)
                return shap_values, explainer
                
            except Exception as e2:
                print(f"SHAP GradientExplainer also failed: {e2}")
                return None, None

    def integrated_gradients_explanation(self, image, steps=50):
        """Generate Integrated Gradients explanation."""
        
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Generating Integrated Gradients explanation...")
        
        # Ensure correct format
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Create baseline (black image)
        baseline = np.zeros_like(image)
        
        # Generate path from baseline to image
        alphas = np.linspace(0, 1, steps + 1)
        
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and image
            interpolated_image = baseline + alpha * (image - baseline)
            
            # Calculate gradients
            with tf.GradientTape() as tape:
                tape.watch(interpolated_image)
                predictions = self.model(interpolated_image)
                
                # For binary classification, take the positive class
                if predictions.shape[1] == 1:
                    target_output = predictions[:, 0]
                else:
                    target_output = predictions[:, 1]  # Assume melanoma class is index 1
            
            # Get gradients
            grads = tape.gradient(target_output, interpolated_image)
            gradients.append(grads.numpy())
        
        # Calculate integrated gradients
        gradients = np.array(gradients)
        integrated_grads = np.mean(gradients, axis=0) * (image - baseline)
        
        return integrated_grads[0]  # Remove batch dimension

    def explain_image(self, image_path, save_results=True):
        """Generate comprehensive explanations for a single image."""
        
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load and preprocess image
        if isinstance(image_path, str):
            image = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(224, 224)
            )
            image = tf.keras.preprocessing.image.img_to_array(image)
        else:
            image = image_path  # Assume it's already an array
        
        # Get model prediction
        pred_image = image.copy()
        if pred_image.max() > 1.0:
            pred_image = pred_image / 255.0
        
        prediction = self.model.predict(np.expand_dims(pred_image, axis=0), verbose=0)
        
        if prediction.shape[1] == 1:
            pred_prob = prediction[0][0]
            pred_class = "Melanoma" if pred_prob > 0.5 else "Non-melanoma"
        else:
            pred_class_idx = np.argmax(prediction[0])
            pred_prob = prediction[0][pred_class_idx]
            pred_class = "Melanoma" if pred_class_idx == 1 else "Non-melanoma"
        
        print(f"Model prediction: {pred_class} ({pred_prob:.3f})")
        
        explanations = {
            "prediction": {
                "class": pred_class,
                "probability": float(pred_prob),
                "raw_output": prediction.tolist()
            }
        }
        
        # LIME explanation
        if LIME_AVAILABLE:
            try:
                lime_exp = self.lime_explanation(image.astype(np.uint8))
                explanations["lime"] = {
                    "available": True,
                    "explanation": lime_exp
                }
                print("✓ LIME explanation generated")
            except Exception as e:
                print(f"✗ LIME explanation failed: {e}")
                explanations["lime"] = {"available": False, "error": str(e)}
        else:
            explanations["lime"] = {"available": False, "error": "LIME not installed"}
        
        # SHAP explanation
        if SHAP_AVAILABLE:
            try:
                shap_values, shap_explainer = self.shap_explanation(pred_image)
                explanations["shap"] = {
                    "available": True,
                    "shap_values": shap_values,
                    "explainer": shap_explainer
                }
                print("✓ SHAP explanation generated")
            except Exception as e:
                print(f"✗ SHAP explanation failed: {e}")
                explanations["shap"] = {"available": False, "error": str(e)}
        else:
            explanations["shap"] = {"available": False, "error": "SHAP not installed"}
        
        # Integrated Gradients explanation
        try:
            ig_attribution = self.integrated_gradients_explanation(pred_image)
            explanations["integrated_gradients"] = {
                "available": True,
                "attribution": ig_attribution
            }
            print("✓ Integrated Gradients explanation generated")
        except Exception as e:
            print(f"✗ Integrated Gradients explanation failed: {e}")
            explanations["integrated_gradients"] = {"available": False, "error": str(e)}
        
        # Save results
        if save_results:
            self._save_explanations(explanations, image, image_path)
        
        return explanations

    def _save_explanations(self, explanations, image, image_path):
        """Save explanation results and visualizations."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create image-specific directory
        if isinstance(image_path, str):
            image_name = Path(image_path).stem
        else:
            image_name = f"image_{timestamp}"
        
        exp_dir = self.results_dir / f"{image_name}_{timestamp}"
        exp_dir.mkdir(exist_ok=True)
        
        # Save original image
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image_to_save = (image * 255).astype(np.uint8)
            else:
                image_to_save = image.astype(np.uint8)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(image_to_save)
            plt.title(f"Original Image\\nPrediction: {explanations['prediction']['class']} "
                     f"({explanations['prediction']['probability']:.3f})")
            plt.axis('off')
            plt.savefig(exp_dir / "original_image.png", bbox_inches='tight', dpi=150)
            plt.close()
        
        # Save LIME visualization
        if explanations["lime"]["available"] and LIME_AVAILABLE:
            try:
                lime_exp = explanations["lime"]["explanation"]
                
                # Get explanation for the predicted class
                pred_class_idx = 1 if explanations['prediction']['class'] == 'Melanoma' else 0
                
                # Positive and negative contributions
                temp, mask = lime_exp.get_image_and_mask(
                    pred_class_idx, positive_only=True, num_features=10, hide_rest=False
                )
                
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(image.astype(np.uint8))
                plt.title("Original Image")
                plt.axis('off')
                
                # Positive contributions
                plt.subplot(1, 3, 2)
                plt.imshow(temp)
                plt.title("LIME: Positive Evidence")
                plt.axis('off')
                
                # Negative contributions
                temp, mask = lime_exp.get_image_and_mask(
                    pred_class_idx, positive_only=False, num_features=10, hide_rest=False
                )
                plt.subplot(1, 3, 3)
                plt.imshow(temp)
                plt.title("LIME: All Evidence")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(exp_dir / "lime_explanation.png", bbox_inches='tight', dpi=150)
                plt.close()
                
            except Exception as e:
                print(f"Error saving LIME visualization: {e}")
        
        # Save SHAP visualization
        if explanations["shap"]["available"] and SHAP_AVAILABLE:
            try:
                shap_values = explanations["shap"]["shap_values"]
                
                if shap_values is not None:
                    # For binary classification, use the positive class
                    if len(shap_values) == 2:
                        shap_vals = shap_values[1][0]  # Melanoma class
                    else:
                        shap_vals = shap_values[0][0]  # Single output
                    
                    plt.figure(figsize=(12, 4))
                    
                    # Original image
                    plt.subplot(1, 3, 1)
                    plt.imshow(image.astype(np.uint8))
                    plt.title("Original Image")
                    plt.axis('off')
                    
                    # SHAP values (summed across channels)
                    plt.subplot(1, 3, 2)
                    shap_sum = np.sum(shap_vals, axis=-1)
                    plt.imshow(shap_sum, cmap='RdBu_r')
                    plt.title("SHAP Attribution")
                    plt.colorbar()
                    plt.axis('off')
                    
                    # Overlay on original
                    plt.subplot(1, 3, 3)
                    plt.imshow(image.astype(np.uint8), alpha=0.7)
                    plt.imshow(shap_sum, cmap='RdBu_r', alpha=0.3)
                    plt.title("SHAP Overlay")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(exp_dir / "shap_explanation.png", bbox_inches='tight', dpi=150)
                    plt.close()
                    
            except Exception as e:
                print(f"Error saving SHAP visualization: {e}")
        
        # Save Integrated Gradients visualization
        if explanations["integrated_gradients"]["available"]:
            try:
                ig_attribution = explanations["integrated_gradients"]["attribution"]
                
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(image.astype(np.uint8))
                plt.title("Original Image")
                plt.axis('off')
                
                # Attribution magnitude
                plt.subplot(1, 3, 2)
                attribution_magnitude = np.sum(np.abs(ig_attribution), axis=-1)
                plt.imshow(attribution_magnitude, cmap='hot')
                plt.title("Integrated Gradients\\nAttribution Magnitude")
                plt.colorbar()
                plt.axis('off')
                
                # Overlay on original
                plt.subplot(1, 3, 3)
                plt.imshow(image.astype(np.uint8), alpha=0.7)
                plt.imshow(attribution_magnitude, cmap='hot', alpha=0.3)
                plt.title("Attribution Overlay")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(exp_dir / "integrated_gradients_explanation.png", bbox_inches='tight', dpi=150)
                plt.close()
                
            except Exception as e:
                print(f"Error saving Integrated Gradients visualization: {e}")
        
        # Save summary JSON (without large arrays)
        summary = {
            "model_path": str(self.model_path),
            "explanation_date": datetime.now().isoformat(),
            "image_name": image_name,
            "prediction": explanations["prediction"],
            "methods_available": {
                "lime": explanations["lime"]["available"],
                "shap": explanations["shap"]["available"],
                "integrated_gradients": explanations["integrated_gradients"]["available"]
            }
        }
        
        with open(exp_dir / "explanation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Explanations saved in: {exp_dir}")
        
        return exp_dir

def explain_melanoma_predictions(model_path=None, image_paths=None):
    """Main function to generate explanations for melanoma predictions."""
    
    if not TF_AVAILABLE:
        print("TensorFlow dependencies not available.")
        return None
    
    print("Starting advanced explainability analysis...")
    
    # Initialize explainer
    explainer = AdvancedExplainer(model_path)
    explainer.load_model()
    
    # Default test images if none provided
    if image_paths is None:
        # Use some sample images for demonstration
        data_dir = Path(Config.DATA_DIR)
        image_paths = list((data_dir / "processed" / "images").glob("*.jpg"))[:3]
    
    if not image_paths:
        print("No images found for explanation. Please provide image paths.")
        return None
    
    all_explanations = []
    
    for image_path in image_paths:
        print(f"\\nExplaining image: {image_path}")
        
        try:
            explanations = explainer.explain_image(image_path)
            all_explanations.append(explanations)
            
        except Exception as e:
            print(f"Failed to explain {image_path}: {e}")
    
    # Create summary report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = explainer.results_dir / f"explainability_report_{timestamp}.json"
    
    report = {
        "model_path": str(explainer.model_path),
        "analysis_date": datetime.now().isoformat(),
        "images_analyzed": len(all_explanations),
        "methods_available": {
            "lime": LIME_AVAILABLE,
            "shap": SHAP_AVAILABLE,
            "integrated_gradients": True
        },
        "summary": {
            "total_predictions": len(all_explanations),
            "melanoma_predictions": sum(1 for exp in all_explanations 
                                      if exp['prediction']['class'] == 'Melanoma'),
            "average_confidence": np.mean([exp['prediction']['probability'] 
                                         for exp in all_explanations])
        }
    }
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\\n=== Explainability Analysis Complete ===")
    print(f"Images analyzed: {len(all_explanations)}")
    print(f"Methods available: LIME={LIME_AVAILABLE}, SHAP={SHAP_AVAILABLE}, IntegratedGradients=True")
    print(f"Report saved: {report_file}")
    
    return report

if __name__ == "__main__":
    report = explain_melanoma_predictions()
    if report:
        print("Advanced explainability analysis completed successfully!")