"""
Model calibration and uncertainty quantification for melanoma classification.
Improves prediction reliability and provides confidence estimates for clinical use.
"""

import os
import json
from pathlib import Path
from datetime import datetime

try:
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    TF_AVAILABLE = True
except ImportError as e:
    print(f"ML dependencies not available: {e}")
    TF_AVAILABLE = False

from config import Config

class ModelCalibrator:
    """Model calibration and uncertainty quantification for melanoma classification."""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.calibrators = {}
        self.calibration_results = {}
        
        # Results directory
        self.results_dir = Path(Config.MODELS_DIR) / "calibration"
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
        
        print(f"Loading model for calibration: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        return self.model

    def load_validation_data(self):
        """Load validation data for calibration."""
        
        # Load the data splits
        splits_dir = Path(Config.DATA_DIR) / "processed" / "splits"
        val_df = pd.read_csv(splits_dir / "val.csv")
        
        print(f"Loading {len(val_df)} validation samples for calibration...")
        
        # Load images and labels
        images = []
        labels = []
        
        for idx, row in val_df.iterrows():
            image_name = row['image_name']
            target = row['target']
            
            # Load image
            image_path = Path(Config.DATA_DIR) / "processed" / "images" / image_name
            
            if image_path.exists():
                try:
                    image = tf.keras.preprocessing.image.load_img(
                        image_path, target_size=(224, 224)
                    )
                    image = tf.keras.preprocessing.image.img_to_array(image)
                    image = image / 255.0  # Normalize
                    
                    images.append(image)
                    labels.append(target)
                    
                except Exception as e:
                    print(f"Error loading {image_name}: {e}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Successfully loaded {len(images)} validation images")
        print(f"Melanoma cases: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        
        return images, labels

    def calculate_calibration_metrics(self, y_true, y_prob, n_bins=10):
        """Calculate calibration metrics including reliability diagram data."""
        
        # Bin probabilities
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Calculate metrics for each bin
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence for this bin
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        # Expected Calibration Error (ECE)
        ece = 0
        for i, (acc, conf, count) in enumerate(zip(bin_accuracies, bin_confidences, bin_counts)):
            if count > 0:
                ece += (count / len(y_prob)) * abs(acc - conf)
        
        # Maximum Calibration Error (MCE)
        mce = max([abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences) if conf > 0])
        
        # Brier Score
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Log Loss
        log_loss_score = log_loss(y_true, y_prob, eps=1e-15)
        
        metrics = {
            "ece": float(ece),
            "mce": float(mce),
            "brier_score": float(brier_score),
            "log_loss": float(log_loss_score),
            "reliability_diagram": {
                "bin_accuracies": [float(x) for x in bin_accuracies],
                "bin_confidences": [float(x) for x in bin_confidences],
                "bin_counts": [int(x) for x in bin_counts],
                "bin_boundaries": [float(x) for x in bin_boundaries]
            }
        }
        
        return metrics

    def platt_scaling(self, y_true, y_prob):
        """Apply Platt scaling (logistic regression) for calibration."""
        
        print("Applying Platt scaling...")
        
        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(y_prob.reshape(-1, 1), y_true)
        
        # Get calibrated probabilities
        calibrated_prob = lr.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        
        self.calibrators['platt'] = lr
        
        return calibrated_prob

    def isotonic_regression_calibration(self, y_true, y_prob):
        """Apply isotonic regression for calibration."""
        
        print("Applying isotonic regression...")
        
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(y_prob, y_true)
        
        # Get calibrated probabilities
        calibrated_prob = iso_reg.predict(y_prob)
        
        self.calibrators['isotonic'] = iso_reg
        
        return calibrated_prob

    def temperature_scaling(self, y_true, logits, learning_rate=0.01, epochs=100):
        """Apply temperature scaling for calibration."""
        
        print("Applying temperature scaling...")
        
        # Initialize temperature parameter
        temperature = tf.Variable(1.0, trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Convert to TensorFlow tensors
        logits_tensor = tf.constant(logits, dtype=tf.float32)
        labels_tensor = tf.constant(y_true, dtype=tf.float32)
        
        # Training loop
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # Apply temperature scaling
                scaled_logits = logits_tensor / temperature
                scaled_probs = tf.nn.sigmoid(scaled_logits)
                
                # Calculate loss (negative log-likelihood)
                loss = tf.keras.losses.binary_crossentropy(labels_tensor, scaled_probs)
                loss = tf.reduce_mean(loss)
            
            # Update temperature
            gradients = tape.gradient(loss, [temperature])
            optimizer.apply_gradients(zip(gradients, [temperature]))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Temperature: {temperature.numpy():.4f}, Loss: {loss.numpy():.4f}")
        
        final_temperature = temperature.numpy()
        print(f"Final temperature: {final_temperature:.4f}")
        
        # Apply final temperature scaling
        scaled_logits = logits / final_temperature
        calibrated_prob = tf.nn.sigmoid(scaled_logits).numpy()
        
        self.calibrators['temperature'] = final_temperature
        
        return calibrated_prob

    def monte_carlo_dropout(self, images, n_samples=100, dropout_rate=0.1):
        """Estimate uncertainty using Monte Carlo Dropout."""
        
        print(f"Estimating uncertainty using Monte Carlo Dropout ({n_samples} samples)...")
        
        # Create a version of the model with dropout enabled during inference
        mc_model = self._create_mc_dropout_model(dropout_rate)
        
        # Collect predictions from multiple forward passes
        predictions = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"MC sample {i+1}/{n_samples}")
            
            pred = mc_model(images, training=True)  # training=True enables dropout
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)  # Shape: (n_samples, n_images, n_classes)
        
        # Calculate mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # For binary classification
        if mean_pred.shape[1] == 1:
            uncertainty = std_pred[:, 0]
            confidence = mean_pred[:, 0]
        else:
            # For multi-class, use entropy as uncertainty measure
            mean_probs = mean_pred
            uncertainty = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
            confidence = np.max(mean_probs, axis=1)
        
        return confidence, uncertainty

    def _create_mc_dropout_model(self, dropout_rate=0.1):
        """Create a model with dropout layers for Monte Carlo estimation."""
        
        # This is a simplified version - in practice, you'd modify your actual model architecture
        # to include dropout layers that can be enabled during inference
        
        # For now, create a wrapper that adds dropout to the final layers
        inputs = self.model.input
        
        # Get the output from the second-to-last layer
        x = self.model.layers[-2].output
        
        # Add dropout
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Add final prediction layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        mc_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Copy weights from original model (except the last layer)
        for i, layer in enumerate(mc_model.layers[:-2]):
            if i < len(self.model.layers) - 1:
                layer.set_weights(self.model.layers[i].get_weights())
        
        return mc_model

    def deep_ensemble_uncertainty(self, images, model_paths=None, n_models=5):
        """Estimate uncertainty using deep ensemble of models."""
        
        if model_paths is None:
            print("No model paths provided for ensemble. Using single model.")
            return self.model.predict(images), np.zeros(len(images))
        
        print(f"Estimating uncertainty using deep ensemble ({len(model_paths)} models)...")
        
        predictions = []
        
        for i, model_path in enumerate(model_paths):
            print(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
            
            model = tf.keras.models.load_model(model_path)
            pred = model.predict(images, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)  # Shape: (n_models, n_images, n_classes)
        
        # Calculate mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # For binary classification
        if mean_pred.shape[1] == 1:
            uncertainty = std_pred[:, 0]
            confidence = mean_pred[:, 0]
        else:
            # For multi-class, use entropy as uncertainty measure
            mean_probs = mean_pred
            uncertainty = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
            confidence = np.max(mean_probs, axis=1)
        
        return confidence, uncertainty

    def calibrate_model(self):
        """Perform comprehensive model calibration."""
        
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Starting model calibration process...")
        
        # Load validation data
        images, labels = self.load_validation_data()
        
        # Get model predictions
        print("Getting model predictions...")
        predictions = self.model.predict(images, batch_size=32, verbose=1)
        
        # Handle different output formats
        if predictions.shape[1] == 1:
            # Binary classification
            y_prob = predictions[:, 0]
        else:
            # Multi-class classification - assume melanoma is class 1
            y_prob = predictions[:, 1]
        
        # Calculate original calibration metrics
        print("Calculating original calibration metrics...")
        original_metrics = self.calculate_calibration_metrics(labels, y_prob)
        
        print(f"Original ECE: {original_metrics['ece']:.4f}")
        print(f"Original MCE: {original_metrics['mce']:.4f}")
        print(f"Original Brier Score: {original_metrics['brier_score']:.4f}")
        
        # Apply different calibration methods
        calibration_methods = {}
        
        # Platt scaling
        try:
            platt_prob = self.platt_scaling(labels, y_prob)
            platt_metrics = self.calculate_calibration_metrics(labels, platt_prob)
            calibration_methods['platt'] = {
                'probabilities': platt_prob,
                'metrics': platt_metrics
            }
            print(f"Platt Scaling ECE: {platt_metrics['ece']:.4f}")
        except Exception as e:
            print(f"Platt scaling failed: {e}")
        
        # Isotonic regression
        try:
            iso_prob = self.isotonic_regression_calibration(labels, y_prob)
            iso_metrics = self.calculate_calibration_metrics(labels, iso_prob)
            calibration_methods['isotonic'] = {
                'probabilities': iso_prob,
                'metrics': iso_metrics
            }
            print(f"Isotonic Regression ECE: {iso_metrics['ece']:.4f}")
        except Exception as e:
            print(f"Isotonic regression failed: {e}")
        
        # Temperature scaling (requires logits)
        try:
            # Approximate logits from probabilities (not ideal, but workable)
            logits = np.log(y_prob / (1 - y_prob + 1e-8))
            temp_prob = self.temperature_scaling(labels, logits)
            temp_metrics = self.calculate_calibration_metrics(labels, temp_prob)
            calibration_methods['temperature'] = {
                'probabilities': temp_prob,
                'metrics': temp_metrics
            }
            print(f"Temperature Scaling ECE: {temp_metrics['ece']:.4f}")
        except Exception as e:
            print(f"Temperature scaling failed: {e}")
        
        # Uncertainty estimation
        print("\\nEstimating prediction uncertainty...")
        
        try:
            mc_confidence, mc_uncertainty = self.monte_carlo_dropout(images, n_samples=50)
            uncertainty_results = {
                'monte_carlo_dropout': {
                    'confidence': mc_confidence.tolist(),
                    'uncertainty': mc_uncertainty.tolist(),
                    'mean_uncertainty': float(np.mean(mc_uncertainty)),
                    'std_uncertainty': float(np.std(mc_uncertainty))
                }
            }
            print(f"Monte Carlo Dropout - Mean uncertainty: {np.mean(mc_uncertainty):.4f}")
        except Exception as e:
            print(f"Monte Carlo Dropout failed: {e}")
            uncertainty_results = {}
        
        # Store results
        self.calibration_results = {
            'model_path': str(self.model_path),
            'calibration_date': datetime.now().isoformat(),
            'original_metrics': original_metrics,
            'calibration_methods': calibration_methods,
            'uncertainty_estimation': uncertainty_results,
            'validation_samples': len(images)
        }
        
        # Save results and create visualizations
        self._save_calibration_results()
        self._create_calibration_plots()
        
        return self.calibration_results

    def _save_calibration_results(self):
        """Save calibration results to file."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"calibration_results_{timestamp}.json"
        
        # Create a copy without large arrays for JSON serialization
        results_copy = self.calibration_results.copy()
        
        # Remove large probability arrays, keep only metrics
        for method in results_copy.get('calibration_methods', {}):
            if 'probabilities' in results_copy['calibration_methods'][method]:
                del results_copy['calibration_methods'][method]['probabilities']
        
        with open(results_file, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"Calibration results saved: {results_file}")

    def _create_calibration_plots(self):
        """Create calibration visualization plots."""
        
        if not TF_AVAILABLE:
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Reliability diagram
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            methods_to_plot = ['original'] + list(self.calibration_results.get('calibration_methods', {}).keys())
            
            for i, method in enumerate(methods_to_plot[:4]):
                row, col = i // 2, i % 2
                ax = axes[row, col]
                
                if method == 'original':
                    metrics = self.calibration_results['original_metrics']
                else:
                    metrics = self.calibration_results['calibration_methods'][method]['metrics']
                
                # Plot reliability diagram
                bin_accs = metrics['reliability_diagram']['bin_accuracies']
                bin_confs = metrics['reliability_diagram']['bin_confidences']
                bin_counts = metrics['reliability_diagram']['bin_counts']
                
                # Only plot bins with samples
                valid_bins = [i for i, count in enumerate(bin_counts) if count > 0]
                
                if valid_bins:
                    x_vals = [bin_confs[i] for i in valid_bins]
                    y_vals = [bin_accs[i] for i in valid_bins]
                    sizes = [bin_counts[i] for i in valid_bins]
                    
                    # Scatter plot with size proportional to bin count
                    ax.scatter(x_vals, y_vals, s=[s*5 for s in sizes], alpha=0.7)
                    
                    # Perfect calibration line
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
                    
                    ax.set_xlabel('Mean Predicted Probability')
                    ax.set_ylabel('Fraction of Positives')
                    ax.set_title(f'{method.title()}\\nECE: {metrics["ece"]:.4f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f"calibration_reliability_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calibration metrics comparison
            if self.calibration_results.get('calibration_methods'):
                methods = ['original'] + list(self.calibration_results['calibration_methods'].keys())
                metrics_names = ['ece', 'mce', 'brier_score']
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                for i, metric in enumerate(metrics_names):
                    values = []
                    
                    # Original metric
                    values.append(self.calibration_results['original_metrics'][metric])
                    
                    # Calibrated metrics
                    for method in self.calibration_results['calibration_methods']:
                        values.append(self.calibration_results['calibration_methods'][method]['metrics'][metric])
                    
                    axes[i].bar(methods, values)
                    axes[i].set_title(f'{metric.upper()}')
                    axes[i].set_ylabel('Score')
                    
                    # Add value labels on bars
                    for j, v in enumerate(values):
                        axes[i].text(j, v + max(values)*0.01, f'{v:.4f}', ha='center')
                
                plt.tight_layout()
                plt.savefig(self.results_dir / f"calibration_metrics_{timestamp}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Calibration plots saved in: {self.results_dir}")
            
        except Exception as e:
            print(f"Error creating calibration plots: {e}")

def calibrate_melanoma_model(model_path=None):
    """Main function to calibrate melanoma classification model."""
    
    if not TF_AVAILABLE:
        print("TensorFlow dependencies not available.")
        return None
    
    print("Starting model calibration for melanoma classification...")
    
    # Initialize calibrator
    calibrator = ModelCalibrator(model_path)
    calibrator.load_model()
    
    # Run calibration
    results = calibrator.calibrate_model()
    
    print("\\n=== Calibration Summary ===")
    print(f"Original ECE: {results['original_metrics']['ece']:.4f}")
    
    if 'calibration_methods' in results:
        for method, data in results['calibration_methods'].items():
            ece = data['metrics']['ece']
            improvement = results['original_metrics']['ece'] - ece
            print(f"{method.title()} ECE: {ece:.4f} (improvement: {improvement:.4f})")
    
    if 'uncertainty_estimation' in results and results['uncertainty_estimation']:
        unc_data = results['uncertainty_estimation'].get('monte_carlo_dropout', {})
        if unc_data:
            print(f"Mean prediction uncertainty: {unc_data['mean_uncertainty']:.4f}")
    
    return results

if __name__ == "__main__":
    results = calibrate_melanoma_model()
    if results:
        print("Model calibration completed successfully!")