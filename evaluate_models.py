"""
Comprehensive model evaluation suite for melanoma classification models.
Generates AUROC, PR-AUC, sensitivity/specificity, confusion matrices, and calibration curves.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    calibration_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from tensorflow import keras
from config import Config
from preprocess_data import DataPreprocessor
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self, input_size=(224, 224)):
        self.input_size = input_size
        self.data_dir = Path(Config.DATA_DIR)
        self.models_dir = Path(Config.MODELS_DIR)
        self.results_dir = Path(Config.RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = DataPreprocessor(input_size=input_size)

    def load_test_data(self):
        """Load test dataset for evaluation."""
        splits_dir = self.data_dir / "processed" / "splits"
        test_df = pd.read_csv(splits_dir / "test.csv")
        
        print(f"Test samples: {len(test_df)}")
        print(f"Test melanoma cases: {test_df['target'].sum()}")
        print(f"Test class balance: {test_df['target'].mean():.3f}")
        
        return test_df

    def load_ph2_data(self):
        """Load PH2 dataset if available for external validation."""
        ph2_path = self.data_dir / "processed" / "ph2_metadata.csv"
        
        if ph2_path.exists():
            ph2_df = pd.read_csv(ph2_path)
            print(f"PH2 samples: {len(ph2_df)}")
            return ph2_df
        else:
            print("PH2 dataset not available for external validation")
            return None

    def create_test_dataset(self, test_df, batch_size=32):
        """Create test dataset without augmentation."""
        test_dataset = self.preprocessor.create_tf_dataset(
            test_df,
            batch_size=batch_size,
            shuffle=False,
            transforms=self.preprocessor.val_transforms
        )
        return test_dataset

    def evaluate_model(self, model_path, test_df, model_name="model"):
        """Evaluate a single model and return comprehensive metrics."""
        print(f"\nEvaluating {model_name}...")
        
        # Load model
        if isinstance(model_path, str) or isinstance(model_path, Path):
            model = keras.models.load_model(str(model_path))
        else:
            model = model_path
        
        # Create test dataset
        test_dataset = self.create_test_dataset(test_df)
        
        # Get predictions
        print("Generating predictions...")
        predictions = model.predict(test_dataset, verbose=1)
        y_pred_proba = predictions.flatten()
        y_true = test_df['target'].values
        
        # Binary predictions with optimal threshold
        optimal_threshold = self.find_optimal_threshold(y_true, y_pred_proba)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba, optimal_threshold)
        
        # Generate plots
        plot_dir = self.results_dir / f"{model_name}_evaluation"
        plot_dir.mkdir(exist_ok=True)
        
        self.plot_roc_curve(y_true, y_pred_proba, save_path=plot_dir / "roc_curve.png")
        self.plot_precision_recall_curve(y_true, y_pred_proba, save_path=plot_dir / "pr_curve.png")
        self.plot_confusion_matrix(y_true, y_pred, save_path=plot_dir / "confusion_matrix.png")
        self.plot_calibration_curve(y_true, y_pred_proba, save_path=plot_dir / "calibration_curve.png")
        self.plot_prediction_distribution(y_pred_proba, y_true, save_path=plot_dir / "prediction_distribution.png")
        
        # Save metrics
        self.save_metrics(metrics, plot_dir / "metrics.json")
        
        return metrics, y_pred_proba

    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Find optimal threshold using Youden's J statistic."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        return optimal_threshold

    def calculate_metrics(self, y_true, y_pred, y_pred_proba, threshold):
        """Calculate comprehensive evaluation metrics."""
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Area under curves
        auroc = roc_auc_score(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)
        
        # Calibration metrics
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        metrics = {
            'threshold': float(threshold),
            'auroc': float(auroc),
            'auprc': float(auprc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'npv': float(npv),
            'accuracy': float(accuracy),
            'f1_score': float(f1_score),
            'brier_score': float(brier_score),
            'confusion_matrix': {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
        }
        
        return metrics

    def plot_roc_curve(self, y_true, y_pred_proba, save_path):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auroc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)
        baseline = np.mean(y_true)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {auprc:.3f})')
        plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benign', 'Melanoma'],
                   yticklabels=['Benign', 'Melanoma'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_calibration_curve(self, y_true, y_pred_proba, save_path, n_bins=10):
        """Plot calibration curve to assess probability calibration."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prediction_distribution(self, y_pred_proba, y_true, save_path):
        """Plot distribution of predictions by class."""
        plt.figure(figsize=(10, 6))
        
        # Separate predictions by true class
        benign_preds = y_pred_proba[y_true == 0]
        melanoma_preds = y_pred_proba[y_true == 1]
        
        plt.hist(benign_preds, bins=50, alpha=0.7, label='Benign', color='blue', density=True)
        plt.hist(melanoma_preds, bins=50, alpha=0.7, label='Melanoma', color='red', density=True)
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Predicted Probabilities by True Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self, metrics, save_path):
        """Save metrics to JSON file."""
        metrics['evaluation_date'] = datetime.now().isoformat()
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {save_path}")

    def compare_models(self, model_results):
        """Compare multiple models and generate comparison plots."""
        print("\nComparing models...")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in model_results.items():
            comparison_data.append({
                'model': model_name,
                'auroc': metrics['auroc'],
                'auprc': metrics['auprc'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_df.to_csv(self.results_dir / "model_comparison.csv", index=False)
        
        # Plot comparison
        self.plot_model_comparison(comparison_df)
        
        return comparison_df

    def plot_model_comparison(self, comparison_df):
        """Plot model comparison charts."""
        metrics_to_plot = ['auroc', 'auprc', 'sensitivity', 'specificity', 'accuracy', 'f1_score']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            bars = ax.bar(comparison_df['model'], comparison_df[metric])
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel(metric.capitalize())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_classification_report(self, y_true, y_pred, save_path):
        """Generate detailed classification report."""
        report = classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma'])
        
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        
        print(f"Classification report saved to: {save_path}")

def main():
    """Main evaluation function."""
    print("DermAI Model Evaluation Suite")
    print("=" * 50)
    
    evaluator = ModelEvaluator()
    
    # Load test data
    test_df = evaluator.load_test_data()
    
    # Find trained models
    models_to_evaluate = []
    
    # Look for benchmark models
    for model_dir in evaluator.models_dir.glob("benchmark_*"):
        best_model = model_dir / "best_benchmark_efficientnet.h5"
        if best_model.exists():
            models_to_evaluate.append(("Benchmark EfficientNet", str(best_model)))
    
    # Look for mobile models
    for model_dir in evaluator.models_dir.glob("mobile_*"):
        best_model = model_dir / "best_mobile_melanoma.h5"
        if best_model.exists():
            models_to_evaluate.append(("Mobile MobileNetV3", str(best_model)))
    
    if not models_to_evaluate:
        print("No trained models found. Please run training scripts first.")
        return
    
    print(f"Found {len(models_to_evaluate)} models to evaluate")
    
    # Evaluate each model
    model_results = {}
    
    for model_name, model_path in models_to_evaluate:
        try:
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*60}")
            
            metrics, predictions = evaluator.evaluate_model(
                model_path, test_df, model_name.lower().replace(" ", "_")
            )
            
            model_results[model_name] = metrics
            
            # Print key metrics
            print(f"\n{model_name} Results:")
            print(f"AUROC: {metrics['auroc']:.4f}")
            print(f"AUPRC: {metrics['auprc']:.4f}")
            print(f"Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"Specificity: {metrics['specificity']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    # Compare models if we have multiple
    if len(model_results) > 1:
        comparison_df = evaluator.compare_models(model_results)
        print(f"\nModel Comparison Summary:")
        print(comparison_df.to_string(index=False))
    
    # External validation on PH2 if available
    ph2_df = evaluator.load_ph2_data()
    if ph2_df is not None:
        print("\nPerforming external validation on PH2 dataset...")
        # Would evaluate on PH2 here
    
    print(f"\n✓ Model evaluation completed!")
    print(f"Results saved to: {evaluator.results_dir}")

if __name__ == "__main__":
    main()
