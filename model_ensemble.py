"""
Model Ensemble Strategy for Melanoma Classification.
Combines predictions from multiple models for improved accuracy and robustness.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Callable
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import json
from datetime import datetime


class ModelEnsemble:
    """
    Model ensemble class for combining predictions from multiple models.
    
    Supports various ensemble strategies including:
    - Simple averaging
    - Weighted averaging  
    - Voting-based methods
    - Meta-learning (stacking)
    """
    
    def __init__(self, ensemble_method: str = 'weighted_average', 
                 meta_learner: str = 'logistic'):
        """
        Initialize model ensemble.
        
        Args:
            ensemble_method: Method for combining predictions
                           ('average', 'weighted_average', 'voting', 'meta_learning')
            meta_learner: Type of meta-learner for stacking ('logistic', 'rf')
        """
        self.ensemble_method = ensemble_method
        self.meta_learner_type = meta_learner
        self.models = []
        self.model_names = []
        self.model_weights = None
        self.meta_learner = None
        self.model_performance = {}
        
        # Supported ensemble methods
        self.ensemble_methods = {
            'average': self._simple_average,
            'weighted_average': self._weighted_average,
            'voting': self._majority_voting,
            'meta_learning': self._meta_learning_predict
        }
        
        if ensemble_method not in self.ensemble_methods:
            raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
    
    def add_model(self, model: Union[tf.keras.Model, str, Path], 
                  model_name: str, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            model: Keras model, model path, or weights path
            model_name: Name identifier for the model
            weight: Weight for weighted averaging
        """
        if isinstance(model, (str, Path)):
            # Load model from path
            model_path = Path(model)
            if model_path.suffix == '.h5' and 'weights' in str(model_path):
                # This is a weights file, need to create model architecture first
                print(f"Warning: Cannot load weights-only file {model_path}")
                print("Please provide full model or create architecture first")
                return
            else:
                # Load full model
                try:
                    model = tf.keras.models.load_model(str(model_path))
                except Exception as e:
                    print(f"Error loading model {model_path}: {e}")
                    return
        
        self.models.append(model)
        self.model_names.append(model_name)
        
        # Initialize weights if using weighted averaging
        if self.model_weights is None:
            self.model_weights = []
        self.model_weights.append(weight)
        
        print(f"✓ Added model '{model_name}' to ensemble")
    
    def evaluate_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate each model individually to determine performance.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Dictionary with individual model performance
        """
        results = {}
        
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            predictions = model.predict(X, verbose=0)
            
            # Handle different prediction formats
            if predictions.shape[1] == 1:
                pred_probs = predictions.flatten()
            else:
                pred_probs = predictions[:, 1]  # Binary classification
            
            binary_preds = (pred_probs > 0.5).astype(int)
            
            # Calculate metrics
            auc = roc_auc_score(y, pred_probs)
            acc = accuracy_score(y, binary_preds)
            prec = precision_score(y, binary_preds, zero_division=0)
            rec = recall_score(y, binary_preds, zero_division=0)
            
            results[name] = {
                'auc': auc,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'predictions': pred_probs
            }
            
            # Store for weight calculation
            self.model_performance[name] = auc
        
        return results
    
    def calculate_optimal_weights(self, X: np.ndarray, y: np.ndarray, 
                                 method: str = 'auc_based') -> np.ndarray:
        """
        Calculate optimal weights for ensemble based on validation performance.
        
        Args:
            X: Validation data
            y: Validation labels
            method: Method for weight calculation ('auc_based', 'inverse_error', 'uniform')
            
        Returns:
            Array of optimal weights
        """
        if method == 'uniform':
            weights = np.ones(len(self.models)) / len(self.models)
        
        elif method == 'auc_based':
            # Evaluate individual models
            individual_results = self.evaluate_individual_models(X, y)
            
            # Weight by AUC performance
            aucs = [individual_results[name]['auc'] for name in self.model_names]
            aucs = np.array(aucs)
            
            # Softmax weighting to ensure positive weights that sum to 1
            weights = np.exp(aucs * 5)  # Temperature scaling
            weights = weights / np.sum(weights)
        
        elif method == 'inverse_error':
            # Weight by inverse of error rate
            individual_results = self.evaluate_individual_models(X, y)
            
            errors = [1 - individual_results[name]['accuracy'] for name in self.model_names]
            errors = np.array(errors) + 1e-8  # Avoid division by zero
            
            weights = 1 / errors
            weights = weights / np.sum(weights)
        
        else:
            raise ValueError(f"Unknown weight calculation method: {method}")
        
        self.model_weights = weights
        print(f"✓ Calculated optimal weights: {dict(zip(self.model_names, weights))}")
        
        return weights
    
    def train_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """
        Train meta-learner for stacking ensemble.
        
        Args:
            X: Training data
            y: Training labels
        """
        # Get predictions from all base models
        base_predictions = []
        
        for model in self.models:
            preds = model.predict(X, verbose=0)
            if preds.shape[1] == 1:
                base_predictions.append(preds.flatten())
            else:
                base_predictions.append(preds[:, 1])
        
        # Stack predictions as features for meta-learner
        meta_features = np.column_stack(base_predictions)
        
        # Train meta-learner
        if self.meta_learner_type == 'logistic':
            self.meta_learner = LogisticRegression(random_state=42)
        elif self.meta_learner_type == 'rf':
            self.meta_learner = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=3
            )
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner_type}")
        
        self.meta_learner.fit(meta_features, y)
        print(f"✓ Trained {self.meta_learner_type} meta-learner on {len(X)} samples")
    
    def _simple_average(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Simple average ensemble."""
        return np.mean(predictions, axis=0)
    
    def _weighted_average(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Weighted average ensemble."""
        if self.model_weights is None:
            return self._simple_average(predictions)
        
        weights = np.array(self.model_weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        weighted_preds = np.average(predictions, axis=0, weights=weights)
        return weighted_preds
    
    def _majority_voting(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Majority voting ensemble."""
        # Convert to binary predictions
        binary_preds = [(pred > 0.5).astype(int) for pred in predictions]
        
        # Majority vote
        votes = np.array(binary_preds)
        majority_vote = np.mean(votes, axis=0)
        
        # Return probabilities based on vote proportion
        return majority_vote
    
    def _meta_learning_predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Meta-learning ensemble prediction."""
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call train_meta_learner first.")
        
        # Stack predictions as features
        meta_features = np.column_stack(predictions)
        
        # Get meta-learner predictions
        meta_preds = self.meta_learner.predict_proba(meta_features)
        
        # Return probability of positive class
        if meta_preds.shape[1] == 2:
            return meta_preds[:, 1]
        else:
            return meta_preds.flatten()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input data
            
        Returns:
            Ensemble predictions
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Add models first.")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            preds = model.predict(X, verbose=0)
            if preds.shape[1] == 1:
                predictions.append(preds.flatten())
            else:
                predictions.append(preds[:, 1])
        
        # Combine predictions using selected method
        ensemble_method = self.ensemble_methods[self.ensemble_method]
        ensemble_preds = ensemble_method(predictions)
        
        return ensemble_preds
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            preds = model.predict(X, verbose=0)
            if preds.shape[1] == 1:
                predictions.append(preds.flatten())
            else:
                predictions.append(preds[:, 1])
        
        predictions = np.array(predictions)
        
        # Calculate ensemble prediction
        ensemble_preds = self.predict(X)
        
        # Calculate uncertainty as standard deviation across models
        uncertainties = np.std(predictions, axis=0)
        
        return ensemble_preds, uncertainties
    
    def evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate ensemble performance.
        
        Args:
            X: Test data
            y: Test labels
            
        Returns:
            Dictionary with performance metrics
        """
        # Get ensemble predictions
        pred_probs = self.predict(X)
        binary_preds = (pred_probs > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'auc': roc_auc_score(y, pred_probs),
            'accuracy': accuracy_score(y, binary_preds),
            'precision': precision_score(y, binary_preds, zero_division=0),
            'recall': recall_score(y, binary_preds, zero_division=0),
            'num_models': len(self.models),
            'ensemble_method': self.ensemble_method
        }
        
        # Add F1 score
        if metrics['precision'] > 0 or metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                           (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        return metrics
    
    def save_ensemble(self, save_path: Union[str, Path]):
        """
        Save ensemble configuration.
        
        Args:
            save_path: Path to save ensemble
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble metadata
        metadata = {
            'ensemble_method': self.ensemble_method,
            'meta_learner_type': self.meta_learner_type,
            'model_names': self.model_names,
            'model_weights': self.model_weights.tolist() if self.model_weights is not None else None,
            'model_performance': self.model_performance,
            'created_at': datetime.now().isoformat()
        }
        
        with open(save_path / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save meta-learner if trained
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, save_path / 'meta_learner.pkl')
        
        print(f"✓ Saved ensemble configuration to {save_path}")
    
    @classmethod
    def load_ensemble(cls, load_path: Union[str, Path], 
                     model_paths: List[Union[str, Path]]):
        """
        Load ensemble from saved configuration.
        
        Args:
            load_path: Path to ensemble configuration
            model_paths: Paths to individual models
            
        Returns:
            Loaded ensemble instance
        """
        load_path = Path(load_path)
        
        # Load metadata
        with open(load_path / 'ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Create ensemble instance
        ensemble = cls(
            ensemble_method=metadata['ensemble_method'],
            meta_learner=metadata['meta_learner_type']
        )
        
        # Load models
        for i, (model_path, model_name) in enumerate(zip(model_paths, metadata['model_names'])):
            weight = metadata['model_weights'][i] if metadata['model_weights'] else 1.0
            ensemble.add_model(model_path, model_name, weight)
        
        # Load meta-learner if exists
        meta_learner_path = load_path / 'meta_learner.pkl'
        if meta_learner_path.exists():
            ensemble.meta_learner = joblib.load(meta_learner_path)
        
        ensemble.model_performance = metadata['model_performance']
        
        print(f"✓ Loaded ensemble from {load_path}")
        return ensemble


def create_diverse_ensemble(benchmark_models: List[Union[str, Path]],
                          mobile_models: List[Union[str, Path]] = None,
                          ensemble_method: str = 'weighted_average') -> ModelEnsemble:
    """
    Create diverse ensemble combining benchmark and mobile models.
    
    Args:
        benchmark_models: List of paths to benchmark models
        mobile_models: List of paths to mobile models (optional)
        ensemble_method: Ensemble method to use
        
    Returns:
        Configured model ensemble
    """
    ensemble = ModelEnsemble(ensemble_method=ensemble_method)
    
    # Add benchmark models
    for i, model_path in enumerate(benchmark_models):
        model_name = f"benchmark_{i+1}"
        ensemble.add_model(model_path, model_name, weight=2.0)  # Higher weight for benchmark
    
    # Add mobile models if provided
    if mobile_models:
        for i, model_path in enumerate(mobile_models):
            model_name = f"mobile_{i+1}"
            ensemble.add_model(model_path, model_name, weight=1.0)  # Lower weight for mobile
    
    return ensemble


def compare_ensemble_methods(models: List[Union[str, Path]], 
                           model_names: List[str],
                           X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Compare different ensemble methods on the same set of models.
    
    Args:
        models: List of model paths
        model_names: List of model names
        X_val: Validation data for weight calculation
        y_val: Validation labels
        X_test: Test data for evaluation
        y_test: Test labels
        
    Returns:
        Comparison results
    """
    methods = ['average', 'weighted_average', 'voting', 'meta_learning']
    results = {}
    
    for method in methods:
        print(f"\nEvaluating {method} ensemble...")
        
        try:
            # Create ensemble
            ensemble = ModelEnsemble(ensemble_method=method)
            
            # Add models
            for model_path, name in zip(models, model_names):
                ensemble.add_model(model_path, name)
            
            # Train method-specific components
            if method == 'weighted_average':
                ensemble.calculate_optimal_weights(X_val, y_val, method='auc_based')
            elif method == 'meta_learning':
                ensemble.train_meta_learner(X_val, y_val)
            
            # Evaluate on test set
            metrics = ensemble.evaluate_ensemble(X_test, y_test)
            results[method] = metrics
            
            print(f"✓ {method}: AUC = {metrics['auc']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"✗ Error with {method}: {e}")
            results[method] = None
    
    return results


# Utility functions for ensemble analysis
def analyze_model_diversity(models: List[tf.keras.Model], 
                          X: np.ndarray) -> Dict:
    """
    Analyze diversity among ensemble models.
    
    Args:
        models: List of trained models
        X: Test data
        
    Returns:
        Diversity analysis results
    """
    predictions = []
    for model in models:
        preds = model.predict(X, verbose=0)
        if preds.shape[1] == 1:
            predictions.append(preds.flatten())
        else:
            predictions.append(preds[:, 1])
    
    predictions = np.array(predictions)
    
    # Calculate pairwise correlations
    correlations = np.corrcoef(predictions)
    
    # Calculate diversity metrics
    avg_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
    min_correlation = np.min(correlations[np.triu_indices_from(correlations, k=1)])
    max_correlation = np.max(correlations[np.triu_indices_from(correlations, k=1)])
    
    # Calculate disagreement rate (for binary predictions)
    binary_preds = (predictions > 0.5).astype(int)
    disagreements = []
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            disagreement = np.mean(binary_preds[i] != binary_preds[j])
            disagreements.append(disagreement)
    
    avg_disagreement = np.mean(disagreements)
    
    return {
        'avg_correlation': avg_correlation,
        'min_correlation': min_correlation,
        'max_correlation': max_correlation,
        'avg_disagreement': avg_disagreement,
        'correlation_matrix': correlations.tolist()
    }