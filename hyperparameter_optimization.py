"""
Hyperparameter Optimization for Melanoma Classification Models.
Provides systematic approaches to optimize batch size, learning rate, 
augmentation parameters, and other hyperparameters for better performance.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import itertools
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
import optuna
from config import Config


class HyperparameterOptimizer:
    """
    Hyperparameter optimization class for melanoma classification models.
    
    Supports various optimization strategies:
    - Grid search
    - Random search
    - Bayesian optimization (using Optuna)
    - Learning rate range testing
    - Batch size optimization
    """
    
    def __init__(self, model_trainer_class, base_config: Dict = None):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            model_trainer_class: Model trainer class (BenchmarkModelTrainer or MobileModelTrainer)
            base_config: Base configuration dictionary
        """
        self.model_trainer_class = model_trainer_class
        self.base_config = base_config or {}
        self.optimization_history = []
        self.best_params = None
        self.best_score = -np.inf
        
        # Results storage
        self.results_dir = Path(Config.MODELS_DIR) / "hyperparameter_optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def grid_search(self, param_grid: Dict, train_data: Tuple, 
                   val_data: Tuple, metric: str = 'val_auc',
                   max_epochs: int = 20) -> Dict:
        """
        Perform grid search over hyperparameters.
        
        Args:
            param_grid: Dictionary of parameters to search
            train_data: (X_train, y_train) tuple
            val_data: (X_val, y_val) tuple  
            metric: Metric to optimize ('val_auc', 'val_accuracy')
            max_epochs: Maximum epochs per trial
            
        Returns:
            Best parameters and results
        """
        print(f"Starting grid search with {len(list(ParameterGrid(param_grid)))} combinations...")
        
        results = []
        grid = ParameterGrid(param_grid)
        
        for i, params in enumerate(grid):
            print(f"\nTrial {i+1}/{len(grid)}: {params}")
            
            # Update configuration
            config = {**self.base_config, **params}
            
            # Train model with current parameters
            try:
                score = self._train_and_evaluate(config, train_data, val_data, 
                                               metric, max_epochs)
                
                result = {
                    'trial': i + 1,
                    'params': params,
                    'score': score,
                    'metric': metric
                }
                results.append(result)
                
                # Update best if improved
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                print(f"✓ {metric}: {score:.4f}")
                
            except Exception as e:
                print(f"✗ Trial failed: {e}")
                continue
        
        # Save results
        self._save_results(results, 'grid_search')
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results
        }
    
    def random_search(self, param_distributions: Dict, n_trials: int,
                     train_data: Tuple, val_data: Tuple, 
                     metric: str = 'val_auc', max_epochs: int = 20) -> Dict:
        """
        Perform random search over hyperparameters.
        
        Args:
            param_distributions: Dictionary of parameter distributions
            n_trials: Number of random trials
            train_data: (X_train, y_train) tuple
            val_data: (X_val, y_val) tuple
            metric: Metric to optimize
            max_epochs: Maximum epochs per trial
            
        Returns:
            Best parameters and results
        """
        print(f"Starting random search with {n_trials} trials...")
        
        results = []
        
        for i in range(n_trials):
            # Sample random parameters
            params = self._sample_params(param_distributions)
            print(f"\nTrial {i+1}/{n_trials}: {params}")
            
            # Update configuration
            config = {**self.base_config, **params}
            
            # Train model with current parameters
            try:
                score = self._train_and_evaluate(config, train_data, val_data,
                                               metric, max_epochs)
                
                result = {
                    'trial': i + 1,
                    'params': params,
                    'score': score,
                    'metric': metric
                }
                results.append(result)
                
                # Update best if improved
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                print(f"✓ {metric}: {score:.4f}")
                
            except Exception as e:
                print(f"✗ Trial failed: {e}")
                continue
        
        # Save results
        self._save_results(results, 'random_search')
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results
        }
    
    def bayesian_optimization(self, param_space: Dict, n_trials: int,
                            train_data: Tuple, val_data: Tuple,
                            metric: str = 'val_auc', max_epochs: int = 20) -> Dict:
        """
        Perform Bayesian optimization using Optuna.
        
        Args:
            param_space: Dictionary defining parameter search space
            n_trials: Number of optimization trials
            train_data: (X_train, y_train) tuple
            val_data: (X_val, y_val) tuple
            metric: Metric to optimize
            max_epochs: Maximum epochs per trial
            
        Returns:
            Best parameters and results
        """
        print(f"Starting Bayesian optimization with {n_trials} trials...")
        
        def objective(trial):
            # Sample parameters from search space
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # Update configuration
            config = {**self.base_config, **params}
            
            try:
                score = self._train_and_evaluate(config, train_data, val_data,
                                               metric, max_epochs)
                print(f"Trial {trial.number}: {params} -> {metric}: {score:.4f}")
                return score
                
            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                return -np.inf
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Extract results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        results = []
        for trial in study.trials:
            if trial.value is not None:
                results.append({
                    'trial': trial.number,
                    'params': trial.params,
                    'score': trial.value,
                    'metric': metric
                })
        
        # Save results
        self._save_results(results, 'bayesian_optimization')
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results,
            'study': study
        }
    
    def learning_rate_range_test(self, train_data: Tuple, val_data: Tuple,
                                min_lr: float = 1e-6, max_lr: float = 1e-1,
                                num_steps: int = 100) -> Dict:
        """
        Perform learning rate range test to find optimal learning rate.
        
        Args:
            train_data: (X_train, y_train) tuple
            val_data: (X_val, y_val) tuple
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            num_steps: Number of steps in range test
            
        Returns:
            Learning rate analysis results
        """
        print(f"Learning rate range test: {min_lr} to {max_lr}")
        
        # Generate learning rate schedule
        lr_schedule = np.logspace(np.log10(min_lr), np.log10(max_lr), num_steps)
        
        results = []
        losses = []
        
        for i, lr in enumerate(lr_schedule):
            print(f"Step {i+1}/{num_steps}: LR = {lr:.2e}")
            
            # Create trainer with current learning rate
            config = {**self.base_config, 'learning_rate': lr}
            
            try:
                # Train for just a few epochs
                score = self._train_and_evaluate(config, train_data, val_data,
                                               'val_loss', max_epochs=3, 
                                               minimize=True)
                
                results.append({
                    'learning_rate': lr,
                    'val_loss': score,
                    'step': i + 1
                })
                losses.append(score)
                
                # Early stopping if loss explodes
                if len(losses) > 1 and losses[-1] > losses[-2] * 2:
                    print("Loss exploding, stopping early")
                    break
                    
            except Exception as e:
                print(f"Failed at LR {lr}: {e}")
                break
        
        # Find optimal learning rate (lowest loss)
        if results:
            best_result = min(results, key=lambda x: x['val_loss'])
            optimal_lr = best_result['learning_rate']
            
            # Plot results
            self._plot_lr_range_test(results)
            
            return {
                'optimal_lr': optimal_lr,
                'results': results,
                'min_loss': best_result['val_loss']
            }
        
        return {'optimal_lr': None, 'results': []}
    
    def batch_size_optimization(self, train_data: Tuple, val_data: Tuple,
                              batch_sizes: List[int] = None,
                              max_epochs: int = 10) -> Dict:
        """
        Optimize batch size for best performance.
        
        Args:
            train_data: (X_train, y_train) tuple
            val_data: (X_val, y_val) tuple
            batch_sizes: List of batch sizes to test
            max_epochs: Maximum epochs per trial
            
        Returns:
            Batch size optimization results
        """
        if batch_sizes is None:
            batch_sizes = [8, 16, 32, 64, 128]
        
        print(f"Testing batch sizes: {batch_sizes}")
        
        results = []
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            config = {**self.base_config, 'batch_size': batch_size}
            
            try:
                score = self._train_and_evaluate(config, train_data, val_data,
                                               'val_auc', max_epochs)
                
                result = {
                    'batch_size': batch_size,
                    'val_auc': score,
                    'status': 'success'
                }
                results.append(result)
                
                print(f"✓ Batch size {batch_size}: AUC = {score:.4f}")
                
            except Exception as e:
                print(f"✗ Batch size {batch_size} failed: {e}")
                results.append({
                    'batch_size': batch_size,
                    'val_auc': 0.0,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Find best batch size
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['val_auc'])
            optimal_batch_size = best_result['batch_size']
        else:
            optimal_batch_size = None
        
        return {
            'optimal_batch_size': optimal_batch_size,
            'results': results
        }
    
    def augmentation_optimization(self, train_data: Tuple, val_data: Tuple,
                                augmentation_params: Dict,
                                n_trials: int = 20) -> Dict:
        """
        Optimize data augmentation parameters.
        
        Args:
            train_data: (X_train, y_train) tuple
            val_data: (X_val, y_val) tuple
            augmentation_params: Dictionary of augmentation parameter ranges
            n_trials: Number of trials for optimization
            
        Returns:
            Optimal augmentation parameters
        """
        print(f"Optimizing augmentation parameters with {n_trials} trials...")
        
        results = []
        
        for i in range(n_trials):
            # Sample augmentation parameters
            aug_params = {}
            for param_name, param_range in augmentation_params.items():
                if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                    if isinstance(param_range[0], float):
                        aug_params[param_name] = np.random.uniform(param_range[0], param_range[1])
                    else:
                        aug_params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                elif isinstance(param_range, list):
                    aug_params[param_name] = np.random.choice(param_range)
            
            print(f"\nTrial {i+1}/{n_trials}: {aug_params}")
            
            # Update configuration with augmentation parameters
            config = {**self.base_config, 'augmentation': aug_params}
            
            try:
                score = self._train_and_evaluate(config, train_data, val_data,
                                               'val_auc', max_epochs=15)
                
                result = {
                    'trial': i + 1,
                    'augmentation_params': aug_params,
                    'val_auc': score
                }
                results.append(result)
                
                print(f"✓ AUC: {score:.4f}")
                
            except Exception as e:
                print(f"✗ Trial failed: {e}")
                continue
        
        # Find best augmentation parameters
        if results:
            best_result = max(results, key=lambda x: x['val_auc'])
            optimal_params = best_result['augmentation_params']
        else:
            optimal_params = None
        
        return {
            'optimal_augmentation': optimal_params,
            'results': results
        }
    
    def _train_and_evaluate(self, config: Dict, train_data: Tuple, 
                          val_data: Tuple, metric: str, max_epochs: int,
                          minimize: bool = False) -> float:
        """
        Train model with given configuration and evaluate.
        
        Args:
            config: Model configuration
            train_data: Training data
            val_data: Validation data
            metric: Metric to return
            max_epochs: Maximum training epochs
            minimize: Whether to minimize the metric
            
        Returns:
            Evaluation score
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create trainer with configuration
        trainer = self.model_trainer_class(**config)
        
        # Prepare data
        if hasattr(trainer, 'create_datasets'):
            # Use trainer's dataset creation if available
            train_df = pd.DataFrame({
                'image_name': [f'train_{i}' for i in range(len(X_train))],
                'target': y_train
            })
            val_df = pd.DataFrame({
                'image_name': [f'val_{i}' for i in range(len(X_val))],
                'target': y_val
            })
            
            # This is a simplified version - in practice you'd need proper dataset handling
            # For now, use direct arrays
            pass
        
        # Create model
        model = trainer.create_model(num_classes=1)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 1e-4))
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=config.get('batch_size', 32),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        
        # Extract metric
        if metric == 'val_auc':
            score = max(history.history['val_auc'])
        elif metric == 'val_accuracy':
            score = max(history.history['val_accuracy'])
        elif metric == 'val_loss':
            score = min(history.history['val_loss'])
        else:
            # Default to AUC
            score = max(history.history.get('val_auc', [0]))
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
        
        return score
    
    def _sample_params(self, param_distributions: Dict) -> Dict:
        """Sample parameters from distributions."""
        params = {}
        
        for param_name, distribution in param_distributions.items():
            if distribution['type'] == 'uniform':
                params[param_name] = np.random.uniform(
                    distribution['low'], 
                    distribution['high']
                )
            elif distribution['type'] == 'log_uniform':
                params[param_name] = np.random.lognormal(
                    np.log(distribution['low']),
                    np.log(distribution['high'] / distribution['low'])
                )
            elif distribution['type'] == 'choice':
                params[param_name] = np.random.choice(distribution['choices'])
            elif distribution['type'] == 'int_uniform':
                params[param_name] = np.random.randint(
                    distribution['low'],
                    distribution['high'] + 1
                )
        
        return params
    
    def _plot_lr_range_test(self, results: List[Dict]):
        """Plot learning rate range test results."""
        lrs = [r['learning_rate'] for r in results]
        losses = [r['val_loss'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Validation Loss')
        plt.title('Learning Rate Range Test')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.results_dir / 'lr_range_test.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Learning rate plot saved to {plot_path}")
    
    def _save_results(self, results: List[Dict], method: str):
        """Save optimization results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{method}_results_{timestamp}.json"
        
        results_data = {
            'method': method,
            'timestamp': timestamp,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'results': results
        }
        
        with open(self.results_dir / filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"✓ Results saved to {filename}")


def create_default_param_grids():
    """Create default parameter grids for optimization."""
    
    # Grid search parameters
    grid_search_params = {
        'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'batch_size': [8, 16, 32],
        'epochs': [50, 100]
    }
    
    # Random search distributions
    random_search_params = {
        'learning_rate': {
            'type': 'log_uniform',
            'low': 1e-6,
            'high': 1e-2
        },
        'batch_size': {
            'type': 'choice',
            'choices': [8, 16, 32, 64]
        },
        'dropout_rate': {
            'type': 'uniform',
            'low': 0.1,
            'high': 0.5
        }
    }
    
    # Bayesian optimization space
    bayesian_space = {
        'learning_rate': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-2,
            'log': True
        },
        'batch_size': {
            'type': 'categorical',
            'choices': [8, 16, 32, 64]
        },
        'dropout_rate': {
            'type': 'float',
            'low': 0.1,
            'high': 0.5
        }
    }
    
    # Augmentation parameters
    augmentation_params = {
        'rotation_range': (0, 30),
        'zoom_range': (0.0, 0.2),
        'brightness_range': (0.8, 1.2),
        'horizontal_flip': [True, False],
        'vertical_flip': [True, False]
    }
    
    return {
        'grid_search': grid_search_params,
        'random_search': random_search_params,
        'bayesian_space': bayesian_space,
        'augmentation': augmentation_params
    }


def run_complete_optimization(model_trainer_class, train_data: Tuple,
                            val_data: Tuple, optimization_methods: List[str] = None):
    """
    Run complete hyperparameter optimization pipeline.
    
    Args:
        model_trainer_class: Model trainer class
        train_data: Training data tuple
        val_data: Validation data tuple
        optimization_methods: List of methods to run
        
    Returns:
        Combined optimization results
    """
    if optimization_methods is None:
        optimization_methods = ['lr_range_test', 'batch_size', 'random_search']
    
    optimizer = HyperparameterOptimizer(model_trainer_class)
    param_grids = create_default_param_grids()
    
    results = {}
    
    # Learning rate range test
    if 'lr_range_test' in optimization_methods:
        print("\n" + "="*50)
        print("LEARNING RATE RANGE TEST")
        print("="*50)
        lr_results = optimizer.learning_rate_range_test(train_data, val_data)
        results['lr_range_test'] = lr_results
    
    # Batch size optimization
    if 'batch_size' in optimization_methods:
        print("\n" + "="*50)
        print("BATCH SIZE OPTIMIZATION")
        print("="*50)
        batch_results = optimizer.batch_size_optimization(train_data, val_data)
        results['batch_size'] = batch_results
    
    # Random search
    if 'random_search' in optimization_methods:
        print("\n" + "="*50)
        print("RANDOM SEARCH OPTIMIZATION")
        print("="*50)
        random_results = optimizer.random_search(
            param_grids['random_search'], 
            n_trials=20,
            train_data=train_data,
            val_data=val_data
        )
        results['random_search'] = random_results
    
    # Grid search (optional, time-consuming)
    if 'grid_search' in optimization_methods:
        print("\n" + "="*50)
        print("GRID SEARCH OPTIMIZATION")
        print("="*50)
        grid_results = optimizer.grid_search(
            param_grids['grid_search'],
            train_data=train_data,
            val_data=val_data
        )
        results['grid_search'] = grid_results
    
    # Bayesian optimization (if optuna available)
    if 'bayesian' in optimization_methods:
        try:
            print("\n" + "="*50)
            print("BAYESIAN OPTIMIZATION")
            print("="*50)
            bayesian_results = optimizer.bayesian_optimization(
                param_grids['bayesian_space'],
                n_trials=30,
                train_data=train_data,
                val_data=val_data
            )
            results['bayesian'] = bayesian_results
        except ImportError:
            print("Optuna not available, skipping Bayesian optimization")
    
    return results