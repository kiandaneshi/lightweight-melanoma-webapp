#!/usr/bin/env python3
"""
Complete DermAI-Melanoma Training Pipeline

Orchestrates the full machine learning pipeline from data download
through model training, evaluation, and export for deployment.

This script provides a complete, reproducible workflow for:
1. Dataset download and preparation
2. Data preprocessing with patient-level splitting
3. Training benchmark and mobile models
4. Comprehensive model evaluation
5. Model export for TensorFlow.js and TensorFlow Lite
6. Grad-CAM visualization generation

Usage:
    python training/run_full_pipeline.py [--steps STEPS]
    
Options:
    --steps: Comma-separated list of steps to run
             (download,preprocess,train_benchmark,train_mobile,evaluate,export,gradcam)
             Default: all steps
    --skip-download: Skip dataset download if already available
    --mobile-only: Train only mobile model (faster for testing)
    --export-only: Only run model export (assumes models exist)
"""

import argparse
import sys
import traceback
from pathlib import Path
from datetime import datetime
import logging

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
import download_dataset
import preprocess_data
import train_benchmark
import train_mobile
import evaluate_models
import export_models
import grad_cam_utils

class PipelineRunner:
    """Orchestrates the complete training pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now()
        self.logger = self.setup_logging()
        
        # Initialize TensorFlow and configuration
        Config.setup_tensorflow()
        Config.setup_logging()
        
    def setup_logging(self):
        """Setup pipeline logging."""
        log_dir = Config.PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('DermAI-Pipeline')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_file = log_dir / f"pipeline_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger

    def print_banner(self):
        """Print pipeline banner."""
        banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          DermAI-Melanoma Training Pipeline                    ║
║                                                                               ║
║  Complete machine learning pipeline for melanoma classification               ║
║  Research demonstration - Not for clinical use                               ║
║                                                                               ║
║  Features:                                                                    ║
║  • Patient-level data splitting to prevent leakage                          ║
║  • Benchmark model (EfficientNet) for maximum performance                   ║
║  • Mobile model (MobileNetV3-Small) optimized for deployment               ║
║  • Comprehensive evaluation with AUROC, PR-AUC, sensitivity/specificity     ║
║  • TensorFlow.js and TensorFlow Lite export                                 ║
║  • Grad-CAM explainability visualization                                    ║
║                                                                               ║
║  ⚠️  IMPORTANT: This is for research and education only                      ║
║     Not approved for medical use or diagnosis                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        self.logger.info("Starting DermAI-Melanoma training pipeline")

    def run_step(self, step_name, step_function, *args, **kwargs):
        """Run a pipeline step with error handling."""
        self.logger.info(f"Starting step: {step_name}")
        step_start = datetime.now()
        
        try:
            result = step_function(*args, **kwargs)
            step_duration = datetime.now() - step_start
            self.logger.info(f"Completed step: {step_name} in {step_duration}")
            return result
            
        except Exception as e:
            self.logger.error(f"Step {step_name} failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def run_download_dataset(self):
        """Step 1: Download and prepare dataset."""
        if self.args.skip_download:
            self.logger.info("Skipping dataset download (--skip-download)")
            return
            
        print("\n" + "="*80)
        print("STEP 1: DATASET DOWNLOAD AND PREPARATION")
        print("="*80)
        
        self.run_step("Dataset Download", download_dataset.main)

    def run_preprocess_data(self):
        """Step 2: Preprocess data with patient-level splitting."""
        print("\n" + "="*80)
        print("STEP 2: DATA PREPROCESSING")
        print("="*80)
        
        self.run_step("Data Preprocessing", preprocess_data.main)

    def run_train_benchmark(self):
        """Step 3: Train benchmark model."""
        if self.args.mobile_only:
            self.logger.info("Skipping benchmark training (--mobile-only)")
            return
            
        print("\n" + "="*80)
        print("STEP 3: BENCHMARK MODEL TRAINING")
        print("="*80)
        
        self.run_step("Benchmark Training", train_benchmark.main)

    def run_train_mobile(self):
        """Step 4: Train mobile model."""
        print("\n" + "="*80)
        print("STEP 4: MOBILE MODEL TRAINING")
        print("="*80)
        
        self.run_step("Mobile Training", train_mobile.main)

    def run_evaluate_models(self):
        """Step 5: Evaluate trained models."""
        print("\n" + "="*80)
        print("STEP 5: MODEL EVALUATION")
        print("="*80)
        
        self.run_step("Model Evaluation", evaluate_models.main)

    def run_export_models(self):
        """Step 6: Export models for deployment."""
        print("\n" + "="*80)
        print("STEP 6: MODEL EXPORT")
        print("="*80)
        
        self.run_step("Model Export", export_models.main)

    def run_grad_cam(self):
        """Step 7: Generate Grad-CAM visualizations."""
        print("\n" + "="*80)
        print("STEP 7: GRAD-CAM GENERATION")
        print("="*80)
        
        self.run_step("Grad-CAM Generation", grad_cam_utils.main)

    def check_prerequisites(self):
        """Check system prerequisites."""
        self.logger.info("Checking system prerequisites...")
        
        try:
            Config.validate_config()
            self.logger.info("Configuration validation passed")
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
            
        # Check available disk space (rough estimate: 10GB minimum)
        import shutil
        free_space = shutil.disk_usage(Config.PROJECT_ROOT).free / (1024**3)  # GB
        if free_space < 10:
            self.logger.warning(f"Low disk space: {free_space:.1f}GB available")
        
        # Check if we're in the right directory
        if not (Config.PROJECT_ROOT / "training").exists():
            self.logger.error("Training directory not found. Please run from project root.")
            return False
            
        return True

    def print_summary(self):
        """Print pipeline completion summary."""
        total_duration = datetime.now() - self.start_time
        
        summary = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           PIPELINE COMPLETED SUCCESSFULLY                     ║
║                                                                               ║
║  Total Duration: {str(total_duration).split('.')[0]:<52} ║
║  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S'):<62} ║
║  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<60} ║
║                                                                               ║
║  Output Directories:                                                          ║
║  • Models: {str(Config.MODELS_DIR):<62} ║
║  • Results: {str(Config.RESULTS_DIR):<61} ║
║  • Exports: {str(Config.EXPORT_DIR):<61} ║
║                                                                               ║
║  Next Steps:                                                                  ║
║  1. Review evaluation results in the results directory                       ║
║  2. Copy TensorFlow.js model to client/public/models/                       ║
║  3. Test the Progressive Web App                                             ║
║  4. Read the documentation in docs/                                          ║
║                                                                               ║
║  ⚠️  Remember: This is for research and education only                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """
        print(summary)
        self.logger.info("Pipeline completed successfully")

    def run_pipeline(self):
        """Run the complete pipeline."""
        if not self.check_prerequisites():
            sys.exit(1)
            
        self.print_banner()
        
        # Define available steps
        all_steps = [
            ('download', self.run_download_dataset),
            ('preprocess', self.run_preprocess_data),
            ('train_benchmark', self.run_train_benchmark),
            ('train_mobile', self.run_train_mobile),
            ('evaluate', self.run_evaluate_models),
            ('export', self.run_export_models),
            ('gradcam', self.run_grad_cam)
        ]
        
        # Determine which steps to run
        if self.args.steps:
            requested_steps = [s.strip() for s in self.args.steps.split(',')]
            steps_to_run = [(name, func) for name, func in all_steps if name in requested_steps]
        else:
            steps_to_run = all_steps
            
        if self.args.export_only:
            steps_to_run = [('export', self.run_export_models)]
        
        self.logger.info(f"Running steps: {[name for name, _ in steps_to_run]}")
        
        # Run selected steps
        try:
            for step_name, step_function in steps_to_run:
                step_function()
                
            self.print_summary()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            print(f"\n❌ Pipeline failed at step: {step_name}")
            print(f"Error: {str(e)}")
            print("\nCheck the log file for detailed error information.")
            sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='DermAI-Melanoma Complete Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python training/run_full_pipeline.py
  
  # Run only specific steps
  python training/run_full_pipeline.py --steps preprocess,train_mobile,export
  
  # Quick mobile-only training for testing
  python training/run_full_pipeline.py --mobile-only --skip-download
  
  # Export models only (assumes training completed)
  python training/run_full_pipeline.py --export-only
        """
    )
    
    parser.add_argument(
        '--steps',
        type=str,
        help='Comma-separated list of steps to run (download,preprocess,train_benchmark,train_mobile,evaluate,export,gradcam)'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download if already available'
    )
    
    parser.add_argument(
        '--mobile-only',
        action='store_true',
        help='Train only mobile model (faster for testing)'
    )
    
    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Only run model export (assumes models exist)'
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    runner = PipelineRunner(args)
    runner.run_pipeline()

if __name__ == "__main__":
    main()
