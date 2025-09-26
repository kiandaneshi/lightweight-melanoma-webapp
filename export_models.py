"""
Export trained models to TensorFlow Lite and TensorFlow.js formats for deployment.
Includes quantization and optimization for mobile/web inference.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

try:
    import tensorflow as tf
    import tensorflowjs as tfjs
    import numpy as np
    import pandas as pd
    TF_AVAILABLE = True
except ImportError as e:
    print(f"ML dependencies not available: {e}")
    TF_AVAILABLE = False

from config import Config
from preprocess_data import DataPreprocessor

class ModelExporter:
    def __init__(self):
        self.models_dir = Path(Config.MODELS_DIR)
        self.export_dir = Path(Config.EXPORT_DIR)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different formats
        (self.export_dir / "tflite").mkdir(exist_ok=True)
        (self.export_dir / "tfjs").mkdir(exist_ok=True)
        (self.export_dir / "saved_model").mkdir(exist_ok=True)

    def find_trained_models(self):
        """Find all trained models ready for export."""
        models = []
        
        # Look for mobile models (primary deployment target)
        for model_dir in self.models_dir.glob("mobile_*"):
            best_model = model_dir / "best_mobile_melanoma.h5"
            final_model = model_dir / "final_mobile_melanoma.h5"
            
            if best_model.exists():
                models.append(("mobile_best", str(best_model)))
            if final_model.exists():
                models.append(("mobile_final", str(final_model)))
        
        # Look for benchmark models
        for model_dir in self.models_dir.glob("benchmark_*"):
            best_model = model_dir / "best_benchmark_efficientnet.h5"
            if best_model.exists():
                models.append(("benchmark", str(best_model)))
        
        return models

    def create_representative_dataset(self, num_samples=100):
        """
        Create representative dataset for quantization calibration.
        This is used for post-training quantization to optimize the model.
        """
        print("Creating representative dataset for quantization...")
        
        preprocessor = DataPreprocessor(input_size=(224, 224))
        
        # Load some training data for calibration
        splits_dir = Path(Config.DATA_DIR) / "processed" / "splits"
        train_df = pd.read_csv(splits_dir / "train.csv")
        
        # Sample representative data
        sample_df = train_df.sample(n=min(num_samples, len(train_df)), random_state=42)
        
        def representative_data_gen():
            for idx, row in sample_df.iterrows():
                image_name = row['image_name']
                image_path = preprocessor.find_image_path(image_name)
                
                if image_path:
                    image = preprocessor.load_and_preprocess_image(
                        image_path, 
                        transforms=preprocessor.val_transforms
                    )
                    if image is not None:
                        # Add batch dimension and convert to float32
                        yield [np.expand_dims(image, axis=0).astype(np.float32)]
        
        return representative_data_gen

    def export_to_tflite(self, model_path, model_name, quantize=True):
        """Export model to TensorFlow Lite format with optional quantization."""
        print(f"Exporting {model_name} to TensorFlow Lite...")
        
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        quant_path = None
        tflite_quant_model = None
        
        if quantize:
            print("Applying post-training quantization...")
            
            # Use representative dataset for quantization
            representative_data = self.create_representative_dataset()
            converter.representative_dataset = representative_data
            
            # Enable different quantization modes
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Export quantized model
            tflite_quant_model = converter.convert()
            
            quant_path = self.export_dir / "tflite" / f"{model_name}_quantized.tflite"
            with open(quant_path, "wb") as f:
                f.write(tflite_quant_model)
            
            print(f"Quantized TFLite model saved to: {quant_path}")
            
            # Get model sizes
            original_size = len(tflite_quant_model) / (1024 * 1024)
            print(f"Quantized model size: {original_size:.2f} MB")
        
        # Also export float32 version (higher accuracy)
        converter_float = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_float.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter_float.convert()
        
        float_path = self.export_dir / "tflite" / f"{model_name}_float32.tflite"
        with open(float_path, "wb") as f:
            f.write(tflite_model)
        
        print(f"Float32 TFLite model saved to: {float_path}")
        
        float_size = len(tflite_model) / (1024 * 1024)
        print(f"Float32 model size: {float_size:.2f} MB")
        
        return quant_path if quantize else float_path, tflite_quant_model if quantize else tflite_model

    def export_to_tfjs(self, model_path, model_name, quantize_uint8=True):
        """Export model to TensorFlow.js format."""
        print(f"Exporting {model_name} to TensorFlow.js...")
        
        # Create output directory
        tfjs_dir = self.export_dir / "tfjs" / model_name
        tfjs_dir.mkdir(exist_ok=True)
        
        # Export to TensorFlow.js
        if quantize_uint8:
            print("Applying uint8 quantization for web deployment...")
            tfjs.converters.convert_tf_keras_model(
                model_path,
                str(tfjs_dir),
                quantization_bytes=1,  # uint8 quantization
                optimize=True,
                weight_shard_size_bytes=1024*1024*4  # 4MB shards for web
            )
        else:
            tfjs.converters.convert_tf_keras_model(
                model_path,
                str(tfjs_dir),
                optimize=True,
                weight_shard_size_bytes=1024*1024*4
            )
        
        print(f"TensorFlow.js model saved to: {tfjs_dir}")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in tfjs_dir.rglob('*') if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        print(f"Total TFJS model size: {total_size_mb:.2f} MB")
        
        # Create model info JSON
        model_info = {
            "model_name": model_name,
            "export_date": datetime.now().isoformat(),
            "input_shape": [224, 224, 3],
            "output_shape": [1],
            "quantized": quantize_uint8,
            "size_mb": round(total_size_mb, 2),
            "preprocessing": {
                "resize": [224, 224],
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            }
        }
        
        with open(tfjs_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        return tfjs_dir

    def export_to_saved_model(self, model_path, model_name):
        """Export to TensorFlow SavedModel format."""
        print(f"Exporting {model_name} to SavedModel format...")
        
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        
        # Save as SavedModel
        saved_model_dir = self.export_dir / "saved_model" / model_name
        model.save(str(saved_model_dir), save_format="tf")
        
        print(f"SavedModel saved to: {saved_model_dir}")
        return saved_model_dir

    def test_tflite_model(self, tflite_path, test_image=None):
        """Test TensorFlow Lite model inference."""
        print(f"Testing TFLite model: {tflite_path}")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        # Test with dummy data if no test image provided
        if test_image is None:
            input_shape = input_details[0]['shape']
            if input_details[0]['dtype'] == np.int8:
                test_image = np.random.randint(-128, 127, input_shape, dtype=np.int8)
            else:
                test_image = np.random.random(input_shape).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_image)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"Test inference successful!")
        print(f"Output: {output_data}")
        
        return True

    def create_deployment_package(self, model_name):
        """Create complete deployment package with all formats."""
        print(f"Creating deployment package for {model_name}...")
        
        package_dir = self.export_dir / f"{model_name}_deployment_package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy TensorFlow.js model
        tfjs_src = self.export_dir / "tfjs" / model_name
        tfjs_dst = package_dir / "tfjs"
        if tfjs_src.exists():
            shutil.copytree(tfjs_src, tfjs_dst, dirs_exist_ok=True)
        
        # Copy TFLite models
        tflite_dir = package_dir / "tflite"
        tflite_dir.mkdir(exist_ok=True)
        
        for tflite_file in (self.export_dir / "tflite").glob(f"{model_name}*"):
            shutil.copy2(tflite_file, tflite_dir)
        
        # Create README for deployment
        readme_content = f"""# {model_name.title()} Deployment Package

This package contains the exported model in multiple formats for different deployment scenarios.

## Contents

### TensorFlow.js (`tfjs/`)
- **Use case**: Web browsers, Progressive Web Apps
- **Files**: model.json, *.bin files
- **Loading**: Use `tf.loadLayersModel('path/to/model.json')`

### TensorFlow Lite (`tflite/`)
- **Use case**: Mobile apps (Android/iOS), edge devices
- **Quantized version**: Smaller size, faster inference, slightly lower accuracy
- **Float32 version**: Larger size, slower inference, higher accuracy

## Model Information
- **Input**: 224x224x3 RGB image
- **Output**: Single probability score (0-1)
- **Preprocessing**: 
  - Resize to 224x224
  - Normalize with ImageNet means/stds
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

## Usage Examples

### TensorFlow.js
```javascript
// Load model
const model = await tf.loadLayersModel('tfjs/model.json');

// Preprocess image
const tensor = tf.browser.fromPixels(imageElement)
  .resizeNearestNeighbor([224, 224])
  .div(255.0)
  .sub([0.485, 0.456, 0.406])
  .div([0.229, 0.224, 0.225])
  .expandDims(0);

// Predict
const prediction = model.predict(tensor);
const probability = await prediction.data();
```

This deployment package provides everything needed to integrate the melanoma classification model into web and mobile applications.
"""
        
        with open(package_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        print(f"Deployment package created: {package_dir}")
        return package_dir

if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("TensorFlow dependencies not available. Install with: pip install tensorflow tensorflowjs")
        exit(1)
        
    exporter = ModelExporter()
    models = exporter.find_trained_models()
    
    if not models:
        print("No trained models found. Train models first using the training scripts.")
        exit(1)
    
    for model_name, model_path in models:
        print(f"Exporting {model_name}...")
        
        # Export to TensorFlow.js (primary deployment target)
        tfjs_dir = exporter.export_to_tfjs(model_path, model_name)
        
        # Export to TFLite for mobile
        tflite_path, _ = exporter.export_to_tflite(model_path, model_name)
        
        # Test the TFLite model
        if tflite_path:
            exporter.test_tflite_model(tflite_path)
        
        # Create deployment package
        exporter.create_deployment_package(model_name)
    
    print("Model export completed!")
