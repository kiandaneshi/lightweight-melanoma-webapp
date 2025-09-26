"""
Clinical workflow integration and batch processing for melanoma classification.
Provides tools for healthcare professionals to integrate AI into clinical practice.
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import uuid

try:
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    TF_AVAILABLE = True
except ImportError as e:
    print(f"ML dependencies not available: {e}")
    TF_AVAILABLE = False

from config import Config

class ClinicalWorkflowManager:
    """Clinical workflow integration for melanoma classification."""
    
    def __init__(self, model_path=None, max_workers=4):
        self.model_path = model_path
        self.model = None
        self.max_workers = max_workers
        self.processing_lock = threading.Lock()
        
        # Workflow directories
        self.workflow_dir = Path(Config.MODELS_DIR) / "clinical_workflow"
        self.batch_dir = self.workflow_dir / "batch_processing"
        self.reports_dir = self.workflow_dir / "reports"
        self.queue_dir = self.workflow_dir / "queue"
        
        # Create directories
        for dir_path in [self.workflow_dir, self.batch_dir, self.reports_dir, self.queue_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Clinical workflow initialized: {self.workflow_dir}")

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
        
        print(f"Loading model for clinical workflow: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        return self.model

    def create_patient_record(self, 
                            patient_id: str,
                            demographic_info: Dict,
                            medical_history: Dict,
                            lesion_info: Dict) -> Dict:
        """Create a structured patient record for clinical workflow."""
        
        record = {
            "patient_id": patient_id,
            "record_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "demographic_info": {
                "age": demographic_info.get("age"),
                "gender": demographic_info.get("gender"),
                "skin_type": demographic_info.get("skin_type"),  # Fitzpatrick scale
                "ethnicity": demographic_info.get("ethnicity")
            },
            "medical_history": {
                "personal_melanoma_history": medical_history.get("personal_melanoma_history", False),
                "family_melanoma_history": medical_history.get("family_melanoma_history", False),
                "atypical_nevi": medical_history.get("atypical_nevi", False),
                "immunosuppressed": medical_history.get("immunosuppressed", False),
                "previous_skin_cancers": medical_history.get("previous_skin_cancers", []),
                "medications": medical_history.get("medications", [])
            },
            "lesion_info": {
                "location": lesion_info.get("location"),
                "size_mm": lesion_info.get("size_mm"),
                "duration": lesion_info.get("duration"),
                "changes_observed": lesion_info.get("changes_observed", []),
                "clinical_notes": lesion_info.get("clinical_notes", "")
            },
            "images": [],
            "ai_analysis": {},
            "clinical_assessment": {},
            "follow_up": {},
            "status": "created"
        }
        
        return record

    def analyze_lesion_batch(self, 
                           image_paths: List[str],
                           patient_records: Optional[List[Dict]] = None,
                           generate_report: bool = True) -> Dict:
        """Perform batch analysis of lesion images for clinical workflow."""
        
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        batch_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"Starting batch analysis: {batch_id}")
        print(f"Processing {len(image_paths)} images...")
        
        # Initialize batch results
        batch_results = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_paths),
            "processed_images": 0,
            "failed_images": 0,
            "results": [],
            "summary": {},
            "processing_time": 0
        }
        
        start_time = datetime.now()
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self._process_single_image, img_path, i, patient_records[i] if patient_records else None): 
                (img_path, i) for i, img_path in enumerate(image_paths)
            }
            
            # Collect results
            for future in as_completed(future_to_image):
                img_path, idx = future_to_image[future]
                
                try:
                    result = future.result()
                    batch_results["results"].append(result)
                    batch_results["processed_images"] += 1
                    
                    if batch_results["processed_images"] % 10 == 0:
                        print(f"Processed {batch_results['processed_images']}/{len(image_paths)} images")
                        
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")
                    batch_results["failed_images"] += 1
                    
                    # Add failed result
                    failed_result = {
                        "image_path": str(img_path),
                        "index": idx,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    batch_results["results"].append(failed_result)
        
        # Calculate processing time
        end_time = datetime.now()
        batch_results["processing_time"] = (end_time - start_time).total_seconds()
        
        # Generate summary statistics
        batch_results["summary"] = self._generate_batch_summary(batch_results["results"])
        
        # Save batch results
        batch_file = self.batch_dir / f"batch_{timestamp}_{batch_id[:8]}.json"
        with open(batch_file, 'w') as f:
            # Create a serializable copy (remove non-serializable objects)
            results_copy = self._make_serializable(batch_results)
            json.dump(results_copy, f, indent=2)
        
        print(f"Batch processing completed!")
        print(f"Processed: {batch_results['processed_images']}/{batch_results['total_images']}")
        print(f"Failed: {batch_results['failed_images']}")
        print(f"Processing time: {batch_results['processing_time']:.2f} seconds")
        print(f"Results saved: {batch_file}")
        
        # Generate clinical report
        if generate_report:
            report_file = self._generate_clinical_report(batch_results, batch_id)
            batch_results["report_file"] = str(report_file)
        
        return batch_results

    def _process_single_image(self, image_path: str, index: int, patient_record: Optional[Dict] = None) -> Dict:
        """Process a single image for clinical analysis."""
        
        try:
            # Load and preprocess image
            if isinstance(image_path, str) and Path(image_path).exists():
                image = tf.keras.preprocessing.image.load_img(
                    image_path, target_size=(224, 224)
                )
                image = tf.keras.preprocessing.image.img_to_array(image)
                image = image / 255.0  # Normalize
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Get prediction with thread safety
            with self.processing_lock:
                prediction = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
            
            # Process prediction
            if prediction.shape[1] == 1:
                probability = float(prediction[0][0])
                confidence = abs(probability - 0.5) * 2  # Simple confidence measure
            else:
                probability = float(prediction[0][1])  # Assume melanoma class is index 1
                confidence = float(np.max(prediction[0]))
            
            # Risk classification
            if probability < 0.3:
                risk_level = "LOW"
                risk_color = "green"
                recommendation = "Routine follow-up"
            elif probability < 0.7:
                risk_level = "MODERATE"
                risk_color = "yellow"
                recommendation = "Close monitoring recommended"
            else:
                risk_level = "HIGH"
                risk_color = "red"
                recommendation = "Urgent dermatological evaluation recommended"
            
            # Create result
            result = {
                "image_path": str(image_path),
                "index": index,
                "success": True,
                "analysis": {
                    "melanoma_probability": probability,
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "risk_color": risk_color,
                    "recommendation": recommendation
                },
                "patient_info": patient_record if patient_record else {},
                "timestamp": datetime.now().isoformat(),
                "model_version": str(self.model_path)
            }
            
            # Add clinical context if patient record provided
            if patient_record:
                result["clinical_context"] = self._assess_clinical_context(
                    result["analysis"], patient_record
                )
            
            return result
            
        except Exception as e:
            return {
                "image_path": str(image_path),
                "index": index,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _assess_clinical_context(self, ai_analysis: Dict, patient_record: Dict) -> Dict:
        """Assess clinical context and modify recommendations based on patient history."""
        
        clinical_context = {
            "risk_factors": [],
            "modified_recommendation": ai_analysis["recommendation"],
            "urgency_level": "standard"
        }
        
        # Check risk factors
        medical_history = patient_record.get("medical_history", {})
        demographic = patient_record.get("demographic_info", {})
        
        risk_factors = []
        
        if medical_history.get("personal_melanoma_history"):
            risk_factors.append("Personal history of melanoma")
        
        if medical_history.get("family_melanoma_history"):
            risk_factors.append("Family history of melanoma")
        
        if medical_history.get("atypical_nevi"):
            risk_factors.append("Atypical nevi present")
        
        if medical_history.get("immunosuppressed"):
            risk_factors.append("Immunosuppressed status")
        
        if demographic.get("skin_type") in ["I", "II"]:  # Fair skin types
            risk_factors.append("Fair skin type")
        
        clinical_context["risk_factors"] = risk_factors
        
        # Modify recommendations based on risk factors
        base_probability = ai_analysis["melanoma_probability"]
        
        if len(risk_factors) > 0:
            if base_probability > 0.5 or len(risk_factors) >= 2:
                clinical_context["modified_recommendation"] = "Urgent dermatological evaluation strongly recommended"
                clinical_context["urgency_level"] = "urgent"
            elif base_probability > 0.3:
                clinical_context["modified_recommendation"] = "Expedited dermatological evaluation recommended"
                clinical_context["urgency_level"] = "expedited"
        
        return clinical_context

    def _generate_batch_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics for batch processing."""
        
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {"error": "No successful analyses"}
        
        probabilities = [r["analysis"]["melanoma_probability"] for r in successful_results]
        risk_levels = [r["analysis"]["risk_level"] for r in successful_results]
        
        summary = {
            "total_analyzed": len(successful_results),
            "probability_statistics": {
                "mean": float(np.mean(probabilities)),
                "median": float(np.median(probabilities)),
                "std": float(np.std(probabilities)),
                "min": float(np.min(probabilities)),
                "max": float(np.max(probabilities))
            },
            "risk_distribution": {
                "LOW": risk_levels.count("LOW"),
                "MODERATE": risk_levels.count("MODERATE"),
                "HIGH": risk_levels.count("HIGH")
            },
            "recommendations": {
                "routine_followup": risk_levels.count("LOW"),
                "close_monitoring": risk_levels.count("MODERATE"),
                "urgent_evaluation": risk_levels.count("HIGH")
            },
            "urgent_cases": len([r for r in successful_results if r["analysis"]["risk_level"] == "HIGH"])
        }
        
        return summary

    def _generate_clinical_report(self, batch_results: Dict, batch_id: str) -> Path:
        """Generate a clinical report for the batch analysis."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f"clinical_report_{timestamp}_{batch_id[:8]}.html"
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DermAI-Melanoma Clinical Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .risk-high {{ color: red; font-weight: bold; }}
        .risk-moderate {{ color: orange; font-weight: bold; }}
        .risk-low {{ color: green; }}
        .urgent {{ background-color: #ffe6e6; padding: 10px; border-left: 4px solid red; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .disclaimer {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>DermAI-Melanoma Clinical Analysis Report</h1>
        <p><strong>Report ID:</strong> {batch_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Images Analyzed:</strong> {batch_results['processed_images']}</p>
    </div>
    
    <div class="disclaimer">
        <h3>⚠️ Important Clinical Disclaimer</h3>
        <p><strong>This is a research tool for educational purposes only. NOT FDA approved for clinical diagnosis.</strong></p>
        <p>AI predictions should be used as a supportive tool alongside clinical judgment. Always follow standard clinical protocols and consult with qualified dermatologists for definitive diagnosis and treatment decisions.</p>
    </div>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <p><strong>Processing Time:</strong> {batch_results['processing_time']:.2f} seconds</p>
        <p><strong>Average Melanoma Probability:</strong> {batch_results['summary'].get('probability_statistics', {}).get('mean', 0):.3f}</p>
        
        <h3>Risk Distribution</h3>
        <ul>
            <li class="risk-high">HIGH RISK: {batch_results['summary'].get('risk_distribution', {}).get('HIGH', 0)} cases</li>
            <li class="risk-moderate">MODERATE RISK: {batch_results['summary'].get('risk_distribution', {}).get('MODERATE', 0)} cases</li>
            <li class="risk-low">LOW RISK: {batch_results['summary'].get('risk_distribution', {}).get('LOW', 0)} cases</li>
        </ul>
    </div>
"""
        
        # Add urgent cases section
        urgent_cases = [r for r in batch_results['results'] if r.get('success') and r.get('analysis', {}).get('risk_level') == 'HIGH']
        
        if urgent_cases:
            html_content += """
    <div class="urgent">
        <h2>🚨 Urgent Cases Requiring Immediate Attention</h2>
        <table>
            <tr>
                <th>Image</th>
                <th>Melanoma Probability</th>
                <th>Confidence</th>
                <th>Recommendation</th>
            </tr>
"""
            for case in urgent_cases:
                analysis = case.get('analysis', {})
                html_content += f"""
            <tr>
                <td>{Path(case['image_path']).name}</td>
                <td class="risk-high">{analysis.get('melanoma_probability', 0):.3f}</td>
                <td>{analysis.get('confidence', 0):.3f}</td>
                <td>{analysis.get('recommendation', 'N/A')}</td>
            </tr>
"""
            html_content += "        </table>\\n    </div>"
        
        # Add detailed results table
        html_content += """
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Index</th>
            <th>Image</th>
            <th>Risk Level</th>
            <th>Probability</th>
            <th>Confidence</th>
            <th>Recommendation</th>
        </tr>
"""
        
        for result in batch_results['results']:
            if result.get('success'):
                analysis = result.get('analysis', {})
                risk_class = f"risk-{analysis.get('risk_level', 'low').lower()}"
                
                html_content += f"""
        <tr>
            <td>{result.get('index', 'N/A')}</td>
            <td>{Path(result['image_path']).name}</td>
            <td class="{risk_class}">{analysis.get('risk_level', 'N/A')}</td>
            <td>{analysis.get('melanoma_probability', 0):.3f}</td>
            <td>{analysis.get('confidence', 0):.3f}</td>
            <td>{analysis.get('recommendation', 'N/A')}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ccc;">
        <p><em>Report generated by DermAI-Melanoma Research System</em></p>
        <p><em>For research and educational use only - not for clinical diagnosis</em></p>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"Clinical report generated: {report_file}")
        return report_file

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    def create_workflow_queue(self, priority_cases: List[str] = None) -> Dict:
        """Create a prioritized workflow queue for clinical processing."""
        
        queue_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        queue = {
            "queue_id": queue_id,
            "created": datetime.now().isoformat(),
            "status": "active",
            "priority_cases": priority_cases or [],
            "regular_cases": [],
            "completed_cases": [],
            "failed_cases": [],
            "processing_statistics": {
                "total_queued": 0,
                "completed": 0,
                "failed": 0,
                "processing_time": 0
            }
        }
        
        # Save queue
        queue_file = self.queue_dir / f"queue_{timestamp}_{queue_id[:8]}.json"
        with open(queue_file, 'w') as f:
            json.dump(queue, f, indent=2)
        
        print(f"Workflow queue created: {queue_file}")
        return queue

def run_clinical_batch_processing(image_directory: str, 
                                model_path: str = None,
                                patient_records_csv: str = None,
                                max_workers: int = 4) -> Dict:
    """Main function to run clinical batch processing."""
    
    if not TF_AVAILABLE:
        print("TensorFlow dependencies not available.")
        return {}
    
    print("Starting clinical batch processing workflow...")
    
    # Initialize workflow manager
    workflow = ClinicalWorkflowManager(model_path, max_workers)
    workflow.load_model()
    
    # Get image files
    image_dir = Path(image_directory)
    if not image_dir.exists():
        print(f"Image directory not found: {image_directory}")
        return {}
    
    # Support multiple image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(ext))
        image_paths.extend(image_dir.glob(ext.upper()))
    
    if not image_paths:
        print(f"No images found in {image_directory}")
        return {}
    
    print(f"Found {len(image_paths)} images for processing")
    
    # Load patient records if provided
    patient_records = None
    if patient_records_csv and Path(patient_records_csv).exists():
        print(f"Loading patient records from {patient_records_csv}")
        try:
            df = pd.read_csv(patient_records_csv)
            patient_records = df.to_dict('records')
            print(f"Loaded {len(patient_records)} patient records")
        except Exception as e:
            print(f"Error loading patient records: {e}")
            patient_records = None
    
    # Run batch analysis
    results = workflow.analyze_lesion_batch(
        image_paths=image_paths,
        patient_records=patient_records,
        generate_report=True
    )
    
    print("\\n=== Clinical Batch Processing Summary ===")
    print(f"Total Images: {results['total_images']}")
    print(f"Successfully Processed: {results['processed_images']}")
    print(f"Failed: {results['failed_images']}")
    print(f"Processing Time: {results['processing_time']:.2f} seconds")
    
    if 'summary' in results and results['summary']:
        summary = results['summary']
        print(f"\\nRisk Distribution:")
        print(f"  HIGH RISK: {summary.get('risk_distribution', {}).get('HIGH', 0)} cases")
        print(f"  MODERATE RISK: {summary.get('risk_distribution', {}).get('MODERATE', 0)} cases")
        print(f"  LOW RISK: {summary.get('risk_distribution', {}).get('LOW', 0)} cases")
        
        urgent_cases = summary.get('urgent_cases', 0)
        if urgent_cases > 0:
            print(f"\\n🚨 URGENT: {urgent_cases} cases require immediate dermatological evaluation")
    
    if 'report_file' in results:
        print(f"\\nClinical report generated: {results['report_file']}")
    
    return results

if __name__ == "__main__":
    # Example usage
    results = run_clinical_batch_processing(
        image_directory="data/test_images",
        max_workers=4
    )
    
    if results:
        print("Clinical workflow processing completed successfully!")