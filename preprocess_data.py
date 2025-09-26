"""
Data preprocessing pipeline for melanoma classification.
Handles image preprocessing, augmentation, and patient-level data splitting.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image, ImageEnhance
import albumentations as A
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config

class DataPreprocessor:
    def __init__(self, input_size=(224, 224)):
        self.input_size = input_size
        self.data_dir = Path(Config.DATA_DIR)
        self.processed_dir = self.data_dir / "processed"
        
        # Define augmentation pipeline
        self.train_transforms = A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_metadata(self):
        """Load and clean the metadata CSV file."""
        metadata_file = self.processed_dir / "train_metadata.csv"
        
        if not metadata_file.exists():
            raise FileNotFoundError("Train metadata not found. Run download_dataset.py first.")
        
        df = pd.read_csv(metadata_file)
        print(f"Loaded {len(df)} samples from metadata")
        
        # Handle different CSV formats
        if 'image_name' not in df.columns and 'image_id' in df.columns:
            df['image_name'] = df['image_id']
        
        # Ensure target column exists
        if 'target' not in df.columns:
            if 'diagnosis' in df.columns:
                # Convert diagnosis to binary target
                melanoma_diagnoses = ['melanoma', 'malignant melanoma']
                df['target'] = df['diagnosis'].str.lower().isin(melanoma_diagnoses).astype(int)
            else:
                raise ValueError("No target or diagnosis column found")
        
        # Add patient ID if not present (critical for preventing data leakage)
        if 'patient_id' not in df.columns:
            print("⚠ No patient_id column found. Creating synthetic patient IDs.")
            print("⚠ This may lead to data leakage in real scenarios.")
            # Create synthetic patient IDs based on image patterns
            df['patient_id'] = df['image_name'].str.extract(r'(ISIC_\d{7})')[0]
            df['patient_id'] = df['patient_id'].fillna(df.index.astype(str))
        
        # Clean and validate data
        df = df.dropna(subset=['image_name', 'target'])
        df['target'] = df['target'].astype(int)
        
        print(f"After cleaning: {len(df)} samples")
        print(f"Melanoma cases: {df['target'].sum()}")
        print(f"Benign cases: {(df['target'] == 0).sum()}")
        print(f"Unique patients: {df['patient_id'].nunique()}")
        
        return df

    def verify_images(self, df):
        """Verify that all images exist and are readable."""
        print("Verifying image files...")
        
        # Find image directory
        image_dirs = [d for d in self.processed_dir.iterdir() if d.is_dir()]
        image_extensions = ['.jpg', '.jpeg', '.png']
        
        valid_indices = []
        missing_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_name = row['image_name']
            image_found = False
            
            # Search for image in all subdirectories
            for img_dir in image_dirs:
                for ext in image_extensions:
                    image_path = img_dir / f"{image_name}{ext}"
                    if image_path.exists():
                        try:
                            # Test if image can be loaded
                            with Image.open(image_path) as img:
                                img.verify()
                            valid_indices.append(idx)
                            image_found = True
                            break
                        except Exception:
                            continue
                if image_found:
                    break
            
            if not image_found:
                missing_count += 1
        
        print(f"Found {len(valid_indices)} valid images")
        print(f"Missing {missing_count} images")
        
        return df.loc[valid_indices].reset_index(drop=True)

    def patient_level_split(self, df, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data by patient ID to prevent data leakage.
        This is critical for medical datasets where the same patient
        may have multiple images.
        """
        print("Performing patient-level data splitting...")
        
        # Get unique patients with their labels
        patient_df = df.groupby('patient_id').agg({
            'target': 'max',  # If any image from patient is melanoma, patient is positive
            'image_name': 'count'
        }).rename(columns={'image_name': 'num_images'}).reset_index()
        
        print(f"Patient-level statistics:")
        print(f"Total patients: {len(patient_df)}")
        print(f"Melanoma patients: {patient_df['target'].sum()}")
        print(f"Benign patients: {(patient_df['target'] == 0).sum()}")
        
        # Split patients into train/val/test
        train_patients, test_patients = train_test_split(
            patient_df['patient_id'],
            test_size=test_size,
            stratify=patient_df['target'],
            random_state=random_state
        )
        
        train_patients, val_patients = train_test_split(
            train_patients,
            test_size=val_size / (1 - test_size),
            stratify=patient_df[patient_df['patient_id'].isin(train_patients)]['target'],
            random_state=random_state
        )
        
        # Create image-level splits
        train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
        val_df = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
        test_df = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)
        
        print(f"\nDataset splits:")
        print(f"Train: {len(train_df)} images from {len(train_patients)} patients")
        print(f"  - Melanoma: {train_df['target'].sum()}")
        print(f"  - Benign: {(train_df['target'] == 0).sum()}")
        print(f"Val: {len(val_df)} images from {len(val_patients)} patients")
        print(f"  - Melanoma: {val_df['target'].sum()}")
        print(f"  - Benign: {(val_df['target'] == 0).sum()}")
        print(f"Test: {len(test_df)} images from {len(test_patients)} patients")
        print(f"  - Melanoma: {test_df['target'].sum()}")
        print(f"  - Benign: {(test_df['target'] == 0).sum()}")
        
        return train_df, val_df, test_df

    def load_and_preprocess_image(self, image_path, transforms=None):
        """Load and preprocess a single image."""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if transforms:
                transformed = transforms(image=image)
                image = transformed['image']
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def create_tf_dataset(self, df, batch_size=32, shuffle=True, transforms=None):
        """Create TensorFlow dataset from dataframe."""
        def generator():
            indices = list(range(len(df)))
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                row = df.iloc[idx]
                image_name = row['image_name']
                target = row['target']
                
                # Find image file
                image_path = self.find_image_path(image_name)
                if image_path is None:
                    continue
                
                # Load and preprocess image
                image = self.load_and_preprocess_image(image_path, transforms)
                if image is None:
                    continue
                
                yield image, target
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.input_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def find_image_path(self, image_name):
        """Find the full path to an image file."""
        image_dirs = [d for d in self.processed_dir.iterdir() if d.is_dir()]
        image_extensions = ['.jpg', '.jpeg', '.png']
        
        for img_dir in image_dirs:
            for ext in image_extensions:
                image_path = img_dir / f"{image_name}{ext}"
                if image_path.exists():
                    return image_path
        
        return None

    def save_splits(self, train_df, val_df, test_df):
        """Save the data splits to CSV files."""
        output_dir = self.processed_dir / "splits"
        output_dir.mkdir(exist_ok=True)
        
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        
        print(f"Saved data splits to {output_dir}")

    def create_data_visualization(self, df):
        """Create visualizations of the dataset."""
        output_dir = self.processed_dir / "visualizations"
        output_dir.mkdir(exist_ok=True)
        
        # Class distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        df['target'].value_counts().plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Class (0=Benign, 1=Melanoma)')
        plt.ylabel('Count')
        
        # Patient distribution
        plt.subplot(1, 3, 2)
        patient_counts = df.groupby('patient_id')['target'].max().value_counts()
        patient_counts.plot(kind='bar')
        plt.title('Patient-level Class Distribution')
        plt.xlabel('Class (0=Benign, 1=Melanoma)')
        plt.ylabel('Number of Patients')
        
        # Images per patient
        plt.subplot(1, 3, 3)
        images_per_patient = df.groupby('patient_id').size()
        plt.hist(images_per_patient, bins=20, alpha=0.7)
        plt.title('Images per Patient Distribution')
        plt.xlabel('Number of Images')
        plt.ylabel('Number of Patients')
        
        plt.tight_layout()
        plt.savefig(output_dir / "data_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved data visualizations to {output_dir}")

def main():
    """Main preprocessing pipeline."""
    print("DermAI Data Preprocessing Pipeline")
    print("=" * 50)
    
    preprocessor = DataPreprocessor(input_size=(224, 224))
    
    try:
        # Load metadata
        df = preprocessor.load_metadata()
        
        # Verify images exist
        df = preprocessor.verify_images(df)
        
        # Patient-level split to prevent data leakage
        train_df, val_df, test_df = preprocessor.patient_level_split(df)
        
        # Save splits
        preprocessor.save_splits(train_df, val_df, test_df)
        
        # Create visualizations
        preprocessor.create_data_visualization(df)
        
        print("\n✓ Data preprocessing completed successfully!")
        print("Ready for model training.")
        
        # Print final statistics
        total_images = len(train_df) + len(val_df) + len(test_df)
        print(f"\nFinal dataset statistics:")
        print(f"Total images: {total_images}")
        print(f"Train: {len(train_df)} ({len(train_df)/total_images:.1%})")
        print(f"Validation: {len(val_df)} ({len(val_df)/total_images:.1%})")
        print(f"Test: {len(test_df)} ({len(test_df)/total_images:.1%})")
        
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()
