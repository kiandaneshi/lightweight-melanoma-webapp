"""
Download and prepare SIIM-ISIC 2020 Melanoma Classification dataset.
Uses Kaggle API for dataset access with proper authentication.
"""

import os
import zipfile
import pandas as pd
from pathlib import Path
import kaggle
from tqdm import tqdm
import shutil
from config import Config

def setup_kaggle_credentials():
    """
    Ensure Kaggle API credentials are properly configured.
    Requires kaggle.json in ~/.kaggle/ or KAGGLE_USERNAME/KAGGLE_KEY env vars.
    """
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    if not kaggle_file.exists():
        print("Kaggle credentials not found. Please ensure one of the following:")
        print("1. Place kaggle.json in ~/.kaggle/")
        print("2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        
        # Check for environment variables
        if 'KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ:
            kaggle_dir.mkdir(exist_ok=True)
            credentials = {
                "username": os.environ['KAGGLE_USERNAME'],
                "key": os.environ['KAGGLE_KEY']
            }
            import json
            with open(kaggle_file, 'w') as f:
                json.dump(credentials, f)
            kaggle_file.chmod(0o600)
            print("Kaggle credentials configured from environment variables.")
        else:
            raise Exception("Kaggle API credentials required")

def download_siim_isic_dataset():
    """
    Download the SIIM-ISIC 2020 Melanoma Classification dataset.
    """
    print("Setting up Kaggle API credentials...")
    setup_kaggle_credentials()
    
    dataset_name = "cdeotte/melanoma-384x384"
    data_dir = Path(Config.DATA_DIR)
    raw_dir = data_dir / "raw"
    
    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading SIIM-ISIC dataset to {raw_dir}...")
    
    try:
        # Download dataset using Kaggle API
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(raw_dir),
            unzip=True,
            quiet=False
        )
        print("Dataset downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Fallback: Trying main competition dataset...")
        
        # Fallback to main competition dataset
        competition_name = "siim-isic-melanoma-classification"
        kaggle.api.competition_download_files(
            competition_name,
            path=str(raw_dir),
            quiet=False
        )
        
        # Extract files
        for zip_file in raw_dir.glob("*.zip"):
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            zip_file.unlink()  # Remove zip file after extraction

def prepare_dataset_structure():
    """
    Organize the downloaded dataset into a structured format.
    """
    data_dir = Path(Config.DATA_DIR)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("Organizing dataset structure...")
    
    # Find CSV files
    train_csv = None
    test_csv = None
    
    for csv_file in raw_dir.glob("*.csv"):
        if "train" in csv_file.name.lower():
            train_csv = csv_file
        elif "test" in csv_file.name.lower():
            test_csv = csv_file
    
    if not train_csv:
        # Look for main dataset CSV
        csv_files = list(raw_dir.glob("*.csv"))
        if csv_files:
            train_csv = csv_files[0]
    
    if train_csv:
        print(f"Found training data: {train_csv}")
        # Copy to processed directory
        shutil.copy2(train_csv, processed_dir / "train_metadata.csv")
        
        # Load and examine the data
        df = pd.read_csv(train_csv)
        print(f"Training samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        if 'target' in df.columns:
            print(f"Melanoma cases: {df['target'].sum()}")
            print(f"Benign cases: {(df['target'] == 0).sum()}")
            print(f"Class balance: {df['target'].mean():.3f}")
    
    if test_csv:
        print(f"Found test data: {test_csv}")
        shutil.copy2(test_csv, processed_dir / "test_metadata.csv")
    
    # Organize image directories
    image_dirs = []
    for img_dir in raw_dir.iterdir():
        if img_dir.is_dir() and any(img_dir.glob("*.jpg")):
            image_dirs.append(img_dir)
    
    print(f"Found {len(image_dirs)} image directories")
    for img_dir in image_dirs:
        target_dir = processed_dir / img_dir.name
        if not target_dir.exists():
            print(f"Moving {img_dir} -> {target_dir}")
            shutil.move(str(img_dir), str(target_dir))

def download_ph2_dataset():
    """
    Download the PH2 dataset for additional validation.
    This is a smaller dataset that can be used for external validation.
    """
    print("PH2 dataset download not implemented in this demo.")
    print("For real implementation, you would download from:")
    print("https://www.fc.up.pt/addi/ph2%20database.html")
    
    # In a real implementation, you would:
    # 1. Download PH2 dataset
    # 2. Extract dermoscopy images
    # 3. Parse ground truth annotations
    # 4. Create CSV metadata file

def verify_dataset():
    """
    Verify that the dataset was downloaded and organized correctly.
    """
    data_dir = Path(Config.DATA_DIR)
    processed_dir = data_dir / "processed"
    
    print("\nVerifying dataset...")
    
    # Check for metadata files
    train_csv = processed_dir / "train_metadata.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        print(f"✓ Training metadata: {len(df)} samples")
        
        # Check for required columns
        required_cols = ['image_name', 'target']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠ Missing columns: {missing_cols}")
        else:
            print("✓ All required columns present")
    else:
        print("✗ Training metadata not found")
    
    # Check for image directories
    image_count = 0
    for img_dir in processed_dir.iterdir():
        if img_dir.is_dir():
            jpg_count = len(list(img_dir.glob("*.jpg")))
            if jpg_count > 0:
                print(f"✓ {img_dir.name}: {jpg_count} images")
                image_count += jpg_count
    
    print(f"Total images found: {image_count}")
    
    if image_count == 0:
        print("⚠ No images found. Check dataset download.")
    
    return train_csv.exists() and image_count > 0

def main():
    """
    Main function to download and prepare the dataset.
    """
    print("DermAI Dataset Download and Preparation")
    print("=" * 50)
    
    try:
        # Download main dataset
        download_siim_isic_dataset()
        
        # Organize dataset structure
        prepare_dataset_structure()
        
        # Optionally download PH2 for validation
        # download_ph2_dataset()
        
        # Verify everything is ready
        if verify_dataset():
            print("\n✓ Dataset preparation completed successfully!")
            print("Ready for preprocessing and training.")
        else:
            print("\n✗ Dataset preparation failed. Please check the logs.")
            
    except Exception as e:
        print(f"\nError during dataset preparation: {e}")
        print("Please check your Kaggle API credentials and internet connection.")

if __name__ == "__main__":
    main()
