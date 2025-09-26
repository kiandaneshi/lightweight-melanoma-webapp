# Installation Guide

## System Requirements

### For Training Pipeline
- **Python**: 3.9 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible GPU recommended (optional but speeds up training significantly)
- **Storage**: 10GB free space for dataset and models

### For Web Application
- **Node.js**: 18.0 or higher
- **NPM**: 9.0 or higher
- **RAM**: 4GB minimum
- **Browser**: Modern browser with WebGL support

## Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/dermai-melanoma.git
cd dermai-melanoma
```

### 2. Training Pipeline Setup
```bash
cd training

# Install Python dependencies
pip install -r requirements.txt

# Download dataset (requires Kaggle API setup)
python download_dataset.py

# Run complete training pipeline
python run_full_pipeline.py
```

### 3. Web Application Setup
```bash
cd web_app

# Install Node.js dependencies
npm install

# Start development server
npm run dev
```

Visit `http://localhost:5000` in your browser.

## Detailed Installation

### Python Environment Setup

#### Using Conda (Recommended)
```bash
# Create new environment
conda create -n dermai python=3.9
conda activate dermai

# Install PyTorch with CUDA support (if you have GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
cd training
pip install -r requirements.txt
```

#### Using pip + venv
```bash
# Create virtual environment
python -m venv dermai_env
source dermai_env/bin/activate  # On Windows: dermai_env\Scripts\activate

# Install dependencies
cd training
pip install -r requirements.txt
```

### Dataset Setup

#### Kaggle API Configuration
1. Create Kaggle account and generate API token
2. Place `kaggle.json` in `~/.kaggle/` directory
3. Run dataset download:
```bash
cd training
python download_dataset.py
```

#### Manual Dataset Download
1. Visit [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification)
2. Download dataset manually
3. Extract to `training/data/` directory

### Web Application Advanced Setup

#### Environment Variables (Optional)
Create `.env` file in `web_app/` directory:
```bash
# Database configuration (if using PostgreSQL)
DATABASE_URL=postgresql://username:password@localhost/dermai

# Development settings
NODE_ENV=development
```

#### Production Build
```bash
cd web_app
npm run build
npm run start
```

## Troubleshooting

### Common Issues

**GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

**Memory Issues During Training**
- Reduce batch size in `training/config.py`
- Use gradient checkpointing
- Train mobile model instead of benchmark model

**Web Application Not Loading**
- Check Node.js version: `node --version`
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`

**Model Loading Issues**
- Verify model files in `models/` directory
- Check browser console for WebGL errors
- Ensure browser supports TensorFlow.js

### Performance Optimization

**Training Acceleration**
- Use GPU if available
- Increase batch size (if memory allows)
- Use mixed precision training (enabled by default)

**Inference Optimization**
- Use mobile model for faster inference
- Enable service worker for caching
- Use WebAssembly backend for TensorFlow.js

## Development Setup

### Code Style
```bash
# Python formatting
pip install black isort
black training/
isort training/

# TypeScript/JavaScript formatting
cd web_app
npm install -D prettier
npx prettier --write .
```

### Testing
```bash
# Python tests
cd training
python -m pytest

# Web application tests
cd web_app
npm run test
```

## Hardware Recommendations

### Minimum Configuration
- **CPU**: 4-core processor
- **RAM**: 8GB
- **Storage**: 50GB available space
- **GPU**: Not required but recommended

### Recommended Configuration
- **CPU**: 8-core processor or better
- **RAM**: 16GB or more
- **Storage**: SSD with 100GB+ available space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)

### Cloud Deployment
The system can be deployed on:
- **Google Colab** (for training)
- **Hugging Face Spaces** (for web app)
- **Vercel/Netlify** (for static deployment)
- **AWS/GCP/Azure** (for full deployment)

---

For additional help, please check the documentation or open an issue on GitHub.