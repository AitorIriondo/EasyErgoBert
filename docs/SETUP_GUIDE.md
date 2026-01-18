# Detailed Setup Guide

This guide covers complete setup from scratch on Windows, Linux, and macOS.

## Prerequisites

- NVIDIA GPU (recommended: RTX 20/30/40/50 series)
- NVIDIA drivers installed
- Conda (Miniconda or Anaconda)
- Git
- Blender 4.0+ (for FBX export)

## Windows Setup

### 1. Install NVIDIA Drivers

1. Download from: https://www.nvidia.com/Download/index.aspx
2. Select your GPU model and download
3. Install and restart

Verify installation:
```cmd
nvidia-smi
```

### 2. Install Miniconda

1. Download from: https://docs.conda.io/en/latest/miniconda.html
2. Run installer, add to PATH when prompted
3. Open "Anaconda Prompt" or restart terminal

### 3. Create Environment

```cmd
:: Create environment with Python 3.11
conda create -n body4d python=3.11 -y
conda activate body4d

:: Install PyTorch with CUDA 12.8 (RTX 40/50 series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

:: For older GPUs (RTX 20/30), use CUDA 11.8:
:: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Clone Repository

```cmd
cd C:\
git clone https://github.com/YOUR_USERNAME/EasyErgoBert.git
cd EasyErgoBert
```

### 5. Install Dependencies

```cmd
pip install -r requirements.txt
```

### 6. Clone MotionBERT

```cmd
git clone https://github.com/Walter0807/MotionBERT.git
```

### 7. Download Checkpoint

1. Go to: https://1drv.ms/f/s!AvAdh0LSjEOlgT67igq_cIoYvO2y?e=bfEc73
2. Navigate to `pose3d` folder
3. Download `FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`
4. Create directory and move file:

```cmd
mkdir MotionBERT\checkpoint\pose3d\FT_MB_lite_MB_ft_h36m_global_lite
move best_epoch.bin MotionBERT\checkpoint\pose3d\FT_MB_lite_MB_ft_h36m_global_lite\
```

### 8. Verify Installation

```cmd
python download_pose3d_checkpoint.py
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 9. Install Blender

1. Download from: https://www.blender.org/download/
2. Install to default location: `C:\Program Files\Blender Foundation\Blender 5.0\`

## Linux Setup

### 1. Install NVIDIA Drivers

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535  # Or latest version

# Verify
nvidia-smi
```

### 2. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 3. Create Environment

```bash
conda create -n body4d python=3.11 -y
conda activate body4d

# PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/EasyErgoBert.git
cd EasyErgoBert
pip install -r requirements.txt
git clone https://github.com/Walter0807/MotionBERT.git
```

### 5. Download Checkpoint

Download from OneDrive and place in:
```bash
mkdir -p MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite
mv best_epoch.bin MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/
```

### 6. Install Blender

```bash
# Ubuntu
sudo snap install blender --classic

# Or download from blender.org
```

## macOS Setup (CPU Only)

Note: macOS does not support NVIDIA CUDA. Pipeline runs on CPU.

### 1. Install Miniconda

```bash
brew install miniconda
conda init zsh  # or bash
```

### 2. Create Environment

```bash
conda create -n body4d python=3.11 -y
conda activate body4d

# CPU-only PyTorch
pip install torch torchvision torchaudio
```

### 3. Continue with clone and setup as Linux

## Verifying GPU Support

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("GPU test: PASSED")
```

## Troubleshooting

### "CUDA not available" on Windows

1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Ensure no CPU-only PyTorch is installed: `pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu128`

### "No module named 'ultralytics'"

```bash
pip install ultralytics
```

### "No module named 'easydict'"

```bash
pip install easydict
```

### MotionBERT import errors

Ensure MotionBERT is in the project directory and the lib folder exists:
```
EasyErgoBert/
└── MotionBERT/
    └── lib/
        └── model/
            └── DSTformer.py
```

### Blender not found

Windows: Use full path:
```cmd
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background ...
```

Linux: Install via snap or add to PATH:
```bash
export PATH=$PATH:/path/to/blender
```

### Out of GPU memory

1. Use a smaller batch or single frames
2. Close other GPU applications
3. Use CPU mode: `--device cpu`

## Environment Variables

Optional environment variables:

```bash
# Custom CUDA path (if not auto-detected)
export CUDA_HOME=/usr/local/cuda-12.8

# Limit GPU memory (useful for shared systems)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Updating

```bash
cd EasyErgoBert
git pull origin main

# Update dependencies if requirements.txt changed
pip install -r requirements.txt --upgrade
```

## Uninstalling

```bash
conda deactivate
conda env remove -n body4d
rm -rf EasyErgoBert  # Or delete the folder manually
```
