#!/bin/bash

# LAB5 - Cluster Setup Script for ensicompute
# This script prepares the environment for GPU execution on the cluster

echo "============================================================================"
echo "LAB5 - Cluster Environment Setup"
echo "============================================================================"
echo ""

# Check if running on nash
if [[ $(hostname) != "nash"* ]]; then
    echo "⚠️  Warning: This script is designed for nash.ensimag.fr"
    echo "   Current host: $(hostname)"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo "📋 Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment already exists
if [ -d "lab5_env" ]; then
    echo ""
    echo "⚠️  Virtual environment 'lab5_env' already exists"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        rm -rf lab5_env
    else
        echo "✓ Using existing environment"
        source lab5_env/bin/activate
        echo ""
        echo "📦 Installed packages:"
        pip list | grep -E "torch|numpy|matplotlib|tqdm"
        echo ""
        echo "============================================================================"
        echo "✅ Setup complete! Environment is ready."
        echo "============================================================================"
        exit 0
    fi
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv lab5_env

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source lab5_env/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q

# Install PyTorch with CUDA support
echo ""
echo "🔥 Installing PyTorch with CUDA 11.8 support..."
echo "   This may take a few minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Failed to install PyTorch with CUDA 11.8"
    echo "   Trying CUDA 12.1 as fallback..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Failed to install PyTorch with CUDA"
        echo "   Installing CPU version as last resort..."
        pip install torch torchvision torchaudio
    fi
fi

# Install other dependencies
echo ""
echo "📦 Installing other dependencies..."
pip install matplotlib tqdm numpy

# Verify installations
echo ""
echo "============================================================================"
echo "🔍 Verifying installations..."
echo "============================================================================"

python3 << EOF
import sys

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
        return True
    except ImportError:
        print(f"✗ {name}: NOT INSTALLED")
        return False

print("\nPackages:")
check_package("NumPy", "numpy")
check_package("Matplotlib", "matplotlib")
check_package("tqdm")

print("\nPyTorch:")
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"  - CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - cuDNN version: {torch.backends.cudnn.version()}")
else:
    print(f"  ⚠️  CUDA not detected (this is normal on nash, GPUs are on compute nodes)")

print("\nTorchVision:")
import torchvision
print(f"✓ TorchVision: {torchvision.__version__}")
EOF

echo ""
echo "============================================================================"
echo "✅ Installation Complete!"
echo "============================================================================"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source lab5_env/bin/activate"
echo ""
echo "2. Test locally (optional):"
echo "   python3 lab5_runner.py --task 1"
echo ""
echo "3. Submit to cluster:"
echo "   ./run_cluster.sh --task=all"
echo ""
echo "4. Monitor your jobs:"
echo "   ./check_status.sh"
echo ""
echo "For detailed instructions, see CLUSTER_GUIDE.md"
echo "============================================================================"
echo ""

# Create necessary directories
mkdir -p cluster_logs/{output,errors,checkpoints}
echo "📁 Created log directories"

echo ""
echo "🎉 You're all set! Ready to run on the GPU cluster."
echo ""

