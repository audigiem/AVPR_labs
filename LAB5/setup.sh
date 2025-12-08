#!/bin/bash

# LAB5 Setup Script
# This script sets up the environment for running LAB5 tasks

echo "=================================================="
echo "LAB5 - Setup and Installation Script"
echo "=================================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv lab5_env

    echo "Activating virtual environment..."
    source lab5_env/bin/activate

    echo "Virtual environment created and activated!"
fi

# Install dependencies
echo ""
echo "Installing required packages..."
echo "This may take a few minutes..."
echo ""

pip install --upgrade pip

# Install PyTorch (CPU version - change if you need CUDA)
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing other dependencies..."
pip install matplotlib tqdm numpy

# Verify installations
echo ""
echo "Verifying installations..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
python3 -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"
python3 -c "import tqdm; print(f'tqdm installed successfully')"
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "To run the tasks:"
echo "  1. Interactive mode: python3 lab5_runner.py"
echo "  2. CLI mode: python3 lab5_runner.py --task 1"
echo "  3. Run all: python3 lab5_runner.py --task all"
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Note: Remember to activate the virtual environment before running:"
    echo "  source lab5_env/bin/activate"
    echo ""
fi

echo "For GPU support, please install PyTorch with CUDA manually."
echo "See: https://pytorch.org/get-started/locally/"
echo ""

