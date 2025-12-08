#!/bin/bash

echo "============================================================================"
echo "Fixing PyTorch Installation for GPU Support"
echo "============================================================================"
echo ""

# Activate virtual environment
if [ -f "lab5_env/bin/activate" ]; then
    echo "Activating virtual environment..."
    source lab5_env/bin/activate
else
    echo "Error: Virtual environment not found!"
    exit 1
fi

echo "Current PyTorch version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "Uninstalling current PyTorch..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "============================================================================"
echo "Verification:"
echo "============================================================================"
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")

# Note: CUDA devices won't be detected on nash (login node)
# They will be available on compute nodes (turing-*)
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("Note: CUDA devices not detected on login node - this is normal")
    print("GPUs will be available when job runs on compute nodes")
EOF

echo ""
echo "============================================================================"
echo "Done! Now submit your job again with:"
echo "  ./run_cluster.sh --task=all"
echo "============================================================================"
