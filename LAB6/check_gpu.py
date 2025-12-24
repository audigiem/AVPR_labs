#!/usr/bin/env python3
"""
GPU Availability Checker
Run this script to verify your GPU setup before training
"""

import sys

print("="*60)
print("GPU CONFIGURATION CHECK")
print("="*60)

# Check Python version
print(f"\nPython Version: {sys.version}")

# Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch is installed")
    print(f"  Version: {torch.__version__}")
except ImportError:
    print("\n✗ PyTorch is NOT installed")
    print("  Install with: pip install torch torchvision")
    sys.exit(1)

# Check CUDA
print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA is available")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    
    # Print details for each GPU
    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    Multi-Processors: {props.multi_processor_count}")
    
    # Test GPU with a simple operation
    print("\nTesting GPU with simple tensor operation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU computation test passed!")
        print(f"  Result tensor shape: {z.shape}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ GPU computation test failed: {e}")
else:
    print("✗ CUDA is NOT available")
    print("  Training will run on CPU (much slower)")
    print("\nPossible reasons:")
    print("  1. PyTorch CPU-only version installed")
    print("  2. No GPU available on this machine")
    print("  3. CUDA drivers not properly installed")
    print("\nTo install PyTorch with CUDA support:")
    print("  Visit: https://pytorch.org/get-started/locally/")

# Check TorchVision
try:
    import torchvision
    print(f"\n✓ TorchVision is installed")
    print(f"  Version: {torchvision.__version__}")
except ImportError:
    print("\n✗ TorchVision is NOT installed")
    print("  Install with: pip install torchvision")

# Check PIL
try:
    from PIL import Image
    print(f"\n✓ Pillow (PIL) is installed")
except ImportError:
    print("\n✗ Pillow is NOT installed")
    print("  Install with: pip install Pillow")

# Check tqdm
try:
    import tqdm
    print(f"✓ tqdm is installed")
except ImportError:
    print("\n✗ tqdm is NOT installed")
    print("  Install with: pip install tqdm")

print("\n" + "="*60)
if torch.cuda.is_available():
    print("READY FOR GPU TRAINING! 🚀")
    print("\nYou can now submit your job with:")
    print("  sbatch train_gpu.sbatch")
else:
    print("GPU NOT AVAILABLE - will train on CPU")
    print("Training will be significantly slower")
print("="*60)
