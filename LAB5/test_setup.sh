#!/bin/bash

# Test script to verify LAB5 setup before cluster submission
# Usage: ./test_setup.sh

echo "============================================================================"
echo "LAB5 - Pre-Cluster Test Script"
echo "============================================================================"
echo ""

ERRORS=0
WARNINGS=0

# Test 1: Check Python
echo "🐍 Test 1: Python Installation"
echo "────────────────────────────────────────────────────────────────────────"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION"
else
    echo "✗ Python 3 not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Test 2: Check virtual environment
echo "📦 Test 2: Virtual Environment"
echo "────────────────────────────────────────────────────────────────────────"
if [ -d "lab5_env" ]; then
    echo "✓ Virtual environment exists"

    # Try to activate and check
    if [ -f "lab5_env/bin/activate" ]; then
        echo "✓ Activation script found"
        source lab5_env/bin/activate

        # Test 3: Check packages
        echo ""
        echo "📚 Test 3: Required Packages"
        echo "────────────────────────────────────────────────────────────────────────"

        python3 << 'EOFPYTHON'
import sys

packages_ok = True

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if not torch.cuda.is_available():
        print("  ℹ️  CUDA not available here (normal on nash, available on compute nodes)")
except ImportError:
    print("✗ PyTorch not installed")
    packages_ok = False

# Check TorchVision
try:
    import torchvision
    print(f"✓ TorchVision {torchvision.__version__}")
except ImportError:
    print("✗ TorchVision not installed")
    packages_ok = False

# Check other packages
required = {
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'tqdm': 'tqdm'
}

for module, name in required.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', '')
        print(f"✓ {name} {version}")
    except ImportError:
        print(f"✗ {name} not installed")
        packages_ok = False

sys.exit(0 if packages_ok else 1)
EOFPYTHON

        if [ $? -ne 0 ]; then
            ERRORS=$((ERRORS + 1))
        fi

    else
        echo "✗ Activation script not found"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "✗ Virtual environment not found"
    echo "  Run: ./setup_cluster.sh"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Test 4: Check required files
echo "📄 Test 4: Required Files"
echo "────────────────────────────────────────────────────────────────────────"

required_files=(
    "lab5_runner.py"
    "task1_hyperparameter_exploration.py"
    "task2_architectural_adaptation.py"
    "task3_data_transformation.py"
    "run_cluster.sh"
    "requirements.txt"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file missing"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""

# Test 5: Check scripts are executable
echo "🔧 Test 5: Script Permissions"
echo "────────────────────────────────────────────────────────────────────────"

executable_scripts=(
    "run_cluster.sh"
    "check_status.sh"
    "setup_cluster.sh"
    "test_setup.sh"
)

for script in "${executable_scripts[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "✓ $script is executable"
        else
            echo "⚠️  $script is not executable (run: chmod +x $script)"
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
done

echo ""

# Test 6: Check data directory
echo "📁 Test 6: Data Directory"
echo "────────────────────────────────────────────────────────────────────────"

if [ -d "data" ]; then
    echo "✓ data/ directory exists"

    if [ -d "data/MNIST" ]; then
        echo "✓ data/MNIST/ directory exists"

        mnist_files=$(find data/MNIST -type f 2>/dev/null | wc -l)
        if [ $mnist_files -gt 0 ]; then
            echo "✓ MNIST data files found ($mnist_files files)"
        else
            echo "⚠️  MNIST directory empty (will be downloaded on first run)"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "⚠️  data/MNIST/ not found (will be created on first run)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "⚠️  data/ directory not found (will be created on first run)"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""

# Test 7: Quick Python syntax check
echo "✨ Test 7: Python Syntax Check"
echo "────────────────────────────────────────────────────────────────────────"

if [ -f "lab5_env/bin/activate" ]; then
    source lab5_env/bin/activate
fi

for pyfile in task*.py lab5_runner.py; do
    if [ -f "$pyfile" ]; then
        python3 -m py_compile "$pyfile" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✓ $pyfile syntax OK"
        else
            echo "✗ $pyfile has syntax errors"
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

echo ""

# Test 8: Check if on nash
echo "🌐 Test 8: Connection to Cluster"
echo "────────────────────────────────────────────────────────────────────────"

if [[ $(hostname) == "nash"* ]]; then
    echo "✓ Connected to nash.ensimag.fr"

    # Check if slurm is available
    if command -v squeue &> /dev/null; then
        echo "✓ SLURM commands available"

        # Check cluster status
        echo ""
        echo "  Cluster quick status:"
        sinfo -o "%10P %5a %10l %6D %6t" 2>/dev/null | head -5
    else
        echo "✗ SLURM commands not found"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "⚠️  Not on nash.ensimag.fr (current: $(hostname))"
    echo "  Connect first: ssh your_login@nash.ensimag.fr"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""

# Test 9: Check disk space
echo "💾 Test 9: Disk Space"
echo "────────────────────────────────────────────────────────────────────────"

available_space=$(df -h . | tail -1 | awk '{print $4}')
echo "✓ Available space: $available_space"

echo ""

# Test 10: Try a minimal import test
echo "🧪 Test 10: Quick Import Test"
echo "────────────────────────────────────────────────────────────────────────"

if [ -f "lab5_env/bin/activate" ]; then
    source lab5_env/bin/activate

    python3 << 'EOFPYTHON'
try:
    import torch
    import torchvision
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import numpy as np
    print("✓ All imports successful")
    exit(0)
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)
EOFPYTHON

    if [ $? -ne 0 ]; then
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "⚠️  Skipping (virtual environment not found)"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""
echo "============================================================================"
echo "📊 Test Summary"
echo "============================================================================"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ All tests passed! Ready for cluster submission."
    echo ""
    echo "Next steps:"
    echo "  1. Submit job: ./run_cluster.sh --task=all"
    echo "  2. Check status: ./check_status.sh"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  Tests passed with $WARNINGS warning(s)"
    echo ""
    echo "You can proceed, but consider addressing the warnings."
    echo ""
    echo "To submit anyway:"
    echo "  ./run_cluster.sh --task=all"
    echo ""
    exit 0
else
    echo "❌ Tests failed with $ERRORS error(s) and $WARNINGS warning(s)"
    echo ""
    echo "Please fix the errors before submitting to cluster."
    echo ""

    if [ ! -d "lab5_env" ]; then
        echo "Suggestion: Run ./setup_cluster.sh to set up the environment"
    fi

    echo ""
    exit 1
fi

