#!/bin/bash
#SBATCH --job-name=LAB6_object_detection
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=a40
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="

# Print GPU information
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Activate your Python environment
echo "Activating virtual environment..."
source .Lab6venv/bin/activate
echo "✓ Virtual environment activated: .Lab6venv"
echo ""

# Optional: Load CUDA module if needed on your cluster
# module load cuda/11.8

# Install requirements if needed (first time only)
# pip install -r requirements.txt

# Print Python and PyTorch versions
echo "Python version:"
python --version
echo ""

echo "PyTorch configuration:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"
echo ""

# Run the training script
echo "Starting training..."
python code/LAB6_Complete.py --all-tasks --use-gpu

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
