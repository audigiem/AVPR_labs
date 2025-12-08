#!/bin/bash

# LAB5 Cluster Execution Script for ensicompute
# Usage: ./run_cluster.sh [--task=1|2|3|all] [--time=4:00:00] [--mem=8GB] [--cpus=8]

# Default values
MEMORY="8GB"
CPUS="8"
TIME_LIMIT="4:00:00"
TASK="all"
JOB_NAME="lab5_training"
PARTITION="rtx6000"  # Options: rtx6000 (turing-[1..11]), v100 (tesla), a40 (ampere)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task=*)
            TASK="${1#*=}"
            shift
            ;;
        --mem=*)
            MEMORY="${1#*=}"
            shift
            ;;
        --cpus=*)
            CPUS="${1#*=}"
            shift
            ;;
        --time=*)
            TIME_LIMIT="${1#*=}"
            shift
            ;;
        --partition=*)
            PARTITION="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task=N         Task to run: 1, 2, 3, or all (default: all)"
            echo "  --mem=SIZE       Memory allocation (default: 8GB)"
            echo "  --cpus=N         Number of CPUs (default: 8)"
            echo "  --time=TIME      Time limit HH:MM:SS (default: 4:00:00)"
            echo "  --partition=P    Partition: rtx6000, v100, a40 (default: rtx6000)"
            echo ""
            echo "Examples:"
            echo "  $0 --task=1"
            echo "  $0 --task=all --mem=16GB --cpus=12 --time=8:00:00"
            echo "  $0 --task=2 --partition=a40"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate task argument
if [[ ! "$TASK" =~ ^(1|2|3|all)$ ]]; then
    echo "Error: Task must be 1, 2, 3, or all"
    exit 1
fi

# Create directories for logs
mkdir -p cluster_logs/{output,errors,checkpoints}

# Timestamp for unique identification
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SLURM_SCRIPT="cluster_logs/slurm_job_${TIMESTAMP}.sh"

# Create SLURM batch script
cat > "$SLURM_SCRIPT" << 'EOFSLURM'
#!/bin/bash
#SBATCH --job-name=JOBNAME_PLACEHOLDER
#SBATCH --output=cluster_logs/output/JOBNAME_PLACEHOLDER_TIMESTAMP_PLACEHOLDER.out
#SBATCH --error=cluster_logs/errors/JOBNAME_PLACEHOLDER_TIMESTAMP_PLACEHOLDER.err
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER
#SBATCH --mem=MEMORY_PLACEHOLDER
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --gres=gpu:1

# ============================================================================
# Job Configuration
# ============================================================================
echo "============================================================================"
echo "LAB5 - Deep Learning Training on ensicompute"
echo "============================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job started"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Node: $SLURM_NODELIST"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ID: $SLURM_JOB_ID"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task: TASK_PLACEHOLDER"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configuration:"
echo "                                - Memory: MEMORY_PLACEHOLDER"
echo "                                - CPUs: CPUS_PLACEHOLDER"
echo "                                - Time Limit: TIME_PLACEHOLDER"
echo "                                - Partition: PARTITION_PLACEHOLDER"
echo "============================================================================"

# ============================================================================
# Environment Setup
# ============================================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up environment..."

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Try to load modules if available (nash has module system issues in SLURM)
if command -v module &> /dev/null; then
    # Load CUDA modules (try multiple versions)
    module load cuda/12.4 2>/dev/null || \
    module load cuda/12.1 2>/dev/null || \
    module load cuda/11.8 2>/dev/null || \
    echo "[INFO] No CUDA module loaded, using system CUDA"

    module load cudnn/8.9 2>/dev/null || \
    module load cudnn 2>/dev/null || \
    echo "[INFO] No cuDNN module loaded, using system cuDNN"

    # Display loaded modules
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loaded modules:"
    module list 2>&1
else
    echo "[INFO] Module system not available, using system CUDA libraries"
    echo "[INFO] CUDA_HOME: $CUDA_HOME"
fi

# Display GPU information
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
echo ""

# Activate virtual environment
ENV_PATH="lab5_env/bin/activate"
if [ -f "$ENV_PATH" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Activating virtual environment..."
    source $ENV_PATH
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Virtual environment not found at $ENV_PATH"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using system Python"
fi

# Set environment variables
export OMP_NUM_THREADS=CPUS_PLACEHOLDER
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Display Python and PyTorch versions
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python version:"
python3 --version
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PyTorch information:"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "============================================================================"

# ============================================================================
# Run Training
# ============================================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting LAB5 training..."
echo ""

TASK_ARG="TASK_PLACEHOLDER"
if [ "$TASK_ARG" == "all" ]; then
    python3 lab5_runner.py --task all
else
    python3 lab5_runner.py --task $TASK_ARG
fi

PYTHON_EXIT_CODE=$?

echo ""
echo "============================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed with exit code: $PYTHON_EXIT_CODE"
echo "============================================================================"

# ============================================================================
# Final Summary
# ============================================================================
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Training completed successfully!"

    # List saved models/checkpoints if any
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for saved models..."
    saved_models=$(find . -name "*.pth" -o -name "*.pt" -newer "$0" 2>/dev/null | head -20)
    if [ ! -z "$saved_models" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Models saved:"
        echo "$saved_models" | while read model; do
            size=$(ls -lh "$model" 2>/dev/null | awk '{print $5}')
            echo "   - $model ($size)"
        done
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No model files found"
    fi

    # Check for result plots
    plots=$(find . -name "*.png" -o -name "*.jpg" -newer "$0" 2>/dev/null | head -20)
    if [ ! -z "$plots" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Plots generated:"
        echo "$plots" | while read plot; do
            size=$(ls -lh "$plot" 2>/dev/null | awk '{print $5}')
            echo "   - $plot ($size)"
        done
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Training failed!"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check error log for details"
fi

echo "============================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job finished"
echo "============================================================================"
EOFSLURM

# Replace placeholders in the SLURM script
sed -i "s/JOBNAME_PLACEHOLDER/${JOB_NAME}_task${TASK}/g" "$SLURM_SCRIPT"
sed -i "s/TIMESTAMP_PLACEHOLDER/${TIMESTAMP}/g" "$SLURM_SCRIPT"
sed -i "s/TIME_PLACEHOLDER/${TIME_LIMIT}/g" "$SLURM_SCRIPT"
sed -i "s/CPUS_PLACEHOLDER/${CPUS}/g" "$SLURM_SCRIPT"
sed -i "s/MEMORY_PLACEHOLDER/${MEMORY}/g" "$SLURM_SCRIPT"
sed -i "s/PARTITION_PLACEHOLDER/${PARTITION}/g" "$SLURM_SCRIPT"
sed -i "s/TASK_PLACEHOLDER/${TASK}/g" "$SLURM_SCRIPT"

# ============================================================================
# Submit Job
# ============================================================================
echo "============================================================================"
echo "Submitting LAB5 training job to SLURM cluster"
echo "============================================================================"
echo "Configuration:"
echo "  Task:      $TASK"
echo "  Memory:    $MEMORY"
echo "  CPUs:      $CPUS"
echo "  Time:      $TIME_LIMIT"
echo "  Partition: $PARTITION"
echo "============================================================================"

JOB_ID=$(sbatch "$SLURM_SCRIPT" 2>&1 | grep -o '[0-9]*' | head -1)

if [ ! -z "$JOB_ID" ]; then
    echo "✓ Job submitted successfully!"
    echo ""
    echo "Job Details:"
    echo "  Job ID:     $JOB_ID"
    echo "  Output:     cluster_logs/output/${JOB_NAME}_task${TASK}_${TIMESTAMP}.out"
    echo "  Errors:     cluster_logs/errors/${JOB_NAME}_task${TASK}_${TIMESTAMP}.err"
    echo "  Script:     $SLURM_SCRIPT"
    echo ""
    echo "============================================================================"
    echo "Monitoring Commands:"
    echo "============================================================================"
    echo "  squeue -u \$USER                                          # Check job status"
    echo "  squeue -j $JOB_ID                                         # Check this job"
    echo "  tail -f cluster_logs/output/${JOB_NAME}_task${TASK}_${TIMESTAMP}.out  # Follow output"
    echo "  tail -f cluster_logs/errors/${JOB_NAME}_task${TASK}_${TIMESTAMP}.err  # Follow errors"
    echo "  scancel $JOB_ID                                           # Cancel job"
    echo "============================================================================"
    echo ""
    echo "You can now safely close this terminal - the job will continue running!"
    echo ""
else
    echo "✗ Failed to submit job"
    echo "Please check:"
    echo "  1. You are connected to nash.ensimag.fr"
    echo "  2. SLURM is available (try: sinfo)"
    echo "  3. Your account has access to the cluster"
    exit 1
fi

