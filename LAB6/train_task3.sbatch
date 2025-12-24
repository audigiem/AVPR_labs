#!/bin/bash
#SBATCH --job-name=LAB6_task3
#SBATCH --output=logs/task3_%j.out
#SBATCH --error=logs/task3_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=a40
#SBATCH --cpus-per-task=4

mkdir -p logs

echo "=========================================="
echo "TASK 3: Data Augmentation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Activate virtual environment
source .Lab6venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

nvidia-smi
echo ""

python code/LAB6_Complete.py --task 3 --use-gpu

echo ""
echo "Task 3 finished at: $(date)"
