# LAB5 - Image Recognition using Deep Convolutional Neural Networks

## Overview
This repository contains the complete implementation of LAB5 exercises for the Artificial Vision and Pattern Recognition course. The lab focuses on exploring deep convolutional neural networks (CNNs) for image recognition tasks using PyTorch and the MNIST dataset.

## 🚀 Quick Start for GPU Cluster (ensicompute)

**For cluster execution, see [CLUSTER_README.md](CLUSTER_README.md)** for quick start guide.

### One-line cluster setup:
```bash
ssh your_login@nash.ensimag.fr
cd ~/Bureau/FIB/cours/AVPR/Labs/LAB5
./setup_cluster.sh && ./run_cluster.sh --task=all
```

### Cluster-specific files:
- `setup_cluster.sh` - Configure environment for cluster
- `run_cluster.sh` - Submit jobs to GPU cluster
- `check_status.sh` - Monitor job status
- `CLUSTER_README.md` - Quick cluster guide
- `CLUSTER_GUIDE.md` - Detailed cluster documentation

## Project Structure
```
LAB5/
├── LAB5.py                              # Simple CNN example (teacher's starter code)
├── LAB5_1.py                            # ResNet transfer learning example (teacher's starter code)
├── task1_hyperparameter_exploration.py  # Task 1: Hyperparameter experiments
├── task2_architectural_adaptation.py    # Task 2: Architecture modification experiments
├── task3_data_transformation.py         # Task 3: Data transformation experiments
├── lab5_runner.py                       # Main runner script (interactive & CLI)
└── README.md                            # This file
```

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
Install the required packages using pip:

```bash
pip install torch torchvision matplotlib tqdm numpy
```

Or create a requirements file:

```bash
# requirements.txt
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
tqdm>=4.60.0
numpy>=1.19.0
```

Then install with:
```bash
pip install -r requirements.txt
```

## Tasks Description

### Task 1: Hyperparameter Exploration
Explores the impact of hyperparameters on ResNet model performance:
- **Learning Rate Adjustment**: Tests different learning rates (0.0001, 0.001, 0.01)
- **Batch Size Variance**: Experiments with batch sizes (32, 64, 128)
- **Epoch Sensitivity**: Evaluates different epoch counts (5, 10, 15)

**Output Files:**
- `task1_learning_rates.png`
- `task1_batch_sizes.png`
- `task1_epochs.png`

### Task 2: Architectural Adaptation
Modifies the ResNet architecture for improved MNIST performance:
- **Layer Modification**: Adds Dropout layers with various rates (0.0, 0.3, 0.5, 0.7)
- **Filter Size Adjustments**: Tests different kernel sizes (3x3, 5x5, 7x7)
- **Feature Map Depth Variation**: Varies the number of filters (32, 64, 128)

**Output Files:**
- `task2_dropout.png`
- `task2_kernel_sizes.png`
- `task2_filter_depth.png`

### Task 3: Data Transformation Techniques
Evaluates data transformations on model performance:
- **Transformation Sequences**: Tests basic, rotation, brightness, and combined transforms
- **Data Augmentation**: Experiments with no augmentation, rotation, affine, and heavy augmentation
- **Normalization Methods**: Compares different normalization techniques

**Output Files:**
- `task3_transformations.png`
- `task3_augmentation.png`
- `task3_normalization.png`
- `task3_augmentation_examples.png`

## Usage

### Method 1: Interactive Menu (Recommended)
Run the main runner script without arguments:

```bash
python lab5_runner.py
```

This will display an interactive menu where you can:
1. Run individual tasks (1, 2, or 3)
2. Run all tasks sequentially
3. Exit the program

### Method 2: Command Line Interface
Run specific tasks directly from the command line:

```bash
# Run Task 1 only
python lab5_runner.py --task 1

# Run Task 2 only
python lab5_runner.py --task 2

# Run Task 3 only
python lab5_runner.py --task 3

# Run all tasks
python lab5_runner.py --task all
```

### Method 3: Run Individual Task Files
You can also run each task file directly:

```bash
python task1_hyperparameter_exploration.py
python task2_architectural_adaptation.py
python task3_data_transformation.py
```

## Expected Runtime

### CPU
- Task 1: ~45-60 minutes (3 experiments)
- Task 2: ~45-60 minutes (3 experiments)
- Task 3: ~45-60 minutes (4 experiments)
- **Total: ~2.5-3 hours**

### GPU (CUDA)
- Task 1: ~15-20 minutes
- Task 2: ~15-20 minutes
- Task 3: ~15-20 minutes
- **Total: ~45-60 minutes**

**Note:** The scripts automatically detect and use GPU if available.

## Understanding the Results

### Output Visualizations
Each task generates PNG files with comparative plots:
- **Loss Plots**: Show training loss over epochs
- **Accuracy Plots**: Display test accuracy progression
- **Bar Charts**: Compare final metrics across configurations

### Console Output
Each experiment prints:
- Current configuration being tested
- Per-epoch loss and accuracy
- Training time per epoch
- Final summary of all results

### Example Output
```
==================================================================
TASK 1: HYPERPARAMETER EXPLORATION
==================================================================

Testing Learning Rate: 0.001
--------------------------------------------------
Epoch [1/5], Loss: 0.2534, Accuracy: 96.45%, Time: 123.45s
Epoch [2/5], Loss: 0.1023, Accuracy: 97.82%, Time: 122.34s
...
```

## GPU Acceleration

The scripts automatically detect and use GPU if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

To verify GPU usage:
```bash
# Check if PyTorch can see your GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Model Checkpoints

Task scripts use ResNet18 with transfer learning:
- Pre-trained weights are automatically downloaded
- Models are adapted for MNIST (1 channel input, 10 classes)
- Each experiment trains a fresh model instance

## Tips for Success

1. **Start Small**: Run one task first to ensure everything works
2. **Monitor Resources**: Watch RAM/VRAM usage during training
3. **GPU Recommended**: Training on CPU is significantly slower
4. **Save Results**: All plots are automatically saved as PNG files
5. **Compare Results**: Look at the summary statistics printed at the end

## Troubleshooting

### Out of Memory Error
- Reduce batch size in the scripts
- Close other applications
- Use CPU if GPU memory is limited

### Slow Training
- Ensure GPU is being used (check console output)
- Reduce number of epochs for initial testing
- Use smaller models if needed

### Import Errors
- Verify all dependencies are installed
- Check Python version (3.7+)
- Try: `pip install --upgrade torch torchvision`

### MNIST Download Issues
- Check internet connection
- Data will be cached in `./data/` after first download
- Delete `./data/` if corrupted and re-run

## Results Analysis

After running all tasks, you should be able to answer:

1. **Which learning rate converges fastest?**
2. **How does batch size affect training speed vs. accuracy?**
3. **Do more epochs always improve accuracy?**
4. **Does dropout help prevent overfitting?**
5. **What kernel size works best for MNIST?**
6. **Does data augmentation improve generalization?**
7. **Which normalization method is most effective?**

## Additional Notes

- The MNIST dataset will be automatically downloaded on first run
- All experiments use the same test set for fair comparison
- Random seeds are not set, so results may vary slightly between runs
- Results are printed to console and saved as visualizations

## Author Information

This implementation was created for the LAB5 assignment of the Artificial Vision and Pattern Recognition course.

## License

This code is provided for educational purposes as part of the course requirements.

---

**Good luck with your experiments! 🚀**

