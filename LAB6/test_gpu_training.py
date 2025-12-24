#!/usr/bin/env python3
"""
Quick test script to verify GPU training works before submitting long jobs
"""

import torch
import torch.nn as nn
import time
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

print("="*60)
print("QUICK GPU TRAINING TEST")
print("="*60)

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load a small model
print("\nLoading Faster R-CNN model...")
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Modify for 2 classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

# Move to device
print(f"Moving model to {device}...")
model = model.to(device)
model.train()

# Create dummy data
print("Creating dummy batch...")
batch_size = 2
images = [torch.rand(3, 800, 800).to(device) for _ in range(batch_size)]
targets = [
    {
        "boxes": torch.tensor([[50, 50, 200, 200]], dtype=torch.float32).to(device),
        "labels": torch.tensor([1], dtype=torch.int64).to(device)
    }
    for _ in range(batch_size)
]

# Forward pass
print("\nRunning forward pass...")
start = time.time()
try:
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    print(f"✓ Forward pass successful!")
    print(f"  Total loss: {losses.item():.4f}")
    print(f"  Time: {time.time() - start:.3f}s")
    
    # Backward pass
    print("\nRunning backward pass...")
    start = time.time()
    losses.backward()
    print(f"✓ Backward pass successful!")
    print(f"  Time: {time.time() - start:.3f}s")
    
    if device.type == 'cuda':
        print(f"\n✓ GPU Memory used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("Your setup is ready for training.")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ TEST FAILED: {e}")
    print("\nPlease check your environment and try again.")
    import traceback
    traceback.print_exc()
