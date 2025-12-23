"""
LAB6: Object Detection Using Deep Learning and Transfer Learning
Complete implementation covering all tasks (1-4)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import time
import json

# ========================================
# CUSTOM DATASET CLASS
# ========================================
class CustomDataset(Dataset):
    """Custom dataset for YOLO-format annotations"""
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        # Case-insensitive image extension check
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"Loaded {len(self.images)} images from {images_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Get corresponding txt label
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.labels_dir, base_name + ".txt")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        
        # Load YOLO-format labels (class, x_center, y_center, w, h)
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x_center, y_center, w, h = map(float, parts[:5])
                        # Convert YOLO format to (x1, y1, x2, y2)
                        x1 = (x_center - w/2) * width
                        y1 = (y_center - h/2) * height
                        x2 = (x_center + w/2) * width
                        y2 = (y_center + h/2) * height
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls) + 1)  # +1 because 0 is background
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


# ========================================
# TASK 1: HYPERPARAMETER EXPLORATION
# ========================================
def task1_hyperparameter_exploration(images_dir, labels_dir):
    """
    Task 1: Test different hyperparameters
    - Learning rates: 0.00001, 0.0001, 0.001
    - Batch sizes: 1, 2, 4
    - Epochs: 2, 3, 5
    """
    print("\n" + "="*60)
    print("TASK 1: HYPERPARAMETER EXPLORATION")
    print("="*60)
    
    results = {}
    
    # Hyperparameter configurations to test
    configs = [
        {"name": "config1_baseline", "lr": 0.0001, "batch_size": 1, "epochs": 2},
        {"name": "config2_high_lr", "lr": 0.001, "batch_size": 1, "epochs": 2},
        {"name": "config3_low_lr", "lr": 0.00001, "batch_size": 1, "epochs": 2},
        {"name": "config4_batch2", "lr": 0.0001, "batch_size": 2, "epochs": 2},
        {"name": "config5_more_epochs", "lr": 0.0001, "batch_size": 1, "epochs": 3},
    ]
    
    for config in configs:
        print(f"\n--- Testing {config['name']} ---")
        print(f"LR: {config['lr']}, Batch Size: {config['batch_size']}, Epochs: {config['epochs']}")
        
        # Create dataset and dataloader
        transform = transforms.ToTensor()
        dataset = CustomDataset(images_dir, labels_dir, transform=transform)
        train_loader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Load model
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        num_classes = 2  # 1 class + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        # Training
        model.train()
        start_time = time.time()
        epoch_losses = []
        
        for epoch in range(config['epochs']):
            epoch_loss = 0
            batch_count = 0
            
            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
                images = list(images)
                targets = list(targets)
                
                # Skip batches with empty targets
                if all(len(t['boxes']) > 0 for t in targets):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    batch_count += 1
            
            avg_loss = epoch_loss / max(batch_count, 1)
            epoch_losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{config['epochs']}] Average Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        
        results[config['name']] = {
            "lr": config['lr'],
            "batch_size": config['batch_size'],
            "epochs": config['epochs'],
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else 0,
            "training_time": training_time
        }
        
        print(f"Training time: {training_time:.2f}s")
        
        # Save checkpoint for this config
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'results': results[config['name']]
        }, f'checkpoint_{config["name"]}.pth')
    
    # Save results summary
    with open('task1_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n--- TASK 1 SUMMARY ---")
    for name, result in results.items():
        print(f"{name}: Final Loss = {result['final_loss']:.4f}, Time = {result['training_time']:.2f}s")
    
    return results


# ========================================
# TASK 2: ARCHITECTURAL ADAPTATION AND TRANSFER LEARNING
# ========================================
def task2_transfer_learning(images_dir, labels_dir):
    """
    Task 2: Test different transfer learning strategies
    - Freezing backbone layers
    - Gradual unfreezing
    - Different backbones (ResNet-50 vs MobileNetV3)
    """
    print("\n" + "="*60)
    print("TASK 2: ARCHITECTURAL ADAPTATION AND TRANSFER LEARNING")
    print("="*60)
    
    results = {}
    
    # Configuration 1: Baseline (no freezing)
    print("\n--- Config 1: No Freezing (Baseline) ---")
    model1, result1 = train_with_config(
        images_dir, labels_dir,
        freeze_backbone=False,
        unfreeze_layer4=False,
        backbone_type="resnet50",
        lr=0.0001,
        epochs=2,
        config_name="no_freeze"
    )
    results["no_freeze"] = result1
    torch.save(model1.state_dict(), 'model_no_freeze.pth')
    
    # Configuration 2: Freeze backbone
    print("\n--- Config 2: Freeze Backbone ---")
    model2, result2 = train_with_config(
        images_dir, labels_dir,
        freeze_backbone=True,
        unfreeze_layer4=False,
        backbone_type="resnet50",
        lr=0.0001,
        epochs=2,
        config_name="freeze_backbone"
    )
    results["freeze_backbone"] = result2
    torch.save(model2.state_dict(), 'model_freeze_backbone.pth')
    
    # Configuration 3: Gradual unfreezing (freeze backbone but unfreeze layer4)
    print("\n--- Config 3: Gradual Unfreezing (Unfreeze Layer4) ---")
    model3, result3 = train_with_config(
        images_dir, labels_dir,
        freeze_backbone=True,
        unfreeze_layer4=True,
        backbone_type="resnet50",
        lr=0.0001,
        epochs=2,
        config_name="gradual_unfreeze"
    )
    results["gradual_unfreeze"] = result3
    torch.save(model3.state_dict(), 'model_gradual_unfreeze.pth')
    
    # Configuration 4: MobileNetV3 backbone
    print("\n--- Config 4: MobileNetV3 Backbone ---")
    model4, result4 = train_with_config(
        images_dir, labels_dir,
        freeze_backbone=False,
        unfreeze_layer4=False,
        backbone_type="mobilenet",
        lr=0.0001,
        epochs=2,
        config_name="mobilenet"
    )
    results["mobilenet"] = result4
    torch.save(model4.state_dict(), 'model_mobilenet.pth')
    
    # Save results
    with open('task2_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n--- TASK 2 SUMMARY ---")
    for name, result in results.items():
        print(f"{name}: Final Loss = {result['final_loss']:.4f}, Time = {result['training_time']:.2f}s")
    
    return results


def train_with_config(images_dir, labels_dir, freeze_backbone, unfreeze_layer4, 
                      backbone_type, lr, epochs, config_name):
    """Helper function to train with specific configuration"""
    
    # Create dataset
    transform = transforms.ToTensor()
    dataset = CustomDataset(images_dir, labels_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    # Load model based on backbone type
    if backbone_type == "resnet50":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    elif backbone_type == "mobilenet":
        model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    # Modify classification head
    num_classes = 2  # 1 class + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Apply freezing strategy
    if freeze_backbone:
        print("Freezing backbone parameters...")
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        if unfreeze_layer4 and backbone_type == "resnet50":
            print("Unfreezing layer4 for gradual unfreezing...")
            for name, param in model.backbone.named_parameters():
                if 'body.layer4' in name:
                    param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Optimizer
    if freeze_backbone:
        # Only optimize unfrozen parameters
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    start_time = time.time()
    epoch_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = list(images)
            targets = list(targets)
            
            # Skip empty targets
            if all(len(t['boxes']) > 0 for t in targets):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                batch_count += 1
        
        avg_loss = epoch_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    
    result = {
        "config_name": config_name,
        "freeze_backbone": freeze_backbone,
        "unfreeze_layer4": unfreeze_layer4,
        "backbone_type": backbone_type,
        "lr": lr,
        "epochs": epochs,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else 0,
        "training_time": training_time
    }
    
    return model, result


# ========================================
# TASK 3: DATA TRANSFORMATION AND AUGMENTATION
# ========================================
def task3_data_augmentation(images_dir, labels_dir):
    """
    Task 3: Test different data augmentation strategies
    - Basic transforms vs augmented
    - Different image sizes
    - Normalization strategies
    """
    print("\n" + "="*60)
    print("TASK 3: DATA TRANSFORMATION AND AUGMENTATION")
    print("="*60)
    
    results = {}
    
    # Config 1: Basic transform (ToTensor only)
    print("\n--- Config 1: Basic Transform (ToTensor only) ---")
    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])
    result1 = train_with_augmentation(images_dir, labels_dir, transform1, "basic_transform")
    results["basic_transform"] = result1
    
    # Config 2: With horizontal flip
    print("\n--- Config 2: With Random Horizontal Flip ---")
    transform2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    result2 = train_with_augmentation(images_dir, labels_dir, transform2, "horizontal_flip")
    results["horizontal_flip"] = result2
    
    # Config 3: With color jitter
    print("\n--- Config 3: With Color Jitter ---")
    transform3 = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])
    result3 = train_with_augmentation(images_dir, labels_dir, transform3, "color_jitter")
    results["color_jitter"] = result3
    
    # Config 4: With normalization (ImageNet stats)
    print("\n--- Config 4: With ImageNet Normalization ---")
    transform4 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    result4 = train_with_augmentation(images_dir, labels_dir, transform4, "normalized")
    results["normalized"] = result4
    
    # Config 5: Combined augmentation
    print("\n--- Config 5: Combined Augmentation ---")
    transform5 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])
    result5 = train_with_augmentation(images_dir, labels_dir, transform5, "combined_augmentation")
    results["combined_augmentation"] = result5
    
    # Save results
    with open('task3_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n--- TASK 3 SUMMARY ---")
    for name, result in results.items():
        print(f"{name}: Final Loss = {result['final_loss']:.4f}, Time = {result['training_time']:.2f}s")
    
    return results


def train_with_augmentation(images_dir, labels_dir, transform, config_name):
    """Helper function to train with specific augmentation"""
    
    # Create dataset with specific transform
    dataset = CustomDataset(images_dir, labels_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    # Load model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Training
    model.train()
    start_time = time.time()
    epoch_losses = []
    num_epochs = 2
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = list(images)
            targets = list(targets)
            
            if all(len(t['boxes']) > 0 for t in targets):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                batch_count += 1
        
        avg_loss = epoch_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    
    # Save model
    torch.save(model.state_dict(), f'model_{config_name}.pth')
    
    result = {
        "config_name": config_name,
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else 0,
        "training_time": training_time
    }
    
    return result


# ========================================
# TASK 4: EVALUATION AND INFERENCE
# ========================================
def task4_evaluation_inference(images_dir, labels_dir, model_path=None, test_image_path=None):
    """
    Task 4: Evaluation and inference
    - Run inference on test images
    - Apply NMS with different thresholds
    - Visualize results
    """
    print("\n" + "="*60)
    print("TASK 4: EVALUATION AND INFERENCE")
    print("="*60)
    
    # Load or train model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training new model...")
        transform = transforms.ToTensor()
        dataset = CustomDataset(images_dir, labels_dir, transform=transform)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        model.train()
        for epoch in range(2):
            for images, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                images = list(images)
                targets = list(targets)
                
                if all(len(t['boxes']) > 0 for t in targets):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
        
        torch.save(model.state_dict(), 'model_task4.pth')
    
    model.eval()
    
    # Run inference on test image
    if test_image_path is None:
        # Use first image from dataset
        dataset = CustomDataset(images_dir, labels_dir, transform=transforms.ToTensor())
        test_image_path = os.path.join(images_dir, dataset.images[0])
    
    print(f"\nRunning inference on: {test_image_path}")
    
    # Load and preprocess image
    image = Image.open(test_image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    
    # Run detection
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]
    
    print(f"\nTotal predictions: {len(boxes)}")
    
    # Test different confidence thresholds
    confidence_thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for conf_threshold in confidence_thresholds:
        print(f"\n--- Confidence Threshold: {conf_threshold} ---")
        mask = scores >= conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        print(f"Detections after confidence filtering: {len(filtered_boxes)}")
        
        # Apply NMS with different IoU thresholds
        iou_thresholds = [0.3, 0.5, 0.7]
        
        for iou_threshold in iou_thresholds:
            if len(filtered_boxes) > 0:
                keep = nms(filtered_boxes, filtered_scores, iou_threshold)
                print(f"  IoU threshold {iou_threshold}: {len(keep)} detections after NMS")
                
                # Visualize results for confidence=0.5 and iou=0.5
                if conf_threshold == 0.5 and iou_threshold == 0.5:
                    visualize_predictions(
                        image, 
                        filtered_boxes[keep], 
                        filtered_scores[keep], 
                        filtered_labels[keep],
                        f"task4_output_conf{conf_threshold}_iou{iou_threshold}.jpg"
                    )
    
    # Compare different NMS strategies
    print("\n--- Comparing NMS Strategies ---")
    mask = scores >= 0.5
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]
    
    # No NMS
    print(f"No NMS: {len(filtered_boxes)} detections")
    visualize_predictions(image, filtered_boxes, filtered_scores, filtered_labels, "task4_no_nms.jpg")
    
    # With NMS (IoU=0.5)
    if len(filtered_boxes) > 0:
        keep = nms(filtered_boxes, filtered_scores, 0.5)
        print(f"With NMS (IoU=0.5): {len(keep)} detections")
        visualize_predictions(
            image, 
            filtered_boxes[keep], 
            filtered_scores[keep], 
            filtered_labels[keep], 
            "task4_with_nms.jpg"
        )
    
    print("\nTask 4 completed. Check generated images for results.")


def visualize_predictions(image, boxes, scores, labels, output_path):
    """Helper function to visualize predictions"""
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        
        # Draw bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        
        # Draw label
        text = f"Class {label.item()} {score.item():.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([(x1, y1 - text_height), (x1 + text_width, y1)], fill="red")
        draw.text((x1, y1 - text_height), text, fill="white", font=font)
    
    image_draw.save(output_path)
    print(f"Saved visualization to {output_path}")


# ========================================
# MAIN EXECUTION
# ========================================
def main():
    """Main execution function"""
    print("="*60)
    print("LAB6: Object Detection Using Deep Learning")
    print("Complete Implementation of All Tasks")
    print("="*60)
    
    # Set paths - adjust these according to your data location
    images_dir = "data/images"
    labels_dir = "data/labels"
    
    # Check if data directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"\nWARNING: Data directories not found!")
        print(f"Expected: {images_dir} and {labels_dir}")
        print(f"Please create these directories and add your dataset.")
        print(f"\nFor demonstration, I'll create example directory structure...")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        print(f"Created: {images_dir} and {labels_dir}")
        print(f"Please add your images and labels to these directories.")
        return
    
    # Execute all tasks
    print("\n" + "="*60)
    print("SELECT TASK TO RUN:")
    print("1. Task 1: Hyperparameter Exploration")
    print("2. Task 2: Architectural Adaptation and Transfer Learning")
    print("3. Task 3: Data Transformation and Augmentation")
    print("4. Task 4: Evaluation and Inference")
    print("5. Run All Tasks")
    print("="*60)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        task1_hyperparameter_exploration(images_dir, labels_dir)
    elif choice == "2":
        task2_transfer_learning(images_dir, labels_dir)
    elif choice == "3":
        task3_data_augmentation(images_dir, labels_dir)
    elif choice == "4":
        task4_evaluation_inference(images_dir, labels_dir)
    elif choice == "5":
        print("\n" + "="*60)
        print("RUNNING ALL TASKS")
        print("="*60)
        
        # Task 1
        task1_hyperparameter_exploration(images_dir, labels_dir)
        
        # Task 2
        task2_transfer_learning(images_dir, labels_dir)
        
        # Task 3
        task3_data_augmentation(images_dir, labels_dir)
        
        # Task 4
        task4_evaluation_inference(images_dir, labels_dir)
        
        print("\n" + "="*60)
        print("ALL TASKS COMPLETED!")
        print("="*60)
    else:
        print("Invalid choice. Please run again and select 1-5.")


if __name__ == "__main__":
    main()

