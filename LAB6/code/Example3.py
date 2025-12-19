# Step 1: Import Necessary Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
from tqdm import tqdm

# -----------------------------
# Step 2: Custom Dataset 
# -----------------------------
class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        # Case-insensitive image extension check
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]

        if len(self.images) == 0:
            raise ValueError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Correctly get corresponding txt label
        base_name = os.path.splitext(img_name)[0]   # remove .JPG or .jpg
        label_path = os.path.join(self.labels_dir, base_name + ".txt")

        # Load image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Load labels
        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f.readlines():
                cls, x_center, y_center, w, h = map(float, line.strip().split())
                x1 = (x_center - w/2) * width
                y1 = (y_center - h/2) * height
                x2 = (x_center + w/2) * width
                y2 = (y_center + h/2) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls)+1)  # 0 = background

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img = self.transform(img)

        return img, target


# -----------------------------
# Step 3: Paths and DataLoader
# -----------------------------
images_dir = "data/images"
labels_dir = "data/labels"

transform = transforms.ToTensor()

dataset = CustomDataset(images_dir, labels_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# -----------------------------
# Step 4: Load Pretrained Faster R-CNN Model
# -----------------------------
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# -----------------------------
# Step 5: Define Optimizer
# -----------------------------
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -----------------------------
# Step 6: Training Loop
# -----------------------------
num_epochs = 2
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for images, targets in tqdm(train_loader):
        images = list(images)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

# -----------------------------
# Step 7: Testing / Inference
# -----------------------------
model.eval()
with torch.no_grad():
    for images, _ in test_loader:
        images = list(images)
        predictions = model(images)
        print(predictions)
        break

# -----------------------------
# Step 8: Save Model Checkpoint
# -----------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs
}, 'fasterrcnn_custom_checkpoint.pth')
