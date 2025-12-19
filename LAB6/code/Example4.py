import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
from tqdm import tqdm

# -----------------------------
# Custom Dataset
# -----------------------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        if len(self.images) == 0:
            raise ValueError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.labels_dir, base_name + ".txt")

        # Load image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Load YOLO-style labels
        boxes, labels = [], []
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
# Dataset and DataLoader
# -----------------------------
images_dir = "data/images"
labels_dir = "data/labels"
transform = transforms.ToTensor()

dataset = CustomDataset(images_dir, labels_dir, transform=transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# -----------------------------
# Load Pretrained Faster R-CNN
# -----------------------------
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace box predictor for 1 class + background
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# -----------------------------
# 5.1 Freeze Backbone
# -----------------------------
for param in model.backbone.parameters():
    param.requires_grad = False  # freeze backbone

for param in model.roi_heads.parameters():
    param.requires_grad = True   # only train ROI heads

# -----------------------------
# 5.2 Optimizer (ROI heads only)
# -----------------------------
optimizer = torch.optim.Adam(model.roi_heads.parameters(), lr=1e-4)

# -----------------------------
# Training Loop (demo)
# -----------------------------
num_epochs = 3
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for images, targets in tqdm(loader):
        images = list(images)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

# -----------------------------
# 5.3 Optional: Gradual Unfreeze
# -----------------------------
# Unfreeze last layer of backbone for fine-tuning
for name, param in model.backbone.body.layer4.named_parameters():
    param.requires_grad = True

# Define optimizer for backbone + ROI heads with different learning rates
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.roi_heads.parameters(), 'lr': 1e-4}
])

# -----------------------------
# Save model
# -----------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, "fasterrcnn_custom_transfer.pth")
