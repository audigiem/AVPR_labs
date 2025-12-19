import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import torchvision.models.detection as detection

# -----------------------------
# 1. Load Pretrained Model
# -----------------------------
# model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# model = detection.ssd300_vgg16(weights="DEFAULT")
model = detection.retinanet_resnet50_fpn(weights="DEFAULT")

model.eval()

# -----------------------------
# 2. Load & Preprocess Image
# -----------------------------
img_path = "family-and-dog.jpg"
image = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.ToTensor()
])

img_tensor = transform(image).unsqueeze(0)

# -----------------------------
# 3. Run Inference
# -----------------------------
with torch.no_grad():
    predictions = model(img_tensor)

# -----------------------------
# 4. Draw Bounding Boxes & Save
# -----------------------------
boxes = predictions[0]["boxes"]
scores = predictions[0]["scores"]

draw = ImageDraw.Draw(image)

# confidence threshold
threshold = 0.5

for box, score in zip(boxes, scores):
    if score >= threshold:
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

# Save output image
output_path = "output.jpg"
image.save(output_path)

print(f"Saved output image to {output_path}")
