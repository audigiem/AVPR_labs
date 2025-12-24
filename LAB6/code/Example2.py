import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torchvision
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

# -------------------------------
# 1. Load pretrained Faster R-CNN
# -------------------------------
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

classes = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

# -------------------------------
# 2. Load and preprocess image
# -------------------------------
img_path = "family-and-dog.jpg"
image = Image.open(img_path).convert("RGB")
image_draw = image.copy()  # copy for drawing

transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(image).unsqueeze(0)

# -------------------------------
# 3. Run detection
# -------------------------------
with torch.no_grad():
    output = model(img_tensor)[0]

boxes = output["boxes"]
labels = output["labels"]
scores = output["scores"]

# -------------------------------
# 4. Apply NMS
# -------------------------------
keep = nms(boxes, scores, iou_threshold=0.5)

# -------------------------------
# 5. Draw boxes + labels
# -------------------------------
draw = ImageDraw.Draw(image_draw)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

threshold = 0.5
for idx in keep:
    score = scores[idx].item()
    if score >= threshold:
        box = boxes[idx]
        label_id = labels[idx].item()
        class_name = classes[label_id]

        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        text = f"{class_name} {score:.2f}"

        # Use draw.textbbox to get text size
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw text background
        draw.rectangle([(x1, y1 - text_height), (x1 + text_width, y1)], fill="red")
        # Draw text
        draw.text((x1, y1 - text_height), text, fill="white", font=font)

# -------------------------------
# 6. Save output image
# -------------------------------
output_path = "output_family_and_dog.jpg"
image_draw.save(output_path)
print(f"Saved labeled image as {output_path}")

# -------------------------------
# 7. Print some detections
# -------------------------------
print("\nFinal detections (top 5):")
for idx in keep[:5]:
    print(
        f"Box={boxes[idx].tolist()}  Class={classes[labels[idx]]}  Score={scores[idx]:.2f}"
    )
