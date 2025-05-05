# Object Detection with YOLO (You Only Look Once)

# In this Colab Notebook, we'll implement object detection using YOLOv8.

# --- 1. Setup and Installation ---
!pip install ultralytics opencv-python matplotlib

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import random
from google.colab import files
import torch
from collections import Counter

plt.rcParams['figure.figsize'] = [12, 8]
plt.style.use('ggplot')

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")

# --- 2. Dataset Preparation ---
!mkdir -p images

coco_image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000087038.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "http://images.cocodataset.org/val2017/000000062808.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
    "http://images.cocodataset.org/val2017/000000578967.jpg",
    "http://images.cocodataset.org/val2017/000000185590.jpg",
]

image_paths = []
for i, url in enumerate(coco_image_urls):
    try:
        response = requests.get(url)
        img_path = f"images/coco_{i+1}.jpg"
        with open(img_path, 'wb') as f:
            f.write(response.content)
        image_paths.append(img_path)
        print(f"Downloaded: {img_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# --- Optional: Upload your own images ---
# uploaded = files.upload()
# for filename in uploaded.keys():
#     img_path = f"images/{filename}"
#     with open(img_path, 'wb') as f:
#         f.write(uploaded[filename])
#     image_paths.append(img_path)
#     print(f"Uploaded: {img_path}")

def display_sample_images(image_paths, num_samples=4):
    samples = random.sample(image_paths, min(num_samples, len(image_paths)))
    fig, axes = plt.subplots(1, len(samples), figsize=(16, 4))
    if len(samples) == 1:
        axes = [axes]
    for i, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

display_sample_images(image_paths)

# --- 3. Load Pretrained YOLOv8 Model ---
model = YOLO('yolov8n.pt')
print(f"Model loaded: YOLOv8n")
print(f"Model Parameters: {model.model.num_params_trainable()} trainable parameters")
print(f"Model Classes: {len(model.names)}")

print("\nExample classes:")
sample_classes = random.sample(list(model.names.items()), min(15, len(model.names)))
for idx, name in sample_classes:
    print(f"{idx}: {name}")

# --- 4. Run Inference ---
def run_inference(model, image_paths):
    results = {}
    for img_path in image_paths:
        result = model(img_path)
        results[img_path] = result[0]
    return results

detection_results = run_inference(model, image_paths)
print(f"Inference completed for {len(detection_results)} images.")

def display_detection_results(results, image_paths, num_samples=4):
    samples = random.sample(image_paths, min(num_samples, len(image_paths)))
    fig, axes = plt.subplots(1, len(samples), figsize=(20, 5))
    if len(samples) == 1:
        axes = [axes]
    for i, img_path in enumerate(samples):
        result = results[img_path]
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Detections: {len(boxes)}")
        axes[i].axis('off')
        for j in range(len(boxes)):
            box = boxes[j]
            label = model.names[classes[j]]
            confidence = confidences[j]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='green', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(box[0], box[1] - 5, f"{label}: {confidence:.2f}",
                         color='white', fontsize=8, bbox=dict(facecolor='green', alpha=0.7))
    plt.tight_layout()
    plt.show()

display_detection_results(detection_results, image_paths)

# --- 5. Exploratory Data Analysis (EDA) ---
def analyze_detections(results):
    all_classes = []
    all_confidences = []
    detections_per_image = []
    for img_path, result in results.items():
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        all_classes.extend([model.names[cls] for cls in classes])
        all_confidences.extend(confidences)
        detections_per_image.append(len(classes))
    return all_classes, all_confidences, detections_per_image

all_classes, all_confidences, detections_per_image = analyze_detections(detection_results)
class_counts = Counter(all_classes)
top_classes = class_counts.most_common(10)

plt.figure(figsize=(10, 6))
classes, counts = zip(*top_classes)
plt.bar(classes, counts)
plt.title('Top 10 Detected Object Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(all_confidences, bins=20, alpha=0.7)
plt.title('Distribution of Confidence Scores')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.axvline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(len(detections_per_image)), detections_per_image)
plt.title('Number of Detections per Image')
plt.xlabel('Image Index')
plt.ylabel('Detections')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

avg_detections = np.mean(detections_per_image)
avg_confidence = np.mean(all_confidences) if all_confidences else 0

print(f"Total detections: {len(all_classes)}")
print(f"Avg detections/image: {avg_detections:.2f}")
print(f"Avg confidence score: {avg_confidence:.4f}")
print(f"Unique object classes detected: {len(class_counts)}")
