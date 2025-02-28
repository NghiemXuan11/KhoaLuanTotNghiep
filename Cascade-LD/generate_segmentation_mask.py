import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# ---- CONFIG ----
DATASET_PATH = "../dataset/TUSimple"  # Path to TuSimple dataset
LABEL_FILE = os.path.join(DATASET_PATH, "test_labels.json")  # JSON labels

# ---- PROCESS LABEL FILE ----
with open(LABEL_FILE, "r") as f:
    annotations = [json.loads(line) for line in f]  # TuSimple stores JSON lines

# ---- GENERATE SEGMENTATION MASKS ----
for ann in tqdm(annotations, desc="Generating masks"):
    img_path = os.path.join(DATASET_PATH, ann["raw_file"])
    
    # Load image to get dimensions
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # Create an empty mask (background=0)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Iterate over lanes
    for lane_idx, lane in enumerate(ann["lanes"]):
        for i in range(len(lane) - 1):
            x1, y1 = lane[i], ann["h_samples"][i]
            x2, y2 = lane[i + 1], ann["h_samples"][i + 1]

            # Skip invalid points
            if x1 < 0 or x2 < 0:
                continue
            
            # Draw line on mask
            cv2.line(mask, (x1, y1), (x2, y2), color=lane_idx + 1, thickness=5)

    # Save mask
    mask_path = os.path.join(DATASET_PATH, img_path.replace(".jpg", ".png"))
    cv2.imwrite(mask_path, mask)

print(f"✅ Segmentation masks save")
