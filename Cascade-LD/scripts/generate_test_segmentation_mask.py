import os
import json
import cv2
import numpy as np
from tqdm import tqdm


# ---- CONFIG ----
DATASET_PATH = "../dataset/TUSimple"  # Path to TuSimple dataset
TEST_LABEL_FILE = os.path.join(DATASET_PATH, "test_label.json")  # JSON labels
THICKNESS = 5  # Thickness of lane lines
ORI_HEIGHT, ORI_WIDTH = 720, 1280
TRAIN_HEIGHT, TRAIN_WIDTH = 256, 512

# ---- PROCESS LABEL FILE ----
height_ratio = TRAIN_HEIGHT / ORI_HEIGHT
width_ratio = TRAIN_WIDTH / ORI_WIDTH
with open(TEST_LABEL_FILE, "r") as f:
    annotations = [json.loads(line) for line in f]  # TuSimple stores JSON lines

# ---- GENERATE SEGMENTATION MASKS ----
for ann in tqdm(annotations, desc="Generating masks"):
    img_path = os.path.join(DATASET_PATH, ann["raw_file"])

    # Create an empty mask (background=0)
    mask = np.zeros((TRAIN_HEIGHT, TRAIN_WIDTH), dtype=np.uint8)

    # Iterate over lanes
    for lane_idx, lane in enumerate(ann["lanes"]):
        for i in range(len(lane) - 1):
            x1, y1 = lane[i], ann["h_samples"][i]
            x2, y2 = lane[i + 1], ann["h_samples"][i + 1]

            # Skip invalid points
            if x1 < 0 or x2 < 0:
                continue

            # Draw line on mask
            cv2.line(mask, (int(x1*width_ratio), int(y1*height_ratio)),
                     (int(x2*width_ratio), int(y2*height_ratio)), lane_idx+1, THICKNESS)
    # Save mask
    mask_path = img_path.replace(".jpg", ".png")
    cv2.imwrite(mask_path, mask)

print(f"✅ TEST Segmentation masks save")

# # ---- VISUALIZE MASK ----
# img_path = 'clips/0601/1494453497604532231/20.jpg'
# mask_path = os.path.join(DATASET_PATH, img_path.replace(".jpg", ".png"))
# img = cv2.imread(os.path.join(DATASET_PATH, img_path))
# img = cv2.resize(img, (TRAIN_WIDTH, TRAIN_HEIGHT))
# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# show_image_results(img, mask, TRAIN_HEIGHT, TRAIN_WIDTH)