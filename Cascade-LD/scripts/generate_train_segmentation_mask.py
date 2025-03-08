import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random


# ---- CONFIG ----
DATASET_PATH = "../dataset/TUSimple"  # Path to TuSimple dataset
TRAIN_LABEL_FILE_JSON = os.path.join(DATASET_PATH, "tusimple_anno_cache.json")
THICKNESS = 5  # Thickness of lane lines
ORI_HEIGHT, ORI_WIDTH = 720, 1280
TRAIN_HEIGHT, TRAIN_WIDTH = 256, 512

# ---- PROCESS LABEL FILE ----
height_ratio = TRAIN_HEIGHT / ORI_HEIGHT
width_ratio = TRAIN_WIDTH / ORI_WIDTH
with open(TRAIN_LABEL_FILE_JSON, "r") as f:
    annotations = json.load(f)
    for image_path in tqdm(annotations):
        annotation = annotations[image_path]
        mask = np.zeros((TRAIN_HEIGHT, TRAIN_WIDTH), dtype=np.uint8)
        for lane_idx, lane in enumerate(annotation):
            for i in range(len(lane) - 1):
                x1, y1 = lane[i]
                x2, y2 = lane[i + 1]

                if x1 < 0 or x2 < 0:
                    continue
                cv2.line(mask, (int(x1 * width_ratio), int(y1 * height_ratio)), (int(x2 * width_ratio), int(y2 * height_ratio)), lane_idx+1, THICKNESS)
        mask_path = os.path.join(DATASET_PATH, image_path.replace(".jpg", ".png"))
        cv2.imwrite(mask_path, mask)

print(f"✅ TRAIN Segmentation masks saved")