from tqdm import tqdm
import os
import json
import numpy as np
import cv2
from show_img_seg_label import get_image_results
import matplotlib.pyplot as plt


# ---- CONFIG ----
DATASET_PATH = "TUSimple"  # Path to TuSimple dataset
TRAIN_LABEL_FILE = os.path.join(DATASET_PATH, "tusimple_anno_cache.json")  # Path to the label file
TEST_LABEL_FILE = os.path.join(DATASET_PATH, "test_label.json")
ORI_HEIGHT, ORI_WIDTH = 720, 1280
TRAIN_HEIGHT, TRAIN_WIDTH = 360, 640
THICKNESS = 3
height_ratio = TRAIN_HEIGHT / ORI_HEIGHT
width_ratio = TRAIN_WIDTH / ORI_WIDTH

def generate_segmentation_mask(lines, path):
    mask = np.zeros((TRAIN_HEIGHT, TRAIN_WIDTH), dtype=np.uint8)
    for lane_idx, lane in enumerate(lines):
        for i in range(len(lane) - 1):
            x1, y1 = lane[i]
            x2, y2 = lane[i + 1]
            cv2.line(mask, (int(x1 * width_ratio), int(y1 * height_ratio)),
                     (int(x2 * width_ratio), int(y2 * height_ratio)), lane_idx + 1, THICKNESS)
    cv2.imwrite(path, mask)

def get_vector(line):
    if len(line) < 2:
        return line
    index_point_center_right = len(line) // 2
    left_point = line[:index_point_center_right]
    right_point = line[index_point_center_right:]
    left_point = np.mean(left_point, axis=0)
    right_point = np.mean(right_point, axis=0)
    vector = right_point - left_point
    return vector

def get_angle(v1, v2, degrees=True):
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Tính tích vô hướng (dot product)
    dot_product = np.dot(v1, v2)
    
    # Tính độ lớn của từng vector
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Một trong hai vector có độ lớn bằng 0, không thể tính góc.")

    # Tính cos(góc) và góc
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Giới hạn để tránh sai số

    # Chuyển sang độ nếu cần
    if degrees:
        angle = np.degrees(angle)

    return angle


def check_lines_valid_train(lines):
    exits_line_0 = int(len(lines[0]) > 0)
    exits_line_3 = int(len(lines[3]) > 0)
    if exits_line_0 and get_angle(get_vector(lines[0]), [-1,0]) > 90:
        return False
    if len(lines[1]) == 0 or get_angle(get_vector(lines[1]), [-1,0]) > 90:
        return False
    if len(lines[2]) == 0 or get_angle(get_vector(lines[2]), [-1,0]) < 90:
        return False
    if exits_line_3 and get_angle(get_vector(lines[3]), [-1,0]) < 90:
        return False
    return str(exits_line_0)+' 1 1 '+str(exits_line_3)

# ---- MAIN ----

# ---- FILTER SAMPLE TRAIN ----
output_path = os.path.join(DATASET_PATH, "train_gt_filtered.txt")
output_filter_path = os.path.join(DATASET_PATH, "filter_train_sample.txt")
output_file = open(output_path, "w")
output_filter_file = open(output_filter_path, "w")
with open(TRAIN_LABEL_FILE, "r") as f:
    annotations = json.load(f)
    for image_path in tqdm(annotations, desc="Filter train masks"):
        annotation = annotations[image_path]
        lines = [[] for _ in range(4)]
        for lane_idx, lane in enumerate(annotation):
            for i in range(len(lane) - 1):
                x1, y1 = lane[i]
                x2, y2 = lane[i + 1]

                if x1 < 0 or x2 < 0:
                    continue
                if len(lines[lane_idx])==0:
                    lines[lane_idx].append([x1, y1])
                lines[lane_idx].append([x2,y2])
        img_path = image_path
        mask_path = img_path.replace('jpg','png')
        check = check_lines_valid_train(lines)
        line_txt = img_path+' '+mask_path
        if not check:
            output_filter_file.write(line_txt + "\n")
            continue
        generate_segmentation_mask(lines, os.path.join(DATASET_PATH, mask_path))
        output_file.write(line_txt +' '+ check + "\n")
output_file.close()
output_filter_file.close()
# ---- FILTER SAMPLE TEST ----
def check_lines_valid_test(angles):
    if len(angles) < 2:
        return False
    angles_gt=[0,0,0,0]
    lines_gt = [[] for _ in range(4)]

    if len(angles) == 2:
        angles_gt[1], angles_gt[2] = angles[0], angles[1]
        lines_gt[1], lines_gt[2] = lines[0], lines[1]
    if len(angles) == 4:
        angles_gt = angles
        lines_gt = lines
    if (len(angles) == 2 or len(angles) == 4):
        if angles_gt[1] > 90 or angles_gt[2] < 90:
            return False
        if angles_gt[0]:
            if angles_gt[0] > 90:
                return False
            else:
                lines_gt[0] = lines[0]
        if angles_gt[3]:
            if angles_gt[3] < 90:
                return False
            else:
                lines_gt[3] = lines[3]
    if len(angles) == 3:
        if angles[0] < 90 and angles[1] < 90 and angles[2] > 90:
            lines_gt[:3] = lines
        elif angles[0] < 90 and angles[1] > 90 and angles[2] > 90:
            lines_gt[1:] = lines
        else: return False
    return lines_gt

output_path = os.path.join(DATASET_PATH, "test_gt_filtered.txt")
output_filter_path = os.path.join(DATASET_PATH, "filter_test_sample.txt")
output_file = open(output_path, "w")
output_filter_file = open(output_filter_path, "w")
with open(TEST_LABEL_FILE, "r") as f:
    annotations = [json.loads(line) for line in f]  # TuSimple stores JSON lines
    # ---- GENERATE SEGMENTATION MASKS ----
    for ann in tqdm(annotations, desc="Filter test masks"):
        lines = []

        # Iterate over lanes
        for lane_idx, lane in enumerate(ann["lanes"]):
            for i in range(len(lane) - 1):
                x1, y1 = lane[i], ann["h_samples"][i]
                x2, y2 = lane[i + 1], ann["h_samples"][i + 1]
                
                if x1 > 0 and len(lines) == lane_idx:
                    lines.append([])
                    lines[-1].append([x1, y1])

                if x2 > 0:
                    if len(lines) == lane_idx:
                        lines.append([])
                    lines[-1].append([x2, y2])

        
        img_path = ann["raw_file"]
        mask_path = img_path.replace('jpg','png')
        angles = [get_angle(get_vector(line), [-1, 0]) for line in lines]
        lines_gt = check_lines_valid_test(angles)
        line_txt = img_path+' '+mask_path
        if not lines_gt:
            # name_file = ann["raw_file"].replace('/','_')
            # img = cv2.imread(os.path.join(DATASET_PATH, img_path))
            # img = cv2.resize(img, (TRAIN_WIDTH, TRAIN_HEIGHT))
            # mask = cv2.imread(os.path.join(DATASET_PATH, mask_path), cv2.IMREAD_GRAYSCALE)
            # mask = cv2.resize(mask, (TRAIN_WIDTH, TRAIN_HEIGHT))
            # im_seg = get_image_results(img, mask, TRAIN_HEIGHT, TRAIN_WIDTH)
            # cv2.imwrite(DATASET_PATH+"/test_filter/"+name_file, np.array(im_seg))
            
            output_filter_file.write(line_txt + "\n")
            continue
        generate_segmentation_mask(lines_gt, os.path.join(DATASET_PATH, mask_path))
        output_file.write(line_txt + "\n")
output_file.close()
output_filter_file.close()

        