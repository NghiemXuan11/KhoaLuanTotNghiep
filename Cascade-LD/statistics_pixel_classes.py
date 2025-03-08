import numpy as np

from data.dali_data import TrainCollect
import tqdm
import torch
import matplotlib.pyplot as plt

DATA_ROOT = '../dataset/TUSimple'
loader = TrainCollect(32, 4, DATA_ROOT, DATA_ROOT + '/train_gt_filtered.txt', 512, 256, 1, 4)

num_pixels_per_class = [[] for _ in range(5)]
for data in tqdm.tqdm(loader):
    seg_labels = data['seg_labels']
    seg_labels = seg_labels.argmax(dim=1)
    for seg_label in seg_labels:
        for i in range(5):
            num_pixels = (seg_label == i).sum().item()
            num_pixels_per_class[i].append(num_pixels)

for i in range(5):
    print(f'Class {i} mean, std, min, max: {np.mean(num_pixels_per_class[i]):.2f}, {np.std(num_pixels_per_class[i]):.2f}, {np.min(num_pixels_per_class[i])}, {np.max(num_pixels_per_class[i])}')

# # save the results list
# with open('statistics_pixel_classes.txt', 'w') as f:
#     for