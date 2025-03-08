import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def color_lanes(image, classes, i, color, HEIGHT, WIDTH):
    buffer_c1 = np.zeros((HEIGHT, WIDTH))
    buffer_c1[classes == i] = color[0]
    image[:, :, 0] += buffer_c1

    buffer_c2 = np.zeros((HEIGHT, WIDTH))
    buffer_c2[classes == i] = color[1]
    image[:, :, 1] += buffer_c2

    buffer_c3 = np.zeros((HEIGHT, WIDTH))
    buffer_c3[classes == i] = color[2]
    image[:, :, 2] += buffer_c3
    return image

def blend(image_orig, image_classes):
	image_classes = image_classes.astype(np.uint8)
	mask = np.zeros(image_classes.shape)

	mask[image_classes.nonzero()] = 255
	mask = mask[:, :, 0]
	mask = Image.fromarray(mask.astype(np.uint8))
	image_classes = Image.fromarray(image_classes)

	image_orig.paste(image_classes, None, mask)
	return image_orig
def get_image_results(image, seg_image, train_height, train_width):
    # normalize to 0-255
    seg_image = seg_image.reshape(train_height, train_width)
    # plot color lane in image
    out_segmentation_viz = np.zeros((train_height, train_width, 3))
    colors = [(255, 1, 1), (1, 255, 1), (1, 1, 255), (255, 255, 1)]
    for i in range(1, 5):
        out_segmentation_viz = color_lanes(
            out_segmentation_viz, seg_image,
            i, colors[i-1], train_height, train_width)
    image_orig = Image.fromarray(image)
    im_seg = blend(image_orig, out_segmentation_viz)

    return im_seg

DATASET_PATH = "../dataset/TUSimple"  # Path to TuSimple dataset
THICKNESS = 5  # Thickness of lane lines
ORI_HEIGHT, ORI_WIDTH = 720, 1280
TRAIN_HEIGHT, TRAIN_WIDTH = 256, 512

if __name__ == "__main__":
    image_path = '../dataset/TUSimple/clips/0530/1492627171538356342_0/20.jpg'
    mask_path = image_path.replace('jpg','png')
    img = cv2.imread(image_path)
    img = cv2.resize(img, (TRAIN_WIDTH, TRAIN_HEIGHT))
    mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)).astype(np.uint8)
    mask = cv2.resize(mask, (TRAIN_WIDTH, TRAIN_HEIGHT))
    im_seg = get_image_results(img, mask, TRAIN_HEIGHT, TRAIN_WIDTH)
    plt.imshow(im_seg)
    plt.show()