import os

import cv2
import numpy as np

def histogram_equalization(image):
    equalized = cv2.equalizeHist(image)
    return equalized

def clahe(image, clip_limit=2.0, grid_size=8):
    tile_grid_size= (grid_size,grid_size)
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(image)
    return clahe_image

def resize_image(image, width=300):
    height = int(image.shape[0] * (width / image.shape[1]))
    return cv2.resize(image, (width, height))


def load_images(image_folder):
    images, titles = [], []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            images.append(cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE))
            titles.append(filename)
    return images, titles
