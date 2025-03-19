import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import math


def load_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to load image.")
        return
    
    return image

def to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def scale_image(image, target_width=200, target_height=200):
    original_height, original_width = image.shape[:2]
    
    aspect_ratio = original_width / original_height
    
    if original_width > original_height:
        new_width = min(target_width, int(target_height * aspect_ratio))
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(target_height, int(target_width / aspect_ratio))
        new_width = int(new_height * aspect_ratio)
    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image

def apply_gaussian_smoothing(image, sigma=1):
    smoothed_image = cv2.GaussianBlur(image, (5, 5), sigma)
    return smoothed_image


def preprocess_image(image):
    
    scaled_image = scale_image(image, 500, 500)
    gray_image = to_grayscale(scaled_image)
    smoothed_image = apply_gaussian_smoothing(gray_image)
    return smoothed_image