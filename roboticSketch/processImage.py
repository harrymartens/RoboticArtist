#!/usr/bin/env python3
import cv2
from roboticSketch.imageProcessing import preprocess_image, canny_lines, findConnectedComponents



def convertImageToLines(image):
    preprocessed_image = preprocess_image(image)
    cannyImage = canny_lines(preprocessed_image)

    countours, lineImage = findConnectedComponents(cannyImage)
    return countours, lineImage
