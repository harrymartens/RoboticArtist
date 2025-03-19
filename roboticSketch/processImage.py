#!/usr/bin/env python3
import cv2

from roboticSketch.initialisation import preprocess_image
from roboticSketch.edgeDetection import cannyLines, realistic_Hough_Transform, abstract_Hough_Transform
from roboticSketch.lineExtraction import findConnectedComponents

# img_path = 'images/shapes.png'
img_path = 'images/obama.jpg'

def show_image(img, image_name="Img"):
    cv2.imshow(image_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Simplify Original Image
# - B&W
# - 64x64
# - Gaussian Filter
# preprocessed_image = preprocess_image(img_path)
# show_image(preprocessed_image, "Grey Image", )


# Extract edges from image:
# line_image = cannyLines(preprocessed_image)
# show_image(line_image, "Line Image", )

# Apply Hough Transform
# hough_line_image, lines = realistic_Hough_Transform(line_image)
# hough_line_image, lines = abstract_Hough_Transform(line_image)
# show_image(hough_line_image, "Hough Image", )


# Skeletonize Extracted Lines
# thinned = cv2.ximgproc.thinning(line_image)
# show_image(thinned, "Thinned Image", )


# countours = findContours(line_image)
# countours = findConnectedComponents(line_image)
# background = threshold_background('images/church.png')
import cv2 as cv
import cv2.ximgproc
from skimage.morphology import skeletonize, thin
import numpy as np

# def skeletonize_image_opencv(image):
#     """
#     Skeletonizes a binary image using OpenCV's thinning function.
    
#     Parameters:
#       image: A binary image (single channel) where foreground pixels are 255 and background 0.
    
#     Returns:
#       A skeletonized image with one-pixel wide lines.
#     """
#     # Ensure the image is binary (if not already)
#     thinned = thin(image)
#     skeletonized = skeletonize(image)
    
#     cv_image_thinned = (thinned.astype(np.uint8)) * 255
#     cv_image_skeletonized = (skeletonized.astype(np.uint8)) * 255
#     show_image(cv_image_thinned, "Thinned Image")
#     show_image(cv_image_skeletonized, "Skeleton Image")


#     return cv_image_skeletonized


def convertImageToLines(image):
    cv2.imwrite('images/cat.jpg', image)
    preprocessed_image = preprocess_image(image)
    line_image = cannyLines(preprocessed_image)
    show_image(line_image, "Line Image" )
    countours = findConnectedComponents(line_image)
    return countours
