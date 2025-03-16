#!/usr/bin/env python3


import cv2

from initialisation import preprocess_image
from edgeDetection import cannyLines, realistic_Hough_Transform, abstract_Hough_Transform
from lineExtraction import findConnectedComponents

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
preprocessed_image = preprocess_image(img_path)
# show_image(preprocessed_image, "Grey Image", )


# Extract edges from image:
line_image = cannyLines(preprocessed_image)
# show_image(line_image, "Line Image", )

# Apply Hough Transform
# hough_line_image, lines = realistic_Hough_Transform(line_image)
# hough_line_image, lines = abstract_Hough_Transform(line_image)
# show_image(hough_line_image, "Hough Image", )


# Skeletonize Extracted Lines
# thinned = cv2.ximgproc.thinning(line_image)
# show_image(thinned, "Thinned Image", )


# countours = findContours(line_image)
countours = findConnectedComponents(line_image)
# background = threshold_background('images/church.png')



