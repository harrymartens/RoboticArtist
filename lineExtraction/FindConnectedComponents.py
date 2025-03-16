import cv2 as cv
import numpy as np
import random

def extract_component_lines(line_image):
    ret, thresh = cv.threshold(line_image, 127, 255, cv.THRESH_BINARY)

    no_ccs, labels = cv.connectedComponents(thresh)

    component_lines = {}

    for i in range(1, no_ccs):
        # Create a binary mask for the current component.
        component_mask = np.uint8(labels == i) * 255

        # We use RETR_EXTERNAL to get only the outer contour.
        contours, hierarchy = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            # If multiple contours are found (which can happen if there are holes),
            # we choose the largest one by area.
            contour = max(contours, key=cv.contourArea)
            component_lines[i] = contour

    return component_lines
    
def findConnectedComponents(line_image):
        
    component_contours = extract_component_lines(line_image)
    
    for label, contour in component_contours.items():
        print(f"Component {label}:")
        print(contour)

    canvas = np.zeros((line_image.shape[0], line_image.shape[1], 3), dtype=np.uint8)
    for contour in component_contours.values():
        cv.drawContours(canvas, [contour], -1, (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)), 1)
    cv.imshow("Component Outlines", canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()

    