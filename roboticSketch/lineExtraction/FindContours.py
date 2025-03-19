import cv2 as cv
import numpy as np


def findContours(line_image):
    
    ret, thresh = cv.threshold(line_image, 127, 255, cv.THRESH_BINARY)
    
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("Number of Contours found = " + str(len(contours))) 

    if len(line_image.shape) == 2:
        canvas = np.zeros((line_image.shape[0], line_image.shape[1], 3), dtype=np.uint8)
    else:
        canvas = np.zeros_like(line_image)
    
    
    for i in range(0,len(contours)):
        cnt = contours[i]
        cv.drawContours(canvas, [cnt], 0, (0, 255, 0), 1)
        
        cv.imshow("Contours", canvas)
        cv.waitKey(0)
        cv.destroyAllWindows()
    # print(contours)
    
