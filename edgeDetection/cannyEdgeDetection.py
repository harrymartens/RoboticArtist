import cv2

def cannyLines(img):
    return cv2.Canny(img, 50, 100)