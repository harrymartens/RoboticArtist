import cv2
import numpy as np


def realistic_Hough_Transform(canny_img):
    lines = cv2.HoughLinesP(canny_img, 1, np.pi/360, 10, minLineLength=1, maxLineGap=2)
    line_image = np.zeros_like(canny_img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
    return line_image, lines


def abstract_Hough_Transform(canny_img):
    lines = cv2.HoughLinesP(canny_img, 1, np.pi/360, 10, minLineLength=1, maxLineGap=10)
    line_image = np.zeros_like(canny_img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
    return line_image, lines
