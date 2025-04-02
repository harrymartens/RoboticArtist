#!/usr/bin/env python3

import cv2
import numpy as np

from speechRecognition.speechInput import recognizeSpeech
from generativeImages.generateImage import generateImage

from roboticSketch.processImage import convertImageToLines
from roboticSketch.imageProcessing.imagePreProcessing import displayImages
from roboticMovement.robotConfig import RoboticArm
from roboticMovement.mapImageToSketch import executeDrawingCommands


# gcloud auth application-default login        
# gcloud auth application-default set-quota-project capable-hash-452722-r5

def speech():
    request = recognizeSpeech()
    return request

def write_prompt():
    request = input("Please enter your prompt: ")
    return request

def upload_image():
    print("You selected Upload Image.")
    
def generate_calibration_grid(min_x, max_x, min_y, max_y, num_columns, num_rows):
    """
    Generates a list of center coordinates for calibration plus symbols,
    evenly spaced across the drawing area.
    """
    calibration_points = []
    x_spacing = (max_x - min_x) / (num_columns - 1)
    y_spacing = (max_y - min_y) / (num_rows - 1)
    for i in range(num_columns):
        for j in range(num_rows):
            center_x = min_x + i * x_spacing
            center_y = min_y + j * y_spacing
            calibration_points.append((center_x, center_y))
    return calibration_points

def draw_plus(arm, center, plus_size):
    """
    Uses the robot's drawing functions to draw a plus symbol at the given center.
    It draws a horizontal line and then a vertical line.
    """
    print(f"Drawing plus at center: {center}")

    half = plus_size / 2
    # --- Draw horizontal segment ---
    # Move to the left endpoint with the pen raised.
    left_x = center[0] - half
    left_y = center[1]
    arm.set_position(left_x, left_y, draw=False)
    # Lower the pen and draw to the right endpoint.
    arm.set_position(left_x, left_y, draw=True)
    right_x = center[0] + half
    right_y = center[1]
    arm.set_position(right_x, right_y, draw=True)
    # Lift the pen.
    arm.set_position(right_x, right_y, draw=False)
    
    # --- Draw vertical segment ---
    # Move to the top endpoint with the pen raised.
    top_x = center[0]
    top_y = center[1] + half
    arm.set_position(top_x, top_y, draw=False)
    # Lower the pen and draw down to the bottom endpoint.
    arm.set_position(top_x, top_y, draw=True)
    bottom_x = center[0]
    bottom_y = center[1] - half
    arm.set_position(bottom_x, bottom_y, draw=True)
    # Lift the pen.
    arm.set_position(bottom_x, bottom_y, draw=False)

if __name__ == "__main__":
    
    options = {
        "1": speech,
        "2": write_prompt,
        "3": upload_image,
    }
    
    choice = input("""
        ====================================================
            Choose Your Interaction
        ====================================================

        Select prompt method:
        [1] Speech
        [2] Write Prompt
        [3] Upload Image

        Enter your choice (1/2/3): 
          """)
    
    
    
    # imageUrl = "images/bird.png"
    # generativeImage = cv2.imread(imageUrl)
    
    while True:
        prompt = options.get(choice, lambda: print("Invalid selection."))() 
        
        generativeImage = generateImage(prompt)
        contours, lineImage = convertImageToLines(generativeImage)
        drawImage = displayImages(generativeImage, lineImage)

        if drawImage:
            roboticArm = RoboticArm()
            executeDrawingCommands(roboticArm, contours)
            break
        else:
            continue
    
    
    # Define your drawing boundaries (robot coordinates).
    
    # arm = RoboticArm()

    # drawing_min_x = 120
    # drawing_max_x = 350
    # drawing_min_y = -170
    # drawing_max_y = 150

    # # Calibration grid parameters.
    # num_columns = 5  # Number of plus symbols along the x-axis.
    # num_rows = 5     # Number of plus symbols along the y-axis.
    # plus_size = 10   # Length of each segment in the plus symbol.

    # # Generate calibration grid points.
    # calibration_points = generate_calibration_grid(drawing_min_x, drawing_max_x,
    #                                             drawing_min_y, drawing_max_y,
    #                                             num_columns, num_rows)

    # # Iterate over each calibration point and draw a plus symbol.
    # for center in calibration_points:
    #     draw_plus(arm, center, plus_size)
    
