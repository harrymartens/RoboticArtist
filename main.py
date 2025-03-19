#!/usr/bin/env python3

import cv2

from speechRecognition.speechInput import recognizeSpeech
# from roboticMovement.patternConstructor import createPatternFromObjects
from generativeImages.generateImage import generateImage
from roboticSketch.processImage import convertImageToLines
from roboticMovement.robotConfig import RoboticArm
from roboticMovement.mapImageToSketch import executeDrawingCommands


# gcloud auth application-default login        
# gcloud auth application-default set-quota-project capable-hash-452722-r5

if __name__ == "__main__":    
    # request = recognizeSpeech()
    
    # request = "A drawing of a cat --in the style of a very simple, single-color line image with all lines being single pixel thickness. The image should consist solely of clear, unbroken black lines without gradients."
    request = "A shark playing a piano--in a rat fink art style. The image should be moderately simple, with a focus on clear outlines"
    
    imageData = generateImage(request)
    
    # imageUrl = "images/obama.jpg"
    # # imageUrl = "images/triangle.png"
    # imageData = cv2.imread(imageUrl)
    
    lineImage = convertImageToLines(imageData)
    
    roboticArm = RoboticArm()

    executeDrawingCommands(roboticArm, lineImage)
    
    # queue = recogniseObjects(request)
    
    # print(queue)
    # createPatternFromObjects(queue)
    
    