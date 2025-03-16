#!/usr/bin/env python3

import math
import random
from abc import ABC, abstractmethod
from xarm.wrapper import XArmAPI
from roboticMovement.emotions import EmotionHappy
from roboticMovement.shapes import Shape, Location, Triangle, Square, Circle, Dot, Line, Rectangle, Polygon, Spiral, Star, Cube, ConcentricCircles, RandomDotsPattern, GridPattern
from roboticMovement.robotConfig import RoboticArm  


def get_location_coords(location_str):
    """
    Convert a location string into a Location object using the following ranges:
      min_x = 150, max_x = 323,
      max_y (top) = 150, min_y (bottom) = -117.
      
    The center is computed as:
      center_x = (min_x + max_x) / 2
      center_y = (max_y + min_y) / 2
      
    Returns a Location object with coordinates based on the input string.
    """
    min_x = 150
    max_x = 323
    min_y = -117
    max_y = 150
    center_x = (min_x + max_x) / 2   
    center_y = (max_y + min_y) / 2 

    if not location_str:
        return Location()
    
    location_str = location_str.lower()
    if location_str == "left":
        return Location(x=min_x, y=center_y)
    elif location_str == "right":
        return Location(x=max_x, y=center_y)
    elif location_str == "center":
        return Location(x=center_x, y=center_y)
    elif location_str == "top":
        return Location(x=center_x, y=max_y)
    elif location_str == "bottom":
        return Location(x=center_x, y=min_y)
    else:
        return Location()

def createPatternFromObjects(shape_defs):
    """
    Takes an array of dictionaries with keys:
      'shape', 'quantity', 'size', and 'location'
    and creates the corresponding shape objects.
    Then, it calls the draw() method on each shape.
    
    Example input:
      [{'shape': 'triangle', 'quantity': 1, 'size': 'large', 'location': None}]
    """
    shapes = []
    
    for shape_def in shape_defs:
        
        shape_type = shape_def.get("shape", "").lower()
        if shape_type:
            quantity = shape_def.get("quantity", 1)
            size = shape_def.get("size", None)
            location_str = shape_def.get("location", None)
        
        emotionType = shape_def.get("emotion", "").lower()
        if emotionType:
            if emotionType == "happy":
                shapes.append(EmotionHappy())
            elif emotionType == "sad":
                print("Draw Sad Picture")
            elif emotionType == "angry":
                print("Draw Angry Picture")
            elif emotionType == "surprised":
                print("Draw Surprised Picture")
            elif emotionType == "fearful":
                print("Draw Fearful Picture")
            elif emotionType == "disgusted":    
                print("Draw Disgusted Picture")
            elif emotionType == "excited":
                print("Draw Excited Picture")
            elif emotionType == "calm":
                print("Draw Calm Picture")
            elif emotionType == "anxious":
                print("Draw Anxious Picture")
            elif emotionType == "bored":
                print("Draw Bored Picture")
            elif emotionType == "confused":
                print("Draw Confused Picture")
            elif emotionType == "hopeful":
                print("Draw Hopeful Picture")
            elif emotionType == "content":
                print("Draw Content Picture")
            elif emotionType == "frustrated":
                print("Draw Frustrated Picture")
            elif emotionType == "melancholy":
                print("Draw Melancholy Picture")
            elif emotionType == "serene":
                print("Draw Serene Picture")
            else:
                print("Unknown emotion type: {emotionType}")
            continue
            
            
        if size:
            size = size.lower()
        else:
            size = "medium"
            
        if (quantity == "many" or quantity == "lots"):
            quantity = random.randint(5, 10)
        elif quantity == "few" or quantity == "couple" or quantity == "some":
            quantity = random.randint(2, 5)
        elif quantity == "several":
            quantity = random.randint(3, 7)
        
        for _ in range(quantity):
            
            loc = get_location_coords(location_str)
            
            if size == "small":
                sideLength = random.uniform(10, 20)
            elif size == "large":
                sideLength = random.uniform(30, 50)
            else:
                sideLength = random.uniform(20, 30)
                   
            if shape_type == "circle":
                radius = sideLength
                shape_instance = Circle(loc, radius=radius, steps=36)

            elif shape_type == "triangle":
                shape_instance = Triangle(loc, length=sideLength)

            elif shape_type == "square":
                shape_instance = Square(loc, length=sideLength)

            elif shape_type == "dot":
                shape_instance = Dot(loc)

            elif shape_type == "line":
                # Create a line from loc to a new point offset by sideLength in the x-direction.
                end_loc = Location(loc.x + sideLength, loc.y)
                shape_instance = Line(start=loc, end=end_loc)

            elif shape_type == "rectangle":
                width = sideLength
                height = sideLength / 2  # Adjust ratio as needed.
                shape_instance = Rectangle(loc, width=width, height=height)

            elif shape_type == "polygon":
                shape_instance = Polygon(loc, random.uniform(3,8), radius=sideLength)

            elif shape_type == "spiral":
                start_radius = sideLength * 0.2
                shape_instance = Spiral(loc, start_radius=start_radius, end_radius=sideLength, revolutions=3)

            elif shape_type == "star":
                shape_instance = Star(loc, points=5, outer_radius=sideLength, inner_radius=sideLength / 2)

            elif shape_type == "cube":
                offset = (sideLength * 0.2, -sideLength * 0.2)
                shape_instance = Cube(loc, side_length=sideLength, offset=offset)

            elif shape_type == "concentriccircles":
                shape_instance = ConcentricCircles(loc, base_radius=sideLength, count=3, step=sideLength / 2)

            elif shape_type == "random dots":
                # Use sideLength to determine the number of dots (e.g., 1 dot per unit length).
                dot_count = int(sideLength)
                shape_instance = RandomDotsPattern(loc, dot_count=dot_count)

            elif shape_type == "grid pattern":
                # Create a 3x3 grid with spacing equal to sideLength.
                shape_instance = GridPattern(loc, rows=3, cols=3, spacing=sideLength)

            else:
                print(f"Unknown shape type: {shape_type}")
                continue
            
            shapes.append(shape_instance)
    
    for shape in shapes:
        shape.draw()
    
def drawShapes():
    for i in range(10):
        square = Square(length=random.uniform(10,40),rotation=random.uniform(0,360) )
        triange = Triangle(length=random.uniform(10,40),rotation=random.uniform(0,360) )
        circle = Circle(radius=random.uniform(10,40) )
        square.draw()
        triange.draw()
        circle.draw()


if __name__ == "__main__":    
    loc = Location(x=100, y=100)
    circle = Circle(loc, radius=10, steps=36)
    circle.draw()
    
    loc2 = Location(150, 200)
    square = Square(length=20, rotation=30 )
    square.draw()
    
