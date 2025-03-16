#!/usr/bin/env python3

import math
import random
from abc import ABC, abstractmethod
import spacy
from collections import deque
import json
import re
from spacy.util import minibatch, compounding
from roboticMovement.shapes import (
    Circle, Triangle, Square, Dot, Line, Rectangle, Polygon, Spiral, Star, Cube, 
    ConcentricCircles, RandomDotsPattern, GridPattern
)

# Load spaCy model (using the small English model; you can change this if needed)
nlp = spacy.load("en_core_web_sm")

# Define keywords for our domain. Note that multi-word shape keywords are included.
SHAPE_KEYWORDS = {
    "circle",
    "triangle",
    "square",
    "dot",
    "line",
    "rectangle",
    "polygon",
    "spiral",
    "star",
    "cube",
    "concentric circles",
    "random dots",
    "grid pattern",
}

# SHAPE_KEYWORDS = {
#     "circle": Circle,
#     "triangle": Triangle,
#     "square": Square,
#     "dot": Dot,
#     "line": Line,
#     "rectangle": Rectangle,
#     "polygon": Polygon,
#     "spiral": Spiral,
#     "star": Star,
#     "cube": Cube,
#     "concentriccircles": ConcentricCircles,
#     "randomdotspattern": RandomDotsPattern,
#     "gridpattern": GridPattern,
# }

SIZE_KEYWORDS = {"small", "medium", "big", "tiny", "regular", "large",  "huge"}
# SIZE_KEYWORDS = {
#     "tiny":random.uniform(1, 10),
#     "small":random.uniform(10, 20),
#     "medium":random.uniform(20, 30),
#     "regular":random.uniform(20, 30),
#     "big":random.uniform(30, 50),
#     "large":random.uniform(30, 50),
#     "huge":random.uniform(50, 100)
#     }

LOCATION_KEYWORDS = {"left", "right", "center", "top", "bottom"}
EMOTION_KEYWORDS = {
    "happy", "sad", "angry", "surprised", "fearful", "disgusted", "excited",
    "calm", "anxious", "bored", "confused", "hopeful", "content", "frustrated",
    "melancholy", "serene"
}

QUANTITY_KEYWORDS = {"many", "lots", "few", "couple", "some", "several"}

# Optional mapping for number words
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def recogniseObjects(input_str):
    """
    Process the input string using spaCy to extract drawing commands.
    Returns a deque of command dictionaries, each containing:
      - shape: one of the shape keywords (supports multi-word phrases)
      - quantity: an integer (default 1 if not specified)
      - size: size descriptor if available (e.g., 'small', 'big')
      - location: location descriptor if available (e.g., 'left', 'right')
    
    Noun chunks are used to group tokens into candidate commands.
    """
    doc = nlp(input_str.lower())
    commandQueue = deque()

    # Process each noun chunk as a candidate command
    for chunk in doc.noun_chunks:
        shape = None
        quantity = 1  # default quantity
        size = None
        location = None
        emotion = None

        # Check the entire chunk text for multi-word shape keywords.
        chunk_text = chunk.text.lower()
        for keyword in SHAPE_KEYWORDS:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, chunk_text):
                shape = keyword
                break

        # Process individual tokens for numbers, size, location, and emotion.
        for token in chunk:
            if token.like_num:
                try:
                    quantity = int(token.text)
                except ValueError:
                    quantity = NUM_WORDS.get(token.text, 1)
            # If not already found via multi-word check, try token lemma.
            if not shape and token.lemma_ in SHAPE_KEYWORDS:
                shape = token.lemma_
            if token.lemma_ in EMOTION_KEYWORDS:
                emotion = token.lemma_
            if token.text in SIZE_KEYWORDS:
                size = token.text
            if token.text in LOCATION_KEYWORDS:
                location = token.text
            if token.text in QUANTITY_KEYWORDS:
                # You could later map this to a specific number if desired.
                quantity = token.text

        # Add a command if a shape was detected.
        if shape:
            commandItem = {
                "shape": shape,
                "quantity": quantity,
                "size": size,
                "location": location
            }
            commandQueue.append(commandItem)

        # Optionally, handle emotion as its own command.
        if emotion:
            commandQueue.append({"emotion": emotion})

    # Fallback: if no noun chunk yielded a command, scan tokens individually.
    if not commandQueue:
        for token in doc:
            if token.lemma_ in SHAPE_KEYWORDS:
                command_item = {
                    "shape": token.lemma_,
                    "quantity": 1,
                    "size": None,
                    "location": None
                }
                commandQueue.append(command_item)

    return commandQueue


if __name__ == "__main__":    
    # Example usage of the updated function
    test_str = "Draw a big square and on the left draw 5 small circles on the left, then add a grid pattern and random dots."
    queue = recogniseObjects(test_str)
    print("Command Queue:", list(queue))