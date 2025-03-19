#!/usr/bin/env python3

import math
import random
from abc import ABC, abstractmethod
import spacy
from collections import deque

# Load spaCy model (using the small English model; you can change this to a transformer model if needed)
nlp = spacy.load("en_core_web_sm")

# Define keywords for our domain
SHAPE_KEYWORDS = {"circle", "triangle", "square"}
SIZE_KEYWORDS = {"small", "big", "large", "tiny", "huge"}
LOCATION_KEYWORDS = {"left", "right", "center", "top", "bottom"}

# Optional mapping for number words
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def createPatternSpacy(input_str):
    """
    Process the input string using spaCy to extract drawing commands.
    Returns a queue (deque) of command dictionaries, each containing:
      - shape: one of 'circle', 'triangle', 'square'
      - quantity: an integer (default 1 if not specified)
      - size: size descriptor if available (e.g., 'small', 'big')
      - location: location descriptor if available (e.g., 'left', 'right')
    
    The function uses noun chunks to group related tokens. If no commands are found, an empty queue is returned.
    """
    doc = nlp(input_str.lower())
    command_queue = deque()

    # Process noun chunks as candidate commands
    for chunk in doc.noun_chunks:
        shape = None
        quantity = 1  # default
        size = None
        location = None

        for token in chunk:
            # Check if token is a number or a number word
            if token.like_num:
                try:
                    quantity = int(token.text)
                except ValueError:
                    quantity = NUM_WORDS.get(token.text, 1)
            
            # Check for shape keywords (using lemma to account for plural forms)
            if token.lemma_ in SHAPE_KEYWORDS:
                shape = token.lemma_
            
            # Check for size keywords
            if token.text in SIZE_KEYWORDS:
                size = token.text
            
            # Check for location keywords
            if token.text in LOCATION_KEYWORDS:
                location = token.text

        if shape:
            command_item = {
                "shape": shape,
                "quantity": quantity,
                "size": size,
                "location": location
            }
            command_queue.append(command_item)

    # Fallback: if no noun chunks yielded a command, scan tokens
    if not command_queue:
        for token in doc:
            if token.lemma_ in SHAPE_KEYWORDS:
                command_item = {
                    "shape": token.lemma_,
                    "quantity": 1,
                    "size": None,
                    "location": None
                }
                command_queue.append(command_item)

    return command_queue

if __name__ == "__main__":    
    # Example usage of the new function
    test_str = "Draw a big square and on the left draw 5 small circles on the left"
    queue = createPatternSpacy(test_str)
    print("Command Queue:", list(queue))
