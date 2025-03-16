#!/usr/bin/env python3


import random
import re
import json

def strip_markers_and_get_annotations(raw_text):
    """
    Processes a raw text string that contains custom markers and returns the final text
    with the markers removed, along with a list of annotation dictionaries.
    Each annotation is a dictionary with keys: 'start', 'end', and 'tag_name'.
    """
    pattern = re.compile(r"<<<(QUANTITY|SIZE|SHAPE|LOCATION)>>>(.*?)<<<END_\1>>>")
    final_text = ""
    annotations = []
    last_index = 0

    for match in pattern.finditer(raw_text):
        start, end = match.span()
        label = match.group(1)
        entity_text = match.group(2)
        # Append text before the marker
        final_text += raw_text[last_index:start]
        entity_start = len(final_text)
        final_text += entity_text
        entity_end = len(final_text)
        annotations.append({
            "start": entity_start,
            "end": entity_end,
            "tag_name": label.upper()
        })
        last_index = end
    # Append remaining text
    final_text += raw_text[last_index:]
    return final_text, annotations

def generate_synthetic_training_data(num_examples, file_path):
    """
    Generates synthetic training data for our custom NER model.
    Each example is stored as:
      {
         "content": "final sentence without markers",
         "annotations": [
              {"start": int, "end": int, "tag_name": "TAG"}
         ]
      }
    
    The sentence templates below include improved and varied structures, e.g.:
      - "draw me a <<<SIZE>>>{size}<<<END_SIZE>>> <<<SHAPE>>>{shape}<<<END_SHAPE>>>."
      - "draw me a <<<SIZE>>>{size1}<<<END_SIZE>>> <<<SHAPE>>>{shape1}<<<END_SHAPE>>> and a <<<SIZE>>>{size2}<<<END_SIZE>>> <<<SHAPE>>>{shape2}<<<END_SHAPE>>>."
      - "draw <<<QUANTITY>>>{quantity}<<<END_QUANTITY>>> <<<SIZE>>>{size}<<<END_SIZE>>> <<<SHAPE>>>{shape}<<<END_SHAPE>>> on the <<<LOCATION>>>{location}<<<END_LOCATION>>>."
      - "draw <<<QUANTITY>>>{quantity1}<<<END_QUANTITY>>> <<<SIZE>>>{size1}<<<END_SIZE>>> <<<SHAPE>>>{shape1}<<<END_SHAPE>>> on the <<<LOCATION>>>{location1}<<<END_LOCATION>>> and <<<QUANTITY>>>{quantity2}<<<END_QUANTITY>>> <<<SIZE>>>{size2}<<<END_SIZE>>> <<<SHAPE>>>{shape2}<<<END_SHAPE>>> on the <<<LOCATION>>>{location2}<<<END_LOCATION>>>."
    
    The generated examples are saved in a JSON file with a top-level "examples" key.
    """
    # Domain keywords
    shapes = ["circle", "triangle", "square"]
    sizes = ["small", "big", "large", "tiny", "huge"]
    locations = ["left", "right", "center", "top", "bottom"]
    quantities = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

    # Define a list of improved sentence templates.
    templates = [
        # Single command without quantity or location
        "draw me a <<<SIZE>>>{size}<<<END_SIZE>>> <<<SHAPE>>>{shape}<<<END_SHAPE>>>.",
        "please draw a <<<SIZE>>>{size}<<<END_SIZE>>> <<<SHAPE>>>{shape}<<<END_SHAPE>>>.",
        "sketch a <<<SIZE>>>{size}<<<END_SIZE>>> <<<SHAPE>>>{shape}<<<END_SHAPE>>>.",
        
        # Two commands without quantity or location
        "draw me a <<<SIZE>>>{size1}<<<END_SIZE>>> <<<SHAPE>>>{shape1}<<<END_SHAPE>>> and a <<<SIZE>>>{size2}<<<END_SIZE>>> <<<SHAPE>>>{shape2}<<<END_SHAPE>>>.",
        "could you draw a <<<SIZE>>>{size1}<<<END_SIZE>>> <<<SHAPE>>>{shape1}<<<END_SHAPE>>> followed by a <<<SIZE>>>{size2}<<<END_SIZE>>> <<<SHAPE>>>{shape2}<<<END_SHAPE>>>?",
        "I want a <<<SIZE>>>{size1}<<<END_SIZE>>> <<<SHAPE>>>{shape1}<<<END_SHAPE>>> and a <<<SIZE>>>{size2}<<<END_SIZE>>> <<<SHAPE>>>{shape2}<<<END_SHAPE>>>.",
        
        # Single command with quantity and location
        "draw <<<QUANTITY>>>{quantity}<<<END_QUANTITY>>> <<<SIZE>>>{size}<<<END_SIZE>>> <<<SHAPE>>>{shape}<<<END_SHAPE>>> on the <<<LOCATION>>>{location}<<<END_LOCATION>>>.",
        "please draw <<<QUANTITY>>>{quantity}<<<END_QUANTITY>>> <<<SHAPE>>>{shape}<<<END_SHAPE>>> at the <<<LOCATION>>>{location}<<<END_LOCATION>>>.",
        
        # Two commands with quantity and location
        "draw <<<QUANTITY>>>{quantity1}<<<END_QUANTITY>>> <<<SIZE>>>{size1}<<<END_SIZE>>> <<<SHAPE>>>{shape1}<<<END_SHAPE>>> on the <<<LOCATION>>>{location1}<<<END_LOCATION>>> and <<<QUANTITY>>>{quantity2}<<<END_QUANTITY>>> <<<SIZE>>>{size2}<<<END_SIZE>>> <<<SHAPE>>>{shape2}<<<END_SHAPE>>> on the <<<LOCATION>>>{location2}<<<END_LOCATION>>>.",
        
        # Mixed structure: one command with full details and one simple command
        "draw a <<<SIZE>>>{size1}<<<END_SIZE>>> <<<SHAPE>>>{shape1}<<<END_SHAPE>>> and then draw <<<QUANTITY>>>{quantity}<<<END_QUANTITY>>> <<<SHAPE>>>{shape2}<<<END_SHAPE>>>.",
    ]
    
    examples = []
    for _ in range(num_examples):
        template = random.choice(templates)
        values = {}
        
        # Check which placeholders exist in the template and fill them accordingly.
        if "<<<QUANTITY>>>{quantity}" in template:
            values["quantity"] = random.choice(quantities)
        if "<<<SIZE>>>{size}" in template:
            values["size"] = random.choice(sizes)
        if "<<<SHAPE>>>{shape}" in template:
            values["shape"] = random.choice(shapes)
        if "<<<LOCATION>>>{location}" in template:
            values["location"] = random.choice(locations)
        
        if "<<<SIZE>>>{size1}" in template:
            values["size1"] = random.choice(sizes)
        if "<<<SHAPE>>>{shape1}" in template:
            values["shape1"] = random.choice(shapes)
        if "<<<QUANTITY>>>{quantity1}" in template:
            values["quantity1"] = random.choice(quantities)
        if "<<<LOCATION>>>{location1}" in template:
            values["location1"] = random.choice(locations)
            
        if "<<<SIZE>>>{size2}" in template:
            values["size2"] = random.choice(sizes)
        if "<<<SHAPE>>>{shape2}" in template:
            values["shape2"] = random.choice(shapes)
        if "<<<QUANTITY>>>{quantity2}" in template:
            values["quantity2"] = random.choice(quantities)
        if "<<<LOCATION>>>{location2}" in template:
            values["location2"] = random.choice(locations)
        
        # Create the raw text with markers
        raw_text = template.format(**values)
        # Process to remove markers and get annotation offsets
        content, annotations = strip_markers_and_get_annotations(raw_text)
        example = {"content": content, "annotations": annotations}
        examples.append(example)
    
    data = {"examples": examples}
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Generated {num_examples} training examples in {file_path}")

# Example usage:
if __name__ == "__main__":
    # Generate 500 examples as a starting point.
    generate_synthetic_training_data(500, "synthetic_training_data.json")