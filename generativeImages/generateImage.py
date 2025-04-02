#!/usr/bin/env python3

from base64 import b64decode
import numpy as np
import cv2

from alive_progress import alive_bar

from openai import OpenAI

client = OpenAI()

def generateImage(prompt):
    
    # prompt = prompt + ". Ensure the image is a minimalistic, high-contrast black and white line drawing of only the specified subject on a plain white background, with bold, clear outlines and no additional details."

    with alive_bar(0, title="Generating image", spinner="horizontal") as bar:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            response_format="b64_json",
        )
        bar()

    for index, image_dict in enumerate(response.data):
        image_data = b64decode(image_dict.b64_json)
        np_array = np.frombuffer(image_data, np.uint8)

        cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        return cv_image
    
    
if __name__ == "__main__":
    prompt = "A cartoon line drawing done using a solid black marker"
    generateImage(prompt)
    