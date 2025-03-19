#!/usr/bin/env python3

import json
from base64 import b64decode
import numpy as np
import cv2



from openai import OpenAI

client = OpenAI()

def generateImage(prompt):

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        response_format="b64_json",
    )

    for index, image_dict in enumerate(response.data):
        image_data = b64decode(image_dict.b64_json)
        np_array = np.frombuffer(image_data, np.uint8)

        # Decode image array into OpenCV format (BGR)
        cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        cv2.imshow("Generated Image", cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return cv_image
    
    
if __name__ == "__main__":
    prompt = "A cartoon line drawing done using a solid black marker"
    generateImage(prompt)
    