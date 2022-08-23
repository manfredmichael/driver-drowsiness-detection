import requests
import json
from PIL import Image
from io import BytesIO, StringIO
from base64 import decodebytes, encodebytes
import numpy as np
import os
import dotenv

dotenv.load_dotenv()

inference_url = os.getenv('INFERENCE_URL') 

def encode_image(pil_img):
    byte_arr = BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

def decode_image(image_bytes):
    image_bytes = image_bytes.encode('ascii')
    image_bytes = decodebytes(image_bytes)
    image_bytes = BytesIO(image_bytes)
    image_bytes = Image.open(image_bytes)
    return np.array(image_bytes)

def inference(webcam, background, virtual_background):
    
    response = requests.post(
            f"{inference_url}/predict",
        files = {'webcam': encode_image(webcam),
                 'background': encode_image(background),
                 'virtual_background': encode_image(virtual_background),
                },
    )
    response = response.json()
    result = decode_image(response['result'])

    steps = {}
    for i, step in enumerate(response['steps']):
        step_name, image_bytes = step
        step_name = f"{i+1}. {step_name}"
        steps[step_name] = decode_image(image_bytes)

    return steps, result
