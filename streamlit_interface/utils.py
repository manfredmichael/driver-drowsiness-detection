import requests
import json
from PIL import Image
from io import BytesIO, StringIO
from base64 import decodebytes, encodebytes
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
import os

inference_url = '127.0.0.1'

def encode_image(pil_img):
    byte_arr = BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img


def inference(webcam, background, virtual_background):
    
    result = requests.post(
            f"http://{inference_url}:5000/predict",
        files = {'webcam': encode_image(webcam),
                 'background': encode_image(background),
                 'virtual_background': encode_image(virtual_background),
                },
    )

    result = (Image.open(BytesIO(result.content)))
    return result