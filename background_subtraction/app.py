from io import BytesIO
from xmlrpc.server import resolve_dotted_attribute
from PIL import Image
from flask import Flask, request, send_file
from background_subtraction import apply_virtual_background
from base64 import decodebytes, encodebytes
import numpy as np

app = Flask(__name__)

def encode_image(pil_img):
    byte_arr = BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route("/predict", methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        # file = request.files['file']
        webcam = request.files['webcam'].read()
        background = request.files['background'].read()
        virtual_background = request.files['virtual_background'].read()
        # webcam = data['webcam']
        # background = data['background']
        # virtual_background = data['virtual_background']

        webcam = np.array(Image.open(BytesIO(decodebytes(webcam))))
        background = np.array(Image.open(BytesIO(decodebytes(background))))
        virtual_background = np.array(Image.open(BytesIO(decodebytes(virtual_background))))

        result = apply_virtual_background(webcam, background, virtual_background)
        result = Image.fromarray(np.uint8(result))
        # annotations = json.load(request.files['data'])['annotations']
        # image = Image.open(BytesIO(file.read()))
        # result = detect_drowsiness(image)
        return serve_pil_image(result)

@app.route('/get_images',methods=['GET'])
def get_images():

    ##reuslt  contains list of path images
    encoded_imges = []
    for image_path in result:
        encoded_imges.append(get_response_image(image_path))
    return jsonify({'result': encoded_imges})

# @app.route("/heatmap", methods=['POST'])
# def evaluate_heatmap():
#     if request.method == 'POST':
#         file = request.files['file']
#         annotations = json.load(request.files['data'])['annotations']
#         image = Image.open(BytesIO(file.read()))
#         count, heatmap = predict(image, annotations, return_density_map=True)
#         return {'count': count, 'heatmap': heatmap} 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)



