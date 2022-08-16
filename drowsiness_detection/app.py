from io import BytesIO
from PIL import Image
from flask import Flask, request 
from drowsiness_detector import detect_drowsiness
import json

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        file = request.files['file']
        # annotations = json.load(request.files['data'])['annotations']
        image = Image.open(BytesIO(file.read()))
        result = detect_drowsiness(image)
        
        return {'drowsy': result['drowsy']} 

# @app.route("/heatmap", methods=['POST'])
# def evaluate_heatmap():
#     if request.method == 'POST':
#         file = request.files['file']
#         annotations = json.load(request.files['data'])['annotations']
#         image = Image.open(BytesIO(file.read()))
#         count, heatmap = predict(image, annotations, return_density_map=True)
#         return {'count': count, 'heatmap': heatmap} 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)



