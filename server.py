from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import generateResponse
import pickle
import cgi
from flask import Flask, request, send_from_directory, render_template
import base64
from io import BytesIO
from PIL import Image
import os
import numpy as np
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\Users\patel\OneDrive - iiit-b\Desktop\coursework\sem 5\Software Engineering Lab\project\deliverable 3\static\upload'    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train():
    status = train_request()
    return status

@app.route('/searchCriminal.html')
def searchCriminal():
    return render_template('searchCriminal.html')

@app.route('/index.html', methods=['POST'])
def home():
    return render_template('index.html')

@app.route('/liveCam.html')
def liveCam():
    return render_template('liveCam.html')


@app.route('/liveInference', methods=['POST', 'GET'])
def live_inference():
    if request.method == "POST":
        
        image = request.get_json()['image']

        # change the image to numpy array
        _, base64_string = image.split(',')
        image_data = base64.b64decode(base64_string)
        image_to_predict = Image.open(BytesIO(image_data))
        image_to_predict = image_to_predict.resize((152, 152))
        image_to_predict = np.array(image_to_predict)
        image_to_predict = image_to_predict.reshape(-1, 152, 152, 1) / 255

        # get the prediction
        prediction, predictionTarget = process_image(image_to_predict)
        return json.dumps({'prediction': prediction, 'image': predictionTarget})

@app.route('/inference', methods=['POST', 'GET'])
def receive_image():
    if request.method == "POST":
        image = request.files['image']

        print("Imgae: " ,image)

        if image.filename == '':
            print("File name is invalid")
            return redirect(request.url)


        filename = image.filename
        basedir = os.path.abspath(os.path.dirname(__file__))
        image.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))

        # change the image to numpy array
        image_to_predict = Image.open(image)
        image_to_predict = image_to_predict.resize((152, 152))
        image_to_predict = np.array(image_to_predict)
        image_to_predict = image_to_predict.reshape(-1, 152, 152, 1) / 255

        # get the prediction
        prediction = process_image(image_to_predict)

# pass the prediction to the frontend
        print("passed ")
        return render_template('searchCriminal.html', prediction=prediction, filename=filename)

    return render_template('searchCriminal.html')


MODEL = None
CRIMINAL_LIST = None


def process_image(data):
    global MODEL, CRIMINAL_LIST
    if MODEL is None:
        MODEL = pickle.load(open("saved_model/model.pickle", 'rb'))
        CRIMINAL_LIST = pickle.load(open("saved_model/labels.pickle", 'rb'))


    result = generateResponse.generate_result(data, model=MODEL)
    return CRIMINAL_LIST[result - 1]

def train_request():
    status, CRIMINAL_LIST, MODEL = generateResponse.train_model()

    pickle.dump(MODEL, open("saved_model/model.pickle", 'wb'))
    pickle.dump(CRIMINAL_LIST, open("saved_model/labels.pickle", 'wb'))

    print("STATUS =", status)
    return status


if __name__ == '__main__':
    app.run(debug=True)
