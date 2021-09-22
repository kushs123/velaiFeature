from flask import Flask, render_template, request;
import cv2;
from predict import predict;
import numpy as np;

import tensorflow as tf;

app = Flask(__name__, template_folder='static')
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def get():
    return render_template('./index.html')

@app.route('/processing', methods=['POST'])
def processing():
    image = request.files["filename"]
    image.save("./photos/"+image.filename)
    img = "./photos/"+image.filename
    # img = img.resize(224,224,1)

    model_path = "./model"
    # print(model)
    predictions = predict(model_path, img)
    print(predictions)
    return "hey"


