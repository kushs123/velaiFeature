from flask import Flask, render_template, request;
import cv2;

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
    img = cv2.imread("./photos/"+image.filename)
    # img = img.resize(224,224,1)

    model = tf.keras.Model("./model/saved_model.pb")
    print(model)
    predictions = model.predict(img)
    print(predictions)
    return "hey"


