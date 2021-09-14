import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from tqdm.notebook import tqdm
import os
import argparse

def predict(args):
    model = load_model(args.model_path)
    im = cv2.imread(args.image_path, 0)
    im = np.array(im)[None,:]
    prob = model.predict(im)
    return prob[0,0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', action='store', help='path of folder containing the model')
    parser.add_argument('--image_path', action='store', help='path of image to process')

    args=parser.parse_args()
    prob = predict(args)
    print("Probability of feature = ", prob)