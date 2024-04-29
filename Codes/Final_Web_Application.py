# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:01:23 2023

@author: rosej
"""

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model
from torchvision import models, transforms
import torch
import torch.nn as nn


app = Flask(__name__)


# Set the template folder path
app = Flask(__name__, template_folder=r"C:/Users/rosej/template")

 

# Load the pre-trained ResNet model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('best_model_first.pth', map_location=device))
model = model.to(device)
model.eval()


def preprocess_image(image):
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image
    
    
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    try:
        image = Image.open(file)
        preprocessed_image = preprocess_image(image)  # Preprocess the image
        # ...
        return "Image uploaded and processed successfully"
    except:
        return "Error processing image"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
     
    file = request.files['file']
    
    image = Image.open(file)
    
    preprocessed_image = preprocess_image(image)
    outputs = model(preprocessed_image)
    _, pred = torch.max(outputs.data, 1)
    pred_label = pred.item()

    if pred_label == 0:
        processed_prediction = "Normal"
    else:
        processed_prediction = "Pneumonia"

    return render_template('prediction_result.html', prediction=processed_prediction)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    
