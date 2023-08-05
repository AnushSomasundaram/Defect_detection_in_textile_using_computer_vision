from flask import Flask, render_template, request, send_from_directory, url_for
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time


import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torchvision.models import vgg16


app = Flask(__name__)

def get_static_files():
    models_dir = os.path.join(app.root_path, 'static', 'models')
    filenames = []
    for filename in os.listdir(models_dir):
        filenames.append(filename.split('.')[0])
    return filenames

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', models=get_static_files())

@app.route('/models', methods=['POST'])
def models():
    image = request.files['image']
    name = request.form.get('name')
    image.save(os.path.join(app.root_path, 'static', 'models', name + '.jpg'))
    return render_template('index.html', models = get_static_files())


@app.route('/processed_image')
def processed_image():
    return send_from_directory('static', 'processed_image.jpg')

@app.route('/process', methods=['POST'])
def process():
    image = request.files['image']
    options = request.form.get('options')
    input_image_name = 'input_image.jpg'
    image.save(os.path.join(app.root_path, 'static', input_image_name))

    model_path = url_for('static', filename = 'models/'+ options +'.jpg')
    image_path = url_for('static', filename = input_image_name)
    result = compare_contours(model_path, image_path)
    time.sleep(5)
    processed_image_name = 'processed_image.jpg'
    cv2.imwrite(os.path.join(app.root_path, 'static', processed_image_name), result)
    time.sleep(2)
    
    processed_image_path = os.path.join(app.root_path, 'static', processed_image_name)
    model_path="/Users/software/Desktop/ai-in-textile/prototype/shape1_vgg16_model.pth"
    predicted_class = predict_vgg16_single(processed_image_path,model_path)
    
    if predicted_class:
                proceed_discard="✅ Correct!"
    else:
                proceed_discard("❌ Incorrect!")
    
    
    
    return render_template('index.html',predicted_class=predicted_class)




def compare_contours(model_path, image_path):
    # Read the model_path image
    image1 = cv2.imread(app.root_path + model_path)
    image2 = cv2.imread(app.root_path + image_path)

    # Resize the images to the same size
    resized_image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    resized_image1 = image1

    # Convert images to grayscale
    gray1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Perform edge detection using Canny
    edges1 = cv2.Canny(blurred1, 50, 150)
    edges2 = cv2.Canny(blurred2, 50, 150)

    # Find contours in the first edge image
    contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours in the second edge image
    contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank canvas with the size of the larger image
    canvas_height = max(resized_image1.shape[0], resized_image2.shape[0])
    canvas_width = max(resized_image1.shape[1], resized_image2.shape[1])
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Draw the contours from the first image onto the canvas using green color
    cv2.drawContours(canvas, contours1, -1, (0, 255, 0), 2)

    # Draw the contours from the second image onto the canvas using red color
    cv2.drawContours(canvas, contours2, -1, (0, 0, 255), 2)
    return canvas

def predict_vgg16_single(image_path,model_path):
    # load custom vgg model trained on azhars pc
    
    model = models.vgg16(pretrained=False)
    
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    
    #preprocessing the image
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_image = Image.open(image_path).convert('RGB')
    
    input_data = preprocess(input_image)
    input_data = input_data.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_data)
        
    #Get the predicted class index
    
    _ , predicted_idx = torch.max(outputs,1)
    predicted_index = predicted_idx.item()
    
    
    if predicted_index==0:
        return True
    else:
        return False


    




if __name__ == '__main__':
    app.run()
