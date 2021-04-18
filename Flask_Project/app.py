import sys

from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from PIL import Image
import os
import numpy as np

from utils import base64_to_pil

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torchvision.models as models
import torch
import torch.nn as nn
from torchvision import transforms

# PATH
WEIGHT_PATH_SCENE = './model/transformerocr_real_scene_7_4.pth'
WEIGHT_PATH_SCAN = './model/best_weight_scan_text.pth'
VGG_MODEL_PATH = './model/vgg_classcification.pth'

# OCR MODEL
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = WEIGHT_PATH_SCENE
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
# SCENE TEXT DETECTOR
scene_detector = Predictor(config)

# SCAN TEXT DETECTOR
config['weights'] = WEIGHT_PATH_SCAN
scan_detector = Predictor(config)

def ocr_model_predict(img,model):
    return model.predict(img)

# INIT PARAMS OF VGG MODEL
num_classes = 2
input_size = 224
vgg_model = models.vgg11_bn()
num_ftrs = vgg_model.classifier[6].in_features
vgg_model.classifier[6] = nn.Linear(num_ftrs, num_classes)
device = torch.device('cpu')
vgg_model.load_state_dict(torch.load(VGG_MODEL_PATH, map_location=device))
data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
def vgg_model_prediction(vgg_model,img):
    return nn.Softmax(vgg_model(data_transforms(img).unsqueeze(0)))


app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')


@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        img_type = vgg_model_prediction(vgg_model, img)
        print(img_type)
        a = np.argmax(img_type)
        if a == 0:  # scan
            result = ocr_model_predict(img, scan_detector)
        else:
            result = ocr_model_predict(img, scene_detector)

        return jsonify(result=result)
    return None


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()