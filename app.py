import torch
import torchvision
import numpy as np
import PIL
import requests
import wget
from io import BytesIO

from flask import Flask, request, jsonify
from disease_classifier import PlantDiseaseClassifier

app = Flask(__name__)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = PlantDiseaseClassifier(pretrained=False)
wget.download('https://github.com/TAASPlapp/plapp-diagnosis-service/releases/download/v2.0/resnet18-plantvillage-42.pt')
checkpoint = torch.load('resnet18-plantvillage-42.pt', map_location={'cuda:0': 'cpu'})
model.load_state_dict(checkpoint['model'])

classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
classes = list(map(lambda c: c.replace('_', ' ').replace('   ', ' - '), classes))

@app.route('/diagnose')
def diagnose():
    url = request.args['plantImageURL']
    response = requests.get(url)
    print(response)
    image = PIL.Image.open(BytesIO(response.content))
    x = transforms(image).unsqueeze(0)
    with torch.no_grad():
        _, pred = torch.max(model(x), 1)
    pred = pred.squeeze(0).item()

    response = {
        'plantId': 0,
        'ill': False if 'healthy' in classes[pred] else True,
        'disease': classes[pred]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
