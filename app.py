import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from flask import Flask, render_template, request
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from Model import ResNet9
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# Load CSV data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
tree_doc = pd.read_csv('tree_doc.csv', encoding='cp1252')

# Load trained model
model = ResNet9(3, 38)
model.load_state_dict(torch.load("green-check-model.pth", map_location=torch.device('cpu')))
model.eval()

# Check for GPU availability
use_gpu = torch.cuda.is_available()

# Initialize Flask app
app = Flask(__name__)

# Prediction function
def prediction(image_path):
    image = Image.open(image_path)
    input_data = TF.to_tensor(image).unsqueeze(0)
    output = model(input_data)
    index = np.argmax(output.detach().numpy())
    return index

# Saliency map function
def Saliency_map(img_tensor, model, ground_truth, use_gpu=False):
    img_tensor = Image.open(img_tensor)
    preprocess = transforms.Compose([transforms.ToTensor()])
    img_tensor = preprocess(img_tensor).unsqueeze(0)

    if use_gpu:
        img_tensor = img_tensor.cuda()

    input = Variable(img_tensor, requires_grad=True)
    if input.grad is not None:
        input.grad.data.zero_()

    model.zero_grad()
    output = model(input)
    ind = torch.LongTensor(1)
    ind[0] = ground_truth
    ind = Variable(ind)
    energy = output[0, ground_truth]
    energy.backward()
    grad = input.grad
    if use_gpu:
        return np.abs(grad.data.cpu().numpy()[0]).max(axis=0)
    return np.abs(grad.data.numpy()[0]).max(axis=0)

# Function to display saliency map
def show_sal(image, file_name):
    plt.imshow(image, cmap='afmhot', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    map_path = os.path.join('static/uploads2', f"{file_name}_map.png")
    plt.savefig(map_path, transparent=True, bbox_inches='tight', pad_inches=0)
    return map_path

# Flask routes
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/aiDoc')
def ai_engine_page():
    return render_template('aiDoc.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        pred = prediction(file_path)

        # Generate saliency map
        sal_map = Saliency_map(file_path, model, pred, use_gpu)
        file_name = os.path.splitext(filename)[0]

        # Save saliency map
        map_path = show_sal(sal_map, f"{file_name}_map")
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        human_tree_doc_img = tree_doc['tree doc'][pred]
        contact_doc_link = tree_doc['contact doc'][pred]

        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=file_path, sal_url=map_path, pred=pred,
                               t_doc=human_tree_doc_img, contact_doc =contact_doc_link)
        return render_template(disease=list(disease_info['disease_name']), contact=list(tree_doc['contact doc']))



if __name__ == '__main__':
    app.run(debug=True)
