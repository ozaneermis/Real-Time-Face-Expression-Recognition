import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image

model = models.resnet18()
model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,7)
model.load_state_dict(torch.load('./68_acc.pth', map_location=torch.device('cpu')))

model.eval()

image = Image.open('./archive/My data/happy.png')
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])
image_tensor = image_transform(image)

image_tensor = image_tensor.unsqueeze(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_tensor = image_tensor.to(device)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Make a prediction
model.eval()
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_label = class_labels[predicted.item()]
    print(f'Predicted Label: {predicted_label}')