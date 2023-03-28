import cv2
import torch
from torchvision import transforms
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Transformation for images
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# Load your trained model
model = models.resnet18()
model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,7)
model.load_state_dict(torch.load('./68_acc.pth', map_location=torch.device('cpu')))
model.eval()

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad','Surprise', 'Neutral', ]

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using Haar cascades
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    for (x,y,w,h) in faces:
        # Extract the face region from the grayscale frame and convert it to a PIL Image object
        face_region_gray = gray_frame[y:y+h,x:x+w]
        face_region_pil_image = Image.fromarray(face_region_gray)

        # Apply image transformations to the face region and convert it to a tensor with an additional batch dimension 
        face_region_tensor = image_transform(face_region_pil_image).unsqueeze(0)

        # Make a prediction using your model and get the predicted label 
        with torch.no_grad():
            outputs = model(face_region_tensor)
            _, predicted_idx = torch.max(outputs.data, 1)
            predicted_label_idx=predicted_idx.item()
            predicted_label=class_labels[predicted_label_idx]

        # Draw a rectangle around the face region and put text with the predicted label above it 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        font=cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(frame,predicted_label,(x,y-10),font,1.5,(255,255,255),1,cv2.LINE_AA)

    # Display the frame with face regions and predictions
    cv2.imshow('frame',frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()