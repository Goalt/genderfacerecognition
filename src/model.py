import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import torch


class FaceRecognitionPipeline():
    def __init__(self, model, haarConfFile, inputSize):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.inputSize = inputSize
        self.faceCascade = cv2.CascadeClassifier(haarConfFile)

        self.model.eval()
        self.model = self.model.to(self.device)

    def faceDetection(self, image):
        # return faceBorder
        facesBorder = self.faceCascade.detectMultiScale(image, 1.3, 5)
        faces = []
        for (x,y,w,h) in facesBorder:
            faces.append(image[y:y + h, x:x + w])
        
        # plt.imshow(faces[0], cmap='gray', vmin=0, vmax=255)
        return faces


    def predict(self, image):
        # returns input to CNN, class and confidence
        
        faces = self.faceDetection(image)
        
        if (len(faces) == 0):
            return (None, None, None)

        testTransforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.inputSize),
            transforms.Grayscale(num_output_channels=1),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])

        input = []
        for i in faces:
            input.append(testTransforms(i))
        input = torch.stack(input) 

        input = input.to(self.device)
        output = self.model(input.to(torch.float32))
        output = torch.sigmoid(output)
        result = torch.round(output)
        
        input = input.cpu()
        return (input, result, output)


class GenderRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def predict(self, x):
        model.eval()
        output = self.forward(x)
        return F.sigmoid(output)


class GenderRecognitionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1620, 550)
        self.fc2 = nn.Linear(550, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1620)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        model.eval()
        output = self.forward(x)
        return torch.sigmoid(output)