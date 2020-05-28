import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import cv2
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import Image

class_names = ['infected', 'Normal']

# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained=False)
device = torch.device("cpu")

# removing last layer
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-2]  # Remove last layer
# due to binary classification setting output size to 2 with sigmoid logistic function
c_feature = [nn.Linear(4096, 710), nn.ReLU(inplace=True), nn.Linear(710, 2), nn.Sigmoid()]
# adding new Layer
features.extend(c_feature)

# Freeze training for all layers, except last two layers
i = 0;
params = vgg16.features.parameters()
for param in vgg16.features.parameters():
    i += 1
    param.require_grad = False

    if (i >= 24):
        break

vgg16.classifier = nn.Sequential(*features)

# vgg16.to(device)
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(THIS_FOLDER, 'vgg16_FC_Only.pth')
res = torch.load(weights_path, map_location=torch.device('cpu'))
vgg16.load_state_dict(res)
vgg16.eval()

loader = transforms.Compose([transforms.Resize((256, 256)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))])


def image_loader(image_name):
    """load image, returns cuda tensor"""

    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    #     image = image.to(device)
    outputs = vgg16(image)
    _, predicted = torch.max(outputs, 1)
    return predicted


# res = image_loader(THIS_FOLDER + '/Dataset/test/infected/dbfc2da5-fc7e-431d-b05d-9a0b580c357f.png')
# print('Predicted: ', ' '.join('%5s' % class_names[res]))