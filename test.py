import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

import os
import cv2
from PIL import Image

from tqdm import tqdm
import albumentations as A
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from dataset import CustomDataset
from model import CombinedModel

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model
model = CombinedModel().to(device)
model.load_state_dict(torch.load('best_model_weights.pt', map_location=device))
summary(model, input_size=(3, 112, 112))
# print(model.eval())

# Function to display image
def display_image(image_tensor, classification_output, bbox_output, actual_class, actual_bbox):
    image = np.array(image_tensor.cpu())
    image = np.transpose(image, (1, 2, 0))  # (c, h, w) -> (h, w, c)

    classification_output = classification_output.cpu().squeeze().numpy()
    bbox_output = bbox_output.cpu().squeeze().numpy()
    actual_bbox = actual_bbox.cpu().squeeze().numpy()

    plt.imshow(image)
    plt.axis('off')

    predicted_bbox = plt.Rectangle((bbox_output[0], bbox_output[1]),
                                   bbox_output[2] - bbox_output[0], bbox_output[3] - bbox_output[1],
                                   linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(predicted_bbox)

    actual_bbox_rect = plt.Rectangle((actual_bbox[0], actual_bbox[1]),
                                     actual_bbox[2] - actual_bbox[0], actual_bbox[3] - actual_bbox[1],
                                     linewidth=2, edgecolor='g', facecolor='none')
    plt.gca().add_patch(actual_bbox_rect)

    plt.title(f'Predicted: {np.argmax(classification_output)}', color='red', loc='left')
    plt.title(f' Actual: {actual_class}', color='green', loc='right')

    plt.show()


# Classes dictionary
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing',
    10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited', 17: 'No entry',
    18: 'General caution', 19: 'Dangerous curve left',
    20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road',
    24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing',
    30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

# Load test data
test_df = pd.read_csv('../data/GBRST/Test.csv')

test_augs = A.Compose([
    A.Resize(112, 112),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['ClassId']))

test = CustomDataset(test_df, test_augs)
test_loader = torch.utils.data.DataLoader(test, batch_size=32)

# print(len(test))
num_samples = len(test_loader.dataset)
random_indices = random.sample(range(num_samples), 25)  # show X amount of random pics

actual_values = []
predict_values = []
for idx, (images, labels, bboxes) in enumerate(test_loader):
    for i in range(len(images)):
        if idx * 32 + i in random_indices:  # Adjust to ensure each image is indexed properly
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            with torch.no_grad():
                classification_output, bbox_output = model(images)

            image = images[i]
            actual_class = labels[i]
            actual_bbox = bboxes[i]

            predict_values.append(classification_output[i].argmax(dim=0).cpu().numpy())
            actual_values.append(actual_class.cpu().numpy())

            # display_image(image, classification_output[i], bbox_output[i], actual_class, actual_bbox)

print(classification_report(actual_values, predict_values))
