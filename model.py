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

import timm
import torch
import torch.nn as nn
import torch.optim as optim

class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.layer6 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 1000),
            nn.ReLU(inplace=True),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),
        )

        self.layer8 = nn.Sequential(
            nn.Linear(256, 43)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.shape[0], -1)

        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return x


# define model for classification + bbox
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()

        self.bbox_regression_branch = timm.create_model('efficientnet_b0', pretrained=True, num_classes=4)

        self.classification_branch = ClassificationNet()

    def forward(self, images):
        classification_output = self.classification_branch(images)

        bbox_output = self.bbox_regression_branch(images)

        return classification_output, bbox_output