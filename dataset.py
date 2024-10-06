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

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # get img
        img_path = '../data/GBRST/' + self.dataframe.iloc[idx]['Path']
        img = Image.open(img_path).convert('RGB')

        # get bbox
        row = self.dataframe.iloc[idx]
        xmin = row['Roi.X1']
        ymin = row['Roi.Y1']
        xmax = row['Roi.X2']
        ymax = row['Roi.Y2']
        bbox = [[xmin, ymin, xmax, ymax]]

        # get class
        class_id = self.dataframe.iloc[idx]['ClassId']

        if self.transform:
            augmented = self.transform(image=np.array(img), bboxes=bbox, ClassId=[class_id])
            img = augmented['image']
            bbox = augmented['bboxes'][0]
            class_id = augmented['ClassId'][0]

        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0  # (h, w, c) -> (c, h, w)
        bbox = torch.Tensor(bbox)

        return img, class_id, bbox