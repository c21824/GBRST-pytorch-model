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

from dataset import CustomDataset
from model import CombinedModel

#load data
device = 'cuda'
df = pd.read_csv('../data/GBRST/Train.csv')


train_augs = A.Compose([
    A.Resize(112, 112),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['ClassId']))

val_augs = A.Compose([
    A.Resize(112, 112),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['ClassId']))

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train = CustomDataset(train_df, train_augs)
val = CustomDataset(val_df, val_augs)

print(len(train), len(val))

train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=32)

# example
# img, class_id, bbox = val[1]
# xmin, ymin, xmax, ymax = bbox
# pt1 = (int(xmin), int(ymin))
# pt2 = (int(xmax), int(ymax))
# if isinstance(img, torch.Tensor):
#     img = img.permute(1, 2, 0).numpy()
# if img.dtype != np.uint8:
#     img = (img * 255).astype(np.uint8)
# bnd_img = cv2.rectangle(img.copy(), pt1, pt2, (255, 0, 0), 2)
# plt.imshow(bnd_img)
# plt.title(f'Class: {class_id}')
# plt.show()


model = CombinedModel().to(device)

#loss  + optimizer
classification_loss = nn.CrossEntropyLoss()
bbox_loss = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

#train + validation
# init loss to compare and save best model later
best_val_loss = float('inf')
best_model_weights = None

epochs = 10
for epoch in range(1, epochs + 1):
    # training
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    train_loader_with_progress = tqdm(train_loader, desc=f'[TRAIN] Epoch {epoch}/{epochs}')
    for images, labels, bboxes in train_loader_with_progress:
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        classification_output, bbox_output = model(images)

        classification_loss_value = classification_loss(classification_output, labels)
        bbox_loss_value = bbox_loss(bbox_output, bboxes)
        total_loss = classification_loss_value + bbox_loss_value

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
        _, predicted = torch.max(classification_output.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        accuracy_train = correct_train / total_train

        postfix_dict = {'classification loss': classification_loss_value.item(), 'bbox loss': bbox_loss_value.item(),
                        'classification acc': accuracy_train}
        train_loader_with_progress.set_postfix(postfix_dict)

    #validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    val_loader_with_progress = tqdm(val_loader, desc=f'[VALID] Epoch {epoch}/{epochs}')

    with torch.no_grad():
        for images, labels, bboxes in val_loader_with_progress:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            classification_output, bbox_output = model(images)

            classification_loss_value = classification_loss(classification_output, labels)
            bbox_loss_value = bbox_loss(bbox_output, bboxes)

            total_loss = classification_loss_value + bbox_loss_value

            val_loss += total_loss.item()
            _, predicted = torch.max(classification_output.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            accuracy_val = correct_val / total_val

            postfix_dict2 = {'classification loss': classification_loss_value.item(),
                             'bbox loss': bbox_loss_value.item(), 'classification acc': accuracy_val}
            val_loader_with_progress.set_postfix(postfix_dict2)

        val_loss /= len(val_loader)
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()

    torch.save(best_model_weights, 'best_model_weights.pt')
