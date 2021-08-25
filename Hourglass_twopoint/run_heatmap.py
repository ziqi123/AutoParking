import torch
import torchvision.transforms as transforms
import cv2
import copy
from PIL import Image
import numpy as np
from dataloader.heatmap_Dataloader import heatmap_Dataloader
import matplotlib.pyplot as plt
# from basic_cnn import BasicCNN
import os
from hourglass import KFSGNet
import torch.nn as nn
import torchvision.transforms as transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 100
learning_rate = 0.0001

transform = transforms.Compose([
    transforms.ToTensor()])


params = dict()
params['data_normalize_factor'] = 256
params['dataset_dir'] = "./"
params['rgb2gray'] = False
params['dataset'] = "CNNDataset"
params['train_batch_sz'] = 16
params['val_batch_sz'] = 1
params['sigma'] = 3

dataloaders, dataset_sizes = heatmap_Dataloader(params)

train_loader = dataloaders['train']
test_loader = dataloaders['val']

# Define your model
# model = BasicCNN()

model = KFSGNet()

# move model to the right device
model.to(device)
model.train()

# Loss and optimizer
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(optimizer.state_dict()['param_groups'][0]['lr'])

# For updating learning rate

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate

print("start")


loss_array = []
for epoch in range(num_epochs):
    tmp = 0
    for i, (data, gt, mask, item, imgPath, heatmaps_targets) in enumerate(train_loader):
        data = data.to(device)
        gt = gt.to(device)
        mask = mask.to(device)
        gt = gt.view(-1, 4)
        heatmaps_targets = heatmaps_targets.to(device)

        # print(heatmaps_targets.shape)

        # Forward pass
        outputs = model(data)

        loss = loss_fn(outputs, heatmaps_targets)

        tmp += loss.item()
        # exit()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("all_peak_points", all_peak_points)
        if i % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}, average_loss: {:.4f}, learning_rate: {}".format(
                epoch + 1, num_epochs, i + 1, total_step, loss.item(), tmp / (i+1), optimizer.state_dict()['param_groups'][0]['lr']))

    loss_array.append(tmp)
    if (epoch + 1) % 3 == 0:
        torch.save(model.state_dict(), '{}heatmap.ckpt'.format(epoch + 1))
