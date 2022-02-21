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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 200
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


def colorize(outputs, gt, path, save_path):
    # outputs = outputs.cuda().data.cpu().numpy()
    gt = gt.cuda().data.cpu().numpy()
    img = cv2.imread(path)

    points_list2 = np.array(gt, dtype=np.float)
    point_size = 1
    point_color = (0, 0, 255)
    point_color2 = (0, 255, 0)
    thickness = 4  # 可以为 0、4、8

    # print(outputs[0])
    for i in range(2):
        cv2.circle(img, (int(points_list2[2*i]), int(points_list2[2*i+1])),
                   point_size, point_color2, thickness)
        cv2.circle(img, (outputs[i][0], outputs[i][1]),
                   point_size, point_color, thickness)

    # print(save_path)
    cv2.imwrite(save_path, img)


def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,15,96,96)
    :return:numpy array (N,15,2)
    """
    N, C, H, W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy, xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x, y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points


def calculate_mask(heatmaps_target):
    """

    :param heatmaps_target: Variable (N,15,96,96)
    :return: Variable (N,15,96,96)
    """
    N, C, _, _ = heatmaps_targets.size()
    N_idx = []
    C_idx = []
    for n in range(N):
        for c in range(C):
            max_v = heatmaps_targets[n, c, :, :].max().data
            if max_v != 0.0:
                N_idx.append(n)
                C_idx.append(c)
    mask = torch.zeros(heatmaps_targets.size())
    mask[N_idx, C_idx, :, :] = 1.
    mask = mask.float().cuda()
    return mask, [N_idx, C_idx]


for epoch in range(num_epochs):
    tmp = 0
    for i, (data, gt, mask, item, imgPath, heatmaps_targets) in enumerate(train_loader):
        data = data.to(device)
        gt = gt.to(device)
        mask = mask.to(device)
        gt = gt.view(-1, 4)
        heatmaps_targets = heatmaps_targets.to(device)
        mask, indices_valid = calculate_mask(heatmaps_targets)

        # print(heatmaps_targets.shape)

        # Forward pass
        outputs = model(data)
        outputs = outputs * mask
        heatmaps_targets = heatmaps_targets * mask

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

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), '{}heatmap4.ckpt'.format(epoch + 1))
