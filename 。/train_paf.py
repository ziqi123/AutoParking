import numpy as np
import torch
import torchvision.transforms as transforms
from dataloader.dataloader_stackedHourglass_paf import heatmap_Dataloader
import os
from models.posenet_PAF import PoseNet
# from models.posenet2 import PoseNet
import torchvision.transforms as transforms
import torch.nn as nn
# from models.Adaptive_wing_loss import HeatmapLoss
from models.FMSE_Loss import HeatmapLoss
from models.MSE_loss import PAFLoss
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,0'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 300
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor()])


params = dict()
params['data_normalize_factor'] = 256
params['dataset_dir'] = "./"
params['rgb2gray'] = False
params['dataset'] = "heatmap_dataset_all"
params['train_batch_sz'] = 32
params['val_batch_sz'] = 1
params['sigma'] = 3

dataloaders, dataset_sizes = heatmap_Dataloader(params)

train_loader = dataloaders['train']
test_loader = dataloaders['val']

# Define your model

# model = PoseNet(256, 4)
model = PoseNet(256, 4, 8)
# move model to the right device
model = nn.DataParallel(model)
model.to(device)

model.train()

# Loss and optimizer
loss_fn = torch.nn.MSELoss()
criterion = nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.1)

print(optimizer.state_dict()['param_groups'][0]['lr'])

# For updating learning rate

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate

print("start")


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


criterion = nn.MSELoss().to(device)
heatmapLoss = HeatmapLoss()
pafLoss = PAFLoss()
for epoch in range(num_epochs):
    tmp = 0
    scheduler.step()
    for i, (data, gt, mask, item, imgPath, heatmaps_targets, vectormap) in enumerate(train_loader):
        # print(i)
        data = data.to(device)
        gt = gt.to(device)
        mask = mask.to(device)
        gt = gt.view(-1, 8)
        heatmaps_targets = heatmaps_targets.to(device)
        vectormap = vectormap.to(device)

        # print("heatmaps_targets.shape", heatmaps_targets.shape)
        # heatmaps_targets = get_peak_points(heatmaps_targets.cpu().data.numpy())
        # print("heatmaps_targets", heatmaps_targets)

        # paf_afterabs = np.abs(vectormap)
        # paf_aftersum = np.sum(paf_afterabs, 0)
        # heatmaps = np.sum(heatmaps_targets, 0)
        # plt.subplot(4, 3, 1)
        # # plt.imshow(data)
        # # plt.subplot(4, 3, 2)
        # plt.imshow(paf_aftersum)
        # plt.subplot(4, 3, 2)
        # plt.imshow(heatmaps)
        # plt.savefig(
        #     "/media/home_bak/ziqi/park/stackedHourglass_256/paf.png")

        # mask, indices_valid = calculate_mask(heatmaps_targets)
        # print("heatmaps_targets.shape", heatmaps_targets.shape)
        # print("imgPath", imgPath)
        # print("gt", gt)
        # heatmaps_targets = get_peak_points(heatmaps_targets.cpu().data.numpy())
        # print("heatmaps_targets", heatmaps_targets)

        # Forward pass
        outputs = model(data)
        # print("heatmaps_targets.shape", heatmaps_targets.shape)
        # print("vectormap.shape", vectormap.shape)
        # print("outputs['heatmaps']", outputs['heatmaps'])
        # outputs = outputs * mask
        # heatmaps_targets = heatmaps_targets * mask
        # print(outputs.shape)

        # loss = criterion(outputs, heatmaps_targets)

        # print('heatmaps', outputs['heatmaps'].shape)
        combined_loss = []
        for j in range(8):
            combined_loss.append(heatmapLoss(
                outputs['heatmaps'][:, j], heatmaps_targets))
        combined_loss = torch.stack(combined_loss, dim=1)

        loss = 0
        for k in range(len(heatmaps_targets)):
            loss = loss + torch.mean(combined_loss[k])

        # print('paf', outputs['paf'].shape)

        combined_paf_loss = []
        for s in range(8):
            combined_paf_loss.append(pafLoss(
                outputs['paf'][:, s], vectormap))
        combined_paf_loss = torch.stack(combined_paf_loss, dim=1)

        for t in range(len(heatmaps_targets)):
            loss = loss + torch.mean(combined_paf_loss[t])

        # paf_loss = loss_fn(outputs['paf'], vectormap)
        # loss += paf_loss
        tmp += loss.item()
        # exit()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}, average_loss: {:.4f}, learning_rate: {}".format(
                epoch + 1, num_epochs, i + 1, total_step, loss.item(), tmp / (i+1), scheduler.get_lr()[0]))

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(),
                   '{}heatmaps_sigma3_1_nstack8_PAF.ckpt'.format(epoch + 1))
