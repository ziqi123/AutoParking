import numpy as np
import torch
import torchvision.transforms as transforms
from dataloader.dataloader_hourglass import heatmap_Dataloader
import os
from network import KFSGNet
import torchvision.transforms as transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
params['train_batch_sz'] = 2
params['val_batch_sz'] = 1
params['sigma'] = 3

dataloaders, dataset_sizes = heatmap_Dataloader(params)

train_loader = dataloaders['train']
test_loader = dataloaders['val']

# Define your model

model = KFSGNet()
# model.load_state_dict(torch.load(
#     '/media/home_bak/ziqi/park/hourglass/10heatmap5.ckpt'))
# move model to the right device
model.to(device)

model.train()

# Loss and optimizer
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 多步长学习率衰减
# 不同的区间采用不同的更新频率，或者是有的区间更新学习率，有的区间不更新学习率
# 其中milestones参数为表示学习率更新的起止区间，在区间[0. 200]内学习率不更新，
# 而在[200, 300]、[300, 320].....[340, 400]的右侧值都进行一次更新；
# gamma参数表示学习率衰减为上次的gamma分之一
# torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                      milestones=[30, 60, 80, 100, 120, 140], gamma=0.5)

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


# def MSE(y_pred, gt):
#     loss = 0
#     loss += 0.5 * np.sum((y_pred - gt)**2)
#     vec_gt = [[0]*3] * 5
#     for w in range(4):
#         vec_gt[w] = np.array([gt[w][0],
#                               gt[w][1]])
#     vector_gt = vec_gt[1]-vec_gt[0]
#     vec_pred = [[0]*3] * 5
#     for v in range(4):
#         vec_pred[w] = np.array([y_pred[w][0],
#                                 y_pred[w][1]])
#     vector_pred = vec_pred[1]-vec_pred[0]
#     loss += (vector_gt[0]*vector_pred[1]-vector_pred[0]*vector_gt[1])**0.5

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


for epoch in range(num_epochs):
    tmp = 0
    for i, (data, gt, mask, item, imgPath, heatmaps_targets) in enumerate(train_loader):
        # print(i)
        data = data.to(device)
        gt = gt.to(device)
        mask = mask.to(device)
        gt = gt.view(-1, 8)
        heatmaps_targets = heatmaps_targets.to(device)
        # mask, indices_valid = calculate_mask(heatmaps_targets)

        # print(heatmaps_targets.shape)
        print("heatmaps_targets.shape", heatmaps_targets.shape)
        heatmaps_targets = get_peak_points(heatmaps_targets.cpu().data.numpy())
        print("imgPath", imgPath)
        print("gt", gt)
        print("heatmaps_targets", heatmaps_targets)

        break

        # Forward pass
        outputs = model(data)
        # outputs = outputs * mask
        # heatmaps_targets = heatmaps_targets * mask
        print(outputs.shape)

        loss = loss_fn(outputs, heatmaps_targets)

        tmp += loss.item()
        # exit()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}, average_loss: {:.4f}, learning_rate: {}".format(
                epoch + 1, num_epochs, i + 1, total_step, loss.item(), tmp / (i+1), optimizer.state_dict()['param_groups'][0]['lr']))

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), '{}heatmap2.ckpt'.format(epoch + 1))

# card2 heatmap 42756
