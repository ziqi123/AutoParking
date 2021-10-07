import torch
import torchvision.transforms as transforms
from dataloader.dataloader_wing import heatmap_Dataloader
import os
from net_wing import KFSGNet
import torchvision.transforms as transforms
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
params['train_batch_sz'] = 16
params['val_batch_sz'] = 1
params['sigma'] = 3

dataloaders, dataset_sizes = heatmap_Dataloader(params)

train_loader = dataloaders['train']
test_loader = dataloaders['val']

# Define your model

model = KFSGNet()

# move model to the right device
model.to(device)

model.train()


def wing_loss(y_true, y_pred, w, epsilon, N_LANDMARK):
    y_pred = y_pred.reshape(-1, N_LANDMARK, 2)
    y_true = y_true.reshape(-1, N_LANDMARK, 2)

    x = y_true - y_pred
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(w > absolute_x,
                         w * torch.log(1.0 + absolute_x / epsilon),
                         absolute_x - c)
    loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0).requires_grad_()
    return loss


# Loss and optimizer
# loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 多步长学习率衰减
# 不同的区间采用不同的更新频率，或者是有的区间更新学习率，有的区间不更新学习率
# 其中milestones参数为表示学习率更新的起止区间，在区间[0. 200]内学习率不更新，
# 而在[200, 300]、[300, 320].....[340, 400]的右侧值都进行一次更新；
# gamma参数表示学习率衰减为上次的gamma分之一
# torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                      milestones=[40, 50, 60, 70, 80, 90, 100], gamma=0.5)

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


for epoch in range(num_epochs):
    tmp = 0
    for i, (data, gt, mask, item, imgPath, heatmaps_targets) in enumerate(train_loader):
        # print(i)
        data = data.to(device)
        gt = gt.to(device)
        mask = mask.to(device)
        gt = gt.view(-1, 8)
        heatmaps_targets = heatmaps_targets.to(device)
        mask, indices_valid = calculate_mask(heatmaps_targets)

        # print(heatmaps_targets.shape)

        # Forward pass
        outputs = model(data)
        outputs = outputs * mask
        heatmaps_targets = heatmaps_targets * mask

        # loss = loss_fn(outputs, heatmaps_targets)
        loss = wing_loss(heatmaps_targets, outputs,
                         w=10.0, epsilon=2.0, N_LANDMARK=192)

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


# card2 heatmap2 37406
