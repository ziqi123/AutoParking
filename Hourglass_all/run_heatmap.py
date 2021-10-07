import torch
import torchvision.transforms as transforms
from dataloader.heatmap_Dataloader_all import heatmap_Dataloader
import os
from hourglass_all import KFSGNet
import torchvision.transforms as transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 150
learning_rate = 0.0001

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
        # mask, indices_valid = calculate_mask(heatmaps_targets)

        # print(heatmaps_targets.shape)

        # Forward pass
        outputs = model(data)
        # outputs = outputs * mask
        # heatmaps_targets = heatmaps_targets * mask

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
        torch.save(model.state_dict(), '{}heatmap3.ckpt'.format(epoch + 1))
