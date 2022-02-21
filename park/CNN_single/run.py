import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from dataloader.DataLoaders import DataLoaders
# from basic_cnn import BasicCNN
from net import network
import matplotlib.pyplot as plt  # plt 用于显示图片
from scipy.optimize import linear_sum_assignment
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
params['train_batch_sz'] = 64
params['val_batch_sz'] = 1

dataloaders, dataset_sizes = DataLoaders(params)

train_loader = dataloaders['train']
test_loader = dataloaders['val']


# Define your model
# model = BasicCNN()

model = network()

# move model to the right device
model.to(device)
model.train()

# Loss and optimizer
loss_fn = torch.nn.MSELoss()

# you can use Nesterov momentum in optim.SGD
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
#                             momentum=0.9, nesterov=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(optimizer.state_dict()['param_groups'][0]['lr'])

# For updating learning rate


# def update_lr(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# Train the model
total_step = len(train_loader)
curr_lr = learning_rate

print("start")


def colorize(outputs, gt, path, save_path):
    outputs = outputs.cuda().data.cpu().numpy()
    gt = gt.cuda().data.cpu().numpy()
    img = cv2.imread(path)

    points_list = np.array(outputs, dtype=np.float)
    points_list2 = np.array(gt, dtype=np.float)
    point_size = 1
    point_color = (0, 0, 255)
    point_color2 = (0, 255, 0)
    thickness = 4  # 可以为 0、4、8

    cv2.circle(img, (np.int(points_list[0]), np.int(points_list[1])),
               point_size, point_color, thickness)

    cv2.circle(img, (int(points_list2[0]), int(points_list2[1])),
               point_size, point_color2, thickness)
    # print(save_path)
    cv2.imwrite(save_path, img)


loss_array = []
for epoch in range(num_epochs):
    tmp = 0
    for i, (data, gt, mask, item, imgPath) in enumerate(train_loader):
        data = data.to(device)
        gt = gt.to(device)
        mask = mask.to(device)
        gt = gt.view(-1, 2)

        # Forward pass
        outputs = model(data)

        # print(outputs.shape)

        for k in range(len(imgPath)):
            f = os.path.basename(imgPath[k])
            save_path2 = os.path.join(
                "/media/home_bak/ziqi/park/CNN_single/train_img", f)
            colorize(outputs[k], gt[k], imgPath[k], save_path2)

        loss = loss_fn(outputs, gt)

        tmp += loss.item()
        # exit()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}, average_loss: {:.4f}, learning_rate: {}".format(
                epoch + 1, num_epochs, i + 1, total_step, loss.item(), tmp / (i+1), optimizer.state_dict()['param_groups'][0]['lr']))

    loss_array.append(tmp)
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), '{}run.ckpt'.format(epoch + 1))


# plt.xlabel('sample')
# plt.ylabel('loss')
# plt.figure()
# plt.plot(loss_array)
# plt.savefig("../CNN/train_loss.png")


# Save the model checkpoint
# torch.save(model.state_dict(), 'resnet.ckpt')
