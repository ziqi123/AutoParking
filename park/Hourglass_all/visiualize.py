import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from dataloader.heatmap_Dataloader_all import heatmap_Dataloader
import os
from hourglass_all import KFSGNet
import torch.nn as nn
import torchvision.transforms as transforms
from heatmappy import Heatmapper
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
params['train_batch_sz'] = 1
params['val_batch_sz'] = 1
params['sigma'] = 3

dataloaders, dataset_sizes = heatmap_Dataloader(params)

train_loader = dataloaders['train']
test_loader = dataloaders['val']

# Define your model
# model = BasicCNN()

model = KFSGNet()
model.load_state_dict(torch.load(
    '/media/home_bak/ziqi/park/Hourglass_all/70heatmap3.ckpt'))

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


def get_acc(path, y, y_hat):
    f = os.path.basename(path)
    file = os.path.join(
        "/media/home_bak/ziqi/park/Hourglass_all/train_img", f)
    # save_img = os.path.join(
    #     "/media/home_bak/ziqi/park/Hourglass_all/error_img", f)
    # img = cv2.imread(file)
    total = 0
    for i in range(2):
        total += ((y[i][0]-y_hat[2*i])**2 + (y[i][1]-y_hat[2*i+1])**2)**0.5
    total /= 2
    # if total > 10:
    #     cv2.imwrite(save_img, img)

    if total <= 5:
        return 1
    else:
        return 0


def colorize(outputs, gt, path, save_path):
    # outputs = outputs.cuda().data.cpu().numpy()
    gt = gt.cuda().data.cpu().numpy()
    img = cv2.imread(path)

    points_list2 = np.array(gt, dtype=np.float)
    point_size = 1
    point_color = (0, 0, 255)
    point_color2 = (0, 255, 0)
    thickness = 4  # ????????? 0???4???8

    # print(outputs[0])
    for i in range(4):
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


loss_array = []
accuracy = 0
for i, (data, gt, mask, item, imgPath, heatmaps_targets) in enumerate(train_loader):
    data = data.to(device)
    gt = gt.to(device)
    mask = mask.to(device)
    gt = gt.view(-1, 8)
    heatmaps_targets = heatmaps_targets.to(device)
    # print(heatmaps_targets.shape)

    # Forward pass
    outputs = model(data)
    # ????????????????????????
    all_peak_points = get_peak_points(
        outputs.cpu().data.numpy())

    # print(all_peak_points)

    for k in range(len(imgPath)):
        f = os.path.basename(imgPath[k])
        save_path2 = os.path.join(
            "/media/home_bak/ziqi/park/Hourglass_all/train_img", f)
        colorize(all_peak_points[k], gt[k], imgPath[k], save_path2)

    # ??????
    gt = gt.cuda().data.cpu().numpy()
    gt = gt.tolist()

    for s in range(len(imgPath)):
        tmp = get_acc(imgPath[s], all_peak_points[s], gt[s])
        accuracy += tmp

    for p in range(len(imgPath)):
        f = os.path.basename(imgPath[p])
        path_gt = os.path.join(
            "/media/home_bak/ziqi/park/Hourglass_all/heatmap_gt_train", f)
        # heatmapper = Heatmapper()
        point = []
        point = [(int(all_peak_points[p][0][0]),
                  int(all_peak_points[p][0][1])),
                 (int(all_peak_points[p][1][0]),
                  int(all_peak_points[p][1][1])),
                 (int(all_peak_points[p][2][0]),
                  int(all_peak_points[p][2][1])),
                 (int(all_peak_points[p][3][0]),
                  int(all_peak_points[p][3][1])),
                 ]
        heatmapper = Heatmapper(opacity=0.9, colours='reveal')
        # print(point)
        # example_img = Image.open(imgPath[p])
        heatmap = heatmapper.heatmap_on_img_path(point, imgPath[p])
        # heatmap = heatmapper.heatmap_on_img(point, example_img)
        heatmap = heatmap.convert("RGB")
        heatmap.save(path_gt)

print("accuracy", accuracy/len(train_loader))
