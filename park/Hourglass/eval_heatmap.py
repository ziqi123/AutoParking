import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from dataloader.heatmap_Dataloader import heatmap_Dataloader
# from basic_cnn import BasicCNN
import os
from hourglass import KFSGNet
import matplotlib.pyplot as plt  # plt 用于显示图片
import torch.nn as nn
from heatmappy import Heatmapper
import matplotlib
import copy
matplotlib.use('Agg')
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

# train_loader = dataloaders['train']
test_loader = dataloaders['val']

# Define your model
# model = BasicCNN()

model = KFSGNet()

# move model to the right device
model.to(device)
model.load_state_dict(torch.load(
    '/media/home_bak/ziqi/park/Hourglass/130heatmap4.ckpt'))


# Loss and optimizer
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(optimizer.state_dict()['param_groups'][0]['lr'])


# Train the model
# total_step = len(train_loader)
curr_lr = learning_rate

print("start")


def get_acc(path, y, y_hat, dis):
    f = os.path.basename(path)
    file = os.path.join(
        "/media/home_bak/ziqi/park/Hourglass/test_img3", f)
    save_img = os.path.join(
        "/media/home_bak/ziqi/park/Hourglass/error_img3", f)
    img = cv2.imread(file)
    total = 0
    for i in range(2):
        total += ((y[i][0]-y_hat[2*i])**2 + (y[i][1]-y_hat[2*i+1])**2)**0.5
    total /= 2
    if total > 10:
        cv2.imwrite(save_img, img)

    if total <= dis:
        return 1
    else:
        return 0


def colorize(outputs, gt, path, save_path, point_path, img_path, img_back_path, gt_back_path):
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
    cv2.imwrite(save_path, img)

    point = np.loadtxt(gt_back_path)
    point[0][0] = point[0][0]+48
    point[0][1] = point[0][1]+48
    point[1][0] = point[1][0]+240-96
    point[1][1] = point[1][1]+48

    img2 = cv2.imread(img_path)
    outputs1 = copy.deepcopy(outputs)
    outputs1[0][0] = outputs[0][0]+48
    outputs1[0][1] = outputs[0][1]+48
    outputs1[1][0] = outputs[1][0]+240-96
    outputs1[1][1] = outputs[1][1]+48
    for j in range(2):
        cv2.circle(img2, (int(point[j][0]), int(point[j][1])),
                   point_size, point_color2, thickness)
        cv2.circle(img2, (outputs1[j][0], outputs1[j][1]),
                   point_size, point_color, thickness)

    # print(save_path)
    cv2.imwrite(img_back_path, img2)

    with open(point_path, "w") as f:
        for k in range(2):
            f.write(str(outputs1[k][0]))
            f.write(' ')
            f.write(str(outputs1[k][1]))
            f.write('\n')


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
accuracy = [0 for i in range(21)]
for i, (data, gt, mask, item, imgPath, heatmaps_targets) in enumerate(test_loader):
    data = data.to(device)
    gt = gt.to(device)
    mask = mask.to(device)
    gt = gt.view(-1, 4)
    heatmaps_targets = heatmaps_targets.to(device)

    # Forward pass
    outputs = model(data)

    # 热图还原为坐标点
    all_peak_points = get_peak_points(
        outputs.cpu().data.numpy())

    # 标点
    for k in range(len(imgPath)):
        f = os.path.basename(imgPath[k])
        save_path2 = os.path.join(
            "/media/home_bak/ziqi/park/Hourglass/test_img3", f)
        point_path = os.path.join(
            "/media/home_bak/ziqi/park/Hourglass/point3", f.strip('.jpg')+'.txt')
        img_path = os.path.join(
            "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/perspective_img", f)
        img_back_path = os.path.join(
            "/media/home_bak/ziqi/park/Hourglass/test_img_back3", f)
        gt_back_path = os.path.join(
            "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/match_img_point", f.strip('.jpg')+'_OA.txt')
        img_back_path = colorize(all_peak_points[k], gt[k], imgPath[k],
                                 save_path2, point_path, img_path, img_back_path, gt_back_path)

    # 计算精度
    for p in range(21):
        for s in range(len(imgPath)):
            tmp = get_acc(imgPath[s], all_peak_points[s], gt[s], p)
            accuracy[p] += tmp

    gt = gt.cuda().data.cpu().numpy()
    gt = gt.tolist()

    for p in range(len(imgPath)):
        f = os.path.basename(imgPath[p])
        path_gt = os.path.join(
            "/media/home_bak/ziqi/park/Hourglass/heatmap_gt3", f)
        # heatmapper = Heatmapper()
        point = []
        point = [(int(all_peak_points[p][0][0]),
                  int(all_peak_points[p][0][1])),
                 (int(all_peak_points[p][1][0]),
                 int(all_peak_points[p][1][1])),
                 ]
        heatmapper = Heatmapper(opacity=0.9, colours='reveal')
        # print(point)
        # example_img = Image.open(imgPath[p])
        heatmap = heatmapper.heatmap_on_img_path(point, imgPath[p])
        # heatmap = heatmapper.heatmap_on_img(point, example_img)
        heatmap = heatmap.convert("RGB")
        heatmap.save(path_gt)

total = len(test_loader)
for x in range(21):
    accuracy[x] = 100 * accuracy[x] / total
    accuracy[x] = round(accuracy[x], 3)

print(accuracy)


# 设置画布大小
plt.figure(figsize=(20, 8))

# 标题
plt.title("accruracy distribution")

plt.bar(range(len(accuracy)), accuracy)

# 横坐标描述
plt.xlabel('pixel')

# 纵坐标描述
plt.ylabel('accuracy(%)')
plt.show()
plt.savefig("/media/home_bak/ziqi/park/Hourglass/accuracy3.png")
