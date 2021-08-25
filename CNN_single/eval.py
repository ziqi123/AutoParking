import torch
import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
import os
import cv2
from PIL import Image
from network_exp_baseline import network
from scipy.optimize import linear_sum_assignment
from dataloader.DataLoaders import DataLoaders
# from basic_cnn import BasicCNN

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
best_test_loss = np.inf

params = dict()
params['data_normalize_factor'] = 256
params['dataset_dir'] = ""
params['rgb2gray'] = False
params['dataset'] = "CNNDataset"
params['train_batch_sz'] = 3
params['val_batch_sz'] = 1

dataloaders, dataset_sizes = DataLoaders(params)
test_loader = dataloaders['val']
train_loader = dataloaders['train']

# 计算四个点的预测点与真值点之间的误差


def get_acc(y, y_hat, dis):
    total = 0

    total += ((y[0]-y_hat[0])**2 + (y[1]-y_hat[1])**2)**0.5

    if total < dis:
        return 1
    else:
        return 0

# 标点


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

    # print("points_list", points_list)

    cv2.circle(img, (int(points_list[0]), int(points_list[1])),
               point_size, point_color, thickness)
    cv2.circle(img, (int(points_list2[0]), int(points_list2[1])),
               point_size, point_color2, thickness)

    # print(save_path)
    cv2.imwrite(save_path, img)


# Define your model
# model = BasicCNN()
model = network()
model.load_state_dict(torch.load(
    '/media/home_bak/ziqi/park/CNN_single/100.ckpt'))

# move model to the right device
model.to(device)

# Loss and optimizer
loss_fn = torch.nn.MSELoss()

# For updating learning rate
total_step = len(test_loader)


total = 0
validation_loss = 0.0
accuracy = [0 for i in range(21)]
loss_array = []

for i, (data, gt, mask, item, imgPath) in enumerate(test_loader):

    data = data.to(device)
    gt = gt.to(device)
    mask = mask.to(device)
    num_pslot = params['train_batch_sz']
    # gt, indices = torch.sort(gt, 1)
    gt = gt.view(-1, 2)

    # Forward pass
    outputs = model(data)

    # 标点
    outputs = outputs.view(-1).cpu()
    gt = gt.view(-1).cpu()
    f = os.path.basename(imgPath[0])
    save_path2 = os.path.join(
        "/media/home_bak/ziqi/park/CNN_single/test_img", f)
    colorize(outputs, gt, imgPath[0], save_path2)
    # print(save_path2)
    # 保存预测值
    txt_file = save_path2.replace('test_img', 'output').replace(
        '.jpg', '.txt')
    t = outputs.cuda().data.cpu().numpy()
    with open(txt_file, "w") as file:
        file.write(str(t[0]))
        file.write(' ')
        file.write(str(t[1]))
        file.write('\n')
        # f.writelines(point)

    # loss = loss_fn(outputs, gt)
    # validation_loss += loss.item()
    # loss_array.append(loss.item())
    # score = 0

    # 计算精度
    for p in range(21):
        tmp = get_acc(outputs, gt, p)
        accuracy[p] += tmp

    # if i % 100 == 0:
    #     print("Step [{}/{}] Loss: {:.4f}".format(i +
    #           1, total_step, loss.item()))

total = len(test_loader)
# print("accuracy", accuracy, total)
# print('Accuracy of the model on the test images: {} %'.format(100 * accuracy / total))
for x in range(21):
    accuracy[x] = 100 * accuracy[x] / total
    accuracy[x] = round(accuracy[x], 3)

print(accuracy)
# validation_loss /= len(test_loader)
# plt.xlabel('sample')
# plt.ylabel('loss')
# plt.figure()
# plt.plot(loss_array)
# plt.savefig("/media/home_bak/ziqi/park/CNN/test_loss.png")


# 设置画布大小
plt.figure(figsize=(20, 8))

# 标题
plt.title("accruracy distribution")

# # 数据
# plt.plot(x, accuracy, label='weight changes', linewidth=3, color='r', marker='o',
#          markerfacecolor='blue', markersize=8)


plt.bar(range(len(accuracy)), accuracy)

# 横坐标描述
plt.xlabel('pixel')

# 纵坐标描述
plt.ylabel('accuracy(%)')


# 设置数字标签
# for a, b in zip(x, accuracy):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.savefig("/media/home_bak/ziqi/park/CNN/accuracy.png")
