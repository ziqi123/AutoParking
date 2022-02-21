import copy
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from dataloader.dataloader_hourglass import heatmap_Dataloader
import os
from network import KFSGNet
import matplotlib.pyplot as plt  # plt 用于显示图片
from heatmappy import Heatmapper
import matplotlib
import math
matplotlib.use('Agg')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 150
learning_rate = 0.001

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

model.load_state_dict(torch.load(
    '/media/home_bak/ziqi/park/hourglass/ckpt2/30heatmap.ckpt'))
model.to(device)

# Loss and optimizer
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(optimizer.state_dict()['param_groups'][0]['lr'])


# Train the model
# total_step = len(train_loader)
curr_lr = learning_rate

print("start")

# 计算点偏差，求精度


def get_acc(path, y, y_hat, dis):
    f = os.path.basename(path)
    file = os.path.join(markImg_192_dir, f)
    save_img = os.path.join(
        error_dir, f)
    img = cv2.imread(file)
    # print(img)
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

# 计算两点与原点的夹角


# def angle_between(p1, p2):
#     ang1 = np.arctan2(*p1[::-1])
#     ang2 = np.arctan2(*p2[::-1])
#     return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def azimuthAngle(x1,  y1,  x2,  y2):
    angle = math.atan2((y2-y1), (x2-x1))
    return (angle * 180 / math.pi)

# 计算角偏差，求精度


def get_angle_acc(point_pred_384_path, point_384_gt_path, pix):
    point_pred = np.loadtxt(point_pred_384_path)
    point_gt = np.loadtxt(point_384_gt_path)
    vec = [[0]*3] * 5
    vec_gt = [[0]*3] * 5
    for w in range(4):
        vec[w] = np.array([point_pred[w][0],
                           point_pred[w][1]])
        vec_gt[w] = np.array([point_gt[w][0],
                             point_gt[w][1]])

    vector1 = vec[2]-vec[0]
    vector2 = vec[3]-vec[1]
    vector_norm1 = vector1/np.linalg.norm(vector1)
    vector_norm2 = vector2/np.linalg.norm(vector2)
    vector_end1 = vec[0]+vector_norm1*20
    vector_end2 = vec[1]+vector_norm2*20

    vector1_gt = vec_gt[2]-vec_gt[0]
    vector2_gt = vec_gt[3]-vec_gt[1]
    vector_norm1_gt = vector1_gt/np.linalg.norm(vector1_gt)
    vector_norm2_gt = vector2_gt/np.linalg.norm(vector2_gt)
    vector_end1_gt = vec_gt[0]+vector_norm1_gt*20
    vector_end2_gt = vec_gt[1]+vector_norm2_gt*20

    # angle1 = angle_between(vec[0], vector_end1)
    # angle1_gt = angle_between(vec_gt[0], vector_end1_gt)

    # angle2 = angle_between(vec[1], vector_end2)
    # angle2_gt = angle_between(vec_gt[1], vector_end2_gt)
    angle1 = azimuthAngle(vec[0][0], vec[0][1], vector_end1[0], vector_end1[1])
    angle1_gt = azimuthAngle(
        vec_gt[0][0], vec_gt[0][1], vector_end1_gt[0], vector_end1_gt[1])
    angle2 = azimuthAngle(vec[1][0], vec[1][1], vector_end2[0], vector_end2[1])
    angle2_gt = azimuthAngle(
        vec_gt[1][0], vec_gt[1][1], vector_end2_gt[0], vector_end2_gt[1])

    total = abs(angle1-angle1_gt)+abs(angle2-angle2_gt)

    total /= 2

    if total <= pix:
        return 1
    else:
        return 0


# 绘制箭头


def cvArrow(img, pt1, pt2, imgPath):
    img = cv2.arrowedLine(img, (int(pt1[0]), int(pt1[1])),
                          (int(pt2[0]), int(pt2[1])), (0, 0, 255), 3, 8, 0, 0.1)
    cv2.imwrite(imgPath, img)

# 画方向


def draw_angle(img_path, point_path):
    point_pred = np.loadtxt(point_path)
    img = cv2.imread(img_path)
    vec = [[0]*3] * 5
    for w in range(4):
        vec[w] = np.array([point_pred[w][0],
                           point_pred[w][1]])
    vector1 = vec[2]-vec[0]
    vector2 = vec[3]-vec[1]
    vector_norm1 = vector1/np.linalg.norm(vector1)
    vector_norm2 = vector2/np.linalg.norm(vector2)
    vector_end1 = vec[0]+vector_norm1*180
    vector_end2 = vec[1]+vector_norm2*180
    # print(vector_end1)
    cvArrow(img, vec[0], vector_end1, img_path)
    cvArrow(img, vec[1], vector_end2, img_path)


# 标点函数
def draw_point(img, point, point_size, point_color, thickness):
    for i in range(4):
        cv2.circle(img, (int(point[i][0]), int(point[i][1])),
                   point_size, point_color, thickness)

# 保存坐标点函数


def save_point(point, point_path):
    with open(point_path, "w") as f:
        for k in range(4):
            f.write(str(point[k][0]))
            f.write(' ')
            f.write(str(point[k][1]))
            f.write('\n')

# 在192*192图上标记预测点和真值点


def colorize(img_path, markImg_192_path, outputs, gt):
    # outputs = outputs.cuda().data.cpu().numpy()
    gt = gt.cuda().data.cpu().numpy()
    img = cv2.imread(img_path)

    points_list = np.array(gt, dtype=np.float)

    for i in range(4):
        cv2.circle(img, (int(points_list[2*i]), int(points_list[2*i+1])),
                   point_size, point_color2, thickness)
    draw_point(img, outputs, point_size, point_color, thickness)

    cv2.imwrite(markImg_192_path, img)


# 将192*192上的预测点和真值还原到384*384上，并在384*384图片上标点
def colorize_back(outputs, img_384_path, markImg_384_path, point_pred_384_path,
                  gt_192_path, point_384_gt_path):
    gt_192 = np.loadtxt(gt_192_path)
    # 从192*192图片上的坐标点变换到384*384图片上的坐标点
    gt_384 = np.array(gt_192, copy=True)
    gt_384[0][0] = gt_192[0][0]+48
    gt_384[0][1] = gt_192[0][1]+48
    gt_384[1][0] = gt_192[1][0]+240-96
    gt_384[1][1] = gt_192[1][1]+48
    gt_384[2][0] = gt_192[2][0]+48
    gt_384[2][1] = gt_192[2][1]+240-96
    gt_384[3][0] = gt_192[3][0]+240-96
    gt_384[3][1] = gt_192[3][1]+240-96

    outputs_384 = np.array(outputs, copy=True)
    outputs_384[0][0] = outputs[0][0]+48
    outputs_384[0][1] = outputs[0][1]+48
    outputs_384[1][0] = outputs[1][0]+240-96
    outputs_384[1][1] = outputs[1][1]+48
    outputs_384[2][0] = outputs[2][0]+48
    outputs_384[2][1] = outputs[2][1]+240-96
    outputs_384[3][0] = outputs[3][0]+240-96
    outputs_384[3][1] = outputs[3][1]+240-96

    img = cv2.imread(img_384_path)

    draw_point(img, gt_384, point_size, point_color2, thickness)
    draw_point(img, outputs_384, point_size, point_color, thickness)

    # print(save_path)
    cv2.imwrite(markImg_384_path, img)

    # 保存坐标点到384*384图片中
    save_point(outputs_384, point_pred_384_path)
    save_point(gt_384, point_384_gt_path)


# 求出热图上的最大值，即为预测点


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


# def color(img_384_path, gt_192_path, test_path):
#     img = cv2.imread(img_384_path)
#     gt_192 = np.loadtxt(gt_192_path)
#     # 从192*192图片上的坐标点变换到384*384图片上的坐标点
#     gt_384 = np.array(gt_192, copy=True)
#     gt_384[0][0] = gt_192[0][0]+48
#     gt_384[0][1] = gt_192[0][1]+48
#     gt_384[1][0] = gt_192[1][0]+240-96
#     gt_384[1][1] = gt_192[1][1]+48
#     gt_384[2][0] = gt_192[2][0]+48
#     gt_384[2][1] = gt_192[2][1]+240-96
#     gt_384[3][0] = gt_192[3][0]+240-96
#     gt_384[3][1] = gt_192[3][1]+240-96

#     draw_point(img, gt_384, point_size, point_color2, thickness)

#     cv2.imwrite(test_path, img)


loss_array = []
accuracy = [0 for i in range(21)]
accuracy_angle = [0 for j in range(10)]
# 384*384数据集
img_384_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/img_384"
# 保存最终标记上预测点和方向的192*192图片
markImg_192_dir = "./mark_test_img_192"
test_dir = "/media/home_bak/ziqi/park/hourglass/test"
# 保存最终标记上预测点和方向的384*384图片
markImg_384_dir = "./mark_test_img_384"
# 192*192上的真值
gt_192_gt_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/point_192"
# 192*192上的预测点还原为384*384上的预测点
point_pred_384_dir = "./point_pred_384"
# 192*192上的真值还原为384*384上的真值
point_384_gt_dir = "./point_384_gt"
error_dir = "/media/home_bak/ziqi/park/hourglass/error_img"
point_size = 1
point_color = (0, 0, 255)
point_color2 = (0, 255, 0)
thickness = 4  # 可以为 0、4、8

for i, (data, gt, mask, item, imgPath, heatmaps_targets) in enumerate(test_loader):
    data = data.to(device)
    gt = gt.to(device)
    mask = mask.to(device)
    gt = gt.view(-1, 8)
    heatmaps_targets = heatmaps_targets.to(device)

    # # Forward pass
    outputs = model(data)

    # # 热图还原为坐标点
    all_peak_points = get_peak_points(
        outputs.cpu().data.numpy())

    # # 标点
    for k in range(len(imgPath)):
        f = os.path.basename(imgPath[k])
        markImg_192_path = os.path.join(markImg_192_dir, f)
        point_pred_384_path = os.path.join(
            point_pred_384_dir, f.strip('.jpg')+'.txt')
        img_384_path = os.path.join(img_384_dir, f)
        markImg_384_path = os.path.join(markImg_384_dir, f)
        gt_192_path = os.path.join(gt_192_gt_dir, f.strip('.jpg')+'_OA.txt')
        point_384_gt_path = os.path.join(
            point_384_gt_dir, f.strip('.jpg')+'.txt')
        # 在192*192图片上标记预测点和真值点
        colorize(imgPath[k], markImg_192_path, all_peak_points[k], gt[k])
        # test_path = os.path.join(
        #     test_dir, f)
        # # 将192*192上的预测点和真值还原到384*384上，并在384*384图片上标点
        # colorize_back(all_peak_points[k], img_384_path, markImg_384_path, point_pred_384_path,
        #               gt_192_path, point_384_gt_path)
#         # color(img_384_path, gt_192_path, test_path)
#         draw_angle(markImg_384_path, point_pred_384_path)

    # 计算点的精度
    for p in range(21):
        for s in range(len(imgPath)):
            tmp = get_acc(imgPath[s], all_peak_points[s], gt[s], p)
            accuracy[p] += tmp

#     # 计算384*384图片上角度的精度
#     for pix in range(10):
#         for c in range(len(imgPath)):
#             f = os.path.basename(imgPath[c])
#             point_pred_384_path = os.path.join(
#                 point_pred_384_dir, f.strip('.jpg')+'.txt')
#             point_384_gt_path = os.path.join(
#                 point_384_gt_dir, f.strip('.jpg')+'.txt')
#             tmp = get_angle_acc(point_pred_384_path, point_384_gt_path, pix)
#             accuracy_angle[pix] += tmp

#     gt = gt.cuda().data.cpu().numpy()
#     gt = gt.tolist()

#     for p in range(len(imgPath)):
#         f = os.path.basename(imgPath[p])
#         path_gt = os.path.join(
#             "./heatmap_gt_test", f)
#         # heatmapper = Heatmapper()
#         point = []
#         point = [(int(all_peak_points[p][0][0]),
#                   int(all_peak_points[p][0][1])),
#                  (int(all_peak_points[p][1][0]),
#                  int(all_peak_points[p][1][1])),
#                  (int(all_peak_points[p][2][0]),
#                  int(all_peak_points[p][2][1])),
#                  (int(all_peak_points[p][3][0]),
#                  int(all_peak_points[p][3][1])),

#                  ]
#         heatmapper = Heatmapper(opacity=0.9, colours='reveal')
#         # print(point)
#         # example_img = Image.open(imgPath[p])
#         heatmap = heatmapper.heatmap_on_img_path(point, imgPath[p])
#         # heatmap = heatmapper.heatmap_on_img(point, example_img)
#         heatmap = heatmap.convert("RGB")
#         heatmap.save(path_gt)

total = len(test_loader)
for x in range(21):
    accuracy[x] = 100 * accuracy[x] / total
    accuracy[x] = round(accuracy[x], 3)

print("accuracy:", accuracy)


# for y in range(10):
#     accuracy_angle[y] = 100 * accuracy_angle[y] / total
#     accuracy_angle[y] = round(accuracy_angle[y], 3)


# print("accuracy_angle:", accuracy_angle)


# # 设置画布大小
# plt.figure(figsize=(20, 8))

# # 标题
# plt.title("accruracy distribution")

# plt.bar(range(len(accuracy)), accuracy)

# # 横坐标描述
# plt.xlabel('pixel')

# # 纵坐标描述
# plt.ylabel('accuracy(%)')
# plt.show()
# plt.savefig("./accuracy3.png")
