from typing_extensions import Annotated
import numpy as np
import os
import cv2
from PIL import Image
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from numpy.lib.shape_base import column_stack
from numpy.lib.type_check import imag  # plt 用于显示图片
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
plt.switch_backend('agg')
# 计算四个点的预测点与真值点之间的误差


def get_acc(y, y_hat, dis, img_path):
    # print(y)
    # print(y_hat)
    total = 0
    for i in range(2):
        total += ((int(y[i][0])-int(y_hat[i][0]))**2 +
                  (int(y[i][1])-int(y_hat[i][1]))**2)**0.5
    total /= 2

    f = os.path.basename(img_path)

    save_path = os.path.join(
        "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/error_img", f)
    read_path = os.path.join(
        "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/fina", f)
    img = cv2.imread(read_path)
    if(total >= 10):
        cv2.imwrite(save_path, img)

    if total < dis:
        return 1
    else:
        return 0

# 转换坐标，标点


def colorize_pic(img_path, output_path, trans_path, fina_path, annt_path2):
    # 读原图
    img_pred = cv2.imread(img_path)
    point_pred = []
    trans_inv = []
    # 测试图片的预测点在384*384图片的坐标
    point_pred = np.loadtxt(output_path)

    # 384*384图片到原图的转换矩阵
    trans_inv = np.loadtxt(trans_path)

    trans_inv = trans_inv.reshape(2, 3)
    trans_inv = np.mat(trans_inv)

    # 把384*384中的点投影到原图上
    # dst = cv2.perspectiveTransform(point_pred, trans_inv)
    column = np.array([1, 1])
    point_pred = np.column_stack((point_pred, column))
    point_pred = point_pred.T
    # dst = cv2.perspectiveTransform(mat, trans_inv)
    dst = np.dot(trans_inv, point_pred)
    dst = dst.T
    # print("dst", dst)

    point_size = 1
    thickness = 4
    # 红色
    point_color = (0, 0, 255)
    point_color2 = (0, 255, 0)

    annt = np.loadtxt(annt_path2)
    # print(annt[0])
    # 把点画在图上
    # 原图中点的坐标
    # point = np.loadtxt(point_path)
    # point = point.T
    # p = np.dot(trans_inv, point)
    # p = p.T
    # p = np.asarray(p)

    # for i in range(2):
    #     cv2.circle(img_pred, (int(p[i][0]), int(p[i][1])),
    #                point_size, point_color2, thickness)

    for i in range(2):
        cv2.circle(img_pred, (int(annt[i][0]), int(annt[i][1])),
                   point_size, point_color2, thickness)

    cv2.imwrite(fina_path, img_pred)

    # dst = dst.reshape(2, 2)
    dst = np.asarray(dst)
    # print("dst", dst)
    # print(type(dst))

    for i in range(2):
        cv2.circle(img_pred, (int(dst[i][0]), int(dst[i][1])),
                   point_size, point_color, thickness)

    cv2.imwrite(fina_path, img_pred)

# 转换坐标，计算单个图片的精度


def pic_accuracy(output_path, trans_path, annt_path2, pix, img_path):
    # 读原图
    point_pred = []
    trans_inv = []
    # 测试图片的预测点在384*384图片的坐标
    point_pred = np.loadtxt(output_path)

    # 384*384图片到原图的转换矩阵
    trans_inv = np.loadtxt(trans_path)

    trans_inv = trans_inv.reshape(2, 3)
    trans_inv = np.mat(trans_inv)

    # 把384*384中的点投影到原图上
    # dst = cv2.perspectiveTransform(point_pred, trans_inv)
    column = np.array([1, 1])
    point_pred = np.column_stack((point_pred, column))
    point_pred = point_pred.T
    # dst = cv2.perspectiveTransform(mat, trans_inv)
    dst = np.dot(trans_inv, point_pred)
    dst = dst.T
    # print("dst", dst)

    annt = np.loadtxt(annt_path2)

    # dst = dst.reshape(2, 2)
    dst = np.asarray(dst)

    point_pred3 = dst

    # print(point_pred3)

    tmp = get_acc(point_pred3, annt, pix, img_path)

    return tmp

# 求测试集的预测点到原图中的精度


def get_accuracy(pix):
    accuracy = 0
    for i in os.listdir(test_dir):
        output_path = os.path.join(
            "/media/home_bak/ziqi/park/Hourglass/point3", i.strip('.jpg')+'.txt')
        trans_path = os.path.join(trans_dir, i.strip('.jpg')+'.txt')
        annt_path2 = os.path.join(annt_path_new, i.strip('.jpg')+'_OA.txt')
        img_path = os.path.join(src_img_dir, i)
        tmp = pic_accuracy(output_path,
                           trans_path, annt_path2, pix, img_path)
        accuracy += tmp
    return accuracy


if __name__ == "__main__":

    test_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/match_img"
    trans_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/trans_inv"
    src_img_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/src_img"
    annt_path_new = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/annt"
    fina_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/fina"

    for i in os.listdir(test_dir):
        output_path = os.path.join(
            "/media/home_bak/ziqi/park/Hourglass/point3", i.strip('.jpg')+'.txt')
        img_path = os.path.join(src_img_dir, i)
        trans_path = os.path.join(trans_dir, i.strip('.jpg')+'.txt')
        fina_path = os.path.join(fina_dir, i)
        annt_path2 = os.path.join(annt_path_new, i.strip('.jpg')+'_OA.txt')
        colorize_pic(img_path, output_path,
                     trans_path, fina_path, annt_path2)

    acc = []
    for k in range(15):
        x1 = get_accuracy(k)
        x1 = 100 * x1 / 8363
        acc.append(x1)

    x1 = round(x1, 3)
    print(acc)
    print(len(acc))

    # 设置画布大小
    plt.figure(figsize=(30, 15))

    # 标题
    plt.title("accruracy distribution")

    # 数据
    plt.bar(range(len(acc)), acc)

    # 横坐标描述
    plt.xlabel('pixel')

    # 纵坐标描述
    plt.ylabel('accuracy')
    # # 设置数字标签
    # for a, b in zip(x, acc):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    plt.savefig(
        "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/accuracy.png")
