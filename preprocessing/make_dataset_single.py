import numpy as np
import os
import cv2
from PIL import Image
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt  # plt 用于显示图片
from tqdm import tqdm

# 标点


def colorize(img, point, save_path, point_color):
    point_size = 1
    thickness = 4  # 可以为 0、4、8
    cv2.circle(img, (int(float(point[0])), int(float(point[1]))),
               point_size, point_color, thickness)
    cv2.imwrite(save_path, img)

# 保存单个点的坐标


def save_point(point, point_crop_path):
    with open(point_crop_path, "w") as f:
        f.write(str(point[0]))
        f.write(' ')
        f.write(str(point[1]))
        f.write('\n')

# 切割图片的一个角


def Crop(img_path, point_dir, save_path, point_crop_path):
    img = cv2.imread(img_path)
    cropped = img[0:192, 0:192]
    cv2.imwrite(save_path, cropped)
    point = []
    with open(point_dir, "r") as f:
        annt = f.readlines()
    for line in annt:
        line_annt = line.strip('\n').split(' ')
        point.append(line_annt[0])
        point.append(line_annt[1])
        break
    colorize(cropped, point, save_path, point_color=(0, 255, 0))
    save_point(point, point_crop_path)


if __name__ == "__main__":
    data_dir = '/media/home_bak/ziqi/park/single_pic/train/pic'
    point_dir = '/media/home_bak/ziqi/park/single_pic/train/point'
    save_dir = '/media/home_bak/ziqi/park/single_pic/train/pic_crop'
    point_crop_dir = '/media/home_bak/ziqi/park/single_pic/train/point_crop'
    path_record = '/media/home_bak/ziqi/park/single_pic/train/path_record.txt'
    f = open(path_record, 'w')
    for i in os.listdir(data_dir):
        img_path = os.path.join(data_dir, i)
        point_path = os.path.join(point_dir, i.strip('.jpg')+'_OA.txt')
        save_path = os.path.join(save_dir, i)

        point_crop_path = os.path.join(
            point_crop_dir, i.strip('.jpg')+'_OA.txt')
        Crop(img_path, point_path, save_path, point_crop_path)
        f.write(save_path)
        f.write('\n')
    f.close()
