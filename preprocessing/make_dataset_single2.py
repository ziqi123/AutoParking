import os
import cv2
import numpy as np

# 标点


def colorize(img, point, save_path, point_color):
    point_size = 1
    thickness = 4  # 可以为 0、4、8
    for i in range(2):
        cv2.circle(img, (int(float(point[2*i])), int(float(point[2*i+1]))),
                   point_size, point_color, thickness)
    cv2.imwrite(save_path, img)

# 保存单个点的坐标


def save_point(point, point_crop_path):
    with open(point_crop_path, "w") as f:
        for i in range(2):
            f.write(str(point[2*i]))
            f.write(' ')
            f.write(str(point[2*i+1]))
            f.write('\n')

# 切割图片的一个角


def Crop(img_path, point_dir, save_path, point_crop_path):
    img = cv2.imread(img_path)
    cropped = img[0:192, 0:383]
    cv2.imwrite(save_path, cropped)
    point = []
    with open(point_dir, "r") as f:
        annt = f.readlines()
    i = 0
    for line in annt:
        i = i+1
        line_annt = line.strip('\n').split(' ')
        point.append(line_annt[0])
        point.append(line_annt[1])
        if i == 2:
            break
    # print(point)
    colorize(cropped, point, save_path, point_color=(0, 255, 0))
    save_point(point, point_crop_path)


if __name__ == "__main__":
    data_dir = '/media/home_bak/ziqi/park/single_pic/train2/pic'
    point_dir = '/media/home_bak/ziqi/park/single_pic/train2/point'
    save_dir = '/media/home_bak/ziqi/park/single_pic/train2/pic_crop'
    point_crop_dir = '/media/home_bak/ziqi/park/single_pic/train2/point_crop'
    path_record = '/media/home_bak/ziqi/park/single_pic/train2/path_record.txt'
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
