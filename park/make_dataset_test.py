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

# 标注文件数据处理


def read_pslot(annt_1024_path):
    with open(annt_1024_path, "r") as f:
        annt = f.readlines()
    l = []
    l_ign = []
    for line in annt:
        line_annt = line.strip('\n').split(' ')

        if len(line_annt) != 13 or line_annt[0] != 'line' or line_annt[-4] == '3':
            continue

        if line_annt[-4] in ['0']:
            l.append(np.array([int(line_annt[i + 1]) for i in range(8)]))
            # continue

        # if line_annt[-4] in ['1', '5']:
        #     l_ign.append(np.array([int(line_annt[i + 1]) for i in range(8)]))
        #     continue
    return l, l_ign

# 标点


def colorize(points_list, img, save_path, item, line, point_color, flag):
    save_path = os.path.join(save_path, str(
        item.strip('.jpg'))+"_"+str(line)+".jpg")
    img2 = img.copy()
    point_size = 1
    thickness = 4  # 可以为 0、4、8
    if flag == 1:
        for i in range(4):
            cv2.circle(img2, (int(points_list[i][0]), int(points_list[i][1])),
                       point_size, point_color, thickness)
    cv2.imwrite(save_path, img2)

 # 画线


# 在原图中标出矩形框
def paint_line(img, dst, img_1024_with_rectangle_dir, num):
    img2 = img.copy()
    cv2.line(img2, (int(dst[0][0]), int(dst[0][1])), (int(
        dst[1][0]), int(dst[1][1])), (255, 0, 0), 5)
    cv2.line(img2, (int(dst[1][0]), int(dst[1][1])), (int(
        dst[2][0]), int(dst[2][1])), (255, 0, 0), 5)
    cv2.line(img2, (int(dst[2][0]), int(dst[2][1])), (int(
        dst[3][0]), int(dst[3][1])), (255, 0, 0), 5)
    cv2.line(img2, (int(dst[3][0]), int(dst[3][1])), (int(
        dst[0][0]), int(dst[0][1])), (255, 0, 0), 5)

    cropimg_path1 = os.path.join(
        img_1024_with_rectangle_dir, i.strip('.jpg')+'_'+str(num)+'.jpg')
    cv2.imwrite(cropimg_path1, img2)


def Crop_pic(ps, img_1024_path, img_1024_with_rectangle_dir, img_384_dir, annt_384_dir, i, trans_inv_dir, img_1024_singleSlot_dir, img_384_with_label_dir, annt_1024_singleSlot_dir):
    img = cv2.imread(img_1024_path)

    vertex_array = np.float32([[0, 0], [383, 0], [383, 383]])
    num = 0
    for line in ps:
        num = num + 1
        # 随机生成4个坐标,对应384*384图片上的随机点
        pointX_384_0 = random.randint(80, 120)
        pointY_384_0 = random.randint(80, 120)

        pointX_384_1 = random.randint(263, 303)
        pointY_384_1 = random.randint(80, 120)

        pointX_384_2 = random.randint(263, 303)
        pointY_384_2 = random.randint(263, 303)

        annt_array = np.float32([[line[0], line[1]], [line[2], line[3]], [
            line[4], line[5]], [line[6], line[7]]])
        annt_384_array_copy = np.float32([[line[0], line[1]], [line[2], line[3]], [
            line[4], line[5]]])
        flag = 0

        # 把真值点标注在单个停车位对应的1024*1024图片上
        # colorize(annt_array, img, img_1024_singleSlot_dir,
        #          i, num, (0, 255, 0), flag)

        # 把原图的四个坐标点记录下来
        annt_1024_singleSlot_path = os.path.join(
            annt_1024_singleSlot_dir, i.strip('.jpg')+'_'+str(num)+'_OA.txt')
        with open(annt_1024_singleSlot_path, "w") as f:
            for w in range(4):
                f.write(str(line[2*w]))
                f.write(' ')
                f.write(str(line[2*w+1]))
                f.write('\n')

        point_384_array = np.float32(
            [[pointX_384_0, pointY_384_0], [pointX_384_1, pointY_384_1], [pointX_384_2, pointY_384_2]])

        # 求仿射变换逆变换矩阵
        trans_inv = cv2.getAffineTransform(
            point_384_array, annt_384_array_copy)

        # 求逆投影变换后的点坐标
        dst = []
        mat = np.array(
            [[0, 0, 1], [383, 0, 1], [383, 383, 1], [0, 383, 1]], dtype=np.float32)
        mat = mat.transpose()
        dst = np.dot(trans_inv, mat)
        dst = dst.transpose()

        # 画线
        paint_line(img, dst, img_1024_with_rectangle_dir, num)

        # 将停车位投影变换后得到在384*384分辨率下的停车位图像
        perspective2_ = np.float32([[dst[0][0], dst[0][1]], [dst[1][0], dst[1][1]], [
            dst[2][0], dst[2][1]]])

        trans = cv2.getAffineTransform(perspective2_, vertex_array)
        dst2 = cv2.warpAffine(img, trans, (384, 384))

        # 保存原图四个内角点在384*384图片上的坐标
        mat2 = np.array([[line[0], line[1], 1], [line[2], line[3], 1], [
                        line[4], line[5], 1], [line[6], line[7], 1]], dtype=np.float32)

        mat2 = mat2.transpose()
        point = np.dot(trans, mat2)
        point = point.transpose()

        img_384_path = os.path.join(
            img_384_dir, i.strip('.jpg')+'_'+str(num)+'.jpg')
        cv2.imwrite(img_384_path, dst2)
        flag = 1
        colorize(point, dst2, img_384_with_label_dir,
                 i, num, (0, 255, 0), flag)

        # 把四个坐标点记录下来
        annt_384_path = os.path.join(
            annt_384_dir, i.strip('.jpg')+'_'+str(num)+'_OA.txt')
        with open(annt_384_path, "w") as f:
            for j in range(4):
                f.write(str(point[j][0]))
                f.write(' ')
                f.write(str(point[j][1]))
                f.write('\n')

        # 把转换矩阵记录下来
        trans_inv_path = os.path.join(
            trans_inv_dir, i.strip('.jpg')+'_'+str(num)+'.txt')
        with open(trans_inv_path, "w") as ff:
            for j in range(2):
                for k in range(3):
                    ff.write(str(trans_inv[j][k]))
                    ff.write(" ")


if __name__ == "__main__":

    # 训练数据集的图片和标注
    img_1024_dir = './img_1024'
    annt_1024_dir = './annt_1024'

    # 数据集处理后的图片和坐标
    # 在原图上画出矩形框的图片
    img_1024_with_rectangle_dir = './img_1024_with_rectangle'
    # 单个停车位的384*384图片
    img_384_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/img_384'
    # 单个停车位的带真值标注的384*384图片
    img_384_with_label_dir = "./img_384_with_label"
    # 384*384图片的真值点坐标
    annt_384_dir = './annt_384'
    cnt = 0

    train_list_file = open("./train_list.txt", "w")
    # 记录转换矩阵
    trans_inv_dir = "./trans_inv"
    # 单个停车位对应的原图
    img_1024_singleSlot_dir = "./img_1024_singleSlot"
    # 1024*1024图片的单个停车位的真值点坐标
    annt_1024_singleSlot_dir = "./annt_1024_singleSlot"

    for i in os.listdir(img_1024_dir):
        annt_1024_path = os.path.join(annt_1024_dir, i.strip('.jpg')+'_OA.txt')
        img_1024_path = os.path.join(img_1024_dir, i)
        ps, _ = read_pslot(annt_1024_path)
        if len(ps) != 0:
            Crop_pic(ps, img_1024_path, img_1024_with_rectangle_dir,
                     img_384_dir, annt_384_dir, i, trans_inv_dir, img_1024_singleSlot_dir, img_384_with_label_dir, annt_1024_singleSlot_dir)

   # 保存训练数据的文件名
    for i in os.listdir(img_384_dir):
        img_384_path = os.path.join(img_384_dir, i)
        train_list_file.write(img_384_path)
        train_list_file.write('\n')

    train_list_file.close()
