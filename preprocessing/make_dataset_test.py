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


def read_pslot(annt_file):
    # print(annt_file)
    with open(annt_file, "r") as f:
        annt = f.readlines()
    # print("annt", annt)
    l = []
    l_ign = []
    for line in annt:
        line_annt = line.strip('\n').split(' ')
        # print(line_annt)

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


def colorize(points_list, img, save_path, item, line, point_color):
    save_path = os.path.join(save_path, str(
        item.strip('.jpg'))+"_"+str(line)+".jpg")
    img2 = img.copy()
    # print(save_path)
    # points_list = 384 * np.abs(np.array(outputs[0], dtype=np.float))
    point_size = 1
    thickness = 4  # 可以为 0、4、8
    for i in range(4):
        cv2.circle(img2, (int(points_list[i][0]), int(points_list[i][1])),
                   point_size, point_color, thickness)

    # print(save_path)
    cv2.imwrite(save_path, img2)

 # 画线


def paint_line(img, dst, cropimg_path, num):
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
        cropimg_path, i.strip('.jpg')+'_'+str(num)+'.jpg')
    cv2.imwrite(cropimg_path1, img2)


def Crop_pic(ps, img_path, cropimg_path, perspective_path, txt_file, i, trans_path, save_path1, save_path2, annt_path_new, img_all_path):
    # single pic
    img = cv2.imread(img_path)

    perspective3 = np.float32([[0, 0], [383, 0], [383, 383], [0, 383]])
    perspective3_ = np.float32([[0, 0], [383, 0], [383, 383]])
    num = 0
    for line in ps:
        num = num + 1
        # 随机生成4个坐标
        arr0 = random.randint(80, 120)
        arr1 = random.randint(80, 120)

        arr2 = random.randint(263, 303)
        arr3 = random.randint(80, 120)

        arr4 = random.randint(263, 303)
        arr5 = random.randint(263, 303)

        arr6 = random.randint(80, 120)
        arr7 = random.randint(263, 303)

        # perspective0 = np.float32([[line[0], line[1]], [line[2], line[3]], [
        #     line[4], line[5]], [line[6], line[7]]])

        # colorize(perspective0, img, save_path1, i, num, (0, 255, 0))

        # perspective1 = np.float32(
        #     [[arr0, arr1], [arr2, arr3], [arr4, arr5], [arr6, arr7]])

        # # 求逆变换矩阵，384*384图到原图的变换矩阵，getPerspectiveTransform（源图像，目标图像）
        # trans_inv = cv2.getPerspectiveTransform(perspective1, perspective0)

        # # 求逆投影变换后的点坐标，perspectiveTransform函数把384*384中的点投影到原图上
        # dst = []
        # mat = np.array(
        #     [[[0, 0], [383, 0], [383, 383], [0, 383]]], dtype=np.float32)
        # dst = cv2.perspectiveTransform(mat, trans_inv)

        perspective0 = np.float32([[line[0], line[1]], [line[2], line[3]], [
            line[4], line[5]], [line[6], line[7]]])
        perspective0_ = np.float32([[line[0], line[1]], [line[2], line[3]], [
            line[4], line[5]]])

        colorize(perspective0, img, save_path1, i, num, (0, 255, 0))
        perspective1_ = np.float32(
            [[arr0, arr1], [arr2, arr3], [arr4, arr5]])

        # 求逆变换矩阵
        # trans_inv = cv2.getPerspectiveTransform(perspective1, perspective0)
        trans_inv = cv2.getAffineTransform(perspective1_, perspective0_)

        # 求逆投影变换后的点坐标
        dst = []
        # mat = np.array(
        #     [[[0, 0], [383, 0], [383, 383], [0, 383]]], dtype=np.float32)
        mat = np.array(
            [[0, 0, 1], [383, 0, 1], [383, 383, 1], [0, 383, 1]], dtype=np.float32)
        mat = mat.transpose()
        # dst = cv2.perspectiveTransform(mat, trans_inv)
        dst = np.dot(trans_inv, mat)
        dst = dst.transpose()

        # 画线
        paint_line(img, dst, cropimg_path, num)

        # 将停车位投影变换后得到在384*384分辨率下的停车位图像

        # perspective2 = np.float32([[dst[0][0][0], dst[0][0][1]], [dst[0][1][0], dst[0][1][1]], [
        #                           dst[0][2][0], dst[0][2][1]], [dst[0][3][0], dst[0][3][1]]])
        perspective2_ = np.float32([[dst[0][0], dst[0][1]], [dst[1][0], dst[1][1]], [
            dst[2][0], dst[2][1]]])

        trans = cv2.getAffineTransform(perspective2_, perspective3_)
        dst2 = cv2.warpAffine(img, trans, (384, 384))

        # 保存原图四个内角点在384*384图片上的坐标
        # mat2 = np.array([[[line[0], line[1]], [line[2], line[3]], [
        #                 line[4], line[5]], [line[6], line[7]]]], dtype=np.float32)
        mat2 = np.array([[line[0], line[1], 1], [line[2], line[3], 1], [
                        line[4], line[5], 1], [line[6], line[7], 1]], dtype=np.float32)

        mat2 = mat2.transpose()
        point = np.dot(trans, mat2)
        point = point.transpose()

        # perspective2 = np.float32([[dst[0][0][0], dst[0][0][1]], [dst[0][1][0], dst[0][1][1]], [
        #                           dst[0][2][0], dst[0][2][1]], [dst[0][3][0], dst[0][3][1]]])

        # trans = cv2.getPerspectiveTransform(perspective2, perspective3)
        # dst2 = cv2.warpPerspective(img, trans, (384, 384))

        # # 保存原图四个内角点在384*384图片上的坐标
        # mat2 = np.array([[[line[0], line[1]], [line[2], line[3]], [
        #                 line[4], line[5]], [line[6], line[7]]]], dtype=np.float32)
        # point = cv2.perspectiveTransform(mat2, trans)

        perspective_path1 = os.path.join(
            perspective_path, i.strip('.jpg')+'_'+str(num)+'.jpg')
        # print(perspective_path)
        cv2.imwrite(perspective_path1, dst2)
        colorize(point, dst2, save_path2, i, num, (0, 255, 0))

        # 把原图中一个停车位对应成一张原图
        pic_all_path = os.path.join(
            img_all_path, i.strip('.jpg')+'_'+str(num)+'.jpg')
        cv2.imwrite(pic_all_path, img)

        # 把四个坐标点记录下来
        txt_file1 = os.path.join(
            txt_file, i.strip('.jpg')+'_'+str(num)+'_OA.txt')
        with open(txt_file1, "w") as f:
            for j in range(4):
                f.write(str(point[j][0]))
                f.write(' ')
                f.write(str(point[j][1]))
                f.write('\n')

        annt_path_new1 = os.path.join(
            annt_path_new, i.strip('.jpg')+'_'+str(num)+'_OA.txt')
        with open(annt_path_new1, "w") as f2:
            f2.write(str(line))
            f2.write('\n')

        # 把转换矩阵记录下来
        trans_path1 = os.path.join(
            trans_path, i.strip('.jpg')+'_'+str(num)+'.txt')
        with open(trans_path1, "w") as ff:
            for j in range(2):
                for k in range(3):
                    ff.write(str(trans_inv[j][k]))
                    ff.write(" ")

# 计算四个点的预测点与真值点之间的误差


def get_acc(y, y_hat, dis):
    # print(y)
    # print(y_hat)
    total = 0
    for i in range(2):
        total += ((int(y[i][0])-int(y_hat[2*i]))**2 +
                  (int(y[i][1])-int(y_hat[2*i+1]))**2)**0.5
    total /= 2

    if total < dis:
        return 1
    else:
        return 0


def output_pic(img_path, output_path, trans_path, fina_path, ps2, pix):
    # 读原图
    img_pred = cv2.imread(img_path)
    point_pred = []
    trans_inv = []
    # 测试图片的预测点在384*384图片的坐标
    point_pred = np.loadtxt(output_path)
    # point_pred = 384*np.expand_dims(point_pred, axis=0)
    point_pred = 384*point_pred
    # print("point_pred", point_pred)

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

    # 把点画在图上
    # 原图中点的坐标
    for i in range(2):
        cv2.circle(img_pred, (int(ps2[2*i]), int(ps2[2*i+1])),
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

    point_pred3 = dst

    tmp = get_acc(point_pred3, ps2, pix)

    return tmp

# 求测试集的预测点到原图中的精度


def output(pix):
    accuracy = 0
    for i in os.listdir(img_all_path):
        output_path = os.path.join(
            "/media/home_bak/ziqi/park/CNN/output", i.strip('.jpg')+'.txt')
        img_path = os.path.join(
            "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/pic_all", i)
        trans_inv = os.path.join(
            "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/trans_inv", i.strip('.jpg')+'.txt')
        fina_path = os.path.join(
            "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/fina", i)
        annt_path2 = os.path.join(
            '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/annotation_new', i.strip('.jpg')+'_OA.txt')

        with open(annt_path2, "r") as f:
            l = f.readlines()
            for line in l:
                line_annt = line.strip('[').strip('\n').strip(']').split()
                tmp = output_pic(img_path, output_path,
                                 trans_inv, fina_path, line_annt, pix)
                accuracy += tmp
    return accuracy


if __name__ == "__main__":
    data_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/pic'
    img_all_path = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/pic_all'
    label_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/annotation'
    crop_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/crop_img'
    perspective_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/perspective_img'
    txt_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/point'
    cnt = 0
    f1 = open(
        "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/train_list.txt", "w")
    # f2 = open("./Ps_locate_dataset/val_list.txt", "w")
    test_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/test_img"
    trans_path = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/trans_inv"
    save_path1 = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/src_img"
    save_path2 = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/perspective2_img"
    annt_path_new = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/annotation_new"

    for i in os.listdir(data_dir):
        # print(i)
        annt_file = os.path.join(label_dir, i.strip('.jpg')+'_OA.txt')
        img_path = os.path.join(data_dir, i)

        ps, _ = read_pslot(annt_file)
        # Crop_pic(ps, img_path, crop_dir,
        #          perspective_dir, txt_dir, i, trans_path, save_path1, save_path2, annt_path_new, img_all_path)

    acc = []
    for k in range(31):
        x1 = output(k)
        x1 = 100 * x1 / 7233
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

   # 保存训练数据的文件名
    for i in os.listdir(perspective_dir):
        perspective_path = os.path.join(perspective_dir, i)
        f1.write(perspective_path)
        f1.write('\n')

    f1.close()
