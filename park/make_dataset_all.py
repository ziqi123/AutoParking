# import os
# import cv2
# import numpy as np
# import copy
# # 标点


# def colorize(img, point, save_path, point_color):
#     point_size = 1
#     thickness = 4  # 可以为 0、4、
#     for i in range(4):
#         cv2.circle(img, (int(float(point[2*i])), int(float(point[2*i+1]))),
#                    point_size, point_color, thickness)
#     cv2.imwrite(save_path, img)

# # 保存单个点的坐标


# def save_point(point, point_crop_path):
#     with open(point_crop_path, "w") as f:
#         for i in range(4):
#             f.write(str(point[2*i]))
#             f.write(' ')
#             f.write(str(point[2*i+1]))
#             f.write('\n')

# # 切割图片的一个角


# def Crop(img_path, point_dir, save_path, save_path2, point_crop_path):
#     img = cv2.imread(img_path)
#     # 将四个角拼成一张图片
#     cropped1 = img[48:144, 48:144]
#     cropped2 = img[48:144, 240:336]
#     cropped3 = img[240:336, 48:144]
#     cropped4 = img[240:336, 240:336]
#     # pj1 = np.zeros((96, 96+96, 3))
#     # # 上面左右拼接
#     # pj1[:, :96, :] = cropped1.copy()
#     # pj1[:, 96:, :] = cropped2.copy()
#     # pj1 = np.array(pj1, dtype=np.uint8)
#     # # 下面左右拼接
#     # pj2 = np.zeros((96, 96+96, 3))
#     # pj2[:, :96, :] = cropped3.copy()
#     # pj2[:, 96:, :] = cropped4.copy()
#     # pj2 = np.array(pj2, dtype=np.uint8)
#     # # 上面与下面拼接
#     # pj = np.zeros((192, 192, 3))
#     # pj[:96, :, :] = pj1.copy()
#     # pj[96:, :, :] = pj2.copy()
#     # pj = np.array(pj, dtype=np.uint8)
#     img_out1 = np.concatenate((cropped1, cropped2), axis=1)
#     img_out2 = np.concatenate((cropped3, cropped4), axis=1)
#     pj = np.concatenate((img_out1, img_out2))

#     # cv2.imwrite(save_path, pj)

#     point = []
#     with open(point_dir, "r") as f:
#         annt = f.readlines()
#     cnt = 0
#     for line in annt:
#         cnt = cnt+1
#         line_annt = line.strip('\n').split(' ')
#         point.append(line_annt[0])
#         point.append(line_annt[1])
#     # 转换四个角的坐标
#     point = list(map(float, point))
#     point2 = copy.deepcopy(point)
#     point2[0] = point[0]-48
#     point2[1] = point[1]-48
#     point2[2] = point[2]-240+96
#     point2[3] = point[3]-48
#     if(point[4] > 0 and point[4] < 192):
#         point2[4] = point[4]-48
#         point2[5] = point[5]-240+96
#         point2[6] = point[6]-240+96
#         point2[7] = point[7]-240+96
#     else:
#         point2[4] = point[6]-48
#         point2[5] = point[7]-240+96
#         point2[6] = point[4]-240+96
#         point2[7] = point[5]-240+96

#     if(point2[0] < 0 or point2[0] > 96):
#         return
#     if(point2[1] < 0 or point2[1] > 96):
#         return
#     if(point2[2] < 96 or point2[2] > 192):
#         return
#     if(point2[3] < 0 or point2[3] > 96):
#         return
#     if(point2[4] < 0 or point2[4] > 96):
#         return
#     if(point2[5] < 96 or point2[5] > 192):
#         return
#     if(point2[6] < 96 or point2[6] > 192):
#         return
#     if(point2[7] < 96 or point2[7] > 192):
#         return

#     # colorize(pj, point2, save_path, point_color=(0, 255, 0))
#     # colorize(pj, point2, save_path2, point_color=(0, 255, 0))
#     # save_point(point2, point_crop_path)
#     # save_point(point2, point_crop_path)
#     cv2.imwrite(save_path, pj)
#     # colorize(pj, point2, save_path, point_color=(0, 255, 0), 0)
#     colorize(pj, point2, save_path2, point_color=(0, 255, 0))
#     save_point(point2, point_crop_path)
#     # point[0] = point2[0]+48
#     # point[1] = point2[1]+48
#     # point[2] = point2[2]+240-96
#     # point[3] = point2[3]+48

#     # point[4] = point2[4]+48
#     # point[5] = point2[5]+240-96
#     # point[6] = point2[6]+240-96
#     # point[7] = point2[7]+240-96
#     # save_point(point, point_dir)


# if __name__ == "__main__":
#     data_dir = './perspective_img'
#     point_dir = './point'
#     save_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/match_img'
#     save_dir2 = './match_img2'
#     point_crop_dir = './match_img_point'
#     path_record = './point_path_record.txt'
#     f = open(path_record, 'w')
#     for i in os.listdir(data_dir):
#         img_path = os.path.join(data_dir, i)
#         point_path = os.path.join(point_dir, i.strip('.jpg')+'_OA.txt')
#         save_path = os.path.join(save_dir, i)

#         point_crop_path = os.path.join(
#             point_crop_dir, i.strip('.jpg')+'_OA.txt')
#         save_path2 = os.path.join(save_dir2, i)
#         Crop(img_path, point_path, save_path, save_path2, point_crop_path)

#     # for j in os.listdir(save_dir):
#     #     save_path = os.path.join(save_dir, j)
#     #     f.write(save_path)
#     #     f.write('\n')

#     f.close()

import os
import cv2
import numpy as np
import copy
# 标点


def colorize(img, point, save_path, point_color):
    point_size = 1
    thickness = 4  # 可以为 0、4、
    for i in range(4):
        cv2.circle(img, (int(float(point[2*i])), int(float(point[2*i+1]))),
                   point_size, point_color, thickness)
    cv2.imwrite(save_path, img)

# 保存单个点的坐标


def save_point(point, point_crop_path):
    with open(point_crop_path, "w") as f:
        for i in range(4):
            f.write(str(point[2*i]))
            f.write(' ')
            f.write(str(point[2*i+1]))
            f.write('\n')

# 切割图片的一个角


def Crop(img_path, point_path, save_path, save_path2, point_crop_path):
    img = cv2.imread(img_path)
    # 将四个角拼成一张图片
    cropped1 = img[48:144, 48:144]
    cropped2 = img[48:144, 240:336]
    cropped3 = img[240:336, 48:144]
    cropped4 = img[240:336, 240:336]
    # pj1 = np.zeros((96, 96+96, 3))
    # # 上面左右拼接
    # pj1[:, :96, :] = cropped1.copy()
    # pj1[:, 96:, :] = cropped2.copy()
    # pj1 = np.array(pj1, dtype=np.uint8)
    # # 下面左右拼接
    # pj2 = np.zeros((96, 96+96, 3))
    # pj2[:, :96, :] = cropped3.copy()
    # pj2[:, 96:, :] = cropped4.copy()
    # pj2 = np.array(pj2, dtype=np.uint8)
    # # 上面与下面拼接
    # pj = np.zeros((192, 192, 3))
    # pj[:96, :, :] = pj1.copy()
    # pj[96:, :, :] = pj2.copy()
    # pj = np.array(pj, dtype=np.uint8)
    img_out1 = np.concatenate((cropped1, cropped2), axis=1)
    img_out2 = np.concatenate((cropped3, cropped4), axis=1)
    pj = np.concatenate((img_out1, img_out2))

    point = []
    with open(point_path, "r") as f:
        annt = f.readlines()
    cnt = 0
    for line in annt:
        cnt = cnt+1
        line_annt = line.strip('\n').split(' ')
        point.append(line_annt[0])
        point.append(line_annt[1])
    # print(point)
    # 转换四个角的坐标
    point = list(map(float, point))
    point2 = copy.deepcopy(point)
    point2[0] = point[0]-48
    point2[1] = point[1]-48
    point2[2] = point[2]-240+96
    point2[3] = point[3]-48
    if(point[4] > 0 and point[4] < 192):
        point2[4] = point[4]-48
        point2[5] = point[5]-240+96
        point2[6] = point[6]-240+96
        point2[7] = point[7]-240+96
    else:
        point2[4] = point[6]-48
        point2[5] = point[7]-240+96
        point2[6] = point[4]-240+96
        point2[7] = point[5]-240+96

    if(point2[0] < 0 or point2[0] > 96):
        return
    if(point2[1] < 0 or point2[1] > 96):
        return
    if(point2[2] < 96 or point2[2] > 192):
        return
    if(point2[3] < 0 or point2[3] > 96):
        return
    if(point2[4] < 0 or point2[4] > 96):
        return
    if(point2[5] < 96 or point2[5] > 192):
        return
    if(point2[6] < 96 or point2[6] > 192):
        return
    if(point2[7] < 96 or point2[7] > 192):
        return

    cv2.imwrite(save_path, pj)
    # colorize(pj, point2, save_path, point_color=(0, 255, 0), 0)
    colorize(pj, point2, save_path2, point_color=(0, 255, 0))
    save_point(point2, point_crop_path)
    # point[0] = point2[0]+48
    # point[1] = point2[1]+48
    # point[2] = point2[2]+240-96
    # point[3] = point2[3]+48

    # point[4] = point2[4]+48
    # point[5] = point2[5]+240-96
    # point[6] = point2[6]+240-96
    # point[7] = point2[7]+240-96
    # save_point(point, point_path)


def img_back(img_path, point_crop_path, save_match_img3_path):
    point = np.loadtxt(point_crop_path)
    point2 = np.array(point, copy=True)
    point2[0][0] = point[0][0]+48
    point2[0][1] = point[0][1]+48
    point2[1][0] = point[1][0]+240-96
    point2[1][1] = point[1][1]+48
    point2[2][0] = point[2][0]+48
    point2[2][1] = point[2][1]+240-96
    point2[3][0] = point[3][0]+240-96
    point2[3][1] = point[3][1]+240-96
    point_size = 1
    thickness = 4  # 可以为 0、4、
    point_color = (0, 255, 255)
    img = cv2.imread(img_path)
    for i in range(4):
        cv2.circle(img, (int(float(point2[i][0])), int(float(point2[i][1]))),
                   point_size, point_color, thickness)
    cv2.imwrite(save_match_img3_path, img)


if __name__ == "__main__":
    data_dir = './perspective_img'
    point_dir = './point'
    img_dir = "./perspective2_img"

    save_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/match_img'
    save_dir2 = './match_img2'
    point_crop_dir = './match_img_point'
    # path_record = './point_path_record.txt'
    path_record = './train_list.txt'
    save_match_img3_dir = "./match_img3"
    f = open(path_record, 'w')
    for i in os.listdir(data_dir):
        img_path = os.path.join(data_dir, i)
        point_path = os.path.join(point_dir, i.strip('.jpg')+'_OA.txt')
        save_path = os.path.join(save_dir, i)
        save_path2 = os.path.join(save_dir2, i)

        point_crop_path = os.path.join(
            point_crop_dir, i.strip('.jpg')+'_OA.txt')
        Crop(img_path, point_path, save_path, save_path2, point_crop_path)
    for j in os.listdir(save_dir):
        save_path = os.path.join(save_dir, j)
        f.write(save_path)
        f.write('\n')
    f.close()
    for k in os.listdir(save_dir):
        img_path = os.path.join(img_dir, k)
        point_crop_path = os.path.join(
            point_crop_dir, k.strip('.jpg')+'_OA.txt')
        save_match_img3_path = os.path.join(save_match_img3_dir, k)
        img_back(img_path, point_crop_path, save_match_img3_path)
