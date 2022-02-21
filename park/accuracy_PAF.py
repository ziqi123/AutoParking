import numpy as np
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# 计算四个点的预测点与真值点之间的误差


def get_acc(y, y_hat, dis, img_path):
    total = 0
    for i in range(2):
        total += ((int(y[i][0])-int(y_hat[i][0]))**2 +
                  (int(y[i][1])-int(y_hat[i][1]))**2)**0.5
    total /= 2

    f = os.path.basename(img_path)

    save_path = os.path.join(
        error_dir, f)
    read_path = os.path.join(
        mark_img_1024_dir, f)
    img = cv2.imread(read_path)

    if(total >= 10):
        cv2.imwrite(save_path, img)

    if total < dis:
        return 1
    else:
        return 0

# 保存坐标点函数


def save_point(point, point_path):
    with open(point_path, "w") as f:
        for k in range(4):
            f.write(str(point[k][0]))
            f.write(' ')
            f.write(str(point[k][1]))
            f.write('\n')

# 转换坐标，标点


def colorize_pic(src_img_1024_path, output_path,
                 trans_inv_path, mark_img_1024_path, point_384_gt_path, point_pred_1024_path, point_gt_1024_path):

    # 读原图
    img_pred = cv2.imread(src_img_1024_path)
    point_pred = []
    trans_inv = []
    # 测试图片的预测点在384*384图片的坐标
    point_pred = np.loadtxt(output_path)

    # 384*384图片到原图的转换矩阵
    trans_inv = np.loadtxt(trans_inv_path)

    trans_inv = trans_inv.reshape(2, 3)
    trans_inv = np.mat(trans_inv)

    # 把384*384中的点投影到原图上
    # dst = cv2.perspectiveTransform(point_pred, trans_inv)
    column = np.array([1, 1, 1, 1])
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

    # annt = np.loadtxt(annt_path2)
    # print(annt[0])
    # 把点画在图上
    # 原图中点的坐标
    point = np.loadtxt(point_384_gt_path)
    point = np.column_stack((point, column))
    point = point.T
    p = np.dot(trans_inv, point)
    p = p.T
    p = np.asarray(p)

    # for i in range(2):
    #     cv2.circle(img_pred, (int(p[i][0]), int(p[i][1])),
    #                point_size, point_color2, thickness)

    for i in range(4):
        cv2.circle(img_pred, (int(p[i][0]), int(p[i][1])),
                   point_size, point_color2, thickness)

    cv2.imwrite(mark_img_1024_path, img_pred)

    # dst = dst.reshape(2, 2)
    dst = np.asarray(dst)
    # print("dst", dst)
    # print(type(dst))

    for i in range(4):
        cv2.circle(img_pred, (int(dst[i][0]), int(dst[i][1])),
                   point_size, point_color, thickness)

    cv2.imwrite(mark_img_1024_path, img_pred)

    save_point(dst, point_pred_1024_path)
    save_point(p, point_gt_1024_path)

# 转换坐标，计算单个图片的精度


def pic_accuracy(src_img_1024_path, pix, point_pred_1024_path, point_gt_1024_path):
    # 读原图
    point_pred = []
    # 测试图片的预测点在384*384图片的坐标
    point_pred = np.loadtxt(point_pred_1024_path)

    gt = np.loadtxt(point_gt_1024_path)

    tmp = get_acc(point_pred, gt, pix, src_img_1024_path)

    return tmp

# 求测试集的预测点到原图中的精度


def get_accuracy(pix):
    accuracy = 0
    for i in os.listdir(mark_test_img_256_dir):
        src_img_1024_path = os.path.join(src_img_1024_dir, i)
        point_pred_1024_path = os.path.join(
            point_pred_1024_dir, i.strip('.jpg')+'.txt')
        point_gt_1024_path = os.path.join(
            point_gt_1024_dir, i.strip('.jpg')+'.txt')
        tmp = pic_accuracy(src_img_1024_path, pix,
                           point_pred_1024_path, point_gt_1024_path)
        accuracy += tmp
    return accuracy


# 计算两点与原点的夹角


# def angle_between(p1, p2):
#     ang1 = np.arctan2(*p1[::-1])
#     ang2 = np.arctan2(*p2[::-1])
#     return np.rad2deg((ang1 - ang2) % (2 * np.pi))

# def azimuthAngle(x1,  y1,  x2,  y2):

#     dx = abs(x2 - x1)
#     dy = abs(y2 - y1)
#     angle = math.atan2(dy , dx)

#     return (angle * 180 / math.pi)

def azimuthAngle(x1,  y1,  x2,  y2):
    angle = math.atan2((y2-y1), (x2-x1))
    return (angle * 180 / math.pi)

# 计算角偏差，求精度


def get_angle_acc(point_pred_1024_path, point_1024_gt_path, pix):
    point_pred = np.loadtxt(point_pred_1024_path)
    point_gt = np.loadtxt(point_1024_gt_path)
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
    vector_end1 = vec[0]+vector_norm1*50
    vector_end2 = vec[1]+vector_norm2*50

    vector1_gt = vec_gt[2]-vec_gt[0]
    vector2_gt = vec_gt[3]-vec_gt[1]
    vector_norm1_gt = vector1_gt/np.linalg.norm(vector1_gt)
    vector_norm2_gt = vector2_gt/np.linalg.norm(vector2_gt)
    vector_end1_gt = vec_gt[0]+vector_norm1_gt*20
    vector_end2_gt = vec_gt[1]+vector_norm2_gt*20

    angle1 = azimuthAngle(vec[0][0], vec[0][1],
                          vector_end1[0],  vector_end1[1])

    # angle_between(vec[0], vector_end1)

    # angle1_gt = angle_between(vec_gt[0], vector_end1_gt)
    angle1_gt = azimuthAngle(vec_gt[0][0], vec_gt[0][1],
                             vector_end1_gt[0],  vector_end1_gt[1])

    # angle2 = angle_between(vec[1], vector_end2)
    # angle2_gt = angle_between(vec_gt[1], vector_end2_gt)
    angle2 = azimuthAngle(vec[1][0], vec[1][1],
                          vector_end2[0],  vector_end2[1])
    angle2_gt = azimuthAngle(vec_gt[1][0], vec_gt[1][1],
                             vector_end2_gt[0],  vector_end2_gt[1])
    # print(angle1)
    # print(angle1_gt)

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
    vector1 = vec[3]-vec[0]
    vector2 = vec[2]-vec[1]
    vector_norm1 = vector1/np.linalg.norm(vector1)
    vector_norm2 = vector2/np.linalg.norm(vector2)
    vector_end1 = vec[0]+vector_norm1*200
    vector_end2 = vec[1]+vector_norm2*200
    # print(vector_end1)
    cvArrow(img, vec[0], vector_end1, img_path)
    cvArrow(img, vec[1], vector_end2, img_path)


if __name__ == "__main__":
    trans_inv_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/trans_inv2"
    src_img_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/img_1024_with_rectangle"
    annt_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/annt_1024_singleSlot"
    mark_img_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/fina"

    # point_384_gt_dir = "/media/home_bak/ziqi/park/hourglass/point_384_gt"
    # point_pred_384_dir = "/media/home_bak/ziqi/park/hourglass/point_pred_384"
    # mark_test_img_384_dir = "/media/home_bak/ziqi/park/hourglass/mark_test_img_384"
    point_256_gt_dir = "/media/home_bak/ziqi/park/stackedHourglass_256/point_256_gt"
    point_pred_256_dir = "/media/home_bak/ziqi/park/stackedHourglass_256/point_pred_256"
    mark_test_img_256_dir = "/media/home_bak/ziqi/park/stackedHourglass_256/mark_test_img_256"

    error_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/error_img"
    point_pred_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/point_pred_1024"
    point_gt_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/point_gt_1024"
    for i in os.listdir(mark_test_img_256_dir):
        output_path = os.path.join(
            point_pred_256_dir, i.strip('.jpg')+'.txt')
        src_img_1024_path = os.path.join(src_img_1024_dir, i)
        trans_inv_path = os.path.join(trans_inv_dir, i.strip('.jpg')+'.txt')
        mark_img_1024_path = os.path.join(mark_img_1024_dir, i)
        annt_1024_path = os.path.join(annt_1024_dir, i.strip('.jpg')+'_OA.txt')
        point_256_gt_path = os.path.join(
            point_256_gt_dir, i.strip('.jpg')+'.txt')
        point_pred_1024_path = os.path.join(
            point_pred_1024_dir, i.strip('.jpg')+'.txt')
        point_gt_1024_path = os.path.join(
            point_gt_1024_dir, i.strip('.jpg')+'.txt')
        colorize_pic(src_img_1024_path, output_path,
                     trans_inv_path, mark_img_1024_path, point_256_gt_path, point_pred_1024_path, point_gt_1024_path)
        draw_angle(mark_img_1024_path, point_pred_1024_path)

    acc = []
    accuracy_angle = [0 for j in range(10)]
    for k in range(15):
        x1 = get_accuracy(k)
        x1 = 100 * x1 / 6842
        acc.append(x1)

    # 计算1024*1024图片上角度的精度
    for pix in range(10):
        for c in os.listdir(mark_test_img_256_dir):
            point_pred_1024_path = os.path.join(
                point_pred_1024_dir, c.strip('.jpg')+'.txt')
            point_gt_1024_path = os.path.join(
                point_gt_1024_dir, c.strip('.jpg')+'.txt')
            tmp = get_angle_acc(point_pred_1024_path, point_gt_1024_path, pix)
            accuracy_angle[pix] += tmp

    for y in range(10):
        accuracy_angle[y] = 100 * accuracy_angle[y] / 7233
        accuracy_angle[y] = round(accuracy_angle[y], 3)

    print("accuracy_angle:", accuracy_angle)

    # x1 = round(x1, 3)
    # print(acc)

    # # 设置画布大小
    # plt.figure(figsize=(30, 15))

    # # 标题
    # plt.title("accruracy distribution")

    # # 数据
    # plt.bar(range(len(acc)), acc)

    # # 横坐标描述
    # plt.xlabel('pixel')

    # # 纵坐标描述
    # plt.ylabel('accuracy')

    # plt.savefig(
    #     "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/accuracy.png")
