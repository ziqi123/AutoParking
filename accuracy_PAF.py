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

    if total > 5:
        cv2.imwrite(save_path, img)

    if total <= dis:
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


def colorize_pic(src_img_1024_path, point_pred_256_path,
                 trans_inv_path, mark_img_1024_path, point_256_gt_path, point_pred_1024_path, point_gt_1024_path):

    # 读原图
    img_pred = cv2.imread(src_img_1024_path)
    point_pred = []
    trans_inv = []
    # 测试图片的预测点在256*256图片的坐标
    point_pred = np.loadtxt(point_pred_256_path)

    # 256*256图片到原图的转换矩阵
    trans_inv = np.loadtxt(trans_inv_path)

    trans_inv = trans_inv.reshape(2, 3)
    trans_inv = np.mat(trans_inv)

    # 把256*256中的点投影到原图上
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

    # 把点画在图上
    # 原图中点的坐标
    point = np.loadtxt(point_256_gt_path)
    point = np.column_stack((point, column))
    point = point.T
    p = np.dot(trans_inv, point)
    p = p.T
    p = np.asarray(p)

    for i in range(4):
        if i < 3:
            img_pred = cv2.line(img_pred, (int(p[i][0]), int(p[i][1])),
                                (int(p[i+1][0]), int(p[i+1][1])), (0, 0, 255), 3, 8)
        else:
            img_pred = cv2.line(img_pred, (int(p[i][0]), int(p[i][1])),
                                (int(p[0][0]), int(p[0][1])), (0, 0, 255), 3, 8)

    # for i in range(4):
    #     cv2.circle(img_pred, (int(p[i][0]), int(p[i][1])),
    #                point_size, point_color2, thickness)

    # gt
    # annt = np.loadtxt(point_gt_1024_path)
    # for i in range(4):
    #     cv2.circle(img_pred, (int(annt[i][0]), int(annt[i][1])),
    #                point_size, point_color, thickness)

    # dst = dst.reshape(2, 2)
    dst = np.asarray(dst)

    # for i in range(4):
    #     cv2.circle(img_pred, (int(dst[i][0]), int(dst[i][1])),
    #                point_size, point_color, thickness)

    cv2.imwrite(mark_img_1024_path, img_pred)

    save_point(dst, point_pred_1024_path)
    save_point(p, point_gt_1024_path)

# 转换坐标，计算单个图片的精度


def pic_accuracy(src_img_1024_path, pix, point_pred_1024_path, point_gt_1024_path):
    # 读原图
    point_pred = []
    # 测试图片的预测点在256*256图片的坐标
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

    angle1_gt = azimuthAngle(vec_gt[0][0], vec_gt[0][1],
                             vector_end1_gt[0],  vector_end1_gt[1])

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

    # 测试数据集的准确度
    #   ************************************************************************************************************************

    # trans_inv_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/trans_256To1024"
    # src_img_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/img_1024_with_rectangle"
    # # annt_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/annt_1024_singleSlot"
    # mark_img_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/fina"

    # point_256_gt_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/annt_256"
    # point_pred_256_dir = "/media/home_bak/ziqi/park/stackedHourglass_256/point_pred_256"
    # mark_test_img_256_dir = "/media/home_bak/ziqi/park/stackedHourglass_256/mark_test_img_256"

    # error_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/error_img"
    # point_pred_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/point_pred_1024"
    # point_gt_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/annt_1024_singleSlot"

    # 训练数据集的准确度
    #   ************************************************************************************************************************

    # trans_inv_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/trans_256To1024"
    # src_img_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/img_1024_with_rectangle"
    # # annt_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_Training_TestSet_v1.0.7_All_3342/annt_1024_singleSlot"
    # mark_img_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/fina"

    # point_256_gt_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/annt_256"
    # point_pred_256_dir = "/media/home_bak/ziqi/park/stackedHourglass_256/point_pred_256"
    # mark_test_img_256_dir = "/media/home_bak/ziqi/park/stackedHourglass_256/mark_test_img_256"

    # error_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/error_img"
    # point_pred_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/point_pred_1024"
    # point_gt_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/annt_1024_singleSlot"

    # for i in os.listdir(mark_test_img_256_dir):
    #     point_pred_256_path = os.path.join(
    #         point_pred_256_dir, i.strip('.jpg')+'.txt')
    #     src_img_1024_path = os.path.join(src_img_1024_dir, i)
    #     trans_inv_path = os.path.join(trans_inv_dir, i.strip('.jpg')+'.txt')
    #     mark_img_1024_path = os.path.join(mark_img_1024_dir, i)
    #     point_256_gt_path = os.path.join(
    #         point_256_gt_dir, i.strip('.jpg')+'_OA.txt')
    #     point_pred_1024_path = os.path.join(
    #         point_pred_1024_dir, i.strip('.jpg')+'.txt')
    #     point_gt_1024_path = os.path.join(
    #         point_gt_1024_dir, i.strip('.jpg')+'.txt')

    #     colorize_pic(src_img_1024_path, point_pred_256_path,
    #              trans_inv_path, mark_img_1024_path, point_256_gt_path, point_pred_1024_path, point_gt_1024_path)
    # draw_angle(mark_img_1024_path, point_pred_1024_path)
    # acc = []
    # accuracy_angle = [0 for j in range(10)]
    # for k in range(15):
    #     x1 = get_accuracy(k)
    #     x1 = 100 * x1 / 6313
    #     x1 = round(x1, 3)
    #     acc.append(x1)

    # print("acc", acc)

    # 计算1024*1024图片上角度的精度
    # for pix in range(10):
    #     for c in os.listdir(mark_test_img_256_dir):
    #         point_pred_1024_path = os.path.join(
    #             point_pred_1024_dir, c.strip('.jpg')+'.txt')
    #         point_gt_1024_path = os.path.join(
    #             point_gt_1024_dir, c.strip('.jpg')+'.txt')
    #         tmp = get_angle_acc(point_pred_1024_path, point_gt_1024_path, pix)
    #         accuracy_angle[pix] += tmp

    # for y in range(10):
    #     accuracy_angle[y] = 100 * accuracy_angle[y] / 6313
    #     accuracy_angle[y] = round(accuracy_angle[y], 3)

    # print("accuracy_angle:", accuracy_angle)

    # x1 = round(x1, 3)
    # print(acc)

    # 测试视频的准确度
    #   ************************************************************************************************************************

    trans_inv_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_20201219152720-00-00/trans_256To1024"
    src_img_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_20201219152720-00-00/img_1024_with_rectangle"
    mark_img_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_20201219152720-00-00/fina"

    point_256_gt_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_20201219152720-00-00/annt_256"
    point_pred_256_dir = "/media/home_bak/ziqi/park/stackedHourglass_256/PLD_BirdView_20201219152720-00-00/point_pred_256"
    mark_test_img_256_dir = "/media/home_bak/ziqi/park/stackedHourglass_256/PLD_BirdView_20201219152720-00-00/mark_test_img_256"

    point_pred_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_20201219152720-00-00/point_pred_1024"
    point_gt_1024_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_20201219152720-00-00/annt_1024_singleSlot"
    video_img_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_20201219152720-00-00/20201219152720-00-00.MP4"

    # for i in os.listdir(mark_test_img_256_dir):
    #     point_pred_256_path = os.path.join(
    #         point_pred_256_dir, i.strip('.jpg')+'.txt')
    #     src_img_1024_path = os.path.join(src_img_1024_dir, i)
    #     trans_inv_path = os.path.join(trans_inv_dir, i.strip('.jpg')+'.txt')
    #     mark_img_1024_path = os.path.join(mark_img_1024_dir, i)
    #     point_256_gt_path = os.path.join(
    #         point_256_gt_dir, i.strip('.jpg')+'_OA.txt')
    #     point_pred_1024_path = os.path.join(
    #         point_pred_1024_dir, i.strip('.jpg')+'.txt')
    #     point_gt_1024_path = os.path.join(
    #         point_gt_1024_dir, i.strip('.jpg')+'.txt')
    #     colorize_pic(src_img_1024_path, point_pred_256_path,
    #                  trans_inv_path, mark_img_1024_path, point_256_gt_path, point_pred_1024_path, point_gt_1024_path)

    # for j in sorted(os.listdir(point_pred_1024_dir)):
    #     tmp = j
    #     pred_point = tmp.split('_')
    #     pred = pred_point[0]+'_'+pred_point[1]
    #     # print(pred)
    #     video_img_path = os.path.join(video_img_dir, pred+'.jpg')
    #     # print(video_img_path)
    #     img_pred = cv2.imread(video_img_path)
    #     point_pred_1024_path = os.path.join(point_pred_1024_dir, j)
    #     point = np.loadtxt(point_pred_1024_path)
    #     for k in range(4):
    #         if k < 3:
    #             video_img = cv2.line(img_pred, (int(point[k][0]), int(point[k][1])),
    #                                  (int(point[k+1][0]), int(point[k+1][1])), (0, 0, 255), 3, 8)
    #         else:
    #             video_img = cv2.line(img_pred, (int(point[k][0]), int(point[k][1])),
    #                                  (int(point[0][0]), int(point[0][1])), (0, 0, 255), 3, 8)
    #         cv2.imwrite(video_img_path, video_img)

    # output video path
    video_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_20201219152720-00-00/demo'
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    # set saved fps
    fps = 30
    img_size = (1024, 1024)
    # get seq name
    seq_name = os.path.dirname(video_img_dir).split('_')[-1]
    # splice video_dir
    video_dir = os.path.join(video_dir, seq_name + '.avi')
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for img in range(801,len(os.listdir(video_img_dir))):
        img = '{}_{}.jpg'.format(video_img_dir.split('/')[-1], img)
        img1 = cv2.imread(os.path.join(video_img_dir, img))
        videowriter.write(img1)

    videowriter.release()

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
