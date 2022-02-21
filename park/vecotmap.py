import os
import cv2
import numpy as np
import copy
import math

# 得到PAF标签


def get_vectormap(self, keypoints, height, width):
    """
    Parameters
    -----------
    Returns
    --------
    """
    num_joints = 8

    vectormap = np.zeros((num_joints, height, width), dtype=np.float32)
    counter = np.zeros((num_joints, height, width), dtype=np.int16)
    all_keypoints = keypoints
    point_num = all_keypoints.shape[0]

    for k in range(point_num):
        v_start = all_keypoints[k][0]
        v_end = all_keypoints[k][1]
        vectormap = cal_vectormap(vectormap, counter, k, v_start, v_end)

    vectormap = vectormap.transpose((1, 2, 0))
    # normalize the PAF (otherwise longer limb gives stronger absolute strength)
    nonzero_vector = np.nonzero(counter)

    for i, y, x in zip(nonzero_vector[0], nonzero_vector[1], nonzero_vector[2]):

        if counter[i][y][x] <= 0:
            continue
        vectormap[y][x][i * 2 + 0] /= counter[i][y][x]
        vectormap[y][x][i * 2 + 1] /= counter[i][y][x]

    mapholder = []
    for i in range(0, 2):
        a = cv2.resize(
            np.array(vectormap[:, :, i]), (height, width), interpolation=cv2.INTER_AREA)
        mapholder.append(a)
    mapholder = np.array(mapholder)
    vectormap = mapholder.transpose(1, 2, 0)

    return vectormap.astype(np.float16)

# 得到单个向量的map（2个）表示


def cal_vectormap(vectormap, countmap, i, v_start, v_end):
    """
    Parameters
    -----------
    Returns
    --------
    """
    _, height, width = vectormap.shape[:3]

    threshold = 1
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]
    length = (vector_x**2 + vector_y**2)**0.5
    if length == 0:
        return vectormap

    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - v_start[0]
            bec_y = y - v_start[1]
            dist = abs(bec_x * norm_y - bec_y * norm_x)

            # orthogonal distance is < then threshold
            if dist > threshold:
                continue
            countmap[i][y][x] += 1
            vectormap[i * 2 + 0][y][x] = norm_x
            vectormap[i * 2 + 1][y][x] = norm_y

    return vectormap


# def get_vectormap(self, label, sign=True):
#     """
#     功能： 生成OpenPose的向量图(PAF标签), 因为每两个关键点的连线有两个方向(x-axis, y-axis), vectormap是heatmap的2倍
#     :param joint: 已标注的真实2D关键点坐标
#     :param sign:
#     :return:
#     """
#     vectormap = np.zeros(
#         (len(self.shuffle_ref)*2, self.output_size[0], self.output_size[1]), dtype=np.float32)
#     countmap = np.zeros(
#         (len(self.shuffle_ref), self.output_size[0], self.output_size[1]), dtype=np.int16)
#     for plane_idx, (j_idx1, j_idx2) in enumerate(self.shuffle_ref):
#         center_from = label[j_idx1]
#         center_to = label[j_idx2]

#         if center_from[0] < -100 or center_from[1] < -100 or center_to[0] < -100 or center_to[1] < -100:
#             continue
#         self.put_vectormap(vectormap, countmap, plane_idx,
#                            center_from, center_to)

#     vectormap = vectormap.transpose((1, 2, 0))
#     nonzeros = np.nonzero(countmap)
#     for p, y, x in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
#         if countmap[p][y][x] <= 0:
#             continue
#         vectormap[y][x][p * 2 + 0] /= countmap[p][y][x]
#         vectormap[y][x][p * 2 + 1] /= countmap[p][y][x]
#     return vectormap.astype(np.float16)


# def put_vectormap(self, vectormap, countmap, plane_idx, center_from, center_to, threshold=1):
#     """
#     功能： 计算每个两个关节点向量的x,y方向上的map
#     :param vectormap:
#     :param countmap:
#     :param plane_idx: 关节点索引
#     :param center_from: 向量终点
#     :param center_to: 向量起点
#     :param threshold: 向量叉乘的范围限定阈值
#     :return:
#     """
#     _, height, width = vectormap.shape[:3]

#     vec_x = center_to[0] - center_from[0]
#     vec_y = center_to[1] - center_from[1]
#     min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
#     min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))
#     max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
#     max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

#     norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
#     if norm == 0:
#         return
#     vec_x /= norm
#     vec_y /= norm
#     for y in range(min_y, max_y):
#         for x in range(min_x, max_x):
#             bec_x = x - center_from[0]
#             bec_y = y - center_from[1]
#             dist = abs(bec_x * vec_y - bec_y * vec_x)

#             if dist > threshold:
#                 continue
#             countmap[plane_idx][y][x] += 1
#             vectormap[plane_idx * 2 + 0][y][x] = vec_x
#             vectormap[plane_idx * 2 + 1][y][x] = vec_y
