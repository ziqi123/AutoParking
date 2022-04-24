import os
import math

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from .inference import tbox
RED = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (115, 181, 34)
BLUE = (255, 0, 0)
CYAN = (255, 128, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
head_or_tail = [RED, CYAN]

template = [
        [0, 0, 0, 0, 1], # side
        [1, 1, 0, 0, 1], # left / right side

        [0, 0, 1, 0, 0], # tail
        [1, 1, 1, 0, 0], # tail left / right side

        [0, 0, 0, 1, 0], # head
        [1, 1, 0, 1, 0], # head left / right side

        [1, 0, 0, 0, 1], # side
        [0, 1, 0, 0, 1], # side

        [1, 0, 1, 0, 0], # tail left / right side
        [0, 1, 1, 0, 0], # tail left / right side

        [1, 0, 0, 1, 0], # head left / right side
        [0, 1, 0, 1, 0], # head left / right side

        [1, 1, 0, 0, 0], # left / right side
        [1, 0, 0, 0, 0], # side
        [0, 1, 0, 0, 0], # side
        [0, 0, 0, 0, 0], # side
    ]

tpid2pose = {
    0: 'side',
    1: 'lrside',
    2: 'tail',
    3: 'taillrside',
    4: 'head',
    5: 'headlrside',
    6: 'lrside',
    7: 'lrside',
    8: 'taillrside',
    9: 'taillrside',
    10: 'headlrside',
    11: 'headlrside',
    12 : 'lrside',
    13: 'side',
    14: 'side',
    15: 'side',
}

name2tag = {
    'side'           : 'S',
    'left_side'      : 'LS',
    'right_side'     : 'RS',

    'tail'           : 'T',
    'tail_left_side' : 'TLS',
    'tail_right_side': 'TRS',

    'head'           : 'H',
    'head_left_side' : 'HLS',
    'head_right_side': 'HRS',
}

edges = [[0, 1], [0, 2], [1, 3], [3, 5], [1, 7], [7, 9], [9, 11],
         [2, 4], [4, 6], [2, 8], [8, 10], [10, 12], [7, 8]]
marker_size = 4
line_width = 1
gap_pixel = 3

def viz_pose(image, preds, maxvals, detboxes):

    num_cars = detboxes.shape[0]
    detboxes = detboxes.astype(int)
    for p in range(num_cars):
        detbox = detboxes[p]
        joints = preds[p]
        scores = maxvals[p].squeeze()
        xmin, ymin, xmax, ymax = detbox
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=DARK_GREEN, thickness=1)


        # Draw raw joint detections
        image = draw_sticks(image, joints, edges, line_width)
        image = draw_joints(image, joints, marker_size, scores)




        # # Record raw joint detections
        # ori_joints, ori_scores = deepcopy(joints), deepcopy(scores)

        # image = cv2.rectangle(image, (xmin, ymin - 20), (xmin + 80, ymin),
        #                       (255, 148, 0), thickness=cv2.FILLED)
        # image = cv2.putText(image, '{:.4f}'.format(boxscore), (xmin, ymin - 5),
        #                     cv2.FONT_HERSHEY_COMPLEX_SMALL, 1., (0, 0, 255), 1)
        # image = cv2.putText(image, '{:.4f}'.format(score), (xmin, int((ymin + ymax) / 2.)),
        #                     cv2.FONT_HERSHEY_COMPLEX_SMALL, 1., (0, 0, 255), 1)

    # cv2.imshow('viz', image)
    # cv2.waitKey(0)
    # exit(1)
    return image



def draw_sticks(image, joints, edges, line_width):

    colors = [DARK_GREEN] * len(edges)

    for i in range(len(edges)):
        pos1 = joints[edges[i][0], :]
        pos2 = joints[edges[i][1], :]
        cp = colors[i]
        if not math.isnan(pos1[0]) and not math.isnan(pos2[0]):

            cv2.line(image, tuple(pos1.astype(int)), tuple(pos2.astype(int)),
                     color=cp,
                     thickness=line_width)

    return image

def draw_joints(image, joints, marker_size, scores=None):

    colors = [BLUE, ORANGE, ORANGE, RED, RED, CYAN, CYAN, YELLOW, YELLOW, PURPLE, PURPLE, DARK_GREEN, DARK_GREEN]
    for j in range(joints.shape[0]):
        pos = joints[j, :]
        if not math.isnan(pos[0]):
            cp = colors[j]
            image = cv2.circle(image, tuple(pos.astype(int)), marker_size, cp, -1)
            # if 1280 - pos.astype(int)[0] < 50:
            #     image = cv2.putText(image, '{:.3f}'.format(scores[j]),
            #                         (pos.astype(int)[0]-50, pos.astype(int)[1] - 2), cv2.FONT_HERSHEY_PLAIN, 1.2, cp, 2)
            # else:
            #     image = cv2.putText(image, '{:.3f}'.format(scores[j]),
            #                         (pos.astype(int)[0], pos.astype(int)[1] - 2), cv2.FONT_HERSHEY_PLAIN, 1.2, cp, 2)
    return image

def draw_head_rect(image, rect, person_color, line_width):

    leftup = rect[0]
    rightdown = rect[1]
    cv2.rectangle(image, leftup, rightdown, (int(person_color[0]), int(person_color[1]), int(person_color[2])),
                  line_width)
    return image

def draw_line(image, pos0, pos1, person_color, line_width):

    if not math.isnan(pos0[0]) and not math.isnan(pos1[0]):
        cv2.line(image, tuple(pos0.astype(int)), tuple(pos1.astype(int)),
                 color=(int(person_color[0]), int(person_color[1]), int(person_color[2])),
                 thickness=line_width)
    return image

def get_points(joints_3d, joints_3d_vis, num_joints):
    points       = np.zeros([num_joints, 2])
    points[:, :] = None
    for i in range(num_joints):
        if joints_3d_vis[i, 0] and joints_3d_vis[i, 1]:
            points[i, 0] = joints_3d[i, 0]
            points[i, 1] = joints_3d[i, 1]
    return points




