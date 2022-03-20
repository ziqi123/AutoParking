# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torchvision
import cv2

from utils.inference import get_max_preds

RED = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (115, 181, 34)
BLUE = (255, 0, 0)
CYAN = (255, 128, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

colors = [BLUE, ORANGE, YELLOW, RED]


def set_order(joints):
    order_joints = np.zeros((4, 2))
    center = np.sum(joints, axis=0)/4
    rel_joints = joints-center
    arctan = np.arctan2(rel_joints[:, -1], rel_joints[:, 0]) * 180 / np.pi
    for i in range(4):
        if arctan[i] < 0:
            arctan[i] = arctan[i]+360
    idx = np.argsort(arctan)
    for j, k in zip([0, 1, 2, 3], idx):
        order_joints[j] = joints[k]

    return order_joints


def save_batch_image_with_joints(batch_image,
                                 batch_joints,
                                 batch_joints_vis,
                                 file_name,
                                 nrow=8,
                                 padding=2, flag='pred', ign_joints=None):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    ign_joints = torch.zeros(nmaps, 10)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            if flag == 'gth':
                # if batch_joints[k][0].cpu().numpy().shape[0] == 1 and ign_joints[k][0].cpu().numpy().shape[0] == 1:
                #     k += 1
                #     continue
                if batch_joints[k][0].cpu().numpy().shape[0] == 1:
                    joints = None
                else:
                    joints = batch_joints[k][0, 1:].cpu().numpy()
                # if ign_joints[k][0].cpu().numpy().shape[0] == 1:
                #     joints_ign = None
                # else:
                #     joints_ign = ign_joints[k][0, 1:].cpu().numpy()
                    # print('real_joints',joints.shape)

            else:
                # print('length',len(batch_joints))
                joints = batch_joints[k].cpu().numpy()
                joints_ign = None
                if joints.shape[0] == 0:
                    k = k+1
                    continue
            if joints is not None:
                joints = np.reshape(joints, (-1, 5, 2))
                # joints=set_order(joints)
                for ps in joints:
                    i = 0
                    last_joint = np.zeros(2)
                    ini_joint = np.zeros(2)
                    ps_order = set_order(ps[:4, :])
                    for idx, joint in enumerate(ps_order):
                        joint[0] = x * width + padding + joint[0]
                        joint[1] = y * height + padding + joint[1]
                        # joint[2]
                        # joint[3]
                        # if joint_vis[0]:
                        cv2.circle(ndarr, (int(joint[0]), int(
                            joint[1])), 2, colors[i], 2)
                        if idx == 0:
                            ini_joint = joint
                        if idx > 0:
                            # print('jjjjjj',int(joint),int(last_joint))
                            cv2.line(ndarr, (int(joint[0]), int(joint[1])), (int(
                                last_joint[0]), int(last_joint[1])), GREEN, 3)
                        last_joint = joint
                        i += 1
                    cv2.line(ndarr, (int(ini_joint[0]), int(ini_joint[1])), (int(
                        last_joint[0]), int(last_joint[1])), GREEN, 3)
                    center_x = x * width + padding + ps[-1, 0]
                    center_y = y * height + padding + ps[-1, 1]
                    cv2.circle(
                        ndarr, (int(center_x), int(center_y)), 5, BLUE, 5)

            # if joints_ign is not None:
            #     joints_ign = np.reshape(joints_ign, (-1, 5, 2))
            #     # joints_ign = set_order(joints_ign)
            #     for ps in joints_ign:
            #         i = 0
            #         last_joint = np.zeros(2)
            #         ini_joint = np.zeros(2)
            #         ps_order = set_order(ps[:4, :])
            #         for idx, joint in enumerate(ps_order):
            #             if joint[0] > 256 or joint[0] < 0:
            #                 raise ValueError
            #             if joint[1] > 256 or joint[1] < 0:
            #                 raise ValueError
            #             joint[0] = x * width + padding + joint[0]
            #             joint[1] = y * height + padding + joint[1]
            #             # joint[2]
            #             # joint[3]
            #             # if joint_vis[0]:
            #             cv2.circle(ndarr, (int(joint[0]), int(
            #                 joint[1])), 2, colors[i], 2)
            #             if idx == 0:
            #                 ini_joint = joint
            #             if idx > 0:
            #                 # print('jjjjjj',int(joint),int(last_joint))
            #                 cv2.line(ndarr, (int(joint[0]), int(joint[1])), (int(
            #                     last_joint[0]), int(last_joint[1])), RED, 3)
            #             last_joint = joint
            #             i += 1
            #         cv2.line(ndarr, (int(ini_joint[0]), int(ini_joint[1])), (int(
            #             last_joint[0]), int(last_joint[1])), RED, 3)
            #         center_x = x * width + padding + ps[-1, 0]
            #         center_y = y * height + padding + ps[-1, 1]
            #         cv2.circle(
            #             ndarr, (int(center_x), int(center_y)), 5, BLUE, 5)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    # print(file_name)
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(input, target, output,
                      prefix):

    # save_batch_image_with_joints(
    #     input, meta['joints'], meta['joints_vis'],
    #     '{}_gt.jpg'.format(prefix)
    # )
    #
    # save_batch_image_with_joints(
    #     input, joints_pred, meta['joints_vis'],
    #     '{}_pred.jpg'.format(prefix)
    # )
    save_batch_heatmaps(
        input, target, '{}_hm_gt.jpg'.format(prefix)
    )
    save_batch_heatmaps(
        input, output, '{}_hm_pred.jpg'.format(prefix)
    )


def save_debug_images_training(input, target, output, prefix):

    # save_batch_image_with_joints(
    #     input, joints, joints_vis,
    #     '{}_gt.jpg'.format(prefix)
    # )

    # save_batch_image_with_joints(
    #     input, joints_pred, joints_vis,
    #     '{}_pred.jpg'.format(prefix)
    # )

    save_batch_heatmaps(
        input, target, '{}_hm_gt.jpg'.format(prefix)
    )
    save_batch_heatmaps(
        input, output, '{}_hm_pred.jpg'.format(prefix)
    )


def save_debug_images_joints(input, gt_joints, gt_joints_vis,
                             joints_pred=None, ign_joints=None, prefix=None):

    save_batch_image_with_joints(
        input, gt_joints, gt_joints_vis,
        '{}_gt.jpg'.format(prefix), flag='gth', ign_joints=ign_joints
    )

    save_batch_image_with_joints(
        input, joints_pred, gt_joints_vis,
        '{}_pred.jpg'.format(prefix)
    )


def viz_infer_joints(input, gt_joints, gt_joints_vis,
                     joints_pred=None, ign_joints=None, prefix=None):
    save_batch_image_with_joints(
        input, gt_joints, gt_joints_vis,
        '{}_gt.jpg'.format(prefix), flag='gth', ign_joints=ign_joints
    )

    save_batch_image_with_joints(
        input, joints_pred, gt_joints_vis,
        '{}_pred.jpg'.format(prefix)
    )
