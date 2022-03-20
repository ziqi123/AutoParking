import os
from tkinter.tix import Y_REGION
import torch
import cv2
import json
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_, \
    get_affine_transform, affine_transform, fliplr_joints, not_crop_but_resize

from utils.inference import get_final_preds, affine_final_preds
from .viz import *
from sample.vis import *


def kp_decode(nnet,
              images,
              inmasks,):
    out = nnet.test([images, inmasks])
    # out_joints = out['pred_boxes'].cpu()

    return out


RED = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (115, 181, 34)
BLUE = (255, 0, 0)
CYAN = (255, 128, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

colors = [BLUE, ORANGE, YELLOW, RED]
mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
show_list = ['20201218142128-00-00.MP4_103.jpg', '20201218142128-00-00.MP4_104.jpg',
             '20201218142128-00-00.MP4_105.jpg', '20201218142128-00-00.MP4_106.jpg', '20201218142128-00-00.MP4_252.jpg']


def kp_detection(nnet, demo_dir, demo_result_dir, annt_result_dir, decode_func=kp_decode, threshold=0.9):
    softmax = nn.Softmax(dim=-1)
    num_images = len(os.listdir(demo_dir))
    for ind in tqdm(os.listdir(demo_dir), ncols=67, desc="locating kps"):
        image_name = os.path.join(demo_dir, ind)
        orimage = cv2.imread(image_name)
        image = cv2.resize(orimage, (384, 384))
        input = image.astype(np.float32) / 255.
        normalize_(input, mean, std)
        height, width = image.shape[0:2]
        inputs = np.zeros((1, 1, 3, 384, 384), dtype=np.float32)
        inmasks = np.zeros((1, 1, 1, 384, 384), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.float)
        inmasks[0, 0] = mask
        inputs[0, 0] = input.transpose(2, 0, 1)
        # B num_roi 3 roi_size[0] roi_size[1]
        batch_images = torch.from_numpy(inputs)
        batch_inmasks = torch.from_numpy(inmasks)
        outputs = decode_func(nnet,
                              images=batch_images,
                              inmasks=batch_inmasks)
        pred_joints = 384 * outputs['pred_boxes'].detach()
        pred_classes = outputs['pred_classes']
        pred = softmax(pred_classes)
        annt_result_file = os.path.join(
            annt_result_dir, ind.strip('.jpg')+'_OA.txt')

        # if ind in show_list:

        mask = pred[:, :, 1] > threshold
        pred_pslots = [joi[mas] for joi, mas in zip(pred_joints, mask)]
        joints = pred_pslots[0].cpu().numpy()
        if joints.shape[0] == 0:
            # k = k + 1
            cv2.imwrite(os.path.join(demo_result_dir, ind), image)
            continue
        joints = np.reshape(joints, (-1, 5, 2))
        for ps in joints:
            i = 0
            last_joint = np.zeros(2)
            ini_joint = np.zeros(2)
            ps_ord = set_order(ps[:4, :])

            for idx, joint in enumerate(ps_ord):
                image = cv2.circle(
                    image, (int(joint[0]), int(joint[1])), 2, colors[i], 3)
                with open(annt_result_file, "a") as file:
                    x = int(joint[0])*(1024/384)
                    y = int(joint[1])*(1024/384)
                    file.write(str(x))
                    file.write(' ')
                    file.write(str(y))
                    file.write('\n')

                if idx == 0:
                    ini_joint = joint
                if idx > 0:
                    image = cv2.line(image, (int(joint[0]), int(joint[1])), (int(
                        last_joint[0]), int(last_joint[1])), GREEN, 2)
                last_joint = joint
                i += 1
            image = cv2.line(image, (int(ini_joint[0]), int(ini_joint[1])), (int(
                last_joint[0]), int(last_joint[1])), GREEN, 2)
            image = cv2.circle(
                image, (int(ps[-1][0]), int(ps[-1][1])), 5, RED, 5)
        cv2.imwrite(os.path.join(demo_result_dir, ind), image)

    return 0


def make_demo(nnet, demo_dir, demo_result_dir, annt_result_dir, threshold=0.985):
    # print('11111111')
    return globals()[system_configs.sampling_function](nnet, demo_dir, demo_result_dir, annt_result_dir, threshold=threshold)


def _box2cs(box,
            aspect_ratio=None,
            scale_factor=None):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h,
                    aspect_ratio,
                    scale_factor)


def _xywh2cs(x, y, w, h,
             aspect_ratio,
             scale_factor):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / 200, h * 1.0 / 200],
        dtype=np.float32)
    # if scale_factor is not None:
    #     scale = scale * (1. + scale_factor)

    return center, scale
