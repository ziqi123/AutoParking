import os
import torch
import cv2
import json
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from config import system_configs
from utils import normalize_

from sample.utils import gaussian_radius, draw_paf, draw_gaussian, \
    visualize_heatmap_return, visualize_paf_return, generate_target
from .post import decode_pose, append_result, draw_keypoints
from utils import crop_image, normalize_, color_jittering_, lighting_, \
    get_affine_transform, affine_transform, fliplr_joints, not_crop_but_resize

from utils.inference import get_final_preds, affine_final_preds
from .viz import *
from sample.vis import *


def kp_decode(nnet,
              images,
              inmasks, ):
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


def make_omit(image, joints, color=GREEN):
    joints = np.reshape(joints, (-1, 4, 2))

    for ps in joints:

        i = 0
        last_joint = np.zeros(2)
        ini_joint = np.zeros(2)
        for idx, joint in enumerate(ps):
            image = cv2.circle(
                image, (int(joint[0]), int(joint[1])), 2, colors[i], 3)
            if idx == 0:
                ini_joint = joint
            if idx > 0:
                image = cv2.line(image, (int(joint[0]), int(joint[1])), (int(last_joint[0]), int(last_joint[1])),
                                 color, 2)
            last_joint = joint
            i += 1
        # image = cv2.circle(image, (int(ps[-1][0]), int(ps[-1][1])), 5, RED, 5)
        image = cv2.line(image, (int(ini_joint[0]), int(ini_joint[1])), (int(last_joint[0]), int(last_joint[1])),
                         color, 2)
    return image


def make_pred(image, joints, color=GREEN):
    joints = np.reshape(joints, (-1, 5, 2))

    for ps in joints:

        i = 0
        last_joint = np.zeros(2)
        ini_joint = np.zeros(2)
        for idx, joint in enumerate(ps[:4, :]):
            image = cv2.circle(
                image, (int(joint[0]), int(joint[1])), 2, colors[i], 3)
            if idx == 0:
                ini_joint = joint
            if idx > 0:
                image = cv2.line(image, (int(joint[0]), int(joint[1])), (int(last_joint[0]), int(last_joint[1])),
                                 color, 2)
            last_joint = joint
            i += 1
        image = cv2.circle(image, (int(ps[-1][0]), int(ps[-1][1])), 5, RED, 5)
        image = cv2.line(image, (int(ini_joint[0]), int(ini_joint[1])), (int(last_joint[0]), int(last_joint[1])),
                         color, 2)
    return image


def make_gth(image, Parkgth):
    pslot = Parkgth[0]
    if pslot.shape[0] > 1:
        for ps in pslot[1:]:
            ps = np.reshape(ps, (4, 2))
            i = 0
            ps = ps.astype(np.int32)
            last_joint = np.zeros(2)
            ini_joint = np.zeros(2)
            for idx, joint in enumerate(ps):
                image = cv2.circle(
                    image, (int(joint[0]), int(joint[1])), 2, colors[i], 3)
                if idx == 0:
                    ini_joint = joint
                if idx > 0:
                    image = cv2.line(image, (int(joint[0]), int(joint[1])),
                                     (int(last_joint[0]), int(last_joint[1])),
                                     GREEN, 2)
                last_joint = joint
                i += 1
            image = cv2.line(image, (int(ini_joint[0]), int(ini_joint[1])),
                             (int(last_joint[0]), int(last_joint[1])),
                             GREEN, 2)
    pslot_ign = Parkgth[1]
    if pslot_ign.shape[0] > 1:
        for ps in pslot_ign[1:]:
            i = 0
            ps = np.reshape(ps, (4, 2))
            ps = set_order(ps)
            ps = ps.astype(np.int32)
            last_joint = np.zeros(2)
            ini_joint = np.zeros(2)
            for idx, joint in enumerate(ps):
                image = cv2.circle(
                    image, (int(joint[0]), int(joint[1])), 2, colors[i], 3)
                if idx == 0:
                    ini_joint = joint
                if idx > 0:
                    image = cv2.line(image, (int(joint[0]), int(joint[1])),
                                     (int(last_joint[0]), int(last_joint[1])),
                                     RED, 2)
                last_joint = joint
                i += 1
            image = cv2.line(image, (int(ini_joint[0]), int(ini_joint[1])),
                             (int(last_joint[0]), int(last_joint[1])),
                             RED, 2)
    return image


def kp_detection(nnet, demo_dir, test_dir, decode_func=kp_decode, threshold=0.9, threshold_pck=5):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    softmax = nn.Softmax(dim=-1)
    TP = 0
    FP = 0
    TN = 0
    dist = []
    list_pck = [0, 0]
    # num_images = len(os.listdir(demo_dir))
    for ind in tqdm(sorted(os.listdir(os.path.join(demo_dir, 'PLD_BirdView_Training_TestSet_v1.0.7_All_3342/pic'))), ncols=67, desc="locating kps"):
        image_name = os.path.join(
            demo_dir, 'PLD_BirdView_Training_TestSet_v1.0.7_All_3342/pic', ind)
        print("image_name", image_name)
        orimage = cv2.imread(image_name)
        gth_name = os.path.join(demo_dir, 'PLD_BirdView_Training_TestSet_v1.0.7_All_3342/annotation',
                                ind.strip('.jpg')+'_OA.txt')

        Parkgth = load_gth(gth_name)
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
        mask = pred[:, :, 1] > threshold
        pred_pslots = [joi[mas] for joi, mas in zip(pred_joints, mask)]
        joints = pred_pslots[0][:, :8].cpu().numpy()
        tmp = pred_pslots[0].cpu().numpy()
        l_s = []
        for j in joints:
            j = np.reshape(j, (4, 2))
            ps = set_order(j)
            ps = np.reshape(ps, (1, 8))
            l_s.append(ps)
        if len(l_s) == 0:
            joints = tmp
        else:
            joints = np.concatenate(l_s, axis=0)
            joints = np.concatenate((joints, tmp[:, 8:]), axis=-1)

        # continue
        TP, FP, TN, store, fail_joints, correct_joints, omit_joints, list_pck = test_pslot(
            Parkgth, joints, TP, FP, TN, ind, list_pck, threshold_pck)
        if store:
            ori = deepcopy(image)
            orimage = deepcopy(image)
            gth = make_gth(orimage, Parkgth)
            if fail_joints is not None:
                fail_pred = make_pred(image, fail_joints, color=YELLOW)
                pred = make_pred(fail_pred, correct_joints, color=GREEN)
            else:
                pred = make_pred(image, correct_joints, color=GREEN)
            if omit_joints is not None:
                gth = make_omit(gth, omit_joints, color=BLUE)
            img = np.concatenate((gth, ori, pred), axis=1)
            cv2.imwrite(os.path.join(test_dir, ind), img)
    recall = TP/(TP+TN)
    precision = TP/(TP+FP)
    print('precision on {} pixels'.format(5), list_pck[0]/list_pck[1])
    return precision, recall


def make_demo(nnet, demo_dir, demo_result_dir, threshold=None, threshold_pck=None):
    return globals()[system_configs.sampling_function](nnet, demo_dir, demo_result_dir, threshold=threshold, threshold_pck=threshold_pck)


def _box2cs(box,
            aspect_ratio=None,
            scale_factor=None):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h,
                    aspect_ratio,
                    scale_factor)


def set_order(joints):
    order_joints = np.zeros((4, 2))
    center = np.sum(joints, axis=0)/4
    rel_joints = joints-center
    # print(rel_joints)
    arctan = np.arctan2(rel_joints[:, -1], rel_joints[:, 0]) * 180 / np.pi
    for i in range(4):
        if arctan[i] < 0:
            arctan[i] = arctan[i]+360
    idx = np.argsort(arctan)
    for j, k in zip([0, 1, 2, 3], idx):
        order_joints[j] = joints[k]

    return order_joints


def make_mask(pslot):
    p = np.reshape(pslot, (4, 2))
    # print('ori_p',p.shape)
    p = set_order(p)
    # print('now_p',p.shape)
    # exit()
    p = np.expand_dims(p, axis=0)
    p = p.astype(np.int32)
    # print('ppp',p)
    # print('pshape',p.shape,type(p))
    im = np.zeros((384, 384), dtype="uint8")
    cv2.polylines(im, p, True, 255, 1)
    cv2.fillPoly(im, p, 255)
    return im


def cal_iou(target, pred):
    and_mask = np.logical_and(target, pred)
    or_mask = np.logical_or(target, pred)
    and_count = np.where(and_mask)
    or_count = np.where(or_mask)
    return and_mask[and_count].shape[0]/or_mask[or_count].shape[0]
    # print([im2].shape)
    # print(img3[im3].shape)


def cal_distance(ps, ps_pred, list, threshold=5):
    ps = ps.reshape((4, 2))
    ps_pred = ps_pred.reshape((4, 2))
    ps = set_order(ps)
    c = 0
    p = 0
    ps_pred = set_order(ps_pred)
    for i, j in zip(ps, ps_pred):
        dist = ((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2) ** 0.5
        if dist < threshold:
            c += 1
        p += 1

    list[0] += c
    list[1] += p

    return list


def test_pslot(all_pslot, pred, TP, FP, TN, ind, list_pck, threshold_pck=5, threshold=0.7, threshold_mask=0.7):
    pred_mask = []
    omit_joints = []
    if pred.shape[0] > 0:
        pred_mask = [make_mask(ps[:8]) for ps in pred]

    pslot = all_pslot[0]
    tmp = []
    if pslot.shape[0] > 1:
        pslot_mask = [make_mask(ps) for ps in pslot[1:]]
        if len(pred_mask) == 0:
            TN += len(pslot_mask)
            omit_joints.append(pslot[1:])
        else:
            for idx, ps in enumerate(pslot_mask):
                iou = []
                for ps_pred in pred_mask:

                    iou.append(cal_iou(ps, ps_pred))
                if max(iou) > threshold:
                    tmp.append(np.argmax(np.array(iou)))
                    list_pck = cal_distance(
                        pslot[1+idx], pred[np.argmax(np.array(iou))][:8], list_pck, threshold_pck)
                    TP += 1
                else:
                    omit_joints.append(pslot[1+idx])
                    TN += 1
    pred_mask_rest = []
    correct_joints = []
    tmp_joints = []
    l = []
    for i in tmp:
        if i not in l:
            l.append(i)
    if len(l) != len(tmp):
        raise ValueError
    for idx, ps in enumerate(pred_mask):
        if idx not in tmp:
            pred_mask_rest.append(ps)
            tmp_joints.append(pred[idx])
        else:
            correct_joints.append(pred[idx])
    pslot_ign = all_pslot[1]
    tmp = []
    if pslot_ign.shape[0] > 1:
        pslot_ign_mask = [make_mask(ps) for ps in pslot_ign[1:]]
        if len(pred_mask_rest) != 0:
            for ps in pslot_ign_mask:
                iou = []
                for ps_pred in pred_mask_rest:
                    iou.append(cal_iou(ps, ps_pred))
                if max(iou) > threshold_mask:
                    tmp.append(np.argmax(np.array(iou)))
    FP += len(pred_mask_rest) - len(tmp)
    fail_joints = []
    l = []
    for i in tmp:
        if i not in l:
            l.append(i)
    if len(l) != len(tmp):
        raise ValueError
    if len(pred_mask_rest) - len(tmp) > 0:
        store = True  # print fail joints
        for idx, ps in enumerate(pred_mask_rest):
            if idx not in tmp:
                fail_joints.append(tmp_joints[idx])
            else:
                correct_joints.append(tmp_joints[idx])
        fail_joints = np.array(fail_joints)

    else:
        store = True  # print correcct joints
        fail_joints = None
    if len(omit_joints) == 0:
        omit_joints = None
    else:
        store = True  # print omit joints
        omit_joints = np.array(omit_joints)
    return TP, FP, TN, store, fail_joints, correct_joints, omit_joints, list_pck


def load_gth(annt):
    with open(annt, "r") as f:
        annt = f.readlines()
    count = 0
    count_ign = 0
    l = []
    l_ign = []
    for line in annt:
        line_annt = line.strip('/n').split(' ')
        # print(len(line_annt),int(line_annt[-4]))
        if line_annt[0] != 'line' or line_annt[-4] == '3' or len(line_annt) != 13:
            continue
        # print(line_annt)
        if line_annt[-4] in ['0', '2']:
            count += 1
            l.append(np.array([int(line_annt[i + 1]) for i in range(8)]))
            continue

        if line_annt[-4] in ['1', '5']:
            count_ign += 1
            l_ign.append(np.array([int(line_annt[i + 1]) for i in range(8)]))
            continue
    parking = np.zeros(
        (count + 1, 8), dtype=np.float) if count != 0 else np.zeros((1, 8), dtype=np.float)
    parking_ignore = np.zeros((count_ign + 1, 8), dtype=np.float) if count_ign != 0 else np.zeros((1, 8),
                                                                                                  dtype=np.float)
    if count_ign != 0:
        for id in range(count_ign):
            parking_ignore[id + 1, :] = (l_ign[id]/4)*1.5
    if count != 0:
        for id in range(count):
            parking[id + 1, :] = (l[id]/4)*1.5

    return [parking, parking_ignore]


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
