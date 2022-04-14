import cv2
import math
import numpy as np
import random
import torch
from utils import crop_image, normalize_, color_jittering_, lighting_, \
    get_affine_transform, affine_transform, fliplr_joints, flipub_joints


flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)


def preprocessing(b_ind,
                  image,
                  detections,
                  rand_color,
                  lighting,
                  data_rng,
                  db,
                  joints,
                  joints_ign,
                  roi_size,
                  num_roi,
                  num_joints,
                  inputs,
                  roiboxes,
                  viz_joints,
                  viz_joints_vis,
                  viz_center,
                  viz_scale,
                  viz_rotation,
                  viz_detbox,
                  tgt_joints,
                  tgt_joints_ign,
                  tgt_areas,
                  inmasks,
                  bsize
                  ):
    joints = joints.copy()
    joints_ign = joints_ign

    if joints.shape[0] > 1:

        ujoints = np.zeros(
            (num_roi, joints.shape[0], joints.shape[1]), dtype=np.float)
        ujoints[0] = joints.astype(float)

    else:
        ujoints = np.zeros((num_roi, 1, 8), dtype=np.float)
    if joints_ign.shape[0] > 1:
        ujoints_ign = np.zeros(
            (num_roi, joints_ign.shape[0], joints_ign.shape[1]), dtype=np.float)
        ujoints_ign[0] = joints_ign.astype(float)
    else:
        ujoints_ign = np.zeros((num_roi, 1, 8), dtype=np.float)
    uraw_box = None

    (inputs, inmasks,
     roiboxes,
     viz_center,
     viz_scale,
     viz_joints,
     viz_joints_vis,
     viz_detbox,
     viz_rotation,
     tgt_joints,
     tgt_joints_ign,
     tgt_areas) \
        = _resize_image(db,
                        b_ind,
                        image,
                        detections,
                        ujoints,
                        ujoints_ign,
                        uraw_box,
                        roi_size,
                        num_roi,
                        num_joints,
                        lighting,
                        rand_color,
                        data_rng,
                        inputs,
                        inmasks,
                        roiboxes,
                        viz_joints,
                        viz_joints_vis,
                        viz_center,
                        viz_scale,
                        viz_rotation,
                        viz_detbox,
                        tgt_joints,
                        tgt_joints_ign,
                        tgt_areas,
                        bsize)

    return (inputs, inmasks,
            roiboxes,
            viz_center,
            viz_scale,
            viz_joints,
            viz_joints_vis,
            viz_detbox,
            viz_rotation,
            tgt_joints,
            tgt_joints_ign,
            tgt_areas)


def _resize_image(db,
                  b_ind: int,
                  raw_image: np.ndarray,
                  detections: np.ndarray,
                  ujoints: np.ndarray,
                  ujoints_ign: np.ndarray,
                  uraw_box: np.ndarray,
                  roi_size: list,
                  num_roi: int,
                  num_joints: None,
                  lighting: bool,
                  rand_color: bool,
                  data_rng: np.ndarray,
                  inputs: np.ndarray,
                  inmasks: np.ndarray,
                  roiboxes: np.ndarray,
                  viz_joints: list,
                  viz_joints_vis: None,
                  viz_center: np.ndarray,
                  viz_scale: np.ndarray,
                  viz_rotation: np.ndarray,
                  viz_detbox: None,
                  tgt_joints: list,
                  tgt_joints_ign: list,
                  tgt_areas: np.ndarray,
                  bsize: int):
    image = raw_image.copy()
    image = cv2.resize(image, (roi_size[0], roi_size[1]))
    ujoints = ujoints/4
    ujoints = add_center(ujoints)
    ujoints_ign = ujoints_ign/4
    ujoints_ign = add_center(ujoints_ign)
    height, width = image.shape[0:2]  # h w
    mask = np.zeros((height, width, 1), dtype=np.float)

    # random horizontal flip
    lr_flip_random = random.random()
    if lr_flip_random <= 0.5:
        image[:] = image[:, ::-1, :]
        width = image.shape[1]
        for i in range(num_roi):
            ujoints[i] = fliplr_joints(
                ujoints[i], width)
            ujoints_ign[i] = fliplr_joints(
                ujoints_ign[i], width)

    ub_flip_random = random.random()
    if ub_flip_random <= 0.5:
        image[:] = image[::-1, :, :]
        height = image.shape[0]
        for i in range(num_roi):
            ujoints[i] = flipub_joints(
                ujoints[i], height)
            ujoints_ign[i] = flipub_joints(
                ujoints_ign[i], height)

    # # TODO Debug
    for i in range(num_roi):
        joints = ujoints[i].copy()
        joints_ign = ujoints_ign[i].copy()
        num_pslot = joints.shape[0]
        num_ign_pslot = joints_ign.shape[0]
        tl = np.array([0, 0], dtype=np.float32)
        br = np.array([width, height], dtype=np.float32)
        size = [width, height]
        c, s = _box2cs(size, aspect_ratio=1)
        sf = 0.05
        s *= (1 + sf)
        t_sf = sf / (1 + sf)
        s = s * np.clip(np.random.randn() * t_sf + 1, 1 - t_sf, 1 + t_sf)
        rf = 90
        r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
            if random.random() <= 0.95 else 0
        viz_center[b_ind][i] = c
        viz_scale[b_ind][i] = s
        viz_rotation[b_ind][i][0] = r
        roiboxes[b_ind][i, 0] = (c[0] - 4 * s[0])
        roiboxes[b_ind][i, 1] = (c[1] - 4 * s[1])
        roiboxes[b_ind][i, 2] = (c[0] + 4 * s[0])
        roiboxes[b_ind][i, 3] = (c[1] + 4 * s[1])

        trans = get_affine_transform(c, s, r, roi_size)
        input = cv2.warpAffine(
            image,
            trans,
            (int(roi_size[0]), int(roi_size[1])),
            flags=cv2.INTER_LINEAR)
        inmask = cv2.warpAffine(
            mask,
            trans,
            (int(roi_size[0]), int(roi_size[1])),
            flags=cv2.INTER_LINEAR, borderValue=255)

        ps_joints = np.reshape(joints, (-1, 5, 2))
        for j in range(num_pslot):
            for k in range(5):
                ps_joints[j, k, :] = affine_transform(
                    ps_joints[j, k, :], trans)
        joints = np.reshape(ps_joints, (-1, 10))
        ign_extra = []
        for idx, i1 in enumerate(joints[1:]):
            for coord in i1:
                if coord < 0 or coord > roi_size[0]:
                    ign_extra.append(idx)
                    break
        l = []
        l_new = []
        joints_ign_extra = None
        if len(ign_extra) != 0:
            # print('ori_joints',joints.shape)
            for j in range(joints.shape[0]-1):
                if j in ign_extra:
                    l.append(np.clip(joints[1+j], 0, roi_size[0]))
                else:
                    l_new.append(joints[1+j])
            if len(l_new) == 0:
                joints = joints[:1]
            else:
                joints = np.concatenate((joints[:1], np.array(l_new)), axis=0)
            joints_ign_extra = np.array(l)
        ps_joints_ign = np.reshape(joints_ign, (-1, 5, 2))
        for j in range(num_ign_pslot):
            for k in range(5):
                ps_joints_ign[j, k, :] = affine_transform(
                    ps_joints_ign[j, k, :], trans)
            ps_joints_ign[j] = np.clip(ps_joints_ign[j], 0, roi_size[0])
        joints_ign = np.reshape(ps_joints_ign, (-1, 10))
        if joints_ign_extra is not None:
            joints_ign = np.concatenate((joints_ign, joints_ign_extra), axis=0)
        tl = affine_transform(tl, get_affine_transform(c, s, 0, roi_size))
        br = affine_transform(br, get_affine_transform(c, s, 0, roi_size))
        tgt_areas.append(((br[0] - tl[0]) / roi_size[1])
                         * ((br[1] - tl[1]) / roi_size[0]))

        # norm and color noise
        input = input.astype(np.float32) / 255.
        if rand_color:
            color_jittering_(data_rng, input)
            if lighting:
                lighting_(data_rng, input, 0.1, db.eig_val, db.eig_vec)
        normalize_(input, db.mean, db.std)
        inputs[b_ind, i] = input.transpose(2, 0, 1)
        joints /= roi_size[1]
        joints_ign /= roi_size[1]
        stack_joints = np.expand_dims(joints, axis=0)
        repeat_joints = stack_joints.repeat(bsize, axis=0)

        stack_joints_ign = np.expand_dims(joints_ign, axis=0)
        repeat_joints_ign = stack_joints_ign.repeat(bsize, axis=0)
        tgt_joints.append(torch.from_numpy(repeat_joints.astype(np.float32)))
        tgt_joints_ign.append(torch.from_numpy(
            repeat_joints_ign.astype(np.float32)))
        inmasks[b_ind, i] = inmask

    return (inputs, inmasks,
            roiboxes,
            viz_center,
            viz_scale,
            viz_joints,
            viz_joints_vis,
            viz_detbox,
            viz_rotation,
            tgt_joints,
            tgt_joints_ign,
            tgt_areas)


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


def _box2cs(size,
            aspect_ratio=None,
            scale_factor=None):
    x, y, w, h = 0, 0, size[0], size[1]
    return _xywh2cs(x, y, w, h,
                    aspect_ratio,
                    scale_factor)


def _xywh2cs(x, y, w, h,
             aspect_ratio,
             scale_factor):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    # print('wh',w,h)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0/200, h * 1.0/200],
        dtype=np.float32)
    return center, scale


def add_center(joints):
    # print('joints',joints.shape)
    list = []
    for joi in joints[0]:
        joi_p = np.reshape(joi, (4, 2))
        center = np.expand_dims(
            (joi_p[0]+joi_p[1]+joi_p[2]+joi_p[3])/4, axis=0)
        joi_p = np.concatenate((joi_p, center), axis=0)
        list.append(joi_p)
    jjj = np.array(list)
    joints = np.expand_dims(np.reshape(jjj, (-1, 10)), axis=0)
    return joints


def get_points(joints_3d, joints_3d_vis, num_joints):
    points = np.zeros([num_joints, 2])
    points[:, :] = None
    for i in range(num_joints):
        if joints_3d_vis[i, 0] and joints_3d_vis[i, 1]:
            points[i, 0] = joints_3d[i, 0]
            points[i, 1] = joints_3d[i, 1]
    return points


def draw_joints(image, joints, marker_size):

    colors = [BLUE, YELLOW, RED, GREEN, PURPLE, CYAN]
    for j in range(joints.shape[0]):
        pos = joints[j, :]
        if not math.isnan(pos[0]):
            cp = colors[(j - 1) % len(colors)]
            image = cv2.circle(image, tuple(
                pos.astype(int)), marker_size, cp, -1)

    return image


def _clip_detections(image, detections):
    detections = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds = ((detections[:, 2] - detections[:, 0]) > 0) & \
        ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections
