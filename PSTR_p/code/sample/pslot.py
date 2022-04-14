import cv2
import numpy as np
import torch

from torch.autograd import Variable
import os
from config import system_configs

from .vis import save_batch_heatmaps, save_batch_image_with_joints
from .preprocessing import preprocessing


def to_varabile(tensor, requires_grad=False, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


def kp_detection(db, k_ind, data_aug, debug, pose_debug=False):
    data_rng = system_configs.data_rng
    batch_size = system_configs.batch_size
    roi_size = db.roi_size  # 64 64 # [h w]
    num_roi = db.num_roi
    lighting = db.configs["lighting"]  # true
    rand_color = db.configs["rand_color"]  # color
    inputs = np.zeros(
        (batch_size, num_roi, 3, roi_size[0], roi_size[1]), dtype=np.float32)
    inmasks = np.zeros(
        (batch_size, num_roi, 1, roi_size[0], roi_size[1]), dtype=np.float32)
    roiboxes = np.zeros((batch_size, num_roi, 4), dtype=np.float32)
    viz_joints = list()
    viz_center = np.zeros((batch_size, num_roi, 2), dtype=np.float32)
    viz_scale = np.zeros((batch_size, num_roi, 2), dtype=np.float32)
    viz_rotation = np.zeros((batch_size, num_roi, 1), dtype=np.float32)

    tgt_joints = list()
    tgt_joints_ign = list()
    tgt_areas = list()

    db_size = db.db_inds.size  # 3918 | 738
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind = (k_ind + 1) % db_size
        image_file = db.image_file(db_ind)
        image = cv2.imread(image_file)
        if not os.path.isfile(image_file):
            print('file_not_exist', image_file)
        ps_joints, ps_joints_ign = db.detections(
            db_ind)  # all in the raw coordinate

        # print('send_in!!!!!!')
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
            = preprocessing(b_ind=b_ind,
                            image=image,
                            detections=None,
                            rand_color=rand_color,
                            lighting=lighting,
                            data_rng=data_rng,
                            db=db,
                            joints=ps_joints,
                            joints_ign=ps_joints_ign,
                            roi_size=roi_size,
                            num_roi=num_roi,
                            num_joints=None,
                            inputs=inputs,
                            roiboxes=roiboxes,
                            viz_joints=viz_joints,
                            viz_joints_vis=None,
                            viz_center=viz_center,
                            viz_scale=viz_scale,
                            viz_rotation=viz_rotation,
                            viz_detbox=None,
                            tgt_joints=tgt_joints,
                            tgt_joints_ign=tgt_joints_ign,
                            tgt_areas=tgt_areas,
                            inmasks=inmasks,
                            bsize=batch_size,
                            )

    if pose_debug:
        viz_inputs = torch.from_numpy(
            inputs.reshape((-1, 3, roi_size[0], roi_size[1])))
        used_tgt_joints = torch.from_numpy(
            roi_size[0] * tgt_joints.reshape((-1, num_joints, 3)))  # 1 6 5 3
        used_tgt_joints_vis = torch.from_numpy(
            roi_size[0] * tgt_joints_vis.reshape((-1, num_joints, 3)))  # 1 6 5 3
        # used_tgt_areas      = torch.from_numpy(tgt_areas)  # 1 6 5 1

        debug_path = "./debug/sfsp/train_pose_debug/"
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        save_batch_image_with_joints(viz_inputs, used_tgt_joints, used_tgt_joints_vis,
                                     '{}gt_iter{}.jpg'.format(debug_path, k_ind))
        # save_batch_heatmaps(viz_inputs, viz_targets,
        #                     '{}hm_iter{}.jpg'.format(debug_path, k_ind))
        exit(1)

    batch_images = torch.from_numpy(inputs)
    batch_inmasks = torch.from_numpy(inmasks)
    used_viz_inputs = torch.from_numpy(inputs)
    return {
        "xs": [batch_images, batch_inmasks],
        "ys": [used_viz_inputs, *tgt_joints, *tgt_joints_ign]
    }, k_ind


def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)
