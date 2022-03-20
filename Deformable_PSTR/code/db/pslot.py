from test.viz import *
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from config import system_configs
from db.detection import DETECTION
from torchvision.ops import RoIAlign
from skimage.io import imread
from torch.autograd import Variable, gradcheck
from torchvision import transforms, utils
import torch
from tqdm import tqdm
import math
import cv2
import xml.etree.ElementTree as ET
import pickle
import numpy as np
import json
import sys

sys.path.insert(0, "data/coco/PythonAPI/")


def to_varabile(tensor, requires_grad=False, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)


class PSLOT(DETECTION):
    def __init__(self, db_config, split):
        super(PSLOT, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir
        self.roi_size = db_config["roi_size"]
        self.num_roi = db_config["num_roi"]

        self.roi_height, self.roi_width = db_config["roi_size"]
        self.aspect_ratio = self.roi_width * 1.0 / self.roi_height
        # self.scale_factor = db_config["scale_factor"]

        self.detbox_path = db_config["detbox_path"]

        self._split = split
        self._labelset = {
            "train": "train/data/annt_256",
            "test": "val/data/annt_256",
            "demo": "val/data/annt_256",

        }[self._split]
        self._imageset = {
            "train": "train/data/img_256",
            "test": "val/data/img_256",
            "demo": "val/data/img_256",

        }[self._split]
        self._cachefile = {
            "train": "train",
            "test": "val",
            "demo": "demo"

        }[self._split]
        self._pslot_dir = (system_configs.data_dir)

        self._label_dir = os.path.join(self._pslot_dir, self._labelset)
        self._label_file = os.path.join(self._label_dir, '{}_OA.txt')

        self._image_dir = os.path.join(self._pslot_dir, self._imageset)
        self._image_file = os.path.join(self._image_dir, "{}.jpg")
        self._data = "pslot"
        self._mean = np.array(
            [0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array(
            [0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array(
            [0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            0
        ]  # 0 car
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }

        self._cache_file = os.path.join(
            cache_dir, "{}.pkl".format(self._cachefile))

        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

    def _load_data(self, debug_paf=False, debug_roi_align=False, debug_raw_gt=False):

        if len(self.detbox_path):
            raise NotImplementedError

        else:
            print("loading from cache file: {}".format(self._cache_file))
            if not os.path.exists(self._cache_file):
                print("No cache file found...")

                self._extract_data()
                if debug_raw_gt or debug_paf or debug_roi_align:
                    pass
                else:
                    with open(self._cache_file, "wb") as f:
                        pickle.dump([self._image_file,
                                     self._image_ids,
                                     self._pslot_joints,
                                     self._pslot_joints_ignore
                                     ], f)
            else:
                with open(self._cache_file, "rb") as f:
                    (self._image_file,
                     self._image_ids,
                     self._pslot_joints,
                     self._pslot_joints_ignore) = pickle.load(f)

            if debug_paf:
                print(len(self._image_ids))  # 3918
                sigma_paf = 5
                variable_width = True
                width_ratio = 767 / (192 * 4)
                height_ratio = 1407 / (352 * 4)
                for i in range(len(self._image_ids)):
                    image_id = self._image_ids[i]
                    image = cv2.imread(self.image_file(i))
                    # print(image.dtype)
                    cv2.imshow('image: {}'.format(i), image)
                    cv2.waitKey(0)
                    out_pafs = np.zeros((2, 720, 1280))

                    for ind, detection in enumerate(self._detections[image_id]):
                        print(detection)
                        xtl, ytl = detection[0], detection[1]  # xmin ymin
                        xbr, ybr = detection[2], detection[3]  # xmax ymax
                        print(xtl, ytl, xbr, ybr)

                        fxtl = (xtl * width_ratio)
                        fytl = (ytl * height_ratio)
                        fxbr = (xbr * width_ratio)
                        fybr = (ybr * height_ratio)

                        xtl = int(fxtl)
                        ytl = int(fytl)
                        xbr = int(fxbr)
                        ybr = int(fybr)

                        tl = np.array([xtl, ytl], dtype=float)
                        br = np.array([xbr, ybr], dtype=float)

                        print(xtl, ytl, xbr, ybr)
                        if xtl >= 0 and ytl >= 0 and xbr <= 1280 and ybr <= 720:
                            part_line_segment = br - tl
                            l = np.linalg.norm(part_line_segment)
                            print('l: {}'.format(l))

                            if l > 1e-2:
                                sigma = sigma_paf
                                if variable_width:
                                    sigma = sigma_paf * l * 0.025
                                v = part_line_segment / l
                                print('v: {}'.format(v))
                                v_per = v[1], -v[0]
                                x, y = np.meshgrid(
                                    np.arange(1280), np.arange(720))

                                dist_along_part = v[0] * \
                                    (x - tl[0]) + v[1] * (y - tl[1])
                                dist_per_part = np.abs(
                                    v_per[0] * (x - tl[0]) + v_per[1] * (y - tl[1]))

                                mask1 = dist_along_part >= 0
                                mask2 = dist_along_part <= l
                                mask3 = dist_per_part <= sigma
                                mask = mask1 & mask2 & mask3
                                out_pafs[0] = out_pafs[0] + \
                                    mask.astype('float32') * v[0]
                                out_pafs[1] = out_pafs[1] + \
                                    mask.astype('float32') * v[1]
                                indicator = np.where(mask, 255, 0)
                                indicator = indicator.astype('uint8')
                                print(indicator.dtype)
                                print('np.sum(mask):{}'.format(np.sum(mask)))
                                print('np.sum(indicator): {}'.format(
                                    np.sum(indicator)))

                    indicator = np.where(np.logical_or(
                        out_pafs[0] != 0, out_pafs[1] != 0), 255, 0)
                    indicator = indicator.astype('uint8')
                    print(indicator.dtype)
                    print('np.sum(mask):{}'.format(np.sum(mask)))
                    print('np.sum(indicator): {}'.format(np.sum(indicator)))
                    print(type(indicator), indicator.dtype, indicator.shape)
                    # indicator = np.stack([indicator]*3, axis=-1)
                    # seg = 0.6 * image + 0.4 * indicator
                    cv2.imshow('paf: {}'.format(i), indicator)
                    cv2.waitKey(0)
                    break
                exit(1)
            if debug_roi_align:
                is_cuda = torch.cuda.is_available()
                crop_height = 192
                crop_width = 352
                roi_align = RoIAlign(
                    (crop_height, crop_width), spatial_scale=1.0, sampling_ratio=-1)

                for i in range(len(self._image_ids)):
                    image_id = self._image_ids[i]
                    next_image_id = self._image_ids[i + 1]

                    detection = self._detections[image_id]
                    detection_2 = self._detections[next_image_id]

                    image_path = self.image_file(i)
                    image_path_2 = self.image_file(i + 1)

                    image_data1 = transforms.ToTensor()(imread(image_path)).unsqueeze(0)
                    boxes_data_1 = detection[:, :4]
                    boxes_data_1 = torch.FloatTensor(boxes_data_1)
                    # box_index_data_1 = torch.IntTensor(np.ones((boxes_data_1.shape[0],), dtype=np.int32) * i)

                    image_data2 = transforms.ToTensor()(imread(image_path_2)).unsqueeze(0)
                    boxes_data_2 = detection_2[:, :4]
                    # print(boxes_data_2.dtype) #  float64

                    boxes_data_2 = torch.FloatTensor(boxes_data_2)
                    # print(boxes_data_2.dtype) # torch.float32

                    # box_index_data_2 = torch.IntTensor(np.ones((boxes_data_2.shape[0],), dtype=np.int32) * (i + 1))

                    image_data = torch.cat((image_data1, image_data2), 0)

                    image_torch = to_varabile(image_data, is_cuda=is_cuda)
                    # boxes = to_varabile(boxes_data, is_cuda=is_cuda)
                    boxes_data_1 = to_varabile(boxes_data_1, is_cuda=is_cuda)
                    boxes_data_2 = to_varabile(boxes_data_2, is_cuda=is_cuda)

                    roi_align_results = roi_align(
                        image_torch, [boxes_data_1, boxes_data_2])
                    print(roi_align_results.shape)

                    crops_torch_data = roi_align_results.data.cpu().numpy().transpose(0, 2, 3, 1)
                    for b in range(crops_torch_data.shape[0]):
                        plt.imshow(crops_torch_data[b])
                        plt.show()

                    exit(1)
            if debug_raw_gt:
                for i in range(len(self._image_ids)):
                    print(i)
                    image_id = self._image_ids[i]
                    print(image_id)
                    detections = self._detections[image_id]
                    # print("detections.shape: {}".format(detections.shape))
                    # print("len(gt_dbs): {}, type(gt_dbs): {}".format(len(gt_dbs), type(gt_dbs)))
                    num_obj = detections.shape[0]
                    image_path = self.image_file(i)
                    print(image_path)
                    image_data = cv2.imread(image_path)
                    print(image_data.shape)
                    # exit()
                    print(detections.shape)
                    for num in range(num_obj):
                        joints_3d = self._joints[image_id][num].astype(int)
                        joints_3d_vis = self._joints_vis[image_id][num].astype(
                            int)
                        box = detections[num, :4].astype(int)
                        joints = self.get_points(joints_3d, joints_3d_vis)
                        image_data = cv2.rectangle(image_data, (box[0], box[1]), (box[2], box[3]),
                                                   color=(0, 255, 0), thickness=1)
                        image_data = self.draw_joints(
                            image_data, joints, marker_size=6)
                        print("box: {}".format(box))

                    cv2.imshow('dbbox', image_data)
                    cv2.waitKey(0)
                    exit(1)

    def _load_coco_data(self):

        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)
        coco_ids = self._coco.getImgIds()

        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        # print(self._coco_categories)  # [{'supercategory': 'none', 'id': 0, 'name': 'car'}]

        self._coco_eval_ids = eval_ids

    def _extract_data(self):

        # self._coco = COCO(self._label_file)
        # self._cat_ids = self._coco.getCatIds()
        self._image_ids = [i.strip('.jpg')
                           for i in os.listdir(self._image_dir)]
        copy_image_ids = deepcopy(self._image_ids)
        self._pslot_joints = {}
        self._pslot_joints_ignore = {}
        for img_id in self._image_ids:
            annt_file = self._label_file.format(img_id)
            if not os.path.isfile(annt_file):
                self._pslot_joints[img_id] = np.zeros((1, 8), dtype=np.float)
                self._pslot_joints_ignore[img_id] = np.zeros(
                    (1, 8), dtype=np.float)
                continue
            with open(annt_file, "r") as f:
                annt = f.readlines()
            l = []
            ll = []
            count = 0
            j = 0
            for line in annt:
                line_annt = line.strip('\n').split(' ')
                j = j+2
                l.append(int(float(line_annt[0])))
                l.append(int(float(line_annt[1])))
                if j % 8 == 0:
                    ll.append(l)
                    l = []
                    count = count+1
            parking = np.zeros(
                (count+1, 8), dtype=np.float) if count != 0 else np.zeros((1, 8), dtype=np.float)
            if count != 0:
                for id in range(count):
                    parking[id+1, :] = ll[id]
            parking_ignore = np.zeros(
                (1, 8), dtype=np.float)
            self._pslot_joints[img_id] = parking
            self._pslot_joints_ignore[img_id] = parking_ignore
        self._image_ids = copy_image_ids

    def detections(self, ind):
        image_id = self._image_ids[ind]
        ps_joints = self._pslot_joints[image_id]
        ps_joints_ign = self._pslot_joints_ignore[image_id]
        return ps_joints.copy(),\
            ps_joints_ign.copy()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None
        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]

    def convert_to_coco_keypoints(self, all_kps, all_scores):
        detections = []
        for image_id in all_kps:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_kps[image_id]:
                category_id = self._classes[cls_ind]
                for kps, scr in zip(all_kps[image_id][cls_ind], all_scores[image_id][cls_ind]):
                    kps = np.concatenate([kps, np.ones_like(scr)], axis=-1)
                    kps = kps.tolist()
                    kps = [item for sublist in kps for item in sublist]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "keypoints": kps,
                        "score": float("{:.2f}".format(1.))
                    }
                    detections.append(detection)

        return detections

    def evaluate_keypoints(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None
        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "keypoints")
        coco_eval.params.kpt_oks_sigmas = np.array([0.25,
                                                    0.79,
                                                    0.79,
                                                    0.72,
                                                    0.72,
                                                    0.62,
                                                    0.62,
                                                    1.07,
                                                    1.07,
                                                    0.87,
                                                    0.87,
                                                    0.89,
                                                    0.89]) / 10.0
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]

    def _extract_det_box(self):

        all_boxes = None
        with open(self.detbox_path, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError('=> Load %s fail!' % self.bbox_file)

        print('=> Total boxes: {}'.format(len(all_boxes)))

        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"] for img_id in coco_image_ids
        ]

        self._detections = {}
        self._joints = {}
        self._joints_vis = {}

        for coco_image_id, image_id in zip(coco_image_ids, self._image_ids):
            bboxes = []
            categories = []
            joints = []
            joints_vis = []
            for box in all_boxes:
                if box['image_id'] == coco_image_id:
                    bbox = box['bbox']
                    bbox[2] = bbox[2] + bbox[0]
                    bbox[3] = bbox[3] + bbox[1]
                    bboxes.append(bbox)
                    category = box['category_id']
                    categories.append(category)
                    joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                    joints_3d_vis = np.ones(
                        (self.num_joints, 3), dtype=np.float)
                    joints.append(joints_3d)
                    joints_vis.append(joints_3d_vis)
            bboxes = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:  # no instances
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
                self._joints[image_id] = np.zeros(
                    (0, self.num_joints, 3), dtype=np.float32)
                self._joints_vis[image_id] = np.zeros(
                    (0, self.num_joints, 3), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack(
                    (bboxes, categories[:, None]))
                self._joints[image_id] = np.stack(joints)
                self._joints_vis[image_id] = np.stack(joints_vis)

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat

    def get_points(self, joints_3d, joints_3d_vis):
        points = np.zeros([self.num_joints, 2])
        points[:, :] = None
        for i in range(self.num_joints):
            if joints_3d_vis[i, 0] and joints_3d_vis[i, 1]:
                points[i, 0] = joints_3d[i, 0]
                points[i, 1] = joints_3d[i, 1]
        return points

    def draw_joints(self, image, joints, marker_size):

        colors = [BLUE, YELLOW, RED, GREEN, PURPLE, CYAN]
        for j in range(joints.shape[0]):
            pos = joints[j, :]
            if not math.isnan(pos[0]):
                cp = colors[(j - 1) % len(colors)]
                pos = pos.astype(int)
                image = cv2.circle(image, tuple(pos), marker_size, cp, -1)
                image = cv2.putText(image, '{}'.format(j),
                                    (pos[0], pos[1] - 2), cv2.FONT_HERSHEY_PLAIN, 3, cp, 3)

        return image


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
