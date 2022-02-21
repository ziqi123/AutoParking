from PIL import Image
from PIL import ImageEnhance
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import random
import cv2
import math
np.seterr(divide='ignore', invalid='ignore')
plt.switch_backend('agg')


class heatmap_dataset(Dataset):
    def __init__(self, ds_dir, sigma, setname='train', transform=None, norm_factor=256, rgb2gray=False, resize=True):

        self.ds_dir = ds_dir
        self.setname = setname
        self.transform = transform
        self.norm_factor = norm_factor
        self.rgb2gray = rgb2gray
        self.__sigma = sigma

        self.resize = resize
        self.c = 0
        self.s = 0
        self.r = 0

        if setname == 'train':
            data = []
            gt = []
            train_list = '/media/home_bak/ziqi/park/stackedHourglass_256/dataset/train.txt'
            f = open(train_list)
            for line in f:
                line_data = line.strip('\n')
                line_gt = line_data.replace(
                    'img_256', 'annt_256').replace('.jpg', '_OA.txt')
                data.append(line_data)
                gt.append(line_gt)
            self.data = data
            self.gt = gt

        if setname == 'val':
            data = []
            gt = []
            test_list = '/media/home_bak/ziqi/park/stackedHourglass_256/dataset/val_correct.txt'
            # test_list = '/media/home_bak/ziqi/park/stackedHourglass_256/dataset/train.txt'
            f = open(test_list)
            for line in f:
                line_data = line.strip('\n')
                line_gt = line_data.replace(
                    'img_256', 'annt_256').replace('.jpg', '_OA.txt')
                data.append(line_data)
                gt.append(line_gt)
            self.data = data
            self.gt = gt

    def __len__(self):
        return len(self.data)

    def get_affine_transform(self, center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            print(scale)
            scale = np.array([scale, scale])

        scale_tmp = scale * 200
        # print('scale_tmp',scale_tmp)
        # print("scale_tmp: {}".format(scale_tmp))

        # print("output_size: {}".format(output_size)) # W H
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)
        # print("src_dir: {}".format(src_dir))
        # print("dst_dir: {}".format(dst_dir))

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        # print("center: {}".format(center))
        src[0, :] = center + scale_tmp * shift
        # print("src[0, :]: {}".format(src[0, :]))

        # print("src_dir: {}".format(src_dir))
        src[1, :] = center + src_dir + scale_tmp * shift

        # print("src[1, :]: {}".format(src[1, :]))
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        # print("src[2:, :]: {}".format(src[2:, :]))
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])
        # print('src', src,dst)
        # print("src:\n{}".format(src))
        # print("dst:\n{}".format(dst))

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        # exit(1)

        return trans

    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def _box2cs(self, size, aspect_ratio=None, scale_factor=None):
        x, y, w, h = 0, 0, size[0], size[1]
        return self._xywh2cs(x, y, w, h,
                             aspect_ratio,
                             scale_factor)

    def _xywh2cs(self, x, y, w, h, aspect_ratio, scale_factor):
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

        return center, scale

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        # Read images
        # data = Image.open(str(self.data[item]))
        data = cv2.imread(str(self.data[item]))
        imgPath = str(self.data[item])

        gt = [[0, 0], [0, 0], [0, 0], [0, 0]]

        # gt = np.loadtxt(str(self.data[item]))
        with open(str(self.gt[item]), "r") as f:
            lines = f.readlines()  # 把全部数据文件读到一个列表lines中

        # 表示矩阵的行，从0行开始
        row = 0
        # 把lines中的数据逐行读取出来
        for line in lines:
            # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
            list = line.strip('\n').split(' ')
            gt[row][0] = float(list[0])
            gt[row][1] = float(list[1])
            # print("point:", list[0], list[1])
            # 然后方阵A的下一行接着读
            row = row + 1

        # gt.sort(key=takeSecond)
        # print("file", imgPath)

        H, W = 256, 256

        # 数据增强
        # data = self.randomBlur(data)
        # data = self.RandomBrightness(data)
        # data = self.RandomHue(data)
        # data = self.RandomSaturation(data)
        # # data = self.randomColor(data)
        # data = self.randomGaussian(data, mean=0.2, sigma=0.3)
        # if self.setname == 'train':
        #     # data = self.randomBlur(data)
        #     data = self.RandomBrightness(data)
        #     data = self.RandomHue(data)
        #     data = self.RandomSaturation(data)
        #     # data = self.randomColor(data)
        #     data = self.randomGaussian(data, mean=0.2, sigma=0.3)

        data = 255 * np.array(data).astype('uint8')
        data = cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)  # PIL转cv2

        # data = self.random_rotate(data)

        if self.rgb2gray:
            t = torchvision.transforms.Grayscale(1)
            data = t(data)

        # Convert to numpy
        data = np.array(data, dtype=np.float32) / self.norm_factor
        gt = np.array(gt, dtype=np.float32)

        size = [256, 256]
        mask = np.zeros((size[0], size[1]), dtype=np.float)

        H, W = 64, 64

        heatmaps = self._putGaussianMaps(gt/4.0, H, W, 1, self.__sigma)

        heatmaps = heatmaps.astype(np.float32)

        vectormap = self.get_vectormap(gt/4.0, H, W)

        vectormap = np.transpose(vectormap, [2, 0, 1])

        # paf_afterabs = np.abs(vectormap)
        # paf_aftersum = np.sum(paf_afterabs, 0)
        # heatmaps = np.sum(heatmaps, 0)
        # plt.subplot(4, 3, 1)
        # plt.imshow(data)
        # plt.subplot(4, 3, 2)
        # plt.imshow(paf_aftersum)
        # plt.subplot(4, 3, 3)
        # plt.imshow(heatmaps)
        # plt.savefig(
        #     "/media/home_bak/ziqi/park/stackedHourglass_256/paf.png")

        c, s = self._box2cs(size, aspect_ratio=1)
        r = 0
        # print(r)
        trans = self.get_affine_transform(c, s, r, size)
        data = cv2.warpAffine(
            data, trans, (size[0], size[1]), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(
            mask, trans, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderValue=255)

        # Expand dims into Pytorch format
        data = np.transpose(data, (2, 0, 1))
        mask = np.expand_dims(mask, 0)

        # Convert to Pytorch Tensors
        data = torch.tensor(data, dtype=torch.float)
        gt = torch.tensor(gt, dtype=torch.float32)
        # print("gt,imgPath:", gt, imgPath)
        mask = torch.tensor(mask, dtype=torch.float)

        return data, gt, mask, item, imgPath, heatmaps, vectormap

    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(
            image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(
            color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        contrast_image = ImageEnhance.Contrast(
            brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        # 调整图像锐度
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

    def random_blur(self, image, sigma):
        _sigma = random.randint(0, sigma)

        return cv2.GaussianBlur(image, (11, 11), _sigma)

    def randomGaussian(self, image, mean, sigma):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """
        def gaussianNoisy(im, mean, sigma):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im
        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:]+boxes[:, :2])/2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width*0.2, width*0.2)
            shift_y = random.uniform(-height*0.2, height*0.2)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(
                    shift_x):, :] = bgr[:height-int(shift_y), :width-int(shift_x), :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height+int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, : width-int(shift_x), :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, : width+int(shift_x),
                                  :] = bgr[: height-int(shift_y), -int(shift_x):, :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[: height+int(shift_y), : width+int(
                    shift_x), :] = bgr[-int(shift_y):, -int(shift_x):, :]

            shift_xy = torch.FloatTensor(
                [[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor(
                [[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def randomScale(self, bgr, boxes):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width*scale), height))
            scale_tensor = torch.FloatTensor(
                [[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:]+boxes[:, :2])/2
            height, width, c = bgr.shape
            h = random.uniform(0.6*height, height)
            w = random.uniform(0.6*width, width)
            x = random.uniform(0, width-w)
            y = random.uniform(0, height-h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if(len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h, x:x+w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im

    def _putGaussianMap(self, center, crop_size_y, crop_size_x, stride, sigma):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:
        """
        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        start = stride / 2.0 - 0.5
        y_range = [i for i in range(int(grid_y))]
        x_range = [i for i in range(int(grid_x))]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap

    def _putGaussianMaps(self, keypoints, crop_size_y, crop_size_x, stride, sigma):
        """

        :param keypoints: (15,2)
        :param crop_size_y: int
        :param crop_size_x: int
        :param stride: int
        :param sigma: float
        :return:
        """
        all_keypoints = keypoints
        point_num = all_keypoints.shape[0]
        heatmaps_this_img = []
        for k in range(point_num):
            heatmap = self._putGaussianMap(
                all_keypoints[k], crop_size_y, crop_size_x, stride, sigma)
            heatmap = heatmap[np.newaxis, ...]
            heatmaps_this_img.append(heatmap)
        # (num_joint,crop_size_y/stride,crop_size_x/stride)
        heatmaps_this_img = np.concatenate(heatmaps_this_img, axis=0)
        return heatmaps_this_img

    def random_rotate(self, image, angle=5.):
        angle = np.random.uniform(-angle, angle)

        h, w, _ = image.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image = cv2.warpAffine(
            image, m, (w, h), borderMode=cv2.BORDER_REFLECT101)

        return image

    def random_distort(self, image, hue, saturation, exposure):
        # determine scale factors
        dhue = np.random.uniform(-hue, hue)
        dsat = np.random.uniform(1. / saturation, saturation)
        dexp = np.random.uniform(1. / exposure, exposure)

        image = 255 * np.array(image).astype('uint8')
        # convert RGB space to HSV space
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')

        # change satuation and exposure
        image[:, :, 1] *= dsat
        image[:, :, 2] *= dexp

        # change hue
        image[:, :, 0] += dhue

        image[:, :, 0] = np.clip(image[:, :, 0], 0., 179.)
        image[:, :, 1] = np.clip(image[:, :, 1], 0., 255.)
        image[:, :, 2] = np.clip(image[:, :, 2], 0., 255.)

        # convert back to RGB from HSV
        return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)

    def random_add_smu(self, image):
        _smu = cv2.imread(
            '/media/home_bak/ziqi/park/Hourglass_nocrop/noise/smu.jpg')
        if random.randint(0, 1):
            rows = random.randint(0, _smu.shape[0] - image.shape[0])
            cols = random.randint(0, _smu.shape[1] - image.shape[1])
            add_smu = _smu[rows:rows + image.shape[0],
                           cols:cols + image.shape[1]]
            image = cv2.bitwise_not(image)
            image = cv2.bitwise_and(add_smu, image)
            image = cv2.bitwise_not(image)
        return image

    def get_heatmap(self, keypoints, target_size, sigma):
        heatmap = np.zeros(
            (4, 256, 256), dtype=np.float32)
        # 全部heatmap都初始化为0
        point_num = keypoints.shape[0]
        # for k in range(point_num):
        #     if keypoints[k][0] < 0 or keypoints[k][1] < 0:
        #         continue
        #     self.put_heatmap(heatmap, k, keypoints[k], sigma)
        # print("keypoints", keypoints)
        for joints in keypoints:
            # print("joints", joints)
            for idx in range(len(joints)):
                # print("point", point)
                self.put_heatmap(heatmap, idx, joints, sigma)

        heatmap = heatmap.transpose((1, 2, 0))
        # background
        heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)
        if target_size:
            heatmap = cv2.resize(heatmap, target_size,
                                 interpolation=cv2.INTER_AREA)  # 插值resize
        return heatmap.astype(np.float16)

    def put_heatmap(self, heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]
        th = 4.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2  # 高斯！！
                exp = d / 2.0 / sigma / sigma  # 高斯！！
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(
                    heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    def get_vectormap(self, joint, height, width):
        """
        功能： 生成OpenPose的向量图(PAF标签), 因为每两个关键点的连线有两个方向(x-axis, y-axis), vectormap是heatmap的2倍
        :param joint: 已标注的真实2D关键点坐标
        :param sign:
        :return:
        """
        num_joints = 4
        limb = list(zip([1, 2, 3, 4],
                        [2, 3, 4, 1]))
        # vectormap = np.zeros(
        #     (len(self.shuffle_ref)*2, self.output_size[0], self.output_size[1]), dtype=np.float32)
        # countmap = np.zeros(
        #     (len(self.shuffle_ref), self.output_size[0], self.output_size[1]), dtype=np.int16)

        vectormap = np.zeros((num_joints * 2, height, width), dtype=np.float32)
        counter = np.zeros((num_joints, height, width), dtype=np.int16)
        for i, (a, b) in enumerate(limb):
            a -= 1
            b -= 1

            v_start = joint[a]
            v_end = joint[b]
            # exclude invisible or unmarked point
            if v_start[0] < -100 or v_start[1] < -100 or v_end[0] < -100 or v_end[1] < -100:
                continue
            self.put_vectormap(
                vectormap, counter, i, v_start, v_end)

        vectormap = vectormap.transpose((1, 2, 0))
        nonzeros = np.nonzero(counter)
        for p, y, x in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
            if counter[p][y][x] <= 0:
                continue
            vectormap[y][x][p * 2 + 0] /= counter[p][y][x]
            vectormap[y][x][p * 2 + 1] /= counter[p][y][x]
        return vectormap.astype(np.float16)

    def put_vectormap(self, vectormap, countmap, plane_idx, center_from, center_to, threshold=1):
        """
        功能： 计算每个两个关节点向量的x,y方向上的map
        :param vectormap:
        :param countmap:
        :param plane_idx: 关节点索引
        :param center_from: 向量终点
        :param center_to: 向量起点
        :param threshold: 向量叉乘的范围限定阈值
        :return:
        """
        _, height, width = vectormap.shape[:3]

        vec_x = center_to[0] - center_from[0]
        vec_y = center_to[1] - center_from[1]
        min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
        min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))
        max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
        max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

        norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
        if norm == 0:
            return
        vec_x /= norm
        vec_y /= norm
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                bec_x = x - center_from[0]
                bec_y = y - center_from[1]
                dist = abs(bec_x * vec_y - bec_y * vec_x)

                if dist > threshold:
                    continue
                countmap[plane_idx][y][x] += 1
                vectormap[plane_idx * 2 + 0][y][x] = vec_x
                vectormap[plane_idx * 2 + 1][y][x] = vec_y


class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res / 48
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape=(self.num_parts, self.output_res,
                       self.output_res), dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            if pt[0] > 0:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue
                ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
                br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(
                    hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms
