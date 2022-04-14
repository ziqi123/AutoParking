import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .matcher import build_matcher
import copy
from sample.vis import save_debug_images_training, save_debug_images_joints, viz_infer_joints
from .position_encoding import build_position_encoding
from .transformer import build_transformer
from .detr_loss import SetCriterion
from .utils import NestedTensor, nested_tensor_from_tensor_list, MLP

BN_MOMENTUM = 0.1


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class kp(nn.Module):
    def __init__(self,
                 flag=False,
                 block=None,
                 layers=None,
                 res_dims=None,
                 res_strides=None,
                 attn_dim=None,
                 num_queries=None,
                 aux_loss=None,
                 pos_type=None,
                 drop_out=0.1,
                 num_heads=None,
                 dim_feedforward=None,
                 enc_layers=None,
                 dec_layers=None,
                 pre_norm=None,
                 return_intermediate=None,
                 kps_dim=None,
                 cls_dim=None,
                 mlp_layers=None,
                 norm_layer=FrozenBatchNorm2d,
                 level=["layer1", "layer2", "layer3", "layer4"],
                 x_res=10,
                 y_res=10):

        super(kp, self).__init__()
        self.flag = flag

        # above all waste not used
        self.norm_layer = norm_layer

        self.inplanes = res_dims[0]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, res_dims[0], layers[0], stride=res_strides[0])
        self.layer2 = self._make_layer(
            block, res_dims[1], layers[1], stride=res_strides[1])
        self.layer3 = self._make_layer(
            block, res_dims[2], layers[2], stride=res_strides[2])
        self.layer4 = self._make_layer(
            block, res_dims[3], layers[3], stride=res_strides[3])

        hidden_dim = attn_dim
        self.aux_loss = aux_loss
        self.position_embedding = build_position_encoding(
            hidden_dim=hidden_dim, type=pos_type)
        # self.position_embedding2 = self.build_pe
        # self.position_embedding = self.get_leant_pe
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 100 h
        self.num_queries = num_queries
        # the same as channel of self.layer4
        # print('res_dims[-1]', res_dims[-1])
        # 512
        input_dim = res_dims[0]+res_dims[1]+res_dims[2]+res_dims[3]
        # print('input_dim', input_dim)
        self.input_proj = nn.Conv2d(res_dims[-1], attn_dim, kernel_size=1)
        self.input_proj2 = nn.Conv2d(input_dim, attn_dim, kernel_size=1)

        self.transformer = build_transformer(hidden_dim=hidden_dim,
                                             dropout=drop_out,
                                             nheads=num_heads,
                                             dim_feedforward=dim_feedforward,
                                             enc_layers=enc_layers,
                                             dec_layers=dec_layers,
                                             pre_norm=pre_norm,
                                             return_intermediate_dec=return_intermediate)
        self.class_embed = nn.Linear(hidden_dim, cls_dim)  # 9 subclasses
        self.joints_embed = MLP(hidden_dim, hidden_dim,
                                kps_dim, mlp_layers)  # 5 keypoints 5 * 2
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # [1, x_res], ANNOT ?(1)
        # 返回（-1.25～1.25）间x_res个均匀间隔的值
        x_interpolate = torch.linspace(-1.25, 1.25,
                                       x_res, requires_grad=False).unsqueeze(0)
        y_interpolate = torch.linspace(-1.25, 1.25, y_res,
                                       requires_grad=False).unsqueeze(0)  # [1, y_res]
        self.register_buffer("x_interpolate", x_interpolate)
        self.register_buffer("y_interpolate", y_interpolate)
        self.x_res = x_res
        self.y_res = y_res
        self.level = level
        # [1, y_res, x_res]
        mask = torch.zeros(1, y_res, x_res, requires_grad=False)
        self.register_buffer("mask", mask)
        self.build_pe()
        self.coord_predictor = MLP(hidden_dim, hidden_dim, 10, num_layers=3)

    def build_pe(self):
        # fixed sine pe
        not_mask = 1 - self.mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        scale = 2 * math.pi  # normalize?
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        num_pos_feats = 32
        temperature = 10000
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=self.mask.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # print('x_embed[:, :, :, None]', x_embed[:, :, :, None].shape)
        # print('dim_t', dim_t.shape)
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # print('pos_x', pos_x.shape)
        # print('pos_y', pos_y.shape)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.register_buffer("pe", pos)

        # learnable pe
        self.row_embed = nn.Embedding(num_pos_feats, self.x_res)
        self.col_embed = nn.Embedding(num_pos_feats, self.y_res)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def get_leant_pe(self):
        y_embed = self.col_embed.weight.unsqueeze(
            -1).expand(-1, -1, self.x_res)
        x_embed = self.row_embed.weight.unsqueeze(1).expand(-1, self.y_res, -1)
        # print('x_embed', x_embed.shape)
        # [128, 10, 10]
        # print('y_embed', y_embed.shape)
        # [128, 10, 10]
        embed = torch.cat([y_embed, x_embed], dim=0).unsqueeze(0)
        return embed

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _train(self, *xs, **kwargs):
        """ This performs the PSTR forward computation.
            Parameters:
                 xs: [batch_images, batch_inmasks]
            Outputs: list of dicts. The expected keys in each dict depends on the losses applied.
        """
        B, num_roi, channel, roi_height, roi_width = xs[0].shape
        images = xs[0].view(-1, channel, roi_height, roi_width)
        masks = xs[1].view(-1, 1, roi_height, roi_width)
        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p1 = self.layer1(p)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        p = p4
        # print('p', p.shape)
        # [16, 512, 8, 8]

        masks_p = F.interpolate(
            masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]  # B, 8, 8

        pos = self.position_embedding(p, masks_p)
        # print('self.input_proj(p)', self.input_proj(p).shape)[16, 128, 8, 8]
        # print('masks_p', masks_p.shape)[16, 8, 8]
        # print('pos', pos.shape)[16, 128, 8, 8]
        # print('self.query_embed.weight', self.query_embed.weight.shape)
        # [100, 128]
        hs = self.transformer(self.input_proj(p), masks_p,
                              self.query_embed.weight, pos)[0][-1]
        # print('hs', hs.shape)
        # [16, 100, 128]
        outputs_class = self.class_embed(hs)
        # outputs_joints = self.joints_embed(hs).sigmoid()
        bboxes = self.bbox_embed(hs).sigmoid()
        out = {'pred_box': bboxes,
               'pred_class': outputs_class}
        # print('outputs_class', outputs_class.shape)
        # [16, 100, 3]
        # print('bboxes', bboxes.shape)
        # [16, 100, 4]

        # out = {'pred_boxes': bboxes,
        #        'pred_classes': outputs_class}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(
        #         bboxes, outputs_class)

        sample = nested_tensor_from_tensor_list([images, masks])
        features1 = nested_tensor_from_tensor_list([p1, ])
        features2 = nested_tensor_from_tensor_list([p2, ])
        features3 = nested_tensor_from_tensor_list([p3, ])
        features4 = nested_tensor_from_tensor_list([p4, ])

        layer = {}
        layer["layer1"] = features1
        layer["layer2"] = features2
        layer["layer3"] = features3
        layer["layer4"] = features4

        # some preperation for STN feature cropping
        person_per_image = hs.size(1)
        # print('person_per_image', person_per_image)
        # [100]
        num_person = person_per_image * hs.size(0)
        # print('num_person', num_person)
        # [1600]
        heights, widths = sample.get_shape().unbind(-1)  # [B] * 2
        # print('sample.get_shape()', sample.get_shape().shape)
        # [16, 2]
        # print('heights', heights.shape)
        # [16]
        # print('widths', widths.shape)
        # [16]
        # 重复张量的元素,person_per_image为每个元素的重复次数
        rh = heights.repeat_interleave(
            person_per_image)  # [person per image * B]
        # print('rh', rh.shape)
        # [1600]
        rw = widths.repeat_interleave(
            person_per_image)  # [person per image * B]
        # print('rw', rw.shape)
        # [1600]
        srcs = [layer[_].decompose()[0] for _ in self.level]
        # print('srcs', srcs[0][0].shape)
        # [64, 64, 64]
        # print('srcs[0]', srcs[0].shape)
        # [16*64*64*64]
        # print('srcs[1]', srcs[1].shape)
        # [16*128*32*32]
        # print('srcs[2]', srcs[2].shape)
        # [16*256*16*16]

        # [person per image * B] * 4
        cx, cy, w, h = bboxes.flatten(
            end_dim=1).unbind(-1)
        # print('cx, cy, w, h', cx[0], cy[0], w[0], h[0])
        # [1600]
        cx, cy, w, h = cx * rw, cy * rh, w * rw, h * rh  # ANNOT (1)

        # STN cropping
        y_grid = (h.unsqueeze(-1) @ self.y_interpolate + cy.unsqueeze(-1) * 2 -
                  1).unsqueeze(-1).unsqueeze(-1)  # [person per image * B, y_res, 1, 1]
        # print('y_grid', y_grid.shape)
        # [1600, 10, 1, 1]
        x_grid = (w.unsqueeze(-1) @ self.x_interpolate + cx.unsqueeze(-1) * 2 -
                  1).unsqueeze(-1).unsqueeze(1)  # [person per image * B, 1, x_res, 1]
        # print('x_grid', x_grid.shape)
        # torch.Size([1600, 1, 10, 1])
        grid = torch.cat([x_grid.expand(-1, self.y_res, -1, -1),
                         y_grid.expand(-1, -1, self.x_res, -1), ],
                         dim=-1)
        # print('grid', grid.shape)
        # torch.Size([800, 10, 10, 2])
        cropped_feature = []
        cropped_pos = []
        for j, l in enumerate(self.level):
            # print('srcs[j][0].expand(num_person, -1, -1, -1)',
            #       srcs[j][0].expand(num_person, -1, -1, -1).shape)
            # print('grid', grid.shape)
            # (input, grid):根据grid中每个位置提供的坐标信息(这里指input中pixel的坐标)，将input中对应位置的像素值填充到grid指定的位置，得到最终的输出。
            f1 = F.grid_sample(srcs[j][0].expand(
                num_person, -1, -1, -1), grid, padding_mode="border")  # [person per image * B, C, y_res, x_res]
            cropped_feature.append(f1)
        # print('cropped_feature0', cropped_feature[0].shape)
        # [1600, 64, 10, 10]
        # print('cropped_feature1', cropped_feature[1].shape)
        # [1600, 128, 10, 10]
        # print('cropped_feature2', cropped_feature[2].shape)
        # [1600, 256, 10, 10]
        # print('cropped_feature3', cropped_feature[3].shape)
        # [1600, 512, 10, 10]
        cropped_feature = torch.cat(cropped_feature, dim=1)
        # print('cropped_feature', cropped_feature.shape)
        # [1600, 960, 10, 10]
        cropped_pos.append(self.pe.expand(num_person, -1, -1, -1))
        # print('cropped_pos0', cropped_pos[0].shape)
        # ([1600, 256, 10, 10])
        cropped_pos.append(self.get_leant_pe().expand(num_person, -1, -1, -1))
        # print('cropped_pos1', cropped_pos[1].shape)
        cropped_pos = torch.cat(cropped_pos, dim=1)
        # print('cropped_pos', cropped_pos.shape)
        # [1600, 512, 10, 10]
        mask = self.mask.bool().expand(num_person, -1, -1)  # ANNOT (2)

        # print('mask', mask.shape)
        # [1600, 10, 10]

        # print('self.input_proj2(cropped_feature)', self.input_proj2(
        #     cropped_feature).shape)
        # [1600, 128, 10, 10]
        # print('self.query_embed.weight',
        #       self.query_embed.weight.shape)
        # [100, 128]
        j_embed = self.transformer(
            self.input_proj2(cropped_feature), mask, self.query_embed.weight, cropped_pos)[0][-1]  # [B, num queries, hidden dim]
        # print('j_embed', j_embed.shape)
        # [800, 50, 128]
        j_coord_ = self.coord_predictor(j_embed).sigmoid()
        # [800, 50, 2]
        # print('j_coord_', j_coord_.shape)
        # [B, Q] * 2
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = j_coord_.unbind(-1)
        # boxes:16, 50, 4
        # print('x, y', x.shape, y.shape)
        # [800,50]
        # print('bboxes[:,:, 0]', bboxes[:, :, 0].shape)
        # print('bboxes[:,:, 0].unsqueeze(-1)',
        #       bboxes[:, :, 0].unsqueeze(-1).shape)
        # print('bboxes.flatten(end_dim=1)', bboxes.flatten(end_dim=1).shape)
        # [16, 4, 1]
        bboxes = bboxes.flatten(end_dim=1)

        x1 = (x1 * 1.25 - 0.625) * \
            bboxes[:, 2].unsqueeze(-1) + bboxes[:, 0].unsqueeze(-1)
        y1 = (y1 * 1.25 - 0.625) * \
            bboxes[:, 3].unsqueeze(-1) + bboxes[:, 1].unsqueeze(-1)

        x2 = (x2 * 1.25 - 0.625) * \
            bboxes[:, 2].unsqueeze(-1) + bboxes[:, 0].unsqueeze(-1)
        y2 = (y2 * 1.25 - 0.625) * \
            bboxes[:, 3].unsqueeze(-1) + bboxes[:, 1].unsqueeze(-1)

        x3 = (x3 * 1.25 - 0.625) * \
            bboxes[:, 2].unsqueeze(-1) + bboxes[:, 0].unsqueeze(-1)
        y3 = (y3 * 1.25 - 0.625) * \
            bboxes[:, 3].unsqueeze(-1) + bboxes[:, 1].unsqueeze(-1)

        x4 = (x4 * 1.25 - 0.625) * \
            bboxes[:, 2].unsqueeze(-1) + bboxes[:, 0].unsqueeze(-1)
        y4 = (y4 * 1.25 - 0.625) * \
            bboxes[:, 3].unsqueeze(-1) + bboxes[:, 1].unsqueeze(-1)

        x5 = (x5 * 1.25 - 0.625) * \
            bboxes[:, 2].unsqueeze(-1) + bboxes[:, 0].unsqueeze(-1)
        y5 = (y5 * 1.25 - 0.625) * \
            bboxes[:, 3].unsqueeze(-1) + bboxes[:, 1].unsqueeze(-1)

        # 限制在0-1之间
        x1 = x1.clamp(0, 1)
        x2 = x2.clamp(0, 1)
        x3 = x3.clamp(0, 1)
        x4 = x4.clamp(0, 1)
        x5 = x5.clamp(0, 1)
        # print('x1', x1.shape)
        # [800, 50]
        y1 = y1.clamp(0, 1)
        y2 = y2.clamp(0, 1)
        y3 = y3.clamp(0, 1)
        y4 = y4.clamp(0, 1)
        y5 = y5.clamp(0, 1)

        # 不同特征层的堆叠
        j_coord = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5], dim=-1)
        j_class = self.class_embed(j_embed)
        # print('j_coord', j_coord.shape)
        # [800, 50, 8]
        # print('j_class', j_class.shape)
        # [800, 50, 3]
        j_coord = j_coord.reshape(
            hs.size(0), -1, 10)
        j_class = j_class.reshape(hs.size(
            0), -1, 3)
        # print('j_coord', j_coord.shape, j_class.shape)
        # [16, 2500, 8]
        # [16, 2500, 3]

        out['pred_boxes'] = j_coord
        out['pred_classes'] = j_class

        return out

    def _test(self, *xs, **kwargs):
        B, num_roi, channel, roi_height, roi_width = xs[0].shape
        images = xs[0].view(-1, channel, roi_height, roi_width)
        masks = xs[1].view(-1, 1, roi_height, roi_width)

        # p = self.conv1(images)
        # p = self.bn1(p)
        # p = self.relu(p)
        # p = self.maxpool(p)
        # p = self.layer1(p)
        # p = self.layer2(p)
        # p = self.layer3(p)
        # p = self.layer4(p)
        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p1 = self.layer1(p)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        p = p4

        masks_p = F.interpolate(
            masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]  # B, 8, 8
        pos = self.position_embedding(p, masks_p)
        hs = self.transformer(self.input_proj(p), masks_p,
                              self.query_embed.weight, pos)[0]
        outputs_class = self.class_embed(hs)
        outputs_joints = self.joints_embed(hs).sigmoid()
        # out = {'pred_boxes': outputs_joints[-1],
        #        'pred_classes': outputs_class[-1]}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(
        #         outputs_joints, outputs_class)
        bboxes = self.bbox_embed(hs).sigmoid()
        sample = nested_tensor_from_tensor_list([images, masks])
        features1 = nested_tensor_from_tensor_list([p1, ])
        features2 = nested_tensor_from_tensor_list([p2, ])
        features3 = nested_tensor_from_tensor_list([p3, ])
        features4 = nested_tensor_from_tensor_list([p4, ])

        layer = {}
        layer["layer1"] = features1
        layer["layer2"] = features2
        layer["layer3"] = features3
        layer["layer4"] = features4

        # some preperation for STN feature cropping
        person_per_image = hs.size(1)
        # print('person_per_image', person_per_image)
        # [100]
        num_person = person_per_image * hs.size(0)
        # print('num_person', num_person)
        # [1600]
        heights, widths = sample.get_shape().unbind(-1)  # [B] * 2
        # print('sample.get_shape()', sample.get_shape().shape)
        # [16, 2]
        # print('heights', heights.shape)
        # [16]
        # print('widths', widths.shape)
        # [16]
        # 重复张量的元素,person_per_image为每个元素的重复次数
        rh = heights.repeat_interleave(
            person_per_image)  # [person per image * B]
        # print('rh', rh.shape)
        # [1600]
        rw = widths.repeat_interleave(
            person_per_image)  # [person per image * B]
        # print('rw', rw.shape)
        # [1600]
        srcs = [layer[_].decompose()[0] for _ in self.level]
        # print('srcs', srcs[0][0].shape)
        # [64, 64, 64]
        # print('srcs[0]', srcs[0].shape)
        # [16*64*64*64]
        # print('srcs[1]', srcs[1].shape)
        # [16*128*32*32]
        # print('srcs[2]', srcs[2].shape)
        # [16*256*16*16]

        # [person per image * B] * 4
        cx, cy, w, h, cx2, cy2, w2, h2, cx3, cy3, w3, h3, cx4, cy4, w4, h4, cx5, cy5, w5, h5 = bboxes.flatten(
            end_dim=1).unbind(-1)
        print('cx, cy, w, h', cx[0], cy[0], w[0], h[0])
        # [1600]
        cx, cy, w, h = cx * rw, cy * rh, w * rw, h * rh  # ANNOT (1)
        cx2, cy2, w2, h2 = cx2 * rw, cy2 * rh, w2 * rw, h2 * rh
        cx3, cy3, w3, h3 = cx3 * rw, cy3 * rh, w3 * rw, h3 * rh
        cx4, cy4, w4, h4 = cx4 * rw, cy4 * rh, w4 * rw, h4 * rh
        cx5, cy5, w5, h5 = cx5 * rw, cy5 * rh, w5 * rw, h5 * rh

        # STN cropping
        y_grid = (h.unsqueeze(-1) @ self.y_interpolate + cy.unsqueeze(-1) * 2 -
                  1).unsqueeze(-1).unsqueeze(-1)  # [person per image * B, y_res, 1, 1]·
        y_grid2 = (h2.unsqueeze(-1) @ self.y_interpolate + cy2.unsqueeze(-1) * 2 -
                   1).unsqueeze(-1).unsqueeze(-1)  # [person per image * B, y_res, 1, 1]·
        y_grid3 = (h3.unsqueeze(-1) @ self.y_interpolate + cy3.unsqueeze(-1) * 2 -
                   1).unsqueeze(-1).unsqueeze(-1)  # [person per image * B, y_res, 1, 1]·
        y_grid4 = (h4.unsqueeze(-1) @ self.y_interpolate + cy4.unsqueeze(-1) * 2 -
                   1).unsqueeze(-1).unsqueeze(-1)  # [person per image * B, y_res, 1, 1]·
        y_grid5 = (h5.unsqueeze(-1) @ self.y_interpolate + cy5.unsqueeze(-1) * 2 -
                   1).unsqueeze(-1).unsqueeze(-1)  # [person per image * B, y_res, 1, 1]·
        # print('y_grid', y_grid.shape)
        # [1600, 10, 1, 1]
        x_grid = (w.unsqueeze(-1) @ self.x_interpolate + cx.unsqueeze(-1) * 2 -
                  1).unsqueeze(-1).unsqueeze(1)  # [person per image * B, 1, x_res, 1]
        x_grid2 = (w2.unsqueeze(-1) @ self.x_interpolate + cx2.unsqueeze(-1) * 2 -
                   1).unsqueeze(-1).unsqueeze(1)  # [person per image * B, 1, x_res, 1]
        x_grid3 = (w3.unsqueeze(-1) @ self.x_interpolate + cx3.unsqueeze(-1) * 2 -
                   1).unsqueeze(-1).unsqueeze(1)  # [person per image * B, 1, x_res, 1]
        x_grid4 = (w4.unsqueeze(-1) @ self.x_interpolate + cx4.unsqueeze(-1) * 2 -
                   1).unsqueeze(-1).unsqueeze(1)  # [person per image * B, 1, x_res, 1]
        x_grid5 = (w5.unsqueeze(-1) @ self.x_interpolate + cx5.unsqueeze(-1) * 2 -
                   1).unsqueeze(-1).unsqueeze(1)  # [person per image * B, 1, x_res, 1]
        # print('x_grid', x_grid.shape)
        # torch.Size([1600, 1, 10, 1])
        grid = torch.cat([x_grid.expand(-1, self.y_res, -1, -1),
                         y_grid.expand(-1, -1, self.x_res, -1), ],
                         dim=-1)
        grid2 = torch.cat([x_grid2.expand(-1, self.y_res, -1, -1),
                           y_grid2.expand(-1, -1, self.x_res, -1), ],
                          dim=-1)
        grid3 = torch.cat([x_grid3.expand(-1, self.y_res, -1, -1),
                           y_grid3.expand(-1, -1, self.x_res, -1), ],
                          dim=-1)
        grid4 = torch.cat([x_grid4.expand(-1, self.y_res, -1, -1),
                           y_grid4.expand(-1, -1, self.x_res, -1), ],
                          dim=-1)
        grid5 = torch.cat([x_grid5.expand(-1, self.y_res, -1, -1),
                           y_grid5.expand(-1, -1, self.x_res, -1), ],
                          dim=-1)
        print('grid', grid.shape)
        # torch.Size([800, 10, 10, 2])
        cropped_feature = []
        cropped_pos = []
        for j, l in enumerate(self.level):
            # print('srcs[j][0].expand(num_person, -1, -1, -1)',
            #       srcs[j][0].expand(num_person, -1, -1, -1).shape)
            # print('grid', grid.shape)
            # (input, grid):根据grid中每个位置提供的坐标信息(这里指input中pixel的坐标)，将input中对应位置的像素值填充到grid指定的位置，得到最终的输出。
            f1 = F.grid_sample(srcs[j][0].expand(
                num_person, -1, -1, -1), grid, padding_mode="border")  # [person per image * B, C, y_res, x_res]
            f2 = F.grid_sample(srcs[j][0].expand(
                num_person, -1, -1, -1), grid2, padding_mode="border")
            f3 = F.grid_sample(srcs[j][0].expand(
                num_person, -1, -1, -1), grid3, padding_mode="border")
            f4 = F.grid_sample(srcs[j][0].expand(
                num_person, -1, -1, -1), grid4, padding_mode="border")
            f5 = F.grid_sample(srcs[j][0].expand(
                num_person, -1, -1, -1), grid5, padding_mode="border")
            cropped_feature.append(f1+f2+f3+f4+f5)
        # print('cropped_feature0', cropped_feature[0].shape)
        # [1600, 64, 10, 10]
        # print('cropped_feature1', cropped_feature[1].shape)
        # [1600, 128, 10, 10]
        # print('cropped_feature2', cropped_feature[2].shape)
        # [1600, 256, 10, 10]
        # print('cropped_feature3', cropped_feature[3].shape)
        # [1600, 512, 10, 10]
        cropped_feature = torch.cat(cropped_feature, dim=1)
        # print('cropped_feature', cropped_feature.shape)
        # [1600, 960, 10, 10]
        cropped_pos.append(self.pe.expand(num_person, -1, -1, -1))
        # print('cropped_pos0', cropped_pos[0].shape)
        # ([1600, 256, 10, 10])
        cropped_pos.append(self.get_leant_pe().expand(num_person, -1, -1, -1))
        # print('cropped_pos1', cropped_pos[1].shape)
        cropped_pos = torch.cat(cropped_pos, dim=1)
        # print('cropped_pos', cropped_pos.shape)
        # [1600, 512, 10, 10]
        mask = self.mask.bool().expand(num_person, -1, -1)  # ANNOT (2)

        # print('mask', mask.shape)
        # [1600, 10, 10]

        # print('self.input_proj2(cropped_feature)', self.input_proj2(
        #     cropped_feature).shape)
        # [1600, 128, 10, 10]
        # print('self.query_embed.weight',
        #       self.query_embed.weight.shape)
        # [100, 128]
        j_embed = self.transformer(
            self.input_proj2(cropped_feature), mask, self.query_embed.weight, cropped_pos)[0][-1]  # [B, num queries, hidden dim]
        # print('j_embed', j_embed.shape)
        # [800, 50, 128]
        j_coord_ = self.coord_predictor(j_embed).sigmoid()
        # [800, 50, 2]
        # print('j_coord_', j_coord_.shape)
        # [B, Q] * 2
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = j_coord_.unbind(-1)
        # boxes:16, 50, 4
        # print('x, y', x.shape, y.shape)
        # [800,50]
        # print('bboxes[:,:, 0]', bboxes[:, :, 0].shape)
        # print('bboxes[:,:, 0].unsqueeze(-1)',
        #       bboxes[:, :, 0].unsqueeze(-1).shape)
        # print('bboxes.flatten(end_dim=1)', bboxes.flatten(end_dim=1).shape)
        # [16, 4, 1]
        bboxes = bboxes.flatten(end_dim=1)

        x1 = (x1 * 1.25 - 0.625) * \
            bboxes[:, 2].unsqueeze(-1) + bboxes[:, 0].unsqueeze(-1)
        y1 = (y1 * 1.25 - 0.625) * \
            bboxes[:, 3].unsqueeze(-1) + bboxes[:, 1].unsqueeze(-1)

        x2 = (x2 * 1.25 - 0.625) * \
            bboxes[:, 6].unsqueeze(-1) + bboxes[:, 4].unsqueeze(-1)
        y2 = (y2 * 1.25 - 0.625) * \
            bboxes[:, 7].unsqueeze(-1) + bboxes[:, 5].unsqueeze(-1)

        x3 = (x3 * 1.25 - 0.625) * \
            bboxes[:, 10].unsqueeze(-1) + bboxes[:, 8].unsqueeze(-1)
        y3 = (y3 * 1.25 - 0.625) * \
            bboxes[:, 11].unsqueeze(-1) + bboxes[:, 9].unsqueeze(-1)

        x4 = (x4 * 1.25 - 0.625) * \
            bboxes[:, 14].unsqueeze(-1) + bboxes[:, 12].unsqueeze(-1)
        y4 = (y4 * 1.25 - 0.625) * \
            bboxes[:, 15].unsqueeze(-1) + bboxes[:, 13].unsqueeze(-1)

        x5 = (x5 * 1.25 - 0.625) * \
            bboxes[:, 18].unsqueeze(-1) + bboxes[:, 16].unsqueeze(-1)
        y5 = (y5 * 1.25 - 0.625) * \
            bboxes[:, 19].unsqueeze(-1) + bboxes[:, 17].unsqueeze(-1)

        # 限制在0-1之间
        x1 = x1.clamp(0, 1)
        x2 = x2.clamp(0, 1)
        x3 = x3.clamp(0, 1)
        x4 = x4.clamp(0, 1)
        x5 = x5.clamp(0, 1)
        # print('x1', x1.shape)
        # [800, 50]
        y1 = y1.clamp(0, 1)
        y2 = y2.clamp(0, 1)
        y3 = y3.clamp(0, 1)
        y4 = y4.clamp(0, 1)
        y5 = y5.clamp(0, 1)

        # 不同特征层的堆叠
        j_coord = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5], dim=-1)
        j_class = self.class_embed(j_embed)
        # print('j_coord', j_coord.shape)
        # [800, 50, 8]
        # print('j_class', j_class.shape)
        # [800, 50, 3]
        j_coord = j_coord.reshape(
            hs.size(0), -1, 10)
        j_class = j_class.reshape(hs.size(
            0), -1, 3)
        # print('j_coord', j_coord.shape, j_class.shape)
        # [16, 2500, 8]
        # [16, 2500, 3]

        out = {'pred_boxes': j_coord,
               'pred_classes': j_class}

        return out

    def forward(self, *xs, **kwargs):
        if self.flag:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

    @ torch.jit.unused
    def _set_aux_loss(self, outputs_coord, outputs_class):
        return [{'pred_boxes': b, 'pred_classes': c} for b, c in zip(outputs_coord[:-1], outputs_class[:-1])]


class AELoss(nn.Module):
    def __init__(self,
                 debug_path=None,
                 aux_loss=True,
                 dec_layers=2,
                 bsize=16,
                 ):
        super(AELoss, self).__init__()
        weight_dict = {'loss_ce': 1,
                       'loss_boxes': 3, }
        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ['cls', 'joints']
        self.debug_path = debug_path
        self.matcher = build_matcher(1, 1, 3, bsize)
        self.criterion = SetCriterion(num_classes=2,
                                      matcher=self.matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=0.1,
                                      losses=losses)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self,
                iteration,
                save,
                viz_split,
                outputs,
                targets, use_indices=True, threshold=0.7):
        """ This performs the PSTR loss computation.
            Parameters:
                 iteration: num of training iters
                 save: save visualization results or not
                 viz_split: save results from test dataset or valid dataset
                 outputs: [batch_images, batch_inmasks]
                 targets: [batch_images, pslot, pslot_ign]
                 use_indices: use matching results or not
            Outputs: list of dicts. The expected keys in each dict depends on the losses applied.
        """

        set_loss = 0
        # print('outputs', outputs['pred_boxes'].shape)
        # print('targets', np.array(targets).shape)
        # [16, 1, 3, 256, 256]
        loss_dict, indices = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        set_loss += sum(loss_dict[k] * weight_dict[k]
                        for k in loss_dict.keys() if k in weight_dict)

        # save intermediate results
        if save:
            which_stack = 0
            if use_indices == False:
                # inference
                save_dir = os.path.join(self.debug_path, viz_split)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_name = 'iter_{}_layer_{}'.format(iteration, which_stack)
                save_path = os.path.join(save_dir, save_name)
                with torch.no_grad():

                    # roi_images
                    gt_viz_inputs = targets[0]
                    raw_batch, num_roi, c, roi_h, roi_w = gt_viz_inputs.shape
                    gt_viz_inputs = gt_viz_inputs.view(-1, c, roi_h, roi_w)
                    pred_joints = roi_h * outputs['pred_boxes'].detach()
                    pred_classes = outputs['pred_classes']
                    pred = self.Softmax(pred_classes)
                    mask = pred[:, :, 1] > pred[:, :, 0]
                    pred_pslots = [joi[mas]
                                   for joi, mas in zip(pred_joints, mask)]
                    viz_tgt_joints = [
                        i.detach()*roi_h for i in targets[1:raw_batch+1]]
                    ign_joints = [
                        i.detach() * roi_h for i in targets[raw_batch + 1:]]

                    viz_infer_joints(gt_viz_inputs,
                                     viz_tgt_joints,
                                     None,
                                     joints_pred=pred_pslots,
                                     ign_joints=ign_joints,
                                     prefix=save_path)
            else:
                # validation
                which_stack = 0
                save_dir = os.path.join(self.debug_path, viz_split)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_name = 'iter_{}_layer_{}'.format(iteration, which_stack)
                save_path = os.path.join(save_dir, save_name)

                with torch.no_grad():
                    gt_viz_inputs = targets[0]
                    raw_batch, num_roi, c, roi_h, roi_w = gt_viz_inputs.shape
                    gt_viz_inputs = gt_viz_inputs.view(-1, c, roi_h, roi_w)
                    pred_joints = roi_h * outputs['pred_boxes'].detach()
                    l_b, l_s, pslot = self._get_src_permutation_idx(indices)

                    cls_b, cls_s, val_pslot = self._make_mask(
                        indices, l_b, l_s, targets, raw_batch, pslot)
                    viz_pred_joints = []
                    k_idx = 0
                    for i in range(raw_batch):
                        if i not in val_pslot:
                            viz_pred_joints.append(torch.tensor([]).cuda())
                        else:
                            viz_pred_joints.append(
                                pred_joints[i][cls_s[k_idx]])
                            k_idx = k_idx+1

                    viz_tgt_joints = [
                        i.detach()*roi_h for i in targets[1:raw_batch+1]]
                    ign_joints = [
                        i.detach()*roi_h for i in targets[raw_batch+1:]]

                    save_debug_images_joints(gt_viz_inputs,
                                             viz_tgt_joints,
                                             None,
                                             joints_pred=viz_pred_joints,
                                             ign_joints=ign_joints,
                                             prefix=save_path)

        return (set_loss.unsqueeze(0), loss_dict,)

    def _get_src_permutation_idx(self, indices):
        l_bidx = []
        l_sidx = []
        valid_pslot = []
        for i, src in enumerate(indices):
            # print(src)
            # print(src[0])
            if src[0][0] != -1:
                l_bidx.append(torch.full_like(src[0], i))
                l_sidx.append(src[0])
                valid_pslot.append(i)

        return l_bidx, l_sidx, valid_pslot

    def _make_mask(self, indices, l_b, l_s, targets, bs, pslot):
        lv_b = []
        lv_s = []
        val_pslot = copy.deepcopy(pslot)

        for id, (i, j) in enumerate(zip(l_b, l_s)):
            if targets[1+pslot[id]].shape[1]-1+targets[1+pslot[id]+bs].shape[1]-1 != i.shape[0]:
                print('contain discrete length between prediction and label')
                raise ValueError

            valid_length = targets[1+pslot[id]].shape[1]-1
            # ignore_length=targets[1+id+bs].shape[1]-1
            sort = indices[pslot[id]][-1]
            mask = sort < valid_length
            if valid_length == 0:
                val_pslot.remove(pslot[id])
                continue
            lv_b.append(i[mask])
            lv_s.append(j[mask])
        return lv_b, lv_s, val_pslot
