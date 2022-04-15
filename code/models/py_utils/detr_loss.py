import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
import copy
from .misc import (NestedTensor, nested_tensor_from_tensor_list,
                   accuracy, get_world_size, interpolate,
                   is_dist_avail_and_initialized)

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, matcher2, matcher3, matcher4, matcher5, matcher6, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.matcher2 = matcher2
        self.matcher3 = matcher3
        self.matcher4 = matcher4
        self.matcher5 = matcher5
        self.matcher6 = matcher6
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.ign_cls_b = 0
        self.ign_cls_s = 0
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.ign_threshold = 0
        self.register_buffer('empty_weight', empty_weight)
        self.dist_list = []
        self.device = 'cuda'

    def loss_labels(self, outputs, targets, indices, indices2, indices3, indices4, indices5, indices6, num_pslots, num_pslots2, num_pslots3, num_pslots4, num_pslots5, num_pslots6):
        """Classification loss
        0:negative sample
        1:visible parking slots
        2:ignored parking slots
        """
        assert 'pred_classes' in outputs
        src_logits = outputs['pred_classes']

        l_b, l_s, pslot = self._get_src_permutation_idx(indices)

        cls_b, cls_s, ign_cls_b, ign_cls_s, _, _, ign_pslot, ign_pslot_idx = self._make_mask(
            indices, l_b, l_s, targets, src_logits.shape[0], pslot)
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[cls_b, cls_s] = 1
        target_classes[ign_cls_b, ign_cls_s] = 2
        loss_ce = F.cross_entropy(src_logits.transpose(
            1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        #     # TODO this should probably be a separate loss, not hacked in this one here
        return losses

    def loss_class(self, outputs, targets, indices, indices2, indices3, indices4, indices5, indices6, num_pslots, num_pslots2, num_pslots3, num_pslots4, num_pslots5, num_pslots6):
        """Classification loss
        0:negative sample
        1:visible parking slots
        2:ignored parking slots
        """
        assert 'pred_class' in outputs
        src_logits = outputs['pred_class']

        l_b, l_s, pslot = self._get_src_permutation_idx(indices2)

        cls_b, cls_s, ign_cls_b, ign_cls_s, _, _, ign_pslot, ign_pslot_idx = self._make_mask(
            indices2, l_b, l_s, targets, src_logits.shape[0], pslot)
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[cls_b, cls_s] = 1
        target_classes[ign_cls_b, ign_cls_s] = 2
        loss_ce = F.cross_entropy(src_logits.transpose(
            1, 2), target_classes, self.empty_weight)
        losses = {'loss_classs': loss_ce}
        #     # TODO this should probably be a separate loss, not hacked in this one here
        return losses

    def loss_boxes(self, outputs, targets, indices, indices2, indices3, indices4, indices5, indices6, num_pslots, num_pslots2, num_pslots3, num_pslots4, num_pslots5, num_pslots6):
        """Compute the losses related to the parking slot location with the L1 regression loss.
           The target boxes are expected in format (p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,center_x, center_y), normalized by the image size.
        """
        assert 'pred_box' in outputs
        bs = outputs['pred_box'].shape[0]
        # 第一个box
        # _get_src_permutation_idx(indices)返回两个值batch_idx和src_idx
        l_b, l_s, pslot = self._get_src_permutation_idx(indices2)
        cls_b, cls_s, ign_cls_b, ign_cls_s, cls_sort, val_pslot, ign_pslot, ign_pslot_idx = self._make_mask(
            indices2, l_b, l_s, targets, bs, pslot)
        # 根据indices的横坐标提取预测输出outputs['pred_boxes']中的对应bbox
        inx = torch.LongTensor([0, 1, 2, 3])
        out = outputs['pred_box'][cls_b, cls_s]
        src_boxes = torch.index_select(out, 1, inx.to(self.device))
        # target_boxes由targets['boxes'] 根据 indices的纵坐标重新排序得到
        tgt_boxes = torch.cat([targets[1+idx][0][J]
                              for idx, J in zip(val_pslot, cls_sort)])
        inxx = torch.LongTensor([0, 4])
        inxxx = torch.LongTensor([1, 5])
        x0 = torch.index_select(
            tgt_boxes, 1, inxx.to(self.device))
        y0 = torch.index_select(
            tgt_boxes, 1, inxxx.to(self.device))
        tgt_min_x = torch.min(x0, 1)[0].T.unsqueeze(1)
        # print('tgt_min_x', tgt_min_x.shape)
        # [70, 1]
        tgt_min_y = torch.min(y0, 1)[0].T.unsqueeze(1)
        tgt_max_x = torch.max(x0, 1)[0].T.unsqueeze(1)
        tgt_max_y = torch.max(y0, 1)[0].T.unsqueeze(1)

        tgt_boxes = torch.cat((tgt_min_x, tgt_min_y, tgt_max_x, tgt_max_y), 1)

        # print('tgt_boxes', tgt_boxes.shape)
        # [25,4]
        tgt_box = box_xyxy_to_cxcywh(tgt_boxes)
        loss_bbox = F.l1_loss(src_boxes, tgt_box, reduction='none')

        # 第二个box
        # _get_src_permutation_idx(indices)返回两个值batch_idx和src_idx
        l_b2, l_s2, pslot2 = self._get_src_permutation_idx(indices3)
        cls_b2, cls_s2, ign_cls_b2, ign_cls_s2, cls_sort2, val_pslot2, ign_pslot2, ign_pslot_idx2 = self._make_mask(
            indices3, l_b2, l_s2, targets, bs, pslot2)
        # 根据indices的横坐标提取预测输出outputs['pred_boxes']中的对应bbox
        inx2 = torch.LongTensor([4, 5, 6, 7])
        out2 = outputs['pred_box'][cls_b2, cls_s2]
        src_boxes2 = torch.index_select(out2, 1, inx2.to(self.device))
        # target_boxes由targets['boxes'] 根据 indices的纵坐标重新排序得到
        tgt_boxes2 = torch.cat([targets[1+idx][0][J]
                                for idx, J in zip(val_pslot2, cls_sort2)])
        inxx = torch.LongTensor([0, 4])
        inxxx = torch.LongTensor([1, 5])
        x0 = torch.index_select(
            tgt_boxes2, 1, inxx.to(self.device))
        y0 = torch.index_select(
            tgt_boxes2, 1, inxxx.to(self.device))
        tgt_min_x = torch.min(x0, 1)[0].T.unsqueeze(1)
        # print('tgt_min_x', tgt_min_x.shape)
        # [70, 1]
        tgt_min_y = torch.min(y0, 1)[0].T.unsqueeze(1)
        tgt_max_x = torch.max(x0, 1)[0].T.unsqueeze(1)
        tgt_max_y = torch.max(y0, 1)[0].T.unsqueeze(1)

        tgt_boxes = torch.cat((tgt_min_x, tgt_min_y, tgt_max_x, tgt_max_y), 1)
        tgt_box2 = box_xyxy_to_cxcywh(tgt_boxes)
        loss_bbox2 = F.l1_loss(src_boxes2, tgt_box2, reduction='none')

        # 第三个box
        # _get_src_permutation_idx(indices)返回两个值batch_idx和src_idx
        l_b3, l_s3, pslot3 = self._get_src_permutation_idx(indices4)
        cls_b3, cls_s3, ign_cls_b3, ign_cls_s3, cls_sort3, val_pslot3, ign_pslot3, ign_pslot_idx3 = self._make_mask(
            indices4, l_b3, l_s3, targets, bs, pslot3)
        # 根据indices的横坐标提取预测输出outputs['pred_boxes']中的对应bbox
        inx = torch.LongTensor([8, 9, 10, 11])
        out = outputs['pred_box'][cls_b, cls_s]
        src_boxes3 = torch.index_select(out, 1, inx.to(self.device))
        # target_boxes由targets['boxes'] 根据 indices的纵坐标重新排序得到
        tgt_boxes = torch.cat([targets[1+idx][0][J]
                              for idx, J in zip(val_pslot3, cls_sort3)])
        inxx = torch.LongTensor([0, 4])
        inxxx = torch.LongTensor([1, 5])
        x0 = torch.index_select(
            tgt_boxes, 1, inxx.to(self.device))
        y0 = torch.index_select(
            tgt_boxes, 1, inxxx.to(self.device))
        tgt_min_x = torch.min(x0, 1)[0].T.unsqueeze(1)
        # print('tgt_min_x', tgt_min_x.shape)
        # [70, 1]
        tgt_min_y = torch.min(y0, 1)[0].T.unsqueeze(1)
        tgt_max_x = torch.max(x0, 1)[0].T.unsqueeze(1)
        tgt_max_y = torch.max(y0, 1)[0].T.unsqueeze(1)

        tgt_boxes = torch.cat((tgt_min_x, tgt_min_y, tgt_max_x, tgt_max_y), 1)

        # print('tgt_boxes', tgt_boxes.shape)
        # [25,4]
        tgt_box3 = box_xyxy_to_cxcywh(tgt_boxes)
        loss_bbox3 = F.l1_loss(src_boxes3, tgt_box3, reduction='none')

        # 第四个box
        # _get_src_permutation_idx(indices)返回两个值batch_idx和src_idx
        l_b4, l_s4, pslot4 = self._get_src_permutation_idx(indices5)
        cls_b4, cls_s4, ign_cls_b4, ign_cls_s4, cls_sort4, val_pslot4, ign_pslot4, ign_pslot_idx4 = self._make_mask(
            indices5, l_b4, l_s4, targets, bs, pslot4)
        # 根据indices的横坐标提取预测输出outputs['pred_boxes']中的对应bbox
        inx = torch.LongTensor([12, 13, 14, 15])
        out = outputs['pred_box'][cls_b4, cls_s4]
        src_boxes4 = torch.index_select(out, 1, inx.to(self.device))
        # target_boxes由targets['boxes'] 根据 indices的纵坐标重新排序得到
        tgt_boxes = torch.cat([targets[1+idx][0][J]
                              for idx, J in zip(val_pslot4, cls_sort4)])
        inxx = torch.LongTensor([0, 4])
        inxxx = torch.LongTensor([1, 5])
        x0 = torch.index_select(
            tgt_boxes, 1, inxx.to(self.device))
        y0 = torch.index_select(
            tgt_boxes, 1, inxxx.to(self.device))
        tgt_min_x = torch.min(x0, 1)[0].T.unsqueeze(1)
        # print('tgt_min_x', tgt_min_x.shape)
        # [70, 1]
        tgt_min_y = torch.min(y0, 1)[0].T.unsqueeze(1)
        tgt_max_x = torch.max(x0, 1)[0].T.unsqueeze(1)
        tgt_max_y = torch.max(y0, 1)[0].T.unsqueeze(1)

        tgt_boxes = torch.cat((tgt_min_x, tgt_min_y, tgt_max_x, tgt_max_y), 1)

        # print('tgt_boxes', tgt_boxes.shape)
        # [25,4]
        tgt_box4 = box_xyxy_to_cxcywh(tgt_boxes)
        loss_bbox4 = F.l1_loss(src_boxes4, tgt_box4, reduction='none')

        # 第五个box
        # _get_src_permutation_idx(indices)返回两个值batch_idx和src_idx
        l_b5, l_s5, pslot5 = self._get_src_permutation_idx(indices6)
        cls_b5, cls_s5, ign_cls_b5, ign_cls_s5, cls_sort5, val_pslot5, ign_pslot5, ign_pslot_idx5 = self._make_mask(
            indices6, l_b5, l_s5, targets, bs, pslot5)
        # 根据indices的横坐标提取预测输出outputs['pred_boxes']中的对应bbox
        inx = torch.LongTensor([0, 1, 2, 3])
        out = outputs['pred_box'][cls_b5, cls_s5]
        src_boxes5 = torch.index_select(out, 1, inx.to(self.device))
        # target_boxes由targets['boxes'] 根据 indices的纵坐标重新排序得到
        tgt_boxes = torch.cat([targets[1+idx][0][J]
                              for idx, J in zip(val_pslot5, cls_sort5)])
        inxx = torch.LongTensor([0, 4])
        inxxx = torch.LongTensor([1, 5])
        x0 = torch.index_select(
            tgt_boxes, 1, inxx.to(self.device))
        y0 = torch.index_select(
            tgt_boxes, 1, inxxx.to(self.device))
        tgt_min_x = torch.min(x0, 1)[0].T.unsqueeze(1)
        # print('tgt_min_x', tgt_min_x.shape)
        # [70, 1]
        tgt_min_y = torch.min(y0, 1)[0].T.unsqueeze(1)
        tgt_max_x = torch.max(x0, 1)[0].T.unsqueeze(1)
        tgt_max_y = torch.max(y0, 1)[0].T.unsqueeze(1)

        tgt_boxes = torch.cat((tgt_min_x, tgt_min_y, tgt_max_x, tgt_max_y), 1)

        # print('tgt_boxes', tgt_boxes.shape)
        # [25,4]
        tgt_box5 = box_xyxy_to_cxcywh(tgt_boxes)
        loss_bbox5 = F.l1_loss(src_boxes5, tgt_box5, reduction='none')

        losses = {}
        losses['loss_bbox'] = (
            loss_bbox.sum() + loss_bbox2.sum() + loss_bbox3.sum() + loss_bbox4.sum() + loss_bbox5.sum()) / num_pslots2

        # print('src_boxes', src_boxes.shape)
        # print('tgt_boxes', tgt_boxes.shape)

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(tgt_box)))
        loss_giou2 = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes2),
            box_cxcywh_to_xyxy(tgt_box2)))
        loss_giou3 = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes3),
            box_cxcywh_to_xyxy(tgt_box3)))
        loss_giou4 = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes4),
            box_cxcywh_to_xyxy(tgt_box4)))
        loss_giou5 = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes5),
            box_cxcywh_to_xyxy(tgt_box5)))
        losses['loss_giou'] = (
            loss_giou.sum() + loss_giou2.sum() + loss_giou3.sum() + loss_giou4.sum() + loss_giou5.sum()) / num_pslots2
        return losses

    def loss_joints(self, outputs, targets, indices, indices2, indices3, indices4, indices5, indices6, num_pslots, num_pslots2, num_pslots3, num_pslots4, num_pslots5, num_pslots6):
        """Compute the losses related to the parking slot location with the L1 regression loss.
           The target boxes are expected in format (p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,center_x, center_y), normalized by the image size.
        """

        assert 'pred_boxes' in outputs
        bs = outputs['pred_boxes'].shape[0]
        l_b, l_s, pslot = self._get_src_permutation_idx(indices)
        cls_b, cls_s, ign_cls_b, ign_cls_s, cls_sort, val_pslot, ign_pslot, ign_pslot_idx = self._make_mask(
            indices, l_b, l_s, targets, bs, pslot)
        src_boxes = outputs['pred_boxes'][cls_b, cls_s]
        src_ign_boxes = outputs['pred_boxes'][ign_cls_b, ign_cls_s]
        tgt_boxes = torch.cat([targets[1+idx][0][J]
                               for idx, J in zip(val_pslot, cls_sort)])
        tgt_ign_boxes = torch.cat([targets[1+idx+bs][0][J]
                                   for idx, J in zip(ign_pslot_idx, ign_pslot)])
        num_pslot = tgt_boxes.shape[0]
        tgt = tgt_boxes[:, : 8].view(-1, 4, 2).flatten(0, 1).detach()
        src = src_boxes[:, : 8].view(-1, 4, 2).flatten(0, 1).detach()
        cost_points = torch.cdist(src, tgt, p=1)
        C = cost_points.view(num_pslot, 4, -1).cpu()
        sizes = [4 for i in range(num_pslot)]
        indices_p = [linear_sum_assignment(c[i]) for i, c in
                     enumerate(C.split(sizes, -1))]
        indices_p = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
            j, dtype=torch.int64)) for i, j in indices_p]
        idx_src, idx_tgt = self._get_src_permutation_idx_points(indices_p)
        num_pslot_ign = tgt_ign_boxes.shape[0]
        tgt_ign = tgt_ign_boxes[:, : 8].view(-1, 4, 2).flatten(0, 1).detach()
        src_ign = src_ign_boxes[:, : 8].view(-1, 4, 2).flatten(0, 1).detach()
        cost_points_ign = torch.cdist(src_ign, tgt_ign, p=1)
        C = cost_points_ign.view(num_pslot_ign, 4, -1).cpu()
        sizes = [4 for i in range(num_pslot_ign)]
        indices_ign = [linear_sum_assignment(c[i]) for i, c in
                       enumerate(C.split(sizes, -1))]
        indices_ign = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                       indices_ign]
        idx_src_ign, idx_tgt_ign = self._get_src_permutation_idx_points(
            indices_ign)
        src_boxes = torch.cat((src_boxes[:, : 8].view(-1, 4, 2).flatten(0, 1)[
            idx_src].view(-1, 8), src_boxes[:, 8:]), axis=-1)
        tgt_boxes = torch.cat((tgt_boxes[:, : 8].view(-1, 4, 2).flatten(0, 1)[
            idx_tgt].view(-1, 8), tgt_boxes[:, 8:]), axis=-1)
        src_ign_boxes = torch.cat((src_ign_boxes[:, : 8].view(-1, 4, 2).flatten(
            0, 1)[idx_src_ign].view(-1, 8), src_ign_boxes[:, 8:]), axis=-1)
        tgt_ign_boxes = torch.cat((tgt_ign_boxes[:, : 8].view(-1, 4, 2).flatten(
            0, 1)[idx_tgt_ign].view(-1, 8), tgt_ign_boxes[:, 8:]), axis=-1)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        loss_bbox_ign = F.l1_loss(
            src_ign_boxes, tgt_ign_boxes, reduction='none')
        losses = {}
        losses['loss_boxes'] = (
            loss_bbox.sum()+loss_bbox_ign.sum()) / num_pslots
        return losses

    def _make_mask(self, indices, l_b, l_s, targets, bs, pslot):
        lv_b = []
        lv_s = []
        li_b = []
        li_s = []
        joints = []
        ign_pslot = []
        val_pslot = copy.deepcopy(pslot)
        ign_pslot_idx = copy.deepcopy(pslot)
        for id, (i, j) in enumerate(zip(l_b, l_s)):
            if targets[1+pslot[id]].shape[1]-1+targets[1+pslot[id]+bs].shape[1]-1 != i.shape[0]:
                print('contain discrete length between prediction and label')
                raise ValueError

            valid_length = targets[1+pslot[id]].shape[1]-1
            ign_length = targets[1+bs+pslot[id]].shape[1]-1
            sort = indices[pslot[id]][-1]
            mask = sort < valid_length
            lv_b.append(i[mask])
            lv_s.append(j[mask])
            ign_mask = ~mask
            li_b.append(i[ign_mask])
            li_s.append(j[ign_mask])
            if valid_length > 0:
                joints.append(sort[mask]+1)
            else:
                val_pslot.remove(pslot[id])
            if ign_length > 0:
                ign_pslot.append(sort[ign_mask]+1-valid_length)
            else:
                ign_pslot_idx.remove(pslot[id])
        return torch.cat(lv_b), torch.cat(lv_s), torch.cat(li_b), torch.cat(li_s), joints, val_pslot, ign_pslot, ign_pslot_idx

    def _get_src_permutation_idx_points(self, indices):
        l_bidx = []
        l_sidx = []
        valid_pslot = []
        for i, src in enumerate(indices):
            l_bidx.append(src[0]+4*i)
            l_sidx.append(src[1]+4*i)
        return torch.cat(l_bidx), torch.cat(l_sidx)

    def _get_src_permutation_idx(self, indices):
        l_bidx = []
        l_sidx = []
        valid_pslot = []
        for i, src in enumerate(indices):
            if src[0][0] != -1:
                l_bidx.append(torch.full_like(src[0], i))
                l_sidx.append(src[0])
                valid_pslot.append(i)
        return l_bidx, l_sidx, valid_pslot

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt[0], i)
                               for i, tgt in enumerate(indices)])
        tgt_idx = torch.cat([tgt[0] for tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets,  indices, indices2, indices3, indices4, indices5, indices6, num_pslots, num_pslots2, num_pslots3, num_pslots4, num_pslots5, num_pslots6, **kwargs):

        loss_map = {
            'cls': self.loss_labels,
            'joints': self.loss_joints,
            'box': self.loss_boxes,
            'class': self.loss_class
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets,  indices, indices2, indices3, indices4, indices5, indices6, num_pslots, num_pslots2, num_pslots3, num_pslots4, num_pslots5, num_pslots6, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts,
                      The expected keys in each dict depends on the losses applied.
        """
        # matching
        indices = self.matcher(outputs, targets)
        bs = outputs['pred_boxes'].shape[0]
        #     print(len(indice))
        num_pslots = sum(tgt.shape[1]-1+tgt_ign.shape[1]-1 for tgt,
                         tgt_ign in zip(targets[1: bs+1], targets[bs+1:]))
        num_pslots = torch.as_tensor(
            [num_pslots], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pslots)
        num_pslots = torch.clamp(num_pslots / get_world_size(), min=1).item()

        # 第一个box
        indices2 = self.matcher2(outputs, targets)
        bs2 = outputs['pred_box'].shape[0]
        #     print(len(indice))
        num_pslots2 = sum(tgt.shape[1]-1+tgt_ign.shape[1]-1 for tgt,
                          tgt_ign in zip(targets[1: bs2+1], targets[bs2+1:]))
        num_pslots2 = torch.as_tensor(
            [num_pslots2], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pslots2)
        num_pslots2 = torch.clamp(num_pslots2 / get_world_size(), min=1).item()

        # 第二个box
        indices3 = self.matcher3(outputs, targets)
        bs3 = outputs['pred_box'].shape[0]
        #     print(len(indice))
        num_pslots3 = sum(tgt.shape[1]-1+tgt_ign.shape[1]-1 for tgt,
                          tgt_ign in zip(targets[1: bs3+1], targets[bs3+1:]))
        num_pslots3 = torch.as_tensor(
            [num_pslots3], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pslots3)
        num_pslots3 = torch.clamp(num_pslots3 / get_world_size(), min=1).item()

        # 第三个box
        indices4 = self.matcher4(outputs, targets)
        bs4 = outputs['pred_box'].shape[0]
        #     print(len(indice))
        num_pslots4 = sum(tgt.shape[1]-1+tgt_ign.shape[1]-1 for tgt,
                          tgt_ign in zip(targets[1: bs4+1], targets[bs4+1:]))
        num_pslots4 = torch.as_tensor(
            [num_pslots4], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pslots4)
        num_pslots4 = torch.clamp(num_pslots4 / get_world_size(), min=1).item()

        # 第四个box
        indices5 = self.matcher5(outputs, targets)
        bs5 = outputs['pred_box'].shape[0]
        #     print(len(indice))
        num_pslots5 = sum(tgt.shape[1]-1+tgt_ign.shape[1]-1 for tgt,
                          tgt_ign in zip(targets[1: bs5+1], targets[bs5+1:]))
        num_pslots5 = torch.as_tensor(
            [num_pslots5], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pslots5)
        num_pslots5 = torch.clamp(num_pslots5 / get_world_size(), min=1).item()

        # 第五个box
        indices6 = self.matcher6(outputs, targets)
        bs6 = outputs['pred_box'].shape[0]
        #     print(len(indice))
        num_pslots6 = sum(tgt.shape[1]-1+tgt_ign.shape[1]-1 for tgt,
                          tgt_ign in zip(targets[1: bs6+1], targets[bs6+1:]))
        num_pslots6 = torch.as_tensor(
            [num_pslots6], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pslots6)
        num_pslots6 = torch.clamp(num_pslots6 / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, indices2, indices3, indices4, indices5, indices6, num_pslots, num_pslots2, num_pslots3, num_pslots4, num_pslots5, num_pslots6))
        if 'aux_outputs' in outputs:
            self.enumerate = enumerate(outputs['aux_outputs'])
            # print()
            for i, aux_outputs in self.enumerate:
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_pslots, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices
