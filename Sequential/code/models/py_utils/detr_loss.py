import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
import copy
from .misc import (NestedTensor, nested_tensor_from_tensor_list,
                   accuracy, get_world_size, interpolate,
                   is_dist_avail_and_initialized)


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, matcher2, weight_dict, eos_coef, losses):
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

    def loss_labels(self, outputs, targets, indices, indices2, num_boxes, num_boxes2):
        """Classification loss
        0:negative sample
        1:visible parking slots
        2:ignored parking slots
        """
        assert 'pred_classes' in outputs
        src_logits = outputs['pred_classes']
        # print('src_logits.shape[0]', src_logits.shape[0])
        l_b, l_s, pslot = self._get_src_permutation_idx(indices)
        cls_b, cls_s, ign_cls_b, ign_cls_s, _, _, ign_pslot, ign_pslot_idx = self._make_mask(
            indices, l_b, l_s, targets, src_logits.shape[0], pslot)
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[cls_b, cls_s] = 1
        target_classes[ign_cls_b, ign_cls_s] = 2
        loss_ce = F.cross_entropy(src_logits.transpose(
            1, 2), target_classes, self.empty_weight)
        loss_ce = loss_ce
        losses = {'loss_ce': loss_ce}
        #     # TODO this should probably be a separate loss, not hacked in this one here
        return losses

    def loss_class(self, outputs, targets, indices, indices2, num_boxes, num_boxes2):
        """Classification loss
        0:negative sample
        1:visible parking slots
        2:ignored parking slots
        """
        assert 'pred_kpt_logits' in outputs
        src_logits = outputs['pred_kpt_logits']
        tmp1 = [targets[0]]
        tmp2 = targets[1:33]
        tmp = [val for val in tmp2 for i in range(10)]
        tmp = tmp1+tmp
        # 321
        targets = tmp
        # print('src_logits.shape[0]', src_logits.shape[0])
        l_b, l_s, pslot = self._get_src_permutation_idx(indices2)
        cls_b, cls_s, ign_cls_b, ign_cls_s, _, _, ign_pslot, ign_pslot_idx = self._make_mask(
            indices2, l_b, l_s, targets, src_logits.shape[0], pslot)

        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[cls_b, cls_s] = 1
        target_classes[ign_cls_b, ign_cls_s] = 2

        loss_ce = F.cross_entropy(src_logits.transpose(
            1, 2), target_classes, self.empty_weight)

        loss_ce = loss_ce
        losses = {'loss_class': loss_ce}
        #     # TODO this should probably be a separate loss, not hacked in this one here
        return losses

    def loss_joints(self, outputs, targets, indices, indices2, num_boxes, num_boxes2):
        """Compute the losses related to the parking slot location with the L1 regression loss.
           The target boxes are expected in format (p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,center_x, center_y), normalized by the image size.
        """
        assert 'pred_box' in outputs
        assert 'pred_boxes' in outputs
        bs = outputs['pred_box'].shape[0]
        bs2 = outputs['pred_boxes'].shape[0]
        l_b, l_s, pslot = self._get_src_permutation_idx(indices)
        l_b2, l_s2, pslot2 = self._get_src_permutation_idx(indices2)

        tmp1 = [targets[0]]
        tmp2 = targets[1:33]
        tmp = [val for val in tmp2 for i in range(10)]
        tmp = tmp1+tmp
        # 321
        targets2 = tmp

        cls_b, cls_s, ign_cls_b, ign_cls_s, cls_sort, val_pslot, ign_pslot, ign_pslot_idx = self._make_mask(
            indices, l_b, l_s, targets, bs, pslot)

        cls_b2, cls_s2, ign_cls_b2, ign_cls_s2, cls_sort2, val_pslot2, ign_pslot2, ign_pslot_idx2 = self._make_mask(
            indices2, l_b2, l_s2, targets2, bs2, pslot2)
        src_boxes = outputs['pred_box'][cls_b, cls_s]
        src_boxes2 = outputs['pred_boxes'][cls_b2, cls_s2]
        src_ign_boxes = outputs['pred_box'][ign_cls_b, ign_cls_s]
        src_ign_boxes2 = outputs['pred_boxes'][ign_cls_b2, ign_cls_s2]
        tgt_boxes = torch.cat([targets[1+idx][0][J]
                              for idx, J in zip(val_pslot, cls_sort)])
        # print('val_pslot, cls_sort', val_pslot, cls_sort)
        tgt_boxes2 = torch.cat([targets2[1+idx][0][J]
                                for idx, J in zip(val_pslot2, cls_sort2)])
        # print('val_pslot2, cls_sort2', val_pslot2, cls_sort2)
        # print('tgt_boxes', tgt_boxes.shape)
        loss_bbox_ign = torch.zeros(1, 1)
        if len(ign_pslot) is not 0:
            tgt_ign_boxes = torch.cat(
                [targets[1+idx+bs][0][J] for idx, J in zip(ign_pslot_idx, ign_pslot)])
            tgt_ign_boxes2 = torch.cat(
                [targets2[1+idx+bs2][0][J] for idx, J in zip(ign_pslot_idx2, ign_pslot2)])
            num_pslot_ign = tgt_ign_boxes.shape[0]
            num_pslot_ign2 = tgt_ign_boxes2.shape[0]
            # print('num_pslot_ign', num_pslot_ign)
            # print('num_pslot_ign2', num_pslot_ign2)
            tgt_ign = tgt_ign_boxes[:,
                                    :8].view(-1, 4, 2).flatten(0, 1).detach()
            tgt_ign2 = tgt_ign_boxes2[:,
                                      :8].view(-1, 4, 2).flatten(0, 1).detach()
            src_ign = src_ign_boxes[:,
                                    :8].view(-1, 4, 2).flatten(0, 1).detach()
            src_ign2 = src_ign_boxes2[:,
                                      :8].view(-1, 4, 2).flatten(0, 1).detach()
            cost_points_ign = torch.cdist(src_ign, tgt_ign, p=1)
            cost_points_ign2 = torch.cdist(src_ign2, tgt_ign2, p=1)
            C = cost_points_ign.view(num_pslot_ign, 4, -1).cpu()
            C2 = cost_points_ign2.view(num_pslot_ign2, 4, -1).cpu()
            sizes = [4 for i in range(num_pslot_ign)]
            sizes2 = [4 for i in range(num_pslot_ign2)]
            indices_ign = [linear_sum_assignment(c[i]) for i, c in
                           enumerate(C.split(sizes, -1))]
            indices_ign2 = [linear_sum_assignment(c[i]) for i, c in
                            enumerate(C2.split(sizes2, -1))]
            indices_ign = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                           indices_ign]
            indices_ign2 = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                            indices_ign2]
            idx_src_ign, idx_tgt_ign = self._get_src_permutation_idx_points(
                indices_ign)
            idx_src_ign2, idx_tgt_ign2 = self._get_src_permutation_idx_points(
                indices_ign2)
            src_ign_boxes = torch.cat((src_ign_boxes[:, :8].view(-1, 4, 2).flatten(
                0, 1)[idx_src_ign].view(-1, 8), src_ign_boxes[:, 8:]), axis=-1)
            src_ign_boxes2 = torch.cat((src_ign_boxes2[:, :8].view(-1, 4, 2).flatten(
                0, 1)[idx_src_ign2].view(-1, 8), src_ign_boxes2[:, 8:]), axis=-1)
            tgt_ign_boxes = torch.cat((tgt_ign_boxes[:, :8].view(-1, 4, 2).flatten(
                0, 1)[idx_tgt_ign].view(-1, 8), tgt_ign_boxes[:, 8:]), axis=-1)
            tgt_ign_boxes2 = torch.cat((tgt_ign_boxes2[:, :8].view(-1, 4, 2).flatten(
                0, 1)[idx_tgt_ign2].view(-1, 8), tgt_ign_boxes2[:, 8:]), axis=-1)
            loss_bbox_ign = F.l1_loss(
                src_ign_boxes, tgt_ign_boxes, reduction='none')
            loss_bbox_ign2 = F.l1_loss(
                src_ign_boxes2, tgt_ign_boxes2, reduction='none')
        num_pslot = tgt_boxes.shape[0]
        num_pslot2 = tgt_boxes2.shape[0]
        tgt = tgt_boxes[:, :8].view(-1, 4, 2).flatten(0, 1).detach()
        tgt2 = tgt_boxes2[:, :8].view(-1, 4, 2).flatten(0, 1).detach()
        src = src_boxes[:, :8].view(-1, 4, 2).flatten(0, 1).detach()
        src2 = src_boxes2[:, :8].view(-1, 4, 2).flatten(0, 1).detach()
        cost_points = torch.cdist(src, tgt, p=1)
        cost_points2 = torch.cdist(src2, tgt2, p=1)
        C = cost_points.view(num_pslot, 4, -1).cpu()
        C2 = cost_points2.view(num_pslot2, 4, -1).cpu()
        sizes = [4 for i in range(num_pslot)]
        sizes2 = [4 for i in range(num_pslot2)]
        indices_p = [linear_sum_assignment(c[i]) for i, c in
                     enumerate(C.split(sizes, -1))]
        indices_p2 = [linear_sum_assignment(c[i]) for i, c in
                      enumerate(C2.split(sizes2, -1))]
        indices_p = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
            j, dtype=torch.int64)) for i, j in indices_p]
        indices_p2 = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
            j, dtype=torch.int64)) for i, j in indices_p2]
        idx_src, idx_tgt = self._get_src_permutation_idx_points(indices_p)
        idx_src2, idx_tgt2 = self._get_src_permutation_idx_points(indices_p2)
        src_boxes = torch.cat((src_boxes[:, :8].view(-1, 4, 2).flatten(0, 1)[
                              idx_src].view(-1, 8), src_boxes[:, 8:]), axis=-1)
        src_boxes2 = torch.cat((src_boxes2[:, :8].view(-1, 4, 2).flatten(0, 1)[
            idx_src2].view(-1, 8), src_boxes2[:, 8:]), axis=-1)
        tgt_boxes = torch.cat((tgt_boxes[:, :8].view(-1, 4, 2).flatten(0, 1)[
                              idx_tgt].view(-1, 8), tgt_boxes[:, 8:]), axis=-1)
        tgt_boxes2 = torch.cat((tgt_boxes2[:, :8].view(-1, 4, 2).flatten(0, 1)[
            idx_tgt2].view(-1, 8), tgt_boxes2[:, 8:]), axis=-1)
        # print('tgt_boxes2', tgt_boxes2)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        loss_bbox2 = F.l1_loss(src_boxes2, tgt_boxes2, reduction='none')
        losses = {}
        losses['loss_boxes'] = (loss_bbox.sum(
        )+loss_bbox_ign.sum()+loss_bbox2.sum()+loss_bbox_ign2.sum()) / num_pslot
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
        # print('l_b, l_s', l_b, l_s)
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
            # print("12")
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

    def get_loss(self, loss, outputs, targets, indices, indices2, num_boxes, num_boxes2, **kwargs):

        loss_map = {
            'cls': self.loss_labels,
            'joints': self.loss_joints,
            'class': self.loss_class
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, indices2, num_boxes, num_boxes2, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts,
                      The expected keys in each dict depends on the losses applied.
        """
        # matching
        # print('targets', len(targets))
        # 33
        # print('targets[0]', targets[0].shape)
        # [16, 3, 10]
        # print('targets[0]', targets[0][0])
        indices = self.matcher(outputs, targets)
        indices2 = self.matcher2(outputs, targets)
        bs = outputs['pred_box'].shape[0]
        #     print(len(indice))
        num_pslots = sum(tgt.shape[1]-1+tgt_ign.shape[1]-1 for tgt,
                         tgt_ign in zip(targets[1:bs+1], targets[bs+1:]))
        num_pslots = torch.as_tensor(
            [num_pslots], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pslots)
        num_pslots = torch.clamp(num_pslots / get_world_size(), min=1).item()

        bs2 = outputs['pred_boxes'].shape[0]
        num_pslots2 = sum(tgt.shape[1]-1+tgt_ign.shape[1]-1 for tgt,
                          tgt_ign in zip(targets[1:bs2+1], targets[bs2+1:]))
        num_pslots2 = torch.as_tensor(
            [num_pslots2], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pslots2)
        num_pslots2 = torch.clamp(num_pslots2 / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, indices2, num_pslots, num_pslots2))
        if 'aux_outputs' in outputs:
            self.enumerate = enumerate(outputs['aux_outputs'])
            for i, aux_outputs in self.enumerate:
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_pslots, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices
