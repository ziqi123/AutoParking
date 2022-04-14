
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_bbox_points: float = 1, cost_class: float = 1, cost_bbox_center: float = 1, batch_size: int = 16):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox_center = cost_bbox_center
        self.cost_bbox_points = cost_bbox_points
        self.batch_size = batch_size
        self.list = [[0, 1, 3, 2], [0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
                     [1, 0, 3, 2], [1, 0, 2, 3], [1, 2, 0, 3], [
                         1, 2, 3, 0], [1, 3, 2, 0], [1, 3, 0, 2],
                     [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 3, 0], [
                         2, 1, 0, 3], [2, 3, 1, 0], [2, 3, 0, 1],
                     [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 1, 0], [3, 2, 0, 1]]
        assert cost_class != 0 or cost_bbox_center != 0 or cost_bbox_points != 0

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_classes": Tensor of dim [batch_size, num_queries, num_classes] with the classification parking slots
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 10] with the predicted parking slots coordinates
            targets: This is a list of 2*batch_size+1 elements, except the first element: batch_images,
                    the left elements are parking_slots and ignored_parking_slots coordinates.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_parking_slots)
        """
        bs, num_queries = outputs["pred_box"].shape[:2]
        # [batch_size * num_queries, num_classes]
        out_prob = outputs["pred_class"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_box"].flatten(
            0, 1)  # [batch_size * num_queries, 10]
        out_bbox_center = out_bbox[:, 8:]
        out_bbox_points = out_bbox[:, :8]
        l = []
        for i in range(self.batch_size):
            a = torch.tensor([]).cuda(
            ) if targets[i + 1].shape[1] == 1 else torch.ones(targets[i + 1].shape[1]-1).cuda()
            b = torch.tensor([]).cuda() if targets[i + self.batch_size + 1].shape[1] == 1 else (
                torch.ones(targets[i + self.batch_size+1].shape[1]-1)*2).cuda()
            l.append(torch.cat((a, b)))

        tgt_ids = torch.cat(l).long()
        l = []

        for i in range(self.batch_size):
            a = torch.tensor([]).cuda() if targets[i +
                                                   1].shape[1] == 1 else targets[i+1][0, 1:]
            b = torch.tensor([]).cuda() if targets[i+self.batch_size +
                                                   1].shape[1] == 1 else targets[i+self.batch_size+1][0, 1:]
            l.append(torch.cat((a, b)))
        tgt_bbox = torch.cat(l)
        tgt_bbox_center = tgt_bbox[:, 8:]
        tgt_bbox_points = tgt_bbox[:, :8]
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox_center = torch.cdist(out_bbox_center, tgt_bbox_center, p=1)
        tgt = tgt_bbox_points.view(-1, 4, 2)
        ll = [torch.cdist(out_bbox_points, tgt[:, i, :].view(-1, 8),
                          p=1).unsqueeze(-1) for i in self.list]
        l2 = torch.cat(ll, axis=-1)
        cost_bbox_points, _ = torch.min(l2, dim=-1)
        C = self.cost_bbox_center * cost_bbox_center + self.cost_class * \
            cost_class+self.cost_bbox_points*cost_bbox_points
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [sum([v.shape[1] - 1, q.shape[1] - 1]) for v, q in
                 zip(targets[1:self.batch_size + 1], targets[1 + self.batch_size:])]
        indices = [linear_sum_assignment(c[i]) if sizes[i] != 0 else (
            np.array([-1]), np.array([-1])) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher2(set_cost_bbox_points, set_cost_class, set_cost_bbox_center, batch_size):
    return HungarianMatcher(cost_bbox_points=set_cost_bbox_points, cost_class=set_cost_class, cost_bbox_center=set_cost_bbox_center, batch_size=batch_size)
