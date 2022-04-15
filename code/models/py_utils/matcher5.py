
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from re import X
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from torch.autograd import Variable


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, batch_size: int = 16):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.batch_size = batch_size
        self.device = 'cuda'
        # self.list = [[0, 1, 3, 2], [0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
        #              [1, 0, 3, 2], [1, 0, 2, 3], [1, 2, 0, 3], [
        #                  1, 2, 3, 0], [1, 3, 2, 0], [1, 3, 0, 2],
        #              [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 3, 0], [
        #                  2, 1, 0, 3], [2, 3, 1, 0], [2, 3, 0, 1],
        #              [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 1, 0], [3, 2, 0, 1]]
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0

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
        indices0 = torch.LongTensor([12,13,14,15])
        # print(outputs["pred_box"].shape)
        # [16, 50, 20]
        out = torch.index_select(
            outputs["pred_box"], 2, indices0.to(self.device))
        # print('out', out.shape)
        # [16,50,4]
        bs, num_queries = out.shape[:2]
        # [batch_size * num_queries, num_classes]

        out_prob = out.flatten(0, 1).softmax(-1)
        # print('out_prob', out_prob.shape)
        # ([800, 3]
        out_bbox = out.flatten(
            0, 1)  # [batch_size * num_queries, 10]
        # print('out_bbox', out_bbox.shape)

        # tgt_ids = torch.cat([v["labels"] for v in targets])
        # tgt_bbox = torch.cat([v["boxes"] for v in targets])

        l = []
        for i in range(self.batch_size):
            a = torch.tensor([]).cuda(
            ) if targets[i + 1].shape[1] == 1 else torch.ones(targets[i + 1].shape[1]-1).cuda()
            b = torch.tensor([]).cuda() if targets[i + self.batch_size + 1].shape[1] == 1 else (
                torch.ones(targets[i + self.batch_size+1].shape[1]-1)*2).cuda()
            l.append(torch.cat((a, b)))

        tgt_ids = torch.cat(l).long()
        # [61]
        # print('tgt_ids', tgt_ids.shape)
        l = []

        for i in range(self.batch_size):
            a = torch.tensor([]).cuda() if targets[i +
                                                   1].shape[1] == 1 else targets[i+1][0, 1:]
            b = torch.tensor([]).cuda() if targets[i+self.batch_size +
                                                   1].shape[1] == 1 else targets[i+self.batch_size+1][0, 1:]
            l.append(torch.cat((a, b)))
        tgt_bbox = torch.cat(l)
        # print('tgt_bbox', tgt_bbox.shape)
        # [61, 10]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # print('tgt_bbox', tgt_bbox.shape)
        # [74, 10]

        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        # print('out_bbox_xyxy', out_bbox_xyxy.shape)

        indices = torch.LongTensor([0, 4])
        indices2 = torch.LongTensor([1, 5])
        indices3 = torch.LongTensor([0, 2])
        indices4 = torch.LongTensor([1, 3])
        x0 = torch.index_select(
            tgt_bbox, 1, indices.to(self.device))
        y0 = torch.index_select(
            tgt_bbox, 1, indices2.to(self.device))
        x1 = torch.index_select(
            out_bbox_xyxy, 1, indices3.to(self.device))
        y1 = torch.index_select(
            out_bbox_xyxy, 1, indices4.to(self.device))
        tgt_min_x = torch.min(x0, 1)[0].T.unsqueeze(1)
        # print('tgt_min_x', tgt_min_x.shape)
        # [70, 1]
        tgt_min_y = torch.min(y0, 1)[0].T.unsqueeze(1)
        tgt_max_x = torch.max(x0, 1)[0].T.unsqueeze(1)
        tgt_max_y = torch.max(y0, 1)[0].T.unsqueeze(1)

        out_min_x = torch.min(x1, 1)[0].T.unsqueeze(1)
        out_min_y = torch.min(y1, 1)[0].T.unsqueeze(1)
        out_max_x = torch.max(x1, 1)[0].T.unsqueeze(1)
        out_max_y = torch.max(y1, 1)[0].T.unsqueeze(1)

        # x = torch.randn(5, 1)
        # y = torch.randn(5, 1)
        # print(x, y)
        # s = torch.cat((x, y), 1)
        # print(s)
        # print('tgt_min_x', tgt_min_x.shape)
        tgt_bbox = torch.cat((tgt_min_x, tgt_min_y, tgt_max_x, tgt_max_y), 1)
        out_bbox = torch.cat((out_min_x, out_min_y, out_max_x, out_max_y), 1)

        # print('tgt_bbox', tgt_bbox.shape)
        # [61, 1, 4]
        # tgt_bbox = tgt_bbox.squeeze(dim=1)
        # print('tgt_bbox', tgt_bbox.shape)
        # [61, 4]
        # print('out_bbox', out_bbox.shape)

        # [800, 4]
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # print('cost_bbox', cost_bbox.shape)
        # ([800, 71]
        # Compute the giou cost betwen boxes
        # for i in range(len(out_bbox)):
        #     for j in range(len(tgt_bbox)):
        box1 = Variable(out_bbox).data.cpu()
        box2 = Variable(tgt_bbox).data.cpu()
        # print('box1', box1.shape)
        # print("boxes1[:, None, :2]", box1[:, None, :2])
        # print('box2', box2.shape)
        # print('boxes2[:, :2]', box2[:, :2])

        cost_giou = -generalized_box_iou(box1, box2)
        # [2, 2]

        # print('cost_giou', cost_giou.shape)

        # Final cost matrix
        C1 = self.cost_giou * cost_giou
        C2 = self.cost_class * cost_class
        # print("C1", C1.shape)
        # print("C2", C2.shape)
        # [800, 77]
        C = self.cost_bbox * cost_bbox.cpu() + self.cost_class * \
            cost_class.cpu() + self.cost_giou * cost_giou.cpu()
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [sum([v.shape[1] - 1, q.shape[1] - 1]) for v, q in
                 zip(targets[1:self.batch_size + 1], targets[1 + self.batch_size:])]
        indices = [linear_sum_assignment(c[i]) if sizes[i] != 0 else (
            np.array([-1]), np.array([-1])) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # cost_bbox_center = torch.cdist(out_bbox_center, tgt_bbox_center, p=1)
        # tgt = tgt_bbox_points.view(-1, 4, 2)
        # ll = [torch.cdist(out_bbox_points, tgt[:, i, :].view(-1, 8),
        #                   p=1).unsqueeze(-1) for i in self.list]
        # l2 = torch.cat(ll, axis=-1)
        # cost_bbox_points, _ = torch.min(l2, dim=-1)
        # C = self.cost_bbox_center * cost_bbox_center + self.cost_class * \
        #     cost_class+self.cost_bbox_points*cost_bbox_points
        # C = C.view(bs, num_queries, -1).cpu()
        # sizes = [sum([v.shape[1] - 1, q.shape[1] - 1]) for v, q in
        #          zip(targets[1:self.batch_size + 1], targets[1 + self.batch_size:])]
        # indices = [linear_sum_assignment(c[i]) if sizes[i] != 0 else (
        #     np.array([-1]), np.array([-1])) for i, c in enumerate(C.split(sizes, -1))]
        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher5(set_cost_class, set_cost_bbox, set_cost_giou, batch_size):
    return HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou, batch_size=batch_size)
