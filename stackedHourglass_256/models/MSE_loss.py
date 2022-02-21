import torch


class PAFLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """

    def __init__(self):
        super(PAFLoss, self).__init__()

    def forward(self, pred, gt):
        cof = 0.0001
        l = (pred - gt)*(pred - gt)*cof
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l  # l of dim bsize
