import torch
import torch.nn as nn
from models.layers_384 import Conv, Hourglass, Residual


class Convert(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convert, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, nstack=8, layer=4, in_channel=256, out_channel=4, increase=0):
        super(PoseNet, self).__init__()
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, in_channel)
        )
        self.hourglass = nn.ModuleList([nn.Sequential(
            Hourglass(layer, in_channel, inc=increase)) for _ in range(nstack)])
        self.feature = nn.ModuleList([nn.Sequential(Residual(in_channel, in_channel), Conv(
            in_channel, in_channel, 1, bn=True, relu=True)) for _ in range(nstack)])
        self.outs = nn.ModuleList(
            [Conv(in_channel, out_channel, 1, bn=False, relu=False) for _ in range(nstack)])
        self.merge_feature = nn.ModuleList(
            [Convert(in_channel, in_channel) for _ in range(nstack - 1)])
        self.merge_pred = nn.ModuleList(
            [Convert(out_channel, in_channel) for _ in range(nstack - 1)])

    def forward(self, x):
        x = self.pre(x)
        heat_maps = []
        for i in range(self.nstack):
            hg = self.hourglass[i](x)
            feature = self.feature[i](hg)
            pred = self.outs[i](feature)
            heat_maps.append(pred)
            if i < self.nstack - 1:
                x = x + self.merge_pred[i](pred) + \
                    self.merge_feature[i](feature)
        return heat_maps
