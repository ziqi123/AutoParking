import torch
import torch.nn as nn

from .py_utils import kp, AELoss, convolution, residual
from config import system_configs

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width,stride)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride=1)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class model(kp):
    def __init__(self, db, flag=False, freeze=False,params=None):
        #create PSTR model
        """
        res18  BasicBlock [2, 2, 2, 2]
        res34  BasicBlock [3, 4, 6, 3]
        res50  Bottleneck [3, 4, 6, 3]
        res101 Bottleneck [3, 4, 23, 3]
        res152 Bottleneck [3, 8, 36, 3]
        """
        layers      = params['layers']
        res_dims    = params['res_dims']
        res_strides = params['res_strides']
        attn_dim    = params['attn_dim']
        dim_feedforward = params['dim_feedforward']

        num_queries = params['num_queries']  # number of joints
        drop_out    = params['drop_out']
        num_heads   = params['num_heads']
        enc_layers  = params['enc_layers']
        dec_layers  = params['dec_layers']
        kps_dim     = params['kps_dim']
        cls_dim=params['cls_dim']
        mlp_layers  = params['mlp_layers']

        aux_loss    = params['aux_loss']
        pos_type    = params['pos_type']
        pre_norm    = params['pre_norm']
        return_intermediate = params['return_intermediate']

        super(model, self).__init__(
            flag=flag,
            layers=layers,
            block=BasicBlock,
            res_dims=res_dims,
            res_strides=res_strides,
            attn_dim=attn_dim,
            num_queries=num_queries,
            aux_loss=aux_loss,
            pos_type=pos_type,
            drop_out=drop_out,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            return_intermediate=return_intermediate,
            kps_dim=kps_dim,
            cls_dim=cls_dim,
            mlp_layers=mlp_layers
        )

debug_path = system_configs.result_dir
bsize=system_configs.batch_size
class loss(AELoss):
    def __init__(self, params=None):
        #apply double matching loss
        super(loss, self).__init__(
           debug_path=debug_path,
           aux_loss=params['aux_loss'],
           dec_layers=params['dec_layers'],
           bsize=bsize
        )
