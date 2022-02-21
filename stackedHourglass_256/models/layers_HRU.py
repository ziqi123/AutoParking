from torch import nn

Pool = nn.MaxPool2d


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size,
                              stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(
            x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class poolLayer(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(poolLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.relu = nn.ReLU()
        self.pool = Pool(2, 2)
        self.conv1 = nn.Conv2d(inp_dim, out_dim, 3, 3, 1, 1, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 3, 1, 1, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.up(out)
        return out


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class ResidualPool(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(ResidualPool, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        # poolLayer
        self.pool = Pool(2, 2)
        self.conv4 = nn.Conv2d(inp_dim, out_dim, 3, 3, 1, 1, 1, 1)
        self.bn4 = nn.BatchNorm2d(out_dim)
        self.conv5 = nn.Conv2d(out_dim, out_dim, 3, 3, 1, 1, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # skipLayer
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x

        poolLayer = self.bn1(x)
        poolLayer = self.relu(poolLayer)
        poolLayer = self.pool(poolLayer)
        poolLayer = self.conv4(poolLayer)
        poolLayer = self.bn4(poolLayer)
        poolLayer = self.relu(poolLayer)
        poolLayer = self.conv5(poolLayer)
        poolLayer = self.up(poolLayer)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual
        out += poolLayer
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = ResidualPool(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = ResidualPool(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2
