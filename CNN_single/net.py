import torch
import torch.nn as nn
from torch.nn import parallel
import torch.nn.functional as F


class WTA(nn.Module):
    def __init__(self):
        super(WTA, self).__init__()

    def forward(self, x):
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
        out = x.new().resize_(num, 1, height, width).zero_()
        for i in range(num):
            out[i, :, :, :] = torch.argmax(x[i, :, :, :], 0)
        return out


class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        in_layers = 2
        layers = 32
        filter_size = 3
        padding = int((filter_size - 1) / 2)
        self.conv = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(in_layers, layers, filter_size,
                                            stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, 1, filter_size, stride=1, padding=padding))

    def forward(self, x):
        out = self.conv(x)
        return out


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.wta = WTA()
        in_layers = 3
        layers = 32
        filter_size = 3
        padding = int((filter_size-1)/2)

        self.init = nn.Conv2d(
            in_layers, layers, filter_size, stride=1, padding=padding)

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size,
                                            stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers * 2, filter_size, stride=1, padding=padding))

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers * 2, layers * 2,
                                            filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers * 2, layers * 2, filter_size, stride=1, padding=padding))

        self.enc3 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers * 2, layers * 2,
                                            filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers * 2, layers * 2, filter_size, stride=1, padding=padding))

        self.enc4 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers * 2, layers * 2,
                                            filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers * 2, layers * 2, filter_size, stride=1, padding=padding))

        self.enc5 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers * 2, layers * 2,
                                            filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers * 2, layers * 2, filter_size, stride=1, padding=padding))

        self.dec5 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers * 2, layers * 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding))

        self.dec4 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers * 2, layers * 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding))

        self.dec3 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers * 2, layers * 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding))

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers*2, layers*2, filter_size, stride=2, padding=padding, output_padding=padding))

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers*2, layers, filter_size, stride=2, padding=padding, output_padding=padding))

        self.prdct = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(layers, layers, filter_size,
                                             stride=1, padding=padding),
                                   nn.ReLU(),
                                   nn.Conv2d(layers, 2, filter_size, stride=1, padding=padding))

        self.fc1 = nn.Linear(1 * 192 * 192, 200)
        self.dropout = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, 2)

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input):

        # input
        input = self.init(input)

        # hourglass with short cuts connections between encoder and decoder
        x1 = self.enc1(input)  # 1/2 input size
        x2 = self.enc2(x1)  # 1/4 input size
        x3 = self.enc3(x2)  # 1/8 input size
        x4 = self.enc4(x3)  # 1/16 input size
        x5 = self.enc5(x4)  # 1/32 input size
        x6 = self.dec5(x5)  # 1/16 input size
        x7 = self.dec4(x6+x4)  # 1/8 input size
        x8 = self.dec3(x7+x3)  # 1/4 input size
        x9 = self.dec2(x8+x2)  # 1/2 input size
        x10 = self.dec1(x9+x1)  # 1/1 input size

        # prediction
        output = self.prdct(x10+input)

        result = self.wta(output)

        # result = self.fc1(result)
        # result = self.fc2(result)
        # result = result.view(-1, 12*12)
        # result = F.relu(self.fc3(result))
        # result = F.relu(self.fc4(result))

        result = result.view(-1, 1 * 192 * 192)
        result = F.relu(self.fc1(result))
        # print("4", out.shape)
        # result = self.dropout(result)
        result = F.relu(self.fc2(result))
        result = self.fc3(result)

        # print("output ", output, output.size())
        # print("result ", result, result.size())

        return result
