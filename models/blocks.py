import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_sobel(in_chan, out_chan):
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(np.float32)
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = nn.Parameter(torch.from_numpy(filter_x), requires_grad=False)
    filter_y = nn.Parameter(torch.from_numpy(filter_y), requires_grad=False)

    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y

    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + 1e-6)
    return torch.sigmoid(g) * input


class Bott_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Bott_Conv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size,
                                   stride, padding, dilation, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class Bott_CBR(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(Bott_CBR, self).__init__()
        self.block = nn.Sequential(
            Bott_Conv(in_channels, out_channels, mid_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CXR(nn.Module):
    use_gn = False

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(CXR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)

        if CXR.use_gn:
            self.norm = nn.GroupNorm(self._get_num_groups(out_channels), out_channels)
        else:
            self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def _get_num_groups(self, channels):
        for num_groups in [32, 16, 8, 4, 2, 1]:
            if channels % num_groups == 0:
                return num_groups
        return 1

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Bott_CXR(nn.Module):
    use_gn = False

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(Bott_CXR, self).__init__()
        self.conv = Bott_Conv(in_channels, out_channels, mid_channels, kernel_size, stride, padding, dilation)

        if Bott_CXR.use_gn:
            self.norm = nn.GroupNorm(self._get_num_groups(out_channels), out_channels)
        else:
            self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def _get_num_groups(self, channels):
        for num_groups in [32, 16, 8, 4, 2, 1]:
            if channels % num_groups == 0:
                return num_groups
        return 1

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, patch_size=1, embed_dim=16, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2)

    def forward(self, x):
        return self.proj(x)


class DownSample(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.proj(x)
