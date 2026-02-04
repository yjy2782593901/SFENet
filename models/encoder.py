import torch
import torch.nn as nn

from .blocks import Bott_Conv, CXR, PatchEmbed, DownSample


class Fourier_Core(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fourier_Core, self).__init__()
        self.conv = Bott_Conv(in_channels * 2, out_channels * 2, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class Fourier_Unit(nn.Module):
    def __init__(self, dim):
        super(Fourier_Unit, self).__init__()
        self.conv = nn.Sequential(
            Bott_Conv(dim, dim, dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            Bott_Conv(dim, dim, dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim)
        )
        self.res_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.res_2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.fc = Fourier_Core(dim, dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x) + self.res_1(x))
        x = self.fc(x) + self.res_2(x)
        x = self.relu(self.bn(x))
        return x


class PSFM(nn.Module):
    def __init__(self, dim):
        super(PSFM, self).__init__()
        self.depthwise_1 = nn.Sequential(
            Bott_Conv(dim, dim, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.depthwise_2 = nn.Sequential(
            Bott_Conv(dim, dim, dim // 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.fu = Fourier_Unit(dim)
        self.depthwise = nn.Sequential(
            Bott_Conv(dim * 2, dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv_final = nn.Sequential(
            Bott_Conv(dim * 2, dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Conv2d(dim, dim, kernel_size=1)
        self.channel_attention_conv = nn.Sequential(
            Bott_Conv(dim, dim, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x0):
        x = x0
        x_global = self.fu(x0)
        x_local_1 = self.depthwise_1(x)
        x_local_2 = self.depthwise_2(x)
        x_local = torch.cat([x_local_1, x_local_2], dim=1)
        x_local = self.depthwise(x_local)
        x = torch.cat([x_global, x_local], dim=1)
        x = self.conv_final(x) + self.res(x0)
        x = self.channel_attention_conv(x)
        x = self.channel_attention(x) * x
        return x


class SFBlock(nn.Module):
    def __init__(self, dim):
        super(SFBlock, self).__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.psfm = PSFM(dim)

    def forward(self, x):
        return self.psfm(self.norm(x)) + x


class SFStage(nn.Module):
    def __init__(self, depth, dim):
        super(SFStage, self).__init__()
        self.blocks = nn.Sequential(*[SFBlock(dim) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)


class SFEncoder(nn.Module):
    def __init__(self, in_chans=3, patch_size=1, embed_dim=[16, 32, 64, 128, 256],
                 depth=[3, 3, 3, 3, 3], embed_kernel_size=3):
        super(SFEncoder, self).__init__()
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(in_chans=in_chans, patch_size=patch_size,
                                      embed_dim=embed_dim[0], kernel_size=embed_kernel_size)

        self.stage1 = SFStage(depth=depth[0], dim=embed_dim[0])
        self.downsample1 = DownSample(input_dim=embed_dim[0])

        self.stage2 = SFStage(depth=depth[1], dim=embed_dim[1])
        self.downsample2 = DownSample(input_dim=embed_dim[1])

        self.stage3 = SFStage(depth=depth[2], dim=embed_dim[2])
        self.downsample3 = DownSample(input_dim=embed_dim[2])

        self.stage4 = SFStage(depth=depth[3], dim=embed_dim[3])
        self.downsample4 = DownSample(input_dim=embed_dim[3])

        self.stage5 = SFStage(depth=depth[4], dim=embed_dim[4])

        self.cbr1 = CXR(embed_dim[0], embed_dim[1], kernel_size=3, stride=1, padding=1)
        self.cbr2 = CXR(embed_dim[1], embed_dim[2], kernel_size=3, stride=1, padding=1)
        self.cbr3 = CXR(embed_dim[2], embed_dim[3], kernel_size=3, stride=1, padding=1)
        self.cbr4 = CXR(embed_dim[3], embed_dim[4], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        layer_output = []
        x = self.patch_embed(x)

        x = self.stage1(x)
        x_ = self.cbr1(x)
        layer_output.append(x_)
        x = self.downsample1(x)

        x = self.stage2(x)
        x_ = self.cbr2(x)
        layer_output.append(x_)
        x = self.downsample2(x)

        x = self.stage3(x)
        x_ = self.cbr3(x)
        layer_output.append(x_)
        x = self.downsample3(x)

        x = self.stage4(x)
        x_ = self.cbr4(x)
        layer_output.append(x_)
        x = self.downsample4(x)

        x = self.stage5(x)
        layer_output.append(x)

        return layer_output
