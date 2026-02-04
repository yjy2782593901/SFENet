import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CXR, Bott_CXR
from .encoder import SFEncoder
from .decoder import SEAM, EPRM, FEIM


class SFENet(nn.Module):
    def __init__(self, embed_dim=[16, 32, 64, 128, 256], depth=[3, 3, 3, 3, 3], input_size=512):
        super(SFENet, self).__init__()

        use_gn = (input_size > 256)
        CXR.use_gn = use_gn
        Bott_CXR.use_gn = use_gn

        self.encoder = SFEncoder(
            in_chans=3,
            patch_size=1,
            embed_dim=embed_dim,
            depth=depth,
            embed_kernel_size=3
        )

        self.seam = SEAM(embed_dim[1], embed_dim[4])

        self.eprm1 = EPRM(embed_dim[1])
        self.eprm2 = EPRM(embed_dim[2])
        self.eprm3 = EPRM(embed_dim[3])
        self.eprm4 = EPRM(embed_dim[4])

        self.feim1 = FEIM(embed_dim[2], embed_dim[1])
        self.feim2 = FEIM(embed_dim[3], embed_dim[2])
        self.feim3 = FEIM(embed_dim[4], embed_dim[3])
        self.feim4 = FEIM(embed_dim[4], embed_dim[4])

        self.predictor1 = nn.Conv2d(embed_dim[1], 1, 1)
        self.predictor2 = nn.Conv2d(embed_dim[2], 1, 1)
        self.predictor3 = nn.Conv2d(embed_dim[3], 1, 1)
        self.predictor4 = nn.Conv2d(embed_dim[4], 1, 1)

    def forward(self, x):
        x_list = self.encoder(x)
        x1 = x_list[0]
        x2 = x_list[1]
        x3 = x_list[2]
        x4 = x_list[3]
        x5 = x_list[4]

        edge = self.seam(x1, x4)
        edge_att = torch.sigmoid(edge)

        x1r = self.eprm1(x1, edge_att)
        x2r = self.eprm2(x2, edge_att)
        x3r = self.eprm3(x3, edge_att)
        x4r = self.eprm4(x4, edge_att)

        x45 = self.feim4(x4r, x5)
        x345 = self.feim3(x3r, x45)
        x2345 = self.feim2(x2r, x345)
        x12345 = self.feim1(x1r, x2345)

        o4 = self.predictor4(x45)
        o4 = F.interpolate(o4, size=x.shape[2:], mode='bilinear', align_corners=False)
        o3 = self.predictor3(x345)
        o3 = F.interpolate(o3, size=x.shape[2:], mode='bilinear', align_corners=False)
        o2 = self.predictor2(x2345)
        o2 = F.interpolate(o2, size=x.shape[2:], mode='bilinear', align_corners=False)
        o1 = self.predictor1(x12345)
        o1 = F.interpolate(o1, size=x.shape[2:], mode='bilinear', align_corners=False)

        oe = F.interpolate(edge_att, size=x.shape[2:], mode='bilinear', align_corners=False)

        if self.training:
            return o1, o2, o3, o4, oe
        else:
            return o1, oe


def build_sfenet(input_size=512, embed_dim=None, depth=None):
    if embed_dim is None:
        embed_dim = [16, 32, 64, 128, 256]
    if depth is None:
        depth = [3, 3, 3, 3, 3]
    return SFENet(embed_dim=embed_dim, depth=depth, input_size=input_size)


if __name__ == "__main__":
    from thop import profile
    from torchinfo import summary

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn((1, 3, 512, 512)).to(device)
    model = SFENet(embed_dim=[16, 32, 64, 128, 256], depth=[3, 3, 3, 3, 3], input_size=512).to(device)

    summary(model, input_size=(1, 3, 512, 512), depth=4, device=device)

    flops, params = profile(model, (x,))
    print('FLOPs: %.2f G, Params: %.2f M' % (flops / 1e9, params / 1e6))

    out1, out2, out3, out4, edge = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: o1={out1.shape}, o2={out2.shape}, o3={out3.shape}, o4={out4.shape}, edge={edge.shape}")
