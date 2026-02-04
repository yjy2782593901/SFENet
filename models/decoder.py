import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log

from .blocks import get_sobel, run_sobel, Bott_CXR, CXR


def carafe(x, normed_mask, kernel_size, group=1, up=1):
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape

    pad = kernel_size // 2
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)

    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')

    unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)

    if m_c == kernel_size * kernel_size:
        normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    elif m_c == kernel_size * kernel_size * up * up:
        normed_mask = normed_mask.reshape(b, up * up, kernel_size * kernel_size, m_h, m_w)
        normed_mask = normed_mask.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unexpected mask channels: {m_c}")

    res = unfold_x * normed_mask
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)

    return res


def adaptive_kernel_aggregate(features, kernel_weights, kernel_size):
    B, C, H, W = features.shape
    pad = kernel_size // 2

    padded_feat = F.pad(features, pad=[pad] * 4, mode='reflect')
    unfolded = F.unfold(padded_feat, kernel_size=kernel_size, stride=1)

    unfolded = unfolded.view(B, C, kernel_size * kernel_size, H, W)
    kernel_weights = kernel_weights.view(B, 1, kernel_size * kernel_size, H, W)

    aggregated = (unfolded * kernel_weights).sum(dim=2)

    return aggregated


def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    B, C, H, W = input_tensor.shape

    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)
    unfold_tensor = unfold_tensor.reshape(B, C, k**2, H, W)

    if sim == 'cos':
        similarity = F.cosine_similarity(
            unfold_tensor[:, :, k * k // 2:k * k // 2 + 1],
            unfold_tensor[:, :, :],
            dim=1
        )
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)
    similarity = similarity.view(B, k * k - 1, H, W)
    return similarity


class HFDE(nn.Module):
    def __init__(self, in_channels, compressed_channels=64, kernel_size=3, use_residual=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.use_residual = use_residual
        self.channel_compressor = nn.Conv2d(in_channels, compressed_channels, 1, bias=False)
        self.lowpass_kernel_generator = nn.Conv2d(
            compressed_channels, kernel_size ** 2, 3, padding=1, bias=True
        )
        hamming_2d = np.outer(np.hamming(kernel_size), np.hamming(kernel_size))
        self.register_buffer('hamming_window', torch.FloatTensor(hamming_2d)[None, None, :, :])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.channel_compressor.weight)
        nn.init.normal_(self.lowpass_kernel_generator.weight, std=0.001)
        nn.init.zeros_(self.lowpass_kernel_generator.bias)

    def _normalize_kernel(self, kernel_weights):
        B, KK, H, W = kernel_weights.shape
        K = self.kernel_size
        kernel_weights = F.softmax(kernel_weights.view(B, KK, -1), dim=1).view(B, KK, H, W)
        kernel_weights = kernel_weights.view(B, H, W, K, K) * self.hamming_window
        kernel_weights = kernel_weights / (kernel_weights.sum(dim=(-1, -2), keepdim=True) + 1e-8)
        return kernel_weights.permute(0, 3, 4, 1, 2).reshape(B, KK, H, W)

    def forward(self, hr_features):
        compressed = self.channel_compressor(hr_features)
        lowpass_kernel = self.lowpass_kernel_generator(compressed)
        lowpass_kernel = self._normalize_kernel(lowpass_kernel)
        lowpass_result = adaptive_kernel_aggregate(hr_features, lowpass_kernel, self.kernel_size)
        highpass_result = hr_features - lowpass_result
        return hr_features + highpass_result if self.use_residual else highpass_result


class HGCU(nn.Module):
    def __init__(self, hr_channels, lr_channels, compressed_channels=64, kernel_size=5,
                 scale=2, groups=1, use_similarity_resample=False, local_window=3, sim_dilation=2):
        super().__init__()
        assert scale == 2

        self.kernel_size = kernel_size
        self.scale = scale
        self.groups = groups
        self.use_similarity_resample = use_similarity_resample
        self.local_window = local_window
        self.sim_dilation = sim_dilation

        self.hr_compressor = nn.Conv2d(hr_channels, compressed_channels, 1, bias=False)
        self.lr_compressor = nn.Conv2d(lr_channels, compressed_channels, 1, bias=False)

        self.kernel_gen_hr = nn.Conv2d(compressed_channels, kernel_size ** 2 * groups * scale ** 2, 3, padding=1)
        self.kernel_gen_lr = nn.Conv2d(compressed_channels, kernel_size ** 2 * groups * scale ** 2, 3, padding=1)

        if use_similarity_resample:
            self.lr_offset = nn.Conv2d(
                compressed_channels + local_window ** 2 - 1,
                2 * groups * scale ** 2,
                kernel_size=1
            )
            self.lr_direct_scale = nn.Conv2d(
                compressed_channels + local_window ** 2 - 1,
                2 * groups * scale ** 2,
                kernel_size=1
            )
            self.hr_offset = nn.Conv2d(
                compressed_channels + local_window ** 2 - 1,
                2 * groups,
                kernel_size=1
            )
            self.hr_direct_scale = nn.Conv2d(
                compressed_channels + local_window ** 2 - 1,
                2 * groups,
                kernel_size=1
            )

            nn.init.normal_(self.lr_offset.weight, std=0.001)
            nn.init.normal_(self.hr_offset.weight, std=0.001)
            nn.init.constant_(self.lr_direct_scale.weight, 0)
            nn.init.constant_(self.lr_direct_scale.bias, 0)
            nn.init.constant_(self.hr_direct_scale.weight, 0)
            nn.init.constant_(self.hr_direct_scale.bias, 0)

            h = torch.arange((-scale + 1) / 2, (scale - 1) / 2 + 1) / scale
            init_pos = torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2)
            init_pos = init_pos.repeat(1, groups, 1).reshape(1, -1, 1, 1)
            self.register_buffer('init_pos', init_pos)

        hamming_2d = np.outer(np.hamming(kernel_size), np.hamming(kernel_size))
        self.register_buffer('hamming_window', torch.FloatTensor(hamming_2d)[None, None, :, :])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.hr_compressor.weight)
        nn.init.xavier_uniform_(self.lr_compressor.weight)
        nn.init.normal_(self.kernel_gen_hr.weight, std=0.001)
        nn.init.normal_(self.kernel_gen_lr.weight, std=0.001)

    def _normalize_kernel(self, kernel_weights):
        B, KC, H, W = kernel_weights.shape
        K = self.kernel_size
        scale = self.scale
        groups = self.groups

        kernel_weights = kernel_weights.view(B, groups * scale ** 2, K ** 2, H, W)
        kernel_weights = F.softmax(kernel_weights, dim=2)
        kernel_weights = kernel_weights.view(B, groups * scale ** 2, H, W, K, K)
        kernel_weights = kernel_weights * self.hamming_window
        kernel_weights = kernel_weights / (kernel_weights.sum(dim=(-1, -2), keepdim=True) + 1e-8)
        kernel_weights = kernel_weights.view(B, groups * scale ** 2, H, W, K * K)
        kernel_weights = kernel_weights.permute(0, 1, 4, 2, 3)
        kernel_weights = kernel_weights.reshape(B, KC, H, W)

        return kernel_weights

    def _sample_with_offset(self, x, offset):
        B, _, H, W = offset.shape
        scale = self.scale
        groups = self.groups

        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij'), dim=0)
        coords = coords.transpose(1, 2).unsqueeze(1).unsqueeze(0)

        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(
            B, 2, -1, scale * H, scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(
            x.reshape(B * groups, -1, x.size(-2), x.size(-1)),
            coords,
            mode='bilinear',
            align_corners=False,
            padding_mode="border"
        ).view(B, -1, scale * H, scale * W)

    def forward(self, hr_features, lr_features):
        compressed_hr = self.hr_compressor(hr_features)
        compressed_lr = self.lr_compressor(lr_features)
        kernel_from_hr = self.kernel_gen_hr(compressed_hr)
        kernel_from_lr = self.kernel_gen_lr(compressed_lr)

        kernel_from_hr_normed = self._normalize_kernel(kernel_from_hr)
        kernel_from_lr_up = carafe(kernel_from_lr, kernel_from_hr_normed,
                                   self.kernel_size, self.groups, self.scale)

        fused_kernel = self._normalize_kernel(kernel_from_hr + kernel_from_lr_up)
        upsampled = carafe(lr_features, fused_kernel, self.kernel_size, self.groups, self.scale)

        if self.use_similarity_resample:
            hr_sim = compute_similarity(compressed_hr, k=self.local_window, dilation=self.sim_dilation, sim='cos')
            lr_sim = compute_similarity(compressed_lr, k=self.local_window, dilation=self.sim_dilation, sim='cos')
            hr_feat_sim = torch.cat([compressed_hr, hr_sim], dim=1)
            lr_feat_sim = torch.cat([compressed_lr, lr_sim], dim=1)

            offset = (self.lr_offset(lr_feat_sim) + F.pixel_unshuffle(self.hr_offset(hr_feat_sim), self.scale)) * \
                     (self.lr_direct_scale(lr_feat_sim) + F.pixel_unshuffle(self.hr_direct_scale(hr_feat_sim), self.scale)).sigmoid() + \
                     self.init_pos
            upsampled = self._sample_with_offset(upsampled, offset)

        return upsampled


class CrackFeatureFusion(nn.Module):
    def __init__(self, hr_channels, lr_channels, compressed_channels=64,
                 use_hfde=True, use_hgcu=True, hfde_kernel_size=3, hgcu_kernel_size=5):
        super().__init__()

        self.hr_channels = hr_channels
        self.lr_channels = lr_channels
        self.use_hfde = use_hfde
        self.use_hgcu = use_hgcu

        if use_hfde:
            self.hfde = HFDE(
                in_channels=hr_channels,
                compressed_channels=compressed_channels,
                kernel_size=hfde_kernel_size,
                use_residual=True
            )

        if use_hgcu:
            self.hgcu = HGCU(
                hr_channels=hr_channels,
                lr_channels=lr_channels,
                compressed_channels=compressed_channels,
                kernel_size=hgcu_kernel_size,
                scale=2,
                groups=1,
            )

    def forward(self, hr_features, lr_features):
        if self.use_hfde:
            hr_enhanced = self.hfde(hr_features)
        else:
            hr_enhanced = hr_features

        if self.use_hgcu:
            lr_upsampled = self.hgcu(hr_enhanced, lr_features)
        else:
            lr_upsampled = F.interpolate(
                lr_features,
                size=hr_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return hr_enhanced, lr_upsampled


class SEAM(nn.Module):
    def __init__(self, low_channels, high_channels):
        super(SEAM, self).__init__()

        self.sobel_x_low, self.sobel_y_low = get_sobel(low_channels, 1)
        self.sobel_x_high, self.sobel_y_high = get_sobel(high_channels, 1)

        self.reduce1 = CXR(low_channels, low_channels, 1, padding=0)
        self.reduce4 = CXR(high_channels, high_channels // 2, 1, padding=0)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(low_channels + high_channels // 2, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            nn.Sigmoid()
        )

        self.balance_factor = nn.Parameter(torch.ones(1) * 0.5)

        self.block = nn.Sequential(
            Bott_CXR(high_channels // 2 + low_channels, high_channels // 2, high_channels // 16, kernel_size=3, stride=1, padding=1),
            Bott_CXR(high_channels // 2, high_channels // 2, high_channels // 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(high_channels // 2, 1, 1)
        )

    def forward(self, x1, x4):
        low_edge_weighted = run_sobel(self.sobel_x_low, self.sobel_y_low, x1)
        high_edge_weighted = run_sobel(self.sobel_x_high, self.sobel_y_high, x4)

        size = low_edge_weighted.size()[2:]
        x1 = self.reduce1(low_edge_weighted)
        x4 = self.reduce4(high_edge_weighted)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)

        concat_features = torch.cat([x1, x4], dim=1)
        weights = self.channel_attention(concat_features)
        w1, w4 = weights[:, 0:1, :, :], weights[:, 1:2, :, :]

        weight_sum = w1 + w4
        scale = 2.0 / (weight_sum + 1e-8)

        x1 = x1 * (w1 * scale).expand_as(x1) * self.balance_factor
        x4 = x4 * (w4 * scale).expand_as(x4) * (1 - self.balance_factor)

        fused = torch.cat((x4, x1), dim=1)
        edge_prior = self.block(fused)

        return edge_prior


class EPRM(nn.Module):
    def __init__(self, dim):
        super(EPRM, self).__init__()
        t = int(abs((log(dim, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = Bott_CXR(dim, dim, dim // 8, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv(x)
        w = self.avg_pool(x)
        w = self.mlp(w.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        w = self.sigmoid(w)
        x = x * w

        return x


class FEIM(nn.Module):
    def __init__(self, hchannel, channel):
        super(FEIM, self).__init__()
        self.crack_fusion = CrackFeatureFusion(
            hr_channels=channel,
            lr_channels=hchannel,
            compressed_channels=min(channel, hchannel) // 2,
            use_hfde=True,
            use_hgcu=True,
            hfde_kernel_size=3,
            hgcu_kernel_size=3
        )
        self.conv1_1 = CXR(hchannel + channel, channel, 1, padding=0)
        self.conv3_1 = CXR(channel // 4, channel // 4, 3, padding=1)
        self.depthwise5_1 = CXR(channel // 4, channel // 4, 3, dilation=2, padding=2)
        self.depthwise7_1 = CXR(channel // 4, channel // 4, 3, dilation=3, padding=3)
        self.depthwise9_1 = CXR(channel // 4, channel // 4, 3, dilation=4, padding=4)
        self.conv1_2 = CXR(channel, channel, 1, padding=0)
        self.conv3_2 = CXR(channel, channel, 3, stride=1, padding=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.fusion_weight = nn.Parameter(torch.ones(2))

    def forward(self, lf, hf):
        hr_enhanced, lr_upsampled = self.crack_fusion(lf, hf)
        fusion_weight = F.softmax(self.fusion_weight, dim=0)
        x = torch.cat((hr_enhanced * fusion_weight[0], lr_upsampled * fusion_weight[1]), dim=1)

        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)

        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.depthwise5_1(xc[1] + x0 + xc[2])
        x2 = self.depthwise7_1(xc[2] + x1 + xc[3])
        x3 = self.depthwise9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_2(x + xx)

        ca = self.channel_attention(x)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_attention(spatial)
        x = x * spatial + x

        return x
