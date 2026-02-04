import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.ops.carafe import normal_init, xavier_init


def carafe(x, normed_mask, kernel_size, group=1, up=1):
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape

    assert m_h == up * h
    assert m_w == up * w

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


def compute_local_similarity(features, window_size=3, dilation=1):
    B, C, H, W = features.shape

    padding = (window_size // 2) * dilation
    unfolded = F.unfold(features, kernel_size=window_size,
                       padding=padding, dilation=dilation)
    unfolded = unfolded.view(B, C, window_size ** 2, H, W)

    center_idx = window_size ** 2 // 2
    center_feat = unfolded[:, :, center_idx:center_idx+1, :, :]

    similarity = F.cosine_similarity(center_feat, unfolded, dim=1)

    similarity = torch.cat([
        similarity[:, :center_idx],
        similarity[:, center_idx+1:]
    ], dim=1)

    return similarity


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


class HighFrequencyDetailEnhancer(nn.Module):
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
        xavier_init(self.channel_compressor, distribution='normal')
        normal_init(self.lowpass_kernel_generator, std=0.001)

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


class HRGuidedContentAwareUpsampler(nn.Module):
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

            normal_init(self.lr_offset, std=0.001)
            normal_init(self.hr_offset, std=0.001)
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
        xavier_init(self.hr_compressor, distribution='uniform')
        xavier_init(self.lr_compressor, distribution='uniform')
        normal_init(self.kernel_gen_hr, std=0.001)
        normal_init(self.kernel_gen_lr, std=0.001)

    def _normalize_kernel(self, kernel_weights):
        B, KC, H, W = kernel_weights.shape
        K = self.kernel_size
        scale = self.scale
        groups = self.groups

        kernel_weights = kernel_weights.view(B, groups * scale ** 2, K ** 2, H, W)
        kernel_weights = F.softmax(kernel_weights, dim=2)
        kernel_weights = kernel_weights.view(B, groups * scale ** 2, K ** 2, H, W)
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
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 compressed_channels=64,
                 use_hfde=True,
                 use_hgcu=True,
                 hfde_kernel_size=3,
                 hgcu_kernel_size=5,
                 use_similarity_resample=True):
        super().__init__()

        self.hr_channels = hr_channels
        self.lr_channels = lr_channels
        self.use_hfde = use_hfde
        self.use_hgcu = use_hgcu

        if use_hfde:
            self.hfde = HighFrequencyDetailEnhancer(
                in_channels=hr_channels,
                compressed_channels=compressed_channels,
                kernel_size=hfde_kernel_size,
                use_residual=True
            )

        if use_hgcu:
            self.hgcu = HRGuidedContentAwareUpsampler(
                hr_channels=hr_channels,
                lr_channels=lr_channels,
                compressed_channels=compressed_channels,
                kernel_size=hgcu_kernel_size,
                scale=2,
                groups=1,
                use_similarity_resample=use_similarity_resample
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


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hr_feat = torch.randn(2, 128, 128, 128).to(device)
    lr_feat = torch.randn(2, 128, 64, 64).to(device)

    fusion = CrackFeatureFusion(
        hr_channels=128, lr_channels=128, compressed_channels=64,
        use_hfde=True, use_hgcu=True, use_similarity_resample=False
    ).to(device)

    hr_out, lr_out = fusion(hr_feat, lr_feat)
    print(f"Input:  HR {hr_feat.shape}, LR {lr_feat.shape}")
    print(f"Output: HR {hr_out.shape}, LR {lr_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in fusion.parameters()):,}")

    loss = hr_out.mean() + lr_out.mean()
    loss.backward()
    print("All tests passed")
