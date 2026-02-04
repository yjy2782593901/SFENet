import torch
import numpy as np
from skimage.morphology import dilation, disk


def get_gt_edge(gt, radius=2, method='dilation'):
    device = gt.device
    dtype = gt.dtype

    if gt.dim() == 3:
        gt = gt.unsqueeze(1)

    gt_binary = (gt > 0.5).cpu().numpy().astype(np.uint8)
    batch_size = gt_binary.shape[0]

    edges = []
    for i in range(batch_size):
        mask = gt_binary[i, 0]

        if mask.sum() == 0:
            edges.append(np.zeros_like(mask, dtype=np.float32))
            continue

        if method == 'dilation':
            dilated = dilation(mask, disk(radius))
            edge = (dilated - mask).astype(np.float32)
        elif method == 'erosion':
            from skimage.morphology import erosion
            eroded = erosion(mask, disk(radius))
            edge = (mask - eroded).astype(np.float32)
        elif method == 'both':
            from skimage.morphology import erosion
            dilated = dilation(mask, disk(radius))
            eroded = erosion(mask, disk(radius))
            edge = (dilated - eroded).astype(np.float32)
        else:
            raise ValueError(f"Unknown method: {method}")

        edges.append(edge)

    edges = np.stack(edges)[:, None, :, :]
    edges = torch.from_numpy(edges).to(device=device, dtype=dtype)

    return edges


def get_gt_edge_batch(gt_batch, radius=2, method='dilation'):
    return get_gt_edge(gt_batch, radius=radius, method=method)


if __name__ == '__main__':
    gt = torch.zeros(2, 1, 128, 128)
    gt[0, 0, 40:60, 50:55] = 1.0
    gt[0, 0, 70:75, 30:90] = 1.0
    gt[1, 0, 20:100, 60:65] = 1.0

    print(f"Input GT shape: {gt.shape}")
    print(f"Crack pixels: {gt.sum().item()}")

    for method in ['dilation', 'erosion', 'both']:
        edge = get_gt_edge(gt, radius=2, method=method)
        print(f"{method}: edge shape {edge.shape}, edge pixels {edge.sum().item()}")
