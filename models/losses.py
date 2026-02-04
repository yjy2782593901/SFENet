import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEIoULoss(nn.Module):
    def __init__(self, bce_weight=3.0, iou_weight=1.0):
        super(BCEIoULoss, self).__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight

    def forward(self, pred, true):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(true, kernel_size=31, stride=1, padding=15) - true)

        wbce = F.binary_cross_entropy_with_logits(pred, true, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * true) * weit).sum(dim=(2, 3))
        union = ((pred + true) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (self.bce_weight * wbce + self.iou_weight * wiou).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class CombinedLoss(nn.Module):
    def __init__(self, edge_weight=1.0, aux_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.seg_loss = BCEIoULoss()
        self.edge_loss = DiceLoss(smooth=1.0)
        self.edge_weight = edge_weight
        self.aux_weight = aux_weight

    def forward(self, outputs, targets, edge_targets=None):
        o1, o2, o3, o4, oe = outputs

        seg_loss = self.seg_loss(o1, targets)

        aux_loss = (self.seg_loss(o2, targets) +
                    self.seg_loss(o3, targets) +
                    self.seg_loss(o4, targets))

        if edge_targets is not None:
            edge_loss = self.edge_loss(oe, edge_targets)
        else:
            edge_loss = torch.tensor(0.0, device=o1.device)

        total_loss = seg_loss + self.aux_weight * aux_loss + self.edge_weight * edge_loss

        loss_dict = {
            'total': total_loss.item(),
            'seg': seg_loss.item(),
            'aux': aux_loss.item(),
            'edge': edge_loss.item() if edge_targets is not None else 0.0
        }

        return total_loss, loss_dict


bic_iou = BCEIoULoss
