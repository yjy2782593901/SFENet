import numpy as np
import os
import cv2
import torch
from tqdm import tqdm
from pathlib import Path


def get_statistics(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp, fp, fn


def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')

            tp = np.sum((pred_img == 1) & (gt_img == 1))
            fp = np.sum((pred_img == 1) & (gt_img == 0))
            fn = np.sum((pred_img == 0) & (gt_img == 1))
            tn = np.sum((pred_img == 0) & (gt_img == 0))

            iou_fg = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0
            iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) != 0 else 0

            iou_list.append((iou_fg + iou_bg) / 2)

        final_iou.append(np.mean(iou_list))

    return np.max(final_iou)

def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    metrics_by_thresh = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        p_list, r_list, f1_list = [], [], []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)

        metrics_by_thresh.append({
            'f1': np.mean(f1_list),
            'p': np.mean(p_list),
            'r': np.mean(r_list),
            'thresh': thresh
        })

    best = max(metrics_by_thresh, key=lambda x: x['f1'])
    return best['f1'], best['p'], best['r'], best['thresh']


def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    best_f1_list, best_p_list, best_r_list = [], [], []

    for pred, gt in zip(pred_list, gt_list):
        metrics_by_thresh = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            metrics_by_thresh.append({'f1': f1, 'p': p, 'r': r})

        best = max(metrics_by_thresh, key=lambda x: x['f1'])
        best_f1_list.append(best['f1'])
        best_p_list.append(best['p'])
        best_r_list.append(best['r'])

    return np.mean(best_f1_list), np.mean(best_p_list), np.mean(best_r_list)


def evaluate_online(model, data_loader, device, epoch, save_path=None, model_outputs_logits=True):
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    model.eval()
    all_pred_probs = []
    all_edge_probs = []
    all_gt_maps = []
    all_filenames = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"Validating Epoch {epoch + 1}"):
            x = data["image"].to(device)
            target = data.get('label', data.get('mask')).to(device)
            raw_output = model(x)

            if isinstance(raw_output, tuple):
                if len(raw_output) == 4:
                    pred_logits = raw_output[0]
                    edge_logits = raw_output[3]
                elif len(raw_output) == 2:
                    pred_logits = raw_output[0]
                    edge_logits = raw_output[1]
                else:
                    pred_logits = raw_output[0]
                    edge_logits = None
            else:
                pred_logits = raw_output
                edge_logits = None

            if model_outputs_logits:
                pred_probs = torch.sigmoid(pred_logits)
            else:
                pred_probs = pred_logits

            target[target > 0] = 1
            all_pred_probs.extend([p.cpu() for p in pred_probs])
            all_gt_maps.extend([g.cpu() for g in target])

            if edge_logits is not None:
                all_edge_probs.extend([e.cpu() for e in edge_logits])

            if "A_paths" in data:
                all_filenames.extend([Path(p).stem for p in data["A_paths"]])

    if not all_pred_probs:
        print("Warning: No evaluation data, skipping metrics calculation.")
        return {'epoch': epoch + 1}

    pred_numpy_list = [(p[0].numpy() * 255) for p in all_pred_probs]
    gt_numpy_list = [(g[0].numpy() * 255) for g in all_gt_maps]

    mIoU = cal_mIoU_metrics(pred_numpy_list, gt_numpy_list)
    ods_f1, ods_p, ods_r, ods_threshold = cal_ODS_metrics(pred_numpy_list, gt_numpy_list)
    ois_f1, ois_p, ois_r = cal_OIS_metrics(pred_numpy_list, gt_numpy_list)

    if save_path:
        print(f"\nSaving predictions with threshold {ods_threshold:.4f} to: {save_path}")
        for i in tqdm(range(len(pred_numpy_list)), desc="Saving images"):
            name = all_filenames[i] if i < len(all_filenames) else f"image_{i:04d}"

            prob_map = pred_numpy_list[i].astype(np.uint8)
            binary_map = (prob_map > (ods_threshold * 255)).astype(np.uint8) * 255
            gt_map = gt_numpy_list[i].astype(np.uint8)

            cv2.imwrite(os.path.join(save_path, f"{name}_pre.png"), binary_map)
            cv2.imwrite(os.path.join(save_path, f"{name}_lab.png"), gt_map)
            cv2.imwrite(os.path.join(save_path, f"{name}_prob.png"), prob_map)

            if all_edge_probs and i < len(all_edge_probs):
                edge_map = (all_edge_probs[i][0].numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, f"{name}_edge.png"), edge_map)

    return {
        'epoch': epoch + 1,
        'mIoU': mIoU,
        'ODS_F1': ods_f1,
        'ODS_P': ods_p,
        'ODS_R': ods_r,
        'ODS_Threshold': ods_threshold,
        'OIS_F1': ois_f1,
        'OIS_P': ois_p,
        'OIS_R': ois_r,
    }

