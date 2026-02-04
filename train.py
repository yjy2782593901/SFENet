import time
import torch
import numpy as np
from tqdm import tqdm
from thop import profile
import argparse
import os
import datetime
import random
import copy

from torch.utils.data import DataLoader
from dataset.dataprocessing import *
from utils.logger import get_logger
from utils.early_stoping import EarlyStopping
from eval.evaluate import evaluate_online

from models.sfenet import SFENet
from models.losses import bic_iou, DiceLoss
from utils.edge_utils import get_gt_edge

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

model = SFENet(embed_dim=[16, 32, 64, 128, 256], depth=[3, 3, 3, 3, 3]).to(device)


def print_time(msg):
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{current_time}] {msg}")


parser = argparse.ArgumentParser('SFENet Training')

parser.add_argument('--dataset_path', type=str, default="datasets/CrackMap")
parser.add_argument('--load_width', type=int, default=512)
parser.add_argument('--load_height', type=int, default=512)

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=200)

parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)

parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'multistep'])
parser.add_argument('--step_size', type=int, default=20)
parser.add_argument('--milestones', type=int, nargs='+', default=[30, 60, 90])
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--T_0', type=int, default=10)
parser.add_argument('--T_mult', type=int, default=2)
parser.add_argument('--eta_min', type=float, default=1e-6)

parser.add_argument('--edge_weight', type=float, default=1.0)
parser.add_argument('--edge_radius', type=int, default=2)
parser.add_argument('--aux_weight1', type=float, default=0.5)
parser.add_argument('--aux_weight2', type=float, default=0.5)

parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--min_delta', type=float, default=0.0005)

parser.add_argument('--exp_name', type=str, default='test_oe')
parser.add_argument('--description', type=str, default='')

parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

print("=" * 80)
print_time("Creating model: SFENet")
print("=" * 80)

criterion = bic_iou().to(device)
edge_criterion = DiceLoss(smooth=1.0).to(device)

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")

if args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
elif args.scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
elif args.scheduler == 'multistep':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
else:
    raise ValueError(f"Unknown scheduler: {args.scheduler}")

cur_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
exp_suffix = f"_{args.exp_name}" if args.exp_name else ""
output_dir = f'out/{os.path.basename(args.dataset_path)}/{cur_time}{exp_suffix}/'
os.makedirs(output_dir, exist_ok=True)
log = get_logger(output_dir, 'train')

dummy_input = torch.randn(1, 3, args.load_height, args.load_width, device=device)
flops, params = profile(copy.deepcopy(model), inputs=(dummy_input,), verbose=False)
args.gflops = f"{flops / 1e9:.2f}G"
args.m_params = f"{params / 1e6:.2f}M"
print_time(f"Model complexity: {args.gflops} FLOPs, {args.m_params} Params")
log.info(f"Model complexity: {args.gflops} FLOPs, {args.m_params} Params")

print("\n" + "=" * 80)
print_time("Loading dataset")
print("=" * 80)

TrainingSet = RoadDataset(f'{args.dataset_path}/train_img', f'{args.dataset_path}/train_lab',
                          args.load_width, args.load_height, augment=True)
train_dataLoader = DataLoader(TrainingSet, batch_size=args.batch_size, shuffle=True)
print(f'Training dataset size = {len(train_dataLoader)}')
log.info(f'Training dataset size = {len(train_dataLoader)}')

ValSet = RoadDataset(f'{args.dataset_path}/val_img', f'{args.dataset_path}/val_lab',
                     args.load_width, args.load_height)
val_dataLoader = DataLoader(ValSet, batch_size=args.batch_size, shuffle=False)
print(f'Validation dataset size = {len(val_dataLoader)}')
log.info(f'Validation dataset size = {len(val_dataLoader)}')

TestSet = RoadDataset(f'{args.dataset_path}/test_img', f'{args.dataset_path}/test_lab',
                      args.load_width, args.load_height)
test_dataLoader = DataLoader(TestSet, batch_size=1, shuffle=False)
print(f'Test dataset size = {len(test_dataLoader)}')
log.info(f'Test dataset size = {len(test_dataLoader)}')

monitor_name = 'ODS_F1+OIS_F1'
early_stopper = EarlyStopping(
    patience=args.patience,
    min_delta=args.min_delta,
    monitor=monitor_name,
    lr_patience=10,
    lr_factor=0.5,
    restore_best_weights=True,
    mode='max'
)
print_time(f"Early stopping: monitor '{monitor_name}', patience={args.patience}")
log.info(f"Early stopping: monitor '{monitor_name}', patience={args.patience}")

start_time = time.time()
max_Metrics = {'epoch': -1, 'mIoU': 0, 'ODS_F1': 0, 'ODS_P': 0, 'ODS_R': 0,
               'OIS_F1': 0, 'OIS_P': 0, 'OIS_R': 0}

print("\n" + "=" * 80)
print_time("Start training")
print("=" * 80)
log.info("=" * 80)
log.info("Start training")
log.info("=" * 80)

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_dataLoader, desc=f'Epoch {epoch + 1}/{args.epochs} Training')

    for i, data in enumerate(train_dataLoader):
        samples = data['image'].to(device)
        targets = data['mask'].to(device)

        targets[targets > 0] = 1
        targets = targets.float()

        outputs = model(samples)
        Mask1, Mask2, Mask3, Mask4, edge_pred = outputs
        Mask1 = Mask1.float()
        Mask2 = Mask2.float()
        Mask3 = Mask3.float()
        Mask4 = Mask4.float()
        edge_pred = edge_pred.float()

        edges_gt = get_gt_edge(targets, radius=args.edge_radius, method='dilation')

        seg_loss = criterion(Mask1, targets) + \
                   0.5 * criterion(Mask2, targets) + \
                   0.5 * criterion(Mask3, targets) + \
                   0.5 * criterion(Mask4, targets)

        edge_loss = edge_criterion(edge_pred, edges_gt)

        loss = seg_loss + args.edge_weight * edge_loss

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bar.set_description(
            f"Loss: {loss.item():.4f} (Seg: {seg_loss.item():.4f}, Edge: {edge_loss.item():.4f})"
        )
        train_bar.update(1)

    train_bar.close()
    avg_train_loss = train_loss / len(train_dataLoader)
    print_time(f'Epoch {epoch + 1}/{args.epochs}, Training Loss: {avg_train_loss:.4f}')
    log.info(f'Epoch {epoch + 1}/{args.epochs}, Training Loss: {avg_train_loss:.4f}')

    scheduler.step()

    print(f'\n{"=" * 80}')
    print_time(f'Epoch {epoch + 1} Validation')
    print("=" * 80)
    log.info(f'{"=" * 80}')
    log.info(f'Epoch {epoch + 1} Validation')
    log.info("=" * 80)

    model.eval()
    val_metrics = evaluate_online(model, val_dataLoader, device, epoch)

    for key, value in val_metrics.items():
        if isinstance(value, float):
            print_time(f'Validation {key} -> {value:.4f}')
            log.info(f"Epoch {epoch} | Validation {key} -> {value:.4f}")

    eval_val = val_metrics.get('ODS_F1', 0) + val_metrics.get('OIS_F1', 0)

    should_stop = early_stopper(
        current_score=eval_val,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        output_dir=output_dir,
        logger=log,
        extra_save_info={'args': args}
    )

    if early_stopper.best_epoch == epoch:
        max_Metrics = val_metrics

    if should_stop:
        print_time("Early stopping triggered.")
        log.info("Early stopping triggered.")
        break

print("\n" + "=" * 80)
print_time("Best validation performance")
print("=" * 80)
log.info("=" * 80)
log.info("Best validation performance")
log.info("=" * 80)

best_metrics_log = {}
for key, value in max_Metrics.items():
    log_key = f'best_val_{key}'
    if isinstance(value, float):
        print_time(f'Best Validation {key} -> {value:.4f}')
        log.info(f'Best Validation {key} -> {value:.4f}')
        best_metrics_log[log_key] = value
    else:
        print_time(f'Best Validation Epoch -> {value}')
        log.info(f'Best Validation Epoch -> {value}')

print("\n" + "=" * 80)
print_time("Test evaluation")
print("=" * 80)
log.info("=" * 80)
log.info("Test evaluation")
log.info("=" * 80)

best_model_path = os.path.join(output_dir, 'checkpoint_best.pth')
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint['model'])
best_epoch = checkpoint['epoch']
print_time(f"Loaded best model from epoch {best_epoch} for final test.")
log.info(f"Loaded best model from epoch {best_epoch} for final test.")

final_results_path = os.path.join(output_dir, 'final_test_results')
final_test_metrics = evaluate_online(
    model=model,
    data_loader=test_dataLoader,
    device=device,
    epoch=best_epoch,
    save_path=final_results_path
)

test_metrics_log = {}
print_time("Final model performance on test set:")
log.info("Final model performance on test set:")
for key, value in final_test_metrics.items():
    log_key = f'test_{key}'
    if isinstance(value, float):
        print_time(f'Test {key} -> {value:.4f}')
        log.info(f'Test {key} -> {value:.4f}')
        test_metrics_log[log_key] = value

if early_stopper:
    summary = early_stopper.get_summary()
    summary_str = (
        f"\n{'='*80}\n"
        f"Early Stopping Summary\n"
        f"{'='*80}\n"
        f"Monitor: {summary['monitor_metric']}\n"
        f"Best Score: {summary['best_score']:.6f}\n"
        f"Best Epoch: {summary['best_epoch']}\n"
        f"Stopped Epoch: {summary['stopped_epoch']}\n"
        f"LR Reductions: {summary['lr_reductions']}\n"
        f"{'='*80}"
    )
    print(summary_str)
    log.info(summary_str)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print_time(f'Total time: {total_time_str}')
log.info(f'Total time: {total_time_str}')

print("\n" + "=" * 80)
print_time("Training completed!")
print("=" * 80)
