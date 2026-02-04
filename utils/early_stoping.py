import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union

class EarlyStopping:
    def __init__(self,
                 patience: int = 15,
                 min_delta: float = 0.001,
                 monitor: str = 'mIoU',
                 lr_patience: int = 8,
                 lr_factor: float = 0.5,
                 restore_best_weights: bool = True,
                 mode: str = 'max',
                 max_lr_reductions: int = 3,
                 verbose: bool = True):

        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.restore_best_weights = restore_best_weights
        self.mode = mode.lower()
        self.max_lr_reductions = max_lr_reductions
        self.verbose = verbose

        if self.mode not in ['max', 'min']:
            raise ValueError(f"Mode must be 'max' or 'min', but got '{mode}'")

        self.best_score = -np.inf if self.mode == 'max' else np.inf
        self.best_epoch = -1
        self.wait = 0
        self.lr_wait = 0
        self.lr_reduced_count = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.history = []

    def __call__(self,
                 current_score: float,
                 epoch: int,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: Optional[Any] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 logger: Optional[Any] = None,
                 extra_save_info: Optional[Dict] = None) -> bool:
        improved = self._is_improved(current_score)

        if improved:
            is_new_best = epoch != self.best_epoch
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            self.lr_wait = 0

            if is_new_best:
                if self.restore_best_weights:
                    self.best_weights = {key: value.cpu().clone() for key, value in model.state_dict().items()}
                if output_dir:
                    self._save_best_model(model, optimizer, lr_scheduler, epoch, output_dir, extra_save_info)
                self._log(f"New best {self.monitor}: {current_score:.6f} (epoch {epoch}). Model saved.", logger)

            return False
        else:
            self.wait += 1
            self.lr_wait += 1
            self._log(f"{self.monitor} no improvement: {current_score:.6f}, wait {self.wait}/{self.patience} epochs", logger)
            if self.lr_wait >= self.lr_patience and self.lr_reduced_count < self.max_lr_reductions:
                self._reduce_lr(optimizer, logger)
                self.lr_wait = 0
                self.lr_reduced_count += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self._log(f"Early stopping triggered at epoch {epoch}.", logger)
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    self._log(f"Restored best weights from epoch {self.best_epoch}.", logger)
                return True
        return False

    def _is_improved(self, score: float) -> bool:
        return (self.mode == 'max' and score > self.best_score + self.min_delta) or \
               (self.mode == 'min' and score < self.best_score - self.min_delta)

    def _reduce_lr(self, optimizer: torch.optim.Optimizer, logger: Optional[Any]):
        old_lr = optimizer.param_groups[0]['lr']
        new_lr = old_lr * self.lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        self._log(f"LR reduced: {old_lr:.2e} -> {new_lr:.2e} (#{self.lr_reduced_count})", logger)

    def _save_best_model(self, model, optimizer, lr_scheduler, epoch, output_dir, extra_save_info):
        checkpoint_path = Path(output_dir) / 'checkpoint_best.pth'
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
            'epoch': epoch,
            'best_score': self.best_score,
            'monitor': self.monitor,
            **(extra_save_info or {})
        }
        torch.save(save_dict, checkpoint_path)

    def _log(self, message: str, logger: Optional[Any]):
        if self.verbose:
            print(message)
        if logger:
            logger.info(message)

    def get_summary(self) -> Dict[str, Any]:
        return {
            'monitor_metric': self.monitor,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'stopped_epoch': self.stopped_epoch if self.stopped_epoch > 0 else 'N/A',
            'lr_reductions': self.lr_reduced_count,
        }
