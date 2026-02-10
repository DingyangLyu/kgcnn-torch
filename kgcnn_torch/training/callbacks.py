"""PyTorch training callbacks for graph neural networks.

Provides a callback system similar to Keras callbacks but implemented purely
with PyTorch. Callbacks are invoked at various stages of the training loop
(epoch begin/end, batch begin/end, training begin/end).
"""
import logging
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class TrainingCallback:
    """Base class for training callbacks.

    A callback receives the model, optimizer, and training state at various
    points in the training loop. Subclass this and override the relevant
    methods to add custom behavior during training.

    Attributes:
        model: The PyTorch model being trained (set by the trainer).
        optimizer: The optimizer (set by the trainer).
    """

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def set_model(self, model: nn.Module):
        """Set the model reference. Called by the trainer before training starts.

        Args:
            model: The PyTorch model being trained.
        """
        self.model = model

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set the optimizer reference. Called by the trainer before training starts.

        Args:
            optimizer: The PyTorch optimizer being used.
        """
        self.optimizer = optimizer

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training.

        Args:
            logs: Dictionary that may contain training configuration info.
        """
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training.

        Args:
            logs: Dictionary containing final training metrics.
        """
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary that may contain metrics from previous epoch.
        """
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing metrics for this epoch,
                  e.g. {'train_loss': ..., 'val_loss': ..., 'lr': ...}.
        """
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each training batch.

        Args:
            batch: Current batch index.
            logs: Dictionary that may contain batch info.
        """
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each training batch.

        Args:
            batch: Current batch index.
            logs: Dictionary containing batch metrics, e.g. {'batch_loss': ...}.
        """
        pass


class LearningRateLoggingCallback(TrainingCallback):
    """Callback that logs the current learning rate at each epoch end.

    This callback reads the learning rate from the optimizer's parameter
    groups and records it in the logs dictionary under the 'lr' key.

    Args:
        verbose: Verbosity level. 0 = silent, 1 = print lr each epoch.
    """

    def __init__(self, verbose: int = 1):
        super().__init__()
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Read and log the learning rate at epoch end.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary to update with 'lr' key.
        """
        if self.optimizer is None:
            return

        logs = logs or {}
        current_lr = self.optimizer.param_groups[0]['lr']
        logs['lr'] = current_lr

        if self.verbose > 0:
            module_logger.info(
                "Epoch %05d: Finished epoch with learning rate: %.2e" % (epoch + 1, current_lr)
            )


class EarlyStoppingCallback(TrainingCallback):
    """Callback that stops training when a monitored metric has stopped improving.

    Training is stopped when the monitored metric does not improve for a given
    number of epochs (patience). Optionally restores the model weights from
    the epoch with the best value of the monitored metric.

    Args:
        patience: Number of epochs with no improvement after which training
            will be stopped. Default is 10.
        monitor: Name of the metric to monitor. Default is 'val_loss'.
        mode: One of 'min' or 'max'. In 'min' mode, training stops when the
            quantity monitored has stopped decreasing; in 'max' mode it stops
            when the quantity has stopped increasing. Default is 'min'.
        min_delta: Minimum change in the monitored metric to qualify as an
            improvement. Default is 0.0.
        restore_best_weights: Whether to restore model weights from the epoch
            with the best monitored metric. Default is True.
        verbose: Verbosity level. 0 = silent, 1 = messages on stop/restore.
    """

    def __init__(self, patience: int = 10, monitor: str = 'val_loss',
                 mode: str = 'min', min_delta: float = 0.0,
                 restore_best_weights: bool = True, verbose: int = 1):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self._best_value = None
        self._best_epoch = 0
        self._best_state_dict = None
        self._wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current value is an improvement over best.

        Args:
            current: Current metric value.
            best: Best metric value seen so far.

        Returns:
            True if current is an improvement.
        """
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Reset state at the beginning of training."""
        self._wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        self._best_value = None
        self._best_epoch = 0
        self._best_state_dict = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Check for improvement at epoch end.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing current metrics.
        """
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        if self._best_value is None or self._is_improvement(current, self._best_value):
            self._best_value = current
            self._best_epoch = epoch
            self._wait = 0
            if self.restore_best_weights and self.model is not None:
                self._best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                if self.verbose > 0:
                    module_logger.info(
                        "Early stopping at epoch %d. Best %s: %.6f at epoch %d." %
                        (epoch + 1, self.monitor, self._best_value, self._best_epoch + 1)
                    )

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Restore best weights at the end of training if early stopping was triggered."""
        if self.restore_best_weights and self._best_state_dict is not None:
            if self.model is not None:
                self.model.load_state_dict(self._best_state_dict)
                if self.verbose > 0:
                    module_logger.info(
                        "Restoring model weights from epoch %d." % (self._best_epoch + 1)
                    )


class ModelCheckpointCallback(TrainingCallback):
    """Callback that saves the model at specified intervals or when a metric improves.

    The model is saved as a PyTorch state_dict using torch.save. The filepath
    can contain named formatting options like '{epoch}' and '{val_loss}' which
    will be filled with the epoch number and metric values.

    Args:
        filepath: Path where the model checkpoint will be saved. May contain
            format placeholders. Default is 'checkpoint.pt'.
        monitor: Metric to monitor for saving the best model. Default is 'val_loss'.
        mode: One of 'min' or 'max'. Determines whether the monitored metric
            should be minimized or maximized. Default is 'min'.
        save_best_only: If True, only save when the monitored metric improves.
            Default is True.
        save_weights_only: If True, save only model.state_dict(). If False,
            save the full model. Default is True.
        verbose: Verbosity level. 0 = silent, 1 = log saves.
    """

    def __init__(self, filepath: str = 'checkpoint.pt', monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = True,
                 save_weights_only: bool = True, verbose: int = 1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        self._best_value = None

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current value is an improvement over best.

        Args:
            current: Current metric value.
            best: Best metric value seen so far.

        Returns:
            True if current is an improvement.
        """
        if self.mode == 'min':
            return current < best
        else:
            return current > best

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Reset best value at the beginning of training."""
        self._best_value = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Potentially save the model at epoch end.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing current metrics.
        """
        logs = logs or {}
        current = logs.get(self.monitor)

        # Build the filepath, substituting any format variables
        filepath = self.filepath
        try:
            format_dict = {'epoch': epoch + 1}
            format_dict.update(logs)
            filepath = filepath.format(**format_dict)
        except (KeyError, ValueError):
            pass

        if self.save_best_only:
            if current is None:
                return

            if self._best_value is None or self._is_improvement(current, self._best_value):
                self._best_value = current
                self._save_model(filepath, epoch)
        else:
            self._save_model(filepath, epoch)

    def _save_model(self, filepath: str, epoch: int):
        """Save the model to disk.

        Args:
            filepath: Full path where to save the model.
            epoch: Current epoch number (0-indexed).
        """
        if self.model is None:
            return

        # Ensure directory exists
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        if self.save_weights_only:
            torch.save(self.model.state_dict(), filepath)
        else:
            torch.save(self.model, filepath)

        if self.verbose > 0:
            module_logger.info("Epoch %05d: Saved model to %s" % (epoch + 1, filepath))
