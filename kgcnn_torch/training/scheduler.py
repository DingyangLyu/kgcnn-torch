"""Learning rate schedulers for kgcnn-torch.

Wraps torch.optim.lr_scheduler with common patterns used in GNN training.
"""
import math
import torch.optim.lr_scheduler as lr_scheduler


class LinearWarmupScheduler(lr_scheduler.LambdaLR):
    """Linear warmup followed by constant learning rate.

    LR ramps linearly from a small fraction to base_lr over warmup_epochs,
    then stays constant. Uses (epoch+1) to avoid LR=0 at epoch 0.
    """

    def __init__(self, optimizer, warmup_epochs: int = 10,
                 last_epoch: int = -1):
        self.warmup_epochs = max(warmup_epochs, 1)

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(self.warmup_epochs)
            return 1.0

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class LinearWarmupExponentialDecay(lr_scheduler.LambdaLR):
    """Linear warmup + exponential decay.

    LR ramps linearly over warmup_epochs, then decays exponentially.
    """

    def __init__(self, optimizer, warmup_epochs: int = None,
                 decay_rate: float = 0.96, decay_epochs: int = None,
                 warmup_steps: int = None, decay_steps: int = None,
                 steps_per_epoch: int = None,
                 last_epoch: int = -1, **kwargs):
        # Accept both epoch-based and step-based parameter names for hyper config
        # compatibility. Keras configs use warmup_steps/decay_steps at the per-batch
        # step level, but this scheduler operates per-epoch. When warmup_epochs or
        # decay_epochs are given they are used directly. When only warmup_steps or
        # decay_steps are given, we convert them to epoch counts by dividing by
        # steps_per_epoch (default 30, a typical estimate for ~1000 samples with
        # batch_size=32). This avoids pathological behaviour where Keras step counts
        # (e.g. warmup_steps=3000) are mistakenly used as epoch counts.
        _spe = steps_per_epoch or int(kwargs.pop("batches_per_epoch", 30))
        if warmup_epochs is not None:
            self.warmup_epochs = max(warmup_epochs, 1)
        elif warmup_steps is not None:
            self.warmup_epochs = max(warmup_steps // _spe, 1)
        else:
            self.warmup_epochs = 10
        self.decay_rate = decay_rate
        if decay_epochs is not None:
            self.decay_epochs = decay_epochs
        elif decay_steps is not None:
            self.decay_epochs = max(decay_steps // _spe, 1)
        else:
            self.decay_epochs = 10

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(self.warmup_epochs)
            decay_steps = epoch - self.warmup_epochs
            return self.decay_rate ** (decay_steps / self.decay_epochs)

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class PolynomialDecayScheduler(lr_scheduler.LambdaLR):
    """Polynomial decay from initial to final learning rate.

    LR = (lr_init - lr_final) * (1 - step/total_steps)^power + lr_final
    """

    def __init__(self, optimizer, total_epochs: int = 500,
                 lr_final_factor: float = 0.01, power: float = 1.0,
                 last_epoch: int = -1):
        self.total_epochs = total_epochs
        self.lr_final_factor = lr_final_factor
        self.power = power

        def lr_lambda(epoch):
            frac = min(float(epoch) / float(total_epochs), 1.0)
            return (1.0 - lr_final_factor) * (1.0 - frac) ** power + lr_final_factor

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class CosineWarmupScheduler(lr_scheduler.LambdaLR):
    """Linear warmup + cosine annealing.

    LR ramps linearly over warmup_epochs, then follows cosine decay.
    """

    def __init__(self, optimizer, warmup_epochs: int = 10,
                 total_epochs: int = 500, min_lr_factor: float = 0.01,
                 last_epoch: int = -1):
        self.warmup_epochs = max(warmup_epochs, 1)
        self.total_epochs = total_epochs
        self.min_lr_factor = min_lr_factor

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(self.warmup_epochs)
            progress = float(epoch - self.warmup_epochs) / float(
                max(total_epochs - self.warmup_epochs, 1))
            return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (
                1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class LinearLearningRateScheduler(lr_scheduler.LambdaLR):
    """Linear learning rate decay (matches Keras LinearLearningRateScheduler).

    Keeps learning_rate_start constant until epoch epo_min, then linearly
    decays to learning_rate_stop by epoch epo.

    This matches the Keras kgcnn>LinearLearningRateScheduler behavior.
    """

    def __init__(self, optimizer, learning_rate_start: float = 1e-3,
                 learning_rate_stop: float = 1e-5, epo_min: int = 100,
                 epo: int = 800, eps: float = 1e-8,
                 last_epoch: int = -1, **kwargs):
        """
        Args:
            optimizer: PyTorch optimizer.
            learning_rate_start: Starting LR (also the LR for epochs < epo_min).
            learning_rate_stop: Target LR at epoch epo.
            epo_min: Epoch to start decay (keeps learning_rate_start before this).
            epo: Total epochs (reaches learning_rate_stop at this epoch).
            eps: Minimum learning rate floor. Default is 1e-8.
            last_epoch: Last epoch index (for resuming training).
        """
        self.lr_start = learning_rate_start
        self.lr_stop = learning_rate_stop
        self.epo_min = epo_min
        self.epo = epo
        self.eps = float(eps)

        def lr_lambda(epoch):
            """Compute LR multiplier for current epoch."""
            if epoch < self.epo_min:
                return 1.0
            elif epoch >= self.epo:
                return max(self.lr_stop, self.eps) / self.lr_start
            else:
                progress = (epoch - self.epo_min) / (self.epo - self.epo_min)
                lr_current = self.lr_start + progress * (self.lr_stop - self.lr_start)
                return max(lr_current, self.eps) / self.lr_start

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class LinearWarmupLinearLearningRateScheduler(lr_scheduler.LambdaLR):
    """Linear warmup + linear decay (matches Keras LinearWarmupLinearLearningRateScheduler).

    LR ramps linearly from 0 to learning_rate_start over epo_warmup epochs,
    then decays linearly from learning_rate_start to learning_rate_stop by
    epoch epo. Floored at eps.

    Used by matbench/materials project configs (mp_log_gvrh, mp_phonons, etc.).
    """

    def __init__(self, optimizer, learning_rate_start: float = 1e-3,
                 learning_rate_stop: float = 1e-5, epo_warmup: int = 0,
                 epo: int = 500, eps: float = 1e-8,
                 last_epoch: int = -1, **kwargs):
        """
        Args:
            optimizer: PyTorch optimizer.
            learning_rate_start: LR at end of warmup / start of decay.
            learning_rate_stop: Target LR at epoch epo.
            epo_warmup: Number of warmup epochs (linear ramp from ~0 to
                learning_rate_start). Default is 0 (no warmup).
            epo: Total epochs (reaches learning_rate_stop at this epoch).
            eps: Minimum learning rate floor. Default is 1e-8.
            last_epoch: Last epoch index (for resuming training).
        """
        self.lr_start = learning_rate_start
        self.lr_stop = learning_rate_stop
        self.epo_warmup = max(epo_warmup, 0)
        self.epo = epo
        self.eps = float(eps)

        def lr_lambda(epoch):
            if self.epo_warmup > 0 and epoch < self.epo_warmup:
                # Linear ramp: (epoch+1)/epo_warmup to avoid LR=0 at epoch 0.
                return float(epoch + 1) / float(self.epo_warmup)
            elif epoch >= self.epo:
                return max(self.lr_stop, self.eps) / self.lr_start
            else:
                # Linear decay from lr_start to lr_stop between epo_warmup and epo.
                progress = (epoch - self.epo_warmup) / max(self.epo - self.epo_warmup, 1)
                lr_current = self.lr_start + progress * (self.lr_stop - self.lr_start)
                return max(lr_current, self.eps) / self.lr_start

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def get_scheduler(name: str, optimizer, **kwargs):
    """Get a scheduler by name.

    Args:
        name: Scheduler name.
        optimizer: PyTorch optimizer.
        **kwargs: Scheduler-specific parameters. ``steps_per_epoch`` is
            consumed by custom schedulers that need to convert Keras
            step-level configs to epoch-level; it is stripped before
            passing kwargs to PyTorch built-in schedulers.

    Returns:
        LR scheduler instance.
    """
    schedulers = {
        "linear_warmup": LinearWarmupScheduler,
        "warmup_exponential": LinearWarmupExponentialDecay,
        "LinearWarmupExponentialDecay": LinearWarmupExponentialDecay,
        "LinearLearningRateScheduler": LinearLearningRateScheduler,
        "LinearWarmupLinearLearningRateScheduler": LinearWarmupLinearLearningRateScheduler,
        "warmup_linear": LinearWarmupLinearLearningRateScheduler,
        "polynomial_decay": PolynomialDecayScheduler,
        "cosine_warmup": CosineWarmupScheduler,
        "step": lr_scheduler.StepLR,
        "exponential": lr_scheduler.ExponentialLR,
        "cosine": lr_scheduler.CosineAnnealingLR,
        "reduce_on_plateau": lr_scheduler.ReduceLROnPlateau,
        "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    }
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {list(schedulers.keys())}")
    cls = schedulers[name]
    # Strip steps_per_epoch for PyTorch built-in schedulers that don't accept it.
    if cls in (lr_scheduler.StepLR, lr_scheduler.ExponentialLR,
               lr_scheduler.CosineAnnealingLR, lr_scheduler.ReduceLROnPlateau):
        kwargs.pop("steps_per_epoch", None)
    return cls(optimizer, **kwargs)
