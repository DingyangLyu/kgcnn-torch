"""Metrics for graph neural network evaluation.

Mirrors the metrics provided in ``kgcnn.metrics.metrics`` (Keras version),
re-implemented as pure PyTorch callables.  Each class can be used as a
standalone function ``metric(pred, target) -> scalar tensor``.
"""
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Basic functional metrics
# ---------------------------------------------------------------------------

def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error."""
    return (pred - target).abs().mean()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error."""
    return ((pred - target) ** 2).mean().sqrt()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error."""
    return ((pred - target) ** 2).mean()


# ---------------------------------------------------------------------------
# Scaled regression metrics (matching Keras ScaledMeanAbsoluteError / RMSE)
# ---------------------------------------------------------------------------

class ScaledMeanAbsoluteError:
    """Scaled Mean Absolute Error.

    Multiplies predictions and targets by a fixed *scale* tensor before
    computing MAE, allowing metrics to be reported in original (unscaled)
    units while training on normalised targets.

    This mirrors ``kgcnn.metrics.metrics.ScaledMeanAbsoluteError`` which
    stores a Keras variable ``self.scale`` and applies ``scale * y`` in
    ``update_state``.

    Args:
        scale: Scalar or array-like scale factor. Default is 1.0 (no scaling).
    """

    def __init__(self, scale=1.0):
        if isinstance(scale, torch.Tensor):
            self.scale = scale.float()
        else:
            self.scale = torch.as_tensor(scale, dtype=torch.float32)

    def set_scale(self, scale):
        """Set scale from numpy array or tensor (matches Keras API)."""
        if isinstance(scale, torch.Tensor):
            self.scale = scale.float()
        else:
            self.scale = torch.as_tensor(scale, dtype=torch.float32)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        s = self.scale.to(pred.device)
        return (s * pred - s * target).abs().mean()


class ScaledRootMeanSquaredError:
    """Scaled Root Mean Squared Error.

    Multiplies predictions and targets by a fixed *scale* tensor before
    computing RMSE.

    Mirrors ``kgcnn.metrics.metrics.ScaledRootMeanSquaredError``.

    Args:
        scale: Scalar or array-like scale factor. Default is 1.0.
    """

    def __init__(self, scale=1.0):
        if isinstance(scale, torch.Tensor):
            self.scale = scale.float()
        else:
            self.scale = torch.as_tensor(scale, dtype=torch.float32)

    def set_scale(self, scale):
        """Set scale from numpy array or tensor (matches Keras API)."""
        if isinstance(scale, torch.Tensor):
            self.scale = scale.float()
        else:
            self.scale = torch.as_tensor(scale, dtype=torch.float32)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        s = self.scale.to(pred.device)
        return ((s * pred - s * target) ** 2).mean().sqrt()


class ScaledForceMeanAbsoluteError:
    """Scaled force MAE with padded-atom detection and per-atom normalisation.

    Mirrors ``kgcnn.metrics.metrics.ScaledForceMeanAbsoluteError``.

    For padded batches the target tensor has shape ``(batch, max_atoms, 3)``.
    Padding rows (all-zero in the target) are detected and excluded so that
    the per-structure MAE is normalised by the *actual* number of atoms.

    Args:
        scale: Scale factor applied to pred/target before error computation.
            Shape should broadcast with ``(1, 1, 1)`` or ``(1, 1, D)``.
        find_padded_atoms: If True (default), detect padding atoms as rows
            where all target components are zero and exclude them.
        squeeze_states: If True and the last dim of scale is 1, squeeze it
            before reshaping (matches Keras behaviour).
    """

    def __init__(self, scale=1.0, find_padded_atoms: bool = True,
                 squeeze_states: bool = True):
        self.find_padded_atoms = find_padded_atoms
        self.squeeze_states = squeeze_states
        self.set_scale(scale)

    def set_scale(self, scale):
        """Set the scale tensor, reshaping to broadcast with (B, N, 3)."""
        if isinstance(scale, np.ndarray):
            scale = torch.from_numpy(scale).float()
        elif not isinstance(scale, torch.Tensor):
            scale = torch.as_tensor(scale, dtype=torch.float32)
        else:
            scale = scale.float()
        # Reshape to (1, 1, ...) to broadcast with (batch, atoms, 3).
        if scale.dim() == 0:
            scale = scale.reshape(1, 1, 1)
        elif scale.dim() == 1:
            if self.squeeze_states and scale.shape[-1] == 1:
                scale = scale.squeeze(-1)
            scale = scale.unsqueeze(0).unsqueeze(0)
        elif scale.dim() == 2:
            if self.squeeze_states and scale.shape[-1] == 1:
                scale = scale.squeeze(-1)
            scale = scale.unsqueeze(1).unsqueeze(2)
        self.scale = scale

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute scaled force MAE.

        Args:
            pred: Predicted forces of shape ``(batch, max_atoms, 3)``.
            target: Target forces of shape ``(batch, max_atoms, 3)``.

        Returns:
            Scalar force MAE.
        """
        s = self.scale.to(pred.device)

        if self.find_padded_atoms:
            # Detect padding atoms: rows where all target components ≈ 0.
            is_real = ~torch.all(
                torch.isclose(target, torch.tensor(0.0, device=target.device)), dim=2
            )  # (batch, max_atoms)
            row_count = is_real.sum(dim=1).clamp(min=1).float()  # (batch,)
            norm = 1.0 / row_count  # (batch,)
        else:
            norm = 1.0 / target.shape[1]

        y_true = s * target
        y_pred = s * pred

        diff = (y_true - y_pred).abs()  # (batch, atoms, 3)
        # Mean over xyz, sum over atoms, normalise by real atom count.
        per_struct = diff.mean(dim=2).sum(dim=1) * norm  # (batch,)
        if not self.squeeze_states:
            per_struct = per_struct.mean(dim=-1)
        return per_struct.mean()


# ---------------------------------------------------------------------------
# Classification metrics with NaN handling
# ---------------------------------------------------------------------------

class BinaryAccuracyNoNaN:
    """Binary accuracy ignoring NaN labels.

    Mirrors ``kgcnn.metrics.metrics.BinaryAccuracyNoNaN``.

    Computes accuracy per sample (along the last axis) then averages across
    samples (macro-average), matching the Keras ``MeanMetricWrapper`` behaviour.

    Args:
        threshold: Decision boundary for converting probabilities to binary
            predictions. Default is 0.5.
    """

    def __init__(self, threshold: float = 0.5):
        if threshold is not None and (threshold <= 0 or threshold >= 1):
            raise ValueError(
                f"Invalid threshold. Expected value in (0, 1), got {threshold}."
            )
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        is_not_nan = ~torch.isnan(target)
        pred_binary = (pred > self.threshold).to(target.dtype)
        target_clean = torch.where(is_not_nan, target, torch.zeros_like(target))
        correct = (pred_binary == target_clean).float()

        if target.dim() >= 2:
            # Per-sample accuracy along last axis, then macro-average.
            counts = (correct * is_not_nan.float()).sum(dim=-1)
            norm = is_not_nan.float().sum(dim=-1).clamp(min=1)
            return (counts / norm).mean()
        else:
            # Flat tensor: global accuracy.
            count = is_not_nan.float().sum().clamp(min=1)
            return (correct * is_not_nan.float()).sum() / count


class AUCNoNaN:
    """ROC-AUC ignoring NaN labels.

    Uses ``sklearn.metrics.roc_auc_score``. NaN entries in the target are
    removed before computing the AUC. Returns 0.0 when fewer than 2 classes
    are present.

    Mirrors ``kgcnn.metrics.metrics.AUCNoNaN``.
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from sklearn.metrics import roc_auc_score

        pred_np = pred.detach().cpu().numpy().ravel()
        target_np = target.detach().cpu().numpy().ravel()

        valid = ~np.isnan(target_np)
        target_valid = target_np[valid]
        pred_valid = pred_np[valid]

        if len(target_valid) < 2 or len(set(target_valid.astype(int))) < 2:
            return torch.tensor(0.0)

        try:
            score = roc_auc_score(target_valid, pred_valid)
        except ValueError:
            score = 0.0

        return torch.tensor(float(score))


class BalancedBinaryAccuracyNoNaN:
    """Balanced binary accuracy ignoring NaN labels.

    Computes ``(sensitivity + specificity) / 2`` while excluding NaN entries.

    Mirrors ``kgcnn.metrics.metrics.BalancedBinaryAccuracyNoNaN``.

    Args:
        threshold: Decision boundary. Default is 0.5.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-7  # match keras.config.epsilon()
        is_not_nan = ~torch.isnan(target)
        pred_binary = (pred > self.threshold).float()
        target_clean = torch.where(is_not_nan, target, torch.zeros_like(target))
        valid = is_not_nan.float()

        positives = (target_clean == 1).float() * valid
        tp = (pred_binary * positives).sum()
        fn = ((1 - pred_binary) * positives).sum()
        sensitivity = tp / (tp + fn + eps)

        negatives = (target_clean == 0).float() * valid
        tn = ((1 - pred_binary) * negatives).sum()
        fp = (pred_binary * negatives).sum()
        specificity = tn / (tn + fp + eps)

        return (sensitivity + specificity) / 2.0


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
ScaledMAE = ScaledMeanAbsoluteError
ScaledRMSE = ScaledRootMeanSquaredError
ForceMAE = ScaledForceMeanAbsoluteError
