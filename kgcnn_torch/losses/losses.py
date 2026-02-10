"""Loss functions for graph neural network training."""
import torch
import torch.nn as nn


class ForceMeanAbsoluteError(nn.Module):
    """MAE loss for force predictions.

    Computes element-wise absolute error averaged over all components.

    Args:
        per_molecule: If True and ``batch`` is provided, compute per-atom MAE
            for each molecule first, then average over molecules. This prevents
            large molecules from dominating the loss. Default: False.
    """

    def __init__(self, per_molecule: bool = False):
        super().__init__()
        self.per_molecule = per_molecule

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                batch: torch.Tensor = None, batch_size: int = None) -> torch.Tensor:
        if self.per_molecule and batch is not None:
            from kgcnn_torch.ops.scatter import scatter_reduce_mean
            if batch_size is None:
                batch_size = int(batch.max().item()) + 1
            per_atom_err = (pred - target).abs().mean(dim=-1)  # (N,)
            per_mol = scatter_reduce_mean(batch, per_atom_err.unsqueeze(-1), batch_size)  # (B, 1)
            return per_mol.mean()
        return (pred - target).abs().mean()


class EnergyForceLoss(nn.Module):
    """Combined energy + force loss.

    L = energy_weight * MAE(E_pred, E_target) + force_weight * MAE(F_pred, F_target)
    """

    def __init__(self, energy_weight: float = 1.0, force_weight: float = 100.0):
        super().__init__()
        self.energy_weight = energy_weight
        self.force_weight = force_weight

    def forward(self, energy_pred: torch.Tensor, energy_target: torch.Tensor,
                force_pred: torch.Tensor, force_target: torch.Tensor) -> torch.Tensor:
        energy_loss = (energy_pred - energy_target).abs().mean()
        force_loss = (force_pred - force_target).abs().mean()
        return self.energy_weight * energy_loss + self.force_weight * force_loss


class BinaryCrossentropyNoNaN(nn.Module):
    """Binary cross-entropy loss that ignores NaN targets.

    For datasets with missing labels (encoded as NaN), this loss masks out
    NaN entries before computing the binary cross-entropy. Predictions and
    targets at NaN positions are replaced with zeros so they do not
    contribute to the loss.
    """

    def __init__(self, from_logits: bool = False, reduction: str = "mean"):
        """Initialize.

        Args:
            from_logits: If True, input predictions are raw logits and
                sigmoid is applied internally via BCEWithLogitsLoss.
                If False (default, matching Keras BinaryCrossentropy),
                predictions are assumed to be probabilities.
            reduction: Reduction mode ('mean', 'sum', 'none').
        """
        super().__init__()
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute binary cross-entropy ignoring NaN targets.

        Args:
            pred: Predictions of shape (...).
            target: Targets of shape (...), may contain NaN values.

        Returns:
            Scalar loss (or unreduced if reduction='none').
        """
        is_nan = torch.isnan(target)
        clean_target = torch.where(is_nan, torch.zeros_like(target), target)
        # For NaN positions, use safe default values to avoid log(0) or log(1-1)
        if self.from_logits:
            clean_pred = torch.where(is_nan, torch.zeros_like(pred), pred)
            loss = nn.functional.binary_cross_entropy_with_logits(
                clean_pred, clean_target, reduction="none"
            )
        else:
            # Clamp predictions to [eps, 1-eps] to prevent log(0) at NaN positions
            eps = torch.finfo(pred.dtype).eps
            clean_pred = torch.where(is_nan, torch.full_like(pred, 0.5), pred)
            clean_pred = clean_pred.clamp(min=eps, max=1.0 - eps)
            loss = nn.functional.binary_cross_entropy(
                clean_pred, clean_target, reduction="none"
            )

        # Zero out loss at NaN positions.
        loss = torch.where(is_nan, torch.zeros_like(loss), loss)

        if self.reduction == "mean":
            # Average only over non-NaN entries.
            count = (~is_nan).sum().clamp(min=1)
            return loss.sum() / count.float()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class RaggedValuesMeanAbsoluteError(nn.Module):
    """MAE loss that handles variable-length (ragged) predictions.

    For graph-level predictions where different graphs may produce different
    numbers of outputs (e.g., variable-size node predictions gathered per
    graph), this loss flattens / concatenates all values before computing MAE.

    Expects either:
        - Flat tensors of matching shape (standard MAE).
        - Lists of tensors with variable first dimension, which are
          concatenated before computing MAE.
    """

    def forward(self, pred, target) -> torch.Tensor:
        """Compute MAE over all values.

        Args:
            pred: Predictions -- either a Tensor or a list of Tensors.
            target: Targets -- either a Tensor or a list of Tensors.

        Returns:
            Scalar MAE.
        """
        if isinstance(pred, (list, tuple)):
            pred = torch.cat(pred, dim=0)
        if isinstance(target, (list, tuple)):
            target = torch.cat(target, dim=0)
        return (pred - target).abs().mean()


class DisjointForceMeanAbsoluteError(nn.Module):
    """Force MAE loss for disjoint graph representation.

    In disjoint representation, all atoms from the batch are concatenated into
    a single tensor of shape (N_total, 3) (or (N_total, 3, S) for multiple
    states). This loss computes the MAE over force components, optionally
    handling padded atoms (zero-force padding in padded-disjoint mode).

    Args:
        squeeze_states: If True (default), assume a single state and
            squeeze the state dimension. If False, average over states.
        padded_disjoint: If True, detect and mask padded (all-zero) atoms
            so they do not contribute to the loss.
    """

    def __init__(self, squeeze_states: bool = True, padded_disjoint: bool = False):
        super().__init__()
        self.squeeze_states = squeeze_states
        self.padded_disjoint = padded_disjoint

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute force MAE for disjoint representation.

        Args:
            pred: Predicted forces of shape (N, 3) or (N, 3, S).
            target: Target forces of shape (N, 3) or (N, 3, S).

        Returns:
            Scalar MAE loss.
        """
        if self.padded_disjoint:
            # Mask out padded atoms: atoms where all target components are zero.
            # For shape (N, 3): mask shape is (N,)
            # For shape (N, 3, S): mask shape is (N, S)
            is_close = torch.isclose(target, torch.zeros(1, device=target.device, dtype=target.dtype))
            mask = ~is_close.all(dim=1)  # (N,) or (N, S)

            # Zero out predictions at padded positions.
            if pred.dim() == 2:
                # (N, 3) case: mask is (N,), unsqueeze to (N, 1)
                pred = pred * mask.unsqueeze(1).float()
            else:
                # (N, 3, S) case: mask is (N, S), unsqueeze to (N, 1, S)
                pred = pred * mask.unsqueeze(1).float()

            row_count = mask.sum(dim=0).clamp(min=1)  # scalar or (S,)
            norm = 1.0 / row_count.float()
        else:
            norm = 1.0 / target.shape[0]

        diff = (target - pred).abs()
        # Mean over the spatial (3) dimension, then sum over atoms.
        out = diff.mean(dim=1)  # (N,) or (N, S)
        out = out.sum(dim=0) * norm  # scalar or (S,)

        if not self.squeeze_states and out.dim() > 0:
            # Average over states dimension (matching Keras: ops.mean(out, axis=-1))
            out = out.mean()

        return out
