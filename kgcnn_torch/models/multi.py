"""Multi-model merging utilities.

Mirrors Keras ``kgcnn.models.multi.merge_models`` — combines multiple
graph models by merging their outputs (e.g. concatenation) with an
optional output MLP.

Usage::

    from kgcnn_torch.models.multi import merge_models

    merged = merge_models([model_a, model_b], output_mlp={"units": [64, 1], "input_dim": 128})
    out = merged(batch)
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.mlp import MLP


class MergeModels(nn.Module):
    """Merge multiple graph models by combining their outputs.

    Each sub-model receives the same PyG batch object.  Their outputs
    are merged (default: concatenated along the feature axis) and
    optionally passed through a final MLP.

    Args:
        model_list (list): List of :class:`nn.Module` graph models.
        merge_type (str): How to merge outputs. Currently supports
            ``"concat"`` (alias ``"concatenate"``).
        output_mlp: Optional MLP applied after merging.  Can be either:
            - A ``dict`` of kwargs forwarded to :class:`~kgcnn_torch.layers.mlp.MLP`
              (must include ``units`` and ``input_dim``).
            - An existing :class:`nn.Module`.
    """

    def __init__(self, model_list: list, merge_type: str = "concat",
                 output_mlp=None):
        super().__init__()
        self.models = nn.ModuleList(model_list)
        self.merge_type = merge_type

        if merge_type not in ("concat", "concatenate"):
            raise NotImplementedError(
                "Unknown merge type '%s' for models" % merge_type)

        self.mlp = None
        if output_mlp is not None:
            if isinstance(output_mlp, dict):
                self.mlp = MLP(**output_mlp)
            elif isinstance(output_mlp, nn.Module):
                self.mlp = output_mlp

    def forward(self, data):
        """Run each sub-model on *data* and merge their outputs.

        Args:
            data: PyG ``Data`` / ``Batch`` object forwarded to every
                sub-model.

        Returns:
            torch.Tensor: Merged (and optionally MLP-transformed) output.
        """
        outputs = [model(data) for model in self.models]

        if self.merge_type in ("concat", "concatenate"):
            out = torch.cat(outputs, dim=-1)

        if self.mlp is not None:
            out = self.mlp(out)

        return out


def merge_models(model_list: list, merge_type: str = "concat",
                 output_mlp: dict = None):
    r"""Merge a list of models by combining their output.

    Functional wrapper around :class:`MergeModels` that mirrors the Keras
    ``kgcnn.models.multi.merge_models`` API.

    Args:
        model_list (list): List of graph models (:class:`nn.Module`).
        merge_type (str): How to merge the output (default ``"concat"``).
        output_mlp (dict): Kwargs forwarded to
            :class:`~kgcnn_torch.layers.mlp.MLP` for a final MLP after
            the merged output.  Must include ``units`` and ``input_dim``.

    Returns:
        :class:`MergeModels`: An ``nn.Module`` that runs all models and
        merges their predictions.
    """
    return MergeModels(model_list, merge_type=merge_type,
                       output_mlp=output_mlp)
