"""Force prediction model wrapper.

Wraps an energy-predicting GNN model to also predict forces via the negative
gradient of the energy with respect to atomic positions.
"""
import torch
import torch.nn as nn
from typing import Union, Optional


class EnergyForceModel(nn.Module):
    """Wrapper model that predicts both energy and forces from a GNN.

    Given a GNN that predicts total energy from atomic positions and other
    features, this model computes forces as the negative gradient of the
    predicted energy with respect to atomic coordinates:

    .. math::

        \\vec{F_i} = - \\nabla_i E_{\\text{total}}

    The inner energy model receives a PyG-style data object (or dict) and
    must return a scalar or per-graph energy prediction.

    Example:

    .. code-block:: python

        from kgcnn_torch.models.force import EnergyForceModel
        energy_model = ...  # Your GNN that predicts energy
        model = EnergyForceModel(energy_model, coordinate_input="pos")
        # data.pos must have requires_grad=True
        energy, forces = model(data)
    """

    def __init__(self,
                 energy_model: nn.Module,
                 coordinate_input: str = "pos",
                 output_as_dict: Union[bool, tuple] = True,
                 output_squeeze_states: bool = False,
                 is_physical_force: bool = True):
        """Initialize force model.

        Args:
            energy_model (nn.Module): A PyTorch module that takes a data object
                and returns energy predictions. The model should accept
                ``forward(data)`` where ``data`` is a PyG Data/Batch object.
            coordinate_input (str): The attribute name on the data object that
                contains atomic coordinates. Default is 'pos'.
            output_as_dict (bool or tuple): If True, output a dict with keys
                'energy' and 'force'. If a tuple of two strings, use those as
                dict keys. If False, return a tuple (energy, force).
            output_squeeze_states (bool): If True, squeeze the state dimension
                of forces when there is only one energy state.
            is_physical_force (bool): If True, return the physical force
                (negative gradient). If False, return the raw gradient.
        """
        super().__init__()
        self.energy_model = energy_model
        self.coordinate_input = coordinate_input
        self.output_squeeze_states = output_squeeze_states
        self.is_physical_force = is_physical_force

        if isinstance(output_as_dict, bool):
            self.output_as_dict_use = output_as_dict
            self.output_as_dict_names = ("energy", "force")
        elif isinstance(output_as_dict, (list, tuple)):
            self.output_as_dict_use = True
            self.output_as_dict_names = (output_as_dict[0], output_as_dict[1])
        else:
            self.output_as_dict_use = False
            self.output_as_dict_names = ("energy", "force")

    def forward(self, data, **kwargs):
        """Forward pass: compute energy and derive forces.

        The coordinate tensor on the data object must have
        ``requires_grad=True`` for gradient computation.

        Args:
            data: A PyG Data or Batch object. Must have the coordinate
                attribute (default 'pos') set and requiring gradients.
            **kwargs: Additional keyword arguments passed to the energy model.

        Returns:
            dict or tuple: Energy and force predictions. Format depends on
                ``output_as_dict``.
        """
        # Get coordinate tensor
        if isinstance(data, dict):
            coords = data[self.coordinate_input]
        else:
            coords = getattr(data, self.coordinate_input)

        # Ensure coordinates require gradient
        if not coords.requires_grad:
            coords.requires_grad_(True)
            if isinstance(data, dict):
                data[self.coordinate_input] = coords
            else:
                setattr(data, self.coordinate_input, coords)

        # Forward through energy model
        energy = self.energy_model(data, **kwargs)

        # Compute gradient of energy w.r.t. coordinates
        # energy shape: (batch_size, n_states) or (batch_size,) or scalar
        if energy.dim() == 0:
            # Scalar energy
            e_grad = torch.autograd.grad(
                energy, coords, create_graph=self.training,
                retain_graph=True, allow_unused=True)[0]
            if e_grad is None:
                e_grad = torch.zeros_like(coords)
        elif energy.dim() == 1:
            # (batch_size,) - sum over batch
            e_grad = torch.autograd.grad(
                energy.sum(), coords, create_graph=self.training,
                retain_graph=True, allow_unused=True)[0]
            if e_grad is None:
                e_grad = torch.zeros_like(coords)
        else:
            # (batch_size, n_states) - compute gradient per state and stack
            eng_sum = energy.sum(dim=0)
            n_states = eng_sum.shape[-1]
            grads = []
            for i in range(n_states):
                g = torch.autograd.grad(
                    eng_sum[i], coords, create_graph=self.training,
                    retain_graph=True, allow_unused=True)[0]
                if g is None:
                    g = torch.zeros_like(coords)
                grads.append(g.unsqueeze(-1))
            e_grad = torch.cat(grads, dim=-1)

            if self.output_squeeze_states and n_states == 1:
                e_grad = e_grad.squeeze(-1)

        # Compute physical force as negative gradient
        if self.is_physical_force:
            force = -e_grad
        else:
            force = e_grad

        # Format output
        if self.output_as_dict_use:
            return {
                self.output_as_dict_names[0]: energy,
                self.output_as_dict_names[1]: force,
            }
        else:
            return energy, force

    def get_config(self) -> dict:
        """Get configuration dictionary.

        Returns:
            dict: Configuration of this force model wrapper.
        """
        return {
            "coordinate_input": self.coordinate_input,
            "output_as_dict": self.output_as_dict_use,
            "output_as_dict_names": list(self.output_as_dict_names),
            "output_squeeze_states": self.output_squeeze_states,
            "is_physical_force": self.is_physical_force,
        }
