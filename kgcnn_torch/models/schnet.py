"""SchNet model.

Reference: Schutt et al., SchNet: A continuous-filter convolutional neural network
for modeling quantum interactions (2017).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.conv import SchNetInteraction
from kgcnn_torch.layers.geom import GaussBasisLayer, compute_edge_distances, shift_periodic_lattice
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class SchNetModel(nn.Module):
    """SchNet model for molecular property prediction.

    # TODO: Add output_scaling support (StandardScaler-like per-target scaling)
    # to match Keras make_model_weighted / output_scaling functionality.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Atom positions (N, 3).
        - data.edge_index: Edge indices (2, M).
        - data.batch: Batch assignment (N,).
        - Optional (crystal): data.edge_image, data.lattice, data.batch_edge.
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 4,
                 units: int = 128,
                 edge_dim: int = 20,
                 gauss_bins: int = 20,
                 gauss_distance: float = 4.0,
                 gauss_sigma: float = 0.4,
                 gauss_offset: float = 0.0,
                 interaction_activation: str = "shifted_softplus",
                 interaction_pooling: str = "sum",
                 node_pooling: str = "sum",
                 last_mlp_units: list = None,
                 last_mlp_activation: str = "shifted_softplus",
                 output_units: list = None,
                 output_activation: str = "shifted_softplus",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 make_distance: bool = True,
                 expand_distance: bool = True,
                 use_output_mlp: bool = True):
        """Initialize SchNet model.

        Args:
            node_dim: Embedding dimension for atomic numbers.
            depth: Number of SchNet interaction blocks.
            units: Hidden dimension for interaction layers.
            edge_dim: Gaussian expansion dimension (must match gauss_bins if expand_distance).
            gauss_bins: Number of Gaussian basis functions.
            gauss_distance: Maximum distance for Gaussian expansion.
            gauss_sigma: Width of Gaussian basis functions.
            gauss_offset: Offset for Gaussian basis functions.
            interaction_activation: Activation function in interaction blocks.
            interaction_pooling: Pooling method in interaction aggregation.
            node_pooling: Pooling for graph-level readout.
            last_mlp_units: Hidden dims for per-node MLP after interactions.
            last_mlp_activation: Activation for last MLP.
            output_units: Hidden dims for output MLP (after pooling).
            output_activation: Activation for output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to embed atomic numbers.
            num_embeddings: Vocabulary size for embedding.
            make_distance: Whether to compute distances from positions.
            expand_distance: Whether to expand distances with Gaussian basis.
            use_output_mlp: Whether to apply output MLP after pooling.
        """
        super().__init__()
        self.output_embedding = output_embedding
        if last_mlp_units is None:
            last_mlp_units = [128, 64]
        if output_units is None:
            output_units = [64]

        self.make_distance = make_distance
        self.expand_distance = expand_distance
        self.use_output_mlp = use_output_mlp
        self.use_node_embedding = use_node_embedding

        # Auto-derive edge_dim from gauss_bins when using Gaussian expansion
        if expand_distance:
            edge_dim = gauss_bins

        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)

        if expand_distance:
            self.gauss_basis = GaussBasisLayer(
                bins=gauss_bins, distance=gauss_distance,
                sigma=gauss_sigma, offset=gauss_offset
            )

        self.dense_in = nn.Linear(node_dim, units)

        self.interactions = nn.ModuleList()
        for _ in range(depth):
            self.interactions.append(SchNetInteraction(
                units=units, edge_dim=edge_dim,
                activation=interaction_activation,
                pooling_method=interaction_pooling
            ))

        self.last_mlp = MLP(
            units=last_mlp_units, input_dim=units,
            activation=last_mlp_activation
        )

        self.pooling = PoolingNodes(pooling_method=node_pooling)

        if use_output_mlp:
            out_units = output_units + [num_targets]
            out_act = [output_activation] * len(output_units) + ["linear"]
            self.output_mlp = MLP(
                units=out_units,
                input_dim=last_mlp_units[-1],
                activation=out_act
            )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        z = data.z
        edge_index = data.edge_index
        batch = data.batch

        # Node embedding
        if self.use_node_embedding:
            n = self.node_embedding(z.long())
        else:
            n = z

        # Edge features
        if self.make_distance:
            pos = data.pos
            # Handle periodic systems
            if hasattr(data, 'edge_image') and data.edge_image is not None:
                pos_j = pos[edge_index[0]]
                batch_edge = data.batch_edge if hasattr(data, 'batch_edge') else batch[edge_index[0]]
                pos_j = shift_periodic_lattice(pos_j, data.edge_image, data.lattice, batch_edge)
                pos_i = pos[edge_index[1]]
                diff = pos_j - pos_i
                dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
                dist = torch.sqrt(dist_sq + torch.finfo(dist_sq.dtype).eps)
            else:
                # Distance is symmetric: |source - target| == |target - source|.
                dist = compute_edge_distances(pos, edge_index)
            ed = dist
        else:
            ed = data.edge_attr

        if self.expand_distance:
            ed = self.gauss_basis(ed)

        # Model
        n = self.dense_in(n)
        for interaction in self.interactions:
            n = interaction(n, ed, edge_index)

        n = self.last_mlp(n)

        # Graph-level pooling
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(n, batch, batch_size)
        else:
            out = n

        if self.use_output_mlp:
            out = self.output_mlp(out)

        return out


class SchNetCrystalModel(SchNetModel):
    """SchNet model for crystalline materials with periodic boundary conditions.

    This is a dedicated crystal variant that always computes edge distances
    using periodic lattice shifts. It inherits all architecture from SchNetModel
    but overrides the forward pass to require lattice and edge_image inputs.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Atom positions (N, 3).
        - data.edge_index: Edge indices (2, M).
        - data.batch: Batch assignment (N,).
        - data.lattice: Lattice matrix per graph (B, 3, 3).
        - data.edge_image: Periodic image shift vectors per edge (M, 3).
    """

    def __init__(self, **kwargs):
        """Initialize SchNetCrystalModel.

        Accepts all the same keyword arguments as SchNetModel. The make_distance
        parameter is forced to True and expand_distance defaults to True since
        crystal models always compute distances from positions with periodic shifts.
        """
        kwargs.setdefault("make_distance", True)
        kwargs.setdefault("expand_distance", True)
        super().__init__(**kwargs)

    def forward(self, data) -> torch.Tensor:
        """Forward pass for periodic crystal systems.

        Args:
            data: PyG Data batch object with lattice and edge_image attributes.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        z = data.z
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch
        edge_image = data.edge_image
        lattice = data.lattice

        # Node embedding
        if self.use_node_embedding:
            n = self.node_embedding(z.long())
        else:
            n = z

        # Edge features with periodic boundary conditions
        if self.make_distance:
            batch_edge = batch[edge_index[0]]
            pos_j = pos[edge_index[0]]
            pos_j = shift_periodic_lattice(pos_j, edge_image, lattice, batch_edge)
            pos_i = pos[edge_index[1]]
            diff = pos_j - pos_i
            dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
            ed = torch.sqrt(dist_sq + torch.finfo(dist_sq.dtype).eps)
        else:
            ed = data.edge_attr

        if self.expand_distance:
            ed = self.gauss_basis(ed)

        # Model
        n = self.dense_in(n)
        for interaction in self.interactions:
            n = interaction(n, ed, edge_index)

        n = self.last_mlp(n)

        # Graph-level pooling
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(n, batch, batch_size)
        else:
            out = n

        if self.use_output_mlp:
            out = self.output_mlp(out)

        return out
