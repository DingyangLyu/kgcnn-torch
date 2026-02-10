"""CGCNN (Crystal Graph Convolutional Neural Network) model.

Reference: Xie & Grossman, Crystal Graph Convolutional Neural Networks for an Accurate
and Interpretable Prediction of Material Properties (2018).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.conv import CGCNNLayer
from kgcnn_torch.layers.geom import (
    GaussBasisLayer, DisplacementVectorsUnitCell, DisplacementVectorsASU,
    FracToRealCoordinates, EuclideanNorm
)
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes, PoolingWeightedNodes
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class CGCNNModel(nn.Module):
    """Crystal Graph Convolutional Neural Network.

    Designed for crystal and materials property prediction. Uses gated convolution
    layers where messages are formed by concatenating source, target, and edge features,
    then applying a sigmoid gate and softplus filter with residual connections.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.edge_index: Edge indices (2, M).
        - data.edge_attr: Edge features (M, edge_dim), e.g. Gaussian-expanded distances.
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 conv_units: int = None,
                 depth: int = 3,
                 gauss_bins: int = 40,
                 gauss_distance: float = 8.0,
                 gauss_sigma: float = 0.4,
                 gauss_offset: float = 0.0,
                 conv_activation: str = "softplus",
                 conv_gate_activation: str = "sigmoid",
                 node_pooling: str = "mean",
                 output_units: list = None,
                 output_activation: str = "softplus",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 expand_distance: bool = True,
                 make_distance: bool = False,
                 edge_input_dim: int = 1,
                 batch_normalization: bool = True):
        """Initialize CGCNN model.

        Args:
            node_dim: Embedding dimension for atomic numbers.
            conv_units: Internal dimension for convolutions (Keras conv_layer_args["units"]).
                If None, defaults to node_dim.
            depth: Number of CGCNN convolution layers.
            gauss_bins: Number of Gaussian basis functions for distance expansion.
            gauss_distance: Maximum distance for Gaussian expansion.
            gauss_sigma: Width of Gaussian basis functions.
            gauss_offset: Offset for Gaussian basis functions.
            conv_activation: Activation function for convolution filter (softplus).
            conv_gate_activation: Activation function for convolution gate (sigmoid).
            node_pooling: Pooling method for graph-level readout ('mean' or 'sum').
            output_units: Hidden dims for output MLP. If None, [node_dim, node_dim].
            output_activation: Activation for output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to embed atomic numbers.
            num_embeddings: Vocabulary size for embedding.
            expand_distance: Whether to expand distances with Gaussian basis.
            make_distance: Whether edge_attr contains raw distances to expand.
            edge_input_dim: Dimension of raw edge features when expand_distance=False.
            batch_normalization: Whether to use batch normalization in conv layers
                (default True, matching Keras).
        """
        super().__init__()
        self.output_embedding = output_embedding
        if conv_units is None:
            conv_units = node_dim
        if output_units is None:
            output_units = [64]

        self.use_node_embedding = use_node_embedding
        self.expand_distance = expand_distance
        self.make_distance = make_distance
        self.batch_normalization = batch_normalization

        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)

        # Initial projection: node_dim -> conv_units (Keras Dense(conv_units))
        self.dense_in = nn.Linear(node_dim, conv_units)

        if expand_distance:
            self.gauss_basis = GaussBasisLayer(
                bins=gauss_bins, distance=gauss_distance,
                sigma=gauss_sigma, offset=gauss_offset
            )
            edge_features = gauss_bins
        else:
            edge_features = edge_input_dim

        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(CGCNNLayer(
                node_features=conv_units, edge_features=edge_features,
                activation=conv_activation, gate_activation=conv_gate_activation,
                batch_normalization=batch_normalization
            ))

        self.pooling = PoolingNodes(pooling_method=node_pooling)
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        # Keras default: use_bias=[True, False] -- final layer has no bias.
        output_use_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=conv_units,
            activation=out_act,
            use_bias=output_use_bias
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
        edge_attr = data.edge_attr
        batch = data.batch

        # Node embedding
        if self.use_node_embedding:
            x = self.node_embedding(z.long())
        else:
            x = z

        # Initial projection (matches Keras)
        x = self.dense_in(x)

        # Edge features
        if self.expand_distance:
            ed = self.gauss_basis(edge_attr)
        else:
            ed = edge_attr

        # Compute batch_size once for batch normalization and pooling
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # CGCNN convolution layers with residual connections
        for conv in self.convs:
            if self.batch_normalization:
                x = conv(x, ed, edge_index, batch=batch, batch_size=batch_size)
            else:
                x = conv(x, ed, edge_index)

        # Graph-level pooling
        if self.output_embedding == "graph":
            out = self.pooling(x, batch, batch_size)
        else:
            out = x
        out = self.output_mlp(out)
        return out


class CGCNNCrystalModel(nn.Module):
    """CGCNN model for crystalline materials with periodic boundary conditions.

    Computes interatomic distances from fractional coordinates, lattice matrices,
    and cell translations (periodic image vectors), following the Keras
    ``make_crystal_model`` implementation.

    Supports two crystal representations:
        - ``'unit'``: Standard unit cell with cell translation vectors.
        - ``'asu'``: Asymmetric unit with symmetry operations and multiplicities
          (uses weighted pooling with site multiplicities).

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.frac_coords: Fractional coordinates (N, 3).
        - data.edge_index: Edge indices (2, M).
        - data.cell_translations: Per-edge cell translation vectors (M, 3).
        - data.lattice: Lattice matrix per graph (B, 3, 3).
        - data.batch: Batch assignment (N,).
        - Optional (ASU): data.symmetry_ops (M, 4, 4), data.multiplicities (N, 1).
    """

    def __init__(self,
                 node_dim: int = 64,
                 conv_units: int = None,
                 depth: int = 3,
                 gauss_bins: int = 40,
                 gauss_distance: float = 8.0,
                 gauss_sigma: float = 0.4,
                 gauss_offset: float = 0.0,
                 conv_activation: str = "softplus",
                 conv_gate_activation: str = "sigmoid",
                 node_pooling: str = "mean",
                 output_units: list = None,
                 output_activation: str = "softplus",
                 output_use_bias: list = None,
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 expand_distance: bool = True,
                 representation: str = "unit",
                 batch_normalization: bool = True):
        """Initialize CGCNNCrystalModel.

        Args:
            node_dim: Embedding dimension for atomic numbers.
            conv_units: Internal dimension for convolutions. If None, defaults to node_dim.
            depth: Number of CGCNN convolution layers.
            gauss_bins: Number of Gaussian basis functions for distance expansion.
            gauss_distance: Maximum distance for Gaussian expansion.
            gauss_sigma: Width of Gaussian basis functions.
            gauss_offset: Offset for Gaussian basis functions.
            conv_activation: Activation function for convolution filter.
            conv_gate_activation: Activation function for convolution gate.
            node_pooling: Pooling method for graph-level readout.
            output_units: Hidden dims for output MLP.
            output_activation: Activation for output MLP.
            output_use_bias: List of bools for per-layer bias in output MLP.
                If None, defaults to [True, ..., True, False] (matching Keras [True, False]).
            num_targets: Number of output targets.
            output_embedding: Output type ('graph' only for CGCNN).
            use_node_embedding: Whether to embed atomic numbers.
            num_embeddings: Vocabulary size for embedding.
            expand_distance: Whether to expand distances with Gaussian basis.
            representation: Crystal representation, 'unit' or 'asu'.
            batch_normalization: Whether to use batch normalization in conv layers
                (default True, matching Keras).
        """
        super().__init__()
        self.output_embedding = output_embedding
        self.representation = representation
        if conv_units is None:
            conv_units = node_dim
        if output_units is None:
            output_units = [64]

        self.use_node_embedding = use_node_embedding
        self.expand_distance = expand_distance
        self.batch_normalization = batch_normalization

        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)

        # Initial projection: node_dim -> conv_units (Keras Dense(conv_units))
        self.dense_in = nn.Linear(node_dim, conv_units)

        # Geometric layers for crystal distance computation
        if representation == "asu":
            self.displacement = DisplacementVectorsASU()
        else:
            self.displacement = DisplacementVectorsUnitCell()
        self.frac_to_real = FracToRealCoordinates()
        self.euclidean_norm = EuclideanNorm(axis=-1, keepdims=True)

        if expand_distance:
            self.gauss_basis = GaussBasisLayer(
                bins=gauss_bins, distance=gauss_distance,
                sigma=gauss_sigma, offset=gauss_offset
            )
            edge_features = gauss_bins
        else:
            # When not expanding, use raw edge feature dimension (1 for distances)
            edge_features = 1

        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(CGCNNLayer(
                node_features=conv_units, edge_features=edge_features,
                activation=conv_activation, gate_activation=conv_gate_activation,
                batch_normalization=batch_normalization
            ))

        if representation == "asu":
            self.pooling = PoolingWeightedNodes(pooling_method=node_pooling)
        else:
            self.pooling = PoolingNodes(pooling_method=node_pooling)

        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        if output_use_bias is None:
            output_use_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=conv_units,
            activation=out_act,
            use_bias=output_use_bias
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass for periodic crystal systems.

        Args:
            data: PyG Data batch object with crystal attributes.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        z = data.z
        frac_coords = data.frac_coords
        edge_index = data.edge_index
        cell_translations = data.cell_translations
        lattice = data.lattice
        batch = data.batch

        # Node embedding
        if self.use_node_embedding:
            x = self.node_embedding(z.long())
        else:
            x = z

        x = self.dense_in(x)

        # Reshape lattice from PyG-batched (B*3, 3) to (B, 3, 3) if needed
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        if lattice.dim() == 2:
            lattice = lattice.reshape(batch_size, 3, 3)

        # Compute distances from fractional coordinates
        if self.representation == "asu":
            symmetry_ops = data.symmetry_ops
            disp_frac = self.displacement(frac_coords, edge_index, symmetry_ops, cell_translations)
        else:
            disp_frac = self.displacement(frac_coords, edge_index, cell_translations)

        # Convert fractional displacement to real space
        batch_edge = batch[edge_index[0]]
        disp_real = self.frac_to_real(disp_frac, lattice, batch_edge)

        # Compute Euclidean distances
        edge_distances = self.euclidean_norm(disp_real)

        # Gaussian expansion
        if self.expand_distance:
            ed = self.gauss_basis(edge_distances)
        else:
            ed = edge_distances

        # CGCNN convolution layers
        for conv in self.convs:
            if self.batch_normalization:
                x = conv(x, ed, edge_index, batch=batch, batch_size=batch_size)
            else:
                x = conv(x, ed, edge_index)

        # Graph-level pooling
        if self.output_embedding != "graph":
            raise ValueError("Unsupported output embedding for CGCNN; only 'graph' is supported.")

        if self.representation == "asu":
            multiplicities = data.multiplicities
            out = self.pooling(x, multiplicities, batch, batch_size)
        else:
            out = self.pooling(x, batch, batch_size)

        out = self.output_mlp(out)
        return out
