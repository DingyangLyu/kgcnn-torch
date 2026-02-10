"""MAT (Molecule Attention Transformer) model.

Reference: Maziarka et al., Molecule Attention Transformer (2020).
https://arxiv.org/abs/2002.08264

This model operates on padded (dense) graph representations rather than
sparse PyG-style batches. Inputs are expected as:
    - node_input: (B, N, F) node features or (B, N) integer atom types
    - xyz_input: (B, N, 3) atomic coordinates
    - adjacency: (B, N, N) or (B, N, N, 1) adjacency matrix
    - node_mask: (B, N) boolean mask for valid nodes
    - adj_mask: (B, N, N) boolean mask for valid adjacency entries

The attention mechanism uses **feature-wise** attention following the Keras
reference implementation: Q and K are expanded along different spatial axes
and multiplied element-wise (not dot-product), producing attention weights
of shape (B, N, N, F) per head. Each head is a separate MATAttentionHead
instance, and results are concatenated or summed then projected.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class MATDistanceMatrix(nn.Module):
    """Compute pairwise distance matrix from 3D coordinates.

    Matches the Keras MATDistanceMatrix: computes squared distances, applies
    masking by adding 1/epsilon to masked positions before exp (so masked
    positions become exp(-large) ~ 0), then zeros out masked positions.

    Args:
        trafo: How to transform distances. One of 'exp' or 'softmax'.
    """

    def __init__(self, trafo: str = "exp"):
        super().__init__()
        if trafo not in ("exp", "softmax"):
            raise ValueError(
                f"trafo must be 'exp' or 'softmax', got '{trafo}'"
            )
        self.trafo = trafo

    def forward(self, xyz: torch.Tensor, node_mask: torch.Tensor):
        """Compute distance matrix.

        Args:
            xyz: Coordinates of shape (B, N, 3).
            node_mask: Mask of shape (B, N, 1), float, 1 for valid nodes.

        Returns:
            dist: Distance-based matrix of shape (B, N, N, 1).
            dist_mask: Validity mask of shape (B, N, N, 1), float.
        """
        # Pairwise squared distances: (B, N, N, 1)
        # diff: (B, 1, N, 3) - (B, N, 1, 3) -> (B, N, N, 3)
        diff = xyz.unsqueeze(1) - xyz.unsqueeze(2)
        dist = (diff * diff).sum(dim=-1, keepdim=True)  # (B, N, N, 1)

        # Build pairwise mask from node_mask (B, N, 1)
        # diff_mask: (B, 1, N, 1) * (B, N, 1, 1) -> broadcast to (B, N, N, 1)
        # but we need product over last dim of the 3-component mask
        # Keras: mask is (B, N, 3), diff_mask = expand(mask,1) * expand(mask,2) -> (B, N, N, 3)
        # dist_mask = prod(diff_mask, axis=-1, keepdims=True) -> (B, N, N, 1)
        # Since our node_mask is (B, N, 1), pairwise is just outer product
        dist_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # (B, N, N, 1)
        dist_mask_bool = dist_mask.bool()

        eps = 1e-7  # Keras backend epsilon

        if self.trafo == "exp":
            # Add 1/eps to masked positions before exp, so exp(-1/eps) ~ 0
            dist = dist + torch.where(
                dist_mask_bool,
                torch.zeros_like(dist),
                torch.ones_like(dist) / eps,
            )
            dist = torch.exp(-dist)
        elif self.trafo == "softmax":
            # Add -1/eps to masked positions before softmax
            dist = dist + torch.where(
                dist_mask_bool,
                torch.zeros_like(dist),
                -torch.ones_like(dist) / eps,
            )
            dist = F.softmax(dist, dim=2)

        dist = dist * dist_mask
        return dist, dist_mask


class MATAttentionHead(nn.Module):
    """Single feature-wise attention head for MAT.

    Uses element-wise (feature-wise) attention, NOT dot-product attention.
    Q is expanded along axis=2: (B, N, 1, F), K along axis=1: (B, 1, N, F).
    qk = Q * K / scale gives (B, N, N, F) -- element-wise product.
    The combined attention (qk + distance + adjacency) has shape (B, N, N, F)
    and is applied to V via feature-wise weighted sum.

    Args:
        units: Dimension of Q, K, V projections (feature dimension per head).
        lambda_attention: Weight for QK self-attention component.
        lambda_distance: Weight for distance matrix component.
        lambda_adjacency: Weight for adjacency matrix component.
            If None, computed as 1 - lambda_attention - lambda_distance.
        add_identity: Whether to add identity matrix to adjacency.
        dropout: Dropout rate on attention weights.
    """

    def __init__(self,
                 units: int = 8,
                 input_dim: int = None,
                 lambda_attention: float = 0.3,
                 lambda_distance: float = 0.3,
                 lambda_adjacency: float = None,
                 add_identity: bool = False,
                 dropout: float = None):
        super().__init__()
        self.units = units
        self.add_identity = add_identity
        self.lambda_attention = lambda_attention
        self.lambda_distance = lambda_distance
        if lambda_adjacency is not None:
            self.lambda_adjacency = lambda_adjacency
        else:
            self.lambda_adjacency = 1.0 - lambda_attention - lambda_distance
        self.scale = units ** -0.5

        # input_dim is the dimension of h coming in (embedding_units).
        # If not specified, assume input_dim == units.
        in_dim = input_dim if input_dim is not None else units
        self.dense_q = nn.Linear(in_dim, units)
        self.dense_k = nn.Linear(in_dim, units)
        self.dense_v = nn.Linear(in_dim, units)

        self._dropout = dropout
        if self._dropout is not None and self._dropout > 0:
            self.layer_dropout = nn.Dropout(p=self._dropout)
        else:
            self._dropout = None

    def forward(self, h, a_d, a_g, h_mask, a_d_mask, a_g_mask):
        """Forward pass.

        Args:
            h: Node features of shape (B, N, F_in).
            a_d: Distance matrix of shape (B, N, N, 1).
            a_g: Adjacency matrix of shape (B, N, N, 1).
            h_mask: Node mask of shape (B, N, 1), float.
            a_d_mask: Distance mask of shape (B, N, N, 1), float.
            a_g_mask: Adjacency mask of shape (B, N, N, 1), float.

        Returns:
            Output node features of shape (B, N, units).
        """
        eps = 1e-7

        # Q: (B, N, units) -> (B, N, 1, units)
        q = self.dense_q(h).unsqueeze(2)
        # K: (B, N, units) -> (B, 1, N, units)
        k = self.dense_k(h).unsqueeze(1)
        # V: (B, N, units), masked
        v = self.dense_v(h) * h_mask

        # Feature-wise attention: element-wise product, NOT dot product
        # qk: (B, N, N, units)
        # Keep Keras semantics: qk = q * k / (units**-0.5) = q * k * sqrt(units)
        qk = q * k / self.scale

        # Mask self-attention: (B, 1, N, 1) * (B, N, 1, 1) -> (B, N, N, 1)
        qk_mask = h_mask.unsqueeze(1) * h_mask.unsqueeze(2)  # (B, N, N, 1)
        qk_mask_bool = qk_mask.bool()

        # Add -1/eps to masked positions before softmax
        qk = qk + torch.where(
            qk_mask_bool.expand_as(qk),
            torch.zeros_like(qk),
            -torch.ones_like(qk) / eps,
        )
        qk = F.softmax(qk, dim=2)
        qk = qk * qk_mask.expand_as(qk)

        # Add identity to adjacency (optional)
        if self.add_identity:
            N = a_g.shape[1]
            eye = torch.eye(N, dtype=a_g.dtype, device=a_g.device)
            eye = eye.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
            a_g = a_g + eye

        # Weighted combination: all have shape broadcastable to (B, N, N, units)
        qk = self.lambda_attention * qk
        a_d = self.lambda_distance * a_d.to(h.dtype)
        a_g = self.lambda_adjacency * a_g.to(h.dtype)
        # a_d and a_g are (B, N, N, 1), they broadcast over feature dim
        att = qk + a_d + a_g  # (B, N, N, units)

        if self._dropout is not None:
            att = self.layer_dropout(att)

        # Apply attention to values: feature-wise weighted sum
        # v: (B, N, units) -> transpose to (B, units, N)
        # att: (B, N, N, units) -> transpose to (B, units, N, N)
        # Then matmul: (B, units, N, N) @ (B, units, N, 1) -> (B, units, N, 1)
        # Squeeze and transpose back to (B, N, units)
        v_t = v.permute(0, 2, 1)  # (B, units, N)
        att_t = att.permute(0, 3, 1, 2)  # (B, units, N, N)
        hp = torch.einsum('...ij,...jk->...ik', att_t, v_t.unsqueeze(3))
        hp = hp.squeeze(3)  # (B, units, N)
        hp = hp.permute(0, 2, 1)  # (B, N, units)

        hp = hp * h_mask
        return hp


class MATGlobalPool(nn.Module):
    """Global sum pooling over padded node dimension."""

    def forward(self, h: torch.Tensor):
        """Pool node features to graph-level by summing over node axis.

        Args:
            h: Node features of shape (B, N, F), already masked.

        Returns:
            Pooled features of shape (B, F).
        """
        return h.sum(dim=1)


class MATModel(nn.Module):
    """Molecule Attention Transformer (MAT) model.

    A transformer-style architecture for molecular property prediction that
    combines feature-wise self-attention, 3D distance information, and graph
    adjacency. Each transformer block has multiple separate attention heads
    (each an independent MATAttentionHead), whose outputs are concatenated
    (or summed) and projected.

    This model operates on padded (dense) representations. Inputs:
        - node_input: (B, N, F) float features or (B, N) integer atom types.
        - xyz_input: (B, N, 3) atomic coordinates.
        - adjacency: (B, N, N) or (B, N, N, 1) adjacency matrix.
        - node_mask: (B, N) boolean mask for valid nodes.
        - adj_mask: (B, N, N) boolean mask for valid adjacency entries.

    Architecture (matching Keras reference):
        1. Optional embedding for integer atom types.
        2. Linear projection to embedding dimension (no bias).
        3. Compute distance matrix with exp transform and masking.
        4. Transformer blocks (depth layers), each with:
           a. LayerNorm -> N separate MATAttentionHead instances ->
              concat/sum -> Dense projection -> Residual
           b. LayerNorm -> Feed-forward MLP -> Dense projection ->
              Mask -> Residual
        5. Final LayerNorm -> Mask -> Global sum pool -> Output MLP.

    Args:
        embedding_units: Hidden dimension for the transformer.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads per block.
        attention_units: Feature dimension per attention head.
        merge_heads: How to merge heads: 'concat' or 'sum'.
        lambda_attention: Weight for QK self-attention component.
        lambda_distance: Weight for distance matrix component.
        lambda_adjacency: Weight for adjacency component (None = auto).
        add_identity: Whether to add identity to adjacency in attention.
        attention_dropout: Dropout rate for attention weights.
        distance_trafo: Distance transform method ('exp' or 'softmax').
        units_ff: Hidden dimensions for the feed-forward MLP.
        ff_activations: Activations for the feed-forward MLP layers.
        output_units: Hidden dimensions for the output MLP.
        output_activations: Activations for the output MLP layers.
        num_targets: Number of output targets.
        use_node_embedding: If True, embed integer node inputs via nn.Embedding.
        num_embeddings: Vocabulary size for the node embedding.
        input_node_dim: Dimension of input node features (when not embedding).
    """

    def __init__(self,
                 embedding_units: int = 32,
                 depth: int = 5,
                 num_heads: int = 8,
                 attention_units: int = 8,
                 merge_heads: str = "concat",
                 lambda_attention: float = 0.3,
                 lambda_distance: float = 0.3,
                 lambda_adjacency: float = None,
                 add_identity: bool = False,
                 attention_dropout: float = 0.1,
                 distance_trafo: str = "exp",
                 units_ff: list = None,
                 ff_activations: list = None,
                 output_units: list = None,
                 output_activations: list = None,
                 num_targets: int = 1,
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 input_node_dim: int = 64,
                 output_embedding: str = "graph",
                 # Legacy parameters for backward compatibility with tests
                 units_ff_legacy: int = None,
                 ff_activation: str = None,
                 dropout: float = None,
                 distance_method: str = None,
                 output_activation: str = None,
                 ):
        super().__init__()

        # Handle legacy parameter names for backward compat with test
        if distance_method is not None and distance_trafo == "exp":
            distance_trafo = distance_method
        if dropout is not None and attention_dropout == 0.1:
            attention_dropout = dropout

        # Handle units_ff as int (legacy) or list
        if units_ff is not None and isinstance(units_ff, int):
            units_ff = [units_ff, units_ff, units_ff]

        # Default feed-forward units/activations matching Keras defaults
        if units_ff is None:
            if units_ff_legacy is not None:
                units_ff = [units_ff_legacy, units_ff_legacy, units_ff_legacy]
            else:
                units_ff = [32, 32, 32]
        if ff_activations is None:
            if ff_activation is not None:
                ff_activations = [ff_activation] * (len(units_ff) - 1) + ["linear"]
            else:
                ff_activations = ["relu", "relu", "linear"]

        # Default output MLP
        if output_units is None:
            output_units = [32, 16, num_targets]
        else:
            output_units = list(output_units) + [num_targets]
        if output_activations is None:
            if output_activation is not None:
                output_activations = [output_activation] * (len(output_units) - 1) + ["linear"]
            else:
                output_activations = ["relu"] * (len(output_units) - 1) + ["linear"]

        self.output_embedding = output_embedding
        self.use_node_embedding = use_node_embedding
        self.depth = depth
        self.embedding_units = embedding_units
        self.num_heads = num_heads
        self.merge_heads = merge_heads

        # Distance matrix computation
        self.distance_layer = MATDistanceMatrix(trafo=distance_trafo)

        # Node embedding / input projection
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, input_node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        self.input_projection = nn.Linear(input_node_dim, embedding_units, bias=False)

        # Adjacency projection: Dense(1, use_bias=False) applied when adjacency has feature dim.
        # Matches Keras: adj = Dense(1, use_bias=False)(adj) when has_edge_dim is True.
        self.adj_projection = nn.Linear(1, 1, bias=False)

        # Transformer blocks
        self.attention_norms = nn.ModuleList()
        self.attention_heads = nn.ModuleList()  # ModuleList of ModuleLists
        self.attention_projections = nn.ModuleList()
        self.ff_norms = nn.ModuleList()
        self.ff_mlps = nn.ModuleList()
        self.ff_projections = nn.ModuleList()

        for _ in range(depth):
            # Match Keras LayerNormalization default epsilon (1e-3).
            self.attention_norms.append(nn.LayerNorm(embedding_units, eps=1e-3))

            # Create num_heads separate MATAttentionHead instances
            heads = nn.ModuleList([
                MATAttentionHead(
                    units=attention_units,
                    input_dim=embedding_units,
                    lambda_attention=lambda_attention,
                    lambda_distance=lambda_distance,
                    lambda_adjacency=lambda_adjacency,
                    add_identity=add_identity,
                    dropout=attention_dropout,
                )
                for _ in range(num_heads)
            ])
            self.attention_heads.append(heads)

            # Projection after merging heads
            if merge_heads in ("add", "sum", "reduce_sum"):
                proj_in = attention_units
            else:
                proj_in = attention_units * num_heads
            self.attention_projections.append(
                nn.Linear(proj_in, embedding_units, bias=False)
            )

            # Feed-forward sub-layer
            # Match Keras LayerNormalization default epsilon (1e-3).
            self.ff_norms.append(nn.LayerNorm(embedding_units, eps=1e-3))

            # Build FF MLP as sequential of (Linear, Activation) pairs
            ff_layers = []
            ff_in = embedding_units
            for j, ff_out in enumerate(units_ff):
                ff_layers.append(nn.Linear(ff_in, ff_out))
                ff_layers.append(_get_activation_module(ff_activations[j]))
                ff_in = ff_out
            self.ff_mlps.append(nn.Sequential(*ff_layers))

            # Projection after FF MLP to match embedding_units
            self.ff_projections.append(
                nn.Linear(units_ff[-1], embedding_units, bias=False)
            )

        # Final layer norm
        # Match Keras LayerNormalization default epsilon (1e-3).
        self.final_norm = nn.LayerNorm(embedding_units, eps=1e-3)

        # Pooling
        self.pool = MATGlobalPool()

        # Output MLP: build as sequential of (Linear, Activation) pairs
        out_layers = []
        out_in = embedding_units
        for j, out_dim in enumerate(output_units):
            out_layers.append(nn.Linear(out_in, out_dim))
            out_layers.append(_get_activation_module(output_activations[j]))
            out_in = out_dim
        self.output_mlp = nn.Sequential(*out_layers)

    def forward(self,
                node_input: torch.Tensor,
                xyz_input: torch.Tensor,
                adjacency: torch.Tensor,
                node_mask: torch.Tensor,
                adj_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            node_input: Node features (B, N, F) or integer types (B, N).
            xyz_input: Coordinates (B, N, 3).
            adjacency: Adjacency matrix (B, N, N) or (B, N, N, 1).
            node_mask: Boolean/float node mask (B, N).
            adj_mask: Boolean/float adjacency mask (B, N, N).

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        # Node features
        if self.use_node_embedding:
            n = self.node_embedding(node_input.long())  # (B, N, input_node_dim)
        else:
            n = node_input.float()

        # Expand masks to have trailing feature dim, matching Keras
        # n_mask: (B, N) -> (B, N, 1)
        n_mask = node_mask.float().unsqueeze(-1)  # (B, N, 1)
        # adj_mask: (B, N, N) -> (B, N, N, 1)
        a_mask = adj_mask.float().unsqueeze(-1)  # (B, N, N, 1)

        # Compute distance matrix from coordinates
        dist, dist_mask = self.distance_layer(xyz_input, n_mask)
        # dist: (B, N, N, 1), dist_mask: (B, N, N, 1)

        # Process adjacency to ensure shape (B, N, N, 1)
        if adjacency.dim() == 3:
            # (B, N, N) -> (B, N, N, 1)
            adj = adjacency.unsqueeze(-1).float()
        else:
            adj = adjacency.float()

        # Apply adjacency projection (matches Keras Dense(1, use_bias=False) for has_edge_dim)
        adj = self.adj_projection(adj)

        # Project node features to embedding dimension
        h = self.input_projection(n)  # (B, N, embedding_units)

        h_mask = n_mask  # (B, N, 1)

        # Transformer blocks
        for i in range(self.depth):
            # 1. Norm + Attention + Residual
            hn = self.attention_norms[i](h)

            # Run each head separately
            head_outputs = []
            for head in self.attention_heads[i]:
                ho = head(hn, dist, adj, h_mask, dist_mask, a_mask)
                head_outputs.append(ho)

            # Merge heads
            if self.merge_heads in ("add", "sum", "reduce_sum"):
                hu = sum(head_outputs)
            else:
                hu = torch.cat(head_outputs, dim=-1)

            hu = self.attention_projections[i](hu)
            h = h + hu  # Residual connection

            # 2. Norm + MLP + Residual
            hn = self.ff_norms[i](h)
            hu = self.ff_mlps[i](hn)
            hu = self.ff_projections[i](hu)
            hu = hu * h_mask  # Mask to keep padded positions zero
            h = h + hu  # Residual connection

        # Final layer norm
        out = self.final_norm(h)

        # Global pooling: mask then sum
        if self.output_embedding == "graph":
            out = out * h_mask
            out = self.pool(out)  # (B, embedding_units)
            out = self.output_mlp(out)  # (B, num_targets)
        else:
            out = self.output_mlp(out)
            out = out * h_mask

        return out


def _get_activation_module(name: str) -> nn.Module:
    """Get activation module by name, compatible with project registry.

    Falls back to the kgcnn_torch activation registry for custom activations.
    """
    from kgcnn_torch.ops.activ import get_activation
    return get_activation(name)
