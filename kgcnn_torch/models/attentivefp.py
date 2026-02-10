"""AttentiveFP model.

Reference: Xiong et al., Pushing the Boundaries of Molecular Representation for
Drug Discovery with the Graph Attention Mechanism (2020).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.attention import AttentiveHeadFP
from kgcnn_torch.layers.pooling import PoolingNodesAttentive
from kgcnn_torch.layers.update import GRUUpdate
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class AttentiveFPModel(nn.Module):
    """AttentiveFP model for graph-level property prediction.

    Implements the Attentive FP architecture with graph attention convolution
    layers augmented by GRU updates, followed by an attentive graph-level
    pooling mechanism.

    Expects PyG Data batch with:
        - data.x or data.z: Node features (N,) int or (N, F) float.
        - data.edge_index: Edge indices (2, M).
        - data.edge_attr: Optional edge features (M, edge_dim).
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth_ato: int = 2,
                 depth_mol: int = 2,
                 units: int = 32,
                 use_edge_features: bool = True,
                 edge_dim: int = 0,
                 attention_activation: str = "leaky_relu2",
                 attention_activation_context: str = "elu",
                 pooling_activation: str = "leaky_relu2",
                 pooling_activation_context: str = "elu",
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "relu",
                 num_targets: int = 1,
                 dropout: float = 0.1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        """Initialize AttentiveFP model.

        Args:
            node_dim: Dimension of node features after embedding.
            depth_ato: Number of atom-level attention + GRU iterations.
            depth_mol: Number of molecule-level attentive pooling iterations.
            units: Hidden dimension for attention layers and GRU.
            use_edge_features: Whether to incorporate edge features in attention.
            edge_dim: Dimension of edge features (used only if use_edge_features is True).
            attention_activation: Activation for the attention alignment network.
            attention_activation_context: Context activation in AttentiveHeadFP.
            pooling_activation: Activation for attentive graph pooling alignment.
            pooling_activation_context: Context activation in attentive graph pooling.
            node_pooling: Initial pooling method for the attentive pooling start.
            output_units: Hidden dims for output MLP. If None, [units, units].
            output_activation: Activation for output MLP.
            num_targets: Number of output targets.
            dropout: Dropout rate after each GRU update.
            output_embedding: Output embedding mode, "graph" for graph-level or "node" for node-level.
            use_node_embedding: Whether to embed integer node features.
            num_embeddings: Vocabulary size for node embedding.
        """
        super().__init__()
        self.output_embedding = output_embedding
        if output_units is None:
            output_units = [25, 10]

        self.use_node_embedding = use_node_embedding
        self.use_edge_features = use_edge_features
        self.depth_ato = depth_ato
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        self.dense_in = nn.Linear(node_dim, units)

        # Atom-level: AttentiveHeadFP layers paired with GRU updates.
        # First layer always uses edge features (Keras convention),
        # subsequent layers use edge features only if explicitly set.
        self.attention_layers = nn.ModuleList()
        self.gru_updates = nn.ModuleList()

        for i in range(depth_ato):
            in_features = units
            # Only first layer uses edge features (matches Keras: first call has
            # use_edge_features=True, subsequent calls use default=False)
            layer_use_edge = use_edge_features if i == 0 else False
            self.attention_layers.append(AttentiveHeadFP(
                in_features=in_features,
                units=units,
                use_edge_features=layer_use_edge,
                edge_dim=edge_dim,
                activation=attention_activation,
                activation_context=attention_activation_context
            ))
            self.gru_updates.append(GRUUpdate(
                input_dim=units,
                hidden_dim=units
            ))

        # Molecule-level attentive pooling (Xiong et al. 2020, Section 2.3).
        self.attentive_pooling = PoolingNodesAttentive(
            units=units,
            depth=depth_mol,
            input_dim=units,
            pooling_method=node_pooling,
            activation=pooling_activation,
            activation_context=pooling_activation_context
        )

        # Output MLP
        # Keras default: activation=["relu", "relu", "sigmoid"], use_bias=[True, True, False]
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["sigmoid"]
        out_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=units,
            activation=out_act,
            use_bias=out_bias
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        if self.use_node_embedding:
            x = data.z if hasattr(data, 'z') and data.z is not None else data.x
            x = self.node_embedding(x.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            x = self.node_projection(x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        x = self.dense_in(x)

        # Atom-level iterations: attention followed by GRU update.
        # Keras: first iteration has no dropout, subsequent iterations do.
        for i in range(self.depth_ato):
            # AttentiveHeadFP computes attention-weighted messages.
            h_att = self.attention_layers[i](x, edge_index, edge_attr)  # (N, units)
            # GRU update: h_new = GRU(attention_output, h_old).
            x = self.gru_updates[i](h_att, x)  # (N, units)
            # Dropout after GRU update (Keras skips first iteration)
            if i > 0 and self.dropout is not None:
                x = self.dropout(x)

        # Molecule-level attentive pooling or node-level output.
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.attentive_pooling(x, batch, batch_size)  # (B, units)
        else:
            out = x

        # Output MLP
        out = self.output_mlp(out)
        return out
