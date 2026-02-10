"""GAT (Graph Attention Network) model.

Reference: Velickovic et al., Graph Attention Networks (2018).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.attention import AttentionHeadGAT
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.ops.activ import get_activation
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class GATModel(nn.Module):
    """Graph Attention Network.

    Expects PyG Data batch with:
        - data.x or data.z: Node features (N,) int or (N, F) float.
        - data.edge_index: Edge indices (2, M).
        - data.edge_attr: Optional edge features (M, edge_dim).
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 3,
                 attention_units: int = 32,
                 attention_heads_num: int = 5,
                 attention_heads_concat: bool = False,
                 attention_activation: str = "leaky_relu2",
                 use_edge_features: bool = True,
                 edge_dim: int = 0,
                 node_pooling: str = "mean",
                 output_units: list = None,
                 output_activation: str = "relu",
                 output_use_bias: list = None,
                 output_final_activation: str = "sigmoid",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        """Initialize GAT model.

        Args:
            node_dim: Dimension of node features after embedding.
            depth: Number of GAT layers.
            attention_units: Units per attention head.
            attention_heads_num: Number of attention heads.
            attention_heads_concat: If True, concatenate heads; else average.
            attention_activation: Activation in attention computation.
            use_edge_features: Whether to use edge features in attention.
            edge_dim: Edge feature dimension.
            node_pooling: Pooling method for readout.
            output_units: Hidden dims for output MLP.
            output_activation: Activation for output MLP.
            output_use_bias: Per-layer use_bias for output MLP. Defaults to
                [True, True, False] matching the Keras reference.
            num_targets: Number of output targets.
            use_node_embedding: Whether to embed integer node features.
            num_embeddings: Vocabulary size.
        """
        super().__init__()
        if output_units is None:
            output_units = [25, 10]
        if output_use_bias is None:
            output_use_bias = [True] * len(output_units) + [False]

        self.output_embedding = output_embedding
        self.use_node_embedding = use_node_embedding
        self.attention_heads_num = attention_heads_num
        self.attention_heads_concat = attention_heads_concat
        self.depth = depth
        self.use_edge_features = use_edge_features
        self.edge_proj = None
        if use_edge_features:
            if edge_dim and edge_dim > 0:
                self._effective_edge_dim = edge_dim
            else:
                self._effective_edge_dim = attention_units
                self.edge_proj = nn.LazyLinear(attention_units)
        else:
            self._effective_edge_dim = 0

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        self.dense_in = nn.Linear(node_dim, attention_units)

        # Each layer has multiple attention heads
        self.attention_layers = nn.ModuleList()
        for i in range(depth):
            in_dim = attention_units * attention_heads_num if (i > 0 and attention_heads_concat) else attention_units
            heads = nn.ModuleList([
                AttentionHeadGAT(
                    in_features=in_dim, units=attention_units,
                    use_edge_features=use_edge_features, edge_dim=self._effective_edge_dim,
                    activation=attention_activation, use_final_activation=False
                )
                for _ in range(attention_heads_num)
            ])
            self.attention_layers.append(heads)

        if attention_heads_concat:
            self.activation_after_average = None
        else:
            self.activation_after_average = get_activation(attention_activation)

        final_dim = attention_units * attention_heads_num if attention_heads_concat else attention_units
        self.pooling = PoolingNodes(pooling_method=node_pooling)
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + [output_final_activation]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=final_dim,
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
        if self.use_edge_features and edge_attr is None:
            edge_attr = torch.zeros(
                edge_index.size(1), self._effective_edge_dim,
                device=x.device, dtype=x.dtype
            )
        if self.edge_proj is not None and edge_attr is not None:
            edge_attr = self.edge_proj(edge_attr)

        for heads in self.attention_layers:
            head_outs = [head(x, edge_index, edge_attr) for head in heads]
            if self.attention_heads_concat:
                x = torch.cat(head_outs, dim=-1)
            else:
                x = torch.stack(head_outs, dim=0).mean(dim=0)
                if self.activation_after_average is not None:
                    x = self.activation_after_average(x)

        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(x, batch, batch_size)
        else:
            out = x
        out = self.output_mlp(out)
        return out
