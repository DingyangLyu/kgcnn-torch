"""GATv2 (Graph Attention Network v2) model.

Reference: Brody et al., How Attentive are Graph Attention Networks? (2021).
Uses AttentionHeadGATV2 which applies activation BEFORE the attention vector,
giving it stronger expressivity than the original GAT.
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.attention import AttentionHeadGATV2
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.ops.activ import get_activation
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class GATv2Model(nn.Module):
    """GATv2 model with improved attention mechanism.

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
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        super().__init__()
        if output_units is None:
            output_units = [25, 10]

        self.output_embedding = output_embedding
        self.use_node_embedding = use_node_embedding
        self.attention_heads_num = attention_heads_num
        self.attention_heads_concat = attention_heads_concat
        self.depth = depth
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        self.dense_in = nn.Linear(node_dim, attention_units)

        # Each layer has multiple GATv2 attention heads
        self.attention_layers = nn.ModuleList()
        for i in range(depth):
            in_dim = attention_units * attention_heads_num if (i > 0 and attention_heads_concat) else attention_units
            heads = nn.ModuleList([
                AttentionHeadGATV2(
                    in_features=in_dim, units=attention_units,
                    use_edge_features=use_edge_features, edge_dim=edge_dim,
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
        out_act = [output_activation] * len(output_units) + ["sigmoid"]
        out_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=final_dim,
            activation=out_act,
            use_bias=out_bias
        )

    def forward(self, data) -> torch.Tensor:
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
        if self.use_edge_features and self.edge_dim > 0 and edge_attr is None:
            edge_attr = torch.zeros(
                edge_index.size(1), self.edge_dim,
                device=x.device, dtype=x.dtype
            )

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
