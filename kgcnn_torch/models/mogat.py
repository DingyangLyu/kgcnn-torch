"""MoGAT (Molecular Graph Attention Transformer) model.

Uses AttentiveHeadFP layers with GRU updates and multi-layer self-attention
readout matching the Keras implementation.

Key architecture (Keras MoGAT):
  1. Collect embeddings from EACH attention layer into list_emb
  2. Apply PoolingNodesAttentive to each layer's embedding
  3. Concatenate along axis=1 (creating a sequence of super-node embeddings)
  4. Apply scaled dot-product self-attention: at = Attention()([out, out])
  5. Element-wise multiply attention output with pooled features
  6. Flatten and feed through output MLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from kgcnn_torch.layers.attention import AttentiveHeadFP
from kgcnn_torch.layers.update import GRUUpdate
from kgcnn_torch.layers.pooling import PoolingNodesAttentive
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class MoGATModel(nn.Module):
    """MoGAT model for molecular property prediction.

    Matches Keras implementation with multi-layer self-attention readout:
      - Collect per-layer node embeddings
      - Attentive pool each layer's embedding to graph level
      - Self-attention across layers
      - Multiply, flatten, output MLP

    Expects PyG Data batch with:
        - data.z or data.x: Node features.
        - data.edge_index: Edge indices (2, M).
        - data.batch: Batch assignment (N,).
        - Optional: data.edge_attr: Edge features.
    """

    def __init__(self,
                 node_dim: int = 64,
                 depthato: int = 2,
                 depthmol: int = 2,
                 units: int = 32,
                 edge_dim: int = 0,
                 # Keras MoGAT always uses edge features in the *first* AttentiveHeadFP_ call.
                 # Keep default True so parity works without extra ctor wiring.
                 use_edge_features: bool = True,
                 activation: str = "leaky_relu2",
                 dropout: float = 0.2,
                 output_units: list = None,
                 output_activation: str = "relu",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        super().__init__()
        if output_units is None:
            output_units = []

        self.output_embedding = output_embedding
        self.use_node_embedding = use_node_embedding
        self.use_edge_features = use_edge_features
        self.depthato = depthato
        self.dropout_rate = dropout

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # Keras: nk = Dense(units=attention_args['units'])(n)
        self.dense_in = nn.Linear(node_dim, units)

        # Attention layers and GRU updates
        self.attention_layers = nn.ModuleList()
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First layer uses edge features, subsequent layers may not
        # Keras: first call uses use_edge_features=True, subsequent use default from attention_args
        for i in range(depthato):
            use_ef = use_edge_features if i == 0 else False
            self.attention_layers.append(AttentiveHeadFP(
                in_features=units, units=units,
                use_edge_features=use_ef,
                edge_dim=edge_dim,
                activation=activation,
                activation_context="elu"
            ))
            self.gru_layers.append(GRUUpdate(input_dim=units, hidden_dim=units))
            self.dropout_layers.append(
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
            )

        # Per-layer attentive pooling (one PoolingNodesAttentive per layer)
        self.layer_pools = nn.ModuleList()
        for _ in range(depthato):
            self.layer_pools.append(PoolingNodesAttentive(
                units=units, depth=depthmol,
                activation=activation,
                activation_context="elu"
            ))

        # Attention dropout for self-attention
        self.attention_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        # Keras `keras.layers.Attention(use_scale=True)` uses a learned scalar `scale`
        # (not 1/sqrt(d_k)). Keep parity by matching that behavior.
        self.attn_scale = nn.Parameter(torch.tensor(1.0))

        # Output MLP: input is depthato * units (flattened after self-attention)
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        if output_embedding == "graph":
            self.output_mlp = MLP(
                units=out_units,
                input_dim=depthato * units,
                activation=out_act
            )
        else:
            self.output_mlp = MLP(
                units=out_units,
                input_dim=units,
                activation=out_act
            )

    def forward(self, data) -> torch.Tensor:
        if self.use_node_embedding:
            x = data.z if hasattr(data, 'z') and data.z is not None else data.x
            x = self.node_embedding(x.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            x = self.node_projection(x)
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = getattr(data, 'edge_attr', None)

        # Keras: nk = Dense(units=attention_args['units'])(n)
        nk = self.dense_in(x)

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Collect per-layer embeddings
        # Keras: first layer uses edge features, followed by dropout
        #        subsequent layers may not use edge features
        list_emb = []
        for i in range(self.depthato):
            if i == 0 and self.use_edge_features:
                ck = self.attention_layers[i](nk, edge_index, edge_attr)
            else:
                ck = self.attention_layers[i](nk, edge_index, edge_attr)
            nk = self.gru_layers[i](ck, nk)
            nk = self.dropout_layers[i](nk)
            list_emb.append(nk)

        if self.output_embedding == "graph":
            # Per-layer attentive pooling -> graph-level embedding per layer
            # Keras: out = [PoolingNodesAttentive(...)([count_nodes, ni, batch_id_node]) for ni in list_emb]
            pooled = []
            for i, ni in enumerate(list_emb):
                pi = self.layer_pools[i](ni, batch, batch_size)  # (B, units)
                pooled.append(pi)

            # Stack to create sequence: (B, depthato, units)
            # Keras: out = [ExpandDims(axis=1)(x) for x in out]
            #        out = Concatenate(axis=1)(out)
            out = torch.stack(pooled, dim=1)  # (B, depthato, units)

            # Scaled dot-product self-attention across layers
            # Keras: at = Attention(dropout=dropout, use_scale=True, score_mode="dot")([out, out])
            # Keras Attention with use_scale=True applies a *learned scalar* to dot scores.
            scores = torch.matmul(out, out.transpose(-2, -1)) * self.attn_scale  # (B, depthato, depthato)
            attn_weights = F.softmax(scores, dim=-1)  # (B, depthato, depthato)
            attn_weights = self.attention_dropout(attn_weights)
            at = torch.matmul(attn_weights, out)  # (B, depthato, units)

            # Element-wise multiply attention output with pooled features
            # Keras: out = at * out
            out = at * out  # (B, depthato, units)

            # Flatten: (B, depthato * units)
            # Keras: out = Flatten()(out)
            out = out.reshape(out.size(0), -1)
        else:
            # Node-level output: sum all layer embeddings (matching Keras Add()(list_emb))
            out = torch.stack(list_emb, dim=0).sum(dim=0)

        # Output MLP
        out = self.output_mlp(out)
        return out
