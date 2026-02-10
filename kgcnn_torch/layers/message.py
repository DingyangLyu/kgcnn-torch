"""Message passing base layers for graph neural networks."""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
from kgcnn_torch.layers.aggr import AggregateLocalEdges


class MessagePassingBase(nn.Module):
    """Base class for message passing networks.

    Subclasses should implement:
        - message_function(n_in, n_out, edges=None) -> messages
        - update_nodes(nodes, aggregated) -> updated_nodes
    """

    def __init__(self, pooling_method: str = "sum"):
        super().__init__()
        self.pooling_method = pooling_method
        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)

    def message_function(self, n_in: torch.Tensor, n_out: torch.Tensor,
                         edges: torch.Tensor = None) -> torch.Tensor:
        """Compute messages from gathered node pairs and optional edge features.

        Args:
            n_in: Receiving (target) node features gathered for edges (M, F).
            n_out: Sending (source) node features gathered for edges (M, F).
            edges: Optional edge features (M, E).

        Returns:
            Messages of shape (M, F').
        """
        raise NotImplementedError("Implement message_function in subclass.")

    def update_nodes(self, nodes: torch.Tensor,
                     aggregated: torch.Tensor) -> torch.Tensor:
        """Update node features from aggregated messages.

        Args:
            nodes: Current node features (N, F).
            aggregated: Aggregated messages (N, F').

        Returns:
            Updated node features (N, F'').
        """
        raise NotImplementedError("Implement update_nodes in subclass.")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """Standard message passing forward.

        Args:
            x: Node features (N, F).
            edge_index: Edge indices (2, M).
            edge_attr: Optional edge features (M, E).

        Returns:
            Updated node features (N, F'').
        """
        num_nodes = x.size(0)
        n_in = gather_nodes_ingoing(x, edge_index)  # target
        n_out = gather_nodes_outgoing(x, edge_index)  # source

        msg = self.message_function(n_in, n_out, edge_attr)
        agg = self.aggr(msg, edge_index, num_nodes)
        return self.update_nodes(x, agg)


class MatMulMessages(nn.Module):
    """Matrix multiplication on messages.

    For each edge: x_i' = A_i @ x_i where A is (M, F', F) and x is (M, F).
    Used by NMPN.
    """

    def forward(self, mat: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            mat: Transformation matrices (M, F', F).
            edges: Edge embeddings (M, F).

        Returns:
            Transformed messages (M, F').
        """
        return torch.bmm(mat, edges.unsqueeze(-1)).squeeze(-1)
