"""Gather functions for graph neural networks.

In PyG convention: edge_index[0] = source, edge_index[1] = target.
These are pure functions, not nn.Modules.
"""
import torch


def gather_nodes_outgoing(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Gather source/outgoing node features for each edge.

    Args:
        x: Node features of shape (N, F).
        edge_index: Edge indices of shape (2, M), PyG convention.

    Returns:
        Source node features of shape (M, F).
    """
    return x[edge_index[0]]


def gather_nodes_ingoing(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Gather target/ingoing node features for each edge.

    Args:
        x: Node features of shape (N, F).
        edge_index: Edge indices of shape (2, M), PyG convention.

    Returns:
        Target node features of shape (M, F).
    """
    return x[edge_index[1]]


def gather_nodes(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Gather both target and source node features, concatenated.

    Matches Keras GatherNodes(split_indices=(0,1)) which returns [target, source].
    In PyG convention: edge_index[1]=target, edge_index[0]=source.

    Args:
        x: Node features of shape (N, F).
        edge_index: Edge indices of shape (2, M), PyG convention.

    Returns:
        Concatenated [target, source] features of shape (M, 2*F).
    """
    return torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=-1)


def gather_state(state: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Repeat graph-level state for each node according to batch assignment.

    Args:
        state: Graph-level features of shape (B, F).
        batch: Batch assignment for each node of shape (N,).

    Returns:
        Repeated state of shape (N, F).
    """
    return state[batch]


def gather_edges_pairs(edges: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
    """Gather reverse edge features for each edge (for directed message passing).

    Handles negative indices (convention for edges without a reverse pair):
    negative indices are replaced with 0 before gathering, and features at
    those positions are zeroed out.

    Args:
        edges: Edge features of shape (M, F).
        pair_index: Index of the reverse edge for each edge, shape (M,).
            Negative values indicate no reverse edge exists.

    Returns:
        Reverse edge features of shape (M, F).
    """
    valid_mask = pair_index >= 0
    safe_index = torch.where(valid_mask, pair_index, torch.zeros_like(pair_index))
    gathered = edges[safe_index]
    gathered = torch.where(valid_mask.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    return gathered
