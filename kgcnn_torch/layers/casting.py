"""Casting layers for converting between graph representations.

Supports conversions between batched (padded), ragged (variable-length lists),
and disjoint (PyG-style flat) graph tensor formats.

In disjoint representation, a batch of graphs is stored as a single large graph
with batch assignment tensors (graph_id_node, graph_id_edge).

PyG convention: edge_index shape is (2, M) with edge_index[0] = source,
edge_index[1] = target.
"""
import torch
import torch.nn as nn
from kgcnn_torch.ops.scatter import scatter_reduce_sum


def _pad_left(t: torch.Tensor) -> torch.Tensor:
    """Prepend a zero-valued element along dimension 0."""
    pad_shape = (1,) + t.shape[1:]
    return torch.cat([torch.zeros(pad_shape, dtype=t.dtype, device=t.device), t], dim=0)


def _cat_one(t: torch.Tensor) -> torch.Tensor:
    """Prepend a scalar 1 to a 1D tensor."""
    return torch.cat([torch.ones(1, dtype=t.dtype, device=t.device), t], dim=0)


class _CastBatchedDisjointBase(nn.Module):
    """Base class for casting between batched and disjoint graph representations.

    Args:
        reverse_indices: Whether to reverse index order (swap source/target). Default is False.
        dtype_batch: Dtype for batch ID tensor. Default is torch.int64.
        dtype_index: Dtype for index tensor. Default is None (keep original).
        padded_disjoint: Whether to keep padding in disjoint representation. Default is False.
        uses_mask: Whether the padding is marked by a boolean mask or by a length tensor.
            Default is False.
        remove_padded_disjoint_from_batched_output: Whether to remove the first padding element
            on batched output in case of padded disjoint. Default is True.
    """

    def __init__(self,
                 reverse_indices: bool = False,
                 dtype_batch: torch.dtype = torch.int64,
                 dtype_index: torch.dtype = None,
                 padded_disjoint: bool = False,
                 uses_mask: bool = False,
                 remove_padded_disjoint_from_batched_output: bool = True):
        super().__init__()
        self.reverse_indices = reverse_indices
        self.dtype_batch = dtype_batch
        self.dtype_index = dtype_index
        self.padded_disjoint = padded_disjoint
        self.uses_mask = uses_mask
        self.remove_padded_disjoint_from_batched_output = remove_padded_disjoint_from_batched_output


class CastBatchedIndicesToDisjoint(_CastBatchedDisjointBase):
    """Cast batched node features and edge indices to disjoint graph representation.

    Converts padded batched tensors of shape (B, N_max, ...) into flat disjoint tensors
    compatible with PyG-style message passing.

    For padded disjoint, all padded nodes are assigned to a dummy first graph with a single
    node and self-loop, so padding does not interfere with message passing.

    Input:
        [nodes, edge_indices, total_nodes_or_mask, total_edges_or_mask]

    Output:
        [node_attr, edge_index, graph_id_node, graph_id_edge,
         node_id, edge_id, nodes_count, edges_count]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: list) -> list:
        """Convert batched nodes and edge indices to disjoint format.

        Args:
            inputs: List of [nodes, edge_indices, nodes_in_batch, edges_in_batch].
                - nodes: (B, N_max, F, ...) node features.
                - edge_indices: (B, M_max, 2) edge index pairs.
                - nodes_in_batch: (B,) number of nodes per graph, or (B, N_max) boolean mask.
                - edges_in_batch: (B,) number of edges per graph, or (B, M_max) boolean mask.

        Returns:
            List of [node_attr, edge_index, graph_id_node, graph_id_edge,
                      node_id, edge_id, nodes_count, edges_count].
        """
        nodes, edge_indices, node_pad, edge_pad = inputs

        if self.dtype_index is not None:
            edge_indices = edge_indices.to(dtype=self.dtype_index)

        dtype_batch = self.dtype_batch

        if not self.uses_mask:
            node_len = node_pad.to(dtype=dtype_batch)
            edge_len = edge_pad.to(dtype=dtype_batch)
            B, N_max = nodes.shape[0], nodes.shape[1]
            M_max = edge_indices.shape[1]
            node_id = torch.arange(N_max, dtype=dtype_batch, device=nodes.device).unsqueeze(0).expand(B, -1)
            edge_id = torch.arange(M_max, dtype=dtype_batch, device=nodes.device).unsqueeze(0).expand(B, -1)
            node_mask = node_id < node_len.unsqueeze(-1)
            edge_mask = edge_id < edge_len.unsqueeze(-1)
        else:
            node_mask = node_pad
            edge_mask = edge_pad
            node_len = node_mask.to(dtype=dtype_batch).sum(dim=1)
            edge_len = edge_mask.to(dtype=dtype_batch).sum(dim=1)
            B, N_max = nodes.shape[0], nodes.shape[1]
            M_max = edge_indices.shape[1]
            node_id = torch.arange(N_max, dtype=dtype_batch, device=nodes.device).unsqueeze(0).expand(B, -1)
            edge_id = torch.arange(M_max, dtype=dtype_batch, device=nodes.device).unsqueeze(0).expand(B, -1)

        if not self.padded_disjoint:
            # Flatten using boolean mask
            nodes_flatten = nodes[node_mask]
            edge_indices_flatten = edge_indices[edge_mask]
            node_id_flat = node_id[node_mask]
            edge_id_flat = edge_id[edge_mask]

            node_splits = torch.cat([torch.zeros(1, dtype=dtype_batch, device=nodes.device),
                                     torch.cumsum(node_len, dim=0)], dim=0)
            graph_id_node = torch.repeat_interleave(
                torch.arange(B, dtype=dtype_batch, device=nodes.device), node_len)
            graph_id_edge = torch.repeat_interleave(
                torch.arange(B, dtype=dtype_batch, device=nodes.device), edge_len)

            offset_edge_indices = node_splits[graph_id_edge].unsqueeze(-1)
            offset_edge_indices = offset_edge_indices.expand_as(edge_indices_flatten)
            disjoint_indices = edge_indices_flatten + offset_edge_indices.to(edge_indices_flatten.dtype)
        else:
            # Padded disjoint: flatten all, prepend dummy graph
            nodes_flatten = nodes.reshape(-1, *nodes.shape[2:])
            edge_indices_flatten = edge_indices.reshape(-1, *edge_indices.shape[2:])
            node_len_flat = torch.full((B,), N_max, dtype=dtype_batch, device=nodes.device)
            edge_len_flat = torch.full((B,), M_max, dtype=dtype_batch, device=nodes.device)
            node_mask_flatten = node_mask.reshape(-1)
            edge_mask_flatten = edge_mask.reshape(-1)
            node_id_flat = node_id.reshape(-1)
            edge_id_flat = edge_id.reshape(-1)

            # Prepend dummy entries for the padding graph
            nodes_flatten = _pad_left(nodes_flatten)
            edge_indices_flatten = _pad_left(edge_indices_flatten)
            node_id_flat = _pad_left(node_id_flat)
            edge_id_flat = _pad_left(edge_id_flat)
            node_len_flat = _cat_one(node_len_flat)
            edge_len_flat = _cat_one(edge_len_flat)
            node_mask_flatten = _pad_left(node_mask_flatten)
            edge_mask_flatten = _pad_left(edge_mask_flatten)

            total_nodes = nodes_flatten.shape[0]
            total_edges = edge_indices_flatten.shape[0]

            graph_id_node = torch.repeat_interleave(
                torch.arange(node_len_flat.shape[0], dtype=dtype_batch, device=nodes.device),
                node_len_flat)[:total_nodes]
            graph_id_edge = torch.repeat_interleave(
                torch.arange(edge_len_flat.shape[0], dtype=dtype_batch, device=nodes.device),
                edge_len_flat)[:total_edges]

            graph_id_node = torch.where(node_mask_flatten, graph_id_node,
                                        torch.zeros_like(graph_id_node))
            graph_id_edge = torch.where(edge_mask_flatten, graph_id_edge,
                                        torch.zeros_like(graph_id_edge))
            node_id_flat = torch.where(node_mask_flatten, node_id_flat,
                                       torch.zeros_like(node_id_flat))
            edge_id_flat = torch.where(edge_mask_flatten, edge_id_flat,
                                       torch.zeros_like(edge_id_flat))

            node_splits = torch.cat([torch.zeros(1, dtype=dtype_batch, device=nodes.device),
                                     torch.cumsum(node_len_flat, dim=0)], dim=0)
            offset_edge_indices = node_splits[graph_id_edge].unsqueeze(-1)
            offset_edge_indices = offset_edge_indices.expand_as(edge_indices_flatten)
            disjoint_indices = edge_indices_flatten + offset_edge_indices.to(edge_indices_flatten.dtype)

            disjoint_indices = torch.where(edge_mask_flatten.unsqueeze(-1), disjoint_indices,
                                           torch.zeros_like(disjoint_indices))

            # Update counts: add padding counts for dummy graph
            pad_node_count = (node_len_flat[1:] - node_len).sum(dim=0, keepdim=True)
            node_len = torch.cat([pad_node_count, node_len], dim=0)
            pad_edge_count = (edge_len_flat[1:] - edge_len).sum(dim=0, keepdim=True)
            edge_len = torch.cat([pad_edge_count, edge_len], dim=0)

        # Transpose to PyG convention: (2, M)
        disjoint_indices = disjoint_indices.t().contiguous()

        if self.reverse_indices:
            disjoint_indices = disjoint_indices.flip(0)

        return [nodes_flatten, disjoint_indices, graph_id_node, graph_id_edge,
                node_id_flat, edge_id_flat, node_len, edge_len]


class CastBatchedAttributesToDisjoint(_CastBatchedDisjointBase):
    """Cast batched attributes to disjoint graph representation.

    Similar to CastBatchedIndicesToDisjoint but for attribute tensors only,
    without any index adjustment. Produces batch-ID tensor assignment.

    Input:
        [attr, total_attr_or_mask]

    Output:
        [attr, graph_id, item_id, item_counts]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: list) -> list:
        """Convert batched attributes to disjoint format.

        Args:
            inputs: List of [attr, total_attr].
                - attr: (B, N_max, F, ...) batched attributes.
                - total_attr: (B,) lengths per graph, or (B, N_max) boolean mask.

        Returns:
            List of [attr, graph_id, item_id, item_counts].
        """
        nodes, node_pad = inputs
        dtype_batch = self.dtype_batch

        if not self.uses_mask:
            node_len = node_pad.to(dtype=dtype_batch)
            B, N_max = nodes.shape[0], nodes.shape[1]
            node_id = torch.arange(N_max, dtype=dtype_batch, device=nodes.device).unsqueeze(0).expand(B, -1)
            node_mask = node_id < node_len.unsqueeze(-1)
        else:
            node_mask = node_pad
            node_len = node_mask.to(dtype=dtype_batch).sum(dim=1)
            B, N_max = nodes.shape[0], nodes.shape[1]
            node_id = torch.arange(N_max, dtype=dtype_batch, device=nodes.device).unsqueeze(0).expand(B, -1)

        if not self.padded_disjoint:
            nodes_flatten = nodes[node_mask]
            graph_id_node = torch.repeat_interleave(
                torch.arange(B, dtype=dtype_batch, device=nodes.device), node_len)
            node_id_flat = node_id[node_mask]
        else:
            nodes_flatten = nodes.reshape(-1, *nodes.shape[2:])
            node_len_flat = torch.full((B,), N_max, dtype=dtype_batch, device=nodes.device)
            node_mask_flatten = node_mask.reshape(-1)
            node_id_flat = node_id.reshape(-1)

            nodes_flatten = _pad_left(nodes_flatten)
            node_id_flat = _pad_left(node_id_flat)
            node_len_flat = _cat_one(node_len_flat)
            node_mask_flatten = _pad_left(node_mask_flatten)

            total_nodes = nodes_flatten.shape[0]
            graph_id = torch.repeat_interleave(
                torch.arange(node_len_flat.shape[0], dtype=dtype_batch, device=nodes.device),
                node_len_flat)[:total_nodes]
            graph_id_node = torch.where(node_mask_flatten, graph_id, torch.zeros_like(graph_id))
            node_id_flat = torch.where(node_mask_flatten, node_id_flat, torch.zeros_like(node_id_flat))

            pad_count = (node_len_flat[1:] - node_len).sum(dim=0, keepdim=True)
            node_len = torch.cat([pad_count, node_len], dim=0)

        return [nodes_flatten, graph_id_node, node_id_flat, node_len]


class CastDisjointToBatchedAttributes(_CastBatchedDisjointBase):
    """Cast disjoint attributes back to batched (padded) tensor representation.

    Reconstructs batched tensor from flat disjoint representation using ID tensors.

    Input:
        [attr, graph_id_attr, (attr_id), attr_counts]

    Output:
        Tensor of shape (B, N_max, F, ...) or (Tensor, mask) if return_mask=True.
    """

    def __init__(self, static_output_shape: tuple = None, return_mask: bool = False, **kwargs):
        """Initialize layer.

        Args:
            static_output_shape: Static shape for the second dimension (N_max,).
                If None, uses the max count from attr_counts.
            return_mask: If True, also return a boolean mask of valid positions.
        """
        super().__init__(**kwargs)
        self.static_output_shape = static_output_shape
        self.return_mask = return_mask

    def forward(self, inputs: list):
        """Convert disjoint attributes to batched format.

        Args:
            inputs: List of [attr, graph_id_attr, (attr_id), attr_counts].
                - attr: ([N], F, ...) flat attributes.
                - graph_id_attr: ([N],) batch assignment.
                - attr_id: ([N],) optional sub-graph ID for each item.
                - attr_counts: (B,) number of items per graph.

        Returns:
            Batched tensor of shape (B, N_max, F, ...), or tuple (tensor, mask)
            if return_mask is True.
        """
        if len(inputs) == 4:
            attr, graph_id_attr, attr_id, attr_len = inputs
        else:
            attr, graph_id_attr, attr_len = inputs
            attr_id = None

        if self.static_output_shape is not None:
            N_max = self.static_output_shape[0]
        else:
            N_max = int(attr_len.max().item())

        B = attr_len.shape[0]

        if not self.padded_disjoint:
            if attr_id is None:
                # Compute within-graph position for each item
                attr_splits = torch.cat([torch.zeros(1, dtype=attr_len.dtype, device=attr.device),
                                         torch.cumsum(attr_len, dim=0)], dim=0)
                attr_id = torch.arange(graph_id_attr.shape[0], dtype=graph_id_attr.dtype,
                                       device=attr.device)
                offset = torch.repeat_interleave(attr_splits[:-1], attr_len)
                attr_id = attr_id - offset
        else:
            if attr_id is None:
                raise ValueError(
                    "Require sub-graph IDs in addition to batch IDs for padded disjoint graphs.")

        # Scatter into flat (B * N_max, ...) then reshape
        flat_size = B * N_max
        indices = graph_id_attr * N_max + attr_id.to(graph_id_attr.dtype)
        out = scatter_reduce_sum(indices.long(), attr, flat_size)
        out = out.reshape(B, N_max, *attr.shape[1:])

        out_mask = None
        if self.return_mask:
            ones = torch.ones(attr.shape[0], dtype=torch.bool, device=attr.device)
            mask_flat = torch.zeros(flat_size, dtype=torch.bool, device=attr.device)
            mask_flat.scatter_(0, indices.long(), ones)
            out_mask = mask_flat.reshape(B, N_max)

        if self.padded_disjoint and self.remove_padded_disjoint_from_batched_output:
            out = out[1:]
            if out_mask is not None:
                out_mask = out_mask[1:]

        if self.return_mask:
            return out, out_mask
        return out


class CastDisjointToBatchedGraphState(_CastBatchedDisjointBase):
    """Cast graph-level state tensor from disjoint representation.

    For padded disjoint, removes the dummy first graph that represents padding.

    Input:
        Tensor of shape (B,) or (B+1, ...) for padded disjoint.

    Output:
        Tensor of shape (B, ...).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Remove padding graph state if padded disjoint.

        Args:
            inputs: Graph state tensor of shape (B, ...) or (B+1, ...).

        Returns:
            Graph state tensor of shape (B, ...).
        """
        if self.padded_disjoint and self.remove_padded_disjoint_from_batched_output:
            return inputs[1:]
        return inputs


class CastBatchedGraphStateToDisjoint(_CastBatchedDisjointBase):
    """Cast graph-level state tensor to disjoint representation.

    For padded disjoint, prepends a zero-valued dummy graph state for the padding graph.

    Input:
        Tensor of shape (B, ...).

    Output:
        Tensor of shape (B, ...) or (B+1, ...) for padded disjoint.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add padding graph state if padded disjoint.

        Args:
            inputs: Graph state tensor of shape (B, ...).

        Returns:
            Graph state tensor of shape (B, ...) or (B+1, ...).
        """
        if self.padded_disjoint:
            return _pad_left(inputs)
        return inputs


class CastRaggedAttributesToDisjoint(_CastBatchedDisjointBase):
    """Cast ragged (variable-length list) attributes to disjoint representation.

    Takes a list of tensors with variable first dimension and produces a single
    flat tensor with batch assignment information.

    Input:
        List of tensors, each of shape (N_i, F, ...) for graph i.

    Output:
        [attr, graph_id, item_id, item_counts]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: list) -> list:
        """Convert ragged attributes to disjoint format.

        Args:
            inputs: List of B tensors, each of shape (N_i, F, ...).

        Returns:
            List of [attr, graph_id, item_id, item_counts].
                - attr: ([N], F, ...) concatenated attributes.
                - graph_id: ([N],) batch assignment.
                - item_id: ([N],) within-graph item index.
                - item_counts: (B,) number of items per graph.
        """
        device = inputs[0].device
        dtype_batch = self.dtype_batch

        lengths = torch.tensor([t.shape[0] for t in inputs], dtype=dtype_batch, device=device)
        attr = torch.cat(inputs, dim=0)

        graph_id = torch.repeat_interleave(
            torch.arange(len(inputs), dtype=dtype_batch, device=device), lengths)

        # Within-graph item IDs
        item_ids = []
        for n in lengths:
            item_ids.append(torch.arange(n.item(), dtype=dtype_batch, device=device))
        item_id = torch.cat(item_ids, dim=0)

        return [attr, graph_id, item_id, lengths]


class CastRaggedIndicesToDisjoint(_CastBatchedDisjointBase):
    """Cast ragged node features and edge indices to disjoint representation.

    Takes lists of per-graph node tensors and edge index tensors and produces
    a single flat disjoint graph with adjusted edge indices.

    Input:
        [nodes_list, edge_indices_list]

    Output:
        [node_attr, edge_index, graph_id_node, graph_id_edge,
         node_id, edge_id, nodes_count, edges_count]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: list) -> list:
        """Convert ragged nodes and edge indices to disjoint format.

        Args:
            inputs: List of [nodes_list, edge_indices_list].
                - nodes_list: List of B tensors, each (N_i, F).
                - edge_indices_list: List of B tensors, each (M_i, 2).

        Returns:
            List of [node_attr, edge_index, graph_id_node, graph_id_edge,
                      node_id, edge_id, nodes_count, edges_count].
        """
        nodes_list, edge_indices_list = inputs
        device = nodes_list[0].device
        dtype_batch = self.dtype_batch

        node_lengths = torch.tensor([n.shape[0] for n in nodes_list],
                                    dtype=dtype_batch, device=device)
        edge_lengths = torch.tensor([e.shape[0] for e in edge_indices_list],
                                    dtype=dtype_batch, device=device)

        B = len(nodes_list)
        nodes_flatten = torch.cat(nodes_list, dim=0)
        edge_indices_flatten = torch.cat(edge_indices_list, dim=0)

        if self.dtype_index is not None:
            edge_indices_flatten = edge_indices_flatten.to(dtype=self.dtype_index)

        graph_id_node = torch.repeat_interleave(
            torch.arange(B, dtype=dtype_batch, device=device), node_lengths)
        graph_id_edge = torch.repeat_interleave(
            torch.arange(B, dtype=dtype_batch, device=device), edge_lengths)

        node_id_parts = [torch.arange(n.item(), dtype=dtype_batch, device=device) for n in node_lengths]
        node_id = torch.cat(node_id_parts, dim=0)
        edge_id_parts = [torch.arange(e.item(), dtype=dtype_batch, device=device) for e in edge_lengths]
        edge_id = torch.cat(edge_id_parts, dim=0)

        # Compute node offset for each edge
        node_splits = torch.cat([torch.zeros(1, dtype=dtype_batch, device=device),
                                 torch.cumsum(node_lengths, dim=0)], dim=0)
        offset = torch.repeat_interleave(node_splits[:-1], edge_lengths)
        offset = offset.unsqueeze(-1).expand_as(edge_indices_flatten)
        disjoint_indices = edge_indices_flatten + offset.to(edge_indices_flatten.dtype)

        # Transpose to PyG convention: (2, M)
        disjoint_indices = disjoint_indices.t().contiguous()

        if self.reverse_indices:
            disjoint_indices = disjoint_indices.flip(0)

        return [nodes_flatten, disjoint_indices, graph_id_node, graph_id_edge,
                node_id, edge_id, node_lengths, edge_lengths]


class CastDisjointToRaggedAttributes(_CastBatchedDisjointBase):
    """Cast disjoint attributes back to ragged (list of tensors) representation.

    Input:
        [attr, graph_id, item_id, item_counts]

    Output:
        List of B tensors, each of shape (N_i, F, ...).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: list) -> list:
        """Convert disjoint attributes to a list of per-graph tensors.

        Args:
            inputs: List of [attr, graph_id, item_id, item_counts].
                - attr: ([N], F, ...) flat attributes.
                - graph_id: ([N],) batch assignment.
                - item_id: ([N],) within-graph item index (unused here).
                - item_counts: (B,) number of items per graph.

        Returns:
            List of B tensors, each of shape (N_i, F, ...).
        """
        attr, graph_id, item_id, item_counts = inputs
        return list(torch.split(attr, item_counts.tolist(), dim=0))
