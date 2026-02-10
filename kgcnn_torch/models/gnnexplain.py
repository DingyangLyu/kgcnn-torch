"""GNNExplainer for explaining GNN predictions.

Based on Ying et al. (https://arxiv.org/abs/1903.03894).
Learns edge, feature, and node masks that explain a GNN's prediction on a given graph.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class GNNInterface:
    """Interface class that GNN models must implement to be explainable.

    The GNNExplainer requires a model that implements these methods.
    Typically this is a wrapper around an existing PyTorch GNN model.
    """

    def predict(self, data, **kwargs):
        """Return the prediction for the input graph.

        Args:
            data: PyG-style Data object or similar graph input.

        Returns:
            Prediction tensor.
        """
        raise NotImplementedError("Implement in subclass.")

    def masked_predict(self, data, edge_mask, feature_mask, node_mask, **kwargs):
        """Return prediction with masks applied.

        Args:
            data: Graph input.
            edge_mask: Tensor of shape (M, 1) masking edges.
            feature_mask: Tensor of shape (F, 1) masking node features.
            node_mask: Tensor of shape (N, 1) masking nodes.

        Returns:
            Masked prediction tensor.
        """
        raise NotImplementedError("Implement in subclass.")

    def get_number_of_nodes(self, data):
        """Return number of nodes in the graph."""
        raise NotImplementedError("Implement in subclass.")

    def get_number_of_edges(self, data):
        """Return number of edges in the graph."""
        raise NotImplementedError("Implement in subclass.")

    def get_number_of_node_features(self, data):
        """Return number of node features."""
        raise NotImplementedError("Implement in subclass.")

    def get_explanation(self, data, edge_mask, feature_mask, node_mask, **kwargs):
        """Convert learned masks into an explanation.

        Args:
            data: Original graph input.
            edge_mask: Learned edge mask (sigmoid-activated).
            feature_mask: Learned feature mask (sigmoid-activated).
            node_mask: Learned node mask (sigmoid-activated).

        Returns:
            Explanation in model-specific format.
        """
        raise NotImplementedError("Implement in subclass.")

    def present_explanation(self, explanation, **kwargs):
        """Present an explanation to the user (e.g., visualization)."""
        raise NotImplementedError("Implement in subclass.")


class GNNExplainerOptimizer(nn.Module):
    """Learns edge, feature, and node masks to explain a GNN prediction.

    Optimizes masks via gradient descent with sparsity regularization.
    Masks are stored as raw logits and activated with sigmoid.
    """

    def __init__(self, gnn_model: GNNInterface, graph_instance,
                 edge_mask_loss_weight: float = 1e-4,
                 edge_mask_norm_ord: float = 1.0,
                 feature_mask_loss_weight: float = 1e-4,
                 feature_mask_norm_ord: float = 1.0,
                 node_mask_loss_weight: float = 0.0,
                 node_mask_norm_ord: float = 1.0):
        """Initialize GNNExplainerOptimizer.

        Args:
            gnn_model: GNN model implementing GNNInterface.
            graph_instance: The graph to explain.
            edge_mask_loss_weight: Regularization weight for edge mask sparsity.
            edge_mask_norm_ord: Norm order for edge mask regularization.
            feature_mask_loss_weight: Regularization weight for feature mask sparsity.
            feature_mask_norm_ord: Norm order for feature mask regularization.
            node_mask_loss_weight: Regularization weight for node mask sparsity.
            node_mask_norm_ord: Norm order for node mask regularization.
        """
        super().__init__()
        self.gnn_model = gnn_model

        num_edges = gnn_model.get_number_of_edges(graph_instance)
        num_features = gnn_model.get_number_of_node_features(graph_instance)
        num_nodes = gnn_model.get_number_of_nodes(graph_instance)

        # Initialize mask logits with 5.0 (sigmoid(5) ≈ 0.993, nearly transparent)
        self.edge_mask = nn.Parameter(torch.full((num_edges, 1), 5.0))
        self.feature_mask = nn.Parameter(torch.full((num_features, 1), 5.0))
        self.node_mask = nn.Parameter(torch.full((num_nodes, 1), 5.0))

        # Store the target prediction (what we want to explain)
        with torch.no_grad():
            self.register_buffer(
                "output_to_explain",
                gnn_model.predict(graph_instance).detach().clone()
            )

        self.edge_mask_loss_weight = edge_mask_loss_weight
        self.edge_mask_norm_ord = edge_mask_norm_ord
        self.feature_mask_loss_weight = feature_mask_loss_weight
        self.feature_mask_norm_ord = feature_mask_norm_ord
        self.node_mask_loss_weight = node_mask_loss_weight
        self.node_mask_norm_ord = node_mask_norm_ord

    def get_mask(self, mask_identifier: str) -> torch.Tensor:
        """Get activated mask by name.

        Args:
            mask_identifier: One of 'edge', 'feature', 'node'.

        Returns:
            Sigmoid-activated mask, or ones if that mask's weight is 0.
        """
        if mask_identifier == "edge":
            if self.edge_mask_loss_weight > 0:
                return torch.sigmoid(self.edge_mask)
            return torch.ones_like(self.edge_mask)
        elif mask_identifier == "feature":
            if self.feature_mask_loss_weight > 0:
                return torch.sigmoid(self.feature_mask)
            return torch.ones_like(self.feature_mask)
        elif mask_identifier == "node":
            if self.node_mask_loss_weight > 0:
                return torch.sigmoid(self.node_mask)
            return torch.ones_like(self.node_mask)
        raise ValueError(f"mask_identifier must be 'edge', 'feature' or 'node', got '{mask_identifier}'")

    def forward(self, graph_instance):
        """Forward pass: apply masks and return prediction + regularization loss.

        Args:
            graph_instance: Graph input.

        Returns:
            Tuple of (masked_prediction, regularization_loss).
        """
        edge_mask = self.get_mask("edge")
        feature_mask = self.get_mask("feature")
        node_mask = self.get_mask("node")

        y_pred = self.gnn_model.masked_predict(
            graph_instance, edge_mask, feature_mask, node_mask)

        # Compute regularization losses encouraging sparsity
        reg_loss = torch.tensor(0.0, device=y_pred.device)
        if self.edge_mask_loss_weight > 0:
            reg_loss = reg_loss + self.edge_mask_loss_weight * torch.norm(
                torch.sigmoid(self.edge_mask), p=self.edge_mask_norm_ord)
        if self.feature_mask_loss_weight > 0:
            reg_loss = reg_loss + self.feature_mask_loss_weight * torch.norm(
                torch.sigmoid(self.feature_mask), p=self.feature_mask_norm_ord)
        if self.node_mask_loss_weight > 0:
            reg_loss = reg_loss + self.node_mask_loss_weight * torch.norm(
                torch.sigmoid(self.node_mask), p=self.node_mask_norm_ord)

        return y_pred, reg_loss


class GNNExplainer:
    """Explains GNN predictions by learning edge, feature, and node masks.

    Based on Ying et al. (https://arxiv.org/abs/1903.03894).

    Example usage::

        class MyGNNWrapper(GNNInterface):
            def __init__(self, model):
                self.model = model
                self.model.eval()

            def predict(self, data, **kwargs):
                with torch.no_grad():
                    return self.model(data)

            def masked_predict(self, data, edge_mask, feature_mask, node_mask, **kwargs):
                # Apply masks to data before prediction
                data.edge_weight = edge_mask.squeeze(-1)
                data.x = data.x * feature_mask.T  # broadcast (N, F) * (1, F)
                return self.model(data)

            def get_number_of_nodes(self, data): return data.num_nodes
            def get_number_of_edges(self, data): return data.edge_index.shape[1]
            def get_number_of_node_features(self, data): return data.x.shape[1]

            def get_explanation(self, data, edge_mask, feature_mask, node_mask, **kwargs):
                return {"edge_mask": edge_mask, "feature_mask": feature_mask, "node_mask": node_mask}

        gnn = MyGNNWrapper(trained_model)
        explainer = GNNExplainer(gnn)
        info = explainer.explain(data, epochs=200)
        explanation = explainer.get_explanation()
    """

    def __init__(self, gnn: GNNInterface,
                 optimizer_options: Optional[dict] = None,
                 lr: float = 0.01,
                 epochs: int = 100,
                 loss_fn: str = "mse"):
        """Initialize GNNExplainer.

        Args:
            gnn: GNN model implementing GNNInterface.
            optimizer_options: Dict of kwargs for GNNExplainerOptimizer.
            lr: Learning rate for mask optimization.
            epochs: Number of optimization epochs.
            loss_fn: Loss function name ('mse' or 'cross_entropy').
        """
        self.gnn = gnn
        self.optimizer_options = optimizer_options or {}
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.gnnx_optimizer = None
        self.graph_instance = None

    def explain(self, graph_instance, output_to_explain=None,
                epochs: int = None, lr: float = None,
                inspection: bool = False, verbose: bool = False,
                device: torch.device = None) -> Optional[dict]:
        """Find masks explaining the GNN's prediction on graph_instance.

        Args:
            graph_instance: Graph input to explain.
            output_to_explain: Override the target output. If None, uses the
                GNN's own prediction.
            epochs: Override number of epochs.
            lr: Override learning rate.
            inspection: If True, return dict with per-epoch losses and predictions.
            verbose: If True, print progress.
            device: Device to run on.

        Returns:
            If inspection=True, returns dict with training history.
            Otherwise None (masks stored internally).
        """
        epochs = epochs or self.epochs
        lr = lr or self.lr
        self.graph_instance = graph_instance

        # Create optimizer module
        self.gnnx_optimizer = GNNExplainerOptimizer(
            self.gnn, graph_instance, **self.optimizer_options)

        if device is not None:
            self.gnnx_optimizer = self.gnnx_optimizer.to(device)

        if output_to_explain is not None:
            self.gnnx_optimizer.output_to_explain = output_to_explain.detach()

        # Set up optimizer (only optimize mask parameters)
        optimizer = torch.optim.Adam(self.gnnx_optimizer.parameters(), lr=lr)

        # Select loss function
        if self.loss_fn == "cross_entropy":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        # Inspection tracking
        history = {
            "total_loss": [], "pred_loss": [], "reg_loss": [],
            "edge_mask_loss": [], "feature_mask_loss": [],
            "node_mask_loss": [], "predictions": []
        } if inspection else None

        # Optimization loop
        self.gnnx_optimizer.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            y_pred, reg_loss = self.gnnx_optimizer(graph_instance)
            pred_loss = criterion(y_pred, self.gnnx_optimizer.output_to_explain)
            total_loss = pred_loss + reg_loss

            total_loss.backward()
            optimizer.step()

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs}: loss={total_loss.item():.6f} "
                      f"(pred={pred_loss.item():.6f}, reg={reg_loss.item():.6f})")

            if inspection:
                with torch.no_grad():
                    history["total_loss"].append(total_loss.item())
                    history["pred_loss"].append(pred_loss.item())
                    history["reg_loss"].append(reg_loss.item())
                    history["predictions"].append(y_pred.detach().cpu().numpy())
                    opt = self.gnnx_optimizer
                    if opt.edge_mask_loss_weight > 0:
                        el = opt.edge_mask_loss_weight * torch.norm(
                            torch.sigmoid(opt.edge_mask), p=opt.edge_mask_norm_ord).item()
                        history["edge_mask_loss"].append(el)
                    if opt.feature_mask_loss_weight > 0:
                        fl = opt.feature_mask_loss_weight * torch.norm(
                            torch.sigmoid(opt.feature_mask), p=opt.feature_mask_norm_ord).item()
                        history["feature_mask_loss"].append(fl)
                    if opt.node_mask_loss_weight > 0:
                        nl = opt.node_mask_loss_weight * torch.norm(
                            torch.sigmoid(opt.node_mask), p=opt.node_mask_norm_ord).item()
                        history["node_mask_loss"].append(nl)

        self.gnnx_optimizer.eval()

        if inspection:
            return history

    def get_explanation(self, **kwargs):
        """Get explanation from the learned masks.

        Calls GNNInterface.get_explanation with the optimized masks.

        Returns:
            Explanation in model-specific format.

        Raises:
            RuntimeError: If explain() has not been called yet.
        """
        if self.graph_instance is None or self.gnnx_optimizer is None:
            raise RuntimeError("Call explain() before get_explanation().")

        edge_mask = self.gnnx_optimizer.get_mask("edge").detach()
        feature_mask = self.gnnx_optimizer.get_mask("feature").detach()
        node_mask = self.gnnx_optimizer.get_mask("node").detach()

        return self.gnn.get_explanation(
            self.graph_instance, edge_mask, feature_mask, node_mask, **kwargs)

    def get_masks(self):
        """Get the learned masks directly.

        Returns:
            Dict with 'edge', 'feature', 'node' masks (sigmoid-activated, detached).

        Raises:
            RuntimeError: If explain() has not been called yet.
        """
        if self.gnnx_optimizer is None:
            raise RuntimeError("Call explain() before get_masks().")

        return {
            "edge": self.gnnx_optimizer.get_mask("edge").detach(),
            "feature": self.gnnx_optimizer.get_mask("feature").detach(),
            "node": self.gnnx_optimizer.get_mask("node").detach(),
        }

    def present_explanation(self, explanation, **kwargs):
        """Present an explanation using the GNN's present_explanation method."""
        return self.gnn.present_explanation(explanation, **kwargs)
