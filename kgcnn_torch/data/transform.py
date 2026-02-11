"""Data transformation and scaling utilities (numpy-based).

These operate on numpy arrays before conversion to PyG Data objects.
For torch-based scaling during training, see kgcnn_torch.layers.scale.
"""
import json
import numpy as np
from typing import Optional, List, Union, Tuple, Dict


class StandardScaler:
    """Standard scaling: (X - mean) / std.

    Numpy-based scaler for preprocessing, compatible with sklearn API.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.atleast_1d(np.mean(X, axis=0))
        self.scale_ = np.atleast_1d(np.std(X, axis=0))
        self.scale_[self.scale_ < 1e-7] = 1.0
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.mean_

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def get_weights(self) -> dict:
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    def set_weights(self, weights: dict):
        self.mean_ = np.array(weights["mean"])
        self.scale_ = np.array(weights["scale"])
        self._is_fitted = True

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.get_weights(), f)

    def load(self, filepath: str):
        with open(filepath, 'r') as f:
            self.set_weights(json.load(f))

    def partial_fit(self, X: np.ndarray, y=None, sample_weight=None, **kwargs):
        """Online computation of mean and std on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Uses the parallel algorithm from Chan et al. (1982) for numerically
        stable incremental mean and variance computation.

        Args:
            X: Array of shape (n_samples, n_features).
            y: Ignored (API compatibility).
            sample_weight: Not used.

        Returns:
            self: Fitted scaler.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]

        if not self._is_fitted:
            # First call: initialize from this batch
            self._n_samples_seen = X.shape[0]
            self.mean_ = np.mean(X, axis=0)
            self._var = np.var(X, axis=0)
            self.scale_ = np.sqrt(self._var)
            self.scale_[self.scale_ < 1e-7] = 1.0
            self._is_fitted = True
            return self

        # Incremental update (Chan et al. 1982, Eq 1.5a,b)
        new_n = X.shape[0]
        new_mean = np.mean(X, axis=0)
        new_var = np.var(X, axis=0)

        old_n = self._n_samples_seen
        old_mean = self.mean_
        old_var = self._var

        total_n = old_n + new_n
        delta = new_mean - old_mean

        self.mean_ = (old_n * old_mean + new_n * new_mean) / total_n
        self._var = (old_n * (old_var + delta ** 2 * new_n / total_n) +
                     new_n * new_var) / total_n
        self._n_samples_seen = total_n

        self.scale_ = np.sqrt(self._var)
        self.scale_[self.scale_ < 1e-7] = 1.0
        return self

    def get_scaling(self):
        """Get scale of shape (1, n_properties).

        Returns:
            np.ndarray or None: Scale array expanded with leading axis,
                or None if scaler has not been fitted.
        """
        if self.scale_ is None:
            return None
        return np.expand_dims(self.scale_, axis=0)

    def get_config(self) -> dict:
        return {}

    def set_config(self, config: dict):
        pass


class StandardLabelScaler(StandardScaler):
    """Standard scaler for labels/targets.

    Wraps ``StandardScaler`` with dataset-level methods matching Keras API.
    """

    def fit(self, y: np.ndarray, **kwargs):
        return super().fit(y)

    def transform(self, y: np.ndarray, **kwargs) -> np.ndarray:
        return super().transform(y)

    def inverse_transform(self, y: np.ndarray, **kwargs) -> np.ndarray:
        return super().inverse_transform(y)

    def fit_dataset(self, dataset: List[Dict[str, np.ndarray]],
                    y_key: str = "graph_labels"):
        """Fit scaler from a list of graph dictionaries.

        Args:
            dataset: List of dicts with graph data.
            y_key: Key for label values. Default is 'graph_labels'.

        Returns:
            self
        """
        y = np.array([item[y_key] for item in dataset])
        return self.fit(y)

    def transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                          y_key: str = "graph_labels",
                          copy: bool = True) -> List[Dict[str, np.ndarray]]:
        """Transform labels in a list of graph dictionaries.

        Args:
            dataset: List of dicts with graph data.
            y_key: Key for label values.
            copy: Whether to copy data. Default is True.

        Returns:
            Transformed dataset.
        """
        y = np.array([item[y_key] for item in dataset])
        out = self.transform(y)
        for i, item in enumerate(dataset):
            item[y_key] = out[i]
        return dataset

    def inverse_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                                  y_key: str = "graph_labels",
                                  copy: bool = True) -> List[Dict[str, np.ndarray]]:
        """Inverse-transform labels in a list of graph dictionaries.

        Args:
            dataset: List of dicts with graph data.
            y_key: Key for label values.
            copy: Whether to copy data. Default is True.

        Returns:
            Inverse-transformed dataset.
        """
        y = np.array([item[y_key] for item in dataset])
        out = self.inverse_transform(y)
        for i, item in enumerate(dataset):
            item[y_key] = out[i]
        return dataset

    def fit_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                              y_key: str = "graph_labels",
                              copy: bool = True) -> List[Dict[str, np.ndarray]]:
        """Fit and transform labels in one step.

        Args:
            dataset: List of dicts with graph data.
            y_key: Key for label values.
            copy: Whether to copy data. Default is True.

        Returns:
            Transformed dataset.
        """
        self.fit_dataset(dataset, y_key=y_key)
        return self.transform_dataset(dataset, y_key=y_key, copy=copy)


class ExtensiveMolecularScaler:
    """Extensive property scaler.

    Removes per-atom energy offset using ridge regression on atom counts,
    then scales residuals.

    For extensive properties like total energy that scale with system size.
    """

    def __init__(self, standardize_scale: bool = True):
        self._standardize_scale = standardize_scale
        self.ridge = None
        self.scale_ = None
        self._fit_atom_mask = None
        self._is_fitted = False

    # Fixed vocabulary size matching Keras convention.
    max_atomic_number = 95

    def fit(self, X: np.ndarray, atomic_number: list = None, **kwargs):
        """Fit scaler.

        Args:
            X: Target values (N,) or (N, D).
            atomic_number: List of arrays, each containing atomic numbers per molecule.
        """
        from sklearn.linear_model import Ridge

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]

        # Build atom count matrix with fixed size (matches Keras)
        max_z = self.max_atomic_number
        counts = np.zeros((len(atomic_number), max_z), dtype=np.float64)
        for i, z_arr in enumerate(atomic_number):
            z_arr = np.asarray(z_arr).ravel()
            for z in z_arr:
                zi = int(z)
                if zi < max_z:
                    counts[i, zi] += 1

        # Mask for atoms that appear
        self._fit_atom_mask = counts.sum(axis=0) > 0

        # Ridge regression (alpha=1e-9 matches Keras)
        self.ridge = Ridge(alpha=1e-9, fit_intercept=False)
        self.ridge.fit(counts[:, self._fit_atom_mask], X)

        # Residual (sklearn squeezes (N,1) predict to (N,), so reshape)
        offset = self.ridge.predict(counts[:, self._fit_atom_mask])
        if offset.ndim == 1:
            offset = offset[:, None]
        residual = X - offset
        self.scale_ = np.std(residual, axis=0)
        self.scale_[self.scale_ < 1e-7] = 1.0
        self._is_fitted = True
        return self

    def _count_matrix(self, atomic_number: list) -> np.ndarray:
        max_z = len(self._fit_atom_mask)
        counts = np.zeros((len(atomic_number), max_z), dtype=np.float64)
        for i, z_arr in enumerate(atomic_number):
            z_arr = np.asarray(z_arr).ravel()
            for z in z_arr:
                zi = int(z)
                if zi < max_z:
                    counts[i, zi] += 1
                else:
                    import warnings
                    warnings.warn(
                        f"Atomic number {zi} exceeds max_atomic_number={max_z}. "
                        f"Atom will be ignored in scaling.", stacklevel=2
                    )
        return counts[:, self._fit_atom_mask]

    def transform(self, X: np.ndarray, atomic_number: list = None, **kwargs) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        counts = self._count_matrix(atomic_number)
        offset = self.ridge.predict(counts)
        if offset.ndim == 1:
            offset = offset[:, None]
        residual = X - offset
        if self._standardize_scale:
            return residual / self.scale_
        return residual

    def inverse_transform(self, X: np.ndarray, atomic_number: list = None, **kwargs) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        counts = self._count_matrix(atomic_number)
        offset = self.ridge.predict(counts)
        if offset.ndim == 1:
            offset = offset[:, None]
        if self._standardize_scale:
            return X * self.scale_ + offset
        return X + offset

    def fit_transform(self, X: np.ndarray, atomic_number: list = None, **kwargs) -> np.ndarray:
        self.fit(X, atomic_number=atomic_number)
        return self.transform(X, atomic_number=atomic_number)

    def get_weights(self) -> dict:
        return {
            "coef": self.ridge.coef_.tolist(),
            "scale": self.scale_.tolist(),
            "atom_mask": self._fit_atom_mask.tolist()
        }

    def set_weights(self, weights: dict):
        from sklearn.linear_model import Ridge
        self.ridge = Ridge(alpha=1e-9, fit_intercept=False)
        self.ridge.coef_ = np.array(weights["coef"])
        self.ridge.intercept_ = 0.0
        self.scale_ = np.array(weights["scale"])
        self._fit_atom_mask = np.array(weights["atom_mask"], dtype=bool)
        self._is_fitted = True

    def save(self, filepath: str):
        """Save scaler weights to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.get_weights(), f)

    def load(self, filepath: str):
        """Load scaler weights from JSON."""
        with open(filepath, "r") as f:
            self.set_weights(json.load(f))

    def _predict(self, atomic_number: list) -> np.ndarray:
        """Predict the offset from atomic numbers. Requires :obj:`fit()` called previously.

        Args:
            atomic_number: List of arrays of atomic numbers.
                Example: ``[np.array([7,1,1,1]), ...]``.

        Returns:
            np.ndarray: Predicted offset of shape ``(n_samples, n_properties)``.
        """
        if self._fit_atom_mask is None:
            raise ValueError(
                "`ExtensiveMolecularScaler` has not been fitted yet. Cannot predict.")
        counts = self._count_matrix(atomic_number)
        offset = self.ridge.predict(counts)
        if offset.ndim == 1:
            offset = offset[:, None]
        return offset

    def _plot_predict(self, molecular_property: np.ndarray,
                      atomic_number: list):
        """Debug function to check prediction quality.

        Creates a scatter plot of fitted (predicted) vs actual molecular
        properties with MAE per property dimension.

        Args:
            molecular_property: Array of actual properties, shape ``(N,)`` or ``(N, D)``.
            atomic_number: List of arrays of atomic numbers.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        import matplotlib.pyplot as plt

        molecular_property = np.array(molecular_property)
        if molecular_property.ndim <= 1:
            molecular_property = np.expand_dims(molecular_property, axis=-1)
        predict_prop = self._predict(atomic_number)
        if predict_prop.ndim <= 1:
            predict_prop = np.expand_dims(predict_prop, axis=-1)
        mae = np.mean(np.abs(molecular_property - predict_prop), axis=0)
        fig = plt.figure()
        for i in range(predict_prop.shape[-1]):
            plt.scatter(predict_prop[:, i], molecular_property[:, i], alpha=0.3,
                        label="Pos: " + str(i) + " MAE: {0:0.4f} ".format(mae[i]))
        plt.plot(np.arange(np.amin(molecular_property), np.amax(molecular_property), 0.05),
                 np.arange(np.amin(molecular_property), np.amax(molecular_property), 0.05),
                 color='red')
        plt.xlabel('Fitted')
        plt.ylabel('Actual')
        plt.legend(loc='upper left', fontsize='x-small')
        plt.show()
        return fig

    def get_scaling(self):
        """Get scale of shape (1, n_properties).

        Returns:
            np.ndarray or None: Scale array expanded with leading axis,
                or None if scaler has not been fitted.
        """
        if self.scale_ is None:
            return None
        return np.expand_dims(self.scale_, axis=0)

    def get_config(self) -> dict:
        return {"standardize_scale": self._standardize_scale}

    def set_config(self, config: dict):
        self._standardize_scale = config.get("standardize_scale", True)

    def fit_dataset(self, dataset: List[Dict[str, np.ndarray]],
                    X_key: str = "graph_labels",
                    atomic_number_key: str = "atomic_number"):
        """Fit scaler from a list of graph dictionaries.

        Args:
            dataset: List of dicts with graph data.
            X_key: Key for property values.
            atomic_number_key: Key for atomic number arrays.

        Returns:
            self
        """
        X = np.array([item[X_key] for item in dataset])
        atoms = [item[atomic_number_key] for item in dataset]
        return self.fit(X, atomic_number=atoms)

    def transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                          X_key: str = "graph_labels",
                          atomic_number_key: str = "atomic_number",
                          copy: bool = True) -> List[Dict[str, np.ndarray]]:
        """Transform properties in a list of graph dictionaries.

        Args:
            dataset: List of dicts with graph data.
            X_key: Key for property values.
            atomic_number_key: Key for atomic number arrays.
            copy: Whether to copy data. Default is True.

        Returns:
            Transformed dataset.
        """
        X = np.array([item[X_key] for item in dataset])
        atoms = [item[atomic_number_key] for item in dataset]
        out = self.transform(X, atomic_number=atoms)
        for i, item in enumerate(dataset):
            item[X_key] = out[i]
        return dataset

    def inverse_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                                  X_key: str = "graph_labels",
                                  atomic_number_key: str = "atomic_number",
                                  copy: bool = True) -> List[Dict[str, np.ndarray]]:
        """Inverse-transform properties in a list of graph dictionaries.

        Args:
            dataset: List of dicts with graph data.
            X_key: Key for property values.
            atomic_number_key: Key for atomic number arrays.
            copy: Whether to copy data. Default is True.

        Returns:
            Inverse-transformed dataset.
        """
        X = np.array([item[X_key] for item in dataset])
        atoms = [item[atomic_number_key] for item in dataset]
        out = self.inverse_transform(X, atomic_number=atoms)
        for i, item in enumerate(dataset):
            item[X_key] = out[i]
        return dataset

    def fit_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                              X_key: str = "graph_labels",
                              atomic_number_key: str = "atomic_number",
                              copy: bool = True) -> List[Dict[str, np.ndarray]]:
        """Fit and transform in one step.

        Args:
            dataset: List of dicts with graph data.
            X_key: Key for property values.
            atomic_number_key: Key for atomic number arrays.
            copy: Whether to copy data. Default is True.

        Returns:
            Transformed dataset.
        """
        self.fit_dataset(dataset, X_key=X_key, atomic_number_key=atomic_number_key)
        return self.transform_dataset(dataset, X_key=X_key,
                                      atomic_number_key=atomic_number_key, copy=copy)


class ExtensiveMolecularLabelScaler(ExtensiveMolecularScaler):
    """Extensive property scaler with label-oriented API.

    Same as ``ExtensiveMolecularScaler`` but uses ``y`` for labels and
    ``X`` or ``atomic_number`` for atomic numbers. Mirrors the Keras
    ``ExtensiveMolecularLabelScaler`` interface.
    """

    def fit(self, y=None, X=None, atomic_number=None, sample_weight=None, **kwargs):
        """Fit scaler to label data.

        Args:
            y: Label array of shape (N,) or (N, D).
            X: List of arrays of atomic numbers (used if atomic_number is None).
            atomic_number: List of arrays of atomic numbers.
            sample_weight: Not used (API compatibility).

        Returns:
            self
        """
        if y is None:
            raise ValueError("ExtensiveMolecularLabelScaler requires 'y' argument.")
        atomic_number = atomic_number if atomic_number is not None else X
        return super().fit(y, atomic_number=atomic_number)

    def transform(self, y=None, X=None, atomic_number=None, **kwargs):
        """Transform labels.

        Args:
            y: Label array of shape (N,) or (N, D).
            X: List of arrays of atomic numbers (used if atomic_number is None).
            atomic_number: List of arrays of atomic numbers.

        Returns:
            Scaled labels.
        """
        if y is None:
            raise ValueError("ExtensiveMolecularLabelScaler requires 'y' argument.")
        atomic_number = atomic_number if atomic_number is not None else X
        return super().transform(y, atomic_number=atomic_number)

    def inverse_transform(self, y=None, X=None, atomic_number=None, **kwargs):
        """Inverse-transform labels.

        Args:
            y: Scaled label array.
            X: List of arrays of atomic numbers (used if atomic_number is None).
            atomic_number: List of arrays of atomic numbers.

        Returns:
            Labels in original scale.
        """
        if y is None:
            raise ValueError("ExtensiveMolecularLabelScaler requires 'y' argument.")
        atomic_number = atomic_number if atomic_number is not None else X
        return super().inverse_transform(y, atomic_number=atomic_number)

    def fit_transform(self, y=None, X=None, atomic_number=None, **kwargs):
        self.fit(y=y, X=X, atomic_number=atomic_number)
        return self.transform(y=y, X=X, atomic_number=atomic_number)


class ForceStandardScaler(ExtensiveMolecularScaler):
    """Extensive scaler for jointly scaling energy and forces.

    Extends ``ExtensiveMolecularScaler`` to handle both energy and force targets.
    The energy is scaled by removing per-atom offsets (via ridge regression on
    atom counts) and optionally dividing by the residual standard deviation.
    Forces are scaled by the same standard deviation factor.

    Units for energy and forces must be consistent.

    Supports two calling conventions (matching Keras ``EnergyForceExtensiveLabelScaler``):

    1. **Tuple form**: ``fit(y=(energy, forces), X=mol_num)``
    2. **Separate args form**: ``fit(y=energy, force=forces, atomic_number=mol_num)``

    Example:

    .. code-block:: python

        import numpy as np
        from kgcnn_torch.data.transform import ForceStandardScaler
        energy = np.random.rand(5).reshape((5, 1))
        mol_num = [np.array([6, 1, 1, 1, 1]), np.array([7, 1, 1, 1]),
            np.array([6, 6, 1, 1, 1, 1]), np.array([6, 6, 1, 1]),
            np.array([6, 6, 1, 1, 1, 1, 1, 1])]
        force = [np.random.rand(len(m) * 3).reshape((len(m), 3)) for m in mol_num]
        scaler = ForceStandardScaler()
        scaler.fit(y=(energy, force), X=mol_num)
        scaled_e, scaled_f = scaler.transform(y=(energy, force), X=mol_num)
        orig_e, orig_f = scaler.inverse_transform(y=(scaled_e, scaled_f), X=mol_num)
    """

    def __init__(self, standardize_scale: bool = True,
                 energy: str = "energy", force: str = "force",
                 atomic_number: str = "atomic_number",
                 sample_weight: str = None):
        """Initialize ForceStandardScaler.

        Args:
            standardize_scale (bool): Whether to divide by residual std. Default is True.
            energy (str): Key name for energy in dataset dicts. Default is ``"energy"``.
            force (str): Key name for forces in dataset dicts. Default is ``"force"``.
            atomic_number (str): Key name for atomic numbers in dataset dicts.
                Default is ``"atomic_number"``.
            sample_weight (str): Key name for sample weights. Default is None.
        """
        super().__init__(standardize_scale=standardize_scale)
        self._energy = energy
        self._force = force
        self._atomic_number = atomic_number
        self._sample_weight = sample_weight

    def _verify_input(self, X, y, force, atomic_number):
        """Verify and normalize input arguments.

        Supports both tuple form ``y=(energy, forces)`` and separate args
        ``y=energy, force=forces, atomic_number=atoms``, matching the Keras
        ``EnergyForceExtensiveLabelScaler`` interface.

        Args:
            X: Atomic numbers (used if atomic_number is None).
            y: Either tuple of (energy, forces) or just energy array.
            force: Separate forces argument (if not included in y).
            atomic_number: Separate atomic numbers argument.

        Returns:
            tuple: (x_input, energy, forces, atoms)
        """
        if y is None:
            raise ValueError("ForceStandardScaler requires 'y' argument, got None.")
        if force is not None:
            if isinstance(y, (tuple, list)) and len(y) == 2:
                energy, forces = y[0], force
            else:
                energy, forces = y, force
        else:
            if isinstance(y, (tuple, list)) and len(y) == 2:
                energy, forces = y
            else:
                raise ValueError(
                    "y must be a tuple of (energy, forces), or pass forces "
                    "separately via the 'force' argument.")
        if atomic_number is not None:
            atoms = atomic_number
            x_input = X
        else:
            atoms = X
            x_input = None
        return x_input, energy, forces, atoms

    def fit(self, y=None, *,
            X: List[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            force: Optional[List[np.ndarray]] = None,
            atomic_number: Optional[List[np.ndarray]] = None,
            **kwargs):
        """Fit scaler to energy and force data.

        The scaler is fit on the energy data only (forces are scaled by the same
        factor derived from energy residuals).

        Args:
            y: Tuple of ``(energy, forces)`` or just energy array (if ``force``
                is given separately).
            X (list): List of arrays of atomic numbers per molecule (used if
                ``atomic_number`` is None).
            sample_weight: Not used (kept for API compatibility).
            force (list): List of force arrays. If provided, ``y`` is treated
                as energy only.
            atomic_number (list): List of arrays of atomic numbers. If provided,
                ``X`` is ignored for atom counts.

        Returns:
            self
        """
        _, energy, forces, atoms = self._verify_input(X, y, force, atomic_number)
        return super().fit(energy, atomic_number=atoms)

    def transform(self, y=None, *,
                  X: List[np.ndarray] = None,
                  force: Optional[List[np.ndarray]] = None,
                  atomic_number: Optional[List[np.ndarray]] = None,
                  copy: bool = True, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Scale energy and forces.

        Args:
            y: Tuple of ``(energy, forces)`` or just energy array.
            X (list): List of arrays of atomic numbers.
            force (list): List of force arrays (if not in y).
            atomic_number (list): List of arrays of atomic numbers.
            copy (bool): Whether to copy the data. Default is True.

        Returns:
            tuple: ``(scaled_energy, scaled_forces)``.
        """
        _, energy, forces, atoms = self._verify_input(X, y, force, atomic_number)
        scaled_energy = super().transform(energy, atomic_number=atoms)

        if self._standardize_scale:
            # scale_ shape is (D,) where D = number of energy states.
            # For single-state (D=1), this is a scalar broadcast.
            s = self.scale_.ravel()
            if copy:
                scaled_forces = [np.array(f, dtype=np.float64) / s for f in forces]
            else:
                scaled_forces = forces
                for f in scaled_forces:
                    f[:] = f / s
        else:
            scaled_forces = [np.array(f) for f in forces] if copy else forces

        return scaled_energy, scaled_forces

    def inverse_transform(self, y=None, *,
                          X: List[np.ndarray] = None,
                          force: Optional[List[np.ndarray]] = None,
                          atomic_number: Optional[List[np.ndarray]] = None,
                          copy: bool = True, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Inverse-scale energy and forces.

        Args:
            y: Tuple of ``(scaled_energy, scaled_forces)`` or just energy array.
            X (list): List of arrays of atomic numbers.
            force (list): List of force arrays (if not in y).
            atomic_number (list): List of arrays of atomic numbers.
            copy (bool): Whether to copy the data. Default is True.

        Returns:
            tuple: ``(energy, forces)`` in original scale.
        """
        _, energy, forces, atoms = self._verify_input(X, y, force, atomic_number)
        orig_energy = super().inverse_transform(energy, atomic_number=atoms)

        if self._standardize_scale:
            s = self.scale_.ravel()
            if copy:
                orig_forces = [np.array(f, dtype=np.float64) * s for f in forces]
            else:
                orig_forces = forces
                for f in orig_forces:
                    f[:] = f * s
        else:
            orig_forces = [np.array(f) for f in forces] if copy else forces

        return orig_energy, orig_forces

    def fit_transform(self, y=None, *,
                      X: List[np.ndarray] = None,
                      force: Optional[List[np.ndarray]] = None,
                      atomic_number: Optional[List[np.ndarray]] = None,
                      copy: bool = True, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Fit and transform in one step.

        Args:
            y: Tuple of ``(energy, forces)`` or just energy array.
            X (list): List of arrays of atomic numbers.
            force (list): List of force arrays (if not in y).
            atomic_number (list): List of arrays of atomic numbers.
            copy (bool): Whether to copy the data. Default is True.

        Returns:
            tuple: ``(scaled_energy, scaled_forces)``.
        """
        self.fit(y=y, X=X, force=force, atomic_number=atomic_number)
        return self.transform(y=y, X=X, force=force, atomic_number=atomic_number, copy=copy)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "energy": self._energy,
            "force": self._force,
            "atomic_number": self._atomic_number,
            "sample_weight": self._sample_weight,
        })
        return config

    def set_config(self, config: dict):
        self._energy = config.get("energy", "energy")
        self._force = config.get("force", "force")
        self._atomic_number = config.get("atomic_number", "atomic_number")
        self._sample_weight = config.get("sample_weight", None)
        config_super = {k: v for k, v in config.items()
                        if k not in ("energy", "force", "atomic_number", "sample_weight")}
        super().set_config(config_super)

    def fit_dataset(self, dataset: List[Dict[str, np.ndarray]], **fit_params):
        """Fit scaler from a list of graph dictionaries.

        Uses key names from ``__init__`` (energy, force, atomic_number).

        Args:
            dataset (list): List of dicts with graph data.
            fit_params: Additional fit parameters.

        Returns:
            self
        """
        atoms_key = self._atomic_number
        energy_key, force_key = self._energy, self._force
        return self.fit(
            X=[item[atoms_key] for item in dataset],
            y=([item[energy_key] for item in dataset],
               [item[force_key] for item in dataset]),
            sample_weight=([item[self._sample_weight] for item in dataset]
                           if self._sample_weight is not None else None),
            **fit_params
        )

    def transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                          copy: bool = True, copy_dataset: bool = False,
                          ) -> List[Dict[str, np.ndarray]]:
        """Transform a list of graph dictionaries.

        Uses key names from ``__init__`` (energy, force, atomic_number).

        Args:
            dataset (list): List of dicts with graph data.
            copy (bool): Whether to copy data. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.

        Returns:
            list: Transformed dataset.
        """
        atoms_key = self._atomic_number
        energy_key, force_key = self._energy, self._force
        if copy_dataset:
            dataset = dataset.copy()
        out_e, out_f = self.transform(
            X=[item[atoms_key] for item in dataset],
            y=([item[energy_key] for item in dataset],
               [item[force_key] for item in dataset]),
            copy=copy,
        )
        for i, item in enumerate(dataset):
            item[energy_key] = out_e[i] if out_e.ndim > 1 else out_e[i:i + 1]
            item[force_key] = out_f[i]
        return dataset

    def inverse_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                                  copy: bool = True, copy_dataset: bool = False,
                                  ) -> List[Dict[str, np.ndarray]]:
        """Inverse-transform a list of graph dictionaries.

        Uses key names from ``__init__`` (energy, force, atomic_number).

        Args:
            dataset (list): List of dicts with graph data.
            copy (bool): Whether to copy data. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.

        Returns:
            list: Inverse-transformed dataset.
        """
        atoms_key = self._atomic_number
        energy_key, force_key = self._energy, self._force
        if copy_dataset:
            dataset = dataset.copy()
        out_e, out_f = self.inverse_transform(
            X=[item[atoms_key] for item in dataset],
            y=([item[energy_key] for item in dataset],
               [item[force_key] for item in dataset]),
            copy=copy,
        )
        for i, item in enumerate(dataset):
            item[energy_key] = out_e[i] if out_e.ndim > 1 else out_e[i:i + 1]
            item[force_key] = out_f[i]
        return dataset

    def fit_transform_dataset(self, dataset: List[Dict[str, np.ndarray]],
                              copy: bool = True, copy_dataset: bool = False,
                              **fit_params) -> List[Dict[str, np.ndarray]]:
        """Fit and transform dataset in one step.

        Args:
            dataset (list): List of dicts with graph data.
            copy (bool): Whether to copy data. Default is True.
            copy_dataset (bool): Whether to copy full dataset. Default is False.
            fit_params: Additional fit parameters.

        Returns:
            list: Transformed dataset.
        """
        self.fit_dataset(dataset=dataset, **fit_params)
        return self.transform_dataset(dataset=dataset, copy=copy, copy_dataset=copy_dataset)


# Alias for Keras API compatibility
EnergyForceExtensiveLabelScaler = ForceStandardScaler


class QMGraphLabelScaler:
    """Scaler for QM (quantum mechanics) graph-level labels.

    Supports two modes:

    1. **Simple mode** (default): Per-column standard scaling for multi-target
       QM property prediction. Created with ``QMGraphLabelScaler()``.

    2. **Composition mode** (Keras-compatible): A list of different scalers,
       one per target column. Allows mixing ``ExtensiveMolecularScaler`` for
       extensive properties and ``StandardScaler`` for intensive ones.
       Created with ``QMGraphLabelScaler(scaler=[scaler1, scaler2, ...])``.

    Example (simple mode):

    .. code-block:: python

        import numpy as np
        from kgcnn_torch.data.transform import QMGraphLabelScaler
        labels = np.random.rand(100, 5)
        scaler = QMGraphLabelScaler()
        scaler.fit(labels)
        scaled = scaler.transform(labels)

    Example (composition mode):

    .. code-block:: python

        from kgcnn_torch.data.transform import QMGraphLabelScaler, ExtensiveMolecularScaler, StandardScaler
        scaler = QMGraphLabelScaler(scaler=[ExtensiveMolecularScaler(), StandardScaler()])
        scaler.fit(y=data, atomic_number=mol_num)
    """

    def __init__(self, scaler: Optional[list] = None,
                 targets: Optional[List[str]] = None):
        """Initialize QMGraphLabelScaler.

        Args:
            scaler: Optional list of scaler objects (one per target) for
                composition mode. If None, uses simple per-column StandardScaler.
            targets: Optional list of target property names for reference.
        """
        self.targets = targets
        self.mean_ = None
        self.scale_ = None
        self._is_fitted = False

        if scaler is not None:
            if not isinstance(scaler, list):
                raise TypeError(
                    "Scaler for QMGraphLabelScaler must be list, got '%s'." % type(scaler).__name__)
            self._scaler_list = list(scaler)
            self._composed = True
        else:
            self._scaler_list = None
            self._composed = False

    def fit(self, y: np.ndarray = None, atomic_number=None,
            sample_weight: Optional[np.ndarray] = None, **kwargs):
        """Fit the scaler to the label data.

        Args:
            y: Label array of shape (N, n_targets).
            atomic_number: List of arrays of atomic numbers (for composition mode
                with ExtensiveMolecularScaler sub-scalers).
            sample_weight: Not used (kept for API compatibility).

        Returns:
            self
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]

        if self._composed:
            if len(self._scaler_list) != y.shape[1]:
                raise ValueError(
                    "QMGraphLabelScaler has %d scalers but got %d targets." %
                    (len(self._scaler_list), y.shape[1]))
            for i, scaler in enumerate(self._scaler_list):
                col = y[:, i:i + 1]  # Shape (N, 1)
                scaler.fit(col, atomic_number=atomic_number,
                           sample_weight=sample_weight)
            # Build composite scale_
            self.scale_ = np.concatenate(
                [np.atleast_1d(s.scale_).ravel()[:1] for s in self._scaler_list])
            self.mean_ = None  # Not meaningful for mixed scalers
        else:
            self.mean_ = np.mean(y, axis=0)
            self.scale_ = np.std(y, axis=0)
            self.scale_[self.scale_ < 1e-7] = 1.0

        self._is_fitted = True
        return self

    def transform(self, y: np.ndarray = None, atomic_number=None, **kwargs) -> np.ndarray:
        """Transform labels.

        Args:
            y: Label array of shape (N, n_targets).
            atomic_number: List of arrays of atomic numbers (composition mode).

        Returns:
            Scaled labels.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]

        if self._composed:
            out_cols = []
            for i, scaler in enumerate(self._scaler_list):
                col = y[:, i:i + 1]
                out_cols.append(scaler.transform(col, atomic_number=atomic_number))
            return np.concatenate(out_cols, axis=-1)
        else:
            return (y - self.mean_) / self.scale_

    def inverse_transform(self, y: np.ndarray = None, atomic_number=None,
                          **kwargs) -> np.ndarray:
        """Inverse-transform labels to original scale.

        Args:
            y: Scaled label array of shape (N, n_targets).
            atomic_number: List of arrays of atomic numbers (composition mode).

        Returns:
            Labels in original scale.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]

        if self._composed:
            out_cols = []
            for i, scaler in enumerate(self._scaler_list):
                col = y[:, i:i + 1]
                out_cols.append(scaler.inverse_transform(
                    col, atomic_number=atomic_number))
            return np.concatenate(out_cols, axis=-1)
        else:
            return y * self.scale_ + self.mean_

    def fit_transform(self, y: np.ndarray = None, atomic_number=None,
                      **kwargs) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y, atomic_number=atomic_number)
        return self.transform(y, atomic_number=atomic_number)

    def get_scaling(self):
        """Get scale of shape (1, n_properties).

        Returns:
            np.ndarray or None: Scale array expanded with leading axis,
                or None if scaler has not been fitted.
        """
        if self.scale_ is None:
            return None
        return np.expand_dims(self.scale_, axis=0)

    def get_weights(self) -> dict:
        if self._composed:
            return {"scaler": [s.get_weights() for s in self._scaler_list]}
        weights = {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}
        if self.targets is not None:
            weights["targets"] = self.targets
        return weights

    def set_weights(self, weights: dict):
        if self._composed:
            for i, w in enumerate(weights["scaler"]):
                self._scaler_list[i].set_weights(w)
        else:
            self.mean_ = np.array(weights["mean"])
            self.scale_ = np.array(weights["scale"])
            if "targets" in weights:
                self.targets = weights["targets"]
        self._is_fitted = True

    def save(self, filepath: str):
        """Save scaler to JSON file."""
        if self._composed:
            data = {
                "class_name": type(self).__name__,
                "config": self.get_config(),
                "weights": self.get_weights()
            }
        else:
            data = self.get_weights()
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load scaler from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        if "weights" in data and "config" in data:
            # Composed format
            self.set_config(data["config"])
            self.set_weights(data["weights"])
        else:
            # Simple format
            self.set_weights(data)

    def get_config(self) -> dict:
        if self._composed:
            return {
                "scaler": [
                    {"class_name": type(s).__name__,
                     "config": s.get_config()} for s in self._scaler_list
                ],
                "targets": self.targets,
            }
        return {"targets": self.targets}

    def set_config(self, config: dict):
        self.targets = config.get("targets", None)
        if "scaler" in config and self._composed:
            for i, sc in enumerate(config["scaler"]):
                self._scaler_list[i].set_config(sc["config"])
