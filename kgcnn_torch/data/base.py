"""Base data structures for kgcnn-torch data pipeline.

Provides MemoryGraphList (list of GraphDict), MemoryGraphDataset (persistent dataset with file management),
and DownloadDataset (download + extract) ported from the Keras KGCNN package, plus to_pyg_list() for
converting to PyG Data objects.
"""
import logging
import os
import pickle
import numpy as np
import pandas as pd
from typing import Union, List, Callable, Optional
from copy import copy

from kgcnn_torch.graph.base import GraphDict

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Utility functions (ported from kgcnn.data.utils)
# ---------------------------------------------------------------------------

def save_pickle_file(obj, file_path: str, **kwargs):
    """Save object to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, **kwargs)


def load_pickle_file(file_path: str, **kwargs):
    """Load object from pickle file."""
    with open(file_path, 'rb') as f:
        obj = pickle.load(f, **kwargs)
    return obj


def save_json_file(obj, file_path: str, **kwargs):
    """Save object to JSON file."""
    import json
    with open(file_path, 'w') as json_file:
        json.dump(obj, json_file, **kwargs)


def load_json_file(file_path: str, **kwargs):
    """Load object from JSON file."""
    import json
    with open(file_path, 'r') as json_file:
        file_read = json.load(json_file, **kwargs)
    return file_read


def pad_np_array_list_batch_dim(values: list, dtype: str = None):
    r"""Pad a list of numpy arrays along first dimension.

    Args:
        values (list): List of :obj:`np.ndarray` .
        dtype (str): Data type of values tensor. Defaults to None.

    Returns:
        tuple: Padded and mask :obj:`np.ndarray` of values.
    """
    max_shape = np.amax([x.shape for x in values], axis=0)
    final_shape = np.concatenate([np.array([len(values)], dtype="int64"), np.array(max_shape, dtype="int64")])
    padded = np.zeros(final_shape, dtype=values[0].dtype)
    mask = np.zeros(final_shape, dtype="bool")
    for i, x in enumerate(values):
        index = [i] + [slice(0, int(j)) for j in x.shape]
        padded[tuple(index)] = x
        mask[tuple(index)] = True
    if dtype is not None:
        padded = padded.astype(dtype=dtype)
    return padded, mask


# ---------------------------------------------------------------------------
# MemoryGraphList
# ---------------------------------------------------------------------------

class MemoryGraphList(list):
    r"""List of :obj:`GraphDict` objects stored in memory.

    Inherits from a python list. The graph properties are defined by tensor-like (numpy) arrays
    in :obj:`GraphDict`, which are the items of the list.

    Provides assign/obtain property, map_list, clean, and to_pyg_list() for PyG conversion.
    """

    _require_validate = True

    def __init__(self, iterable: list = None):
        iterable = iterable if iterable is not None else []
        super(MemoryGraphList, self).__init__(iterable)
        self.logger = module_logger
        self.validate()

    def validate(self):
        for i, x in enumerate(self):
            if not isinstance(x, GraphDict):
                self[i] = GraphDict(x)

    def assign_property(self, key: str, value: list):
        """Assign a list of values to the respective :obj:`GraphDict` items.

        Args:
            key (str): Name of the property.
            value (list): List of values for property `key`.

        Returns:
            self
        """
        if value is None:
            return self
        if not isinstance(value, list):
            raise TypeError("Expected type `list` to assign graph properties.")
        if len(self) == 0:
            self.empty(len(value))
        if len(self) != len(value):
            raise ValueError("Can only store graph attributes from list with same length.")
        for i, x in enumerate(value):
            self[i].assign_property(key, x)
        return self

    def obtain_property(self, key: str) -> Union[List, None]:
        """Returns a list with the values of all graphs for the given property name.
        If none of the graphs have this property, returns None.

        Args:
            key (str): The string name of the property.
        """
        prop_list = [x.obtain_property(key) for x in self]
        if all([x is None for x in prop_list]):
            self.logger.warning("Property '%s' is not set on any graph." % key)
            return None
        return prop_list

    def __getitem__(self, item) -> Union[GraphDict, "MemoryGraphList"]:
        if isinstance(item, int):
            return super(MemoryGraphList, self).__getitem__(item)
        if isinstance(item, slice):
            return MemoryGraphList(super(MemoryGraphList, self).__getitem__(item))
        if isinstance(item, (list, tuple)):
            return MemoryGraphList([super(MemoryGraphList, self).__getitem__(int(i)) for i in item])
        if isinstance(item, np.ndarray):
            return MemoryGraphList([super(MemoryGraphList, self).__getitem__(int(i)) for i in item])
        if isinstance(item, (np.uint8, np.int32, np.int64)):
            return super(MemoryGraphList, self).__getitem__(int(item))
        raise TypeError("Unsupported type '%s' for `MemoryGraphList` items." % type(item))

    def __setitem__(self, key, value):
        if not isinstance(value, GraphDict):
            raise TypeError("Require a `GraphDict` as list item.")
        super(MemoryGraphList, self).__setitem__(key, value)

    def __repr__(self):
        return "<{} [{}]>".format(type(self).__name__, "" if len(self) == 0 else self[0].__repr__() + " ...")

    def append(self, graph):
        assert isinstance(graph, GraphDict), "Must append `GraphDict` to self."
        super(MemoryGraphList, self).append(graph)

    def insert(self, index: int, value) -> None:
        assert isinstance(value, GraphDict), "Must insert `GraphDict` to self."
        super(MemoryGraphList, self).insert(index, value)

    def __add__(self, other):
        assert isinstance(other, MemoryGraphList), "Must add `MemoryGraphList` to self."
        return MemoryGraphList(super(MemoryGraphList, self).__add__(other))

    def copy(self, deep_copy: bool = False):
        """Copy data in the list."""
        if not deep_copy:
            return MemoryGraphList([GraphDict(x) for x in self])
        else:
            return MemoryGraphList([x.copy() for x in self])

    def empty(self, length: int):
        """Create an empty list in place. Overwrites existing list.

        Args:
            length (int): Length of the empty list.

        Returns:
            self
        """
        if length is None:
            return self
        if length < 0:
            raise ValueError("Length of empty list must be >=0.")
        self.clear()
        for _ in range(length):
            super(MemoryGraphList, self).append(GraphDict())
        return self

    def update(self, other) -> None:
        assert isinstance(other, MemoryGraphList), "Must update `MemoryGraphList`."
        assert len(other) == len(self), "Length of list for update must match."
        for i in range(len(self)):
            self[i].update(other[i])

    @property
    def length(self):
        """Length of list."""
        return len(self)

    @length.setter
    def length(self, value: int):
        raise ValueError("Can not set length. Please use 'empty()' to initialize an empty list.")

    def _to_tensor(self, item: dict, make_copy=True):
        if not make_copy:
            self.logger.warning("At the moment always a copy is made for tensor().")
        props: list = self.obtain_property(item["name"])
        is_ragged = item.get("ragged", False)
        dtype = item.get("dtype", None)
        if is_ragged:
            # For PyTorch we just concatenate + return row lengths (no TF ragged tensor)
            values = np.concatenate(props, axis=0)
            if dtype is not None:
                values = values.astype(dtype)
            row_lengths = np.array([len(x) for x in props], dtype="int64")
            return values, row_lengths
        else:
            return pad_np_array_list_batch_dim(props, dtype=dtype)[0]

    def tensor(self, items: Union[list, dict], make_copy=True):
        r"""Make tensor objects from multiple graph properties in list.

        Args:
            items (list): List of dicts specifying graph properties via 'name' key.
                Required dict-keys should be 'name' and optionally 'ragged', 'shape', 'dtype'.
            make_copy (bool): Whether to copy the data. Default is True.

        Returns:
            list: List of arrays/tensors.
        """
        if isinstance(items, dict):
            if all([isinstance(value, dict) for value in items.values() if value is not None]) and "name" not in items:
                return {key: self._to_tensor(value, make_copy=make_copy) for key, value in items.items() if
                        value is not None}
            return self._to_tensor(items, make_copy=make_copy)
        elif isinstance(items, (tuple, list)):
            return [self._to_tensor(x, make_copy=make_copy) for x in items]
        else:
            raise TypeError("Wrong type, expected e.g. [{'name': 'edge_indices', 'ragged': True}, {...}, ...]")

    def map_list(self, method: Union[str, Callable], **kwargs):
        r"""Map a method over this list and apply on each :obj:`GraphDict`.

        Args:
            method (str, Callable): Name of the :obj:`GraphDict` method, or a callable.
            kwargs: Kwargs for `method`.

        Returns:
            self
        """
        if isinstance(method, str):
            for i, x in enumerate(self):
                if hasattr(x, method):
                    getattr(x, method)(**kwargs)
                else:
                    x.apply_preprocessor(name=method, **kwargs)
        elif isinstance(method, dict):
            raise NotImplementedError("Serialization for method in `map_list` is not yet supported")
        else:
            for i, x in enumerate(self):
                method(x, **kwargs)
        return self

    def clean(self, inputs: Union[list, str]):
        r"""Remove graphs that are missing required properties.

        Args:
            inputs (list, str): Property name(s) that must exist on each graph. Can also be a list of dicts
                with 'name' key (model input config style).

        Returns:
            np.ndarray: Array of indices that were removed.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        invalid_graphs = []
        for item in inputs:
            if isinstance(item, dict):
                item_name = item["name"]
            else:
                item_name = item
            props = self.obtain_property(item_name)
            if props is None:
                self.logger.warning("Can not clean property '%s' as it was not assigned to any graph." % item)
                continue
            for i, x in enumerate(props):
                if x is None or not hasattr(x, "__getitem__"):
                    self.logger.info("Property '%s' is not defined for graph '%s'." % (item_name, i))
                    invalid_graphs.append(i)
                elif not isinstance(x, np.ndarray):
                    self.logger.info("Property '%s' is not a numpy array for graph '%s'." % (item_name, i))
                    invalid_graphs.append(i)
                elif len(x.shape) > 0:
                    if len(x) <= 0:
                        self.logger.info("Property '%s' is an empty list for graph '%s'." % (item_name, i))
                        invalid_graphs.append(i)
        invalid_graphs = np.unique(np.array(invalid_graphs, dtype="int"))
        invalid_graphs = np.flip(invalid_graphs)
        if len(invalid_graphs) > 0:
            self.logger.warning("Found invalid graphs for properties. Removing graphs '%s'." % invalid_graphs)
        else:
            self.logger.info("No invalid graphs for assigned properties found.")
        for i in invalid_graphs:
            self.pop(int(i))
        return invalid_graphs

    def rename_property_on_graphs(self, old_property_name: str, new_property_name: str) -> list:
        """Change the name of a graph property on all graphs in the list.

        Args:
            old_property_name (str): Old property name.
            new_property_name (str): New property name.

        Returns:
            list: List indices of replaced property names.
        """
        replaced_list = []
        for i, x in enumerate(self):
            if old_property_name in x:
                self[i][new_property_name] = x[old_property_name]
                self[i].pop(old_property_name)
                replaced_list.append(i)
        return replaced_list

    # Alias
    set = assign_property
    get = obtain_property

    def to_pyg_list(self,
                    node_key: str = "node_number",
                    pos_key: str = "node_coordinates",
                    edge_key: str = "edge_indices",
                    edge_attr_keys: list = None,
                    label_key: str = "graph_labels",
                    lattice_key: str = "graph_lattice",
                    image_key: str = "range_image",
                    graph_size_key: str = "graph_size") -> list:
        """Convert MemoryGraphList to list of PyG Data objects.

        IMPORTANT: Swaps edge index convention from KGCNN (target, source)
        to PyG (source, target).

        Args:
            node_key: Property name for node features.
            pos_key: Property name for node positions.
            edge_key: Property name for edge indices.
            edge_attr_keys: List of property names for edge features.
            label_key: Property name for graph labels.
            lattice_key: Property name for lattice matrix (crystals).
            image_key: Property name for periodic image (crystals).
            graph_size_key: Property name for graph size.

        Returns:
            List of torch_geometric.data.Data objects.
        """
        import torch
        from torch_geometric.data import Data

        pyg_list = []
        for g in self:
            data_dict = {}

            # Node features
            nodes = g.obtain_property(node_key)
            if nodes is not None:
                nodes = np.asarray(nodes)
                if nodes.dtype.kind in ('U', 'S', 'O'):
                    pass
                else:
                    data_dict['z'] = torch.tensor(nodes, dtype=torch.long if nodes.ndim == 1 else torch.float)

            # Positions
            pos = g.obtain_property(pos_key)
            if pos is not None:
                data_dict['pos'] = torch.tensor(np.asarray(pos), dtype=torch.float)

            # Edge indices: swap KGCNN convention [target, source] -> PyG [source, target]
            edges = g.obtain_property(edge_key)
            if edges is not None:
                edges = np.asarray(edges)
                if edges.ndim == 2 and edges.shape[1] == 2:
                    ei = edges[:, [1, 0]].T
                elif edges.ndim == 2 and edges.shape[0] == 2:
                    ei = edges[[1, 0]]
                else:
                    ei = edges
                data_dict['edge_index'] = torch.tensor(ei, dtype=torch.long)

            # Edge attributes
            if edge_attr_keys:
                edge_attr_parts = []
                for attr_key in edge_attr_keys:
                    val = g.obtain_property(attr_key)
                    if val is not None:
                        t = torch.tensor(np.asarray(val), dtype=torch.float)
                        if t.dim() == 1:
                            t = t.unsqueeze(-1)
                        edge_attr_parts.append(t)
                if edge_attr_parts:
                    data_dict['edge_attr'] = torch.cat(edge_attr_parts, dim=-1)

            # Edge weight
            edge_weight = g.obtain_property("edge_weight")
            if edge_weight is not None:
                data_dict['edge_weight'] = torch.tensor(np.asarray(edge_weight), dtype=torch.float)

            # Labels
            labels = g.obtain_property(label_key)
            if labels is not None:
                labels = np.atleast_1d(np.asarray(labels))
                if labels.dtype.kind == 'i':
                    data_dict['y'] = torch.tensor(labels, dtype=torch.long)
                else:
                    data_dict['y'] = torch.tensor(labels, dtype=torch.float)

            # Force labels
            force = g.obtain_property("force")
            if force is None:
                force = g.obtain_property("node_gradient")
            if force is not None:
                data_dict['force'] = torch.tensor(np.asarray(force), dtype=torch.float)

            # Stress labels
            stress = g.obtain_property("stress")
            if stress is None:
                stress = g.obtain_property("graph_stress")
            if stress is not None:
                data_dict['stress'] = torch.tensor(np.asarray(stress), dtype=torch.float)

            # Total energy
            energy = g.obtain_property("energy")
            if energy is None:
                energy = g.obtain_property("graph_energy")
            if energy is not None:
                data_dict['energy'] = torch.tensor(np.asarray(energy), dtype=torch.float)

            # Crystal-specific
            lattice = g.obtain_property(lattice_key)
            if lattice is not None:
                data_dict['lattice'] = torch.tensor(np.asarray(lattice), dtype=torch.float)

            image = g.obtain_property(image_key)
            if image is not None:
                data_dict['edge_image'] = torch.tensor(np.asarray(image), dtype=torch.float)

            # Graph size
            gsize = g.obtain_property(graph_size_key)
            if gsize is not None:
                data_dict['graph_size'] = torch.tensor(int(gsize), dtype=torch.long)

            # Node attributes
            node_attr = g.obtain_property("node_attributes")
            if node_attr is None:
                node_num = g.obtain_property("node_number")
                if node_num is not None:
                    node_attr = np.expand_dims(np.asarray(node_num, dtype=np.float32), axis=-1)
            if node_attr is not None:
                data_dict['x'] = torch.tensor(np.asarray(node_attr), dtype=torch.float)

            # Edge number / edge type
            edge_type = g.obtain_property("edge_number")
            if edge_type is not None:
                et = np.asarray(edge_type)
                data_dict['edge_type'] = torch.tensor(et, dtype=torch.long)

            # Edge pair index (for DMPNN, CMPNN)
            edge_pair = g.obtain_property("edge_pair_index")
            if edge_pair is not None:
                data_dict['edge_pair_index'] = torch.tensor(np.asarray(edge_pair), dtype=torch.long)

            # Range indices
            range_idx = g.obtain_property("range_indices")
            if range_idx is not None and edge_key != "range_indices":
                ri = np.asarray(range_idx)
                if ri.ndim == 2 and ri.shape[1] == 2:
                    ri = ri[:, [1, 0]].T
                data_dict['range_edge_index'] = torch.tensor(ri, dtype=torch.long)

            # Angle/triplet indices
            angle_idx = g.obtain_property("angle_indices")
            if angle_idx is not None:
                ai = np.asarray(angle_idx)
                if ai.ndim == 2 and ai.shape[1] == 2:
                    ai = ai.T
                data_dict['angle_index'] = torch.tensor(ai, dtype=torch.long)

            # Graph-level state (for MEGNet)
            graph_attr = g.obtain_property("graph_attributes")
            if graph_attr is not None:
                data_dict['state'] = torch.tensor(np.asarray(graph_attr), dtype=torch.float)

            # Crystal: symmetry operations
            sym_ops = g.obtain_property("symmetry_ops")
            if sym_ops is not None:
                data_dict['symmetry_ops'] = torch.tensor(np.asarray(sym_ops), dtype=torch.float)

            # Crystal: cell translations
            cell_trans = g.obtain_property("cell_translations")
            if cell_trans is not None:
                data_dict['cell_translations'] = torch.tensor(np.asarray(cell_trans), dtype=torch.float)

            # Set num_nodes explicitly
            if 'z' in data_dict:
                data_dict['num_nodes'] = len(data_dict['z'])
            elif 'pos' in data_dict:
                data_dict['num_nodes'] = len(data_dict['pos'])
            elif 'x' in data_dict:
                data_dict['num_nodes'] = len(data_dict['x'])

            pyg_data = Data(**data_dict)
            pyg_list.append(pyg_data)

        return pyg_list

    def save(self, filepath: str):
        """Save graph list to pickle file."""
        save_pickle_file([x.to_dict() for x in self], filepath)

    def load(self, filepath: str):
        """Load graph list from pickle file."""
        in_list = load_pickle_file(filepath)
        self.clear()
        for x in in_list:
            super(MemoryGraphList, self).append(GraphDict(x))
        return self


# ---------------------------------------------------------------------------
# MemoryGraphDataset
# ---------------------------------------------------------------------------

class MemoryGraphDataset(MemoryGraphList):
    r"""Dataset class for lists of graph tensor dictionaries stored on file and fit into memory.

    Inherits from :obj:`MemoryGraphList` with additional information about disk location
    (data directory, file directory, file name, dataset name) and methods for loading,
    saving, and managing datasets.
    """

    fits_in_memory = True

    def __init__(self,
                 data_directory: str = None,
                 dataset_name: str = None,
                 file_name: str = None,
                 file_directory: str = None,
                 verbose: int = 10):
        super(MemoryGraphDataset, self).__init__()
        self.logger = logging.getLogger("kgcnn.data." + dataset_name) if dataset_name is not None else module_logger
        self.logger.setLevel(verbose)
        self.data_directory = data_directory
        self.file_name = file_name
        self.file_directory = file_directory
        self.dataset_name = dataset_name
        self.data_frame = None
        self.data_keys = None
        self.data_unit = None
        self.label_names = None
        self.label_units = None

    def _verify_data_directory(self) -> Union[str, None]:
        if self.data_directory is None:
            self.warning("Data directory is not set.")
            return None
        if not os.path.exists(os.path.realpath(self.data_directory)):
            self.error("Data directory does not exist.")
        return self.data_directory

    @property
    def file_path(self):
        """Construct filepath from 'file_name'."""
        self._verify_data_directory()
        if self.file_name is None:
            self.warning("Can not determine file path, missing `file_name`.")
            return None
        return os.path.join(self.data_directory, self.file_name)

    @property
    def file_directory_path(self):
        """Construct file-directory path from 'data_directory' and 'file_directory'."""
        self._verify_data_directory()
        if self.file_directory is None:
            self.warning("Can not determine file directory, missing `file_directory`.")
            return None
        return os.path.join(self.data_directory, self.file_directory)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def save(self, filepath: str = None):
        """Save all graph properties to pickle file.

        Args:
            filepath (str): Full path of output file. Default saves to `dataset_name.kgcnn.pickle`.
        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, self.dataset_name + ".kgcnn.pickle")
        self.info("Pickle dataset...")
        save_pickle_file([x.to_dict() for x in self], filepath)
        return self

    def load(self, filepath: str = None):
        """Load graph properties from a pickled file.

        Args:
            filepath (str): Full path of input file. Default loads from `dataset_name.kgcnn.pickle`.
        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, self.dataset_name + ".kgcnn.pickle")
        self.info("Load pickled dataset...")
        in_list = load_pickle_file(filepath)
        self.clear()
        for x in in_list:
            super(MemoryGraphList, self).append(GraphDict(x))
        return self

    def read_in_table_file(self, file_path: str = None, **kwargs):
        """Read a data frame from file path (CSV or Excel).

        Args:
            file_path (str): File path to table file. Default is None.
            kwargs: Kwargs for pandas read_csv.

        Returns:
            self
        """
        if file_path is None:
            file_path = os.path.join(self.data_directory, self.file_name)

        file_path_base = os.path.splitext(file_path)[0]

        for file_extension in [".csv"]:
            if os.path.exists(file_path_base + file_extension):
                self.data_frame = pd.read_csv(file_path_base + file_extension, **kwargs)
                return self
        for file_extension in [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]:
            if os.path.exists(file_path_base + file_extension):
                self.data_frame = pd.read_excel(file_path_base + file_extension, **kwargs)
                return self

        self.warning("Unsupported data extension of '%s' for table file." % file_path)
        return self

    def assert_valid_model_input(self, hyper_input: Union[list, dict], raise_error_on_fail: bool = True):
        """Check whether dataset has graph properties requested by model input.

        Args:
            hyper_input (list): List of dicts with "name" and "shape" keys.
            raise_error_on_fail (bool): Whether to raise an error if assertion failed.
        """
        dataset = self

        def message_error(msg):
            if raise_error_on_fail:
                raise ValueError(msg)
            else:
                dataset.error(msg)

        def message_warning(msg):
            dataset.warning(msg)

        if isinstance(hyper_input, dict):
            if "name" in hyper_input and "shape" in hyper_input:
                hyper_input = [hyper_input]
            else:
                hyper_input = list(hyper_input.values())

        for x in hyper_input:
            if x is None:
                message_warning("Found 'None' in place of model input. Skipping this input.")
                continue
            if not isinstance(x, dict):
                message_error("Wrong type of list item. Found '%s' but must be `dict`." % type(x))

        for x in hyper_input:
            if x is None:
                continue
            if "name" not in x:
                message_error("Can not infer name from '%s' for model input." % x)
            data = [dataset[i].obtain_property(x["name"]) for i in range(len(dataset))]
            prop_in_data = [y is None for y in data]
            if all(prop_in_data):
                message_error("Property %s is not defined for any graph." % x["name"])
            if any(prop_in_data):
                message_warning("Property %s is not defined for all graphs. Please run clean()." % x["name"])

            if hasattr(data[0], "shape") and "shape" in x:
                shape_element = data[0].shape
                shape_input = x["shape"]
                if len(shape_input) != len(shape_element):
                    message_error("Mismatch in rank for model input {} vs. {}".format(shape_element, shape_input))
                for i, dim in enumerate(shape_input):
                    if dim is not None:
                        if shape_element[i] != dim:
                            message_error(
                                "Mismatch in shape for model input {} vs. {}".format(shape_element, shape_input))
            else:
                message_error("Can not check shape for '%s'." % x["name"])

    def collect_files_in_file_directory(self, file_column_name: str = None, table_file_path: str = None,
                                        read_method_file: Callable = None, update_counter: int = 1000,
                                        append_file_content: bool = True,
                                        read_method_return_list: bool = False) -> list:
        """Collect single files in :obj:`file_directory` by names in CSV table file.

        Args:
            file_column_name (str): Column name in table file that holds file names.
            table_file_path (str): Path to table file.
            read_method_file (Callable): Callable read-file method to return (processed) file content.
            update_counter (int): Loop counter to show progress. Default is 1000.
            append_file_content (bool): Whether to append or add return of read_method_file.
            read_method_return_list (bool): Whether read_method_file returns a list.

        Returns:
            list: File content loaded from single files.
        """
        self.read_in_table_file(table_file_path)
        if self.data_frame is None:
            raise FileNotFoundError("Can not find '.csv' table path '%s'." % table_file_path)

        if file_column_name is None:
            raise ValueError("Please specify column for '.csv' file which contains file names.")

        if file_column_name not in self.data_frame.columns:
            raise ValueError(
                "Can not find file names of column '%s' in '%s'" % (file_column_name, self.data_frame.columns))

        name_file_list = self.data_frame[file_column_name].values
        num_files = len(name_file_list)

        if not os.path.exists(self.file_directory_path):
            raise ValueError("No file directory found at '%s'." % self.file_directory_path)

        self.info("Read %s single files." % num_files)
        out_list = []
        for i, x in enumerate(name_file_list):
            file_loaded = read_method_file(os.path.join(self.file_directory_path, x))
            if append_file_content:
                if read_method_return_list:
                    out_list.append(file_loaded[0])
                else:
                    out_list.append(file_loaded)
            else:
                if read_method_return_list:
                    out_list += file_loaded
                else:
                    out_list += [file_loaded]
            if i % update_counter == 0:
                self.info("... Read {0} file {1} from {2}".format(os.path.splitext(x)[1], i, num_files))

        return out_list

    def set_methods(self, method_list: List[dict]) -> None:
        """Apply a list of serialized class-methods on the dataset.

        Args:
            method_list (list): List of dicts specifying class methods.
                The dict key is the method name, the value is kwargs.
        """
        for method_item in method_list:
            for method, kwargs in method_item.items():
                if hasattr(self, method):
                    getattr(self, method)(**kwargs)
                else:
                    self.error("Class does not have method '%s'." % method)

    def get_train_test_indices(self,
                               train: str = "train",
                               test: str = "test",
                               valid: Optional[str] = None,
                               split_index: Union[int, list] = None,
                               shuffle: bool = False,
                               seed: int = None
                               ) -> List[List[np.ndarray]]:
        """Get train and test indices from graph list.

        Args:
            train (str): Name of graph property for train split assignment. Default 'train'.
            test (str): Name of graph property for test split assignment. Default 'test'.
            valid (str): Name of graph property for validation assignment. Default None.
            split_index (int, list): Split index to get indices for.
            shuffle (bool): Whether to shuffle splits. Default is False.
            seed (int): Random seed for shuffle. Default is None.

        Returns:
            list: List of tuples/triples of train, test, (validation) split indices.
        """
        out_indices = []

        if split_index is None:
            train_to_check = self.obtain_property(train)
            train_to_check = [
                np.expand_dims(x, axis=0) if len(x.shape) < 1 else x for x in train_to_check if x is not None]
            split_index = list(np.sort(np.unique(np.concatenate(train_to_check, axis=0))))
        if not isinstance(split_index, (list, tuple)):
            split_index_list: List[int] = [split_index]
        else:
            split_index_list: List[int] = split_index

        for split_index in split_index_list:
            graph_index_split_list: List[np.ndarray] = []

            for property_name in [train, test, valid]:
                if property_name is None:
                    continue

                graph_index_list: List[int] = []
                split_prop: List = self.obtain_property(property_name)
                for index, split_list in enumerate(split_prop):
                    if split_list is not None:
                        if split_index in split_list:
                            graph_index_list.append(index)

                graph_index_array = np.array(graph_index_list)
                graph_index_split_list.append(graph_index_array)

            if shuffle:
                np.random.seed(seed)
                for graph_index_array in graph_index_split_list:
                    np.random.shuffle(graph_index_array)

            out_indices.append(graph_index_split_list)

        return out_indices

    def relocate(self, data_directory: str = None, file_name: str = None, file_directory: str = None):
        """Change file information. Does not copy files on disk.

        Args:
            data_directory (str): Full path to directory.
            file_name (str): Generic filename.
            file_directory (str): Name or relative path to a directory containing sorted files.

        Returns:
            self
        """
        self.data_directory = data_directory
        self.file_name = file_name
        self.file_directory = file_directory
        return self

    def set_multi_target_labels(self, graph_labels: str = "graph_labels", multi_target_indices: list = None,
                                data_unit: Union[str, list] = None):
        """Select multiple targets in labels.

        Args:
            graph_labels (str): Name of the property that holds multiple targets.
            multi_target_indices (list): List of indices of targets to select.
            data_unit (str, list): Optional list of data units.

        Returns:
            tuple: List of label names and label units.
        """
        labels = np.array(self.obtain_property(graph_labels))
        label_names = self.label_names if hasattr(self, "label_names") else None
        label_units = self.label_units if hasattr(self, "label_units") else None
        if data_unit is not None:
            label_units = data_unit
        if len(labels.shape) <= 1:
            labels = np.expand_dims(labels, axis=-1)

        if multi_target_indices is not None:
            labels = labels[:, multi_target_indices]
            if label_names is not None:
                label_names = [label_names[i] for i in multi_target_indices]
            if label_units is not None:
                label_units = [label_units[i] for i in multi_target_indices]
        self.info("Labels '%s' in '%s' have shape '%s'." % (label_names, label_units, labels.shape))
        self.assign_property(graph_labels, [x for x in labels])
        return label_names, label_units

    def set_train_test_indices_k_fold(self, n_splits: int = 5, shuffle: bool = False, random_state: int = None,
                                      train: str = "train", test: str = "test"):
        """Set train/test indices for each graph from k-fold cross-validation.

        Args:
            n_splits (int): Number of splits.
            shuffle (bool): Whether to shuffle indices.
            random_state (int): Random seed.
            train (str): Property to assign train indices to.
            test (str): Property to assign test indices to.
        """
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for x in self:
            x.set(train, [])
            x.set(test, [])
        for fold, (train_index, test_index) in enumerate(kf.split(np.expand_dims(np.arange(len(self)), axis=-1))):
            for i in train_index:
                self[i].set(train, list(self[i].get(train)) + [fold])
            for i in test_index:
                self[i].set(test, list(self[i].get(test)) + [fold])


# Alias
MemoryGeometricGraphDataset = MemoryGraphDataset


# ---------------------------------------------------------------------------
# DownloadDataset — imported from download.py (single source of truth)
# ---------------------------------------------------------------------------
from kgcnn_torch.data.download import DownloadDataset  # noqa: E402
