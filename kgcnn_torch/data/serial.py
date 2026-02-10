"""Dataset deserialization from configuration dictionaries.

Mirrors Keras ``kgcnn.data.serial`` — resolves dataset classes by name with
a global registry and a module-name fallback convention, and optionally
calls post-instantiation methods (``prepare_data``, ``read_in_memory``, etc.).

Usage::

    from kgcnn_torch.data.serial import deserialize

    # From string (resolves via kgcnn_torch.data.datasets.<name>)
    ds = deserialize("QM7Dataset")

    # From dict (with optional config and methods)
    ds = deserialize({
        "class_name": "MD17Dataset",
        "config": {"trajectory_name": "aspirin_dft"},
    })
"""
import importlib
import logging
from typing import Union

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

# Maps well-known base dataset categories to their modules.
# Datasets not listed here are resolved via the fallback convention
# ``kgcnn_torch.data.datasets.<class_name>``.
global_dataset_register = {
    "KgcnnGraphDataset": {
        "class_name": "KgcnnGraphDataset",
        "module_name": "kgcnn_torch.data.datasets._base",
    },
    "MoleculeNetDataset": {
        "class_name": "MoleculeNetDataset",
        "module_name": "kgcnn_torch.data.moleculenet",
    },
    "QMDataset": {
        "class_name": "QMDataset",
        "module_name": "kgcnn_torch.data.qm",
    },
    "GraphTUDataset": {
        "class_name": "GraphTUDataset",
        "module_name": "kgcnn_torch.data.tudataset",
    },
    "CrystalDataset": {
        "class_name": "CrystalDataset",
        "module_name": "kgcnn_torch.data.crystal",
    },
}


def deserialize(dataset: Union[str, dict]):
    """Deserialize a dataset class from a string name or configuration dict.

    Supports three forms:

    1. **String**: Treated as a class name. Looked up in ``global_dataset_register``
       first, then falls back to ``kgcnn_torch.data.datasets.<name>``.
    2. **Dict with class_name only**: Same resolution logic, instantiated with
       ``config`` kwargs (empty dict if not provided).
    3. **Dict with class_name + module_name**: Directly imports from the given module.

    After instantiation, any entries in ``"methods"`` are called in order::

        {"methods": [{"read_in_memory": {}}, {"map_list": {"method_name": "set_range"}}]}

    Args:
        dataset: Dataset class name (str) or serialization dict with at least
            ``class_name``.

    Returns:
        Dataset instance.

    Raises:
        NotImplementedError: If the module or class cannot be found.
    """
    if not isinstance(dataset, (dict, str)):
        module_logger.warning("Cannot deserialize dataset %s." % dataset)
        return dataset

    if isinstance(dataset, str):
        dataset = {"class_name": dataset, "config": {}}

    class_name = dataset["class_name"]

    # Resolve module name
    if class_name in global_dataset_register:
        resolved_name = global_dataset_register[class_name]["class_name"]
        module_name = global_dataset_register[class_name]["module_name"]
    else:
        resolved_name = class_name
        module_name = dataset.get(
            "module_name", "kgcnn_torch.data.datasets.%s" % resolved_name)

    try:
        ds_class = getattr(importlib.import_module(str(module_name)), str(resolved_name))
        config = dataset.get("config", {})
        ds_instance = ds_class(**config)
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            "Unknown dataset identifier '%s', which is not in "
            "kgcnn_torch.data.datasets." % resolved_name)

    # Call post-instantiation methods in order.
    if "methods" in dataset:
        method_list = dataset["methods"]
        for method_item in method_list:
            for method, kwargs in method_item.items():
                if hasattr(ds_instance, method):
                    getattr(ds_instance, method)(**kwargs)
                else:
                    module_logger.error(
                        "Dataset class does not have method '%s'." % method)

    return ds_instance
