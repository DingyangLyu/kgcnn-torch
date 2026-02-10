"""Serialization helpers for saving and loading model configurations and objects."""
import importlib
import logging
from typing import Any


def serialize(obj) -> dict:
    """General serialization scheme for objects. Requires module information.

    Serializes an object by storing its module name, class name, and optionally
    its config, weights, and methods.

    Args:
        obj: Object to serialize. Should have ``get_config()`` and optionally
            ``get_weights()`` and ``get_methods()`` methods.

    Returns:
        dict: Serialized representation with keys: 'module_name', 'class_name',
            and optionally 'config', 'weights', 'methods'.
    """
    obj_dict = {}
    obj_dict["module_name"] = type(obj).__module__
    obj_dict["class_name"] = type(obj).__name__
    if hasattr(obj, "get_config"):
        obj_dict["config"] = obj.get_config()
    if hasattr(obj, "get_weights"):
        obj_dict["weights"] = obj.get_weights()
    if hasattr(obj, "get_methods"):
        obj_dict["methods"] = obj.get_methods()
    return obj_dict


def deserialize(obj_dict: dict) -> Any:
    """General deserialization scheme for objects. Requires module information.

    Reconstructs an object from its serialized dictionary representation.

    Args:
        obj_dict (dict): Serialized object dictionary with at least
            'module_name' and 'class_name' keys.

    Returns:
        Any: The reconstructed object.

    Raises:
        NotImplementedError: If the module or class cannot be found.
    """
    class_name = obj_dict["class_name"]
    module_name = obj_dict["module_name"]
    try:
        obj_class = getattr(importlib.import_module(str(module_name)), str(class_name))
        config = obj_dict.get("config", {})
        obj = obj_class(**config)
    except ModuleNotFoundError:
        raise NotImplementedError(
            "Unknown identifier '%s', which is not in modules in kgcnn_torch." % class_name)

    if hasattr(obj, "set_weights") and "weights" in obj_dict:
        obj.set_weights(obj_dict["weights"])

    # Call class methods if methods are in obj_dict. Order is important.
    if "methods" in obj_dict:
        method_list = obj_dict["methods"]
        if hasattr(obj, "set_methods"):
            obj.set_methods(method_list)
        else:
            for method_item in method_list:
                for method, kwargs in method_item.items():
                    if hasattr(obj, method):
                        getattr(obj, method)(**kwargs)
                    else:
                        logging.error(
                            "Class for deserialization does not have method '%s'." % method)
    return obj
