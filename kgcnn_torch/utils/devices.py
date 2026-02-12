"""Device detection and management utilities for PyTorch."""
import torch
from typing import Union


def check_device() -> dict:
    """Check for available compute devices (GPUs).

    Returns:
        dict: Dictionary with device information including:
            - cuda_available: Whether CUDA is available.
            - device_name: List of GPU device names.
            - device_id: List of GPU device IDs.
            - device_memory: Memory info per device (allocated/cached in GB).
    """
    cuda_is_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()

    physical_device_name = [
        torch.cuda.get_device_name(i) for i in range(device_count)
    ]
    logical_device_list = list(range(device_count))
    memory_info = [
        {
            "allocated": round(torch.cuda.memory_allocated(i) / 1024**3, 1),
            "cached": round(torch.cuda.memory_reserved(i) / 1024**3, 1),
        }
        for i in logical_device_list
    ]

    out_info = {
        "cuda_available": str(cuda_is_available),
        "device_name": str(physical_device_name),
        "device_id": str(logical_device_list),
        "device_memory": str(memory_info),
    }
    return out_info


def set_cuda_device(device_id: Union[int, list]):
    """Set the CUDA device by ID.

    It is recommended to use environment variables instead:

    .. code-block:: python

        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    Args:
        device_id (int or list): ID(s) of the GPU(s) to use.
    """
    if isinstance(device_id, int):
        torch.cuda.set_device(device_id)
    elif isinstance(device_id, list):
        # Set the first device as default; multiple devices
        # typically handled by DataParallel or DistributedDataParallel.
        torch.cuda.set_device(device_id[0])
    else:
        raise TypeError("device_id must be int or list of int, got %s" % type(device_id).__name__)


def get_device(device: str = "auto") -> torch.device:
    """Get a torch.device, with 'auto' selecting CUDA if available.

    Args:
        device (str): Device string: 'auto', 'cpu', 'cuda', 'cuda:0', etc.

    Returns:
        torch.device: The resolved device.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """Get GPU memory information for a specific device.

    Args:
        device_id (int): GPU device index. Default is 0.

    Returns:
        dict: Dictionary with memory info in bytes:
            - total: Total GPU memory.
            - allocated: Currently allocated memory.
            - cached: Currently cached (reserved) memory.
            - free: Free memory (total - allocated).
    """
    if not torch.cuda.is_available():
        return {"total": 0, "allocated": 0, "cached": 0, "free": 0}

    total = torch.cuda.get_device_properties(device_id).total_memory
    allocated = torch.cuda.memory_allocated(device_id)
    cached = torch.cuda.memory_reserved(device_id)
    free = total - allocated

    return {
        "total": total,
        "allocated": allocated,
        "cached": cached,
        "free": free,
    }
