.. _installation:
   :maxdepth: 3

Installation
============

Requirements
------------

kgcnn-torch requires **PyTorch** and **PyTorch Geometric (PyG)** to be installed.
Please install them first following the official instructions:

* PyTorch: https://pytorch.org/get-started/locally/
* PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Install from source
-------------------

Clone the repository and install in editable mode::

   git clone https://github.com/aimat-lab/kgcnn-torch.git
   cd kgcnn-torch
   pip install -e .

To install with all optional dependencies::

   pip install -e ".[all]"

Optional dependencies include RDKit, pymatgen, ASE, and other packages used for
molecular and crystal graph construction.

GPU Setup
---------

For GPU acceleration, ensure that the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit-archive>`_
is installed and that your PyTorch installation is built with CUDA support.
You can verify CUDA availability in PyTorch with::

   import torch
   print(torch.cuda.is_available())

Make sure the installed CUDA version matches the version PyTorch was compiled against.
See the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ for compatible version combinations.
