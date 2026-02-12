.. kgcnn-torch documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

KGCNN-Torch Documentation
===================================

The package `kgcnn-torch <https://github.com/aimat-lab/kgcnn-torch>`__ is a PyTorch implementation of KGCNN
for building graph neural networks on molecules and materials.
It provides layers, models, and data utilities built on top of
`PyTorch <https://pytorch.org/>`__ and `PyTorch Geometric (PyG) <https://pytorch-geometric.readthedocs.io/en/latest/>`__ .

Focus of kgcnn-torch is (batched) graph learning for molecules **kgcnn_torch.molecule** and materials **kgcnn_torch.crystal**.
Below you can find explanations and information on how to use kgcnn-torch.
See `Reference` under `Package Content` for code documentation.

.. toctree::
   :maxdepth: 3
   :caption: General:

   intro
   installation
   data.ipynb
   models.ipynb
   layers.ipynb
   literature.ipynb
   molecules.ipynb
   forces.ipynb


Package Content
===================================

.. toctree::
   :maxdepth: 3
   :caption: Reference:

   kgcnn_torch

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
