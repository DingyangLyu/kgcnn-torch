.. _intro:
   :maxdepth: 3

Introduction
============


The package `kgcnn-torch <https://github.com/aimat-lab/kgcnn-torch>`__ is a PyTorch (PyG) implementation of KGCNN,
providing several layer classes and model architectures to build graph convolution models for molecules and materials.
It is built on top of `PyTorch <https://pytorch.org/>`__ and `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`__ .

This is a **PyTorch-only** package. Unlike the Keras-based multi-backend kgcnn, kgcnn-torch uses PyTorch and PyG natively
for all graph operations, leveraging PyG's efficient sparse tensor and message passing infrastructure.

Graph Representation
--------------------

kgcnn-torch uses the **disjoint graph format** from `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`__ .
Graphs are represented as PyG ``Data`` objects, and batches of graphs are handled by PyG's ``Batch`` class,
which merges multiple small graphs into a single large disjoint graph.

**Disjoint Graph**

* ``x``: Node attributes of shape ``([N], F)`` and dtype *float*
* ``edge_attr``: Edge attributes of shape ``([M], F)`` and dtype *float*
* ``edge_index``: Edge indices of shape ``(2, [M])`` and dtype *long*
* ``batch``: Graph assignment vector of shape ``([N], )`` and dtype *long*

Where ``[N]`` is the total number of nodes across all graphs in the batch, and ``[M]`` is the total number of edges.

The ``edge_index`` convention follows PyG: ``edge_index[0]`` contains the **source** (sender) node indices and
``edge_index[1]`` contains the **target** (receiver) node indices for each edge.

Models and Layers
-----------------

kgcnn-torch includes **27 published GNN model architectures** from the literature, including:
GCN, GAT, GATv2, GIN, rGIN, GraphSAGE, DMPNN, CMPNN, DGIN, EGNN, INorp, GNNFilm,
SchNet, PAiNN, DimeNetPP, MEGNet, CGCNN, HDNNP2nd, MXMNet, AttentiveFP, MoGAT,
HamNet, MAT, MEGAN, RGCN, NMPN, and GNNExplain.

All models are composed from reusable layer building blocks in ``kgcnn_torch.layers``, including:

* ``aggr`` -- Aggregation operations (sum, mean, max, softmax, set2set)
* ``conv`` -- Graph convolution implementations
* ``gather`` -- Gather node features along edges
* ``geom`` -- Geometric operations (distances, angles, spherical harmonics)
* ``pooling`` -- Graph-level readout pooling
* ``mlp`` -- MLP variants
* ``attention`` -- Attention mechanisms
* ``norm`` -- Graph-aware normalization layers
* ``message`` -- Message passing base layers

Data Pipeline
-------------

The data pipeline in ``kgcnn_torch.data`` provides utilities for loading and preprocessing graph datasets.
It supports molecular datasets (via SMILES/SDF with RDKit or OpenBabel), crystal datasets (via pymatgen),
force field datasets, and standard graph benchmark datasets (TUDatasets, MoleculeNet, QM datasets).

Each dataset class handles automatic downloading, caching, and conversion to PyG-compatible graph objects.
