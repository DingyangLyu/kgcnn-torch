kgcnn\_torch.data.transform module
====================================

Data transformation and scaling utilities (numpy-based) for preprocessing
graph data before conversion to PyG Data objects.

This module provides several scaler classes:

- :class:`~kgcnn_torch.data.transform.StandardScaler` -- Standard (Z-score) scaling.
- :class:`~kgcnn_torch.data.transform.StandardLabelScaler` -- Standard scaling with dataset-level methods.
- :class:`~kgcnn_torch.data.transform.ExtensiveMolecularScaler` -- Extensive property scaling via ridge regression on atom counts.
- :class:`~kgcnn_torch.data.transform.ExtensiveMolecularLabelScaler` -- Label-oriented API for extensive scaling.
- :class:`~kgcnn_torch.data.transform.ForceStandardScaler` -- Joint energy/force scaling.
- :class:`~kgcnn_torch.data.transform.QMGraphLabelScaler` -- Multi-target QM label scaling.

Standard Scalers
----------------

.. autoclass:: kgcnn_torch.data.transform.StandardScaler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: kgcnn_torch.data.transform.StandardLabelScaler
   :members:
   :undoc-members:
   :show-inheritance:

Extensive Molecular Scalers
---------------------------

.. autoclass:: kgcnn_torch.data.transform.ExtensiveMolecularScaler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: kgcnn_torch.data.transform.ExtensiveMolecularLabelScaler
   :members:
   :undoc-members:
   :show-inheritance:

Force Scaler
------------

.. autoclass:: kgcnn_torch.data.transform.ForceStandardScaler
   :members:
   :undoc-members:
   :show-inheritance:

QM Graph Label Scaler
---------------------

.. autoclass:: kgcnn_torch.data.transform.QMGraphLabelScaler
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: kgcnn_torch.data.transform
   :members:
   :undoc-members:
   :show-inheritance:
