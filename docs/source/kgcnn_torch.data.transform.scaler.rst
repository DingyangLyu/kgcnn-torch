kgcnn\_torch.data.transform.scaler
====================================

.. note::

   In kgcnn-torch, all scaler classes are located in the single module
   :mod:`kgcnn_torch.data.transform` rather than in a separate ``scaler``
   subpackage. This page provides a focused view of the scaler classes
   for users familiar with the Keras ``kgcnn.data.transform.scaler`` layout.

For full documentation of every scaler, see :doc:`kgcnn_torch.data.transform`.

Scalers Overview
----------------

Standard Scalers
~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   kgcnn_torch.data.transform.StandardScaler
   kgcnn_torch.data.transform.StandardLabelScaler

Extensive Molecular Scalers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   kgcnn_torch.data.transform.ExtensiveMolecularScaler
   kgcnn_torch.data.transform.ExtensiveMolecularLabelScaler

Force and Energy Scalers
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   kgcnn_torch.data.transform.ForceStandardScaler
   kgcnn_torch.data.transform.EnergyForceExtensiveLabelScaler

QM Scalers
~~~~~~~~~~

.. autosummary::
   :nosignatures:

   kgcnn_torch.data.transform.QMGraphLabelScaler
