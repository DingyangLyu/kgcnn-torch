# kgcnn-torch

`kgcnn-torch` is a native [PyTorch](https://pytorch.org/) and
[PyTorch Geometric](https://pyg.org/) implementation of graph neural networks
for molecular and materials-science prediction. It provides reusable layers,
models, datasets, and training utilities using PyG's `Data` and `Batch`
interfaces.

This is an independent PyTorch/PyG port inspired by the model definitions and
data tooling in [aimat-lab/kgcnn](https://github.com/aimat-lab/gcnn_keras).
It is not an official release of the original project.

## Highlights

- Native `torch.nn.Module` implementations designed for PyG data objects.
- Molecular, crystal, node-classification, and energy-and-force workflows.
- Periodic-boundary-condition support for crystal models.
- An `EnergyForceModel` wrapper that derives forces from predicted energies.
- Dataset loaders, preprocessing utilities, cross-validation, early stopping,
  checkpoints, learning-rate scheduling, metrics, and label scaling.
- Example notebooks and JSON-based training configurations.

## Implemented models

The repository includes implementations of GCN, GAT, GATv2, GIN, rGIN,
GraphSAGE, RGCN, SchNet, DimeNet++, DMPNN, CMPNN, DGIN, NMPN, INorp, PAiNN,
EGNN, HamNet, AttentiveFP, MAT, MEGAN, MoGAT, CGCNN, MEGNet, HDNNP2nd,
MXMNet, GNNFiLM, and GNNExplainer utilities. Some architectures also provide
crystal-aware variants.

### Conventional graph networks

| Model | Class | Notes |
| --- | --- | --- |
| GCN | `GCNModel` | Graph convolution for integer or floating-point node features. |
| GAT / GATv2 | `GATModel`, `GATv2Model` | Multi-head graph attention with optional edge features. |
| GIN / rGIN | `GINModel`, `rGINModel` | Graph-isomorphism networks; rGIN adds residual connections. |
| GraphSAGE | `GraphSAGEModel` | Inductive, sampling-oriented graph representation learning. |
| RGCN | `RGCNModel` | Relation-aware graph convolution. |

### Message passing and geometric models

| Model | Class | Notes |
| --- | --- | --- |
| SchNet | `SchNetModel` | Continuous-filter convolution with Gaussian distance expansion. |
| DimeNet++ | `DimeNetPPModel` | Directional message passing with radial and spherical basis functions. |
| PAiNN | `PAiNNModel` | Equivariant scalar and vector message passing. |
| EGNN | `EGNNModel` | E(n)-equivariant graph network. |
| DMPNN / CMPNN | `DMPNNModel`, `CMPNNModel` | Directed and communicative message passing for molecular prediction. |
| NMPN / DGIN / INorp | `NMPNModel`, `DGINModel`, `INorpModel` | Neural message passing and interaction-network variants. |
| HamNet / HDNNP2nd | `HamNetModel`, `HDNNP2ndModel` | Hamiltonian and high-dimensional neural-network-potential variants. |

### Attention, materials, and other architectures

| Model family | Class | Notes |
| --- | --- | --- |
| AttentiveFP / MAT / MEGAN / MoGAT | corresponding `*Model` classes | Molecular fingerprinting and attention-based graph models. |
| CGCNN / MEGNet | `CGCNNModel`, `MEGNetModel` | Crystal and materials-property prediction. |
| SchNetCrystal | `SchNetCrystalModel` | SchNet variant with periodic-boundary-condition support. |
| MXMNet | `MXMNetModel` | Molecular mechanics-inspired multiplex graph network. |
| GNNFiLM / GNNExplain | `GNNFilmModel`, `GNNExplainModel` | Feature-wise modulation and explanation utilities. |

## Layer system

All layers live in [`kgcnn_torch/layers`](kgcnn_torch/layers) and use the PyG
edge convention: `edge_index[0]` is the source node and `edge_index[1]` is the
target node.

### Geometry and basis functions

`layers/geom.py` contains the geometric building blocks used by molecular and
crystal models:

| Layer or function | Purpose |
| --- | --- |
| `compute_edge_distances(pos, edge_index)` | Euclidean distance for each edge. |
| `compute_edge_direction_normalized(pos, edge_index)` | Normalized edge direction vectors. |
| `shift_periodic_lattice(...)` | Coordinates shifted under periodic boundary conditions. |
| `GaussBasisLayer` | Gaussian basis expansion used by SchNet. |
| `BesselBasisLayer` | Radial Bessel basis with trainable frequency parameters. |
| `SphericalBasisLayer` | Combined radial Bessel and spherical-harmonic basis for DimeNet++. |
| `CosCutOffEnvelope` | Cosine cutoff envelope for finite interaction radii. |

### Convolutions, attention, aggregation, and pooling

| Area | Available components |
| --- | --- |
| Convolutions | `GCNConv`, `SchNetCFconv`, `SchNetInteraction`, `GINConv`, `GINEConv`, `CGCNNLayer` |
| Attention | `AttentionHeadGAT`, `AttentionHeadGATV2`, `MultiHeadGATV2Layer` |
| Edge aggregation | `Aggregate`, `AggregateLocalEdges`, `AggregateLocalEdgesAttention`, `AggregateLocalEdgesLSTM`, `RelationalAggregateLocalEdges` |
| Graph pooling | `PoolingNodes`, `PoolingWeightedNodes`, `PoolingEmbeddingAttention`, `PoolingNodesAttentive` |

Aggregation supports common reduction modes such as sum, mean, max, and min.
The attentive pooling implementation includes the iterative GRU-and-attention
refinement used by AttentiveFP.

### MLPs, normalization, and utilities

`MLP` supports per-layer widths and activations, optional dropout, and batch,
layer, graph, group, or unit normalization. Graph-aware normalization is
provided by `GraphBatchNorm`, `GraphLayerNorm`, and `GraphNormalization`.

Other useful components include `gather_nodes_outgoing`, `gather_nodes_ingoing`,
`GRUUpdate`, `ResidualLayer`, `StandardLabelScaler`, and
`ExtensiveMolecularLabelScaler`. Low-level scatter reductions and activations
are available under [`kgcnn_torch/ops`](kgcnn_torch/ops), including
`scatter_reduce_sum`, `scatter_reduce_mean`, `scatter_reduce_max`,
`scatter_reduce_min`, and `scatter_reduce_softmax`.

## Training system

The training module provides a framework-independent PyTorch training loop in
[`kgcnn_torch/training`](kgcnn_torch/training):

| Function | Purpose |
| --- | --- |
| `train_epoch` | Trains one epoch; supports scalar and dictionary outputs. |
| `eval_epoch` | Evaluates a loader with optional metrics and inverse scaling. |
| `fit` | Full loop with callbacks, schedulers, checkpoints, and early stopping. |

```python
from kgcnn_torch.training.trainer import fit

history = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=500,
    metrics={"mae": mae, "rmse": rmse},
    scheduler=scheduler,
    callbacks=[early_stopping, checkpoint],
    device="cuda",
    scaler=scaler,
)
```

### Callbacks and schedulers

`EarlyStoppingCallback`, `ModelCheckpointCallback`, and
`LearningRateLoggingCallback` provide the standard training lifecycle hooks.
The scheduler factory supports linear warm-up, warm-up plus exponential or
cosine decay, polynomial decay, linear decay, `ReduceLROnPlateau`, `StepLR`,
`ExponentialLR`, and `CosineAnnealingLR`.

### Losses, metrics, and scaling

- `EnergyForceLoss` combines energy and force objectives.
- `ForceMeanAbsoluteError` and `DisjointForceMeanAbsoluteError` evaluate forces
  while handling molecule-wise normalization or disjoint graphs.
- `BinaryCrossentropyNoNaN`, `BinaryAccuracyNoNaN`,
  `BalancedBinaryAccuracyNoNaN`, and `AUCNoNaN` handle datasets with missing
  labels.
- Regression metrics include MAE, MSE, RMSE, and scaled MAE/RMSE variants.
- `StandardLabelScaler` applies standard scaling; `ExtensiveMolecularLabelScaler`
  learns element reference energies with ridge regression for extensive targets.

## Hyperparameter configurations

JSON configurations are loaded by `HyperParameter` from
[`kgcnn_torch/training/hyper.py`](kgcnn_torch/training/hyper.py). A file can
store multiple model configurations keyed by model name:

```json
{
  "SchNet": {
    "model": {
      "config": {
        "num_features": 128,
        "num_filters": 128,
        "num_interactions": 6,
        "cutoff": 10.0,
        "num_gaussians": 50,
        "output_dim": 1
      }
    },
    "training": {
      "fit": {"epochs": 500, "batch_size": 32},
      "compile": {"optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}}, "loss": "mae"},
      "cross_validation": {"n_splits": 5, "shuffle": true}
    }
  }
}
```

The [`training_scripts/hyper`](training_scripts/hyper) directory contains
ready-to-run configurations for ESOL, FreeSolv, Lipophilicity, QM7/QM9,
MUTAG, Mutagenicity, PROTEINS, ClinTox, SIDER, Tox21, Cora, MD17, ISO17,
MatBench, and Materials Project tasks.

For `GNNFilm` and `RGCN`, `output_final_activation` defaults to `"linear"` to
work naturally with logits-based losses such as `BCEWithLogitsLoss` and
`CrossEntropyLoss`. Set it explicitly to `"softmax"` when reproducing a
configuration that expects probability outputs.

## Data pipeline

PyG `Data` is the standard graph representation. The dataset pipeline can
convert `MemoryGraphList`-style kgcnn data into PyG objects.

| Source kgcnn attribute | PyG attribute | Meaning |
| --- | --- | --- |
| `node_number` | `z` | Atomic numbers, shape `(N,)`. |
| `node_coordinates` | `pos` | Cartesian coordinates, shape `(N, 3)`. |
| `edge_indices` | `edge_index` | Edges, shape `(2, M)`. |
| `graph_labels` | `y` | Graph-level labels. |
| `graph_lattice` | `lattice` | Lattice matrices, shape `(B, 3, 3)`. |
| `range_image` | `edge_image` | Periodic image offsets, shape `(M, 3)`. |
| `angle_indices` | `angle_index` | Angle triplets for DimeNet++. |

> **Edge-order note:** the original kgcnn convention is `[target, source]`,
> while PyG uses `[source, target]`. `to_pyg_list()` performs this conversion.

The `graph`, `molecule`, `crystal`, and `io` modules cover generic graph
preprocessing, RDKit/Open Babel molecule processing, periodic crystal graph
construction, and data I/O respectively.

## Installation

Python 3.9 or later is required. Install a PyTorch build appropriate for your
CPU or CUDA environment first, then install PyG following its
[official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
git clone https://github.com/DinyangLyu/kgcnn-torch.git
cd kgcnn-torch
pip install -e .
```

Optional dependency groups are available for common workflows:

```bash
pip install -e ".[molecule]"  # RDKit, Open Babel, ASE
pip install -e ".[crystal]"   # pymatgen, pyxtal
pip install -e ".[vis]"       # matplotlib
pip install -e ".[extras]"    # scikit-learn, PyYAML
pip install -e ".[all]"       # all optional dependencies
pip install -e ".[dev]"       # test dependencies
```

## Quick start

```python
import torch
from torch_geometric.data import Data
from kgcnn_torch.models.schnet import SchNetModel

model = SchNetModel(
    node_dim=64,
    depth=4,
    units=128,
    gauss_bins=20,
    gauss_distance=4.0,
    num_targets=1,
)

data = Data(
    z=torch.tensor([6, 1, 1, 1, 1]),
    pos=torch.randn(5, 3),
    edge_index=torch.tensor(
        [[0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 0, 0, 0]]
    ),
    batch=torch.zeros(5, dtype=torch.long),
)

prediction = model(data)
```

### Energy and force prediction

```python
from kgcnn_torch.models.force import EnergyForceModel
from kgcnn_torch.models.schnet import SchNetModel

energy_model = SchNetModel(node_dim=128, depth=6, units=128, num_targets=1)
model = EnergyForceModel(
    energy_model=energy_model,
    coordinate_input="pos",
    output_as_dict=True,
    is_physical_force=True,  # force = -dE/dR
)

data.pos.requires_grad_(True)
result = model(data)
energy, forces = result["energy"], result["force"]
```

## Training examples

Training scripts use JSON hyperparameter configurations in
[`training_scripts/hyper`](training_scripts/hyper).

```bash
# Graph-level property prediction
python training_scripts/train_graph.py \
  --hyper training_scripts/hyper/hyper_esol.json \
  --category SchNet \
  --device cuda \
  --output results/

# Energy-and-force prediction
python training_scripts/train_force.py \
  --hyper training_scripts/hyper/hyper_md17_revised.json \
  --category SchNet \
  --output results/
```

See [`notebooks`](notebooks) for end-to-end tutorials and
[`CONVERSION_GUIDE.md`](CONVERSION_GUIDE.md) for implementation and API
differences from the Keras project.

## Project layout

```text
kgcnn_torch/
  data/              Dataset loaders, graph containers, and transforms
  layers/            Message passing, geometry, pooling, and neural layers
  models/            GNN architectures and the energy/force wrapper
  training/          Trainer, callbacks, schedulers, metrics, and configuration
  molecule/          Molecule conversion and preprocessing helpers
  crystal/           Crystal and periodic-structure helpers
training_scripts/    Command-line training entry points and configurations
notebooks/           Tutorials and workflow examples
tests/               Unit tests
```

## Testing

After installing development dependencies, run:

```bash
pytest tests
```

Some top-level training and dataset checks download data or require optional
scientific packages; they are intentionally not part of the minimal unit-test
command above.

## Acknowledgements

This project was developed by porting and adapting concepts, model
configurations, and parts of the data-processing approach from
[kgcnn](https://github.com/aimat-lab/gcnn_keras), the Keras graph-convolution
library maintained by **aimat-lab**. We thank its authors and contributors for
their foundational open-source work. Please cite and acknowledge the original
project when this port is used in research; see the upstream project's
[citation guidance](https://github.com/aimat-lab/gcnn_keras#citing).

The upstream `kgcnn` project is distributed under the MIT License. This
repository retains the required upstream copyright notice in
[`LICENSE`](LICENSE).

## License

`kgcnn-torch` is distributed under the [MIT License](LICENSE). It includes
adapted work from `aimat-lab/kgcnn`; see the license file and acknowledgements
above for attribution.
