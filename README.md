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
