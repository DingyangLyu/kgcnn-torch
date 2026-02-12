# Notebooks

Example notebooks demonstrating usage of `kgcnn-torch` (PyTorch Graph Neural Networks).

## Tutorials

| Notebook | Description |
|---|---|
| [tutorial_pytorch_gpu.ipynb](tutorial_pytorch_gpu.ipynb) | GPU device selection, torch.compile(), mixed precision, DataLoader parallelism, and memory management |
| [tutorial_config_training.ipynb](tutorial_config_training.ipynb) | Config-driven training with HyperParameter files |
| [tutorial_custom_crystal_dataset.ipynb](tutorial_custom_crystal_dataset.ipynb) | Building a custom crystal dataset for GNN training |
| [tutorial_custom_moleculenet.ipynb](tutorial_custom_moleculenet.ipynb) | Creating a custom MoleculeNet-style dataset |
| [tutorial_custom_qm_dataset.ipynb](tutorial_custom_qm_dataset.ipynb) | Creating a custom quantum chemistry dataset |
| [tutorial_graph_dict.ipynb](tutorial_graph_dict.ipynb) | Working with GraphDict and MemoryGraphList data structures |
| [tutorial_hyper_optuna.ipynb](tutorial_hyper_optuna.ipynb) | Hyperparameter optimization with Optuna |
| [tutorial_model_loading_options.ipynb](tutorial_model_loading_options.ipynb) | Model loading, saving, and checkpoint options |
| [tutorial_optimizer.ipynb](tutorial_optimizer.ipynb) | Optimizers and learning rate schedulers |

## Workflows

| Notebook | Description |
|---|---|
| [example_transfer_learning.ipynb](example_transfer_learning.ipynb) | Pretrain NMPN on ESOL, freeze backbone, fine-tune output MLP on FreeSolv |
| [workflow_molecule_regression.ipynb](workflow_molecule_regression.ipynb) | End-to-end molecule property regression workflow |
| [workflow_qm_regression.ipynb](workflow_qm_regression.ipynb) | Quantum chemistry regression workflow |

## Showcases

| Notebook | Description |
|---|---|
| [showcase_energy_force_model.ipynb](showcase_energy_force_model.ipynb) | Energy and force prediction with equivariant GNNs |
| [showcase_hdnnp2nd_dipole.ipynb](showcase_hdnnp2nd_dipole.ipynb) | HDNNP2nd model for dipole moment prediction |

## Explanations

| Notebook | Description |
|---|---|
| [graph_explanation/explain_GNNExplain_mutagenicity.ipynb](graph_explanation/explain_GNNExplain_mutagenicity.ipynb) | GNNExplainer for graph classification on the Mutagenicity dataset |
| [graph_explanation/explain_GNNExplain_cora.ipynb](graph_explanation/explain_GNNExplain_cora.ipynb) | GNNExplainer for node classification on the Cora citation network |
