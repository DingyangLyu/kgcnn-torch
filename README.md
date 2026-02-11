# kgcnn-torch

基于 **PyTorch** 和 **PyTorch Geometric (PyG)** 的图神经网络库，面向分子和材料科学领域的属性预测任务。

本项目从 [kgcnn](https://github.com/aimat-lab/gcnn_keras) v4.0.2（Keras 3 多后端版）完整转换而来，提供 **28 种** GNN 模型的 PyTorch 原生实现，所有模型均通过与 Keras 原始实现的数值对齐验证（含前向传播 + 反向传播训练对齐，支持跨框架 PyTorch↔TensorFlow 验证）。

---

## 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
  - [图级别属性预测](#1-图级别属性预测)
  - [能量与力场预测](#2-能量与力场预测mlip)
  - [在代码中使用模型](#3-在代码中使用模型)
  - [力场模型（EnergyForceModel）](#4-力场模型energyforcemodel)
- [支持的模型](#支持的模型)
  - [经典图神经网络](#经典图神经网络)
  - [连续滤波与消息传递](#连续滤波与消息传递)
  - [等变网络](#等变网络)
  - [注意力机制](#注意力机制)
  - [材料科学专用](#材料科学专用)
  - [其他模型](#其他模型)
- [层系统](#层系统)
  - [几何与基函数层](#几何与基函数层)
  - [图卷积层](#图卷积层)
  - [注意力层](#注意力层)
  - [聚合与池化层](#聚合与池化层)
  - [MLP 与归一化层](#mlp-与归一化层)
  - [辅助层](#辅助层)
- [训练系统](#训练系统)
  - [训练器 (Trainer)](#训练器-trainer)
  - [回调系统 (Callbacks)](#回调系统-callbacks)
  - [学习率调度器](#学习率调度器)
  - [损失函数](#损失函数)
  - [评估指标](#评估指标)
  - [数据缩放](#数据缩放)
- [超参数配置](#超参数配置)
  - [配置文件格式](#配置文件格式)
  - [预置配置](#预置配置)
- [数据管线](#数据管线)
- [项目结构](#项目结构)
- [测试](#测试)
- [与 Keras 版的关键差异](#与-keras-版的关键差异)
- [许可证](#许可证)

---

## 特性

- **28 种 GNN 模型**：涵盖经典 GNN、消息传递、等变网络、注意力机制、材料科学专用模型
- **力场预测**：通过 `EnergyForceModel` 包装器，任何能量预测模型均可自动计算力 ($\vec{F}_i = -\nabla_i E$)
- **周期性边界条件**：原生支持晶体材料（lattice + edge_image），SchNet/PAiNN/DimeNetPP 等均有 Crystal 变体
- **完整训练管线**：内置交叉验证、早停、模型检查点、学习率调度、数据缩放
- **与 Keras 数值对齐**：全部 28 个模型均通过前向+训练对齐验证，支持 Keras torch 后端（同框架）和 TensorFlow 后端（跨框架）双重验证
- **PyG 生态兼容**：基于 `torch_geometric.data.Data` 标准格式，可无缝使用 PyG 数据集和 DataLoader

---

## 安装

### 基础安装

```bash
git clone https://gitee.com/baiyuan1/kgcnn-torch.git
cd kgcnn-torch
pip install -e .
```

### 核心依赖

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Python | >= 3.9 | |
| PyTorch | >= 2.0 | 核心深度学习框架 |
| PyTorch Geometric | >= 2.4 | 图数据处理与 DataLoader |
| NumPy | >= 1.22 | 数值计算 |
| pandas | >= 1.4 | 数据处理 |
| SciPy | >= 1.7 | 科学计算（几何、多项式基函数等） |
| NetworkX | >= 2.6 | 图操作 |
| requests | — | 数据集下载 |

### 可选依赖

```bash
# 分子数据处理（SMILES 解析、分子图构建、分子动力学等）
pip install -e ".[molecule]"   # rdkit, openbabel-wheel, ase

# 晶体数据处理（CIF 解析、周期性图构建等）
pip install -e ".[crystal]"    # pymatgen, pyxtal

# 可视化
pip install -e ".[vis]"        # matplotlib

# 额外功能（数据缩放、指标、YAML 配置等）
pip install -e ".[extras]"     # scikit-learn, pyyaml

# 全部可选依赖
pip install -e ".[all]"

# 开发依赖（测试）
pip install -e ".[dev]"
```

---

## 快速开始

### 1. 图级别属性预测

使用内置训练脚本，通过 JSON 超参数配置文件一键训练：

```bash
# 在 ESOL 数据集上训练 SchNet 模型（5 折交叉验证）
python training_scripts/train_graph.py \
    --hyper training_scripts/hyper/hyper_esol.json \
    --category SchNet \
    --output results/

# 指定特定折数
python training_scripts/train_graph.py \
    --hyper training_scripts/hyper/hyper_esol.json \
    --category GIN \
    --fold 0 1 2 \
    --output results/

# 使用 GPU
python training_scripts/train_graph.py \
    --hyper training_scripts/hyper/hyper_qm9.json \
    --category PAiNN \
    --device cuda \
    --output results/
```

**支持的模型（train_graph.py）**：SchNet, PAiNN, DimeNetPP, GCN, GAT, GATv2, GIN, EGNN, DMPNN, GraphSAGE, Megnet, AttentiveFP, CGCNN, NMPN, INorp, MEGAN, RGCN, GNNFilm, rGIN, MXMNet, MoGAT, CMPNN, DGIN, HamNet, HDNNP2nd, MAT

### 2. 能量与力场预测（MLIP）

```bash
# 在 MD17 Revised 数据集上训练 SchNet 力场模型
python training_scripts/train_force.py \
    --hyper training_scripts/hyper/hyper_md17_revised.json \
    --category SchNet \
    --output results/
```

力场训练通过 `torch.autograd.grad` 自动计算原子力（$F = -\nabla_{\text{pos}} E$），支持的模型：**SchNet, PAiNN, DimeNetPP, EGNN**。

### 3. 在代码中使用模型

```python
import torch
from torch_geometric.data import Data, Batch
from kgcnn_torch.models.schnet import SchNetModel

# 构建模型
model = SchNetModel(
    node_dim=64,       # 原子嵌入维度
    depth=4,           # 交互层数
    units=128,         # 隐藏层维度
    gauss_bins=20,     # 高斯基函数数量
    gauss_distance=4.0,# 高斯展开最大距离
    num_targets=1      # 输出目标数
)

# 准备 PyG 格式数据
data = Data(
    z=torch.tensor([6, 1, 1, 1, 1]),                    # 原子序数
    pos=torch.randn(5, 3),                                # 原子坐标
    edge_index=torch.tensor([[0,0,0,0,1,2,3,4],
                              [1,2,3,4,0,0,0,0]]),        # 边索引 [source, target]
    batch=torch.zeros(5, dtype=torch.long)                # batch 分配
)

# 前向推理
prediction = model(data)  # shape: (1, num_targets)
```

### 4. 力场模型（EnergyForceModel）

```python
from kgcnn_torch.models.schnet import SchNetModel
from kgcnn_torch.models.force import EnergyForceModel

# 用任意能量模型构建力场模型
energy_model = SchNetModel(node_dim=128, depth=6, units=128, num_targets=1)
model = EnergyForceModel(
    energy_model=energy_model,
    coordinate_input="pos",        # 坐标属性名
    output_as_dict=True,           # 输出为 dict（也可设为 False 返回 tuple）
    is_physical_force=True         # F = -∇E（物理力）
)

# data.pos 需要 requires_grad=True
data.pos.requires_grad_(True)
result = model(data)
energy = result["energy"]  # 总能量
forces = result["force"]   # 原子力 (N, 3)
```

---

## 支持的模型

### 经典图神经网络

| 模型 | 类名 | 论文 | 说明 |
|------|------|------|------|
| **GCN** | `GCNModel` | Kipf & Welling, 2017 | 半监督分类图卷积网络，支持整数/浮点节点特征 |
| **GAT** | `GATModel` | Velickovic et al., 2018 | 图注意力网络，多头注意力（concat/average），支持边特征 |
| **GATv2** | `GATv2Model` | Brody et al., 2021 | 改进版 GAT，$\alpha_{ij} = a^T \sigma(W[n_i \| n_j])$（先激活再计算注意力） |
| **GIN** | `GINModel` | Xu et al., 2019 | 图同构网络，$h_i' = (1+\varepsilon)h_i + \sum_j h_j$，可学习 $\varepsilon$ |
| **rGIN** | `rGINModel` | — | GIN 的残差变体，带跳跃连接 |
| **GraphSAGE** | `GraphSAGEModel` | Hamilton et al., 2017 | 基于采样的图表示学习 |
| **RGCN** | `RGCNModel` | Schlichtkrull et al., 2018 | 关系图卷积，支持多关系类型 |

### 连续滤波与消息传递

| 模型 | 类名 | 论文 | 说明 |
|------|------|------|------|
| **SchNet** | `SchNetModel` | Schütt et al., 2017 | 连续滤波卷积，高斯基函数展开距离，shifted softplus 激活 |
| **DimeNetPP** | `DimeNetPPModel` | Klicpera et al., 2020 | 方向性消息传递，球面基函数（距离+角度），三体交互 |
| **DMPNN** | `DMPNNModel` | Yang et al., 2019 | 有向消息传递，分子属性预测 |
| **CMPNN** | `CMPNNModel` | — | Chemprop 变体，增强特征 |
| **NMPN** | `NMPNModel` | Gilmer et al., 2017 | 神经消息传递网络 |
| **DGIN** | `DGINModel` | — | 双图同构网络 |
| **INorp** | `INorpModel` | — | 图交互网络 |

### 等变网络

| 模型 | 类名 | 论文 | 说明 |
|------|------|------|------|
| **PAiNN** | `PAiNNModel` | Schütt et al., 2021 | 等变消息传递，维护标量 $(N,F)$ + 矢量 $(N,3,F)$ 双通道特征 |
| **EGNN** | `EGNNModel` | Satorras et al., 2021 | 等变图神经网络，坐标等变性 |
| **HamNet** | `HamNetModel` | — | 哈密顿神经网络变体 |

### 注意力机制

| 模型 | 类名 | 论文 | 说明 |
|------|------|------|------|
| **AttentiveFP** | `AttentiveFPModel` | Xiong et al., 2020 | 基于注意力的分子指纹，GRU + 迭代图嵌入精炼 |
| **MAT** | `MATModel` | — | 分子注意力 Transformer，使用 dense padded 表示 |
| **MEGAN** | `MEGANModel` | — | 多头等变图注意力网络 |
| **MoGAT** | `MoGATModel` | — | 多阶 GAT |

### 材料科学专用

| 模型 | 类名 | 论文 | 说明 |
|------|------|------|------|
| **CGCNN** | `CGCNNModel` | Xie & Grossman, 2018 | 晶体图卷积，门控机制 |
| **MEGNet** | `MEGNetModel` | Chen et al., 2019 | 材料图网络，全局状态更新 |
| **HDNNP2nd** | `HDNNP2ndModel` | Behler & Parrinello | 高维神经网络势，原子化能量 |
| **SchNetCrystal** | `SchNetCrystalModel` | — | SchNet 晶体变体，原生周期性边界条件 |

### 其他模型

| 模型 | 类名 | 说明 |
|------|------|------|
| **MXMNet** | `MXMNetModel` | 分子力学网络 |
| **GNNFilm** | `GNNFilmModel` | Feature-wise Linear Modulation |
| **GNNExplain** | `GNNExplainModel` | GNN 可解释性框架 |

---

## 层系统

所有层位于 `kgcnn_torch/layers/`，遵循 PyG 边索引约定（`edge_index[0]` = source, `edge_index[1]` = target）。

### 几何与基函数层

位于 `layers/geom.py`，提供距离计算和基函数展开：

| 层 / 函数 | 说明 |
|-----------|------|
| `compute_edge_distances(pos, edge_index)` | 计算边的欧氏距离 |
| `compute_edge_direction_normalized(pos, edge_index)` | 计算归一化方向向量 |
| `shift_periodic_lattice(pos, edge_image, lattice, batch)` | 周期性边界条件下的坐标偏移 |
| `GaussBasisLayer(bins, distance, sigma)` | 高斯基函数展开（SchNet 使用） |
| `BesselBasisLayer(num_radial, cutoff)` | 贝塞尔函数基（DimeNet 使用），可训练频率参数 |
| `SphericalBasisLayer(num_radial, num_spherical, cutoff)` | 球面基函数 = 径向贝塞尔 + 球谐函数（DimeNet++ 使用） |
| `CosCutOffEnvelope(cutoff)` | 余弦截断包络 $\frac{1}{2}(\cos(\pi r / r_c) + 1)$ |
| `NodePosition` | 从边索引提取源/目标节点坐标 |

### 图卷积层

位于 `layers/conv.py`：

| 层 | 说明 | 关键参数 |
|----|------|---------|
| `GCNConv` | 标准图卷积 $\sigma(W \cdot x \cdot e)$ + 聚合 | `in_features`, `out_features`, `activation` |
| `SchNetCFconv` | SchNet 连续滤波卷积：边特征→滤波器变换 | `units`, `edge_dim` |
| `SchNetInteraction` | SchNet 完整交互块（CFconv + Dense + 残差连接） | `units`, `edge_dim`, `activation` |
| `GINConv` | GIN 卷积 $h' = (1+\varepsilon)h + \sum h_j$ | `units`, `activation` |
| `GINEConv` | GIN + 边特征版本 | `units`, `edge_dim` |
| `CGCNNLayer` | CGCNN 门控卷积 | `units` |

### 注意力层

位于 `layers/attention.py`：

| 层 | 说明 |
|----|------|
| `AttentionHeadGAT` | 单头 GAT 注意力：$\alpha_{ij} = \text{softmax}(a^T [Wn_i \| Wn_j])$ |
| `AttentionHeadGATV2` | GATv2 注意力：$\alpha_{ij} = a^T \sigma(W[n_i \| n_j])$ |
| `MultiHeadGATV2Layer` | 多头 GATv2（支持输出注意力 logits，用于 MEGAN） |

### 聚合与池化层

**聚合层** (`layers/aggr.py`)：

| 层 | 说明 |
|----|------|
| `Aggregate` | 通用 scatter 聚合（按索引） |
| `AggregateLocalEdges` | 边→目标节点聚合，支持 sum/mean/max/min |
| `AggregateLocalEdgesAttention` | softmax 加权聚合 |
| `AggregateLocalEdgesLSTM` | LSTM 序列聚合 |
| `RelationalAggregateLocalEdges` | 按关系类型分别聚合 |

**池化层** (`layers/pooling.py`)：

| 层 | 说明 |
|----|------|
| `PoolingNodes` | 节点→图级别聚合（sum/mean/max/min） |
| `PoolingWeightedNodes` | 带权重的节点池化 |
| `PoolingEmbeddingAttention` | softmax 加权池化 |
| `PoolingNodesAttentive` | AttentiveFP 风格的 GRU + 注意力迭代精炼池化 |

### MLP 与归一化层

**MLP** (`layers/mlp.py`)：

```python
from kgcnn_torch.layers.mlp import MLP

mlp = MLP(
    units=[128, 64, 1],          # 各层输出维度
    input_dim=256,                # 输入维度
    activation="relu",            # 激活函数（支持列表形式逐层指定）
    use_bias=True,
    use_dropout=False,
    dropout_rate=0.0,
    use_normalization=False,
    normalization_technique="batch"  # batch/layer/graph/group/unit_norm
)
```

**归一化层** (`layers/norm.py`)：

| 层 | 说明 |
|----|------|
| `GraphBatchNorm` | 按图分组的 BatchNorm（非全局 BatchNorm） |
| `GraphLayerNorm` | 按图分组的 LayerNorm |
| `GraphNormalization` | 通用按图归一化 |

### 辅助层

| 模块 | 位置 | 说明 |
|------|------|------|
| `gather_nodes_outgoing / ingoing` | `layers/gather.py` | 根据 edge_index 提取源/目标节点特征（纯函数） |
| `GRUUpdate` | `layers/update.py` | GRU 状态更新层 |
| `ResidualLayer` | `layers/update.py` | 残差连接 $x + \sigma(\text{Dense}(x))$ |
| `StandardLabelScaler` | `layers/scale.py` | 目标值标准化（均值/标准差缩放） |
| `ExtensiveMolecularLabelScaler` | `layers/scale.py` | 分子能量缩放（Ridge 回归计算各元素参考能量） |

### 底层算子

| 模块 | 位置 | 说明 |
|------|------|------|
| `scatter_reduce_sum/mean/max/min/softmax` | `ops/scatter.py` | Scatter 归约操作 |
| `get_activation(name)` | `ops/activ.py` | 激活函数工厂，支持 relu, sigmoid, tanh, swish, silu, elu, softplus, shifted_softplus, leaky_relu, linear |
| `glorot_orthogonal_` | `initializers/initializers.py` | 正交 + Glorot 缩放初始化（DimeNet++ 使用） |

---

## 训练系统

### 训练器 (Trainer)

位于 `kgcnn_torch/training/trainer.py`，提供三个核心函数：

| 函数 | 说明 |
|------|------|
| `train_epoch(model, loader, optimizer, loss_fn, device)` | 训练一个 epoch，支持标量和 dict 输出（力场模型） |
| `eval_epoch(model, loader, loss_fn, metrics, device, scaler)` | 评估，支持逆缩放后计算指标 |
| `fit(model, train_loader, val_loader, optimizer, loss_fn, ...)` | 完整训练循环：早停、检查点、调度器、回调 |

`fit()` 函数的关键参数：

```python
history = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=500,
    metrics={"mae": mae, "rmse": rmse},
    scheduler=scheduler,
    callbacks=[early_stop, checkpoint],
    device="cuda",
    scaler=scaler  # 逆变换后计算指标
)
```

### 回调系统 (Callbacks)

位于 `kgcnn_torch/training/callbacks.py`：

| 回调 | 说明 | 关键参数 |
|------|------|---------|
| `EarlyStoppingCallback` | 指标无改善时停止训练 | `patience`, `monitor`, `mode`('min'/'max'), `min_delta`, `restore_best_weights` |
| `ModelCheckpointCallback` | 保存最优模型 | `filepath`（支持 `{epoch}`, `{val_loss}` 占位符）, `monitor`, `save_best_only` |
| `LearningRateLoggingCallback` | 记录当前学习率 | — |

回调基类 `TrainingCallback` 提供以下 hook：
- `on_train_begin/end(logs)`
- `on_epoch_begin/end(epoch, logs)`
- `on_batch_begin/end(batch, logs)`

### 学习率调度器

位于 `kgcnn_torch/training/scheduler.py`，通过 `get_scheduler(name, optimizer, **kwargs)` 工厂函数创建：

| 调度器 | 名称标识 | 说明 |
|--------|---------|------|
| `LinearWarmupScheduler` | `linear_warmup` | 线性预热，之后保持不变 |
| `LinearWarmupExponentialDecay` | `LinearWarmupExponentialDecay` | 线性预热 + 指数衰减 |
| `PolynomialDecayScheduler` | `polynomial_decay` | 多项式衰减 |
| `CosineWarmupScheduler` | `cosine_warmup` | 线性预热 + 余弦退火 |
| `LinearLearningRateScheduler` | `LinearLearningRateScheduler` | 保持→线性衰减（兼容 Keras） |
| `LinearWarmupLinearLearningRateScheduler` | `warmup_linear` | 预热→线性衰减（MatBench 配置使用） |
| `ReduceLROnPlateau` | `ReduceLROnPlateau` | PyTorch 内置，指标停滞时降低 LR |
| `StepLR` | `step` | 固定步长衰减 |
| `ExponentialLR` | `exponential` | 指数衰减 |
| `CosineAnnealingLR` | `cosine` | 余弦退火 |

### 损失函数

位于 `kgcnn_torch/losses/losses.py`：

| 损失函数 | 说明 |
|---------|------|
| `EnergyForceLoss(energy_weight, force_weight)` | 能量 + 力联合损失：$L = w_E \cdot \text{MAE}(E) + w_F \cdot \text{MAE}(F)$ |
| `ForceMeanAbsoluteError(per_molecule)` | 力 MAE，可选按分子归一化（防止大分子主导） |
| `DisjointForceMeanAbsoluteError` | 不相交图表示下的力 MAE，支持 padding 检测 |
| `BinaryCrossentropyNoNaN` | 忽略 NaN 标签的二元交叉熵（用于缺失标签数据集） |
| `RaggedValuesMeanAbsoluteError` | 处理变长（ragged）预测的 MAE |

### 评估指标

位于 `kgcnn_torch/metrics/metrics.py`：

**回归指标**：

| 指标 | 说明 |
|------|------|
| `mae(pred, target)` | 平均绝对误差 |
| `rmse(pred, target)` | 均方根误差 |
| `mse(pred, target)` | 均方误差 |
| `ScaledMeanAbsoluteError(scale)` | 缩放后 MAE（在原始单位下评估） |
| `ScaledRootMeanSquaredError(scale)` | 缩放后 RMSE |
| `ScaledForceMeanAbsoluteError(scale)` | 缩放后力 MAE，支持 padding 原子检测 |

**分类指标**：

| 指标 | 说明 |
|------|------|
| `BinaryAccuracyNoNaN(threshold)` | 忽略 NaN 的二元准确率 |
| `BalancedBinaryAccuracyNoNaN(threshold)` | 忽略 NaN 的平衡准确率 $(sensitivity + specificity) / 2$ |
| `AUCNoNaN` | 忽略 NaN 的 ROC-AUC（基于 sklearn） |

### 数据缩放

| 缩放器 | 说明 |
|--------|------|
| `StandardLabelScaler` | 标准缩放：$(y - \mu) / \sigma$，`fit/transform/inverse_transform` 接口 |
| `ExtensiveMolecularLabelScaler` | 广延量缩放：通过 Ridge 回归学习各元素参考能量，$E_{\text{corrected}} = E - \sum_i E_{\text{ref}}(Z_i)$ |

---

## 超参数配置

### 配置文件格式

超参数通过 JSON 文件管理，由 `HyperParameter` 类加载（`kgcnn_torch/training/hyper.py`）。每个 JSON 文件可包含多个模型的配置，以模型名为键：

```json
{
  "SchNet": {
    "model": {
      "config": {
        "model_name": "SchNet",
        "num_features": 128,
        "num_filters": 128,
        "num_interactions": 6,
        "cutoff": 10.0,
        "num_gaussians": 50,
        "readout": "sum",
        "output_dim": 1
      }
    },
    "training": {
      "fit": {
        "epochs": 500,
        "batch_size": 32,
        "early_stopping_patience": 50
      },
      "compile": {
        "optimizer": {"class_name": "Adam", "config": {"lr": 5e-4}},
        "loss": "mae"
      },
      "scheduler": {
        "class_name": "ReduceLROnPlateau",
        "patience": 25,
        "factor": 0.5,
        "min_lr": 1e-6
      },
      "scaler": {
        "class_name": "StandardLabelScaler"
      },
      "cross_validation": {
        "n_splits": 5,
        "shuffle": true
      }
    },
    "data": {
      "dataset": {
        "class_name": "ESOL",
        "config": {"root": "data/"}
      },
      "data_unit": "log mol/L"
    }
  }
}
```

**GNNFilm / RGCN 输出头激活说明**：

- 两个模型均支持 `output_final_activation` 参数。
- 默认值为 `linear`（更适配 `BCEWithLogitsLoss` / `CrossEntropyLoss` 这类 logits loss）。
- 若你要严格复刻 Keras literature 默认行为，可显式设置 `output_final_activation: "softmax"`。

**力场训练配置示例**（`hyper_md17_revised.json`）：

```json
{
  "SchNet": {
    "model": {
      "config": {
        "num_features": 256,
        "num_filters": 256,
        "num_interactions": 6,
        "cutoff": 5.0,
        "num_gaussians": 25,
        "readout": "sum",
        "output_dim": 1
      }
    },
    "training": {
      "fit": {
        "epochs": 1000,
        "batch_size": 32,
        "early_stopping_patience": 100,
        "energy_key": "energy",
        "force_key": "force"
      },
      "compile": {
        "optimizer": {"class_name": "Adam", "config": {"lr": 1e-3}},
        "loss": "energy_force",
        "energy_weight": 0.02,
        "force_weight": 0.98
      },
      "scheduler": {
        "class_name": "ReduceLROnPlateau",
        "patience": 50,
        "factor": 0.5,
        "min_lr": 1e-7
      },
      "scaler": {
        "class_name": "EnergyForceExtensiveLabelScaler"
      },
      "cross_validation": {
        "n_splits": 3,
        "shuffle": true
      }
    },
    "data": {
      "dataset": {
        "class_name": "MD17Revised",
        "config": {"trajectory_name": "toluene"}
      },
      "data_unit": "kcal/mol"
    }
  }
}
```

### 预置配置

位于 `training_scripts/hyper/`：

| 配置文件 | 任务类型 | 数据集 |
|---------|---------|--------|
| `hyper_esol.json` | 分子回归 | ESOL（溶解度） |
| `hyper_freesolv.json` | 分子回归 | FreeSolv（水合自由能） |
| `hyper_lipop.json` | 分子回归 | Lipophilicity（亲脂性） |
| `hyper_qm7.json` | 量子属性 | QM7 |
| `hyper_qm9.json` | 量子属性 | QM9 |
| `hyper_qm9_energies.json` | 量子能量 | QM9 |
| `hyper_qm9_orbitals.json` | 量子轨道 | QM9 |
| `hyper_mutag.json` | 图分类 | MUTAG |
| `hyper_mutagenicity.json` | 图分类 | Mutagenicity |
| `hyper_proteins.json` | 图分类 | PROTEINS |
| `hyper_clintox.json` | 分子分类 | ClinTox（毒性） |
| `hyper_sider.json` | 分子分类 | SIDER（副作用） |
| `hyper_tox21mol.json` | 分子分类 | Tox21 |
| `hyper_cora.json` / `hyper_cora_lu.json` | 节点分类 | Cora |
| `hyper_md17.json` / `hyper_md17_revised.json` | 力场预测 | MD17 / MD17 Revised |
| `hyper_iso17.json` | 力场预测 | ISO17 |
| `hyper_mp_e_form.json` | 材料属性 | Materials Project（形成能） |
| `hyper_mp_gap.json` | 材料属性 | Materials Project（带隙） |
| `hyper_mp_dielectric.json` | 材料属性 | Materials Project（介电常数） |
| `hyper_mp_is_metal.json` | 材料分类 | Materials Project（金属性） |
| `hyper_mp_phonons.json` | 材料属性 | Materials Project（声子） |
| `hyper_mp_perovskites.json` | 材料属性 | Materials Project（钙钛矿） |
| `hyper_mp_log_gvrh.json` / `hyper_mp_log_kvrh.json` | 材料属性 | Materials Project（体积/剪切模量） |
| `hyper_mp_jdft2d.json` | 材料属性 | Materials Project（2D 材料） |

---

## 数据管线

kgcnn-torch 使用 PyG 的 `Data` 对象作为标准数据格式。内置的数据处理管线支持从 KGCNN 原始格式（`MemoryGraphList`）到 PyG 格式的转换：

### PyG Data 属性约定

| KGCNN 属性 | PyG 属性 | 说明 |
|------------|----------|------|
| `node_number` | `z` | 原子序数 (N,) |
| `node_coordinates` | `pos` | 原子坐标 (N, 3) |
| `edge_indices` | `edge_index` | 边索引 (2, M)，**自动执行 KGCNN→PyG 约定转换** |
| `graph_labels` | `y` | 图级标签 |
| `graph_lattice` | `lattice` | 晶格矩阵 (B, 3, 3) |
| `range_image` | `edge_image` | 周期性镜像偏移向量 (M, 3) |
| `angle_indices` | `angle_index` | 角度三元组索引（DimeNet++ 使用） |

> **重要**: KGCNN 使用 `[target, source]` 边索引约定，而 PyG 使用 `[source, target]`。转换在 `to_pyg_list()` 中自动完成。

### 图数据处理模块

- **`kgcnn_torch/graph/`**：通用图预处理（邻接矩阵、几何计算、周期性边界）
- **`kgcnn_torch/molecule/`**：分子数据处理（RDKit/OpenBabel 集成、SMILES 解析、分子图构建、分子编码器）
- **`kgcnn_torch/crystal/`**：晶体数据处理（周期性图构建、元素周期表嵌入）
- **`kgcnn_torch/io/`**：数据 I/O 工具

---

## 项目结构

```
kgcnn-torch/
├── kgcnn_torch/                     # 核心库
│   ├── models/                      # 28 个 GNN 模型 + force.py + multi.py + gnnexplain.py
│   │   ├── schnet.py                #   SchNet / SchNetCrystal
│   │   ├── painn.py                 #   PAiNN（等变消息传递）
│   │   ├── dimenetpp.py             #   DimeNet++（方向性消息传递）
│   │   ├── force.py                 #   EnergyForceModel 包装器
│   │   ├── gcn.py, gat.py, gin.py   #   经典 GNN
│   │   └── ...                      #   其余模型
│   ├── layers/                      # 构建模型的基础层
│   │   ├── conv.py                  #   图卷积层
│   │   ├── attention.py             #   注意力层
│   │   ├── geom.py                  #   几何层与基函数
│   │   ├── aggr.py                  #   聚合层
│   │   ├── pooling.py               #   池化层
│   │   ├── mlp.py                   #   多层感知机
│   │   ├── norm.py                  #   归一化层
│   │   ├── gather.py                #   节点特征提取（纯函数）
│   │   ├── update.py                #   GRU/残差更新层
│   │   ├── scale.py                 #   数据缩放层
│   │   └── ...
│   ├── ops/                         # 底层算子
│   │   ├── scatter.py               #   scatter_reduce 操作
│   │   └── activ.py                 #   激活函数注册表
│   ├── training/                    # 训练基础设施
│   │   ├── trainer.py               #   训练循环（train_epoch/eval_epoch/fit）
│   │   ├── callbacks.py             #   早停、检查点、LR 记录
│   │   ├── scheduler.py             #   6 种自定义 + 4 种 PyTorch 内置调度器
│   │   ├── hyper.py                 #   超参数管理器
│   │   └── history.py               #   训练历史记录
│   ├── losses/                      # 损失函数
│   │   └── losses.py                #   EnergyForceLoss, BinaryCE(NaN), ForcMAE 等
│   ├── metrics/                     # 评估指标
│   │   └── metrics.py               #   MAE, RMSE, ScaledMAE, AUC(NaN), BalancedAcc 等
│   ├── initializers/                # 权重初始化
│   │   └── initializers.py          #   glorot_orthogonal_
│   ├── graph/                       # 图数据预处理（邻接矩阵、几何）
│   ├── crystal/                     # 晶体结构处理（周期性边界）
│   ├── molecule/                    # 分子数据处理（RDKit/OpenBabel）
│   ├── io/                          # 数据 I/O
│   └── utils/                       # 通用工具
│
├── training_scripts/                # 训练入口脚本
│   ├── train_graph.py               #   图级别属性预测（28 个模型）
│   ├── train_force.py               #   能量/力场预测（4 个模型）
│   ├── train_node.py                #   节点级别预测
│   └── hyper/                       #   25+ 超参数配置文件
│
├── scripts/                         # 开发与验证脚本
│   ├── align_*_layerwise.py         #   逐层数值对齐验证（Keras vs PyTorch）
│   ├── align_*_model.py             #   模型级别对齐验证
│   ├── run_layerwise_alignment.py   #   批量运行对齐测试
│   └── smoke_all_models.py          #   全模型冒烟测试
│
├── tests/                           # 单元测试
│   ├── test_ops_scatter.py          #   scatter 操作测试
│   ├── test_layers.py               #   层测试（21 项）
│   ├── test_models.py               #   模型测试（10 项）
│   ├── test_data_pipeline.py        #   数据管线测试（8 项）
│   └── test_alignment.py            #   Keras↔PyTorch 对齐测试
│
├── CONVERSION_GUIDE.md              # Keras→PyTorch 详细转换文档（~13000 行）
├── SDP.md                           # Keras vs Torch 对比检查报告
├── pyproject.toml                   # 项目配置与依赖
└── requirements.txt                 # 依赖列表
```

---

## 测试

```bash
# 运行全部测试
pytest tests/

# 运行特定测试文件
pytest tests/test_models.py -v
pytest tests/test_layers.py -v
pytest tests/test_ops_scatter.py -v

# 运行全模型冒烟测试
python scripts/smoke_all_models.py
```

**测试覆盖**：

| 测试文件 | 测试数量 | 内容 |
|---------|---------|------|
| `test_ops_scatter.py` | 7 | scatter reduce 操作（sum/mean/max/min/softmax）、空输入、梯度流 |
| `test_layers.py` | 21 | 所有层类型：Gather、聚合、几何、卷积、注意力、池化、MLP、归一化、更新、缩放 |
| `test_models.py` | 10 | GCN、GAT、SchNet、PAiNN、DimeNetPP 的前向/反向传播 |
| `test_data_pipeline.py` | 8 | GraphDict、MemoryGraphList、to_pyg_list 转换、DataLoader 集成、StandardScaler |
| `test_alignment.py` | — | Keras↔PyTorch 数值对齐验证 |

---

## 与 Keras 版的关键差异

| 方面 | KGCNN (Keras) | kgcnn-torch (PyTorch) |
|------|--------------|----------------------|
| 边索引约定 | `[target, source]` | `[source, target]`（PyG 标准） |
| 数据格式 | 自定义 ragged tensor | `torch_geometric.data.Data` |
| 模型定义 | Keras Functional API | `nn.Module` 子类 |
| 训练循环 | `model.fit()` | 自定义 `fit()` 函数 |
| 层共享 | Keras 层天然支持共享 | 需注意 `nn.Module` 引用语义 |
| 力计算 | 自定义梯度实现 | `torch.autograd.grad` |
| 数据集 | 30+ 内置数据集类 | 通过 PyG 数据集 + 自定义适配 |
| GNNFilm/RGCN 输出头默认值 | literature 默认 `softmax` | 默认 `output_final_activation="linear"`；如需严格复刻可显式设 `"softmax"` |

---

## 许可证

MIT
