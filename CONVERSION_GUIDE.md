# KGCNN → kgcnn-torch 转换详细文档

## 目录

1. [项目概述](#1-项目概述)
2. [为什么要转换](#2-为什么要转换)
3. [总体架构对比](#3-总体架构对比)
4. [Edge Index 约定转换（核心难点）](#4-edge-index-约定转换核心难点)
5. [基础操作层转换](#5-基础操作层转换)
   - 5.1 [Scatter 操作](#51-scatter-操作)
   - 5.2 [自定义激活函数](#52-自定义激活函数)
   - 5.3 [权重初始化器](#53-权重初始化器)
6. [核心层转换详解](#6-核心层转换详解)
   - 6.1 [Gather 层](#61-gather-层)
   - 6.2 [聚合层 (Aggregation)](#62-聚合层-aggregation)
   - 6.3 [图卷积层 (Convolution)](#63-图卷积层-convolution)
   - 6.4 [几何层 (Geometry)](#64-几何层-geometry)
   - 6.5 [注意力层 (Attention)](#65-注意力层-attention)
   - 6.6 [池化层 (Pooling)](#66-池化层-pooling)
   - 6.7 [MLP 层](#67-mlp-层)
   - 6.8 [归一化层 (Normalization)](#68-归一化层-normalization)
   - 6.9 [更新层 (Update)](#69-更新层-update)
   - 6.10 [缩放层 (Scale)](#610-缩放层-scale)
7. [模型转换详解](#7-模型转换详解)
   - 7.1 [GCN](#71-gcn)
   - 7.2 [GAT](#72-gat)
   - 7.3 [SchNet](#73-schnet)
   - 7.4 [PAiNN](#74-painn)
   - 7.5 [DimeNet++](#75-dimenetpp)
8. [数据管道转换](#8-数据管道转换)
9. [训练基础设施转换](#9-训练基础设施转换)
10. [转换中遇到的坑与解决方案](#10-转换中遇到的坑与解决方案)
11. [测试策略](#11-测试策略)
12. [新项目文件结构](#12-新项目文件结构)
13. [代码统计](#13-代码统计)

---

## 1. 项目概述

**原项目**: KGCNN v4.0.2 — 基于 Keras 3 的图神经网络库，支持 TensorFlow/PyTorch/JAX 三后端，专注于分子和材料科学。位于 `gcnn_keras-master/`。

**新项目**: kgcnn-torch v0.1.0 — 基于 PyTorch + PyTorch Geometric (PyG) 的纯 PyTorch 实现，保留 KGCNN 的模型架构和数据处理能力。位于 `kgcnn-torch/`。

**转换范围**:
- 5 个核心 GNN 模型: GCN, GAT, SchNet, PAiNN, DimeNet++
- 完整的层级库: scatter ops, gather, aggregation, convolution, geometry, attention, pooling, MLP, normalization, update, scale
- 数据管道: GraphDict, MemoryGraphList → PyG Data 转换
- 训练基础设施: 训练循环, 学习率调度器, 指标, 损失函数
- 框架无关模块: graph/, molecule/, crystal/ 直接复制

---

## 2. 为什么要转换

| 方面 | Keras 3 (KGCNN) | PyTorch + PyG (kgcnn-torch) |
|------|------------------|----------------------------|
| GNN 生态 | 有限，无专用 GNN DataLoader | PyG 是最成熟的 GNN 框架 |
| 数据表示 | 需手动处理 padded/ragged/disjoint 转换 | PyG 原生 disjoint 表示，自动 batching |
| 调试 | Keras functional API 难以逐步调试 | 标准 Python 类，可以 pdb 逐行 |
| 社区 | GNN 相关 Keras 社区较小 | 大量 PyG 论文实现可直接复用 |
| 部署 | 需要选择后端 | 标准 torch.jit / ONNX 导出 |

---

## 3. 总体架构对比

### 3.1 Keras 3 架构 (原)

```
model_inputs → CastBatchedIndicesToDisjoint → model_disjoint() → template_cast_output
                     ↓
         (padded → disjoint 转换)
                     ↓
         n, x, disjoint_indices, batch_id_node, count_nodes
                     ↓
         Embedding → Dense → [Interaction × depth] → MLP → Pooling → Output
```

Keras 模型通过 **functional API** 构建——所有层通过函数调用链接成一个计算图：

```python
# 原 KGCNN 风格 (Keras functional API)
n = Embedding(**input_node_embedding)(n)
n = Dense(units, activation='linear')(n)
for i in range(depth):
    n = SchNetInteraction(**interaction_args)([n, ed, disjoint_indices])
out = PoolingNodes(**pooling_args)([count_nodes, n, batch_id_node])
model = keras.models.Model(inputs=model_inputs, outputs=out)
```

关键特点:
- 层的 `build()` 方法支持**延迟形状推断**（不需要显式指定 input_dim）
- `get_config()` / `from_config()` 支持完整的序列化
- 需要 `CastBatchedIndicesToDisjoint` 层在 padded/ragged/disjoint 表示之间转换
- `compute_output_shape()` 需要每个自定义层手动实现

### 3.2 PyTorch + PyG 架构 (新)

```
PyG DataLoader (auto-batching) → model.forward(data) → predictions
                                        ↓
                                data.z, data.pos, data.edge_index, data.batch
                                        ↓
                                Embedding → Linear → [Interaction × depth] → MLP → Pooling → Output
```

PyTorch 模型是标准的 `nn.Module` 类:

```python
# 新 kgcnn-torch 风格 (nn.Module)
class SchNetModel(nn.Module):
    def __init__(self, node_dim, depth, units, ...):
        super().__init__()
        self.node_embedding = nn.Embedding(95, node_dim)
        self.dense_in = nn.Linear(node_dim, units)  # 必须指定 input_dim!
        self.interactions = nn.ModuleList([
            SchNetInteraction(units, edge_dim) for _ in range(depth)
        ])
        self.pooling = PoolingNodes(pooling_method="sum")

    def forward(self, data):
        n = self.node_embedding(data.z.long())
        n = self.dense_in(n)
        for interaction in self.interactions:
            n = interaction(n, ed, data.edge_index)
        out = self.pooling(n, data.batch, batch_size)
        return out
```

关键区别:
- **无需 CastBatchedIndicesToDisjoint** — PyG 的 DataLoader 自动把多个图合并为 disjoint 表示
- **`nn.Linear` 必须指定 `in_features`** — 没有 Keras 的延迟构建机制
- **没有 `get_config()`** — PyTorch 不需要，直接 `torch.save(model.state_dict(), ...)`
- **没有 `compute_output_shape()`** — 不需要静态形状推断
- **没有 `build()`** — 所有参数在 `__init__` 中创建

---

## 4. Edge Index 约定转换（核心难点）

**这是整个转换中最关键、最容易出错的部分。**

### 4.1 两种约定

| | KGCNN | PyG |
|---|---|---|
| `edge_index[0]` | **接收/目标节点 (receive/target)** | **发送/源节点 (source)** |
| `edge_index[1]` | **发送/源节点 (send/source)** | **接收/目标节点 (target)** |
| 存储格式 | `(M, 2)` numpy 数组 | `(2, M)` tensor |
| 聚合方向 | 聚合到 `edge_index[0]` | 聚合到 `edge_index[1]` |

### 4.2 KGCNN 的定义

在 `kgcnn/__init__.py` 中:
```python
__index_receive__ = 0   # edge_index 的第 0 列/行 = 接收（目标）节点
__index_send__ = 1      # edge_index 的第 1 列/行 = 发送（源）节点
```

原 KGCNN 中的 Gather:
```python
# 原 kgcnn/layers/gather.py
class GatherNodesOutgoing(Layer):  # 取"发送"节点的特征
    def call(self, inputs):
        node, edge_index = inputs
        return ops.take(node, edge_index[1], axis=0)  # edge_index[1] = source
```

原 KGCNN 中的 Aggregation:
```python
# 原 kgcnn/layers/aggr.py
class AggregateLocalEdges(Layer):  # 聚合到"接收"节点
    def call(self, inputs):
        node, edges, edge_index = inputs
        return scatter_reduce(edge_index[0], edges, ...)  # edge_index[0] = target
```

### 4.3 PyG 的定义

PyG 中 `edge_index[0]` = source, `edge_index[1]` = target：
```python
# PyG 约定
src = edge_index[0]  # 源节点（消息发出方）
tgt = edge_index[1]  # 目标节点（消息接收方）
# 消息从 src → tgt 流动，聚合到 tgt
```

### 4.4 转换实现

**在 `to_pyg_list()` 中一次性交换**（`data/base.py:178-182`）:

```python
def to_pyg_list(self, ...):
    for g in self:
        edges = g.obtain_property(edge_key)
        if edges is not None:
            edges = np.asarray(edges)
            if edges.ndim == 2 and edges.shape[1] == 2:
                # KGCNN (M, 2): [target, source] → PyG (2, M): [source, target]
                ei = edges[:, [1, 0]].T  # 先交换列，再转置
            data_dict['edge_index'] = torch.tensor(ei, dtype=torch.long)
```

**交换之后，全部代码统一使用 PyG 约定**:

```python
# kgcnn_torch/layers/gather.py
def gather_nodes_outgoing(x, edge_index):
    return x[edge_index[0]]   # source 节点特征（PyG 约定）

def gather_nodes_ingoing(x, edge_index):
    return x[edge_index[1]]   # target 节点特征（PyG 约定）

# kgcnn_torch/layers/aggr.py
class AggregateLocalEdges(nn.Module):
    def forward(self, edges, edge_index, num_nodes):
        target_idx = edge_index[1]   # 聚合到 target（PyG 约定）
        return self._aggregate(edges, target_idx, num_nodes)
```

### 4.5 对比表

| 操作 | KGCNN 实现 | kgcnn-torch 实现 |
|------|-----------|-----------------|
| 取源节点特征 | `node[edge_index[1]]` | `node[edge_index[0]]` |
| 取目标节点特征 | `node[edge_index[0]]` | `node[edge_index[1]]` |
| 消息聚合 | `scatter(edge_index[0], ...)` | `scatter(edge_index[1], ...)` |
| 存储形状 | `(M, 2)` numpy | `(2, M)` torch.long |

---

## 5. 基础操作层转换

### 5.1 Scatter 操作

**原** (`kgcnn/ops/scatter.py`): 使用 Keras backend 抽象层，底层调用各后端的 scatter 实现。

**新** (`kgcnn_torch/ops/scatter.py`): 直接使用 PyTorch 原生 `torch.scatter_reduce`。

```python
# 核心实现
def scatter_reduce_sum(indices, values, dim_size):
    idx = _expand_indices(indices, values)  # 将 1D 索引扩展到 values 的维度
    out = torch.zeros(dim_size, *values.shape[1:], dtype=values.dtype, device=values.device)
    return out.scatter_reduce(0, idx, values, reduce='sum')
```

提供 5 种操作: `sum`, `mean`, `max`, `min`, `softmax`。

softmax 的实现比较特殊——需要先做 scatter_max 减去最大值（数值稳定），再 exp，再 scatter_sum 归一化:

```python
def scatter_reduce_softmax(indices, values, dim_size, normalize=True):
    if normalize:
        out_max = zeros.scatter_reduce(0, idx, values, reduce='amax')
        values = values - out_max[indices]  # 数值稳定性
    values_exp = torch.exp(values)
    out_sum = zeros.scatter_reduce(0, idx, values_exp, reduce='sum')
    return values_exp / out_sum[indices]
```

### 5.2 自定义激活函数

**原** (`kgcnn/ops/activ.py`): 定义为纯函数，通过 Keras 的 `@keras.saving.register_keras_serializable` 装饰器注册。

**新** (`kgcnn_torch/ops/activ.py`): 定义**函数 + Module 包装器 + 注册表**三层结构。

```python
# 1. 纯函数
def shifted_softplus(x):
    return F.softplus(x) - math.log(2.0)

# 2. Module 包装器
class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return shifted_softplus(x)

# 3. 注册表 + 工厂函数
_ACTIVATION_REGISTRY = {
    "shifted_softplus": ShiftedSoftplus,
    "swish": Swish, "relu": nn.ReLU, "silu": nn.SiLU, ...
}

def get_activation(name: str) -> nn.Module:
    """每次调用返回一个新的 Module 实例!"""
    if name is None or name == "linear":
        return nn.Identity()
    return _ACTIVATION_REGISTRY[key]()  # 注意这里的 ()，创建新实例
```

> **关键**: `get_activation()` 必须每次返回**新实例**。PyTorch 的 `nn.Module` 不能被多个父 Module 共享——如果一个 `nn.ReLU()` 实例被放入两个不同的 `nn.Sequential`，它只会被注册到最后一个，导致另一个的参数丢失。详见[第 10 节的坑](#10-转换中遇到的坑与解决方案)。

### 5.3 权重初始化器

**原** (`kgcnn/initializers/initializers.py`): 继承 `keras.initializers.Initializer`，通过 `kernel_initializer="kgcnn>glorot_orthogonal"` 字符串引用。

**新** (`kgcnn_torch/initializers/initializers.py`): 纯函数式就地初始化。

```python
def glorot_orthogonal_(tensor, scale=2.0):
    """Glorot-orthogonal 初始化 (DimeNet++ 使用)。"""
    with torch.no_grad():  # 必须! 否则会报 "in-place operation on leaf variable" 错误
        nn.init.orthogonal_(tensor)
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        s = math.sqrt(scale / ((fan_in + fan_out) * torch.var(tensor).item()))
        tensor.mul_(s)
    return tensor
```

使用方式:
```python
# 原 KGCNN
Dense(emb_size, kernel_initializer="kgcnn>glorot_orthogonal")

# 新 kgcnn-torch
linear = nn.Linear(emb_size, emb_size)
glorot_orthogonal_(linear.weight)
if linear.bias is not None:
    nn.init.zeros_(linear.bias)
```

---

## 6. 核心层转换详解

### 6.1 Gather 层

**原**: `GatherNodesOutgoing`, `GatherNodesIngoing` 是 Keras Layer 类，有 `build()`, `call()`, `get_config()`。

**新**: **纯函数**（不是 Module），因为没有可训练参数:

```python
# kgcnn_torch/layers/gather.py
def gather_nodes_outgoing(x, edge_index):
    """取源节点特征。"""
    return x[edge_index[0]]  # PyG: edge_index[0] = source

def gather_nodes_ingoing(x, edge_index):
    """取目标节点特征。"""
    return x[edge_index[1]]  # PyG: edge_index[1] = target

def gather_nodes(x, edge_index):
    """拼接源和目标节点特征。"""
    return torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
```

### 6.2 聚合层 (Aggregation)

**原**: `AggregateLocalEdges` 接收 `[node, edges, edge_index]` 三元组。

**新**: `AggregateLocalEdges` 简化为 `(edges, edge_index, num_nodes)`:

```python
# 原 KGCNN
out = AggregateLocalEdges(pooling_method="sum")([node, edges, edge_index])
# node 在原始代码中其实只用于获取 dim_size

# 新 kgcnn-torch
out = AggregateLocalEdges(pooling_method="sum")(edges, edge_index, num_nodes)
```

还额外提供了 `Aggregate` — 通用索引聚合（不绑定到 edge_index 语义），DimeNet++ 的 edge→edge 聚合需要用到:

```python
class Aggregate(nn.Module):
    def forward(self, values, indices, dim_size):
        return self._pool_fn(indices, values, dim_size)
```

### 6.3 图卷积层 (Convolution)

#### GCNConv

**原** (`kgcnn/layers/conv.py` 的 `GCN` 类): 一个 Keras Layer，内部包含 `Dense` + `GatherNodesOutgoing` + `AggregateWeightedLocalEdges`。

**新**: 独立的 `nn.Module`:

```python
class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, ...):
        self.linear = nn.Linear(in_features, out_features)  # 必须显式指定 in_features
        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)

    def forward(self, x, edge_index, edge_weight):
        x_trans = self.linear(x)
        x_j = gather_nodes_outgoing(x_trans, edge_index)  # 源节点
        messages = x_j * edge_weight                        # 加权
        out = self.aggr(messages, edge_index, num_nodes)    # 聚合到目标
        return self.activation(out)
```

#### SchNetInteraction

结构几乎一致，核心区别是 Keras 的 `Dense` → PyTorch 的 `nn.Linear` (需要 in_features):

```python
# 原 KGCNN
self.lay_cfconv = SchNetCFconv(**cfconv_pool)  # 内部的 Dense 不需要指定输入维度

# 新 kgcnn-torch
self.cfconv = SchNetCFconv(edge_dim, units, ...)  # 必须传入 edge_dim
```

### 6.4 几何层 (Geometry)

这是**最复杂的层文件** (`kgcnn/layers/geom.py` 原文件 49KB)。

#### 距离和方向计算

**原**: 用 Keras Layer 封装 (`NodePosition`, `NodeDistanceEuclidean`, `EdgeDirectionNormalized`)。

**新**: 用纯函数:

```python
def compute_edge_distances(pos, edge_index, eps=True):
    diff = pos[edge_index[0]] - pos[edge_index[1]]
    dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
    if eps:
        dist_sq = dist_sq + torch.finfo(dist_sq.dtype).eps
    return torch.sqrt(dist_sq)
```

#### GaussBasisLayer (SchNet 用)

```python
# 原 KGCNN — 使用 keras ops
class GaussBasisLayer(Layer):
    def call(self, inputs):
        return ops.exp(-self.gamma * ops.square(inputs - self.offset - self.centers))

# 新 kgcnn-torch — 使用 register_buffer 存储非参数常量
class GaussBasisLayer(nn.Module):
    def __init__(self, bins, distance, sigma, offset):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, distance, bins))
        self.gamma = 1.0 / (sigma * sigma * 2.0)

    def forward(self, dist):
        return torch.exp(-self.gamma * (dist - self.offset - self.centers).pow(2))
```

#### BesselBasisLayer (DimeNet 用)

**原**: 频率通过 `add_weight()` 注册为可训练权重。

**新**: 使用 `nn.Parameter`:

```python
class BesselBasisLayer(nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent):
        freq = torch.arange(1, num_radial + 1, dtype=torch.float) * math.pi
        self.frequencies = nn.Parameter(freq)  # 可训练

    def forward(self, dist):
        d_scaled = dist / self.cutoff
        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)
```

#### SphericalBasisLayer (DimeNet 用)

最复杂的基函数层。使用 scipy 预计算 spherical bessel 零点和归一化因子，存为 buffer:

```python
class SphericalBasisLayer(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, ...):
        # 预计算 (numpy) → register_buffer
        bessel_zeros = spherical_bessel_jn_zeros(num_spherical, num_radial)  # scipy
        self.register_buffer("bessel_zeros", torch.tensor(bessel_zeros))
        self.register_buffer("bessel_norm", torch.tensor(bessel_norm))

    def forward(self, dist, angles, angle_index):
        # 1. 径向部分: spherical bessel functions (PyTorch 递推实现)
        rbf = torch_spherical_bessel_jn(d_scaled * self.bessel_zeros[n, k], n)
        # 2. 角度部分: spherical harmonics Y_l(cos(theta))
        cbf = torch_spherical_harmonics_yl(angles, n)
        # 3. 径向 × 角度
        return rbf_env * cbf
```

### 6.5 注意力层 (Attention)

#### AttentionHeadGAT

逻辑完全一致，API 区别:

```python
# 原 KGCNN — 各子层是 Keras Layer 实例
class AttentionHeadGAT(Layer):
    def __init__(self, ...):
        self.lay_linear_trafo = Dense(units, ...)
        self.lay_alpha = Dense(1, ...)
        self.gather_n = GatherNodesOutgoing()
        self.pool_attention = AggregateLocalEdgesAttention()
    def call(self, inputs):
        node, edge, edge_index = inputs  # 解包列表输入
        ...

# 新 kgcnn-torch — 使用 nn.Linear + 纯函数
class AttentionHeadGAT(nn.Module):
    def __init__(self, in_features, units, ...):
        self.linear_trafo = nn.Linear(in_features, units)  # 必须 in_features
        self.linear_alpha = nn.Linear(concat_dim, 1)
        self.pool_attention = AggregateLocalEdgesAttention()
    def forward(self, x, edge_index, edge_attr=None):
        n_i = gather_nodes_ingoing(x, edge_index)   # 函数调用，不是 Layer
        n_j = gather_nodes_outgoing(x, edge_index)
        ...
```

### 6.6 池化层 (Pooling)

**原**: 接收 `[count_nodes, n, batch_id_node]` 三元组。

**新**: 使用 `(x, batch, batch_size)` — 更符合 PyG 风格:

```python
# 原 KGCNN
out = PoolingNodes(pooling_method="sum")([count_nodes, n, batch_id_node])

# 新 kgcnn-torch
out = PoolingNodes(pooling_method="sum")(n, batch, batch_size)
```

### 6.7 MLP 层

**最大区别: PyTorch 的 `nn.Linear` 必须指定 `input_dim`。**

```python
# 原 KGCNN — 不需要 input_dim，Keras 会自动推断
mlp = MLP(units=[128, 64], activation="relu")

# 新 kgcnn-torch — 必须传入 input_dim
mlp = MLP(units=[128, 64], input_dim=256, activation="relu")
```

实现上构建一个 `nn.Sequential`，依次放 Linear + (Dropout) + (BatchNorm) + Activation:

```python
class MLP(nn.Module):
    def __init__(self, units, input_dim, activation="relu", ...):
        layers = []
        in_dim = input_dim
        for out_dim in units:
            layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            if use_normalization:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(get_activation(activation))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
```

### 6.8 归一化层 (Normalization)

**原**: `GraphBatchNormalization` 和 `GraphNormalization` 都接收 `[x, batch_id, count]`。

**新**: `GraphBatchNorm` 直接包装 `nn.BatchNorm1d`（因为 disjoint 表示下 `(N, F)` 可以直接用），`GraphNormalization` 手动实现 per-graph 统计:

```python
class GraphNormalization(nn.Module):
    """Per-graph normalization (不是 per-batch!)"""
    def forward(self, x, batch, batch_size=None):
        # 计算每个图的均值和方差
        counts = scatter_reduce_sum(batch, ones, batch_size)
        mean = scatter_reduce_sum(batch, x, batch_size) / counts
        var = scatter_reduce_sum(batch, (x - mean[batch]).pow(2), batch_size) / counts
        return (x - mean[batch]) / (var + eps).sqrt()[batch] * gamma + beta
```

### 6.9 更新层 (Update)

**GRUUpdate**: 直接包装 `nn.GRUCell`:

```python
# 原 KGCNN
class GRUUpdate(Layer):
    def __init__(self, ...):
        self.gru = keras.layers.GRUCell(units)
    def call(self, inputs):
        message, hidden = inputs
        return self.gru(message, [hidden])[0]

# 新 kgcnn-torch
class GRUUpdate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)  # 需要 input_dim
    def forward(self, message, hidden):
        return self.gru_cell(message, hidden)
```

**ResidualLayer**: `output = activation(linear(x)) + x`:

```python
class ResidualLayer(nn.Module):
    def __init__(self, units, activation="swish"):
        self.linear = nn.Linear(units, units)
        self.activation = get_activation(activation)
    def forward(self, x):
        return x + self.activation(self.linear(x))
```

### 6.10 缩放层 (Scale)

**原**: Keras Layer 使用 `add_weight()` 存储均值和标准差。

**新**: 使用 `register_buffer` 存储（非参数，但随模型保存/迁移）:

```python
class StandardLabelScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean_", torch.tensor(0.0))
        self.register_buffer("std_", torch.tensor(1.0))

    def fit(self, y: np.ndarray):
        self.mean_ = torch.tensor(np.mean(y, axis=0))
        self.std_ = torch.tensor(np.std(y, axis=0))

    def transform(self, y: torch.Tensor):
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y_scaled: torch.Tensor):
        return y_scaled * self.std_ + self.mean_
```

---

## 7. 模型转换详解

### 7.1 GCN

**论文**: Kipf & Welling, 2017

**最简单的模型。** 结构:

```
Atomic Numbers → Embedding(95, node_dim)
    → Linear(node_dim, gcn_units)
    → [GCNConv(gcn_units, gcn_units) × depth]
    → PoolingNodes("sum")
    → MLP → num_targets
```

**原 vs 新的关键区别**:
1. 输入: 原接收 `[n, e, disjoint_indices, batch_id, count_nodes]` 列表 → 新直接接收 PyG `Data` 对象
2. 不需要 `CastBatchedIndicesToDisjoint` — PyG DataLoader 已经处理了

```python
# 新 GCNModel.forward()
def forward(self, data):
    x = data.z
    edge_index = data.edge_index
    edge_weight = data.edge_weight
    batch = data.batch
    x = self.node_embedding(x.long())
    x = self.dense_in(x)
    for conv in self.convs:
        x = conv(x, edge_index, edge_weight)
    batch_size = int(batch.max().item()) + 1
    out = self.pooling(x, batch, batch_size)
    out = self.output_mlp(out)
    return out
```

### 7.2 GAT

**论文**: Velickovic et al., 2018

**多头注意力。** 结构:

```
Atomic Numbers → Embedding → Linear(node_dim, units)
    → [Multi-Head Attention × depth]
        每层: heads = [AttentionHeadGAT(in_dim, units) × num_heads]
              if concat: x = cat(heads, dim=-1)  → out_dim = units * num_heads
              else:      x = mean(heads)          → out_dim = units
    → PoolingNodes → MLP → num_targets
```

**Concat 模式的维度变化**: 第一层输入是 `units`，输出是 `units * num_heads`。从第二层开始，输入变成 `units * num_heads`。所以每层的 `in_features` 不同:

```python
for i in range(depth):
    in_dim = attention_units * num_heads if (i > 0 and concat) else attention_units
    heads = nn.ModuleList([
        AttentionHeadGAT(in_features=in_dim, units=attention_units, ...)
        for _ in range(num_heads)
    ])
```

### 7.3 SchNet

**论文**: Schutt et al., 2017

结构:

```
Atomic Numbers → Embedding(95, node_dim)
    → Linear(node_dim, units)
    → [SchNetInteraction(units, edge_dim) × depth]
        SchNetInteraction:
            x → Dense(linear) → CFconv(x, gauss_basis(dist), edge_index) → Dense(act) → Dense(linear) → + x
    → MLP(last_mlp_units)
    → PoolingNodes("sum")
    → MLP(output_units + [num_targets])
```

**周期性体系支持**: SchNet 和 PAiNN 都支持晶体系统（需要 `edge_image` 和 `lattice`）:

```python
# 在 forward() 中处理周期性
if hasattr(data, 'edge_image') and data.edge_image is not None:
    pos_j = pos[edge_index[0]]
    pos_j = shift_periodic_lattice(pos_j, data.edge_image, data.lattice, batch_edge)
    pos_i = pos[edge_index[1]]
    diff = pos_j - pos_i
    dist = torch.sqrt((diff * diff).sum(dim=-1, keepdim=True) + 1e-8)
else:
    dist = compute_edge_distances(pos, edge_index)
```

### 7.4 PAiNN

**论文**: Schutt et al., 2021

**最复杂的等变模型。** 维护两套特征:
- **标量 z**: `(N, F)` — 普通节点特征
- **向量 v**: `(N, 3, F)` — 等变向量特征（3 是空间维度）

结构:

```
Atomic Numbers → Embedding(95, node_dim) → z: (N, F)
EquivariantInitialize → v: (N, 3, F) = zeros

pos → BesselBasis(dist) → rbf: (M, num_radial)
pos → CosCutOff(dist)   → env: (M, 1)
pos → NormalizedDir      → rij: (M, 3)

for i in range(depth):
    # 消息传递
    ds, dv = PAiNNConv(z, v, rbf, env, rij, edge_index)
    z = z + ds
    v = v + dv
    # 更新
    ds, dv = PAiNNUpdate(z, v)
    z = z + ds
    v = v + dv
    # 可选归一化
    if equiv_normalization: v = LayerNorm(v)
    if node_normalization: z = BatchNorm(z)

out = PoolingNodes(z, batch) → MLP → num_targets
```

#### PAiNNConv 详解

核心等变消息传递:

```python
class PAiNNConv(nn.Module):
    def forward(self, z, v, rbf, envelope, rij, edge_index):
        # 1. 标量处理: z → Dense → act → Dense(3F)
        s = self.act1(self.dense1(z))    # (N, F)
        s = self.phi(s)                   # (N, 3F)
        s = gather_nodes_outgoing(s, edge_index)  # (M, 3F) 取源节点

        # 2. RBF 滤波: rbf → Dense(3F) × envelope
        w = self.w(rbf)                   # (M, 3F) — 注意这里输入是 num_radial 维!
        w = w * envelope                  # 乘截断包络
        sw = s * w                        # (M, 3F) 逐元素

        # 3. 拆分为 3 个通道
        sw1, sw2, sw3 = torch.chunk(sw, 3, dim=-1)  # 每个 (M, F)

        # 4. 标量更新: ds = aggregate(sw1)
        ds = self.aggr_s(sw1, edge_index, num_nodes)

        # 5. 向量更新: dv = aggregate(sw2 * vj + sw3 * rij)
        vj = gather_nodes_outgoing(v, edge_index)   # (M, 3, F)
        dv1 = sw2.unsqueeze(1) * vj                 # (M, 3, F)
        dv2 = sw3.unsqueeze(1) * rij.unsqueeze(2)   # (M, 3, F)
        dv = dv1 + dv2
        # 聚合 (M, 3, F) → (N, 3, F) 需要先 reshape
        dv_flat = dv.reshape(M, 3 * F)
        dv_agg = self.aggr_v(dv_flat, edge_index, num_nodes)
        dv = dv_agg.reshape(num_nodes, 3, F)

        return ds, dv
```

#### PAiNNUpdate 详解

标量-向量交互:

```python
class PAiNNUpdate(nn.Module):
    def forward(self, z, v):
        v_v = self.lin_v(v)               # (N, 3, F) — Linear 作用在最后一维
        v_u = self.lin_u(v)               # (N, 3, F)

        v_prod = (v_u * v_v).sum(dim=1)   # (N, F) 标量积
        v_norm = sqrt(sum(v_v^2, dim=1))  # (N, F) 范数

        a = cat([z, v_norm], dim=-1)       # (N, 2F)
        a = act(Dense(a))                  # (N, F)
        a = Dense(a)                       # (N, 3F)
        a_vv, a_sv, a_ss = chunk(a, 3)    # 每个 (N, F)

        dv = a_vv.unsqueeze(1) * v_u       # (N, 3, F) 向量更新
        ds = v_prod * a_sv + a_ss          # (N, F) 标量更新
        return ds, dv
```

### 7.5 DimeNet++

**论文**: Klicpera et al., 2020

**三体模型**——不仅使用节点对（边），还使用边对（角度/三元组）。

结构:

```
Atomic Numbers → EmbeddingDimeBlock(95, emb_size)

pos → BesselBasis(dist)   → rbf: (M, num_radial)
pos → SphericalBasis(dist, angles, angle_index) → sbf: (K, S*R)
pos → compute_angles(v12, angle_index) → angles: (K, 1)

# 初始嵌入
rbf_emb = Dense(rbf)                                           # (M, emb_size)
n_pairs = gather_nodes(n, edge_index)                           # (M, 2*emb_size)
x = Dense(cat([n_pairs, rbf_emb]))                             # (M, emb_size)
ps = DimNetOutputBlock(n, x, rbf, edge_index)                   # (N, num_targets)

for i in range(num_blocks):
    x = DimNetInteractionPPBlock(x, rbf, sbf, angle_index)     # (M, emb_size)
    p_update = DimNetOutputBlock(n, x, rbf, edge_index)         # (N, num_targets)
    ps = ps + p_update                                          # 残差累加

out = PoolingNodes(ps, batch) → [Optional MLP] → num_targets
```

#### DimNetInteractionPPBlock 详解

```python
def forward(self, x, rbf, sbf, angle_index):
    x_ji = Dense(x)            # (M, emb_size)  — ji 边变换
    x_kj = Dense(x)            # (M, emb_size)  — kj 边变换

    # 径向基变换
    rbf_w = Dense(Dense(rbf))   # (M, emb_size)
    x_kj = x_kj * rbf_w

    # 下投影 + 取三元组
    x_kj = Dense(x_kj)         # (M, int_emb_size) — 降维
    x_kj = x_kj[angle_index[0]]  # (K, int_emb_size) — 按三元组索引取

    # 球面基变换
    sbf_w = Dense(Dense(sbf))   # (K, int_emb_size)
    x_kj = x_kj * sbf_w

    # 关键: edge→edge 聚合 (不是 edge→node!)
    x_kj_agg = Aggregate(x_kj, angle_index[1], num_edges)  # (M, int_emb_size)

    # 上投影
    x_kj_agg = Dense(x_kj_agg)  # (M, emb_size) — 升维

    # 残差
    x2 = x_ji + x_kj_agg
    for layer in before_skip_residuals:
        x2 = layer(x2)
    x = x + Dense(x2)          # skip connection
    for layer in after_skip_residuals:
        x = layer(x)
    return x
```

> **关键**: DimeNet++ 的 InteractionBlock 中的聚合是**边→边**（三元组 K 聚合回边 M），不是通常的边→节点。所以这里用的是 `Aggregate`（通用聚合），而不是 `AggregateLocalEdges`（专门按 edge_index[1] 聚合到目标节点）。

#### 角度计算

原 KGCNN 有专门的 `EdgeAngle` 层，新实现直接在模型 forward 中计算:

```python
# 向量 v12 = pos_target - pos_source（指向目标的方向）
v12 = pos_i - pos_j  # (M, 3)

# 取角度对应的两个边向量
vec_a = v12[angle_index[0]]  # (K, 3)
vec_b = v12[angle_index[1]]  # (K, 3)

# 余弦→角度
cos_angle = (vec_a * vec_b).sum(dim=-1) / (norm(vec_a) * norm(vec_b) + 1e-8)
angles = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
```

---

## 8. 数据管道转换

### 8.1 保留的部分

**GraphDict** 和 **MemoryGraphList** 几乎原样保留——它们是纯 Python/numpy 数据结构，与框架无关。

**graph/**, **molecule/**, **crystal/** 三个目录直接复制——它们使用 numpy、RDKit、pymatgen 等，不依赖任何深度学习框架。

### 8.2 添加的 `to_pyg_list()` 方法

这是数据管道中最重要的新增方法，将 KGCNN 格式的 numpy 图转换为 PyG Data 对象:

```python
def to_pyg_list(self, node_key="node_number", pos_key="node_coordinates",
                edge_key="edge_indices", label_key="graph_labels", ...):
    pyg_list = []
    for g in self:
        data_dict = {}

        # 节点特征
        nodes = g.obtain_property(node_key)
        data_dict['z'] = torch.tensor(nodes, dtype=torch.long)

        # 坐标
        pos = g.obtain_property(pos_key)
        data_dict['pos'] = torch.tensor(pos, dtype=torch.float)

        # ★ 核心: 交换 edge index 约定
        edges = g.obtain_property(edge_key)
        ei = edges[:, [1, 0]].T   # KGCNN [target, source] → PyG [source, target]
        data_dict['edge_index'] = torch.tensor(ei, dtype=torch.long)

        # 标签
        labels = g.obtain_property(label_key)
        data_dict['y'] = torch.tensor(labels, dtype=torch.float)

        # 晶体相关
        lattice = g.obtain_property("graph_lattice")
        if lattice is not None:
            data_dict['lattice'] = torch.tensor(lattice, dtype=torch.float)
        edge_image = g.obtain_property("range_image")
        if edge_image is not None:
            data_dict['edge_image'] = torch.tensor(edge_image, dtype=torch.float)

        # 角度索引 (DimeNet++)
        angle_idx = g.obtain_property("angle_indices")
        if angle_idx is not None:
            data_dict['angle_index'] = torch.tensor(angle_idx.T, dtype=torch.long)

        pyg_list.append(Data(**data_dict))
    return pyg_list
```

### 8.3 KGCNN 属性名 → PyG 属性名映射

| KGCNN 属性名 | PyG Data 属性名 | 说明 |
|---|---|---|
| `node_number` | `data.z` | 原子序数 |
| `node_coordinates` | `data.pos` | 原子坐标 |
| `edge_indices` | `data.edge_index` | 边索引 **(已交换)** |
| `graph_labels` | `data.y` | 图级标签 |
| `graph_lattice` | `data.lattice` | 晶格矩阵 |
| `range_image` | `data.edge_image` | 周期性像 |
| `angle_indices` | `data.angle_index` | 角度/三元组索引 |
| `edge_weight` | `data.edge_weight` | 边权重 |

### 8.4 PyG InMemoryDataset 封装

为每种数据集类型提供 PyG InMemoryDataset 包装:

```python
# kgcnn_torch/data/qm.py
class QMDataset(InMemoryDataset):
    def process(self):
        # 1. 使用 KGCNN 的 MemoryGraphDataset 加载原始数据
        kgcnn_dataset = MemoryGraphDataset(...)
        kgcnn_dataset.prepare_data()
        # 2. 转换为 PyG 格式
        data_list = kgcnn_dataset.to_pyg_list()
        # 3. 标准 PyG 保存
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```

### 8.5 Transform (数据变换)

`StandardScaler` 和 `ExtensiveMolecularScaler` 保持纯 numpy 实现（用于 `fit/transform` 预处理），与框架无关:

```python
class StandardScaler:
    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def save(self, path: str):  # JSON 序列化
        json.dump({"mean_": self.mean_.tolist(), ...}, ...)

    def load(self, path: str):
        data = json.load(...)
        self.mean_ = np.array(data["mean_"])
```

---

## 9. 训练基础设施转换

### 9.1 从 Keras `model.compile/fit` 到 PyTorch 手动训练循环

**原 KGCNN**:
```python
model = make_model(**hyper.model_config)
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))
```

**新 kgcnn-torch**:
```python
model = SchNetModel(**config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

history = fit(
    model, train_loader, val_loader,
    optimizer=optimizer, loss_fn=loss_fn,
    scheduler=scheduler, epochs=100,
    early_stopping_patience=50,
    checkpoint_path="best_model.pt"
)
```

### 9.2 训练循环

`trainer.py` 提供三个核心函数:

```python
def train_epoch(model, loader, optimizer, loss_fn, device):
    """单个训练 epoch。"""
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y.unsqueeze(-1))
        loss.backward()
        optimizer.step()

def eval_epoch(model, loader, loss_fn, device, metrics):
    """单个验证 epoch（@torch.no_grad()）。"""
    model.eval()
    # 收集所有预测和目标，计算全局指标

def fit(model, train_loader, val_loader, ...):
    """完整训练循环 + 早停 + checkpoint + LR 调度。"""
```

### 9.3 学习率调度器

**原** (`kgcnn/training/scheduler.py`): 使用 Keras 的 `keras.callbacks.LearningRateScheduler` 或自定义 callback。

**新** (`kgcnn_torch/training/scheduler.py`): 继承 `torch.optim.lr_scheduler.LambdaLR`:

```python
class LinearWarmupExponentialDecay(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, decay_steps, decay_rate):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return decay_rate ** ((step - warmup_steps) / decay_steps)
        super().__init__(optimizer, lr_lambda)

# 工厂函数
scheduler = get_scheduler("warmup_exponential", optimizer,
                          warmup_steps=100, decay_steps=1000, decay_rate=0.96)
```

### 9.4 指标和损失函数

```python
# 指标
class ScaledMAE:
    """使用 scaler 反变换后计算 MAE。"""
    def __call__(self, pred, target):
        pred_inv = self.scaler.inverse_transform(pred)
        target_inv = self.scaler.inverse_transform(target)
        return torch.mean(torch.abs(pred_inv - target_inv))

# 损失函数
class EnergyForceLoss(nn.Module):
    """联合能量 + 力损失: w_e * MSE(energy) + w_f * MAE(forces)"""
    def forward(self, pred_energy, target_energy, pred_forces, target_forces):
        e_loss = F.mse_loss(pred_energy, target_energy)
        f_loss = F.l1_loss(pred_forces, target_forces)
        return self.energy_weight * e_loss + self.force_weight * f_loss
```

---

## 10. 转换中遇到的坑与解决方案

### 坑 1: `nn.Module` 共享实例

**问题**: 在 DimeNet++ 的 InteractionBlock 中，多个 `nn.Sequential` 共享了同一个 activation module 实例:

```python
# 错误写法
act = get_activation("swish")  # 只创建了一个实例
self.dense_ji = nn.Sequential(nn.Linear(emb, emb), act)     # act 注册到 dense_ji
self.dense_kj = nn.Sequential(nn.Linear(emb, emb), act)     # act 被移到 dense_kj!
# 此时 dense_ji 中的 activation 丢失了!
```

**原因**: PyTorch 的 `nn.Module` 只能有一个父 Module。把同一个实例放到两个不同的 Sequential 中，它会被自动从第一个中移除。

**解决**: 每次都创建新实例:

```python
# 正确写法
self.dense_ji = nn.Sequential(nn.Linear(emb, emb), get_activation("swish"))
self.dense_kj = nn.Sequential(nn.Linear(emb, emb), get_activation("swish"))
```

### 坑 2: `glorot_orthogonal_` 就地操作

**问题**: 直接对 `nn.Parameter` 做就地操作会报错:

```python
# 错误
def glorot_orthogonal_(tensor):
    nn.init.orthogonal_(tensor)    # RuntimeError: a leaf Variable that requires grad
    tensor.mul_(s)                 # is being used in an in-place operation
```

**原因**: `nn.Parameter` 默认 `requires_grad=True`，PyTorch 不允许对需要梯度的叶节点做就地操作（会破坏自动求导的计算图）。

**解决**: 包裹在 `torch.no_grad()` 中:

```python
def glorot_orthogonal_(tensor):
    with torch.no_grad():          # 禁用梯度追踪
        nn.init.orthogonal_(tensor)
        tensor.mul_(s)
```

### 坑 3: PAiNN 的 `self.w` 输入维度

**问题**: PAiNNConv 中 `self.w` 接收 RBF 输出（维度 `num_radial`），但误写成了 `units`:

```python
# 错误
self.w = nn.Linear(units, units * 3)  # 输入应该是 rbf 的维度!

# 正确
self.w = nn.Linear(num_radial, units * 3)  # num_radial = RBF 基函数数量
```

**原因**: 原 KGCNN 使用 Keras 的延迟构建，不需要指定 input_dim，所以源代码中看不出输入维度。转换时必须手动追踪每个层的输入张量形状。

### 坑 4: DimeNet++ 的 edge→edge 聚合

**问题**: `AggregateLocalEdges` 默认聚合到 `edge_index[1]`（目标节点），但 DimeNet++ 的 InteractionBlock 需要从三元组 (K) 聚合回边 (M)，不涉及 node-level 的 edge_index。

```python
# 错误: 使用 AggregateLocalEdges
self.lay_pool = AggregateLocalEdges(pooling_method="sum")
# angle_index 是 (2, K)，不是 edge_index!
x_kj = self.lay_pool(x_kj, angle_index, num_edges)  # 语义不对!

# 正确: 使用通用 Aggregate
self.aggr = Aggregate(pooling_method="sum")
x_kj_agg = self.aggr(x_kj, angle_index[1], num_edges)  # 明确按 angle_index[1] 聚合
```

### 坑 5: DimNetOutputBlock 的 `dense_rbf` 输入维度

**问题**: OutputBlock 中 `dense_rbf` 接收的是原始 RBF（维度 `num_radial`），而不是 `emb_size`:

```python
# 错误
self.dense_rbf = nn.Linear(emb_size, emb_size)

# 正确
self.dense_rbf = nn.Linear(num_radial, emb_size)
```

### 坑 6: 测试中 non-leaf tensor 的梯度

**问题**: `torch.randn(...).abs()` 生成的不是叶节点，无法获取 `.grad`:

```python
# 错误
dist = torch.randn(5, 1).abs()  # abs() 创建了计算图，dist 不是叶节点
dist.requires_grad_(True)        # 设置 requires_grad 但仍然不是叶节点
result = layer(dist)
result.sum().backward()
assert dist.grad is not None     # 失败! non-leaf 节点的 grad 被释放了

# 正确
dist = torch.tensor([[0.5], [1.5], [2.5]], requires_grad=True)  # 直接创建叶节点
```

### 坑 7: `nn.Linear` 对高维输入的行为

**发现**: `nn.Linear(F_in, F_out)` 在输入 `(N, 3, F_in)` 时自动作用在最后一维，输出 `(N, 3, F_out)`。这使得 PAiNN 中对 `v: (N, 3, F)` 的变换可以直接用 `nn.Linear`:

```python
v_v = self.lin_v(v)  # v: (N, 3, F) → v_v: (N, 3, F)
# nn.Linear 自动广播到 batch dimensions
```

---

## 11. 测试策略

### 11.1 测试文件

| 文件 | 测试数 | 覆盖内容 |
|---|---|---|
| `test_ops_scatter.py` | 7 | scatter sum/mean/max/min/softmax, empty scatter, gradient flow |
| `test_layers.py` | 21 | gather, aggregation, geom, conv, attention, pooling, mlp, norm, update, scale |
| `test_models.py` | 10 | 5 模型 × 2 (基本 forward+backward + 参数变体) |
| `test_data_pipeline.py` | 8 | GraphDict, MemoryGraphList, to_pyg_list, DataLoader, transform |
| **总计** | **47** | |

### 11.2 测试原则

1. **形状测试**: 验证每个层的输出形状正确
2. **梯度测试**: 验证梯度可以流过（`result.sum().backward()` 后检查 `.grad is not None`）
3. **数值测试**: 对简单情况验证数值正确（如 scatter_sum 对已知输入的结果）
4. **端到端测试**: model forward → backward 不报错
5. **DataLoader 测试**: to_pyg_list → DataLoader → batch 形状正确

### 11.3 运行测试

```bash
conda activate SemMO
cd /home/yuanbai/Downloads/MLIPs/kgcnn-torch
python -m pytest tests/ -v
# ======================== 47 passed, 3 warnings in 3.23s ========================
```

---

## 12. 新项目文件结构

```
kgcnn-torch/
├── pyproject.toml                      # 项目配置 (torch>=2.0, pyg>=2.4)
├── requirements.txt
├── CONVERSION_GUIDE.md                 # 本文档
│
├── kgcnn_torch/
│   ├── __init__.py
│   │
│   ├── ops/                            # 基础操作
│   │   ├── scatter.py                  # scatter_reduce_sum/mean/max/min/softmax
│   │   └── activ.py                    # 自定义激活函数 + 注册表
│   │
│   ├── initializers/
│   │   └── initializers.py             # glorot_orthogonal_ (DimeNet++ 用)
│   │
│   ├── layers/                         # 核心层
│   │   ├── gather.py                   # 纯函数: gather_nodes_outgoing/ingoing/nodes
│   │   ├── aggr.py                     # Aggregate, AggregateLocalEdges, ...Attention
│   │   ├── conv.py                     # GCNConv, SchNetCFconv, SchNetInteraction
│   │   ├── geom.py                     # GaussBasis, BesselBasis, SphericalBasis, CosCutOff
│   │   ├── attention.py                # AttentionHeadGAT, AttentionHeadGATV2
│   │   ├── pooling.py                  # PoolingNodes, PoolingWeightedNodes
│   │   ├── mlp.py                      # MLP (需要 input_dim)
│   │   ├── norm.py                     # GraphBatchNorm, GraphLayerNorm, GraphNormalization
│   │   ├── update.py                   # GRUUpdate, ResidualLayer
│   │   └── scale.py                    # StandardLabelScaler, ExtensiveMolecularLabelScaler
│   │
│   ├── models/                         # 模型
│   │   ├── gcn.py                      # GCNModel
│   │   ├── gat.py                      # GATModel
│   │   ├── schnet.py                   # SchNetModel
│   │   ├── painn.py                    # PAiNNModel (+ PAiNNConv, PAiNNUpdate)
│   │   └── dimenetpp.py                # DimeNetPPModel (+ InteractionPPBlock, OutputBlock)
│   │
│   ├── data/                           # 数据管道
│   │   ├── base.py                     # GraphDict, MemoryGraphList (+to_pyg_list)
│   │   ├── transform.py                # StandardScaler, ExtensiveMolecularScaler
│   │   ├── qm.py                       # QMDataset (PyG InMemoryDataset)
│   │   ├── moleculenet.py              # MoleculeNetDataset
│   │   └── crystal.py                  # CrystalDataset
│   │
│   ├── training/                       # 训练基础设施
│   │   ├── hyper.py                    # HyperParameter 配置管理
│   │   ├── trainer.py                  # train_epoch, eval_epoch, fit
│   │   └── scheduler.py                # LR 调度器
│   │
│   ├── metrics/
│   │   └── metrics.py                  # ScaledMAE, ScaledRMSE, ForceMAE
│   │
│   ├── losses/
│   │   └── losses.py                   # ForceMeanAbsoluteError, EnergyForceLoss
│   │
│   ├── graph/                          # ← 直接从 KGCNN 复制 (纯 numpy)
│   ├── molecule/                       # ← 直接从 KGCNN 复制 (RDKit)
│   └── crystal/                        # ← 直接从 KGCNN 复制 (pymatgen)
│
├── tests/
│   ├── test_ops_scatter.py             # 7 tests
│   ├── test_layers.py                  # 21 tests
│   ├── test_models.py                  # 10 tests
│   └── test_data_pipeline.py           # 8 tests (含 DataLoader 集成测试)
│
└── training_scripts/
    └── train_graph.py                  # CLI 训练脚本
```

---

## 13. 代码统计

| 类别 | 文件数 | 新写代码行数 |
|------|--------|-------------|
| ops (scatter, activ) | 2 | ~200 |
| initializers | 1 | ~30 |
| layers (10 个文件) | 10 | ~800 |
| models (5 个模型) | 5 | ~900 |
| data (base, transform, 3 datasets) | 5 | ~500 |
| training (trainer, scheduler, hyper) | 3 | ~400 |
| metrics + losses | 2 | ~130 |
| tests | 4 | ~350 |
| training scripts | 1 | ~100 |
| **总计 (新写)** | **33** | **~3,400** |
| 复制 (graph/, molecule/, crystal/) | ~20 | ~5,000+ |

**测试覆盖**: 47 个单元测试，全部通过。

---

## API 快速参考

### 创建模型

```python
from kgcnn_torch.models.schnet import SchNetModel

model = SchNetModel(
    node_dim=128, depth=3, units=128,
    edge_dim=20, gauss_bins=20,
    num_targets=1
)
```

### 数据加载

```python
from kgcnn_torch.data.base import MemoryGraphList
from torch_geometric.loader import DataLoader

graphs = MemoryGraphList()
graphs.empty(100)
graphs.set("node_number", [...])
graphs.set("node_coordinates", [...])
graphs.set("edge_indices", [...])      # KGCNN 约定: [target, source]
graphs.set("graph_labels", [...])

pyg_list = graphs.to_pyg_list()         # 自动交换为 PyG 约定
loader = DataLoader(pyg_list, batch_size=32, shuffle=True)
```

### 训练

```python
from kgcnn_torch.training.trainer import fit

history = fit(
    model, train_loader, val_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-4),
    loss_fn=torch.nn.L1Loss(),
    epochs=300, device=torch.device("cuda"),
    early_stopping_patience=50
)
```
