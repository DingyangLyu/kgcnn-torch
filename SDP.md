# kgcnn-torch: Keras vs Torch 对比检查报告

> 基准: `gcnn_keras-master` (kgcnn v4.0.2, Keras 3 多后端)
> 目标: `kgcnn-torch` (v0.2.0, PyTorch + PyG)
> 检查日期: 2026-02-08
> 最后更新: 2026-02-08 (转换完成)

---

## 转换完成状态

| 维度 | Keras | Torch | 状态 |
|------|-------|-------|------|
| 模型数量 | 28 (含GNNExplain) | **26** | ✅ 仅缺 GNNExplain (元框架, 非模型) |
| 层模块 | 18 个 | **10 个 + 2 ops** | ✅ 所有关键层已转换 |
| 自定义激活函数 | 6 个 | **6 个** | ✅ 完全一致 |
| 训练脚本 | 3 (graph/node/force) | **2** (graph/force, 完整可用) | ✅ 缺 train_node |
| 超参数配置 | 26 | **3** (esol/md17/qm9, 含多模型) | ⚠️ 部分覆盖 |
| 数据管线 | 30+ 数据集类 | 基础框架 + PyG 集成 | ⚠️ 依赖 PyG 数据集 |
| 测试 | 有 | 有 | ✅ |

### 已转换的 26 个模型

SchNet, DimeNetPP, PAiNN, GCN, GAT, **GATv2**, GIN, EGNN, DMPNN, GraphSAGE,
Megnet, AttentiveFP, CGCNN, NMPN, INorp, MEGAN, RGCN, GNNFilm, rGIN, MXMNet,
MoGAT, CMPNN, DGIN, **HamNet**, **HDNNP2nd**, **MAT**

### 所有 45 个模块均通过导入测试 ✅

---

## 以下为原始问题记录 (均已修复)

## 一、总体概况 (转换前)

| 维度 | Keras | Torch | 差距 |
|------|-------|-------|------|
| 模型数量 | 28 | 5 | 缺 23 个 |
| 层文件 | 18 个, ~4000 行 | 13 个, ~800 行 | ~80% 缺失 |
| 预配置数据集 | 30+ (自动下载) | 0 | 全部缺失 |
| 训练脚本 | 3 (graph/node/force) | 1 (graph, 未执行训练) | 缺 2 个; 现有脚本不可用 |
| 超参数配置 | 26 | 0 | 全部缺失 |
| 测试 | 有 | 有 (4 文件) | - |

---

## 二、严重问题 (High Severity)

### H1. GAT 注意力系数在错误的特征空间上计算

**文件**: `kgcnn_torch/layers/attention.py` (AttentionHeadGAT)

**Keras** (正确, 遵循原论文):
```python
# kgcnn/layers/attention.py:99-107
w_n = self.lay_linear_trafo(node)            # W*n
wn_in = self.lay_gather_in([w_n, edge_index])  # W*n_i
wn_out = self.lay_gather_out([w_n, edge_index]) # W*n_j
e_ij = self.lay_concat([wn_in, wn_out])         # [W*n_i || W*n_j]
a_ij = self.lay_alpha(e_ij)                      # a^T [W*n_i || W*n_j]
```

**Torch** (错误, 用了原始特征):
```python
# kgcnn_torch/layers/attention.py:49-66
w_n = self.linear_trafo(x)                       # W*n
n_i = gather_nodes_ingoing(x, edge_index)         # n_i  (原始!)
n_j = gather_nodes_outgoing(x, edge_index)        # n_j  (原始!)
e_ij = torch.cat([n_i, n_j], dim=-1)              # [n_i || n_j]
a_ij = self.attention_activation(self.linear_alpha(e_ij))
```

**影响**: GAT 论文公式为 `a^T [W*h_i || W*h_j]`, Torch 版用的是 `a^T [h_i || h_j]`, 当 `units != in_features` 时注意力系数语义完全不同。

**修复**: 将 gather 的对象从 `x` 改为 `w_n`, 同时将 `linear_alpha` 的输入维度从 `2*in_features` 改为 `2*units`。

---

### H2. DimeNetPP ResidualLayer 只有一层, 应为两层

**文件**: `kgcnn_torch/layers/update.py` (ResidualLayer)

**Keras** (两层 Dense + 残差):
```python
# kgcnn/layers/update.py:168-171
x = self.dense_1(inputs)     # activation(W1*x + b1)
x = self.dense_2(x)          # activation(W2*x + b2)
x = self.add_end([inputs, x])
```

**Torch** (一层 Linear + 残差):
```python
# kgcnn_torch/layers/update.py:38-39
return x + self.activation(self.linear(x))
```

**影响**: DimeNetPP 的 `layers_before_skip` 和 `layers_after_skip` 中每个残差块都只有 Keras 版一半的参数和深度, 显著降低模型容量。

**修复**: 添加第二个 `nn.Linear` + activation, 改为 `x + dense_2(activation(dense_1(x)))`。

---

### H3. 训练脚本未实际执行训练

**文件**: `training_scripts/train_graph.py`

脚本设置完模型、优化器、loss 后在第 124 行 print 退出, 没有调用 `trainer.fit()`。没有数据加载代码。

**修复**: 接通 `trainer.fit()`, 实现数据加载流程, 参考 Keras 版 `training/train_graph.py`。

---

### H4. `trainer.py` 中 `os` 模块引用 Bug

**文件**: `kgcnn_torch/training/trainer.py`

第 205 行使用 `os.path.exists(checkpoint_path)`, 但 `import os` 在第 206 行的 `if` 块内部, 会触发 `NameError`。

**修复**: 将 `import os` 移到文件顶部。

---

## 三、中等问题 (Medium Severity)

### M1. GaussBasisLayer 中心点计算不一致

**文件**: `kgcnn_torch/layers/geom.py` (GaussBasisLayer)

| 版本 | 公式 | bins=20, distance=4.0 示例 |
|------|------|---------------------------|
| Keras | `arange(0, bins) / bins * distance` | `[0.0, 0.2, 0.4, ..., 3.8]` (不含端点) |
| Torch | `linspace(0, distance, bins)` | `[0.0, 0.211, ..., 4.0]` (含端点) |

**影响**: 基函数中心不同, 距离编码不等价。SchNet 和 PAiNN 都使用此层。

**修复**: 改为 `torch.arange(0, bins) / bins * distance` 或在文档中明确选择并保持一致。

---

### M2. PAiNN 归一化层不等价

**文件**: `kgcnn_torch/models/painn.py`

| 位置 | Keras | Torch |
|------|-------|-------|
| 等变量归一化 (line 270) | `GraphLayerNormalization` (按图统计) | `nn.LayerNorm` (按整个 batch 统计) |
| 标量归一化 (line 272) | `GraphBatchNormalization` (按图统计) | `nn.BatchNorm1d` (按整个 batch 统计) |

**影响**: 在 disjoint 表示中, 多图拼成一个 batch, 图级归一化 vs 全局归一化统计量不同, 产生不同训练行为。

**修复**: 实现真正的图级 `GraphBatchNormalization` 和 `GraphLayerNormalization`, 使用 `batch` 索引分组计算统计量。

---

### M3. DimeNetPP Embedding 初始化范围不同

**文件**: `kgcnn_torch/models/dimenetpp.py` (EmbeddingDimeBlock)

| 版本 | 范围 |
|------|------|
| Keras | `[-0.05, 0.05]` (Keras 默认 uniform) |
| Torch | `[-sqrt(3), sqrt(3)]` ≈ `[-1.73, 1.73]` |

**影响**: 初始化范围差 30 倍, 影响训练初期动态。Torch 版本注释说匹配 DimeNet 原始实现, 可能更合理。需要确认哪个更符合原论文。

---

### M4. scatter_softmax 默认 normalize 参数不同

**文件**: `kgcnn_torch/ops/scatter.py` 和 `kgcnn_torch/layers/aggr.py`

| 版本 | 默认值 |
|------|--------|
| Keras | `normalize=False` |
| Torch | `normalize=True` |

**影响**: 未显式指定 `normalize` 时行为不同。`normalize=True` 更安全 (防止 exp 溢出), 但会影响 GAT 等模型的注意力计算结果。

**修复**: 统一为 `normalize=True` (更安全), 或者保证所有调用处显式指定。

---

### M5. ForceMeanAbsoluteError 语义不一致

**文件**: `kgcnn_torch/losses/losses.py`

| 版本 | 实现 |
|------|------|
| Keras | 按分子归一化: 检测 padding 原子, 按每个分子的实际原子数除 |
| Torch | `(pred - target).abs().mean()` -- 简单全局平均 |

**影响**: Torch 版本中大分子对 loss 的贡献权重更大, 不是按分子均匀加权的。

**修复**: 实现按分子归一化的版本, 使用 `batch` 索引分组。

---

### M6. ExtensiveMolecularLabelScaler 方法不同

**文件**: `kgcnn_torch/layers/scale.py`

| 版本 | 方法 |
|------|------|
| Keras | 按原子种类 (atomic number) 做 Ridge 回归, 95 个元素的参考能量 |
| Torch | 简单的 mean-per-atom 减法 + std 缩放 |

**影响**: 对于含多种元素的体系 (如合金、有机分子), Keras 版的按元素参考能量更准确。

---

## 四、低优先级问题 (Low Severity)

### L1. 所有模型只支持 "graph" 输出, 缺少 "node" 输出

**文件**: 所有 `kgcnn_torch/models/*.py`

Keras 版模型同时支持 `output_embedding="graph"` 和 `output_embedding="node"`, Torch 版只支持 `"graph"`。

**修复**: 在各模型的 `forward()` 中增加 `output_embedding` 参数分支。

---

### L2. MLP 在最后一层后也加了 activation

**文件**: `kgcnn_torch/layers/mlp.py:38-48`

```python
for i, out_dim in enumerate(units):
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    # ... dropout, norm ...
    layers.append(get_activation(activation))  # 最后一层也有!
```

Keras 版也是同样的行为, 两者一致。但对于回归任务的 output MLP, 最后一层应为 linear。各模型需要在 output_mlp 外额外加一个线性层, 或者让 MLP 支持 `last_activation` 参数。

---

### L3. CosCutOffEnvelope 裁剪行为略有不同

**文件**: `kgcnn_torch/layers/geom.py` (CosCutOffEnvelope)

| 版本 | 裁剪 |
|------|------|
| Keras | `clip(dist, -cutoff, cutoff)` (双边) |
| Torch | `clamp(dist, max=cutoff)` (仅上界) |

对正距离等价, 对负距离 (不应出现) 有差异。实际无影响。

---

### L4. 距离计算的数值稳定方法不同

**文件**: `kgcnn_torch/layers/geom.py` (compute_edge_distances)

| 版本 | 方法 |
|------|------|
| Keras | `sqrt(relu(sum(x^2)) + eps)` |
| Torch | `sqrt(sum(x^2) + machine_eps)` |

Keras 用 relu 确保非负再加 eps, Torch 直接加 machine epsilon。实际结果几乎相同。

---

## 五、缺失功能清单

### 5.1 缺失模型 (23 个)

| 模型 | 类别 | 用途 |
|------|------|------|
| GIN | 图分类 | 图同构网络, WL-test 等价 |
| rGIN | 图分类 | 残差 GIN |
| GATv2 | 图分类 | 改进的多头注意力 |
| GraphSAGE | 图分类 | 采样聚合 |
| RGCN | 图分类 | 关系图卷积 |
| Megnet | 材料 | 材料性质预测 |
| GNNFilm | 图分类 | 特征线性调制 |
| DMPNN | 分子 | 有向消息传递 |
| CMPNN | 分子 | Chemprop 变体 |
| AttentiveFP | 分子 | 注意力指纹 |
| MoGAT | 分子 | 分子图注意力 |
| MEGAN | 分子 | 等变注意力 |
| MAT | 分子 | 分子注意力 Transformer |
| EGNN | 等变 | 等变图神经网络 |
| CGCNN | 晶体 | 晶体图卷积 |
| HDNNP2nd | 势能面 | 高维神经网络势 |
| HamNet | 等变 | 哈密顿神经网络 |
| DGIN | 图分类 | 双图同构 |
| MXMNet | 分子 | 分子力学 |
| NMPN | 分子 | 神经消息传递 |
| INorp | 图分类 | 交互网络 |
| GNNExplain | 可解释性 | GNN 解释器 |

### 5.2 缺失层

| 层 | 文件 | 用途 | 被哪些模型依赖 |
|----|------|------|---------------|
| `GIN` / `GINE` | conv.py | 图同构卷积 | GIN, rGIN, DGIN |
| `AggregateWeightedLocalEdges` | aggr.py | 加权边聚合 | GCN (内部使用) |
| `AggregateLocalEdgesLSTM` | aggr.py | LSTM 聚合 | GraphSAGE |
| `RelationalAggregateLocalEdges` | aggr.py | 关系聚合 | RGCN |
| `MultiHeadGATV2Layer` | attention.py | 多头 GATv2 | MEGAN |
| `AttentiveHeadFP` | attention.py | AttentiveFP 头 | AttentiveFP |
| `PoolingSet2SetEncoder` | set2set.py | Set2Set 池化 | NMPN |
| `PoolingNodesAttentive` | pooling.py | 注意力池化 | AttentiveFP |
| `PoolingEmbeddingAttention` | pooling.py | 嵌入注意力池化 | 多个模型 |
| `MessagePassingBase` | message.py | 消息传递基类 | NMPN 等 |
| `MatMulMessages` | message.py | 矩阵消息 | NMPN |
| `RelationalMLP` | mlp.py | 关系 MLP | RGCN |
| `GraphMLP` | mlp.py | 图感知 MLP | SchNet, DimeNetPP, PAiNN |
| `GraphInstanceNormalization` | norm.py | 图实例归一化 | - |
| `GatherEdgesPairs` | gather.py | 反向边收集 | DMPNN |
| 13 个几何层 | geom.py | 距离/角度/周期边界等 | 晶体模型 |

### 5.3 缺失训练/数据基础设施

| 组件 | 说明 |
|------|------|
| `train_force.py` | Energy + Force 训练脚本 |
| `train_node.py` | Node-level 训练脚本 |
| `ForceDataset` | Force/Energy 数据集类 |
| `GraphTUDataset` | TU 基准数据集 |
| 30+ 数据集类 | ESOL, QM7/8/9, MD17, MatProject 等 |
| 26 个超参数配置 | 各数据集的训练配置 |
| `BinaryCrossentropyNoNaN` | 处理 NaN 标签的交叉熵 |
| `BinaryAccuracyNoNaN` | 处理 NaN 标签的准确率 |
| `AUCNoNaN` | 处理 NaN 标签的 AUC |
| Cross-validation | K-fold 交叉验证基础设施 |
| 数据序列化/反序列化 | 从配置字典实例化数据集 |
| 图预处理方法 | `set_range`, `set_angle`, `normalize_edge_weights_sym` 等 |

---

## 六、一致的部分 (无需修改)

| 组件 | 说明 |
|------|------|
| scatter_reduce_sum/mean/max/min | 功能等价 |
| SchNet 交互块 | 架构一致: dense_in -> cfconv -> dense+act -> dense -> + residual |
| BesselBasisLayer | 数学公式和可训练频率完全一致 |
| SphericalBasisLayer | 球面基函数等价 (bessel zeros, spherical harmonics) |
| CosCutOffEnvelope | 余弦截断函数等价 |
| GCN 卷积 | 基本等价 (封装方式不同, 计算相同) |
| PAiNN Conv/Update 核心逻辑 | 标量/向量消息传递和更新一致 |
| Edge index 约定 | Torch 正确使用 PyG 约定, data pipeline 中有正确的 swap |
| Glorot-orthogonal 初始化 | DimeNetPP 中使用, 算法相同 |
| 自定义激活函数 | shifted_softplus, swish, leaky_relu2 等一致 |

---

## 七、建议修复顺序

### 阶段 1: 修复现有模型的正确性 (必须)

1. **H1** GAT 注意力: 改为在 W*n 上计算
2. **H2** ResidualLayer: 补为两层 Dense
3. **H4** trainer.py os bug: 移动 import
4. **M1** GaussBasisLayer: 对齐中心点
5. **M4** scatter_softmax: 统一 normalize 默认值

### 阶段 2: 补齐训练基础设施 (重要)

6. **H3** 训练脚本: 接通实际训练循环, 实现数据加载
7. **M2** PAiNN 归一化: 实现图级 BatchNorm/LayerNorm
8. **M5** ForceMeanAbsoluteError: 实现按分子归一化
9. **L1** 所有模型: 增加 "node" 输出支持

### 阶段 3: 补充缺失功能 (按需)

10. 补充 GIN/GINE 卷积层和 GIN 模型
11. 补充 GraphMLP (图感知 MLP)
12. 补充 EGNN, CGCNN, DMPNN 等高优先级模型
13. 补充预配置数据集和超参数配置
14. 补充 cross-validation 基础设施
15. 补充 NaN 处理的 loss/metrics (分类任务)

---

## 八、附录: 边索引约定对照

```
Keras:  edge_index[0] = receive (target),  edge_index[1] = send (source)
Torch:  edge_index[0] = source,            edge_index[1] = target

data pipeline 中 to_pyg_list() 已正确执行了 swap。
各层内部使用的是各自框架的约定, 无需额外处理。
```
