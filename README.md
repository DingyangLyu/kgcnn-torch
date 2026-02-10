# kgcnn-torch

基于 PyTorch 和 PyTorch Geometric (PyG) 的图神经网络库，面向分子和材料科学领域的属性预测任务。本项目从 [kgcnn](https://github.com/aimat-lab/gcnn_keras)（Keras 版）转换而来，保持了与原始实现的数值对齐。

## 支持的模型

本库提供 **27 种** 图神经网络模型的 PyTorch 实现：

| 类别 | 模型 |
|------|------|
| 经典 GNN | GCN, GAT, GATv2, GIN, rGIN, GraphSAGE, RGCN |
| 消息传递 | SchNet, DimeNetPP, PAiNN, EGNN, NMPN, DMPNN, CMPNN, DGIN |
| 注意力机制 | AttentiveFP, MAT, MoGAT, MEGAN |
| 材料科学 | CGCNN, MEGNet, HDNNP2nd |
| 其他 | MXMNet, INorp, GNNFilm, HamNet, GNNExplain |

同时支持 **Force 模型**包装器，可用于力场预测（energy + forces）。

## 安装

```bash
pip install -e .
```

### 依赖

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4
- NumPy >= 1.22

可选依赖：

```bash
# 分子数据处理
pip install rdkit

# 晶体数据处理
pip install pymatgen

# 全部安装
pip install -e ".[all]"
```

## 快速开始

### 训练图级别属性预测模型

```bash
python training_scripts/train_graph.py \
    --hyper training_scripts/hyper/hyper_esol.json \
    --category SchNet \
    --output results/
```

### 训练力场模型

```bash
python training_scripts/train_force.py \
    --hyper training_scripts/hyper/hyper_md17.json \
    --category SchNet \
    --output results/
```

### 在代码中使用模型

```python
from kgcnn_torch.models.schnet import SchNetModel

model = SchNetModel(
    node_dim=64,
    units=128,
    num_interactions=3,
    output_dim=1
)
```

## 项目结构

```
kgcnn_torch/
├── models/          # 所有 GNN 模型实现
├── layers/          # 基础层（卷积、注意力、聚合、池化等）
├── ops/             # 底层算子（scatter、激活函数等）
├── training/        # 训练器、回调、学习率调度、超参数管理
├── graph/           # 图数据预处理与构建
├── crystal/         # 晶体结构处理（周期性边界）
├── molecule/        # 分子数据处理（RDKit/OpenBabel）
├── metrics/         # 评估指标（MAE、RMSE 等）
├── losses/          # 损失函数
├── io/              # 数据加载工具
└── utils/           # 辅助工具
training_scripts/    # 训练脚本与超参数配置
scripts/             # 模型对齐验证脚本
tests/               # 单元测试
```

## 许可证

MIT
