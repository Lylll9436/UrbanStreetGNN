# 图自编码器模型使用指南

本目录包含三种图自编码器模型的完整实现：GAE、VGAE 和 GraphMAE。

## 📁 文件结构

```
models/
├── scripts/
│   ├── autoencoder_utils.py       # 共享工具函数
│   ├── gae_model.py                # GAE模型定义
│   ├── vgae_model.py               # VGAE模型定义
│   ├── graphmae_model.py           # GraphMAE模型定义
│   ├── train_gae.py                # GAE训练脚本
│   ├── train_vgae.py               # VGAE训练脚本
│   ├── train_graphmae.py           # GraphMAE训练脚本
│   └── evaluate_autoencoders.py    # 模型对比评估脚本
├── config/
│   ├── gae_config.json             # GAE配置文件
│   ├── vgae_config.json            # VGAE配置文件
│   └── graphmae_config.json        # GraphMAE配置文件
├── gnn_models/                     # 保存训练好的模型
├── data/                           # 保存训练数据和嵌入
└── outputs/                        # 保存可视化结果
```

## 🚀 快速开始

### 1. 训练单个模型

#### GAE
```bash
cd models/scripts
python train_gae.py
```

#### VGAE
```bash
cd models/scripts
python train_vgae.py
```

#### GraphMAE
```bash
cd models/scripts
python train_graphmae.py
```

### 2. 对比所有模型

训练完所有模型后，运行对比评估：

```bash
cd models/scripts
python evaluate_autoencoders.py
```

## ⚙️ 配置文件说明

每个模型都有独立的JSON配置文件，可以调整以下参数：

### 模型参数 (`model`)
- `node_features`: 节点特征维度（默认2：度数、中心性）
- `edge_features`: 边特征维度（默认3：类型、宽度、长度）
- `hidden_dim`: 隐藏层维度（默认256）
- `embedding_dim`: 图嵌入维度（默认128）
- `num_layers`: GNN层数（默认3）
- `dropout`: Dropout率（默认0.2）
- `conv_type`: 卷积类型 (`gcn`, `gat`, `sage`)

### GraphMAE额外参数
- `mask_ratio`: 节点掩码比例（默认0.15）

### 训练参数 (`training`)
- `num_epochs`: 训练轮数（默认100）
- `learning_rate`: 学习率（默认0.001）
- `weight_decay`: L2正则化系数（默认1e-5）
- `train_ratio`: 训练集比例（默认0.8）
- `val_ratio`: 验证集比例（默认0.2）
- `seed`: 随机种子（默认42）

### VGAE额外参数
- `kl_weight`: KL散度权重（默认0.1）
- `kl_annealing`: KL退火策略配置

### GraphMAE额外参数
- `link_weight`: 链接预测损失权重（默认0.3）

## 📊 输出说明

### 训练过程输出

每个模型训练后会生成：

1. **模型文件**
   - `{model_name}_model.pt`: 最终模型权重
   - `best_{model_name}_model.pt`: 最佳验证AUC的模型权重

2. **图嵌入**
   - `{model_name}_embeddings.pt`: 所有图的嵌入向量 `[num_graphs, embedding_dim]`

3. **训练历史**
   - `{model_name}_training_history.npy`: 训练指标记录
     - `train_loss`: 训练损失
     - `val_loss`: 验证损失
     - `val_precision`: 验证精确率
     - `val_accuracy`: 验证准确率
     - `val_auc`: 验证AUC
     - `val_ap`: 验证平均精度

4. **可视化**
   - `{model_name}_training_curves.png`: 2×2训练曲线图
     - 左上：Train Loss vs Val Loss
     - 右上：Val Precision
     - 左下：Val Accuracy
     - 右下：组合指标

### 对比评估输出

运行 `evaluate_autoencoders.py` 后生成：

1. **models_comparison.png**: 三个模型的性能对比图（6个子图）
2. **embeddings_comparison.png**: 三个模型的嵌入空间可视化（t-SNE）
3. **model_comparison_table.csv**: 性能指标对比表格

## 🔧 自定义配置示例

### 修改隐藏层维度

编辑 `config/gae_config.json`:

```json
{
  "model": {
    "hidden_dim": 512,
    "embedding_dim": 256
  }
}
```

### 修改学习率和训练轮数

```json
{
  "training": {
    "num_epochs": 200,
    "learning_rate": 0.0005
  }
}
```

### 启用VGAE的KL退火

```json
{
  "loss": {
    "kl_annealing": {
      "enabled": true,
      "start_weight": 0.01,
      "end_weight": 0.1,
      "anneal_epochs": 30
    }
  }
}
```

### 调整GraphMAE掩码比例

```json
{
  "model": {
    "mask_ratio": 0.25
  }
}
```

## 📈 评估指标说明

### 链接预测指标

- **Precision（精确率）**: 预测为正样本中真正为正样本的比例
- **Accuracy（准确率）**: 所有预测正确的比例
- **AUC（ROC曲线下面积）**: 模型区分正负样本的能力
- **AP（平均精度）**: Precision-Recall曲线下面积

### 模型特定指标

**VGAE**:
- `recon_loss`: 重建损失
- `kl_loss`: KL散度

**GraphMAE**:
- `feature_loss`: 特征重建损失
- `link_loss`: 链接预测损失

## 🎯 使用图嵌入

训练完成后，可以加载图嵌入用于下游任务：

```python
import torch

# 加载嵌入
data = torch.load('../data/gae_embeddings.pt')
embeddings = data['embeddings']  # [num_graphs, embedding_dim]
graph_ids = data['graph_ids']

# 使用嵌入进行分析
print(f"嵌入形状: {embeddings.shape}")
print(f"图ID: {graph_ids}")

# 计算图之间的相似度
similarity = torch.mm(embeddings, embeddings.t())
```

## 🔍 模型选择建议

### GAE
- ✅ 简单快速，适合baseline
- ✅ 训练稳定
- ❌ 表达能力相对较弱

### VGAE
- ✅ 学习连续潜在分布
- ✅ 对噪声鲁棒
- ✅ 适合街道网络分析
- ⚠️ 需要调整KL权重

### GraphMAE
- ✅ 最新SOTA方法
- ✅ 掩码预训练，表达能力强
- ✅ 特征重建+链接预测双任务
- ⚠️ 训练时间较长

## 🐛 常见问题

### Q: 训练过程中损失为NaN
A: 降低学习率，检查输入数据是否有异常值

### Q: 验证指标不提升
A: 尝试调整模型结构（增加层数/维度），或修改学习率调度策略

### Q: 内存不足
A: 减小 `hidden_dim` 和 `embedding_dim`，或使用GPU训练

### Q: 如何修改图池化策略
A: 在各模型的编码器中修改 `global_mean_pool` 和 `global_max_pool` 组合方式

## 📝 引用

如果使用本代码，请引用相关论文：

**GAE/VGAE**:
```
Kipf, T. N., & Welling, M. (2016). Variational graph auto-encoders. 
NIPS Workshop on Bayesian Deep Learning.
```

**GraphMAE**:
```
Hou, Z., et al. (2022). GraphMAE: Self-Supervised Masked Graph Autoencoders.
KDD 2022.
```

## 📧 联系

如有问题，请查看代码注释或修改配置文件进行调试。

