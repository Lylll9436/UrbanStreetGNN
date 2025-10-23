# 城市街道网络图自编码器项目

基于PyTorch Geometric实现的三种图自编码器模型（GAE、VGAE、GraphMAE），用于学习城市街道网络的图级别嵌入表示。

## 🎯 项目目标

将城市街道网络的每个ego-graph编码为固定维度的向量表示，用于：
- 街道网络结构分析
- 图相似度计算
- 下游任务（如强化学习、分类、聚类等）

## 📋 模型概览

| 模型 | 类型 | 核心思想 | 优势 | 适用场景 |
|------|------|----------|------|----------|
| **GAE** | 确定性自编码器 | 编码-解码重建边 | 简单稳定 | 快速baseline |
| **VGAE** | 变分自编码器 | 学习潜在分布 | 鲁棒性强 | **推荐用于街道网络** |
| **GraphMAE** | 掩码自编码器 | 掩码特征重建 | SOTA性能 | 特征丰富场景 |

## 🏗️ 模型架构图

### 1️⃣ GAE (Graph AutoEncoder) 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GAE 模型架构                                     │
└─────────────────────────────────────────────────────────────────────────┘

输入图数据
    │
    │  节点特征: [N, 2]  (度数, 中心性)
    │  边特征:   [E, 3]  (类型, 宽度, 长度)
    │  边索引:   [2, E]
    │
    ▼
┌───────────────────────────────────────┐
│     节点特征编码器 (MLP)               │
│  Linear(2 → 256) → ReLU → Dropout     │
│  → Linear(256 → 256)                  │
└───────────────┬───────────────────────┘
                │  节点嵌入: [N, 256]
                ▼
┌───────────────────────────────────────┐
│     GNN消息传递 (×3层)                │
│  ┌─────────────────────────────────┐ │
│  │  GCNConv(256 → 256)             │ │
│  │  → ReLU → Dropout               │ │
│  └─────────────────────────────────┘ │
│  ┌─────────────────────────────────┐ │
│  │  GCNConv(256 → 256)             │ │
│  │  → ReLU → Dropout               │ │
│  └─────────────────────────────────┘ │
│  ┌─────────────────────────────────┐ │
│  │  GCNConv(256 → 256)             │ │
│  │  → ReLU → Dropout               │ │
│  └─────────────────────────────────┘ │
└───────────────┬───────────────────────┘
                │  更新后节点嵌入: [N, 256]
                ▼
┌───────────────────────────────────────┐
│     图级别池化                         │
│  ┌─────────────┬─────────────┐       │
│  │ Mean Pool   │ Max Pool    │       │
│  │ [1, 256]    │ [1, 256]    │       │
│  └──────┬──────┴──────┬──────┘       │
│         │ Concatenate │              │
│         └──────┬──────┘              │
│                │  [1, 512]           │
└────────────────┼─────────────────────┘
                 ▼
┌───────────────────────────────────────┐
│     图嵌入投影层                       │
│  Linear(512 → 256) → ReLU → Dropout  │
│  → Linear(256 → 128)                 │
└───────────────┬───────────────────────┘
                │
                ▼
        【图嵌入 z】
          [1, 128]  ◄────── 这是最终输出的图表示
                │
                ├──────────────┐
                │              │
                ▼              ▼
    ┌───────────────────┐   ┌──────────────────────┐
    │  节点投影          │   │  链接预测解码器       │
    │  z → [N, 256]     │   │  ┌────────────────┐  │
    └──────┬────────────┘   │  │ 内积解码        │  │
           │                │  │ σ(z[i]·z[j])   │  │
           │                │  └────────────────┘  │
           └────────────────┴──► 边存在概率         │
                               [E]                 │
                               └──────────────────┘

损失函数: 重建损失 (Binary Cross Entropy)
         L = -[log(p_pos) + log(1 - p_neg)]
```

### 2️⃣ VGAE (Variational Graph AutoEncoder) 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      VGAE 模型架构 (变分版本)                            │
└─────────────────────────────────────────────────────────────────────────┘

输入图数据
    │
    ▼
┌───────────────────────────────────────┐
│     节点特征编码器 (MLP)               │
│  Linear(2 → 256) → ReLU → Dropout     │
│  → Linear(256 → 256)                  │
└───────────────┬───────────────────────┘
                │  节点嵌入: [N, 256]
                ▼
┌───────────────────────────────────────┐
│     GNN消息传递 (×3层)                │
│  GCNConv → ReLU → Dropout (×3)       │
└───────────────┬───────────────────────┘
                │  更新后节点嵌入: [N, 256]
                ▼
┌───────────────────────────────────────┐
│     图级别池化                         │
│  Mean Pool ⊕ Max Pool → [1, 512]     │
└───────────────┬───────────────────────┘
                │
                ├─────────────┬─────────────┐
                ▼             ▼             ▼
    ┌──────────────────┐ ┌──────────────────┐
    │  均值分支 (μ)     │ │  对数方差分支     │
    │  Linear(512→256) │ │  (log σ²)        │
    │  →ReLU→Dropout   │ │  Linear(512→256) │
    │  →Linear(256→128)│ │  →ReLU→Dropout   │
    │                  │ │  →Linear(256→128)│
    └────────┬─────────┘ └────────┬─────────┘
             │                    │
             │  μ: [1, 128]       │  log σ²: [1, 128]
             │                    │
             └──────┬─────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │   重参数化技巧 (Reparameterization)  │
    │                                     │
    │   ε ~ N(0, I)                      │
    │   z = μ + σ · ε                    │
    │                                     │
    └──────────────┬──────────────────────┘
                   │
                   ▼
           【图嵌入 z】
             [1, 128]  ◄────── 从潜在分布采样
                   │
                   ├──────────────┬──────────────┐
                   │              │              │
                   ▼              ▼              ▼
         ┌──────────────┐  ┌─────────────┐  ┌─────────────┐
         │ 节点投影      │  │ 链接解码     │  │ KL散度计算   │
         │ z→[N, 256]   │  │ σ(z[i]·z[j])│  │ KL(q||p)    │
         └──────┬───────┘  └──────┬──────┘  └──────┬──────┘
                │                 │                 │
                └─────────────────┴─────────────────┘
                                  │
                                  ▼
                        ┌──────────────────────┐
                        │   总损失函数          │
                        │  L = L_recon         │
                        │      + β·L_KL        │
                        │                      │
                        │  L_recon: 重建损失    │
                        │  L_KL: KL散度        │
                        │  β: KL权重(可退火)   │
                        └──────────────────────┘

损失函数: L_total = BCE(重建边) + β·KL(N(μ,σ²) || N(0,I))
         支持KL退火: β从0.01逐渐增加到0.1
```

### 3️⃣ GraphMAE (Graph Masked AutoEncoder) 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GraphMAE 模型架构 (掩码版本)                          │
└─────────────────────────────────────────────────────────────────────────┘

输入图数据
    │
    │  节点特征: [N, 2]
    │  边特征:   [E, 3]
    │
    ▼
┌───────────────────────────────────────┐
│     掩码策略 (Masking Strategy)        │
│  ┌─────────────────────────────────┐ │
│  │ 随机选择15%的节点进行掩码        │ │
│  │ 掩码节点特征 → [MASK] token     │ │
│  │                                 │ │
│  │ 原始特征: [1.2, 0.8] ────┐     │ │
│  │ 掩码特征: [0.0, 0.0] ◄───┘     │ │
│  └─────────────────────────────────┘ │
└───────────────┬───────────────────────┘
                │  掩码后特征: [N, 2]
                ▼
┌───────────────────────────────────────┐
│     节点特征编码器 (含掩码token)       │
│  Mask Token: 可学习参数 [1, 2]        │
│  Linear(2 → 256) → ReLU → Dropout    │
│  → Linear(256 → 256)                 │
└───────────────┬───────────────────────┘
                │  节点嵌入: [N, 256]
                ▼
┌───────────────────────────────────────┐
│     GNN编码器 (×3层)                  │
│  GCNConv → ReLU → Dropout (×3)       │
│  【掩码节点通过消息传递聚合邻居信息】  │
└───────────────┬───────────────────────┘
                │
                ├──────────────────────────┬──────────────────┐
                │                          │                  │
                ▼  节点嵌入: [N, 256]      ▼                  ▼
    ┌───────────────────────┐   ┌──────────────────┐  ┌─────────────┐
    │   图级别池化           │   │  特征解码器       │  │ 链接解码器   │
    │  Mean⊕Max → [1, 512]  │   │  ┌────────────┐  │  │ (辅助任务)   │
    └──────────┬────────────┘   │  │ Linear     │  │  │             │
               │                │  │ 256→256    │  │  │ 重建边连接   │
               ▼                │  │ →ReLU      │  │  └─────┬───────┘
    ┌──────────────────────┐   │  │ →Linear    │  │        │
    │  图嵌入投影层          │   │  │ 256→2      │  │        │
    │  Linear(512→256)      │   │  └─────┬──────┘  │        │
    │  →ReLU→Dropout        │   │        │         │        │
    │  →Linear(256→128)     │   │        ▼         │        │
    └──────────┬───────────┘   │  重建特征:        │        │
               │                │  [N, 2]          │        │
               ▼                └──────┬───────────┘        │
                                       │                    │
       【图嵌入 z】                     │                    │
         [1, 128] ◄──────────────────┐│                    │
                                      ││                    │
                                      ││                    │
                 ┌────────────────────┴┴────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────────────────┐
    │            双任务损失函数                         │
    │  ┌───────────────────┬──────────────────────┐  │
    │  │ 特征重建损失       │  链接重建损失         │  │
    │  │ L_feature         │  L_link              │  │
    │  │                   │                      │  │
    │  │ 仅对被掩码节点计算 │  边连接BCE损失        │  │
    │  │ MSE(预测, 真实)   │  (辅助任务)          │  │
    │  └─────────┬─────────┴─────────┬────────────┘  │
    │            │                   │               │
    │            └──────────┬────────┘               │
    │                       ▼                        │
    │        L_total = L_feature + λ·L_link         │
    │                 (λ = 0.3)                      │
    └───────────────────────────────────────────────┘

核心创新:
1. 掩码预训练: 随机掩盖15%节点特征，强制模型学习鲁棒表示
2. 双任务学习: 特征重建(主) + 链接预测(辅)
3. 可学习掩码token: 自适应学习最优掩码表示
```

### 4️⃣ 完整训练与评估流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      图自编码器训练与评估完整流程                             │
└─────────────────────────────────────────────────────────────────────────────┘

                               ┌──────────────────┐
                               │  原始数据准备     │
                               │  ego_graphs.pkl  │
                               └────────┬─────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────┐
                     │     数据预处理与转换              │
                     │  • NetworkX → PyG Data           │
                     │  • 提取节点/边特征                │
                     │  • 编码道路类型                  │
                     └────────┬─────────────────────────┘
                              │
                              ▼
                     ┌─────────────────────┐
                     │   数据集划分         │
                     │  Train: 80%         │
                     │  Val:   20%         │
                     └────────┬────────────┘
                              │
                              ├─────────────┬─────────────┐
                              ▼             ▼             ▼
                    ┌───────────────┐ ┌──────────────┐ ┌────────────────┐
                    │  训练 GAE     │ │  训练 VGAE   │ │  训练 GraphMAE │
                    │               │ │              │ │                │
                    │  • 编码器     │ │  • 编码器    │ │  • 掩码编码器  │
                    │  • 解码器     │ │  • 变分层    │ │  • 双解码器    │
                    │  • 重建损失   │ │  • 解码器    │ │  • 双任务损失  │
                    │               │ │  • Recon+KL  │ │                │
                    └───────┬───────┘ └──────┬───────┘ └───────┬────────┘
                            │                │                 │
                            │   每个Epoch执行:                 │
                            │   ┌─────────────────────────┐   │
                            │   │ 1. 前向传播计算损失      │   │
                            │   │ 2. 反向传播更新参数      │   │
                            │   │ 3. 验证集评估性能        │   │
                            │   │ 4. 记录训练指标          │   │
                            │   │ 5. 保存最佳模型          │   │
                            │   └─────────────────────────┘   │
                            │                │                 │
                            ▼                ▼                 ▼
                  ┌──────────────────────────────────────────────────┐
                  │              验证集评估指标                       │
                  │  ┌──────────┬──────────┬──────────┬──────────┐  │
                  │  │ Val Loss │ Precision│ Accuracy │   AUC    │  │
                  │  └──────────┴──────────┴──────────┴──────────┘  │
                  │                                                  │
                  │  最佳模型保存策略: 保存最高val_auc的模型          │
                  └─────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                  ┌─────────────────────────────────────────────┐
                  │           训练结果输出                       │
                  │  ┌──────────────────┬────────────────────┐  │
                  │  │ 模型权重          │ 图嵌入              │  │
                  │  │ • gae_model.pt   │ • gae_embeddings   │  │
                  │  │ • vgae_model.pt  │ • vgae_embeddings  │  │
                  │  │ • mae_model.pt   │ • mae_embeddings   │  │
                  │  └──────────────────┴────────────────────┘  │
                  │  ┌──────────────────┬────────────────────┐  │
                  │  │ 训练历史          │ 训练曲线            │  │
                  │  │ • history.npy    │ • curves.png       │  │
                  │  │   - train_loss   │   ┌──────┬──────┐  │  │
                  │  │   - val_loss     │   │Loss  │Prec  │  │  │
                  │  │   - val_prec     │   ├──────┼──────┤  │  │
                  │  │   - val_acc      │   │Acc   │Comb  │  │  │
                  │  │   - val_auc      │   └──────┴──────┘  │  │
                  │  └──────────────────┴────────────────────┘  │
                  └─────────────────────┬───────────────────────┘
                                        │
                                        ▼
                  ┌─────────────────────────────────────────────┐
                  │         模型对比评估                         │
                  │  ┌────────────────────────────────────────┐ │
                  │  │ evaluate_autoencoders.py               │ │
                  │  │                                        │ │
                  │  │ 1. 加载三个模型的训练历史               │ │
                  │  │ 2. 对比性能指标                        │ │
                  │  │ 3. 生成对比图表 (2×3)                  │ │
                  │  │    ├─ Train Loss vs Val Loss          │ │
                  │  │    ├─ Precision Comparison            │ │
                  │  │    ├─ Accuracy Comparison             │ │
                  │  │    ├─ AUC Comparison                  │ │
                  │  │    ├─ Final Metrics Bar Chart         │ │
                  │  │    └─ Model Summary Table             │ │
                  │  │                                        │ │
                  │  │ 4. 嵌入空间可视化 (t-SNE)              │ │
                  │  │    ├─ GAE embeddings                  │ │
                  │  │    ├─ VGAE embeddings                 │ │
                  │  │    └─ GraphMAE embeddings             │ │
                  │  │                                        │ │
                  │  │ 5. 导出性能对比表格 (CSV)              │ │
                  │  └────────────────────────────────────────┘ │
                  └─────────────────────┬───────────────────────┘
                                        │
                                        ▼
                  ┌─────────────────────────────────────────────┐
                  │          最终输出                            │
                  │                                             │
                  │  📊 可视化结果:                             │
                  │    • models_comparison.png (6子图)         │
                  │    • embeddings_comparison.png (t-SNE)     │
                  │                                             │
                  │  📈 性能报告:                               │
                  │    • model_comparison_table.csv            │
                  │                                             │
                  │  💾 模型产物:                               │
                  │    • 3个最佳模型 (best_*.pt)                │
                  │    • 3组图嵌入 (*_embeddings.pt)            │
                  │    • 形状: [num_graphs, 128]               │
                  │                                             │
                  │  ✅ 用于下游任务:                           │
                  │    • 图分类/聚类                            │
                  │    • 相似度计算                             │
                  │    • 强化学习状态表示                       │
                  └─────────────────────────────────────────────┘

运行命令:
  • 单个模型:    python train_gae.py / train_vgae.py / train_graphmae.py
  • 批量训练:    python train_all_models.py
  • 快速测试:    python test_models_quick.py
  • 对比评估:    python evaluate_autoencoders.py
```

## 🚀 快速开始

### 1. 环境要求

```bash
torch >= 1.10.0
torch-geometric >= 2.0.0
numpy
matplotlib
scikit-learn
pandas
seaborn
```

### 2. 数据准备

确保数据文件存在：
```
models/data/ego_graphs.pkl
```

数据格式：包含NetworkX图的列表，每个图包含：
- 节点特征：`degree`（度数）、`centrality`（中心性）
- 边特征：`highway`（类型）、`width`（宽度）、`length`（长度）

### 3. 快速测试

验证模型是否能正常运行：

```bash
cd models/scripts
python test_models_quick.py
```

### 4. 训练模型

#### 方式1：训练单个模型

```bash
# 训练GAE
python train_gae.py

# 训练VGAE
python train_vgae.py

# 训练GraphMAE
python train_graphmae.py
```

#### 方式2：批量训练所有模型

```bash
python train_all_models.py
```

### 5. 对比评估

```bash
python evaluate_autoencoders.py
```

## 📊 输出结果

### 每个模型产生的文件

```
models/
├── gnn_models/
│   ├── gae_model.pt                    # GAE最终模型
│   ├── best_gae_model.pt               # GAE最佳模型
│   ├── vgae_model.pt                   # VGAE最终模型
│   ├── best_vgae_model.pt              # VGAE最佳模型
│   ├── graphmae_model.pt               # GraphMAE最终模型
│   └── best_graphmae_model.pt          # GraphMAE最佳模型
├── data/
│   ├── gae_embeddings.pt               # GAE图嵌入
│   ├── vgae_embeddings.pt              # VGAE图嵌入
│   ├── graphmae_embeddings.pt          # GraphMAE图嵌入
│   ├── gae_training_history.npy        # GAE训练历史
│   ├── vgae_training_history.npy       # VGAE训练历史
│   └── graphmae_training_history.npy   # GraphMAE训练历史
└── outputs/
    ├── gae_training_curves.png         # GAE训练曲线（2×2）
    ├── vgae_training_curves.png        # VGAE训练曲线（2×2）
    ├── graphmae_training_curves.png    # GraphMAE训练曲线（2×2）
    ├── models_comparison.png           # 三模型对比（2×3）
    ├── embeddings_comparison.png       # 嵌入可视化（t-SNE）
    └── model_comparison_table.csv      # 性能对比表格
```

### 图嵌入格式

每个模型的嵌入文件包含：

```python
{
    'embeddings': Tensor,      # [num_graphs, embedding_dim]
    'graph_ids': List[int],    # 图ID列表
    'config': Dict             # 模型配置
}
```

### 训练曲线布局

每个模型的训练曲线包含4个子图：
- **左上**: Train Loss vs Val Loss
- **右上**: Val Precision
- **左下**: Val Accuracy  
- **右下**: 组合指标（Precision + Accuracy + AUC）

## ⚙️ 配置文件详解

### 通用参数

```json
{
  "model": {
    "node_features": 2,        // 节点特征维度
    "edge_features": 3,        // 边特征维度
    "hidden_dim": 256,         // 隐藏层维度
    "embedding_dim": 128,      // 图嵌入维度（重要！）
    "num_layers": 3,           // GNN层数
    "dropout": 0.2,            // Dropout率
    "conv_type": "gcn"         // 卷积类型：gcn/gat/sage
  },
  "training": {
    "num_epochs": 100,         // 训练轮数
    "learning_rate": 0.001,    // 学习率
    "weight_decay": 1e-5,      // L2正则化
    "train_ratio": 0.8,        // 训练集比例
    "val_ratio": 0.2,          // 验证集比例
    "seed": 42                 // 随机种子
  }
}
```

### VGAE特有参数

```json
{
  "training": {
    "kl_weight": 0.1           // KL散度权重
  },
  "loss": {
    "kl_annealing": {
      "enabled": true,         // 启用KL退火
      "start_weight": 0.01,    // 起始权重
      "end_weight": 0.1,       // 结束权重
      "anneal_epochs": 30      // 退火轮数
    }
  }
}
```

### GraphMAE特有参数

```json
{
  "model": {
    "mask_ratio": 0.15         // 节点掩码比例
  },
  "training": {
    "link_weight": 0.3         // 链接预测损失权重
  }
}
```

## 🔧 高级使用

### 加载和使用图嵌入

```python
import torch

# 加载嵌入
data = torch.load('models/data/vgae_embeddings.pt')
embeddings = data['embeddings']  # [100, 128]
graph_ids = data['graph_ids']

# 计算图相似度
similarity_matrix = torch.mm(embeddings, embeddings.t())

# 找到最相似的图
graph_idx = 0
similarities = similarity_matrix[graph_idx]
top_k_similar = torch.topk(similarities, k=5)
print(f"与图 {graph_idx} 最相似的图: {top_k_similar.indices}")
```

### 加载训练好的模型进行推理

```python
from vgae_model import create_vgae_model
from autoencoder_utils import load_config, convert_ego_graphs_to_pytorch

# 加载配置和模型
config = load_config('models/config/vgae_config.json')
model = create_vgae_model(config['model'])
model.load_state_dict(torch.load('models/gnn_models/best_vgae_model.pt'))
model.eval()

# 加载数据
graphs = convert_ego_graphs_to_pytorch('models/data/ego_graphs.pkl')

# 推理
with torch.no_grad():
    for graph in graphs:
        embedding = model.get_graph_embedding(graph)
        print(f"图嵌入: {embedding.shape}")
```

### 自定义图池化策略

修改模型编码器中的池化方式：

```python
# 当前：Mean + Max Pooling
mean_pool = global_mean_pool(h, batch)
max_pool = global_max_pool(h, batch)
graph_rep = torch.cat([mean_pool, max_pool], dim=1)

# 可选：仅使用Mean Pooling
graph_rep = global_mean_pool(h, batch)

# 可选：添加Sum Pooling
from torch_geometric.nn import global_add_pool
sum_pool = global_add_pool(h, batch)
graph_rep = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
```

## 📈 性能优化建议

### 提升训练速度
- 使用GPU：`device = 'cuda'`
- 减小 `hidden_dim` 和 `embedding_dim`
- 减少 `num_layers`
- 增大 `batch_size`（需修改DataLoader）

### 提升模型性能
- 增加 `hidden_dim` 和 `num_layers`
- 使用 `conv_type: "gat"`（注意力机制）
- 调整学习率调度策略
- 增加训练轮数

### 针对VGAE
- 调整 `kl_weight`（建议0.01-0.5）
- 启用KL退火

### 针对GraphMAE
- 调整 `mask_ratio`（建议0.1-0.3）
- 调整 `link_weight`（建议0.2-0.5）

## 📊 评估指标说明

### 链接预测指标

- **Loss**: 重建损失，越低越好
- **Precision**: 精确率，范围[0, 1]，越高越好
- **Accuracy**: 准确率，范围[0, 1]，越高越好
- **AUC**: ROC曲线下面积，范围[0, 1]，越高越好（**主要指标**）
- **AP**: 平均精度，范围[0, 1]，越高越好

### 最佳模型保存策略

默认按 `val_auc` 保存最佳模型，可修改为其他指标：

```python
# 在训练脚本中修改
if val_metrics['precision'] > best_val_precision:
    best_val_precision = val_metrics['precision']
    torch.save(model.state_dict(), config['paths']['best_model_save'])
```

## 🐛 故障排除

### 问题1: CUDA out of memory
**解决方案**:
- 减小 `hidden_dim` 和 `embedding_dim`
- 使用CPU训练
- 减少图的数量

### 问题2: 训练损失不下降
**解决方案**:
- 检查学习率（尝试0.0001-0.01）
- 检查数据是否正确加载
- 增加模型容量（hidden_dim）

### 问题3: 验证指标不稳定
**解决方案**:
- 增加验证集大小
- 使用固定随机种子
- 延长训练轮数

### 问题4: 嵌入质量差
**解决方案**:
- 增加 `embedding_dim`（如256）
- 增加 `num_layers`（如4-5层）
- 使用VGAE而非GAE
- 延长训练时间

## 📚 参考文献

**GAE/VGAE**:
- Kipf, T. N., & Welling, M. (2016). Variational graph auto-encoders. NIPS Workshop on Bayesian Deep Learning.

**GraphMAE**:
- Hou, Z., et al. (2022). GraphMAE: Self-Supervised Masked Graph Autoencoders. KDD 2022.

**PyTorch Geometric**:
- Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. ICLR Workshop.

## 📝 项目结构

```
models/
├── scripts/                            # 脚本目录
│   ├── autoencoder_utils.py            # 工具函数
│   ├── gae_model.py                    # GAE模型
│   ├── vgae_model.py                   # VGAE模型
│   ├── graphmae_model.py               # GraphMAE模型
│   ├── train_gae.py                    # GAE训练
│   ├── train_vgae.py                   # VGAE训练
│   ├── train_graphmae.py               # GraphMAE训练
│   ├── train_all_models.py             # 批量训练
│   ├── evaluate_autoencoders.py        # 模型对比
│   ├── test_models_quick.py            # 快速测试
│   └── README_AUTOENCODERS.md          # 使用指南
├── config/                             # 配置文件
│   ├── gae_config.json
│   ├── vgae_config.json
│   └── graphmae_config.json
├── gnn_models/                         # 保存的模型
├── data/                               # 数据和嵌入
└── outputs/                            # 可视化结果
```

## ✅ 下一步工作

训练完成后，可以：

1. **分析图嵌入空间**
   - t-SNE/UMAP可视化
   - 聚类分析
   - 相似度分析

2. **下游任务应用**
   - 图分类
   - 图聚类
   - 强化学习的状态表示

3. **模型改进**
   - 尝试不同的GNN架构
   - 添加边特征编码器
   - 实现图级别的对比学习

## 📧 技术支持

遇到问题请：
1. 查看 `README_AUTOENCODERS.md` 详细文档
2. 运行 `test_models_quick.py` 快速诊断
3. 检查配置文件参数设置
4. 查看代码注释和docstrings

