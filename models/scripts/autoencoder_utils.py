"""
图自编码器共享工具函数
提供数据转换、评估指标、可视化等通用功能
"""

import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import global_mean_pool, global_max_pool
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, accuracy_score


def encode_highway_type(highway_type: str) -> int:
    """
    编码道路类型为数值
    
    Args:
        highway_type: 道路类型字符串
        
    Returns:
        编码后的整数值
    """
    type_mapping = {
        'footway': 0,
        'secondary': 1,
        'tertiary': 2,
        'primary': 3,
    }
    return type_mapping.get(highway_type, 0)


def extract_node_centrality(nx_graph: nx.Graph) -> Dict[int, List[float]]:
    """
    提取节点中心性特征
    
    Args:
        nx_graph: NetworkX图对象
        
    Returns:
        节点中心性特征字典 {node_id: [degree_centrality, betweenness, closeness]}
    """
    degree_centrality = nx.degree_centrality(nx_graph)
    betweenness_centrality = nx.betweenness_centrality(nx_graph)
    closeness_centrality = nx.closeness_centrality(nx_graph)
    
    centrality_features = {}
    for node in nx_graph.nodes():
        centrality_features[node] = [
            degree_centrality.get(node, 0),
            betweenness_centrality.get(node, 0),
            closeness_centrality.get(node, 0)
        ]
    
    return centrality_features


def convert_ego_graphs_to_pytorch(pkl_path: str) -> List[Data]:
    """
    将ego-graphs转换为PyTorch Geometric格式
    
    Args:
        pkl_path: pickle文件路径
        
    Returns:
        PyTorch Geometric Data对象列表
    """
    print("加载pkl文件...")
    with open(pkl_path, 'rb') as f:
        ego_graphs = pickle.load(f)
    
    print("转换图数据...")
    pytorch_graphs = []
    
    for i, graph_data in enumerate(ego_graphs):
        nx_graph = graph_data['graph']
        node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
        
        # 提取节点特征：度数 + 中心性
        node_features = []
        for node in nx_graph.nodes():
            degree = nx_graph.nodes[node]['degree']
            centrality = nx_graph.nodes[node]['centrality']
            
            features = [
                degree,
                centrality,
            ]
            node_features.append(features)
                
        # 提取边特征和索引
        edge_features = []
        edge_indices = []
        for u, v, data in nx_graph.edges(data=True):
            try:
                highway_encoded = encode_highway_type(data.get('highway', 'unknown'))
                width = float(data.get('width', 0.0))
                length = float(data.get('length', 0.0))
                
                features = [
                    highway_encoded,
                    width,
                    length
                ]
                edge_features.append(features)
                edge_indices.append([node_mapping[u], node_mapping[v]])
            except (ValueError, TypeError) as e:
                print(f"警告：跳过边 ({u}, {v})，数据转换错误: {e}")
                continue
        
        # 创建Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            graph_id=graph_data.get('id', i)
        )
        pytorch_graphs.append(data)
        
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(ego_graphs)} 个图")
    
    print(f"转换完成！共处理 {len(pytorch_graphs)} 个图")
    return pytorch_graphs


def compute_link_prediction_metrics(
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    pos_pred: torch.Tensor,
    neg_pred: torch.Tensor
) -> Dict[str, float]:
    """
    计算链接预测评估指标
    
    Args:
        pos_edge_index: 正样本边索引 [2, num_pos_edges]
        neg_edge_index: 负样本边索引 [2, num_neg_edges]
        pos_pred: 正样本预测概率 [num_pos_edges]
        neg_pred: 负样本预测概率 [num_neg_edges]
        
    Returns:
        评估指标字典 {'auc', 'ap', 'precision', 'accuracy'}
    """
    # 合并预测和标签
    preds = torch.cat([pos_pred, neg_pred]).cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_pred.size(0)),
        torch.zeros(neg_pred.size(0))
    ]).cpu().numpy()
    
    # 计算AUC和AP
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    
    # 二值化预测（阈值0.5）
    binary_preds = (preds > 0.5).astype(int)
    
    # 计算Precision和Accuracy
    precision = precision_score(labels, binary_preds, zero_division=0)
    accuracy = accuracy_score(labels, binary_preds)
    
    return {
        'auc': float(auc),
        'ap': float(ap),
        'precision': float(precision),
        'accuracy': float(accuracy)
    }


def get_pos_neg_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取正样本边和负样本边
    
    Args:
        edge_index: 原始边索引 [2, num_edges]
        num_nodes: 节点数量
        num_neg_samples: 负样本数量（默认与正样本相同）
        
    Returns:
        (正样本边索引, 负样本边索引)
    """
    # 正样本边
    pos_edge_index = edge_index
    
    # 负采样
    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)
    
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples
    )
    
    return pos_edge_index, neg_edge_index


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    model_name: str = "Model"
) -> None:
    """
    绘制训练曲线（2x2布局）
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
        model_name: 模型名称
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Training Curves', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 左上：Train Loss vs Val Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右上：Val Precision
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_precision'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Validation Precision', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # 左下：Val Accuracy
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_accuracy'], 'm-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Validation Accuracy', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # 右下：组合指标
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2)
    ax4.plot(epochs, history['val_accuracy'], 'm-', label='Accuracy', linewidth=2)
    if 'val_auc' in history:
        ax4.plot(epochs, history['val_auc'], 'c-', label='AUC', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Combined Metrics', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存至: {save_path}")


def split_data(
    graphs: List[Data],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Data], List[Data]]:
    """
    划分训练集和验证集
    
    Args:
        graphs: 图数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
        
    Returns:
        (训练集, 验证集)
    """
    import random
    random.seed(seed)
    
    total = len(graphs)
    train_size = int(total * train_ratio)
    
    # 随机打乱数据
    graphs_shuffled = graphs.copy()
    random.shuffle(graphs_shuffled)
    
    train_graphs = graphs_shuffled[:train_size]
    val_graphs = graphs_shuffled[train_size:]
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_graphs)} 个图")
    print(f"  验证集: {len(val_graphs)} 个图")
    
    return train_graphs, val_graphs


def load_config(config_path: str) -> Dict:
    """
    加载JSON配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def save_config(config: Dict, config_path: str) -> None:
    """
    保存配置到JSON文件
    
    Args:
        config: 配置字典
        config_path: 保存路径
    """
    import json
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"配置已保存至: {config_path}")


def print_model_summary(model: torch.nn.Module, model_name: str) -> None:
    """
    打印模型摘要信息
    
    Args:
        model: PyTorch模型
        model_name: 模型名称
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"{model_name} 模型摘要")
    print(f"{'='*60}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"{'='*60}\n")

