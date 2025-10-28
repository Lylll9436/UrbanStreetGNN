"""
图自编码器共享工具函数
提供数据转换、评估指标、可视化等通用功能
"""

import pickle
import sys
import math
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import global_mean_pool, global_max_pool
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    accuracy_score,
    roc_curve
)


def _render_progress(description: str, current: int, total: int, bar_length: int = 40) -> None:
    """
    渲染终端进度条
    """
    if total <= 0:
        return
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f"\r{description} [{bar}] {current}/{total}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _progress_print(message: str) -> None:
    """
    在进度条场景下安全打印信息
    """
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(message)


def _safe_float(value, default: float = 0.0) -> float:
    """
    安全地将任意值转换为float；若遇到None/非法字符串/NaN则返回默认值
    """
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"nan", "none", "null"}:
            return default
        value = stripped
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(float_value):
        return default
    return float_value


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

def convert_route_graphs_to_pytorch(pkl_path: str) -> List[Data]:
    """
    将route-graphs转换为PyTorch Geometric格式
    
    Args:
        pkl_path: pickle文件路径
        
    Returns:
        PyTorch Geometric Data对象列表
    """
    print("加载pkl文件...")
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)

    # 兼容封装成字典的最新数据格式，自动提取图列表
    if isinstance(raw_data, dict):
        candidate_keys = ['route_graphs', 'ego_graphs', 'graphs']
        route_graphs = None
        for key in candidate_keys:
            graph_list = raw_data.get(key)
            if isinstance(graph_list, list):
                route_graphs = graph_list
                print(f"检测到封装格式，使用键 '{key}' 中的 {len(route_graphs)} 个图数据")
                break
        if route_graphs is None:
            raise ValueError("未在 pkl 文件中找到 route_graphs/ego_graphs 列表")
    else:
        route_graphs = raw_data

    total_graphs = len(route_graphs)
    print(f"转换图数据，共 {total_graphs} 个 route 图...")
    pytorch_graphs: List[Data] = []
    skipped_empty = 0

    processed_graphs: List[Dict[str, Any]] = []
    feature_sum = np.zeros(9, dtype=np.float64)
    feature_sq_sum = np.zeros(9, dtype=np.float64)
    edge_sum = np.zeros(2, dtype=np.float64)
    edge_sq_sum = np.zeros(2, dtype=np.float64)
    total_nodes = 0
    total_edges = 0
    
    for idx, graph_data in enumerate(route_graphs, start=1):
        nx_graph = graph_data['graph']
        is_directed = nx_graph.is_directed()
        node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
        
        node_features_list: List[List[float]] = []
        for node in nx_graph.nodes():
            node_data = nx_graph.nodes[node]
            length_value = _safe_float(node_data.get('length', 0.0))
            frontage_raw = _safe_float(node_data.get('frontage_L_mean', 0.0))
            if length_value > 0:
                # 将沿街立面长度转换为单位道路长度占比，并限制在 [0, 1]，仍沿用原字段表示占比
                frontage_l_mean = min(max(frontage_raw / length_value, 0.0), 1.0)
            else:
                frontage_l_mean = 0.0
            features = [
                length_value,
                _safe_float(node_data.get('width', 0.0)),
                _safe_float(node_data.get('height_mean', 0.0)),
                frontage_l_mean,
                _safe_float(node_data.get('public_den', 0.0)),
                _safe_float(node_data.get('transport_den', 0.0)),
                _safe_float(node_data.get('nvdi_mean', 0.0)),
                _safe_float(node_data.get('hop_level', 0.0)),
                1.0 if bool(node_data.get('is_center', False)) else 0.0,
            ]
            node_features_list.append(features)
                
        edge_features_list: List[List[float]] = []
        edge_indices_list: List[List[int]] = []
        for u, v, data in nx_graph.edges(data=True):
            try:
                intersection_coords = data.get('intersection_coords', (0.0, 0.0))
                if isinstance(intersection_coords, (list, tuple)) and len(intersection_coords) >= 2:
                    x_coord = _safe_float(intersection_coords[0])
                    y_coord = _safe_float(intersection_coords[1])
                else:
                    x_coord = 0.0
                    y_coord = 0.0
                features = [x_coord, y_coord]
                edge_features_list.append(features.copy())
                edge_indices_list.append([node_mapping[u], node_mapping[v]])

                if not is_directed:
                    edge_features_list.append(features.copy())
                    edge_indices_list.append([node_mapping[v], node_mapping[u]])
            except (ValueError, TypeError) as e:
                _progress_print(f"警告：跳过边 ({u}, {v})，数据转换错误: {e}")
                continue
        
        if len(node_features_list) == 0 or len(edge_indices_list) == 0:
            _progress_print(f"警告：跳过图 {graph_data.get('id', idx)}，节点或边为空")
            skipped_empty += 1
            _render_progress("转换 route 图数据", idx, total_graphs)
            continue
        
        node_array = np.array(node_features_list, dtype=np.float32)
        edge_array = np.array(edge_features_list, dtype=np.float32)
        edge_indices_array = np.array(edge_indices_list, dtype=np.int64)

        feature_sum += node_array.sum(axis=0)
        feature_sq_sum += np.square(node_array).sum(axis=0)
        total_nodes += node_array.shape[0]

        if edge_array.size > 0:
            edge_sum += edge_array.sum(axis=0)
            edge_sq_sum += np.square(edge_array).sum(axis=0)
            total_edges += edge_array.shape[0]

        processed_graphs.append({
            'graph_id': graph_data.get('id', idx),
            'node_features': node_array,
            'edge_features': edge_array,
            'edge_indices': edge_indices_array
        })

        _render_progress("转换 route 图数据", idx, total_graphs)
    
    if total_nodes == 0:
        print("⚠️ 未找到有效的图数据，返回空列表")
        return pytorch_graphs

    node_means = feature_sum / total_nodes
    node_sq_means = feature_sq_sum / total_nodes
    node_vars = np.maximum(node_sq_means - np.square(node_means), 1e-12)
    node_stds = np.sqrt(node_vars)

    if total_edges > 0:
        edge_means = edge_sum / total_edges
        edge_sq_means = edge_sq_sum / total_edges
        edge_vars = np.maximum(edge_sq_means - np.square(edge_means), 1e-12)
        edge_stds = np.sqrt(edge_vars)
    else:
        edge_means = np.zeros(2, dtype=np.float32)
        edge_stds = np.ones(2, dtype=np.float32)

    node_means_tensor = torch.from_numpy(node_means.astype(np.float32))
    node_stds_tensor = torch.from_numpy(node_stds.astype(np.float32))
    edge_means_tensor = torch.from_numpy(edge_means.astype(np.float32))
    edge_stds_tensor = torch.from_numpy(edge_stds.astype(np.float32))

    for info in processed_graphs:
        node_tensor = torch.from_numpy(info['node_features'])
        node_tensor = (node_tensor - node_means_tensor) / node_stds_tensor
        node_tensor = torch.nan_to_num(node_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        edge_tensor = torch.from_numpy(info['edge_features'])
        if edge_tensor.numel() > 0:
            edge_tensor = (edge_tensor - edge_means_tensor) / edge_stds_tensor
            edge_tensor = torch.nan_to_num(edge_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            edge_tensor = torch.empty((0, edge_means_tensor.numel()), dtype=torch.float32)

        edge_index_tensor = torch.from_numpy(info['edge_indices']).t().contiguous().long()

        data = Data(
            x=node_tensor.float(),
            edge_index=edge_index_tensor,
            edge_attr=edge_tensor.float(),
            graph_id=info['graph_id']
        )

        pytorch_graphs.append(data)
    
    print(f"转换完成！共处理 {len(pytorch_graphs)} 个图，跳过 {skipped_empty} 个空图")
    return pytorch_graphs

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
        raw_data = pickle.load(f)

    # Elena 最新数据可能以字典形式封装
    if isinstance(raw_data, dict):
        ego_graphs = raw_data.get('ego_graphs', [])
        if not ego_graphs:
            raise ValueError("未在数据文件中找到 'ego_graphs' 列表")
        roads_info = raw_data.get('roads')
        if roads_info is not None:
            print(f"附带道路数据记录数: {len(roads_info)}")
    else:
        ego_graphs = raw_data
    
    total_graphs = len(ego_graphs)
    print(f"转换图数据，共 {total_graphs} 个 ego 图...")
    pytorch_graphs: List[Data] = []
    skipped_empty = 0

    processed_graphs: List[Dict[str, Any]] = []
    feature_sum = np.zeros(9, dtype=np.float64)
    feature_sq_sum = np.zeros(9, dtype=np.float64)
    edge_sum = np.zeros(2, dtype=np.float64)
    edge_sq_sum = np.zeros(2, dtype=np.float64)
    total_nodes = 0
    total_edges = 0
    
    for idx, graph_data in enumerate(ego_graphs, start=1):
        nx_graph = graph_data['graph']
        is_directed = nx_graph.is_directed()
        node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
        
        node_features_list: List[List[float]] = []
        for node in nx_graph.nodes():
            node_data = nx_graph.nodes[node]
            
            length_value = _safe_float(node_data.get('length', 0.0))
            frontage_raw = _safe_float(node_data.get('frontage_L_mean', 0.0))
            if length_value > 0:
                frontage_l_mean = min(max(frontage_raw / length_value, 0.0), 1.0)
            else:
                frontage_l_mean = 0.0

            features = [
                length_value,
                _safe_float(node_data.get('width', 0.0)),
                _safe_float(node_data.get('height_mean', 0.0)),
                frontage_l_mean,
                _safe_float(node_data.get('public_den', 0.0)),
                _safe_float(node_data.get('transport_den', 0.0)),
                _safe_float(node_data.get('nvdi_mean', 0.0)),
                _safe_float(node_data.get('hop_level', 0.0)),
                1.0 if bool(node_data.get('is_center', False)) else 0.0,
            ]
            node_features_list.append(features)
                
        edge_features_list: List[List[float]] = []
        edge_indices_list: List[List[int]] = []
        for u, v, data in nx_graph.edges(data=True):
            try:
                intersection_coords = data.get('intersection_coords', (0.0, 0.0))
                if isinstance(intersection_coords, (list, tuple)) and len(intersection_coords) >= 2:
                    x_coord = _safe_float(intersection_coords[0])
                    y_coord = _safe_float(intersection_coords[1])
                else:
                    x_coord = 0.0
                    y_coord = 0.0
                features = [x_coord, y_coord]
                edge_features_list.append(features.copy())
                edge_indices_list.append([node_mapping[u], node_mapping[v]])

                if not is_directed:
                    edge_features_list.append(features.copy())
                    edge_indices_list.append([node_mapping[v], node_mapping[u]])
            except (ValueError, TypeError) as e:
                _progress_print(f"警告：跳过边 ({u}, {v})，数据转换错误: {e}")
                continue
        
        if len(node_features_list) == 0 or len(edge_indices_list) == 0:
            _progress_print(f"警告：跳过图 {graph_data.get('id', idx)}，节点或边为空")
            skipped_empty += 1
            _render_progress("转换 ego 图数据", idx, total_graphs)
            continue
        
        node_array = np.array(node_features_list, dtype=np.float32)
        edge_array = np.array(edge_features_list, dtype=np.float32)
        edge_indices_array = np.array(edge_indices_list, dtype=np.int64)

        feature_sum += node_array.sum(axis=0)
        feature_sq_sum += np.square(node_array).sum(axis=0)
        total_nodes += node_array.shape[0]

        if edge_array.size > 0:
            edge_sum += edge_array.sum(axis=0)
            edge_sq_sum += np.square(edge_array).sum(axis=0)
            total_edges += edge_array.shape[0]

        processed_graphs.append({
            'graph_id': graph_data.get('id', idx),
            'node_features': node_array,
            'edge_features': edge_array,
            'edge_indices': edge_indices_array
        })

        _render_progress("转换 ego 图数据", idx, total_graphs)
        
    if total_nodes == 0:
        print("⚠️ 未找到有效的图数据，返回空列表")
        return pytorch_graphs

    node_means = feature_sum / total_nodes
    node_sq_means = feature_sq_sum / total_nodes
    node_vars = np.maximum(node_sq_means - np.square(node_means), 1e-12)
    node_stds = np.sqrt(node_vars)

    if total_edges > 0:
        edge_means = edge_sum / total_edges
        edge_sq_means = edge_sq_sum / total_edges
        edge_vars = np.maximum(edge_sq_means - np.square(edge_means), 1e-12)
        edge_stds = np.sqrt(edge_vars)
    else:
        edge_means = np.zeros(2, dtype=np.float32)
        edge_stds = np.ones(2, dtype=np.float32)

    node_means_tensor = torch.from_numpy(node_means.astype(np.float32))
    node_stds_tensor = torch.from_numpy(node_stds.astype(np.float32))
    edge_means_tensor = torch.from_numpy(edge_means.astype(np.float32))
    edge_stds_tensor = torch.from_numpy(edge_stds.astype(np.float32))

    for info in processed_graphs:
        node_tensor = torch.from_numpy(info['node_features'])
        node_tensor = (node_tensor - node_means_tensor) / node_stds_tensor
        node_tensor = torch.nan_to_num(node_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        edge_tensor = torch.from_numpy(info['edge_features'])
        if edge_tensor.numel() > 0:
            edge_tensor = (edge_tensor - edge_means_tensor) / edge_stds_tensor
            edge_tensor = torch.nan_to_num(edge_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            edge_tensor = torch.empty((0, edge_means_tensor.numel()), dtype=torch.float32)

        edge_index_tensor = torch.from_numpy(info['edge_indices']).t().contiguous().long()

        data = Data(
            x=node_tensor.float(),
            edge_index=edge_index_tensor,
            edge_attr=edge_tensor.float(),
            graph_id=info['graph_id']
        )
        
        pytorch_graphs.append(data)
        
    print(f"转换完成！共处理 {len(pytorch_graphs)} 个图，跳过 {skipped_empty} 个空图")
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
    
    # 根据 Youden Index 自适应选择阈值
    try:
        fpr, tpr, thresholds = roc_curve(labels, preds)
        if thresholds.size > 0:
            youden_index = tpr - fpr
            best_idx = youden_index.argmax()
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = 0.5
    except ValueError:
        best_threshold = 0.5

    binary_preds = (preds > best_threshold).astype(int)
    
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

    if neg_edge_index.numel() == 0 and num_nodes > 0:
        device = edge_index.device if edge_index.is_cuda else torch.device("cpu")
        loops = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
        if loops.numel() == 0:
            neg_edge_index = torch.empty((2, 0), dtype=edge_index.dtype, device=device)
        else:
            neg_edge_index = torch.stack([loops, loops], dim=0)
            target = max(num_neg_samples, 1)
            if neg_edge_index.size(1) < target:
                repeat = math.ceil(target / neg_edge_index.size(1))
                neg_edge_index = neg_edge_index.repeat(1, repeat)
            neg_edge_index = neg_edge_index[:, :target]
    
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
    ����JSON�����ļ�
    
    Args:
        config_path: �����ļ�·��
        
    Returns:
        �����ֵ�
    """
    import json
    config_path = Path(config_path).resolve()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ��̬����paths�ֶ�，ʵ���������Ŀ¼��ת��Ϊ���·��
    base_dir = config_path.parent
    paths = config.get('paths')
    if isinstance(paths, dict):
        resolved_paths: Dict[str, str] = {}
        for key, value in paths.items():
            value_path = Path(value)
            if not value_path.is_absolute():
                value_path = (base_dir / value_path).resolve()
            resolved_paths[key] = str(value_path)
        config['paths'] = resolved_paths

    # ��¼����Ŀ¼，������Ҫ�������ļ�ʱʹ��
    config['_config_dir'] = str(base_dir)
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

