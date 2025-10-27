"""
GraphMAE (Graph Masked AutoEncoder) 模型实现
基于掩码重建的图自编码器，通过掩码预训练学习鲁棒表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from typing import Dict, Tuple, Optional
import random


class GraphMAEEncoder(nn.Module):
    """
    GraphMAE编码器
    
    对掩码后的图进行编码
    """
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "gcn"
    ):
        """
        初始化GraphMAE编码器
        
        Args:
            node_features: 节点特征维度
            edge_features: 边特征维度
            hidden_dim: 隐藏层维度
            embedding_dim: 图嵌入维度
            num_layers: GNN层数
            dropout: Dropout率
            conv_type: 卷积类型
        """
        super(GraphMAEEncoder, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        
        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 掩码token（可学习参数）
        self.mask_token = nn.Parameter(torch.zeros(1, node_features))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # GNN层
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if conv_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == "gat":
                conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            elif conv_type == "sage":
                conv = SAGEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")
            self.conv_layers.append(conv)
        
        # 图级别投影
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(
        self,
        data: Data,
        mask_nodes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: 图数据
            mask_nodes: 被掩码的节点索引
            
        Returns:
            (图嵌入, 节点嵌入)
        """
        x, edge_index = data.x, data.edge_index
        
        # 应用掩码
        if mask_nodes is not None and len(mask_nodes) > 0:
            x = x.clone()
            x[mask_nodes] = self.mask_token
        
        # 节点特征编码
        h = self.node_encoder(x)
        
        # GNN消息传递
        for conv in self.conv_layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 图级别pooling
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        graph_rep = torch.cat([mean_pool, max_pool], dim=1)
        
        # 图嵌入
        graph_embedding = self.graph_projection(graph_rep)
        
        return graph_embedding, h


class GraphMAEDecoder(nn.Module):
    """
    GraphMAE解码器
    
    重建被掩码的节点特征
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        node_features: int = 2
    ):
        """
        初始化GraphMAE解码器
        
        Args:
            hidden_dim: 隐藏层维度
            node_features: 节点特征维度
        """
        super(GraphMAEDecoder, self).__init__()
        
        self.feature_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_features)
        )
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        解码节点特征
        
        Args:
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            
        Returns:
            重建的节点特征 [num_nodes, node_features]
        """
        return self.feature_decoder(node_embeddings)


class GraphMAEModel(nn.Module):
    """
    完整的GraphMAE模型
    
    通过掩码节点特征重建学习图表示
    """
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "gcn",
        mask_ratio: float = 0.15
    ):
        """
        初始化GraphMAE模型
        
        Args:
            node_features: 节点特征维度
            edge_features: 边特征维度
            hidden_dim: 隐藏层维度
            embedding_dim: 图嵌入维度
            num_layers: GNN层数
            dropout: Dropout率
            conv_type: 卷积类型
            mask_ratio: 掩码比例（0-1之间）
        """
        super(GraphMAEModel, self).__init__()
        
        self.encoder = GraphMAEEncoder(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type
        )
        
        self.decoder = GraphMAEDecoder(
            hidden_dim=hidden_dim,
            node_features=node_features
        )
        
        self.embedding_dim = embedding_dim
        self.mask_ratio = mask_ratio
        
        # 链接预测解码器（用于评估）
        self.link_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def mask_nodes(self, num_nodes: int, mask_ratio: float) -> torch.Tensor:
        """
        随机选择要掩码的节点
        
        Args:
            num_nodes: 总节点数
            mask_ratio: 掩码比例
            
        Returns:
            被掩码的节点索引
        """
        num_mask = int(num_nodes * mask_ratio)
        mask_indices = torch.randperm(num_nodes)[:num_mask]
        return mask_indices
    
    def encode(
        self,
        data: Data,
        mask_nodes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码图
        
        Args:
            data: 图数据
            mask_nodes: 被掩码的节点
            
        Returns:
            (图嵌入, 节点嵌入)
        """
        return self.encoder(data, mask_nodes)
    
    def decode_features(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        解码节点特征
        
        Args:
            node_embeddings: 节点嵌入
            
        Returns:
            重建的节点特征
        """
        return self.decoder(node_embeddings)
    
    def decode_links(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        解码链接（用于评估）
        
        Args:
            node_embeddings: 节点嵌入
            edge_index: 边索引
            
        Returns:
            边概率
        """
        projected = self.link_decoder(node_embeddings)
        
        row, col = edge_index
        edge_logits = (projected[row] * projected[col]).sum(dim=1)
        edge_probs = torch.sigmoid(edge_logits)
        
        return edge_probs
    
    def forward(
        self,
        data: Data,
        use_mask: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            data: 图数据
            use_mask: 是否使用掩码
            
        Returns:
            (图嵌入, 节点嵌入, 重建特征, 掩码节点索引)
        """
        # 生成掩码
        mask_nodes = None
        if use_mask and self.training:
            mask_nodes = self.mask_nodes(data.x.size(0), self.mask_ratio)
        
        # 编码
        graph_embedding, node_embeddings = self.encode(data, mask_nodes)
        
        # 解码节点特征
        reconstructed_features = self.decode_features(node_embeddings)
        
        return graph_embedding, node_embeddings, reconstructed_features, mask_nodes
    
    def feature_recon_loss(
        self,
        data: Data,
        reconstructed_features: torch.Tensor,
        mask_nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算特征重建损失
        
        Args:
            data: 图数据
            reconstructed_features: 重建的特征
            mask_nodes: 被掩码的节点（仅对这些节点计算损失）
            
        Returns:
            重建损失
        """
        if mask_nodes is not None and len(mask_nodes) > 0:
            # 只对被掩码的节点计算损失
            loss = F.mse_loss(
                reconstructed_features[mask_nodes],
                data.x[mask_nodes]
            )
        else:
            # 对所有节点计算损失
            loss = F.mse_loss(reconstructed_features, data.x)
        
        return loss
    
    def link_recon_loss(
        self,
        data: Data,
        node_embeddings: torch.Tensor,
        pos_edge_index: Optional[torch.Tensor] = None,
        neg_edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算链接重建损失（辅助任务）
        
        Args:
            data: 图数据
            node_embeddings: 节点嵌入
            pos_edge_index: 正样本边
            neg_edge_index: 负样本边
            
        Returns:
            链接重建损失
        """
        if pos_edge_index is None:
            pos_edge_index = data.edge_index
        
        # 解码正样本
        pos_edge_probs = self.decode_links(node_embeddings, pos_edge_index)
        
        # 负采样
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=data.x.size(0),
                num_neg_samples=pos_edge_index.size(1)
            )
        
        # 解码负样本
        neg_edge_probs = self.decode_links(node_embeddings, neg_edge_index)
        
        # BCE损失
        pos_loss = -torch.log(pos_edge_probs + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_edge_probs + 1e-15).mean()
        
        return pos_loss + neg_loss
    
    def loss(
        self,
        data: Data,
        link_weight: float = 0.3
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失 = 特征重建损失 + 链接重建损失
        
        Args:
            data: 图数据
            link_weight: 链接重建损失权重
            
        Returns:
            (总损失, 损失详情字典)
        """
        graph_embedding, node_embeddings, reconstructed_features, mask_nodes = self.forward(data)
        
        # 特征重建损失
        feature_loss = self.feature_recon_loss(data, reconstructed_features, mask_nodes)
        
        # 链接重建损失
        link_loss = self.link_recon_loss(data, node_embeddings)
        
        total_loss = feature_loss + link_weight * link_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'feature': feature_loss.item(),
            'link': link_loss.item()
        }
        
        return total_loss, loss_dict
    
    def test(
        self,
        data: Data,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        测试模式：链接预测
        
        Args:
            data: 图数据
            pos_edge_index: 正样本边
            neg_edge_index: 负样本边
            
        Returns:
            (正样本预测, 负样本预测)
        """
        self.eval()
        with torch.no_grad():
            _, node_embeddings, _, _ = self.forward(data, use_mask=False)
            
            pos_pred = self.decode_links(node_embeddings, pos_edge_index)
            neg_pred = self.decode_links(node_embeddings, neg_edge_index)
        
        return pos_pred, neg_pred
    
    def get_graph_embedding(self, data: Data) -> torch.Tensor:
        """
        获取图嵌入（推理模式）
        
        Args:
            data: 图数据
            
        Returns:
            图嵌入 [1, embedding_dim]
        """
        self.eval()
        with torch.no_grad():
            graph_embedding, _, _, _ = self.forward(data, use_mask=False)
        return graph_embedding


def create_graphmae_model(config: Dict) -> GraphMAEModel:
    """
    根据配置创建GraphMAE模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        GraphMAE模型实例
    """
    model = GraphMAEModel(
        node_features=config.get('node_features', 2),
        edge_features=config.get('edge_features', 3),
        hidden_dim=config.get('hidden_dim', 256),
        embedding_dim=config.get('embedding_dim', 128),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.2),
        conv_type=config.get('conv_type', 'gcn'),
        mask_ratio=config.get('mask_ratio', 0.15)
    )
    
    return model

