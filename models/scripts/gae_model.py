"""
GAE (Graph AutoEncoder) 模型实现
基于PyTorch Geometric的图自编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from typing import Dict, Tuple, Optional


class GAEEncoder(nn.Module):
    """
    GAE编码器
    
    将节点特征编码为节点级别的隐藏表示
    """
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "gcn"
    ):
        """
        初始化GAE编码器
        
        Args:
            node_features: 节点特征维度
            edge_features: 边特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            dropout: Dropout率
            conv_type: 卷积类型 ('gcn', 'gat', 'sage')
        """
        super(GAEEncoder, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
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
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: PyTorch Geometric Data对象
            
        Returns:
            节点嵌入 [num_nodes, hidden_dim]
        """
        x, edge_index = data.x, data.edge_index
        
        # 节点特征编码
        h = self.node_encoder(x)
        
        # GNN消息传递
        for conv in self.conv_layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h


class GAEDecoder(nn.Module):
    """
    GAE解码器
    
    使用节点级表示计算边存在概率
    """
    
    def __init__(self):
        super(GAEDecoder, self).__init__()
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        解码边连接
        
        Args:
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            边存在概率 [num_edges]
        """
        row, col = edge_index
        edge_logits = (node_embeddings[row] * node_embeddings[col]).sum(dim=1)
        edge_probs = torch.sigmoid(edge_logits)
        return edge_probs


class GAEModel(nn.Module):
    """
    完整的GAE模型
    
    组合编码器和解码器，实现图自编码功能
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
        初始化GAE模型
        
        Args:
            node_features: 节点特征维度
            edge_features: 边特征维度
            hidden_dim: 隐藏层维度
            embedding_dim: 图嵌入维度
            num_layers: GNN层数
            dropout: Dropout率
            conv_type: 卷积类型
        """
        super(GAEModel, self).__init__()
        
        self.encoder = GAEEncoder(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type
        )
        self.decoder = GAEDecoder()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        # 图级别读出网络
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def _graph_readout(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        将节点嵌入汇聚为图嵌入
        """
        batch = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)
        mean_pool = global_mean_pool(node_embeddings, batch)
        max_pool = global_max_pool(node_embeddings, batch)
        graph_rep = torch.cat([mean_pool, max_pool], dim=1)
        return self.graph_projection(graph_rep)
    
    def encode(self, data: Data) -> torch.Tensor:
        """
        编码图为节点级嵌入
        
        Args:
            data: 图数据
            
        Returns:
            节点嵌入 [num_nodes, hidden_dim]
        """
        return self.encoder(data)
    
    def decode(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        从节点嵌入解码边
        
        Args:
            node_embeddings: 节点嵌入
            edge_index: 边索引
            
        Returns:
            边概率
        """
        return self.decoder(node_embeddings, edge_index)
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: 图数据
            
        Returns:
            (图嵌入, 正样本边重建概率)
        """
        # 节点编码
        node_embeddings = self.encode(data)
        graph_embedding = self._graph_readout(node_embeddings)
        
        # 解码（重建正样本边）
        edge_probs = self.decode(node_embeddings, data.edge_index)
        
        return graph_embedding, edge_probs
    
    def recon_loss(
        self,
        data: Data,
        pos_edge_index: Optional[torch.Tensor] = None,
        neg_edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算重建损失
        
        Args:
            data: 图数据
            pos_edge_index: 正样本边（默认使用data.edge_index）
            neg_edge_index: 负样本边（自动采样）
            
        Returns:
            重建损失
        """
        if pos_edge_index is None:
            pos_edge_index = data.edge_index
        
        num_nodes = data.x.size(0)
        
        # 编码
        node_embeddings = self.encode(data)
        
        # 解码正样本
        pos_edge_probs = self.decode(node_embeddings, pos_edge_index)
        
        # 负采样
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )
        
        # 解码负样本
        neg_edge_probs = self.decode(node_embeddings, neg_edge_index)
        
        # 二元交叉熵损失
        pos_loss = -torch.log(pos_edge_probs + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_edge_probs + 1e-15).mean()
        
        loss = pos_loss + neg_loss
        
        return loss
    
    def test(
        self,
        data: Data,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        测试模式：返回正负样本预测
        
        Args:
            data: 图数据
            pos_edge_index: 正样本边
            neg_edge_index: 负样本边
            
        Returns:
            (正样本预测概率, 负样本预测概率)
        """
        self.eval()
        with torch.no_grad():
            node_embeddings = self.encode(data)
            graph_embedding = self._graph_readout(node_embeddings)
            num_nodes = data.x.size(0)
            
            pos_pred = self.decode(node_embeddings, pos_edge_index)
            neg_pred = self.decode(node_embeddings, neg_edge_index)
        
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
            node_embeddings = self.encode(data)
            embedding = self._graph_readout(node_embeddings)
        return embedding


def create_gae_model(config: Dict) -> GAEModel:
    """
    根据配置创建GAE模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        GAE模型实例
    """
    model = GAEModel(
        node_features=config.get('node_features', 2),
        edge_features=config.get('edge_features', 3),
        hidden_dim=config.get('hidden_dim', 256),
        embedding_dim=config.get('embedding_dim', 128),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.2),
        conv_type=config.get('conv_type', 'gcn')
    )
    
    return model

