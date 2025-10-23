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
    
    将图结构编码为图级别的embedding向量
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
        初始化GAE编码器
        
        Args:
            node_features: 节点特征维度
            edge_features: 边特征维度
            hidden_dim: 隐藏层维度
            embedding_dim: 图嵌入维度
            num_layers: GNN层数
            dropout: Dropout率
            conv_type: 卷积类型 ('gcn', 'gat', 'sage')
        """
        super(GAEEncoder, self).__init__()
        
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
        
        # 图级别pooling后的投影层
        # 使用mean + max pooling，所以输入维度是 2 * hidden_dim
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: PyTorch Geometric Data对象
            
        Returns:
            图嵌入向量 [1, embedding_dim]
        """
        x, edge_index = data.x, data.edge_index
        
        # 节点特征编码
        h = self.node_encoder(x)
        
        # GNN消息传递
        for conv in self.conv_layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 图级别pooling
        # 创建batch张量（单图情况下全为0）
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        mean_pool = global_mean_pool(h, batch)  # [1, hidden_dim]
        max_pool = global_max_pool(h, batch)    # [1, hidden_dim]
        
        # 合并pooling结果
        graph_rep = torch.cat([mean_pool, max_pool], dim=1)  # [1, 2*hidden_dim]
        
        # 投影到embedding空间
        graph_embedding = self.graph_projection(graph_rep)  # [1, embedding_dim]
        
        return graph_embedding


class GAEDecoder(nn.Module):
    """
    GAE解码器
    
    从图嵌入重建边连接（链接预测）
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        """
        初始化GAE解码器
        
        Args:
            embedding_dim: 图嵌入维度
            hidden_dim: 隐藏层维度
        """
        super(GAEDecoder, self).__init__()
        
        # 将图embedding投影回节点级别表示
        self.node_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        graph_embedding: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        解码边连接
        
        Args:
            graph_embedding: 图嵌入 [1, embedding_dim]
            edge_index: 边索引 [2, num_edges]
            num_nodes: 节点数量
            
        Returns:
            边存在概率 [num_edges]
        """
        # 从图embedding生成节点表示（广播）
        # 这里简化处理：所有节点共享相同的图级别表示
        node_embeddings = self.node_projection(graph_embedding)  # [1, hidden_dim]
        node_embeddings = node_embeddings.expand(num_nodes, -1)  # [num_nodes, hidden_dim]
        
        # 内积解码器：计算边两端节点的相似度
        row, col = edge_index
        edge_logits = (node_embeddings[row] * node_embeddings[col]).sum(dim=1)
        
        # Sigmoid激活得到概率
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
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type
        )
        
        self.decoder = GAEDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        
        self.embedding_dim = embedding_dim
    
    def encode(self, data: Data) -> torch.Tensor:
        """
        编码图为embedding
        
        Args:
            data: 图数据
            
        Returns:
            图嵌入 [1, embedding_dim]
        """
        return self.encoder(data)
    
    def decode(
        self,
        graph_embedding: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        从embedding解码边
        
        Args:
            graph_embedding: 图嵌入
            edge_index: 边索引
            num_nodes: 节点数量
            
        Returns:
            边概率
        """
        return self.decoder(graph_embedding, edge_index, num_nodes)
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: 图数据
            
        Returns:
            (图嵌入, 正样本边重建概率)
        """
        # 编码
        graph_embedding = self.encode(data)
        
        # 解码（重建正样本边）
        edge_probs = self.decode(graph_embedding, data.edge_index, data.x.size(0))
        
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
        graph_embedding = self.encode(data)
        
        # 解码正样本
        pos_edge_probs = self.decode(graph_embedding, pos_edge_index, num_nodes)
        
        # 负采样
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )
        
        # 解码负样本
        neg_edge_probs = self.decode(graph_embedding, neg_edge_index, num_nodes)
        
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
            graph_embedding = self.encode(data)
            num_nodes = data.x.size(0)
            
            pos_pred = self.decode(graph_embedding, pos_edge_index, num_nodes)
            neg_pred = self.decode(graph_embedding, neg_edge_index, num_nodes)
        
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
            embedding = self.encode(data)
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

