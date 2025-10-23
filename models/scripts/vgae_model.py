"""
VGAE (Variational Graph AutoEncoder) 模型实现
基于变分推断的图自编码器，学习连续潜在分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from typing import Dict, Tuple, Optional


class VGAEEncoder(nn.Module):
    """
    VGAE编码器
    
    将图编码为潜在分布的均值和对数方差
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
        初始化VGAE编码器
        
        Args:
            node_features: 节点特征维度
            edge_features: 边特征维度
            hidden_dim: 隐藏层维度
            embedding_dim: 图嵌入维度
            num_layers: GNN层数
            dropout: Dropout率
            conv_type: 卷积类型
        """
        super(VGAEEncoder, self).__init__()
        
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
        
        # 均值分支
        self.mean_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 对数方差分支
        self.logvar_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: 图数据
            
        Returns:
            (均值, 对数方差) 每个形状为 [1, embedding_dim]
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
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        graph_rep = torch.cat([mean_pool, max_pool], dim=1)
        
        # 变分层：输出均值和对数方差
        mu = self.mean_projection(graph_rep)
        logvar = self.logvar_projection(graph_rep)
        
        return mu, logvar


class VGAEDecoder(nn.Module):
    """
    VGAE解码器
    
    从图嵌入重建边连接
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        """
        初始化VGAE解码器
        
        Args:
            embedding_dim: 图嵌入维度
            hidden_dim: 隐藏层维度
        """
        super(VGAEDecoder, self).__init__()
        
        self.node_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        解码边连接
        
        Args:
            z: 图嵌入（从潜在分布采样）
            edge_index: 边索引
            num_nodes: 节点数量
            
        Returns:
            边概率
        """
        node_embeddings = self.node_projection(z)
        node_embeddings = node_embeddings.expand(num_nodes, -1)
        
        row, col = edge_index
        edge_logits = (node_embeddings[row] * node_embeddings[col]).sum(dim=1)
        edge_probs = torch.sigmoid(edge_logits)
        
        return edge_probs


class VGAEModel(nn.Module):
    """
    完整的VGAE模型
    
    变分图自编码器，学习图的潜在分布表示
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
        初始化VGAE模型
        
        Args:
            node_features: 节点特征维度
            edge_features: 边特征维度
            hidden_dim: 隐藏层维度
            embedding_dim: 图嵌入维度
            num_layers: GNN层数
            dropout: Dropout率
            conv_type: 卷积类型
        """
        super(VGAEModel, self).__init__()
        
        self.encoder = VGAEEncoder(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type
        )
        
        self.decoder = VGAEDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        
        self.embedding_dim = embedding_dim
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：从N(mu, var)采样
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            采样的潜在向量
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def encode(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码图为潜在分布参数
        
        Args:
            data: 图数据
            
        Returns:
            (均值, 对数方差)
        """
        return self.encoder(data)
    
    def decode(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        从潜在向量解码边
        
        Args:
            z: 潜在向量
            edge_index: 边索引
            num_nodes: 节点数量
            
        Returns:
            边概率
        """
        return self.decoder(z, edge_index, num_nodes)
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: 图数据
            
        Returns:
            (图嵌入z, 均值mu, 对数方差logvar)
        """
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        计算KL散度损失
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            KL散度
        """
        # KL(N(mu, var) || N(0, 1))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl
    
    def recon_loss(
        self,
        data: Data,
        z: torch.Tensor,
        pos_edge_index: Optional[torch.Tensor] = None,
        neg_edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算重建损失
        
        Args:
            data: 图数据
            z: 潜在向量
            pos_edge_index: 正样本边
            neg_edge_index: 负样本边
            
        Returns:
            重建损失
        """
        if pos_edge_index is None:
            pos_edge_index = data.edge_index
        
        num_nodes = data.x.size(0)
        
        # 解码正样本
        pos_edge_probs = self.decode(z, pos_edge_index, num_nodes)
        
        # 负采样
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )
        
        # 解码负样本
        neg_edge_probs = self.decode(z, neg_edge_index, num_nodes)
        
        # 二元交叉熵损失
        pos_loss = -torch.log(pos_edge_probs + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_edge_probs + 1e-15).mean()
        
        return pos_loss + neg_loss
    
    def loss(
        self,
        data: Data,
        kl_weight: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失 = 重建损失 + KL散度
        
        Args:
            data: 图数据
            kl_weight: KL散度权重
            
        Returns:
            (总损失, 损失详情字典)
        """
        z, mu, logvar = self.forward(data)
        
        recon = self.recon_loss(data, z)
        kl = self.kl_loss(mu, logvar)
        
        total_loss = recon + kl_weight * kl
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon.item(),
            'kl': kl.item()
        }
        
        return total_loss, loss_dict
    
    def test(
        self,
        data: Data,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        测试模式
        
        Args:
            data: 图数据
            pos_edge_index: 正样本边
            neg_edge_index: 负样本边
            
        Returns:
            (正样本预测, 负样本预测)
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(data)
            z = mu  # 测试时使用均值
            num_nodes = data.x.size(0)
            
            pos_pred = self.decode(z, pos_edge_index, num_nodes)
            neg_pred = self.decode(z, neg_edge_index, num_nodes)
        
        return pos_pred, neg_pred
    
    def get_graph_embedding(self, data: Data) -> torch.Tensor:
        """
        获取图嵌入（推理模式，使用均值）
        
        Args:
            data: 图数据
            
        Returns:
            图嵌入 [1, embedding_dim]
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(data)
        return mu


def create_vgae_model(config: Dict) -> VGAEModel:
    """
    根据配置创建VGAE模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        VGAE模型实例
    """
    model = VGAEModel(
        node_features=config.get('node_features', 2),
        edge_features=config.get('edge_features', 3),
        hidden_dim=config.get('hidden_dim', 256),
        embedding_dim=config.get('embedding_dim', 128),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.2),
        conv_type=config.get('conv_type', 'gcn')
    )
    
    return model

