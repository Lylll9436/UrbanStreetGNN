"""
GraphMAE模型训练脚本
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from graphmae_model import create_graphmae_model
from autoencoder_utils import (
    convert_ego_graphs_to_pytorch,
    split_data,
    load_config,
    compute_link_prediction_metrics,
    get_pos_neg_edges,
    plot_training_curves,
    print_model_summary
)


def evaluate_graphmae(
    model: torch.nn.Module,
    graphs: List,
    device: str,
    link_weight: float
) -> Dict[str, float]:
    """
    评估GraphMAE模型性能
    
    Args:
        model: GraphMAE模型
        graphs: 图数据列表
        device: 设备
        link_weight: 链接损失权重
        
    Returns:
        评估指标字典
    """
    model.eval()
    
    total_loss = 0.0
    total_feature_loss = 0.0
    total_link_loss = 0.0
    all_pos_preds = []
    all_neg_preds = []
    all_pos_edges = []
    all_neg_edges = []
    
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            
            # 计算损失
            loss, loss_dict = model.loss(graph, link_weight)
            total_loss += loss_dict['total']
            total_feature_loss += loss_dict['feature']
            total_link_loss += loss_dict['link']
            
            # 获取正负样本边
            pos_edge_index, neg_edge_index = get_pos_neg_edges(
                graph.edge_index,
                graph.x.size(0)
            )
            
            # 获取预测
            pos_pred, neg_pred = model.test(graph, pos_edge_index, neg_edge_index)
            
            all_pos_preds.append(pos_pred)
            all_neg_preds.append(neg_pred)
            all_pos_edges.append(pos_edge_index)
            all_neg_edges.append(neg_edge_index)
    
    # 合并所有预测
    pos_preds_cat = torch.cat(all_pos_preds)
    neg_preds_cat = torch.cat(all_neg_preds)
    pos_edges_cat = torch.cat(all_pos_edges, dim=1)
    neg_edges_cat = torch.cat(all_neg_edges, dim=1)
    
    # 计算指标
    metrics = compute_link_prediction_metrics(
        pos_edges_cat,
        neg_edges_cat,
        pos_preds_cat,
        neg_preds_cat
    )
    
    metrics['loss'] = total_loss / len(graphs)
    metrics['feature_loss'] = total_feature_loss / len(graphs)
    metrics['link_loss'] = total_link_loss / len(graphs)
    
    return metrics


def train_graphmae(config: Dict) -> None:
    """
    训练GraphMAE模型
    
    Args:
        config: 配置字典
    """
    print("="*60)
    print("GraphMAE 模型训练")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n📊 步骤1: 加载数据")
    data_path = config['paths']['data']
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    graphs = convert_ego_graphs_to_pytorch(data_path)
    
    # 划分数据集
    print("\n📊 步骤2: 划分数据集")
    train_graphs, val_graphs = split_data(
        graphs,
        train_ratio=config['training']['train_ratio'],
        val_ratio=config['training']['val_ratio'],
        seed=config['training']['seed']
    )
    
    # 创建模型
    print("\n🏗️ 步骤3: 创建模型")
    model = create_graphmae_model(config['model'])
    model = model.to(device)
    print_model_summary(model, "GraphMAE")
    print(f"掩码比例: {config['model']['mask_ratio']}")
    
    # 优化器和调度器
    print("\n⚙️ 步骤4: 配置优化器")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **config['optimizer']['scheduler_params']
    )
    
    # 训练循环
    print("\n🚂 步骤5: 开始训练")
    print(f"训练轮数: {config['training']['num_epochs']}")
    
    link_weight = config['training']['link_weight']
    
    best_val_auc = 0.0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_feature_loss': [],
        'val_link_loss': [],
        'val_precision': [],
        'val_accuracy': [],
        'val_auc': [],
        'val_ap': []
    }
    
    for epoch in range(config['training']['num_epochs']):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        
        for graph in train_graphs:
            graph = graph.to(device)
            
            # 前向传播
            loss, loss_dict = model.loss(graph, link_weight)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss_dict['total']
        
        avg_train_loss = total_train_loss / len(train_graphs)
        
        # 验证阶段
        val_metrics = evaluate_graphmae(model, val_graphs, device, link_weight)
        
        # 更新学习率
        scheduler.step(val_metrics['loss'])
        
        # 记录历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_feature_loss'].append(val_metrics['feature_loss'])
        training_history['val_link_loss'].append(val_metrics['link_loss'])
        training_history['val_precision'].append(val_metrics['precision'])
        training_history['val_accuracy'].append(val_metrics['accuracy'])
        training_history['val_auc'].append(val_metrics['auc'])
        training_history['val_ap'].append(val_metrics['ap'])
        
        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), config['paths']['best_model_save'])
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Feature: {val_metrics['feature_loss']:.4f} | "
                  f"Link: {val_metrics['link_loss']:.4f} | "
                  f"Precision: {val_metrics['precision']:.4f} | "
                  f"Accuracy: {val_metrics['accuracy']:.4f} | "
                  f"AUC: {val_metrics['auc']:.4f} | "
                  f"LR: {current_lr:.6f}")
    
    # 保存最终模型
    print("\n💾 步骤6: 保存结果")
    torch.save(model.state_dict(), config['paths']['model_save'])
    
    # 保存训练历史
    np.save(config['paths']['history_save'], training_history)
    
    # 绘制训练曲线
    plot_training_curves(
        training_history,
        config['paths']['training_curve'],
        model_name="GraphMAE"
    )
    
    # 生成图嵌入
    print("\n📊 步骤7: 生成图嵌入")
    model.eval()
    all_embeddings = []
    all_graph_ids = []
    
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            embedding = model.get_graph_embedding(graph)
            all_embeddings.append(embedding.cpu())
            all_graph_ids.append(graph.graph_id)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # 保存嵌入
    torch.save({
        'embeddings': all_embeddings,
        'graph_ids': all_graph_ids,
        'config': config
    }, config['paths']['embeddings_save'])
    
    print(f"\n✅ 训练完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最佳验证AUC: {best_val_auc:.4f}")
    print(f"图嵌入形状: {all_embeddings.shape}")
    print("\n生成的文件:")
    print(f"- {config['paths']['model_save']}")
    print(f"- {config['paths']['best_model_save']}")
    print(f"- {config['paths']['embeddings_save']}")
    print(f"- {config['paths']['training_curve']}")
    print(f"- {config['paths']['history_save']}")


def main() -> None:
    """主函数"""
    # 加载配置
    config_path = "../config/graphmae_config.json"
    config = load_config(config_path)
    
    # 训练模型
    train_graphmae(config)


if __name__ == "__main__":
    main()

