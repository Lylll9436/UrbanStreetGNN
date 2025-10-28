"""
VGAE模型训练脚本
"""

import os
import sys
import io
import logging
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from contextlib import redirect_stdout

from vgae_model import create_vgae_model
from autoencoder_utils import (
    convert_route_graphs_to_pytorch,
    split_data,
    load_config,
    compute_link_prediction_metrics,
    get_pos_neg_edges,
    plot_training_curves,
    print_model_summary
)


def setup_logger(subdir: str, enable_file: bool) -> Tuple[logging.Logger, Optional[Path]]:
    """
    初始化日志记录器
    """
    logger = logging.getLogger(f"{subdir}_trainer")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    log_path: Optional[Path] = None
    if enable_file:
        base_dir = (Path(__file__).resolve().parent / f"../outputs/logs/{subdir}").resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        log_path = base_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger, log_path


def get_kl_weight(epoch: int, config: Dict) -> float:
    """
    计算KL散度权重（支持退火策略）
    
    Args:
        epoch: 当前轮数
        config: 配置字典
        
    Returns:
        KL权重
    """
    kl_annealing = config['loss'].get('kl_annealing', {})
    
    if not kl_annealing.get('enabled', False):
        return config['training']['kl_weight']
    
    start_weight = kl_annealing['start_weight']
    end_weight = kl_annealing['end_weight']
    anneal_epochs = kl_annealing['anneal_epochs']
    
    if epoch < anneal_epochs:
        # 线性退火
        weight = start_weight + (end_weight - start_weight) * (epoch / anneal_epochs)
    else:
        weight = end_weight
    
    return weight


def evaluate_vgae(
    model: torch.nn.Module,
    graphs: List,
    device: str,
    kl_weight: float
) -> Dict[str, float]:
    """
    评估VGAE模型性能
    
    Args:
        model: VGAE模型
        graphs: 图数据列表
        device: 设备
        kl_weight: KL散度权重
        
    Returns:
        评估指标字典
    """
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    all_pos_preds = []
    all_neg_preds = []
    all_pos_edges = []
    all_neg_edges = []
    
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            
            # 计算损失
            loss, loss_dict = model.loss(graph, kl_weight)
            total_loss += loss_dict['total']
            total_recon_loss += loss_dict['recon']
            total_kl_loss += loss_dict['kl']
            
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
    metrics['recon_loss'] = total_recon_loss / len(graphs)
    metrics['kl_loss'] = total_kl_loss / len(graphs)
    
    return metrics


def train_vgae(config: Dict, trial_mode: bool = False) -> Dict[str, Any]:
    """
    训练VGAE模型
    
    Args:
        config: 配置字典
    """
    logger, log_path = setup_logger("vgae", enable_file=not trial_mode)

    def log_info(message: str) -> None:
        logger.info(message)
        print(message)

    if log_path is not None:
        log_info(f"日志文件: {log_path}")

    if not trial_mode:
        log_info("="*60)
        log_info("VGAE 模型训练")
        log_info("="*60)
        log_info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not trial_mode:
        log_info(f"使用设备: {device}")
    
    # 加载数据
    if not trial_mode:
        log_info("\n📊 步骤1: 加载数据")
    data_path = config['paths']['data']
    if not os.path.exists(data_path):
        log_info(f"❌ 数据文件不存在: {data_path}")
        return {"best_val_auc": 0.0, "training_history": {}}
    
    graphs = convert_route_graphs_to_pytorch(data_path)
    
    # 划分数据集
    if not trial_mode:
        log_info("\n📊 步骤2: 划分数据集")
    train_graphs, val_graphs = split_data(
        graphs,
        train_ratio=config['training']['train_ratio'],
        val_ratio=config['training']['val_ratio'],
        seed=config['training']['seed']
    )
    
    # 创建模型
    if not trial_mode:
        log_info("\n🏗️ 步骤3: 创建模型")
    model = create_vgae_model(config['model'])
    model = model.to(device)
    if not trial_mode and log_path is not None:
        summary_buffer = io.StringIO()
        with redirect_stdout(summary_buffer):
            print_model_summary(model, "VGAE")
        for line in summary_buffer.getvalue().splitlines():
            log_info(line)
    else:
        print_model_summary(model, "VGAE")
    
    # 优化器和调度器
    if not trial_mode:
        log_info("\n⚙️ 步骤4: 配置优化器")
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
    if not trial_mode:
        log_info("\n🚂 步骤5: 开始训练")
        log_info(f"训练轮数: {config['training']['num_epochs']}")

    def display_epoch_progress(epoch_idx: int,
                               total_epochs: int,
                               train_loss_value: float,
                               val_metrics_value: Dict[str, float],
                               kl_weight_value: float) -> None:
        bar_len = 30
        progress = (epoch_idx + 1) / total_epochs
        filled = int(bar_len * progress)
        bar = '█' * filled + '-' * (bar_len - filled)
        base_msg = (f"[{bar}] Epoch {epoch_idx + 1}/{total_epochs} "
                    f"Train Loss: {train_loss_value:.4f} | KL_w: {kl_weight_value:.4f}")

        def _fmt(metric_value: float) -> str:
            return "N/A" if metric_value is None or np.isnan(metric_value) else f"{metric_value:.4f}"

        base_msg += (f" | Val Loss: {_fmt(val_metrics_value.get('loss'))} "
                     f"| Recon: {_fmt(val_metrics_value.get('recon_loss'))} "
                     f"| KL: {_fmt(val_metrics_value.get('kl_loss'))} "
                     f"| AUC: {_fmt(val_metrics_value.get('auc'))}")
        if not trial_mode:
            log_info(base_msg)
    
    best_val_auc = 0.0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_precision': [],
        'val_accuracy': [],
        'val_auc': [],
        'val_ap': [],
        'kl_weight': []
    }
    
    last_val_metrics = {
        'loss': float('nan'),
        'recon_loss': float('nan'),
        'kl_loss': float('nan'),
        'precision': float('nan'),
        'accuracy': float('nan'),
        'auc': float('nan'),
        'ap': float('nan')
    }

    for epoch in range(config['training']['num_epochs']):
        # 获取当前KL权重
        kl_weight = get_kl_weight(epoch, config)
        
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        
        for graph in train_graphs:
            graph = graph.to(device)
            
            # 前向传播
            loss, loss_dict = model.loss(graph, kl_weight)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss_dict['total']
        
        avg_train_loss = total_train_loss / len(train_graphs)
        
        val_metrics = evaluate_vgae(model, val_graphs, device, kl_weight)
        last_val_metrics = val_metrics
        scheduler.step(val_metrics['loss'])
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), config['paths']['best_model_save'])
        
        # 记录历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_recon_loss'].append(val_metrics['recon_loss'])
        training_history['val_kl_loss'].append(val_metrics['kl_loss'])
        training_history['val_precision'].append(val_metrics['precision'])
        training_history['val_accuracy'].append(val_metrics['accuracy'])
        training_history['val_auc'].append(val_metrics['auc'])
        training_history['val_ap'].append(val_metrics['ap'])
        training_history['kl_weight'].append(kl_weight)

        display_epoch_progress(epoch,
                               config['training']['num_epochs'],
                               avg_train_loss,
                               val_metrics,
                               kl_weight)
    
    # 保存最终模型
    if not trial_mode:
        log_info("\n💾 步骤6: 保存结果")
        torch.save(model.state_dict(), config['paths']['model_save'])
        
        # 保存训练历史
        np.save(config['paths']['history_save'], training_history)
        
        # 绘制训练曲线
        plot_training_curves(
            training_history,
            config['paths']['training_curve'],
            model_name="VGAE"
        )
        
        # 生成图嵌入
        log_info("\n📊 步骤7: 生成图嵌入")
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
        
        log_info(f"\n✅ 训练完成！")
        log_info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_info(f"最佳验证AUC: {best_val_auc:.4f}")
        log_info(f"图嵌入形状: {all_embeddings.shape}")
        log_info("\n生成的文件:")
        log_info(f"- {config['paths']['model_save']}")
        log_info(f"- {config['paths']['best_model_save']}")
        log_info(f"- {config['paths']['embeddings_save']}")
        log_info(f"- {config['paths']['training_curve']}")
        log_info(f"- {config['paths']['history_save']}")
    else:
        if not np.isnan(last_val_metrics.get('loss', float('nan'))):
            log_info(f"[Trial] 最佳验证AUC: {best_val_auc:.4f}")

    return {
        "best_val_auc": best_val_auc,
        "best_val_metrics": last_val_metrics,
        "training_history": training_history
    }


def main() -> None:
    """主函数"""
    # 加载配置
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir / "../config/vgae_config.json"
    config = load_config(config_path)
    
    # 训练模型
    train_vgae(config)


if __name__ == "__main__":
    main()

