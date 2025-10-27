"""
图自编码器模型对比评估脚本
对比GAE、VGAE、GraphMAE三种模型的性能
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
from sklearn.manifold import TSNE
import seaborn as sns

from autoencoder_utils import load_config


def load_training_history(model_name: str) -> Dict:
    """
    加载训练历史
    
    Args:
        model_name: 模型名称 ('gae', 'vgae', 'graphmae')
        
    Returns:
        训练历史字典
    """
    history_path = f"../data/{model_name}_training_history.npy"
    
    if not os.path.exists(history_path):
        print(f"⚠️  训练历史文件不存在: {history_path}")
        return None
    
    history = np.load(history_path, allow_pickle=True).item()
    return history


def load_embeddings(model_name: str) -> Dict:
    """
    加载图嵌入
    
    Args:
        model_name: 模型名称
        
    Returns:
        嵌入数据字典
    """
    embedding_path = f"../data/{model_name}_embeddings.pt"
    
    if not os.path.exists(embedding_path):
        print(f"⚠️  嵌入文件不存在: {embedding_path}")
        return None
    
    data = torch.load(embedding_path, map_location='cpu')
    return data


def plot_comparison_metrics(histories: Dict[str, Dict], save_path: str) -> None:
    """
    绘制三个模型的对比图表
    
    Args:
        histories: 模型训练历史字典 {model_name: history}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Graph AutoEncoder Models Comparison', fontsize=16, fontweight='bold')
    
    colors = {
        'gae': '#1f77b4',
        'vgae': '#ff7f0e',
        'graphmae': '#2ca02c'
    }
    
    model_labels = {
        'gae': 'GAE',
        'vgae': 'VGAE',
        'graphmae': 'GraphMAE'
    }
    
    # 1. Train Loss对比
    ax1 = axes[0, 0]
    for model_name, history in histories.items():
        if history is None:
            continue
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 
                label=model_labels[model_name], 
                color=colors[model_name], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Train Loss', fontsize=11)
    ax1.set_title('Training Loss Comparison', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Val Loss对比
    ax2 = axes[0, 1]
    for model_name, history in histories.items():
        if history is None:
            continue
        epochs = range(1, len(history['val_loss']) + 1)
        ax2.plot(epochs, history['val_loss'], 
                label=model_labels[model_name], 
                color=colors[model_name], linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Val Loss', fontsize=11)
    ax2.set_title('Validation Loss Comparison', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Val Precision对比
    ax3 = axes[0, 2]
    for model_name, history in histories.items():
        if history is None:
            continue
        epochs = range(1, len(history['val_precision']) + 1)
        ax3.plot(epochs, history['val_precision'], 
                label=model_labels[model_name], 
                color=colors[model_name], linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Precision', fontsize=11)
    ax3.set_title('Validation Precision Comparison', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # 4. Val Accuracy对比
    ax4 = axes[1, 0]
    for model_name, history in histories.items():
        if history is None:
            continue
        epochs = range(1, len(history['val_accuracy']) + 1)
        ax4.plot(epochs, history['val_accuracy'], 
                label=model_labels[model_name], 
                color=colors[model_name], linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Accuracy', fontsize=11)
    ax4.set_title('Validation Accuracy Comparison', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    # 5. Val AUC对比
    ax5 = axes[1, 1]
    for model_name, history in histories.items():
        if history is None:
            continue
        epochs = range(1, len(history['val_auc']) + 1)
        ax5.plot(epochs, history['val_auc'], 
                label=model_labels[model_name], 
                color=colors[model_name], linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('AUC', fontsize=11)
    ax5.set_title('Validation AUC Comparison', fontsize=13)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.05])
    
    # 6. 最终性能对比（柱状图）
    ax6 = axes[1, 2]
    metrics = ['Precision', 'Accuracy', 'AUC']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (model_name, history) in enumerate(histories.items()):
        if history is None:
            continue
        final_scores = [
            history['val_precision'][-1],
            history['val_accuracy'][-1],
            history['val_auc'][-1]
        ]
        ax6.bar(x + i * width, final_scores, width, 
               label=model_labels[model_name], 
               color=colors[model_name])
    
    ax6.set_xlabel('Metrics', fontsize=11)
    ax6.set_ylabel('Score', fontsize=11)
    ax6.set_title('Final Performance Comparison', fontsize=13)
    ax6.set_xticks(x + width)
    ax6.set_xticklabels(metrics)
    ax6.legend(fontsize=10)
    ax6.set_ylim([0, 1.05])
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 对比图表已保存至: {save_path}")


def plot_embedding_comparison(embeddings: Dict[str, Dict], save_path: str) -> None:
    """
    使用t-SNE可视化不同模型的图嵌入
    
    Args:
        embeddings: 模型嵌入字典 {model_name: embedding_data}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Graph Embeddings Visualization (t-SNE)', fontsize=16, fontweight='bold')
    
    model_labels = {
        'gae': 'GAE',
        'vgae': 'VGAE',
        'graphmae': 'GraphMAE'
    }
    
    for idx, (model_name, emb_data) in enumerate(embeddings.items()):
        if emb_data is None:
            continue
        
        # 获取嵌入
        embeddings_tensor = emb_data['embeddings']
        embeddings_np = embeddings_tensor.detach().cpu().numpy()

        if embeddings_np.size == 0 or np.allclose(embeddings_np.std(axis=0), 0):
            print(f"⚠️ {model_labels[model_name]} 嵌入方差接近 0，跳过 t-SNE 可视化")
            ax = axes[idx]
            ax.axis('off')
            ax.text(0.5, 0.5, '嵌入方差≈0，无法绘制', fontsize=12, ha='center', va='center')
            continue

        # t-SNE降维
        print(f"对 {model_labels[model_name]} 进行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings_np)

        # 绘制
        ax = axes[idx]
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=range(len(embeddings_2d)), 
                           cmap='viridis', 
                           alpha=0.6, 
                           s=50)
        ax.set_title(f'{model_labels[model_name]} Embeddings', fontsize=13)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Graph ID')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 嵌入可视化已保存至: {save_path}")


def generate_comparison_table(histories: Dict[str, Dict]) -> pd.DataFrame:
    """
    生成模型对比表格
    
    Args:
        histories: 模型训练历史字典
        
    Returns:
        对比表格DataFrame
    """
    results = []
    
    for model_name, history in histories.items():
        if history is None:
            continue
        
        result = {
            'Model': model_name.upper(),
            'Final Train Loss': f"{history['train_loss'][-1]:.4f}",
            'Final Val Loss': f"{history['val_loss'][-1]:.4f}",
            'Best Val Precision': f"{max(history['val_precision']):.4f}",
            'Best Val Accuracy': f"{max(history['val_accuracy']):.4f}",
            'Best Val AUC': f"{max(history['val_auc']):.4f}",
            'Best Val AP': f"{max(history['val_ap']):.4f}",
            'Final Val Precision': f"{history['val_precision'][-1]:.4f}",
            'Final Val Accuracy': f"{history['val_accuracy'][-1]:.4f}",
            'Final Val AUC': f"{history['val_auc'][-1]:.4f}",
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    return df


def main() -> None:
    """主函数"""
    print("="*60)
    print("图自编码器模型对比评估")
    print("="*60)
    
    # 加载三个模型的训练历史
    print("\n📊 加载训练历史...")
    histories = {
        'gae': load_training_history('gae'),
        'vgae': load_training_history('vgae'),
        'graphmae': load_training_history('graphmae')
    }
    
    # 加载图嵌入
    print("\n📊 加载图嵌入...")
    embeddings = {
        'gae': load_embeddings('gae'),
        'vgae': load_embeddings('vgae'),
        'graphmae': load_embeddings('graphmae')
    }
    
    # 生成对比表格
    print("\n📋 生成对比表格...")
    comparison_df = generate_comparison_table(histories)
    print("\n" + "="*80)
    print("模型性能对比表")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # 保存表格
    table_path = "../outputs/model_comparison_table.csv"
    comparison_df.to_csv(table_path, index=False)
    print(f"\n✅ 对比表格已保存至: {table_path}")
    
    # 绘制对比图表
    print("\n📊 绘制对比图表...")
    plot_comparison_metrics(
        histories,
        save_path="../outputs/models_comparison.png"
    )
    
    # 可视化嵌入
    print("\n📊 可视化图嵌入...")
    plot_embedding_comparison(
        embeddings,
        save_path="../outputs/embeddings_comparison.png"
    )
    
    print("\n✅ 评估完成！")
    print("\n生成的文件:")
    print("- ../outputs/models_comparison.png")
    print("- ../outputs/embeddings_comparison.png")
    print("- ../outputs/model_comparison_table.csv")


if __name__ == "__main__":
    main()

