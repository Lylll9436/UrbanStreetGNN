"""
快速测试脚本 - 验证模型是否能正常运行
使用少量数据和少量epoch进行快速测试
"""

import torch
import os
from datetime import datetime

from gae_model import create_gae_model
from vgae_model import create_vgae_model
from graphmae_model import create_graphmae_model
from autoencoder_utils import (
    # convert_ego_graphs_to_pytorch,
    convert_route_graphs_to_pytorch,
    load_config
)


def test_model_forward(model_name: str, model: torch.nn.Module, test_data: list) -> bool:
    """
    测试模型前向传播
    
    Args:
        model_name: 模型名称
        model: 模型实例
        test_data: 测试数据
        
    Returns:
        测试是否成功
    """
    try:
        print(f"\n测试 {model_name} 前向传播...")
        model.eval()
        
        with torch.no_grad():
            for i, graph in enumerate(test_data[:3]):  # 只测试前3个图
                if model_name == "GAE":
                    graph_embedding, edge_probs = model(graph)
                elif model_name == "VGAE":
                    z, mu, logvar = model(graph)
                    graph_embedding = mu
                elif model_name == "GraphMAE":
                    graph_embedding, node_embeddings, recon_features, mask_nodes = model(graph, use_mask=False)
                
                print(f"  图 {i+1}: 嵌入形状 = {graph_embedding.shape}")
                
                # 验证形状
                assert graph_embedding.shape[0] == 1, f"批次维度应为1，实际为{graph_embedding.shape[0]}"
                assert graph_embedding.shape[1] == model.embedding_dim, \
                    f"嵌入维度应为{model.embedding_dim}，实际为{graph_embedding.shape[1]}"
        
        print(f"✅ {model_name} 前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} 前向传播测试失败: {str(e)}")
        return False


def test_model_training(model_name: str, model: torch.nn.Module, test_data: list, device: str) -> bool:
    """
    测试模型训练
    
    Args:
        model_name: 模型名称
        model: 模型实例
        test_data: 测试数据
        device: 设备
        
    Returns:
        测试是否成功
    """
    try:
        print(f"\n测试 {model_name} 训练过程（3个epoch）...")
        model.train()
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(3):
            total_loss = 0.0
            
            for graph in test_data[:3]:  # 只用前3个图
                graph = graph.to(device)
                
                if model_name == "GAE":
                    loss = model.recon_loss(graph)
                elif model_name == "VGAE":
                    loss, loss_dict = model.loss(graph, kl_weight=0.1)
                elif model_name == "GraphMAE":
                    loss, loss_dict = model.loss(graph, link_weight=0.3)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / min(3, len(test_data))
            print(f"  Epoch {epoch+1}/3: Loss = {avg_loss:.4f}")
        
        print(f"✅ {model_name} 训练测试通过")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} 训练测试失败: {str(e)}")
        return False


def main() -> None:
    """主函数"""
    print("="*80)
    print("图自编码器模型快速测试")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载测试数据
    print("\n📊 加载测试数据...")
    data_path = r"D:\Architecture\AAA-Master\25Fall\CAADRIA\UrbanStreetGNN\models\data\route_graphs.pkl"
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print("请确保数据文件存在后再运行测试")
        return
    
    graphs = convert_route_graphs_to_pytorch(data_path)
    print(f"加载了 {len(graphs)} 个图，使用前5个进行测试")
    test_data = graphs[:5]
    
    # 测试配置
    test_config = {
        'node_features': 9,
        'edge_features': 2,
        'hidden_dim': 64,  # 减小以加快测试
        'embedding_dim': 32,
        'num_layers': 2,
        'dropout': 0.2,
        'conv_type': 'gcn',
        'mask_ratio': 0.15
    }
    
    # 创建模型
    print("\n🏗️ 创建模型...")
    models = {
        'GAE': create_gae_model(test_config),
        'VGAE': create_vgae_model(test_config),
        'GraphMAE': create_graphmae_model(test_config)
    }
    
    # 测试结果
    test_results = {}
    
    # 测试每个模型
    for model_name, model in models.items():
        print("\n" + "-"*80)
        print(f"测试 {model_name}")
        print("-"*80)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数数量: {total_params:,}")
        
        # 测试前向传播
        forward_pass = test_model_forward(model_name, model, test_data)
        
        # 测试训练
        training_pass = test_model_training(model_name, model, test_data, device)
        
        test_results[model_name] = forward_pass and training_pass
    
    # 打印总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    all_passed = True
    for model_name, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{model_name:12s}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 所有测试通过！模型可以正常使用")
        print("\n下一步:")
        print("1. 运行单个模型训练: python train_gae.py")
        print("2. 运行所有模型训练: python train_all_models.py")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("="*80)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

