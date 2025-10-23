"""
批量训练所有图自编码器模型
依次训练GAE、VGAE、GraphMAE，并生成对比报告
"""

import os
import sys
from datetime import datetime

from train_gae import train_gae
from train_vgae import train_vgae
from train_graphmae import train_graphmae
from autoencoder_utils import load_config


def main() -> None:
    """主函数：依次训练所有模型"""
    print("="*80)
    print("批量训练图自编码器模型")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    models = [
        ('GAE', '../config/gae_config.json', train_gae),
        ('VGAE', '../config/vgae_config.json', train_vgae),
        ('GraphMAE', '../config/graphmae_config.json', train_graphmae)
    ]
    
    training_results = {}
    
    for model_name, config_path, train_fn in models:
        print("\n" + "="*80)
        print(f"开始训练 {model_name}")
        print("="*80)
        
        try:
            # 加载配置
            config = load_config(config_path)
            
            # 训练模型
            start_time = datetime.now()
            train_fn(config)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            training_results[model_name] = {
                'status': 'success',
                'duration': duration
            }
            
            print(f"\n✅ {model_name} 训练完成，耗时: {duration:.2f}秒")
            
        except Exception as e:
            print(f"\n❌ {model_name} 训练失败: {str(e)}")
            training_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # 打印训练总结
    print("\n" + "="*80)
    print("训练总结")
    print("="*80)
    
    for model_name, result in training_results.items():
        if result['status'] == 'success':
            print(f"✅ {model_name:12s} - 成功 (耗时: {result['duration']:.2f}秒)")
        else:
            print(f"❌ {model_name:12s} - 失败 (错误: {result['error']})")
    
    # 运行对比评估
    all_success = all(r['status'] == 'success' for r in training_results.values())
    
    if all_success:
        print("\n" + "="*80)
        print("运行模型对比评估")
        print("="*80)
        
        try:
            from evaluate_autoencoders import main as evaluate_main
            evaluate_main()
            print("\n✅ 对比评估完成！")
        except Exception as e:
            print(f"\n❌ 对比评估失败: {str(e)}")
    else:
        print("\n⚠️  部分模型训练失败，跳过对比评估")
    
    print("\n" + "="*80)
    print(f"所有任务完成！结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()

