#!/usr/bin/env python3
"""
批量预处理鲁棒性分析实验数据
用于预处理多个数据集和预测尺度的组合
"""

import os
import sys
import subprocess
from pathlib import Path

def batch_prepare_data():
    """批量预处理数据"""
    
    # 定义要预处理的数据集和预测尺度组合
    datasets_configs = [
        # {'dataset': 'fujian', 'prediction_scale': '6-0_1', 'seq_length': 36, 'c_out': 6},
        {'dataset': 'fujian', 'prediction_scale': '6-1', 'seq_length': 36, 'c_out': 6},
        # {'dataset': 'fujian', 'prediction_scale': '24-1', 'seq_length': 36, 'c_out': 24},
        # 可以根据需要添加更多配置
        {'dataset': 'DSWE', 'prediction_scale': '24-2', 'seq_length': 144, 'c_out': 12},
    ]
    
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    save_dir = script_dir / 'preprocessed_data'
    
    print(f"🚀 开始批量预处理鲁棒性分析数据")
    print(f"保存目录: {save_dir}")
    print(f"预处理配置数量: {len(datasets_configs)}")
    print("=" * 60)
    
    success_count = 0
    failed_configs = []
    
    for i, config in enumerate(datasets_configs, 1):
        print(f"\n[{i}/{len(datasets_configs)}] 预处理配置: {config}")
        
        # 构建命令
        cmd = [
            sys.executable, 'prepare_robustness_data.py',
            '--dataset', config['dataset'],
            '--prediction_scale', config['prediction_scale'],
            '--seq_length', str(config['seq_length']),
            '--c_out', str(config['c_out']),
            '--save_dir', str(save_dir)
        ]
        
        try:
            # 运行预处理命令
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ 配置 {config} 预处理成功")
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 配置 {config} 预处理失败")
            print(f"错误信息: {e.stderr}")
            failed_configs.append(config)
        except Exception as e:
            print(f"❌ 配置 {config} 预处理遇到异常: {e}")
            failed_configs.append(config)
    
    print("\n" + "=" * 60)
    print(f"📊 批量预处理完成")
    print(f"成功: {success_count}/{len(datasets_configs)}")
    print(f"失败: {len(failed_configs)}")
    
    if failed_configs:
        print(f"\n❌ 失败的配置:")
        for config in failed_configs:
            print(f"  - {config}")
    
    if success_count > 0:
        print(f"\n✅ 预处理成功的数据文件位于: {save_dir}")
        print(f"现在可以使用以下命令运行鲁棒性分析实验:")
        for config in datasets_configs:
            if config not in failed_configs:
                print(f"  python run_robustness_analysis.py --dataset {config['dataset']} --prediction_scale {config['prediction_scale']} --use_preprocessed")

if __name__ == '__main__':
    batch_prepare_data() 