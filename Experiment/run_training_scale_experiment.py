#!/usr/bin/env python3
"""
训练数据规模影响实验运行脚本
用法示例:
python run_training_scale_experiment.py --dataset fujian --prediction_scale 6-0_1
python run_training_scale_experiment.py --dataset DSWE --prediction_scale 24-1
"""

import subprocess
import sys
import argparse

def run_experiment(dataset, prediction_scale, gpu=1, epochs=30):
    """
    运行训练数据规模影响实验
    """
    print(f"🚀 启动训练数据规模影响实验")
    print(f"数据集: {dataset}")
    print(f"预测尺度: {prediction_scale}")
    print(f"GPU: {gpu}")
    print(f"最大轮数: {epochs}")
    
    # 构建命令
    cmd = [
        sys.executable, 'adap_auto/new_hier/ex-experiment/training_scale_experiment.py',
        '--dataset', dataset,
        '--prediction_scale', prediction_scale,
        '--gpu', str(gpu),
        '--epochs', str(epochs),
        '--train_ratios', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0',
        '--batch_size', '128',
        '--lr', '0.0002',
        '--patience', '10',
        '--seed', '42'
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("="*60)
    
    # 运行实验
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 实验完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 实验失败，错误码: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n⚠️  实验被用户中断")
        sys.exit(130)

def main():
    parser = argparse.ArgumentParser(description='Run training data scale impact experiment')
    parser.add_argument('--dataset', type=str, default="DSWE",
                        help='Dataset name (fujian or DSWE)')
    parser.add_argument('--prediction_scale', type=str, default="24-2",
                        help='Prediction scale (e.g., 6-0_1, 24-1, 24-2)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--epochs', type=int, default=80, help='Maximum training epochs')
    
    args = parser.parse_args()
    
    # 验证参数
    valid_scales = ['6-0_1', '6-1', '6-2', '6-4', '24-0_1', '24-1', '24-2', '24-4']
    if args.prediction_scale not in valid_scales:
        print(f"⚠️  警告: 预测尺度 '{args.prediction_scale}' 可能不存在对应的数据文件")
        print(f"常见的预测尺度: {valid_scales}")
        response = input("是否继续？(y/N): ")
        if response.lower() != 'y':
            print("实验已取消")
            sys.exit(0)
    
    run_experiment(args.dataset, args.prediction_scale, args.gpu, args.epochs)

if __name__ == '__main__':
    main() 