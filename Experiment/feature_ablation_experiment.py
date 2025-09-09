import os
import sys
import warnings
# Set CUDA launch blocking for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 抑制 statsmodels 的 FutureWarning 警告
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
warnings.filterwarnings('ignore', message='verbose is deprecated since functions should not print results')

# 获取当前文件的绝对路径，并向上退一层（父目录）
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)  # 将父目录加入模块搜索路径

import pickle
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from adap_auto import adap_auto
from evaluate import MSE, MAPE
import torch.multiprocessing as mp
import time
import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt
from dynamic_data_processor import create_dynamic_data

def seed_everything(seed=42):
    """设置所有随机种子以确保结果可重复"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def inverse_transform_power(data, scaler, power_feature_idx=0):
    """
    对功率数据进行反标准化
    Args:
        data: 标准化后的数据 (numpy array)
        scaler: sklearn StandardScaler对象
        power_feature_idx: 功率特征在scaler中的索引
    Returns:
        反标准化后的功率数据
    """
    mean = scaler.mean_[power_feature_idx]
    scale = scaler.scale_[power_feature_idx]
    return data * scale + mean

def get_feature_combinations(dataset_name='fujian'):
    """
    定义不同的特征组合用于消融实验
    
    Args:
        dataset_name: 数据集名称
    
    Returns:
        dict: 特征组合字典
    """
    if dataset_name.lower() == 'fujian':
        # 福建数据集的特征组合
        feature_combinations = {
            'power_only': ['y'],  # 仅历史功率
            'power_wind': ['y', 'WS10m', 'WD10m', 'WS100m', 'WD100m'],  # 功率 + 风速风向
            'power_wind_core': ['y', 'WS10m', 'WD10m', 'WS100m', 'WD100m', 'Temp_K', 'Pres_Pa'],  # 功率 + 风 + 核心气象
            'all_features': None  # 全量特征（None表示使用所有特征）
        }
    elif dataset_name.lower() == 'dswe':
        # DSWE数据集的特征组合
        feature_combinations = {
            'power_only': ['y'],  # 仅历史功率
            'power_wind': ['y','V','D','air density'],  # 功率 + 风速风向
            'power_wind_core': ['y','V','D','air density','humidity','I','S_a','S_b'],  # 功率 + 风 + 核心气象
            'all_features': None  # 全量特征
        }
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return feature_combinations

def train_model_with_features(dataset_name, prediction_scale, feature_combination_name, feature_list, args, device):
    """
    使用指定特征组合训练模型
    
    Args:
        dataset_name: 数据集名称
        prediction_scale: 预测尺度
        feature_combination_name: 特征组合名称
        feature_list: 特征列表
        args: 命令行参数
        device: 计算设备
    
    Returns:
        dict: 包含性能指标的字典
    """
    print(f"\n{'='*60}")
    print(f"🚀 开始训练 - 数据集: {dataset_name}, 尺度: {prediction_scale}")
    print(f"特征组合: {feature_combination_name}")
    print(f"特征列表: {feature_list if feature_list else '全部特征'}")
    print(f"{'='*60}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    
    # 根据数据集设置CSV路径
    if dataset_name.lower() == 'fujian':
        csv_path = os.path.join(grandparent_dir, 'data/fujian/Offshore Wind Farm Dataset3(WT1).csv')
    elif dataset_name.lower() == 'dswe':
        csv_path = os.path.join(grandparent_dir, 'data/DSWE/Offshore Wind Farm Dataset1(WT5).csv')
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 验证CSV文件存在
    if not os.path.exists(csv_path):
        print(f"❌ 错误: CSV文件不存在: {csv_path}")
        return None
    
    print(f"🔧 使用动态数据处理系统")
    print(f"   序列长度: {args.seq_length}")
    print(f"   预测长度: {args.c_out}")
    print(f"   数据来源: {csv_path}")

    # 创建动态数据
    try:
        model_data = create_dynamic_data(
            csv_path=csv_path,
            seq_length=args.seq_length,
            pred_length=args.c_out,
            split_ratio=args.split_ratio,
            standardize=True,
            feature_groups=feature_list,  # 指定特征组合
            use_macro_only=False,  # 使用完整的宏观+微观图融合
            rho=0.5,
            save_dir=None,
            verbose=True
        )
        
        # 提取数据
        X_train = model_data['X_train'].to(device)
        y_train = model_data['y_train'].to(device)
        X_test = model_data['X_test'].to(device)
        y_test = model_data['y_test'].to(device)
        train_dict = model_data['train_edge_indices']
        test_dict = model_data['test_edge_indices']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        print(f"✅ 动态数据处理完成")
        print(f"   训练集: X{X_train.shape}, y{y_train.shape}")
        print(f"   测试集: X{X_test.shape}, y{y_test.shape}")
        print(f"   实际特征数量: {model_data['num_features']}")
        print(f"   实际特征名称: {feature_names}")
        
    except Exception as e:
        print(f"❌ 动态数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    # 初始化模型
    print(f"🔧 初始化模型 - 输入特征数: {model_data['num_features']}")
    print(f"   特征名称: {feature_names}")
    auto_model = adap_auto(
        n_head=args.n_head,
        hidden_size=args.hidden_size,
        factor=args.factor,
        dropout=args.dropout,
        conv_hidden_size=args.conv_hidden_size,
        MovingAvg_window=args.moving_avg_window,
        activation=args.activation,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        c_in=model_data['num_features'],  # 使用实际的特征数量
        seq_lenth=args.seq_length,
        c_out=args.c_out,
        gruop_dec=args.group_dec
    )

    auto_model.to(device)
    
    # 定义损失函数和优化器
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(auto_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    l1_lambda = args.l1_lambda
    
    # Early stopping parameters
    patience = args.patience
    best_mse = float('inf')
    patience_counter = 0
    training_start_time = time.time()
    
    # 训练循环
    for epoch in range(args.epochs):
        start_time = time.time()
        auto_model.train()
        total_loss = 0
        I = 0
        
        for batch in train_dataloader:
            inputs, targets = batch
            dicts = train_dict[I:I+len(inputs)]
            I += len(inputs)
            optimizer.zero_grad()
            model_output = auto_model(inputs, dicts).squeeze(-1)
            loss = loss_function(model_output, targets)
            l1_norm = sum(p.abs().sum() for p in auto_model.parameters())
            loss = loss + l1_lambda * l1_norm
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        # Validation loss
        auto_model.eval()
        with torch.no_grad():
            prediction = auto_model(X_test, test_dict).squeeze(-1)
            
            # 使用反标准化后的数据计算验证MSE
            y_test_np = y_test.cpu().numpy()
            prediction_np = prediction.cpu().numpy()
            y_test_original = inverse_transform_power(y_test_np, scaler, power_feature_idx=0)
            prediction_original = inverse_transform_power(prediction_np, scaler, power_feature_idx=0)
            val_mse = MSE(y_test_original, prediction_original)
            
        # Check for early stopping
        if val_mse < best_mse:
            best_mse = val_mse
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss / len(train_dataloader):.4f}, Val MSE: {val_mse:.4f}, Time: {epoch_time:.2f}s")

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # 最终评估
    auto_model.eval()
    with torch.no_grad():
        prediction = auto_model(X_test, test_dict).squeeze(-1)
        
    # 添加调试信息
    print(f"🔍 调试信息:")
    print(f"   输入数据形状: {X_test.shape}")
    print(f"   输入数据前5个样本的均值: {X_test[:5].mean(dim=(1,2)).cpu().numpy()}")
    print(f"   预测结果形状: {prediction.shape}")
    print(f"   预测结果前5个样本: {prediction[:5, 0].cpu().numpy()}")
    print(f"   真实值前5个样本: {y_test[:5, 0].cpu().numpy()}")
    
    # Convert to numpy for evaluation
    y_test_np, prediction_np = y_test.cpu().numpy(), prediction.detach().cpu().numpy()
    
    # 对于标准化数据，进行反标准化处理
    y_test_original = inverse_transform_power(y_test_np, scaler, power_feature_idx=0)
    prediction_original = inverse_transform_power(prediction_np, scaler, power_feature_idx=0)
    
    # Calculate performance metrics using original scale data
    mse_result = MSE(y_test_original, prediction_original)
    mape_result = MAPE(y_test_original, prediction_original)
    rmse_result = np.sqrt(mse_result)
    mae_result = np.mean(np.abs(y_test_original - prediction_original))
    
    print(f"✅ 训练完成 - MSE: {mse_result:.6f}, RMSE: {rmse_result:.6f}, MAE: {mae_result:.6f}, MAPE: {mape_result:.6f}")
    print(f"⏱️  训练时间: {total_training_time:.2f}秒, 收敛轮数: {epoch + 1}")
    
    return {
        'dataset': dataset_name,
        'prediction_scale': prediction_scale,
        'feature_combination': feature_combination_name,
        'feature_list': feature_list,
        'feature_count': len(feature_names),
        'actual_features': feature_names,
        'mse': mse_result,
        'rmse': rmse_result,
        'mae': mae_result,
        'mape': mape_result,
        'training_time': total_training_time,
        'converged_epochs': epoch + 1,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

def create_results_table(results_df, dataset_name, prediction_scale, save_dir):
    """
    创建特征消融实验结果表格
    """
    # 创建格式化的表格
    table_data = results_df.copy()
    
    # 格式化数值
    table_data['RMSE'] = table_data['rmse'].round(6)
    table_data['MAE'] = table_data['mae'].round(6)
    table_data['MAPE (%)'] = (table_data['mape'] * 100).round(2)
    table_data['Training Time (s)'] = table_data['training_time'].round(2)
    table_data['Feature Count'] = table_data['feature_count'].astype(int)
    
    # 计算相对于全特征的性能变化
    if 'all_features' in table_data['feature_combination'].values:
        baseline_rmse = table_data[table_data['feature_combination'] == 'all_features']['rmse'].iloc[0]
        baseline_mae = table_data[table_data['feature_combination'] == 'all_features']['mae'].iloc[0]
        
        table_data['RMSE Change (%)'] = ((table_data['rmse'] - baseline_rmse) / baseline_rmse * 100).round(2)
        table_data['MAE Change (%)'] = ((table_data['mae'] - baseline_mae) / baseline_mae * 100).round(2)
    else:
        table_data['RMSE Change (%)'] = 0.0
        table_data['MAE Change (%)'] = 0.0
    
    # 选择要显示的列
    display_columns = ['feature_combination', 'Feature Count', 'RMSE', 'MAE', 'MAPE (%)', 
                      'RMSE Change (%)', 'MAE Change (%)', 'Training Time (s)']
    table_display = table_data[display_columns]
    
    # 重命名列以便显示
    table_display = table_display.rename(columns={
        'feature_combination': 'Feature Combination'
    })
    
    # 保存为CSV
    table_path = save_dir / f'feature_ablation_results_{dataset_name}_{prediction_scale}.csv'
    table_display.to_csv(table_path, index=False)
    print(f"📋 结果表格已保存到: {table_path}")
    
    # 打印表格
    print(f"\n📋 特征消融实验结果 - {dataset_name} {prediction_scale}")
    print("=" * 120)
    print(table_display.to_string(index=False))
    
    return table_display

def create_visualization(results_df, dataset_name, prediction_scale, save_dir):
    """
    创建特征消融实验的可视化图表
    """
    # 设置绘图样式
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 准备数据
    feature_names = results_df['feature_combination'].tolist()
    feature_counts = results_df['feature_count'].tolist()
    rmse_values = results_df['rmse'].tolist()
    mae_values = results_df['mae'].tolist()
    training_times = results_df['training_time'].tolist()
    
    # 1. RMSE vs 特征数量
    ax1.bar(range(len(feature_names)), rmse_values, color='#1f77b4', alpha=0.7)
    ax1.set_xlabel('Feature Combinations')
    ax1.set_ylabel('RMSE')
    ax1.set_title(f'RMSE by Feature Combination\n{dataset_name} - {prediction_scale}')
    ax1.set_xticks(range(len(feature_names)))
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(rmse_values):
        ax1.text(i, v + max(rmse_values) * 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 2. MAE vs 特征数量
    ax2.bar(range(len(feature_names)), mae_values, color='#ff7f0e', alpha=0.7)
    ax2.set_xlabel('Feature Combinations')
    ax2.set_ylabel('MAE')
    ax2.set_title(f'MAE by Feature Combination\n{dataset_name} - {prediction_scale}')
    ax2.set_xticks(range(len(feature_names)))
    ax2.set_xticklabels(feature_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(mae_values):
        ax2.text(i, v + max(mae_values) * 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 特征数量 vs 性能
    ax3.plot(feature_counts, rmse_values, 'o-', linewidth=2, markersize=8, color='#1f77b4', label='RMSE')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(feature_counts, mae_values, 's-', linewidth=2, markersize=8, color='#ff7f0e', label='MAE')
    
    ax3.set_xlabel('Number of Features')
    ax3.set_ylabel('RMSE', color='#1f77b4')
    ax3_twin.set_ylabel('MAE', color='#ff7f0e')
    ax3.set_title(f'Performance vs Feature Count\n{dataset_name} - {prediction_scale}')
    ax3.grid(True, alpha=0.3)
    
    # 4. 训练时间
    ax4.bar(range(len(feature_names)), training_times, color='#2ca02c', alpha=0.7)
    ax4.set_xlabel('Feature Combinations')
    ax4.set_ylabel('Training Time (seconds)')
    ax4.set_title(f'Training Time by Feature Combination\n{dataset_name} - {prediction_scale}')
    ax4.set_xticks(range(len(feature_names)))
    ax4.set_xticklabels(feature_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(training_times):
        ax4.text(i, v + max(training_times) * 0.01, f'{v:.1f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = save_dir / f'feature_ablation_analysis_{dataset_name}_{prediction_scale}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(plot_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"📊 可视化图表已保存到: {plot_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Feature ablation experiment for input features contribution analysis')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='fujian', choices=['fujian', 'DSWE'], 
                        help='Dataset name')
    parser.add_argument('--prediction_scale', type=str, default='6-0_1', 
                        help='Prediction scale (e.g., 6-0_1, 24-1, etc.)')
    parser.add_argument('--feature_combinations', nargs='+', type=str,
                        default=['power_only', 'power_wind', 'power_wind_core', 'all_features'],
                        help='Feature combinations to test')
    
    # 训练相关参数
    parser.add_argument('--gpu', type=int, default=1, help='GPU device id')
    parser.add_argument('--epochs', type=int, default=30, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--l1_lambda', type=float, default=0.15, help='L1 regularization coefficient')
    parser.add_argument('--weight_decay', type=float, default=0.15, help='L2 weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=2, help='Early stopping patience')
    parser.add_argument('--split_ratio', type=float, default=0.99, help='Train/test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 模型架构参数
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=264, help='Hidden size')
    parser.add_argument('--factor', type=int, default=2, help='Factor for attention')
    parser.add_argument('--conv_hidden_size', type=int, default=32, help='Convolution hidden size')
    parser.add_argument('--moving_avg_window', type=int, default=3, help='Moving average window size')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--encoder_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--seq_length', type=int, default=36, help='Sequence length')
    parser.add_argument('--c_out', type=int, default=6, help='Output channels')
    parser.add_argument('--group_dec', action='store_true', default=True, help='Use group decoder')
    
    args = parser.parse_args()
    
    # 设置随机种子
    seed_everything(seed=args.seed)
    
    # 设置设备
    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建结果保存目录
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    results_dir = script_dir / 'results' / 'feature_ablation_experiment'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 开始特征消融实验")
    print(f"数据集: {args.dataset}")
    print(f"预测尺度: {args.prediction_scale}")
    print(f"特征组合: {args.feature_combinations}")
    print(f"结果保存目录: {results_dir}")
    
    # 获取特征组合定义
    feature_combinations = get_feature_combinations(args.dataset)
    
    # 验证特征组合
    invalid_combinations = [combo for combo in args.feature_combinations if combo not in feature_combinations]
    if invalid_combinations:
        print(f"❌ 无效的特征组合: {invalid_combinations}")
        print(f"可用的特征组合: {list(feature_combinations.keys())}")
        return
    
    # 存储所有结果
    all_results = []
    
    # 对每个特征组合进行实验
    for combo_name in args.feature_combinations:
        feature_list = feature_combinations[combo_name]
        
        try:
            result = train_model_with_features(
                dataset_name=args.dataset,
                prediction_scale=args.prediction_scale,
                feature_combination_name=combo_name,
                feature_list=feature_list,
                args=args,
                device=device
            )
            
            if result is not None:
                all_results.append(result)
            else:
                print(f"⚠️  特征组合 {combo_name} 实验失败，跳过...")
                
        except Exception as e:
            print(f"💥 特征组合 {combo_name} 出现错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("❌ 所有实验都失败了，退出程序")
        return
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 按特征组合顺序排序
    combo_order = ['power_only', 'power_wind', 'power_wind_core', 'all_features']
    results_df['combo_order'] = results_df['feature_combination'].map({combo: i for i, combo in enumerate(combo_order)})
    results_df = results_df.sort_values('combo_order').reset_index(drop=True)
    results_df = results_df.drop('combo_order', axis=1)
    
    # 创建结果表格
    table_display = create_results_table(results_df, args.dataset, args.prediction_scale, results_dir)
    
    # 创建可视化
    create_visualization(results_df, args.dataset, args.prediction_scale, results_dir)
    
    # 保存完整结果
    full_results_path = results_dir / f'full_results_{args.dataset}_{args.prediction_scale}.csv'
    results_df.to_csv(full_results_path, index=False)
    
    # 生成分析总结
    print(f"\n🎉 特征消融实验完成！")
    print(f"📁 所有结果已保存到: {results_dir}")
    print(f"📊 完整结果文件: {full_results_path}")
    
    # 输出关键发现
    print(f"\n📈 关键发现:")
    best_rmse_idx = results_df['rmse'].idxmin()
    best_mae_idx = results_df['mae'].idxmin()
    fastest_idx = results_df['training_time'].idxmin()
    
    print(f"   最佳RMSE: {results_df.loc[best_rmse_idx, 'feature_combination']} ({results_df.loc[best_rmse_idx, 'rmse']:.6f})")
    print(f"   最佳MAE: {results_df.loc[best_mae_idx, 'feature_combination']} ({results_df.loc[best_mae_idx, 'mae']:.6f})")
    print(f"   最快训练: {results_df.loc[fastest_idx, 'feature_combination']} ({results_df.loc[fastest_idx, 'training_time']:.2f}s)")
    
    if len(results_df) > 1:
        rmse_improvement = (results_df['rmse'].max() - results_df['rmse'].min()) / results_df['rmse'].max() * 100
        mae_improvement = (results_df['mae'].max() - results_df['mae'].min()) / results_df['mae'].max() * 100
        print(f"   RMSE改进幅度: {rmse_improvement:.2f}%")
        print(f"   MAE改进幅度: {mae_improvement:.2f}%")

if __name__ == '__main__':
    import sys
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  实验被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 