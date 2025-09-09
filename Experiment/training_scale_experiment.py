import os
import sys
# Set CUDA launch blocking for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
import seaborn as sns

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

def is_standardized_data(train_data_path, val_data_path):
    """
    检测数据是否为标准化数据
    Args:
        train_data_path: 训练数据文件路径
        val_data_path: 验证数据文件路径
    Returns:
        bool: True表示是标准化数据，False表示是原始数据
    """
    train_filename = os.path.basename(train_data_path)
    val_filename = os.path.basename(val_data_path)
    
    print(f"🔍 文件名检测: {train_filename}, {val_filename}")
    
    # 主要判断：检查文件名是否明确表示标准化数据
    if train_filename.startswith('std') or val_filename.startswith('std'):
        print(f"✅ 检测结果: 标准化数据 (文件名以'std'开头)")
        return True
    
    if 'std' in train_filename.lower() or 'std' in val_filename.lower():
        print(f"✅ 检测结果: 标准化数据 (文件名包含'std')")
        return True
    
    standardized_keywords = ['standard', 'standardized', 'norm', 'normalized']
    for keyword in standardized_keywords:
        if keyword in train_filename.lower() or keyword in val_filename.lower():
            print(f"✅ 检测结果: 标准化数据 (文件名包含'{keyword}')")
            return True
    
    if ('train_data' in train_filename and 'std' not in train_filename.lower()) or \
       ('val_data' in val_filename and 'std' not in val_filename.lower()):
        print(f"❌ 检测结果: 原始数据 (文件名表明是非标准化数据)")
        return False
    
    data_dir = os.path.dirname(train_data_path)
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    
    if os.path.exists(scaler_path):
        print(f"ℹ️  发现scaler文件: {scaler_path} (但文件名未明确标示为标准化数据)")
        print(f"❌ 检测结果: 原始数据 (优先信任文件名判断)")
        return False
    
    print(f"❌ 检测结果: 原始数据 (未发现标准化数据的明确标志)")
    return False

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

def train_model_with_ratio(dataset_name, prediction_scale, train_ratio, args, device):
    """
    使用指定训练数据比例训练模型
    Args:
        dataset_name: 数据集名称 ('fujian' 或 'DSWE')
        prediction_scale: 预测尺度 (如 '6-0_1')
        train_ratio: 训练数据比例 (0.5-1.0)
        args: 命令行参数
        device: 计算设备
    Returns:
        dict: 包含性能指标和训练信息的字典
    """
    print(f"\n{'='*60}")
    print(f"🚀 开始训练 - 数据集: {dataset_name}, 尺度: {prediction_scale}, 训练比例: {train_ratio*100:.0f}%")
    print(f"{'='*60}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # 上一级
    grandparent_dir = os.path.dirname(parent_dir)  # 上两级
    # 根据数据集设置路径
    if dataset_name.lower() == 'fujian':
        data_dir = os.path.join(grandparent_dir, f'data/fujian')    
        train_dir = os.path.join(data_dir, f'stdtrain_data{prediction_scale}.npy')
        val_dir = os.path.join(data_dir, f'stdval_data{prediction_scale}.npy')
        edge_dir = os.path.join(grandparent_dir, f'new_data/fujian/adag_dict_train_data{prediction_scale}_fused.pkl')
        csv_dir = os.path.join(grandparent_dir, f'data/fujian/Offshore Wind Farm Dataset3(WT1).csv')
    elif dataset_name.lower() == 'dswe':
        data_dir = os.path.join(grandparent_dir, f'data/DSWE')
        train_dir = os.path.join(data_dir, f'stdtrain_data{prediction_scale}.npy')
        val_dir = os.path.join(data_dir, f'stdval_data{prediction_scale}.npy')
        # edge_dir = os.path.join(grandparent_dir, f'new_data/DSWE/adag_dict_train_data{prediction_scale}_fused.pkl')
        edge_dir = os.path.join(grandparent_dir, f'new_data/DSWE/adag_dict_{prediction_scale}.pkl')
        csv_dir = os.path.join(grandparent_dir, f'data/DSWE/Offshore Wind Farm Dataset1(WT5).csv')
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 验证文件存在
    for file_path, file_desc in [(train_dir, "训练数据"), (val_dir, "验证数据"), (edge_dir, "边数据"), (csv_dir, "CSV数据")]:
        if not os.path.exists(file_path):
            print(f"❌ 错误: {file_desc}文件不存在: {file_path}")
            return None
    
    # 检测是否使用标准化数据
    use_standardized = is_standardized_data(train_dir, val_dir)
    print(f"数据类型检测: {'标准化数据' if use_standardized else '原始数据'}")
    
    # 读取CSV元数据
    data = pd.read_csv(csv_dir, nrows=6)
    if dataset_name.lower() == 'dswe':
        cols_to_drop_in_raw_csv = ['Sequence No.']
    elif dataset_name.lower() == 'fujian':
        cols_to_drop_in_raw_csv = ['Site_ID', 'Timestamp']
    df = data.drop(cols_to_drop_in_raw_csv, axis=1)
    
    # 根据数据类型决定是否加载标准化器
    scaler = None
    if use_standardized:
        scaler_path = os.path.join(data_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ 已加载标准化器，特征数量: {len(scaler.mean_)}")
        else:
            print(f"⚠️  警告: 检测到标准化数据但未找到scaler.pkl文件")
            use_standardized = False
    else:
        print(f"ℹ️  使用原始数据，无需加载标准化器")

    # 读取数据
    x_data = torch.tensor(np.load(train_dir)).to(dtype=torch.float32)
    y_data = torch.tensor(np.squeeze(np.load(val_dir)[:, :,0:1], axis=2)).to(dtype=torch.float32)
    x_data, y_data = x_data.to(device), y_data.to(device)
    
    # 读取ADAG边信息
    with open(edge_dir, 'rb') as f: 
        edge_index = pickle.load(f)
    
    # 数据划分 - 先按原比例划分，再调整训练集大小
    original_split_index = int(len(x_data) * args.split_ratio)
    X_original_train, X_test = x_data[0:original_split_index], x_data[original_split_index:]
    y_original_train, y_test = y_data[0:original_split_index], y_data[original_split_index:]
    original_train_dict, test_dict = edge_index[0:original_split_index], edge_index[original_split_index:]
    
    # 根据train_ratio调整训练集大小
    adjusted_train_size = int(len(X_original_train) * train_ratio)
    X_train = X_original_train[:adjusted_train_size]
    y_train = y_original_train[:adjusted_train_size]
    train_dict = original_train_dict[:adjusted_train_size]
    
    print(f"📊 数据统计:")
    print(f"  原始训练集大小: {len(X_original_train)}")
    print(f"  调整后训练集大小: {len(X_train)} ({train_ratio*100:.0f}%)")
    print(f"  测试集大小: {len(X_test)}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    # 初始化模型
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
        c_in=x_data.shape[-1],
        seq_lenth=args.seq_length,
        c_out=args.c_out,
        gruop_dec=args.group_dec,
        train_ratio=train_ratio
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
            
            if use_standardized and scaler is not None:
                # 使用反标准化后的数据计算验证MSE
                y_test_np = y_test.cpu().numpy()
                prediction_np = prediction.cpu().numpy()
                y_test_original = inverse_transform_power(y_test_np, scaler, power_feature_idx=0)
                prediction_original = inverse_transform_power(prediction_np, scaler, power_feature_idx=0)
                val_mse = MSE(y_test_original, prediction_original)
            else:
                val_mse = MSE(y_test, prediction)
            
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
        
    # Convert to numpy for evaluation
    y_test_np, prediction_np = y_test.cpu().numpy(), prediction.detach().cpu().numpy()
    
    if use_standardized and scaler is not None:
        # 对于标准化数据，进行反标准化处理
        y_test_original = inverse_transform_power(y_test_np, scaler, power_feature_idx=0)
        prediction_original = inverse_transform_power(prediction_np, scaler, power_feature_idx=0)
        
        # Calculate performance metrics using original scale data
        mse_result = MSE(y_test_original, prediction_original)
        mape_result = MAPE(y_test_original, prediction_original)
        rmse_result = np.sqrt(mse_result)
        mae_result = np.mean(np.abs(y_test_original - prediction_original))
        
    else:
        # 对于原始数据，直接使用
        mse_result = MSE(y_test_np, prediction_np)
        mape_result = MAPE(y_test_np, prediction_np)
        rmse_result = np.sqrt(mse_result)
        mae_result = np.mean(np.abs(y_test_np - prediction_np))
    
    print(f"✅ 训练完成 - MSE: {mse_result:.6f}, RMSE: {rmse_result:.6f}, MAE: {mae_result:.6f}, MAPE: {mape_result:.6f}")
    print(f"⏱️  训练时间: {total_training_time:.2f}秒, 收敛轮数: {epoch + 1}")
    
    return {
        'dataset': dataset_name,
        'prediction_scale': prediction_scale,
        'train_ratio': train_ratio,
        'mse': mse_result,
        'rmse': rmse_result,
        'mae': mae_result,
        'mape': mape_result,
        'training_time': total_training_time,
        'converged_epochs': epoch + 1,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

def create_visualization(results_df, dataset_name, prediction_scale, save_dir):
    """
    创建训练数据规模影响的可视化图表
    """
    # 设置绘图样式
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. RMSE折线图
    ax1.plot(results_df['train_ratio'] * 100, results_df['rmse'], 'o-', linewidth=2, markersize=8, color='#1f77b4')
    ax1.set_xlabel('Training Data Percentage (%)')
    ax1.set_ylabel('RMSE')
    ax1.set_title(f'RMSE vs Training Data Scale\n{dataset_name} - {prediction_scale}')
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE折线图
    ax2.plot(results_df['train_ratio'] * 100, results_df['mae'], 'o-', linewidth=2, markersize=8, color='#ff7f0e')
    ax2.set_xlabel('Training Data Percentage (%)')
    ax2.set_ylabel('MAE')
    ax2.set_title(f'MAE vs Training Data Scale\n{dataset_name} - {prediction_scale}')
    ax2.grid(True, alpha=0.3)
    
    # 3. 训练时间
    ax3.plot(results_df['train_ratio'] * 100, results_df['training_time'], 'o-', linewidth=2, markersize=8, color='#2ca02c')
    ax3.set_xlabel('Training Data Percentage (%)')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title(f'Training Time vs Training Data Scale\n{dataset_name} - {prediction_scale}')
    ax3.grid(True, alpha=0.3)
    
    # 4. 收敛轮数
    ax4.plot(results_df['train_ratio'] * 100, results_df['converged_epochs'], 'o-', linewidth=2, markersize=8, color='#d62728')
    ax4.set_xlabel('Training Data Percentage (%)')
    ax4.set_ylabel('Converged Epochs')
    ax4.set_title(f'Converged Epochs vs Training Data Scale\n{dataset_name} - {prediction_scale}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = save_dir / f'training_scale_analysis_{dataset_name}_{prediction_scale}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(plot_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"📊 可视化图表已保存到: {plot_path}")
    
    plt.show()

def create_results_table(results_df, dataset_name, prediction_scale, save_dir):
    """
    创建结果表格
    """
    # 创建格式化的表格
    table_data = results_df.copy()
    table_data['Train Ratio (%)'] = (table_data['train_ratio'] * 100).astype(int)
    table_data['RMSE'] = table_data['rmse'].round(6)
    table_data['MAE'] = table_data['mae'].round(6)
    table_data['Training Time (s)'] = table_data['training_time'].round(2)
    table_data['Converged Epochs'] = table_data['converged_epochs'].astype(int)
    table_data['Train Size'] = table_data['train_size'].astype(int)
    
    # 选择要显示的列
    display_columns = ['Train Ratio (%)', 'RMSE', 'MAE', 'Training Time (s)', 'Converged Epochs', 'Train Size']
    table_display = table_data[display_columns]
    
    # 保存为CSV
    table_path = save_dir / f'training_scale_results_{dataset_name}_{prediction_scale}.csv'
    table_display.to_csv(table_path, index=False)
    print(f"📋 结果表格已保存到: {table_path}")
    
    # 打印表格
    print(f"\n📋 训练数据规模影响实验结果 - {dataset_name} {prediction_scale}")
    print("=" * 80)
    print(table_display.to_string(index=False))
    
    return table_display

def main():
    parser = argparse.ArgumentParser(description='Training data scale impact experiment')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='DSWE', choices=['fujian', 'DSWE'], 
                        help='Dataset name')
    parser.add_argument('--prediction_scale', type=str, default='24-2', 
                        help='Prediction scale (e.g., 6-0_1, 24-1, etc.)')
    parser.add_argument('--train_ratios', nargs='+', type=float, 
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='Training data ratios to test')
    
    # 训练相关参数
    parser.add_argument('--gpu', type=int, default=1, help='GPU device id')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--l1_lambda', type=float, default=0.05, help='L1 regularization coefficient')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='L2 weight decay')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
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
    parser.add_argument('--seq_length', type=int, default=144, help='Sequence length')
    parser.add_argument('--c_out', type=int, default=12, help='Output channels')
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
    results_dir = script_dir / 'results' / 'training_scale_experiment'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 开始训练数据规模影响实验")
    print(f"数据集: {args.dataset}")
    print(f"预测尺度: {args.prediction_scale}")
    print(f"训练比例: {args.train_ratios}")
    print(f"结果保存目录: {results_dir}")
    
    # 存储所有结果
    all_results = []
    
    # 对每个训练比例进行实验
    for train_ratio in args.train_ratios:
        try:
            result = train_model_with_ratio(
                dataset_name=args.dataset,
                prediction_scale=args.prediction_scale,
                train_ratio=train_ratio,
                args=args,
                device=device
            )
            
            if result is not None:
                all_results.append(result)
            else:
                print(f"⚠️  训练比例 {train_ratio} 失败，跳过...")
                
        except Exception as e:
            print(f"💥 训练比例 {train_ratio} 出现错误: {e}")
            continue
    
    if not all_results:
        print("❌ 所有实验都失败了，退出程序")
        return
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 创建可视化
    create_visualization(results_df, args.dataset, args.prediction_scale, results_dir)
    
    # 创建结果表格
    table_display = create_results_table(results_df, args.dataset, args.prediction_scale, results_dir)
    
    # 保存完整结果
    full_results_path = results_dir / f'full_results_{args.dataset}_{args.prediction_scale}.csv'
    results_df.to_csv(full_results_path, index=False)
    
    print(f"\n🎉 实验完成！")
    print(f"📁 所有结果已保存到: {results_dir}")
    print(f"📊 完整结果文件: {full_results_path}")

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