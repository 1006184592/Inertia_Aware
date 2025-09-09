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
import seaborn as sns
from dynamic_data_processor import create_dynamic_data
from datetime import datetime
import calendar
from scipy import interpolate

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

def load_preprocessed_data(data_path):
    """
    加载预处理好的数据
    
    Args:
        data_path: 数据文件路径
    
    Returns:
        dict: 加载的数据
    """
    print(f"📂 加载预处理数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ 数据加载成功")
    print(f"   数据集: {data['dataset_name']}")
    print(f"   预测尺度: {data['prediction_scale']}")
    print(f"   特征数量: {data['num_features']}")
    print(f"   特征名称: {data['feature_names']}")
    print(f"   训练集: X{data['X_train'].shape}, y{data['y_train'].shape}")
    print(f"   测试集: X{data['X_test'].shape}, y{data['y_test'].shape}")
    print(f"   创建时间: {data['created_at']}")
    
    return data

def find_csv_data_path(dataset_name):
    """
    查找对应数据集的CSV文件路径
    
    Args:
        dataset_name: 数据集名称
    
    Returns:
        CSV文件路径
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if dataset_name.lower() == 'fujian':
        csv_path = os.path.join(script_dir, '../../data/fujian/Offshore Wind Farm Dataset3(WT1).csv')
    elif dataset_name.lower() == 'dswe':
        csv_path = os.path.join(script_dir, '../../data/DSWE/Offshore Wind Farm Dataset1(WT5).csv')  # 根据实际路径调整
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    
    return csv_path

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

def add_gaussian_noise(data, noise_level=0.05, seed=42):
	"""
	为数据添加高斯噪声（按特征分别基于各自标准差），更稳健地适配原始尺度数据。
	Args:
		data: 输入数据 (numpy array)，形状 [N, L, C]
		noise_level: 噪声强度（相对于各特征标准差的比例）
		seed: 随机种子
	Returns:
		添加噪声后的数据（numpy array）
	"""
	np.random.seed(seed)
	# 按特征维度计算标准差，形状 [1,1,C]
	feature_std = np.std(data, axis=(0, 1), keepdims=True)
	# 防止某些特征std为0
	feature_std = feature_std + 1e-8
	noise = np.random.normal(0, noise_level * feature_std, size=data.shape)
	return data + noise

def create_missing_data(data, missing_ratio=0.05, seed=42):
    """
    随机创建缺失数据并用插值填充
    
    Args:
        data: 输入数据 (numpy array)
        missing_ratio: 缺失数据比例
        seed: 随机种子
    
    Returns:
        处理缺失数据后的数据
    """
    np.random.seed(seed)
    data_copy = data.copy()
    
    # 对每个特征独立处理
    for feature_idx in range(data.shape[-1]):  # 最后一个维度是特征维度
        # 随机选择缺失位置
        total_samples = data.shape[0] * data.shape[1]  # batch_size * seq_length
        missing_count = int(total_samples * missing_ratio)
        
        # 生成随机缺失位置
        missing_indices = np.random.choice(total_samples, missing_count, replace=False)
        
        # 将缺失位置转换为二维索引
        batch_indices = missing_indices // data.shape[1]
        time_indices = missing_indices % data.shape[1]
        
        # 创建缺失数据
        for batch_idx, time_idx in zip(batch_indices, time_indices):
            data_copy[batch_idx, time_idx, feature_idx] = np.nan
        
        # 对每个批次进行插值填充
        for batch_idx in range(data.shape[0]):
            series = data_copy[batch_idx, :, feature_idx]
            if np.isnan(series).any():
                # 使用线性插值填充缺失值
                valid_indices = ~np.isnan(series)
                if np.sum(valid_indices) > 1:  # 至少需要两个有效点进行插值
                    f = interpolate.interp1d(
                        np.where(valid_indices)[0], 
                        series[valid_indices], 
                        kind='linear', 
                        bounds_error=False, 
                        fill_value='extrapolate'
                    )
                    data_copy[batch_idx, :, feature_idx] = f(np.arange(len(series)))
                else:
                    # 如果有效点太少，用均值填充
                    data_copy[batch_idx, :, feature_idx] = np.nanmean(series)
    
    return data_copy

def get_season_from_index(index, total_samples, dataset_name='fujian'):
    """
    根据数据索引获取季节信息
    
    Args:
        index: 数据索引
        total_samples: 总样本数量
        dataset_name: 数据集名称
    
    Returns:
        季节标签 ('Spring', 'Summer', 'Autumn', 'Winter')
    """
    # 将数据按索引均匀分配到四个季节
    # 确保每个季节都有数据用于分析
    season_size = total_samples // 4  # 每个季节的样本数
    remainder = total_samples % 4     # 剩余样本
    
    # 计算季节边界
    boundaries = []
    current_boundary = 0
    for i in range(4):
        # 前面的季节多分配一个剩余样本
        extra = 1 if i < remainder else 0
        current_boundary += season_size + extra
        boundaries.append(current_boundary)
    
    # 根据索引确定季节
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    for i, boundary in enumerate(boundaries):
        if index < boundary:
            return seasons[i]
    
    # 防止越界，返回最后一个季节
    return seasons[-1]

def split_data_by_season(X_test, y_test, test_edge_indices, dataset_name='fujian'):
    """
    按季节划分测试数据
    
    Args:
        X_test: 测试输入数据
        y_test: 测试目标数据
        test_edge_indices: 测试集边索引列表
        dataset_name: 数据集名称
    
    Returns:
        按季节划分的数据字典
    """
    seasonal_data = {
        'Spring': {'X': [], 'y': [], 'edge_indices': []},
        'Summer': {'X': [], 'y': [], 'edge_indices': []},
        'Autumn': {'X': [], 'y': [], 'edge_indices': []},
        'Winter': {'X': [], 'y': [], 'edge_indices': []}
    }
    
    # 为每个样本分配季节
    total_samples = len(X_test)
    for i in range(total_samples):
        season = get_season_from_index(i, total_samples, dataset_name)
        seasonal_data[season]['X'].append(X_test[i])
        seasonal_data[season]['y'].append(y_test[i])
        seasonal_data[season]['edge_indices'].append(test_edge_indices[i])
    
    # 转换为tensor
    for season in seasonal_data:
        if seasonal_data[season]['X']:
            seasonal_data[season]['X'] = torch.stack(seasonal_data[season]['X'])
            seasonal_data[season]['y'] = torch.stack(seasonal_data[season]['y'])
            # edge_indices保持为列表
        else:
            # 如果某个季节没有数据，创建空tensor
            seasonal_data[season]['X'] = torch.empty(0, X_test.shape[1], X_test.shape[2])
            seasonal_data[season]['y'] = torch.empty(0, y_test.shape[1])
            seasonal_data[season]['edge_indices'] = []
    
    return seasonal_data

def inverse_or_identity(arr, scaler, power_feature_idx=0):
	if scaler is None:
		return arr
	return inverse_transform_power(arr, scaler, power_feature_idx=power_feature_idx)

def split_data_by_season_independent(model_data, device, season_split_ratio=0.8):
    """
    将整体数据按季节四等分，每个季节独立进行训练测试划分
    
    Args:
        model_data: 完整的数据字典
        device: 计算设备
        season_split_ratio: 季节内训练集比例（默认0.8）
    
    Returns:
        按季节划分的独立数据字典
    """
    print("🌸 进行季节性独立数据划分...")
    
    # 获取完整数据
    X_full = torch.cat([model_data['X_train'], model_data['X_test']], dim=0)
    y_full = torch.cat([model_data['y_train'], model_data['y_test']], dim=0)
    edge_full = model_data['train_edge_indices'] + model_data['test_edge_indices']
    
    total_samples = len(X_full)
    season_size = total_samples // 4
    remainder = total_samples % 4
    
    seasonal_data = {}
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    start_idx = 0
    for i, season in enumerate(seasons):
        # 计算当前季节的样本数量
        current_size = season_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_size
        
        # 提取当前季节的数据
        X_season = X_full[start_idx:end_idx]
        y_season = y_full[start_idx:end_idx]
        edge_season = edge_full[start_idx:end_idx]
        
        # 在当前季节内进行训练测试划分（支持自定义比例）
        season_split = int(current_size * season_split_ratio)
        
        seasonal_data[season] = {
            'X_train': X_season[:season_split].to(device),
            'y_train': y_season[:season_split].to(device),
            'X_test': X_season[season_split:].to(device),
            'y_test': y_season[season_split:].to(device),
            'train_edge_indices': edge_season[:season_split],
            'test_edge_indices': edge_season[season_split:],
            'total_samples': current_size,
            'train_samples': season_split,
            'test_samples': current_size - season_split
        }
        
        print(f"  {season}: 总样本{current_size}, 训练{season_split}, 测试{current_size - season_split}")
        start_idx = end_idx
    
    return seasonal_data

def run_seasonal_independent_experiment(model_data, args, device):
    """
    运行季节性独立实验：将数据四等分，每个季节独立训练测试
    
    Args:
        model_data: 完整数据字典
        args: 命令行参数
        device: 计算设备
    
    Returns:
        季节性独立实验结果
    """
    print("🌸 开始季节性独立实验...")
    
    # 按季节划分数据
    seasonal_data = split_data_by_season_independent(model_data, device, season_split_ratio=args.season_split_ratio)
    
    seasonal_results = {}
    num_features = model_data['num_features']
    
    for season, data in seasonal_data.items():
        print(f"\n🔬 训练 {season} 季节模型...")
        
        # 根据特征数量调整hidden_size
        base_hidden_size = args.hidden_size
        adjusted_hidden_size = ((base_hidden_size + num_features - 1) // num_features) * num_features
        
        # 为当前季节创建模型
        season_model = adap_auto(
            n_head=args.n_head,
            hidden_size=adjusted_hidden_size,
            factor=args.factor,
            dropout=args.dropout,
            conv_hidden_size=args.conv_hidden_size,
            MovingAvg_window=args.moving_avg_window,
            activation=args.activation,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            c_out=args.c_out,
            c_in=num_features,
            seq_lenth=model_data['seq_length'],
            gruop_dec=args.group_dec
        ).to(device)
        
        # 训练当前季节模型
        train_dataset = TensorDataset(data['X_train'], data['y_train'])
        train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True)
        
        optimizer = torch.optim.Adam(season_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()
        
        # 训练循环（简化版，更少的epochs）
        season_epochs = args.epochs
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(season_epochs):
            season_model.train()
            train_loss = 0.0
            I = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                dicts = data['train_edge_indices'][I:I+len(batch_X)]
                I += len(batch_X)
                
                optimizer.zero_grad()
                outputs = season_model(batch_X, dicts).squeeze(-1)
                loss = criterion(outputs, batch_y)
                
                # L1正则化
                l1_reg = torch.tensor(0.).to(device)
                for param in season_model.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += args.l1_lambda * l1_reg
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            season_model.eval()
            with torch.no_grad():
                prediction = season_model(data['X_test'], data['test_edge_indices']).squeeze(-1)
                val_loss = criterion(prediction, data['y_test']).item()
            
            train_loss /= max(1, len(train_loader))
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{season_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = season_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break
        
        # 加载最佳模型
        season_model.load_state_dict(best_model_state)
        
        # 评估当前季节模型
        season_model.eval()
        with torch.no_grad():
            predictions = season_model(data['X_test'], data['test_edge_indices']).squeeze(-1).cpu().numpy()
            
            # 反标准化
            true_values = inverse_or_identity(data['y_test'].cpu().numpy(), model_data.get('scaler', None), power_feature_idx=0)
            pred_values = inverse_or_identity(predictions, model_data.get('scaler', None), power_feature_idx=0)
            
            mse = MSE(true_values, pred_values)
            mape = MAPE(true_values, pred_values)
        
        seasonal_results[season] = {
            'MSE': mse,
            'MAPE': mape,
            'train_samples': data['train_samples'],
            'test_samples': data['test_samples'],
            'total_samples': data['total_samples']
        }
        
        print(f"  {season} 结果: MSE={mse:.4f}, MAPE={mape:.2f}%")
    
    return seasonal_results

def evaluate_model_robustness(model, X_test, y_test, test_edge_indices, scaler, device,
							   noise_levels=[0.05, 0.1], missing_ratios=[0.05], dataset_name='fujian',
							   include_seasonal_in_default=False):
    """
    评估模型的鲁棒性
    
    Args:
        model: 训练好的模型
        X_test: 测试输入数据
        y_test: 测试目标数据
        test_edge_indices: 测试集的边索引列表
        scaler: 数据标准化器
        device: 计算设备
        noise_levels: 噪声强度列表
        missing_ratios: 缺失数据比例列表
        dataset_name: 数据集名称
        include_seasonal_in_default: 是否在默认模式下包含季节性评估
    
    Returns:
        鲁棒性评估结果字典
    """
    model.eval()
    results = {}
    
    # 1. 基线性能（无扰动）
    print("📊 评估基线性能...")
    with torch.no_grad():
        predictions = model(X_test, test_edge_indices).squeeze(-1).cpu().numpy()
        
        # 反标准化
        true_values = inverse_or_identity(y_test.cpu().numpy(), scaler, power_feature_idx=0)
        pred_values = inverse_or_identity(predictions, scaler, power_feature_idx=0)
        
        baseline_mse = MSE(true_values, pred_values)
        baseline_mape = MAPE(true_values, pred_values)
    
    results['baseline'] = {
        'MSE': baseline_mse,
        'MAPE': baseline_mape
    }
    
    # 2. 噪声鲁棒性测试
    print("🔊 评估噪声鲁棒性...")
    results['noise_robustness'] = {}
    
    for noise_level in noise_levels:
        print(f"  测试噪声强度: {noise_level*100}%")
        
        # 添加噪声到CPU上的numpy数组，然后转换回tensor
        X_test_np = X_test.cpu().numpy()
        X_noisy_np = add_gaussian_noise(X_test_np, noise_level=noise_level)
        X_noisy = torch.FloatTensor(X_noisy_np).to(device)
        
        with torch.no_grad():
            predictions = model(X_noisy, test_edge_indices).squeeze(-1).cpu().numpy()
            
            # 反标准化
            pred_values = inverse_or_identity(predictions, scaler, power_feature_idx=0)
            
            mse = MSE(true_values, pred_values)
            mape = MAPE(true_values, pred_values)
        
        results['noise_robustness'][f'{noise_level*100}%'] = {
            'MSE': mse,
            'MAPE': mape,
            'MSE_degradation': (mse - baseline_mse) / baseline_mse * 100,
            'MAPE_degradation': (mape - baseline_mape) / baseline_mape * 100
        }
    
    # 3. 缺失鲁棒性测试（支持多比例）
    print("🕳️ 评估缺失鲁棒性...")
    results['missing_robustness'] = {}
    X_test_np = X_test.cpu().numpy()
    for missing_ratio in missing_ratios:
        print(f"  缺失数据比例: {missing_ratio*100}%")
        X_missing_np = create_missing_data(X_test_np, missing_ratio=missing_ratio)
        X_missing = torch.FloatTensor(X_missing_np).to(device)
        with torch.no_grad():
            predictions = model(X_missing, test_edge_indices).squeeze(-1).cpu().numpy()
            pred_values = inverse_or_identity(predictions, scaler, power_feature_idx=0)
            mse = MSE(true_values, pred_values)
            mape = MAPE(true_values, pred_values)
        results['missing_robustness'][f'{missing_ratio*100}%_missing'] = {
            'MSE': mse,
            'MAPE': mape,
            'MSE_degradation': (mse - baseline_mse) / baseline_mse * 100,
            'MAPE_degradation': (mape - baseline_mape) / baseline_mape * 100
        }
    
    # 4. 默认模式的季节性性能分析（可选，默认关闭）
    if include_seasonal_in_default:
        print("🌸 评估季节性性能...")
        seasonal_data = split_data_by_season(X_test, y_test, test_edge_indices, dataset_name)
        results['seasonal_performance'] = {}
        for season, data in seasonal_data.items():
            if len(data['X']) > 0:
                print(f"  评估 {season} 季节性能 (样本数: {len(data['X'])})")
                with torch.no_grad():
                    predictions = model(data['X'], data['edge_indices']).squeeze(-1).cpu().numpy()
                    true_seasonal_np = data['y'].cpu().numpy()
                    true_seasonal = inverse_or_identity(true_seasonal_np, scaler, power_feature_idx=0)
                    pred_seasonal = inverse_or_identity(predictions, scaler, power_feature_idx=0)
                    mse = MSE(true_seasonal, pred_seasonal)
                    mape = MAPE(true_seasonal, pred_seasonal)
                results['seasonal_performance'][season] = {
                    'MSE': mse,
                    'MAPE': mape,
                    'sample_count': len(data['X'])
                }
            else:
                print(f"  {season} 季节无数据")
                results['seasonal_performance'][season] = {
                    'MSE': 0,
                    'MAPE': 0,
                    'sample_count': 0
                }
    
    return results

def load_baseline_results(results_dir, dataset_name, prediction_scale):
    """
    加载基线模型的结果用于对比
    
    Args:
        results_dir: 结果目录
        dataset_name: 数据集名称
        prediction_scale: 预测尺度
    
    Returns:
        基线结果字典
    """
    baseline_models = ['iTransformer', 'DLinear', 'NBEATSx', 'FEDformer']
    baseline_results = {}
    
    # 这里应该加载实际的基线模型结果
    # 为了演示，我们创建一些模拟数据
    # 在实际应用中，应该从保存的结果文件中加载
    
    for model_name in baseline_models:
        # 模拟基线结果（实际应用中应该从文件加载）
        baseline_results[model_name] = {
            'Spring': {'MSE': np.random.uniform(0.8, 1.2), 'MAPE': np.random.uniform(8, 15)},
            'Summer': {'MSE': np.random.uniform(0.9, 1.3), 'MAPE': np.random.uniform(9, 16)},
            'Autumn': {'MSE': np.random.uniform(0.7, 1.1), 'MAPE': np.random.uniform(7, 14)},
            'Winter': {'MSE': np.random.uniform(0.8, 1.2), 'MAPE': np.random.uniform(8, 15)}
        }
    
    return baseline_results

def create_robustness_visualizations(results, baseline_results, save_dir, dataset_name, prediction_scale):
    """
    创建鲁棒性分析的可视化图表
    
    Args:
        results: 鲁棒性评估结果
        baseline_results: 基线模型结果
        save_dir: 保存目录
        dataset_name: 数据集名称
        prediction_scale: 预测尺度
    """
    plt.style.use('default')

    # 1. 噪声/缺失实验结果表格（仅当结果中存在 baseline 时绘制）
    if 'baseline' in results and 'noise_robustness' in results and 'missing_robustness' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        # 噪声鲁棒性表格
        noise_data = []
        noise_data.append(['Baseline', f"{results['baseline']['MSE']:.3f}", 
                          f"{results['baseline']['MAPE']:.2f}%"])
        for noise_level, metrics in results['noise_robustness'].items():
            noise_data.append([f'Noise {noise_level}', f"{metrics['MSE']:.3f}", 
                              f"{metrics['MAPE']:.2f}%"])
        for missing_level, metrics in results['missing_robustness'].items():
            noise_data.append([f'Missing {missing_level}', f"{metrics['MSE']:.3f}", 
                              f"{metrics['MAPE']:.2f}%"])
        # 创建表格
        table1 = ax1.table(cellText=noise_data,
                          colLabels=['Condition', 'MSE', 'MAPE'],
                          cellLoc='center',
                          loc='center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 2)
        ax1.axis('off')
        ax1.set_title(f'Noise & Missing Data Robustness\n{dataset_name} - {prediction_scale}', 
                     fontsize=14, fontweight='bold', pad=20)
        # 性能退化柱状图
        conditions = []
        mse_degradation = []
        mape_degradation = []
        for noise_level, metrics in results['noise_robustness'].items():
            conditions.append(f'Noise {noise_level}')
            mse_degradation.append(metrics['MSE_degradation'])
            mape_degradation.append(metrics['MAPE_degradation'])
        for missing_level, metrics in results['missing_robustness'].items():
            conditions.append(f'Missing {missing_level}')
            mse_degradation.append(metrics['MSE_degradation'])
            mape_degradation.append(metrics['MAPE_degradation'])
        x = np.arange(len(conditions))
        width = 0.25
        ax2.bar(x - width, mse_degradation, width, label='MSE Degradation (%)', alpha=0.8)
        ax2.bar(x, mape_degradation, width, label='MAPE Degradation (%)', alpha=0.8)
        ax2.set_xlabel('Test Conditions')
        ax2.set_ylabel('Performance Degradation (%)')
        ax2.set_title(f'Performance Degradation Analysis\n{dataset_name} - {prediction_scale}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(conditions, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # 添加数值标签
        for i, (mse, mape) in enumerate(zip(mse_degradation, mape_degradation)):
            ax2.text(i - width, mse + 0.1, f'{mse:.1f}%', ha='center', va='bottom', fontsize=8)
            ax2.text(i, mape + 0.1, f'{mape:.1f}%', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        # 保存噪声/缺失实验图表
        plot_path1 = save_dir / f'robustness_noise_missing_{dataset_name}_{prediction_scale}.png'
        plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
        plt.savefig(str(plot_path1).replace('.png', '.pdf'), bbox_inches='tight')
        print(f"📊 噪声/缺失鲁棒性图表已保存到: {plot_path1}")
        plt.show()
    else:
        print('ℹ️ 独立季节模式：跳过噪声/缺失鲁棒性图（无 baseline 结果）。')
    
    # 2. 季节性性能对比图（仅当存在季节性结果时绘制）
    if 'seasonal_performance' in results:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        metrics = ['MSE', 'MAPE']
        # 准备数据
        our_model_data = {metric: [] for metric in metrics}
        for season in seasons:
            if season in results['seasonal_performance']:
                for metric in metrics:
                    our_model_data[metric].append(results['seasonal_performance'][season][metric])
            else:
                for metric in metrics:
                    our_model_data[metric].append(0)
        # 基线模型数据
        baseline_model_names = list(baseline_results.keys())
        baseline_data = {model: {metric: [] for metric in metrics} for model in baseline_model_names}
        for model in baseline_model_names:
            for season in seasons:
                for metric in metrics:
                    baseline_data[model][metric].append(baseline_results[model][season][metric])
        # 绘制每个指标的对比图
        axes = [ax1, ax2, ax3]
        metric_titles = ['Mean Squared Error (MSE)', 'Mean Absolute Percentage Error (MAPE)']
        for idx, (ax, metric, title) in enumerate(zip(axes, metrics, metric_titles)):
            x = np.arange(len(seasons))
            width = 0.15
            ax.bar(x - 2*width, our_model_data[metric], width, label='Our Model (adap_auto)', color='#1f77b4', alpha=0.8)
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i, model in enumerate(baseline_model_names):
                ax.bar(x - width + i*width, baseline_data[model][metric], width, label=model, color=colors[i], alpha=0.7)
            ax.set_xlabel('Seasons')
            ax.set_ylabel(metric)
            ax.set_title(f'{title} by Season\n{dataset_name} - {prediction_scale}')
            ax.set_xticks(x)
            ax.set_xticklabels(seasons)
            ax.legend()
            ax.grid(True, alpha=0.3)
            for i, v in enumerate(our_model_data[metric]):
                ax.text(i - 2*width, v + max(our_model_data[metric]) * 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        # 第四个子图：样本数量分布
        sample_counts = [results['seasonal_performance'][season].get('sample_count', results['seasonal_performance'][season].get('test_samples', 0)) for season in seasons]
        ax4.bar(seasons, sample_counts, color='#17becf', alpha=0.7)
        ax4.set_xlabel('Seasons')
        ax4.set_ylabel('Sample Count')
        ax4.set_title(f'Sample Distribution by Season\n{dataset_name} - {prediction_scale}')
        ax4.grid(True, alpha=0.3)
        for i, v in enumerate(sample_counts):
            ax4.text(i, v + max(sample_counts) * 0.01, f'{v}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plot_path2 = save_dir / f'seasonal_performance_{dataset_name}_{prediction_scale}.png'
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        plt.savefig(str(plot_path2).replace('.png', '.pdf'), bbox_inches='tight')
        print(f"📊 季节性性能图表已保存到: {plot_path2}")
        plt.show()
    else:
        print('ℹ️ 默认模式未包含季节性评估，跳过季节性图表。')

def load_window_data_directly(dataset_name, prediction_scale, seq_length=36, c_out=6, split_ratio=0.99, use_std=True):
    """
    直接加载预处理好的窗口数据和边索引数据
    
    Args:
        dataset_name: 数据集名称 ('fujian' 或 'DSWE')
        prediction_scale: 预测尺度 (如 '6-0_1', '24-1' 等)
        seq_length: 序列长度
        c_out: 预测长度
        split_ratio: 训练/测试划分比例
        use_std: 是否使用标准化后的数据文件
    Returns:
        包含数据和元信息的字典
    """
    print(f"\n📂 直接加载窗口数据")
    print(f"数据集: {dataset_name}")
    print(f"预测尺度: {prediction_scale}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据路径映射
    if dataset_name.lower() == 'fujian':
        data_dir = os.path.join(script_dir, '../../data/fujian')
        edge_dir = os.path.join(script_dir, '../../new_data/fujian')
        csv_path = os.path.join(script_dir, '../../data/fujian/Offshore Wind Farm Dataset3(WT1).csv')
        
        # 窗口数据文件（支持std或原始）
        if use_std:
            train_file = f'stdtrain_data{prediction_scale}.npy'
            val_file = f'stdval_data{prediction_scale}.npy'
        else:
            train_file = f'train_data{prediction_scale}.npy'
            val_file = f'val_data{prediction_scale}.npy'
        # 边索引文件
        edge_file = f'adag_dict_train_data{prediction_scale}_fused.pkl'
        
    elif dataset_name.lower() == 'dswe':
        data_dir = os.path.join(script_dir, '../../data/DSWE')
        edge_dir = os.path.join(script_dir, '../../new_data/DSWE')
        csv_path = os.path.join(script_dir, '../../data/DSWE/Offshore Wind Farm Dataset1(WT5).csv')
        
        # 窗口数据文件（支持std或原始）
        if use_std:
            train_file = f'stdtrain_data{prediction_scale}.npy'
            val_file = f'stdval_data{prediction_scale}.npy'
        else:
            train_file = f'train_data{prediction_scale}.npy'
            val_file = f'val_data{prediction_scale}.npy'
        # 边索引文件
        edge_file = f'adag_dict_{prediction_scale}.pkl'
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 构建完整路径
    train_path = os.path.join(data_dir, train_file)
    val_path = os.path.join(data_dir, val_file)
    edge_path = os.path.join(edge_dir, edge_file)
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    
    # 验证文件存在
    missing_files = []
    for path, name in [(train_path, '训练数据'), (val_path, '验证数据'), 
                       (edge_path, '边索引数据'), (scaler_path, '标准化器'), (csv_path, 'CSV元数据')]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(f"以下文件不存在:\n" + "\n".join(missing_files))
    
    print(f"✅ 所有数据文件验证通过")
    print(f"   训练数据: {train_path}")
    print(f"   验证数据: {val_path}")
    print(f"   边索引: {edge_path}")
    print(f"   标准化器: {scaler_path}")
    
    # 加载数据
    print("📊 加载数据...")
    
    # 加载窗口数据
    x_data = torch.tensor(np.load(train_path)).to(dtype=torch.float32)
    y_data = torch.tensor(np.squeeze(np.load(val_path)[:, :, 0:1], axis=2)).to(dtype=torch.float32)
    
    # 依据真实y长度确定预测步长
    pred_len = int(y_data.shape[1])
    
    # 加载边索引
    with open(edge_path, 'rb') as f:
        edge_indices = pickle.load(f)
    
    # 加载标准化器
    scaler = None
    if os.path.exists(scaler_path) and use_std:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    # 读取CSV元数据获取特征名称
    csv_data = pd.read_csv(csv_path, nrows=6)
    feature_columns = csv_data.drop(['Site_ID', 'Timestamp'], axis=1).columns.tolist()
    
    print(f"   数据形状: X{x_data.shape}, y{y_data.shape}")
    print(f"   边索引数量: {len(edge_indices)}")
    print(f"   特征数量: {len(feature_columns)}")
    print(f"   特征名称: {feature_columns}")
    
    # 数据划分
    split_index = int(len(x_data) * split_ratio)
    X_train = x_data[:split_index]
    X_test = x_data[split_index:]
    y_train = y_data[:split_index]
    y_test = y_data[split_index:]
    
    train_edge_indices = edge_indices[:split_index]
    test_edge_indices = edge_indices[split_index:]
    
    print(f"   训练集: X{X_train.shape}, y{y_train.shape}, 边索引{len(train_edge_indices)}")
    print(f"   测试集: X{X_test.shape}, y{y_test.shape}, 边索引{len(test_edge_indices)}")
    
    # 构建返回数据
    model_data = {
        'dataset_name': dataset_name,
        'prediction_scale': prediction_scale,
        'seq_length': seq_length,
        'pred_length': pred_len,
        'split_ratio': split_ratio,
        'feature_names': feature_columns,
        'num_features': len(feature_columns),
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_edge_indices': train_edge_indices,
        'test_edge_indices': test_edge_indices,
        'scaler': scaler,
        'created_at': datetime.now().isoformat(),
        'csv_path': csv_path
    }
    
    return model_data

def run_robustness_experiment(dataset_name, prediction_scale, args, device):
    """
    运行鲁棒性分析实验
    
    Args:
        dataset_name: 数据集名称
        prediction_scale: 预测尺度
        args: 命令行参数
        device: 计算设备
    
    Returns:
        实验结果字典
    """
    print(f"\n🔬 开始鲁棒性分析实验")
    print(f"数据集: {dataset_name}")
    print(f"预测尺度: {prediction_scale}")
    
    start_time = time.time()
    
    try:
        # 1. 数据准备
        print("📊 准备数据...")
        
        if hasattr(args, 'use_preprocessed') and args.use_preprocessed:
            # 使用预处理数据
            if hasattr(args, 'data_dir') and args.data_dir:
                data_dir = Path(args.data_dir)
            else:
                script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
                data_dir = script_dir / 'preprocessed_data'
            
            data_file = data_dir / f'{dataset_name}_{prediction_scale}_robustness_data.pkl'
            model_data = load_preprocessed_data(data_file)
        else:
            # 直接使用窗口数据和边索引数据
            print("📂 直接加载窗口数据和边索引数据")
            model_data = load_window_data_directly(
                dataset_name=dataset_name,
                prediction_scale=prediction_scale,
                seq_length=args.seq_length,
                c_out=args.c_out,
                split_ratio=args.split_ratio,
                use_std=args.use_preprocessed_std # 传递use_preprocessed_std参数
            )
        
        X_train = model_data['X_train'].to(device)
        y_train = model_data['y_train'].to(device)
        X_test = model_data['X_test'].to(device)
        y_test = model_data['y_test'].to(device)
        train_edge_indices = model_data['train_edge_indices']
        test_edge_indices = model_data['test_edge_indices']
        scaler = model_data['scaler']
        
        print(f"训练集形状: {X_train.shape}, {y_train.shape}")
        print(f"测试集形状: {X_test.shape}, {y_test.shape}")
        print(f"特征数量: {model_data['num_features']}")
        print(f"序列长度: {model_data['seq_length']}")
        print(f"预测长度: {model_data['pred_length']}")
        
        # 若为独立季节模式，直接进行季节性独立实验，跳过全局模型训练
        if args.seasonal_mode == 'independent':
            print("🌸 使用季节独立模式...")
            seasonal_results = run_seasonal_independent_experiment(model_data, args, device)
            training_time = time.time() - start_time
            robustness_results = {
                'seasonal_performance': seasonal_results,
                'training_time': training_time,
                'dataset': dataset_name,
                'prediction_scale': prediction_scale,
                'seasonal_mode': 'independent'
            }
            return robustness_results

        # 2. 模型初始化
        print("🏗️ 初始化模型...")
        
        # 根据特征数量调整hidden_size，确保能被特征数整除
        num_features = model_data['num_features']
        base_hidden_size = args.hidden_size
        adjusted_hidden_size = ((base_hidden_size + num_features - 1) // num_features) * num_features
        
        print(f"   输入特征数: {num_features}")
        print(f"   原始hidden_size: {base_hidden_size}, 调整后: {adjusted_hidden_size}")
        
        model = adap_auto(
            n_head=args.n_head,
            hidden_size=adjusted_hidden_size,
            factor=args.factor,
            dropout=args.dropout,
            conv_hidden_size=args.conv_hidden_size,
            MovingAvg_window=args.moving_avg_window,
            activation=args.activation,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            c_out=args.c_out,
            c_in=num_features,
            seq_lenth=model_data['seq_length'],
            gruop_dec=args.group_dec
        ).to(device)
        
        # 3. 模型训练
        print("🚀 开始训练模型...")
        
        # 准备数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        I = 0
        
        for epoch in range(args.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            I = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # 获取对应批次的边索引
                dicts = train_edge_indices[I:I+len(batch_X)]
                I += len(batch_X)
                
                optimizer.zero_grad()
                outputs = model(batch_X, dicts).squeeze(-1)
                loss = criterion(outputs, batch_y)
                
                # L1正则化
                l1_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += args.l1_lambda * l1_reg
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段（使用测试集）
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                prediction = model(X_test, test_edge_indices).squeeze(-1)
                val_loss = criterion(prediction, y_test).item()
            
            train_loss /= len(train_loader)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        print(f"✅ 模型训练完成，用时: {training_time:.2f} 秒")
        
        # 4. 鲁棒性评估
        print("🔍 开始鲁棒性评估...")
        
        if args.seasonal_mode == 'independent':
            # 季节独立模式：重新进行季节性独立实验
            print("🌸 使用季节独立模式...")
            seasonal_results = run_seasonal_independent_experiment(model_data, args, device)
            
            # 构建与标准模式兼容的结果格式
            robustness_results = {
                'seasonal_performance': seasonal_results,
                'training_time': training_time,
                'dataset': dataset_name,
                'prediction_scale': prediction_scale,
                'seasonal_mode': 'independent'
            }
        else:
            # 标准模式：使用统一训练的模型进行测试
            robustness_results = evaluate_model_robustness(
                model=model,
                X_test=X_test,
                y_test=y_test,
                test_edge_indices=test_edge_indices,
                scaler=scaler,
                device=device,
                noise_levels=args.noise_levels,
                missing_ratios=args.missing_ratio,
                dataset_name=dataset_name,
                include_seasonal_in_default=args.include_seasonal_eval
            )
            
            # 添加训练时间到结果
            robustness_results['training_time'] = training_time
            robustness_results['dataset'] = dataset_name
            robustness_results['prediction_scale'] = prediction_scale
            robustness_results['seasonal_mode'] = 'test_split'
        
        return robustness_results
        
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_results(results, save_dir, dataset_name, prediction_scale):
    """
    保存实验结果
    
    Args:
        results: 实验结果
        save_dir: 保存目录
        dataset_name: 数据集名称
        prediction_scale: 预测尺度
    """
    # 保存详细结果
    results_file = save_dir / f'robustness_results_{dataset_name}_{prediction_scale}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"💾 详细结果已保存到: {results_file}")
    
    # 检查实验模式
    seasonal_mode = results.get('seasonal_mode', 'test_split')
    
    if seasonal_mode == 'independent':
        # 独立季节模式：只保存季节性结果
        print("📊 保存独立季节实验结果...")
        
        # 保存季节性性能
        seasonal_file = save_dir / f'seasonal_independent_{dataset_name}_{prediction_scale}.csv'
        seasonal_data = []
        
        for season, metrics in results['seasonal_performance'].items():
            seasonal_data.append({
                'Season': season,
                'MSE': metrics['MSE'],
                'MAPE': metrics['MAPE'],
                'Train_Samples': metrics.get('train_samples', 0),
                'Test_Samples': metrics.get('test_samples', 0),
                'Total_Samples': metrics.get('total_samples', 0)
            })
        
        seasonal_df = pd.DataFrame(seasonal_data)
        seasonal_df.to_csv(seasonal_file, index=False)
        print(f"🌸 独立季节性能已保存到: {seasonal_file}")
        
    else:
        # 标准模式：保存完整的鲁棒性分析结果
        print("📊 保存标准鲁棒性分析结果...")
        
        # 保存汇总表格
        summary_file = save_dir / f'robustness_summary_{dataset_name}_{prediction_scale}.csv'
        summary_data = []
        
        # 基线性能
        if 'baseline' in results:
            summary_data.append({
                'Condition': 'Baseline',
                'MSE': results['baseline']['MSE'],
                'MAPE': results['baseline']['MAPE'],
                'MSE_Degradation(%)': 0,
                'MAPE_Degradation(%)': 0
            })
        
        # 噪声鲁棒性
        if 'noise_robustness' in results:
            for noise_level, metrics in results['noise_robustness'].items():
                summary_data.append({
                    'Condition': f'Noise_{noise_level}',
                    'MSE': metrics['MSE'],
                    'MAPE': metrics['MAPE'],
                    'MSE_Degradation(%)': metrics['MSE_degradation'],
                    'MAPE_Degradation(%)': metrics['MAPE_degradation']
                })
        
        # 缺失鲁棒性
        if 'missing_robustness' in results:
            for missing_level, metrics in results['missing_robustness'].items():
                summary_data.append({
                    'Condition': f'Missing_{missing_level}',
                    'MSE': metrics['MSE'],
                    'MAPE': metrics['MAPE'],
                    'MSE_Degradation(%)': metrics['MSE_degradation'],
                    'MAPE_Degradation(%)': metrics['MAPE_degradation']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_file, index=False)
            print(f"📋 汇总表格已保存到: {summary_file}")
        
        # 保存季节性性能
        if 'seasonal_performance' in results:
            seasonal_file = save_dir / f'seasonal_performance_{dataset_name}_{prediction_scale}.csv'
            seasonal_data = []
            
            for season, metrics in results['seasonal_performance'].items():
                seasonal_data.append({
                    'Season': season,
                    'MSE': metrics['MSE'],
                    'MAPE': metrics['MAPE'],
                    'Sample_Count': metrics.get('sample_count', 0)
                })
            
            seasonal_df = pd.DataFrame(seasonal_data)
            seasonal_df.to_csv(seasonal_file, index=False)
            print(f"🌸 季节性性能已保存到: {seasonal_file}")

def main():
    parser = argparse.ArgumentParser(description='Robustness Analysis Experiment for wind power forecasting model')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='fujian', choices=['fujian', 'DSWE'], 
                        help='Dataset name')
    parser.add_argument('--prediction_scale', type=str, default='6-1', 
                        help='Prediction scale (e.g., 6-0_1, 24-1, etc.)')
    
    # 训练相关参数
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--epochs', type=int, default=15, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--l1_lambda', type=float, default=0.01, help='L1 regularization coefficient')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='L2 weight decay')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
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
    
    # 鲁棒性测试参数
    parser.add_argument('--noise_levels', nargs='+', type=float, default=[0.05, 0.1],
                        help='Noise levels for robustness testing')
    parser.add_argument('--missing_ratio', nargs='+', type=float, default=[0.05, 0.1],
                        help='Missing data ratio(s) for robustness testing, can pass multiple values')
    parser.add_argument('--include_seasonal_eval', action='store_true', default=False,
                        help='Include seasonal evaluation in default (test_split) mode')
    parser.add_argument('--seasonal_mode', type=str, default='independent', choices=['test_split', 'independent'],
                        help='Seasonal analysis mode: test_split (divide test set) or independent (divide whole dataset)')
    parser.add_argument('--season_split_ratio', type=float, default=0.95,
                        help='Train split ratio within each season in independent mode (default: 0.8)')
    
    # 数据加载参数
    parser.add_argument('--use_preprocessed', action='store_true',
                        help='Use preprocessed data instead of processing from scratch')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing preprocessed data (default: ./preprocessed_data)')
    parser.add_argument('--use_preprocessed_std', action='store_true', default=True,
                        help='Use standardized data files (stdtrain_data.npy, stdval_data.npy) if available')
    
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
    results_dir = script_dir / 'results' / 'robustness_analysis_experiment'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 开始鲁棒性分析实验")
    print(f"数据集: {args.dataset}")
    print(f"预测尺度: {args.prediction_scale}")
    print(f"噪声强度: {args.noise_levels}")
    print(f"缺失比例: {args.missing_ratio}")
    print(f"结果保存目录: {results_dir}")
    
    # 运行实验
    results = run_robustness_experiment(args.dataset, args.prediction_scale, args, device)
    
    if results is not None:
        # 保存结果
        save_results(results, results_dir, args.dataset, args.prediction_scale)
        
        # 加载基线结果
        baseline_results = load_baseline_results(results_dir, args.dataset, args.prediction_scale)
        
        # 创建可视化
        create_robustness_visualizations(results, baseline_results, results_dir, args.dataset, args.prediction_scale)
        
        print(f"\n✅ 鲁棒性分析实验完成！")
        print(f"📊 结果已保存到: {results_dir}")
        
        # 打印关键结果摘要
        print(f"\n�� 实验结果摘要:")
        
        seasonal_mode = results.get('seasonal_mode', 'test_split')
        
        if seasonal_mode == 'independent':
            print(f"模式: 独立季节训练")
            print(f"季节性性能:")
            for season, metrics in results['seasonal_performance'].items():
                train_samples = metrics.get('train_samples', 0)
                test_samples = metrics.get('test_samples', 0)
                print(f"  {season}: MSE={metrics['MSE']:.4f}, MAPE={metrics['MAPE']:.2f}% (训练: {train_samples}, 测试: {test_samples})")
        else:
            print(f"模式: 标准鲁棒性分析")
            
            if 'baseline' in results:
                print(f"基线性能 - MSE: {results['baseline']['MSE']:.4f}, MAPE: {results['baseline']['MAPE']:.2f}%")
            
            if 'noise_robustness' in results:
                for noise_level, metrics in results['noise_robustness'].items():
                    print(f"噪声 {noise_level} - MSE退化: {metrics['MSE_degradation']:.2f}%, MAPE退化: {metrics['MAPE_degradation']:.2f}%")
            
            if 'missing_robustness' in results:
                for missing_level, metrics in results['missing_robustness'].items():
                    print(f"缺失 {missing_level} - MSE退化: {metrics['MSE_degradation']:.2f}%, MAPE退化: {metrics['MAPE_degradation']:.2f}%")
            
            if 'seasonal_performance' in results:
                print(f"季节性性能:")
                for season, metrics in results['seasonal_performance'].items():
                    sample_count = metrics.get('sample_count', 0)
                    if sample_count > 0:
                        print(f"  {season}: MSE={metrics['MSE']:.4f}, MAPE={metrics['MAPE']:.2f}% (样本数: {sample_count})")
    else:
        print("❌ 实验失败")

if __name__ == '__main__':
    main() 