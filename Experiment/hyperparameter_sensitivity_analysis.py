#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis Visualization Script (MSE Version).
Correctly isolates one-at-a-time parameter variations for plotting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import matplotlib
import os
from matplotlib import font_manager

warnings.filterwarnings('ignore')
# 第1步：强制matplotlib在PDF/PS中嵌入完整的TrueType字体
# 这个设置对于生成出版级PDF至关重要
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 第2步：首先设置Seaborn的全局样式
# 这会建立一个基础样式，我们再在上面进行自定义修改
sns.set_style("whitegrid")

# 第3步：在Seaborn样式之上，强制设置我们的自定义字体和字号
# 这种“后发制人”的策略，可以防止我们的设置被Seaborn覆盖
font_path = '/home/forecasting/.local/share/fonts/times.ttf'
if os.path.exists(font_path):
    # 将字体文件添加到matplotlib的字体管理器中
    font_manager.fontManager.addfont(font_path)
    # 设置rcParams直接使用该字体的名称
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    print(f"✅ 成功加载并设置字体: {prop.get_name()} from {font_path}")
else:
    print(f"⚠️ 警告: 找不到指定的字体文件 {font_path}。将回退到默认字体。")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

# 第4步：设置所有其他样式参数
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20


def load_and_filter_data(csv_path, target_dataset='24-2'):
    """
    Loads and filters data for a specific dataset.
    """
    print(f"📊 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    dataset_data = df[df['dataset'] == target_dataset].copy()
    print(f"📊 Found {len(dataset_data)} results for dataset '{target_dataset}'")
    return dataset_data


def create_sensitivity_plots(df, save_path):
    """
    Creates a "small multiples" plot with precise one-at-a-time filtering.
    """
    print("📊 Generating hyperparameter sensitivity plot with precise filtering...")

    # --- 核心修改：明确定义默认/基准配置 ---
    # 假设基准配置是您想作为参考的标准配置
    baseline_config = {
        'lr': 0.0002,
        'dropout': 0.05,
        'l1_lambda': 0.05,
        'weight_decay': 0.05,
        'batch_size': 128
    }

    # 在DataFrame中找到完全匹配基准配置的行
    baseline_query = ' & '.join([f"`{k}` == {v}" for k, v in baseline_config.items()])
    baseline_row = df.query(baseline_query)

    if baseline_row.empty:
        print("⚠️ Warning: Baseline configuration not found in the data. Using the first row as a fallback.")
        baseline_mse = df.iloc[0]['mse']
    else:
        baseline_mse = baseline_row.iloc[0]['mse']

    param_config = {
        'lr': {'name': 'Learning Rate', 'values': sorted(df['lr'].unique())},
        'dropout': {'name': 'Dropout Rate', 'values': sorted(df['dropout'].unique())},
        'l1_lambda': {'name': 'L1 Lambda', 'values': sorted(df['l1_lambda'].unique())},
        'weight_decay': {'name': 'Weight Decay', 'values': sorted(df['weight_decay'].unique())},
        'batch_size': {'name': 'Batch Size', 'values': sorted(df['batch_size'].unique())}
    }

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    axes = axes.flatten()

    for i, (param_key, config) in enumerate(param_config.items()):
        ax = axes[i]
        
        values_tested = config['values']
        mses_for_plot = []

        for value in values_tested:
            # --- 核心修改：精确筛选逻辑 ---
            current_config = baseline_config.copy()
            current_config[param_key] = value # 只改变当前正在分析的参数
            
            # 构建精确的查询语句
            query = ' & '.join([f"`{k}` == {v}" for k, v in current_config.items()])
            subset_df = df.query(query)

            if not subset_df.empty:
                # 找到了唯一对应的实验，取其MSE
                mses_for_plot.append(subset_df.iloc[0]['mse'])
            else:
                # 如果找不到完全匹配的实验（例如，在您的日志中某些组合可能没跑）
                # 我们用NaN来表示，这样图中会断开，表示数据缺失
                mses_for_plot.append(np.nan)
        
        # --- 使用修正后的数据进行绘图 ---
        # 注意：因为我们现在是精确匹配单次实验，所以不再有标准差，改用简单的点线图
        ax.plot(range(len(values_tested)), mses_for_plot, marker='o', linestyle='--',
                markersize=8, color='#1B9E77', label='MSE')

        ax.axhline(y=baseline_mse, color='#D95F02', linestyle=':', 
                    label=f'Baseline')
        
        # Formatting
        ax.set_xticks(range(len(values_tested)))
        ax.set_xticklabels([str(v) for v in values_tested])
        ax.set_title(f'{config["name"]}', fontsize=18, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=16)
        ax.set_xlabel('Parameter Value', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.6)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Hyperparameter Sensitivity Analysis (One-at-a-Time)', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved successfully to: {save_path}")
    plt.close()


def main():
    
    # ... (main function logic is the same as your version) ...
    print("🚀 Starting hyperparameter sensitivity analysis...")
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    df = load_and_filter_data(csv_path, target_dataset)
    if df.empty: return
    save_path = save_dir_path / 'hyperparameter_sensitivity_analysis_MSE_corrected.pdf'
    create_sensitivity_plots(df, save_path)
    print(f"📋 Analysis complete! Plot saved to {save_path.name}")

if __name__ == '__main__':
    main()