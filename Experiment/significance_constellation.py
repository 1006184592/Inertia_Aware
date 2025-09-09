import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_significance_constellation():
    """
    创建显著性星状图/雷达图，展示adap_auto与各基线模型的统计显著性对比
    使用福建数据集和DSWE数据集的24-1和24-2预测尺度的p值数据
    """
    # --- 1. 配置绘图样式（参考plot_kl_heatmaps.py） ---
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
    except:
        print("警告: 'Times New Roman' 字体未找到，将使用默认的 serif 字体。")
        
    plt.rcParams['font.size'] = 16          # 基础字号
    plt.rcParams['axes.titlesize'] = 18     # 子图标题
    plt.rcParams['axes.labelsize'] = 16     # 坐标轴标签
    plt.rcParams['xtick.labelsize'] = 14    # X轴刻度标签
    plt.rcParams['ytick.labelsize'] = 12    # Y轴刻度标签
    plt.rcParams['figure.titlesize'] = 20   # 主标题
    plt.rcParams['legend.fontsize'] = 14    # 图例字体

    # --- 2. 读取数据 ---
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # 福建数据集和DSWE数据集的显著性检验结果文件
    fujian_file = os.path.join(current_script_directory, "significance_test_fujian_multi_scale.csv")
    dswe_file = os.path.join(current_script_directory, "significance_test_DSWE_multi_scale.csv")
    
    if not os.path.exists(fujian_file):
        print(f"错误：找不到福建数据集文件: {fujian_file}")
        return
    if not os.path.exists(dswe_file):
        print(f"错误：找不到DSWE数据集文件: {dswe_file}")
        return
    
    # 读取数据
    df_fujian = pd.read_csv(fujian_file)
    df_dswe = pd.read_csv(dswe_file)
    
    # --- 3. 数据筛选和处理 ---
    # 福建数据集: 24-1和24-2预测尺度
    fujian_24_1 = df_fujian[df_fujian['Scale'] == '24-1']
    fujian_24_2 = df_fujian[df_fujian['Scale'] == '24-2']
    
    # DSWE数据集: 24-1和24-2预测尺度
    dswe_24_1 = df_dswe[df_dswe['Scale'] == '24-1']
    dswe_24_2 = df_dswe[df_dswe['Scale'] == '24-2']
    
    # 获取基线模型列表（假设所有数据集都有相同的基线模型）
    baseline_models = fujian_24_1['Baseline_Model'].unique()
    
    # 创建数据字典
    datasets = {
        'Fujian (24-1h)': fujian_24_1,
        'Fujian (24-2h)': fujian_24_2, 
        'DSWE (24-1h)': dswe_24_1,
        'DSWE (24-2h)': dswe_24_2
    }
    
    # --- 4. 创建1x4子图布局 ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 7), subplot_kw=dict(projection='polar'))
    
    # 定义参考p值及其对应的-log10值
    p_values_ref = [0.05, 0.01, 0.001]
    log_p_ref = [-np.log10(p) for p in p_values_ref]
    
    # 定义颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 设置雷达图的最大半径，避免圆太大，同时为极显著点留出空余空间
    MAX_RADIUS = 8  # 对应约p=1e-8的显著性水平
    PLOT_RADIUS = MAX_RADIUS * 1.15  # 实际绘图半径，留出15%的空余空间
    
    # --- 5. 为每个数据集绘制星状图 ---
    for idx, (dataset_name, data) in enumerate(datasets.items()):
        ax = axes[idx]
        
        # 计算角度（基线模型的位置）
        n_models = len(baseline_models)
        angles = np.linspace(0, 2 * np.pi, n_models, endpoint=False).tolist()
        
        # 提取p值并计算-log10(p-value)
        p_values = []
        model_names = []
        extreme_significance_markers = []  # 记录极显著的点
        
        for model in baseline_models:
            model_data = data[data['Baseline_Model'] == model]
            if not model_data.empty:
                p_val = model_data['P_Value'].iloc[0]
                p_values.append(p_val)
                model_names.append(model)
            
        # 计算-log10(p-value)，处理极值情况
        log_p_values = []
        for i, p in enumerate(p_values):
            if p == 0 or p < 1e-50:
                log_p_values.append(MAX_RADIUS)  # 设置为最大半径
                extreme_significance_markers.append(i)  # 标记为极显著
            else:
                log_p = -np.log10(p)
                if log_p > MAX_RADIUS:
                    log_p_values.append(MAX_RADIUS)
                    extreme_significance_markers.append(i)  # 标记为极显著
                else:
                    log_p_values.append(log_p)
        
        # 闭合图形（让第一个点和最后一个点连接）
        angles += angles[:1]
        log_p_values += log_p_values[:1]
        model_names += model_names[:1]
        
        # 绘制星状图
        ax.plot(angles, log_p_values, 'o-', linewidth=2, markersize=6, 
                label=dataset_name, color=colors[idx % len(colors)])
        ax.fill(angles, log_p_values, alpha=0.25, color=colors[idx % len(colors)])
        
        # 为极显著的点添加特殊标记（星号）
        for marker_idx in extreme_significance_markers:
            if marker_idx < len(angles) - 1:  # 避免重复标记闭合点
                ax.plot(angles[marker_idx], log_p_values[marker_idx], 
                       marker='*', markersize=14, color='red', 
                       markeredgecolor='darkred', markeredgewidth=1)
        
        # 添加参考同心圆
        for i, (p_ref, log_p) in enumerate(zip(p_values_ref, log_p_ref)):
            if log_p <= MAX_RADIUS:  # 只显示在范围内的参考线
                circle_angles = np.linspace(0, 2*np.pi, 100)
                circle_radii = [log_p] * len(circle_angles)
                
                if i == 0:  # p=0.05
                    ax.plot(circle_angles, circle_radii, '--', alpha=0.7, color='gray', linewidth=1)
                elif i == 1:  # p=0.01
                    ax.plot(circle_angles, circle_radii, '--', alpha=0.7, color='red', linewidth=1)
                elif i == 2:  # p=0.001
                    ax.plot(circle_angles, circle_radii, '--', alpha=0.7, color='darkred', linewidth=1)
        
        # 设置角度标签（基线模型名称）
        ax.set_xticks(angles[:-1])
        # 简化模型名称以适应更紧凑的布局
        simplified_names = []
        for name in model_names[:-1]:
            if name == 'iTransformer':
                simplified_names.append('iTransf.')
            elif name == 'Autoformer':
                simplified_names.append('AutoF.')
            elif name == 'FEDformer':
                simplified_names.append('FEDF.')
            else:
                simplified_names.append(name)
        ax.set_xticklabels(simplified_names, fontsize=14)
        
        # 设置径向轴
        ax.set_ylim(0, PLOT_RADIUS)
        
        # 添加径向刻度标签
        yticks = [1.3, 2, 3, 4, 6, 8]  # 对应不同的p值水平
        yticks = [y for y in yticks if y <= PLOT_RADIUS]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{val:.1f}' for val in yticks], fontsize=10)
        
        # 设置标题
        ax.set_title(dataset_name, fontsize=16, fontweight='bold', pad=25)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        # 在中心添加"adap_auto"标签
        ax.text(0, 0, 'Ours', ha='center', va='center', 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # --- 6. 添加图例和说明 ---
    # 创建自定义图例，说明参考线的含义
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', alpha=0.7, label='p = 0.05'),
        Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='p = 0.01'),
        Line2D([0], [0], color='darkred', linestyle='--', alpha=0.7, label='p = 0.001'),
        Line2D([0], [0], marker='*', color='red', linestyle='None', 
               markersize=10, label='Extreme significance\n(p < 1e-8)', markeredgecolor='darkred')
    ]
    
    # 在图的底部添加图例
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=4, title='Significance Thresholds', title_fontsize=12)
    
    # 添加总标题
    fig.suptitle('Statistical Significance Constellation Plot: adap_auto vs Baseline Models', 
                 fontsize=18, fontweight='bold', y=0.92)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)  # 为顶部标题和底部图例留出空间
    
    # --- 7. 保存图像 ---
    output_path = os.path.join(current_script_directory, "significance_constellation_plot.pdf")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    
    print(f"\n显著性星状图已保存到:")
    print(f"PDF: {output_path}")
    print(f"PNG: {output_path.replace('.pdf', '.png')}")
    
    # 显示图像
    plt.show()

def print_significance_summary():
    """打印显著性检验结果的汇总信息"""
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    
    print("\n=== 显著性检验结果汇总 ===")
    
    # 读取数据
    fujian_file = os.path.join(current_script_directory, "significance_test_fujian_multi_scale.csv")
    dswe_file = os.path.join(current_script_directory, "significance_test_DSWE_multi_scale.csv")
    
    if os.path.exists(fujian_file):
        df_fujian = pd.read_csv(fujian_file)
        print("\n福建数据集显著性水平统计:")
        
        for scale in ['24-1', '24-2']:
            subset = df_fujian[df_fujian['Scale'] == scale]
            sig_05 = subset['Significant_at_0.05'].sum()
            sig_01 = subset['Significant_at_0.01'].sum()
            total = len(subset)
            print(f"  {scale}预测尺度: {sig_05}/{total} (α=0.05), {sig_01}/{total} (α=0.01)")
    
    if os.path.exists(dswe_file):
        df_dswe = pd.read_csv(dswe_file)
        print("\nDSWE数据集显著性水平统计:")
        
        for scale in ['24-1', '24-2']:
            subset = df_dswe[df_dswe['Scale'] == scale]
            sig_05 = subset['Significant_at_0.05'].sum()
            sig_01 = subset['Significant_at_0.01'].sum()
            total = len(subset)
            print(f"  {scale}预测尺度: {sig_05}/{total} (α=0.05), {sig_01}/{total} (α=0.01)")

    # 打印改进比例摘要
    print("\n=== 改进比例摘要 ===")
    
    if os.path.exists(fujian_file):
        df_fujian = pd.read_csv(fujian_file)
        print("\n福建数据集:")
        for scale in ['24-1', '24-2']:
            subset = df_fujian[df_fujian['Scale'] == scale]
            print(f"  {scale}预测尺度:")
            for _, row in subset.iterrows():
                significance_mark = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
                print(f"    {row['Baseline_Model']:12}: {row['Improvement_Ratio_Percent']:7.2f}% {significance_mark}")
    
    if os.path.exists(dswe_file):
        df_dswe = pd.read_csv(dswe_file)
        print("\nDSWE数据集:")
        for scale in ['24-1', '24-2']:
            subset = df_dswe[df_dswe['Scale'] == scale]
            print(f"  {scale}预测尺度:")
            for _, row in subset.iterrows():
                significance_mark = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
                print(f"    {row['Baseline_Model']:12}: {row['Improvement_Ratio_Percent']:7.2f}% {significance_mark}")

if __name__ == "__main__":
    # 打印汇总信息
    print_significance_summary()
    
    # 绘制星状图
    plot_significance_constellation()