import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import shap
import time
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import os
import random

# --- 自定义模型和评估函数 ---
# 确保你的模块可以被正确导入
from adap_auto import adap_auto
from evaluate import MSE, MAPE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ... (你代码中的所有辅助函数 seed_everything, is_standardized_data 等保持不变) ...
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
    """检测数据是否为标准化数据"""
    train_filename = os.path.basename(train_data_path)
    val_filename = os.path.basename(val_data_path)
    if train_filename.startswith('std') or val_filename.startswith('std'):
        return True
    if 'std' in train_filename.lower() or 'std' in val_filename.lower():
        return True
    standardized_keywords = ['standard', 'standardized', 'norm', 'normalized']
    for keyword in standardized_keywords:
        if keyword in train_filename.lower() or keyword in val_filename.lower():
            return True
    return False

def create_full_graph_dict(data_length, num_nodes):
    """动态创建完全图tensor格式"""
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_list.append([i, j])
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).T
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    full_graph_list = [edge_index.clone() for _ in range(data_length)]
    return full_graph_list

# ==============================================================================
# 1. 全局配置 (Global Configuration) - 保持不变
# ==============================================================================
# ... (你的 DATASET_CONFIGS 和 CONFIG 保持不变) ...
DATASET_CONFIGS = {
    "fujian": {
        "csv_file": "Offshore Wind Farm Dataset3(WT1).csv",
        "hidden_size": 264,
        "feature_map": {
            'Pres_Pa': 'Pressure',
            'RH_pct': 'Humidity (RH)',
            'Cloud': 'Cloud Cover',
            'WS10m': 'WS (10m)',
            'WD10m': 'WD (10m)',
            'Temp_K': 'Temperature',
            'Rad_Jm2': 'Solar Rad.',
            'Precip_m': 'Precipitation',
            'WS100m': 'WS (100m)',
            'WD100m': 'WD (100m)',  
            'y': 'Past Power' 
        },
        "base_model_params": {
            "n_head": 8, "factor": 2, "dropout": 0.5, "conv_hidden_size": 32,
            "MovingAvg_window": 3, "activation": "gelu", "encoder_layers": 1,
            "decoder_layers": 1, "gruop_dec": True
        },
        "scales": {
            "1": {"dataset_name": "1", "seq_lenth": 6, "c_out": 1, "c_in": 11, "prediction_type": "单步", "display_name": "1天单步"},
            "6-0_1": {"dataset_name": "6-0_1", "seq_lenth": 36, "c_out": 1, "c_in": 11, "prediction_type": "单步", "display_name": "6天单步"},
            "6-1": {"dataset_name": "6-1", "seq_lenth": 36, "c_out": 6, "c_in": 11, "prediction_type": "多步", "display_name": "6天多步"},
            "24-1": {"dataset_name": "24-1", "seq_lenth": 36, "c_out": 6, "c_in": 11, "prediction_type": "多步", "display_name": "24天多步"}
        }
    },
    "DSWE": {
        "csv_file": "Offshore Wind Farm Dataset1(WT5).csv",
        "hidden_size": 256,
        "feature_map": {
            'V': 'Wind Speed',
            'D': 'Wind Direction',
            'rho': 'Air Density',
            'H': 'Humidity',
            'I': 'Turbulence Int.', 
            'S_a': 'Wind Shear (Above)',
            'S_b': 'Wind Shear (Below)',
            'y': 'Past Power' 
        },
        "base_model_params": {
            "n_head": 8, "factor": 2, "dropout": 0.5, "conv_hidden_size": 32,
            "MovingAvg_window": 3, "activation": "gelu", "encoder_layers": 1,
            "decoder_layers": 1, "gruop_dec": True
        },
        "scales": {
            "1": {"dataset_name": "1", "seq_lenth": 6, "c_out": 1, "c_in": 8, "prediction_type": "单步", "display_name": "1小时单步"},
            "6-0_1": {"dataset_name": "6-0_1", "seq_lenth": 36, "c_out": 1, "c_in": 8, "prediction_type": "单步", "display_name": "6小时单步"},
            "6-1": {"dataset_name": "6-1", "seq_lenth": 36, "c_out": 6, "c_in": 8, "prediction_type": "多步", "display_name": "6小时多步"},
            "24-1": {"dataset_name": "24-1", "seq_lenth": 144, "c_out": 6, "c_in": 8, "prediction_type": "多步", "display_name": "24小时多步"}
        }
    }
}
CONFIG = {
    "target_dataset": "DSWE",
    "target_scales": ["1"],
    "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    "split_ratio": 0.99,
    "background_samples": 50,
    "test_samples": 200,
    "seed": 42,
    "script_dir": os.path.dirname(os.path.abspath(__file__)),
    "output_dir": Path("shap_analysis_results"),
    "use_adaptive_graph": True,
    "visualization": {"font_path": None, "font_size": 12, "dpi": 300}
}
# ... (get_current_config, 字体设置等保持不变) ...
def get_current_config(dataset_name, scale_name):
    if dataset_name not in DATASET_CONFIGS: raise ValueError(f"未支持的数据集: {dataset_name}")
    dataset_config = DATASET_CONFIGS[dataset_name]
    if scale_name not in dataset_config["scales"]: raise ValueError(f"数据集 {dataset_name} 不支持尺度: {scale_name}")
    scale_config = dataset_config["scales"][scale_name]
    model_params = dataset_config["base_model_params"].copy()
    model_params.update({"hidden_size": dataset_config["hidden_size"], "seq_lenth": scale_config["seq_lenth"], "c_out": scale_config["c_out"], "c_in": scale_config["c_in"]})
    return {"dataset_config": dataset_config, "scale_config": scale_config, "model_params": model_params}

if CONFIG["visualization"]["font_path"] and Path(CONFIG["visualization"]["font_path"]).exists():
    font_prop = fm.FontProperties(fname=CONFIG["visualization"]["font_path"])
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    print("字体路径未找到，使用默认字体。")
plt.rcParams['font.size'] = CONFIG["visualization"]["font_size"]
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 2. 模型封装 (Model Wrapper for SHAP) - 保持不变
# ==============================================================================
# ... (你的 ModelWrapper 类保持不变) ...
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, edge_index_subset, device):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.device = device
        if isinstance(edge_index_subset, list):
            self.edge_index = [edge.to(device) if isinstance(edge, torch.Tensor) else edge for edge in edge_index_subset]
        else:
            self.edge_index = edge_index_subset
    def forward(self, x):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float()
        x = x.to(self.device)
        M_edge = self.edge_index[0:x.shape[0]]
        if isinstance(M_edge, list):
            M_edge = [edge.to(self.device) if isinstance(edge, torch.Tensor) else edge for edge in M_edge]
        return self.model(x, M_edge)

# ==============================================================================
# 3. 核心功能函数 (Core Functional Components) - 保持不变
# ==============================================================================
# ... (你的 load_data_and_model 和 calculate_shap_values 函数保持不变) ...
def load_data_and_model(dataset_name, scale_name, current_config):
    print(f"--- 加载数据集和模型: {dataset_name}数据集 {scale_name}预测")
    seed_everything(seed=CONFIG["seed"])
    dataset_config = current_config["dataset_config"]
    scale_config = current_config["scale_config"]
    model_params = current_config["model_params"]
    script_dir = CONFIG["script_dir"]
    data_dir = os.path.join(script_dir, f'../data/{dataset_name}')
    actual_dataset_name = scale_config["dataset_name"]
    train_dir = os.path.join(script_dir, f'../data/{dataset_name}/stdtrain_data{actual_dataset_name}.npy')
    val_dir = os.path.join(script_dir, f'../data/{dataset_name}/stdval_data{actual_dataset_name}.npy')
    csv_dir = os.path.join(script_dir, f'../data/{dataset_name}/{dataset_config["csv_file"]}')
    if not os.path.exists(train_dir): raise FileNotFoundError(f"❌ 训练数据不存在: {train_dir}")
    if not os.path.exists(val_dir): raise FileNotFoundError(f"❌ 验证数据不存在: {val_dir}")
    # 加载原始特征名
    try:
        data = pd.read_csv(csv_dir, nrows=6)
        if dataset_name == "DSWE":
            df = data.drop(['Sequence No.'], axis=1)
            # 假设DSWE数据集的最后一列是'y'
            original_feature_names = df.columns
        else: # fujian
            df = data.drop(['Site_ID', 'Timestamp'], axis=1)
            # 假设fujian数据集的最后一列是'y'
            original_feature_names = df.columns

        print(f"✅ Original feature names loaded: {original_feature_names}")
        
        # ✨【新增】使用映射字典翻译特征名
        feature_map = dataset_config.get("feature_map", {})
        translated_feature_names = [feature_map.get(name, name) for name in original_feature_names]
        print(f"✅ Translated feature names for paper: {translated_feature_names}")
        
        current_config["model_params"]["c_in"] = len(translated_feature_names)
        
    except FileNotFoundError:
        print(f"Warning: Feature name CSV not found. Using generic names.")
        original_feature_names = [f'feature_{i+1}' for i in range(current_config["model_params"]["c_in"])]
        translated_feature_names = original_feature_names

    x_data = torch.tensor(np.load(train_dir), dtype=torch.float32)
    y_data_raw = np.load(val_dir)
    if scale_config["prediction_type"] == "单步":
        y_data = torch.tensor(y_data_raw[:, 0, 0], dtype=torch.float32).unsqueeze(-1)
        current_config["model_params"]["c_out"] = 1
    else:
        y_data = torch.tensor(y_data_raw[:, :, 0], dtype=torch.float32)
        current_config["model_params"]["c_out"] = scale_config["c_out"]
    use_adaptive_graph = CONFIG["use_adaptive_graph"]
    if use_adaptive_graph:
        if dataset_name == "fujian": edge_dir = os.path.join(script_dir, f'../new_data/fujian/adag_dict_train_data{actual_dataset_name}_fused.pkl')
        elif dataset_name == "DSWE": edge_dir = os.path.join(script_dir, f'../new_data/DSWE/adag_dict_{actual_dataset_name}.pkl')
        else: edge_dir = os.path.join(script_dir, f'../new_data/{dataset_name}/adag_dict_{actual_dataset_name}.pkl')
        if not os.path.exists(edge_dir): use_adaptive_graph = False
        else:
            with open(edge_dir, 'rb') as f: edge_index = pickle.load(f)
            if len(edge_index) != len(x_data): use_adaptive_graph = False
    if not use_adaptive_graph:
        edge_index = create_full_graph_dict(len(x_data), x_data.shape[-1])
    split_index = int(len(x_data) * CONFIG["split_ratio"])
    X_train, X_test = x_data[:split_index], x_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]
    train_dict, test_dict = edge_index[:split_index], edge_index[split_index:]
    auto_model = adap_auto(**model_params).to(CONFIG["device"])
    auto_model.eval()
    X_train, X_test = X_train.to(CONFIG["device"]), X_test.to(CONFIG["device"])
    wrapped_model = ModelWrapper(auto_model, test_dict, CONFIG["device"])
    return X_train, X_test, test_dict, wrapped_model, translated_feature_names, None, True

def calculate_shap_values(wrapped_model, X_train, X_test, current_config):
    print("--- 计算SHAP值 ---")
    background_data = X_train[np.random.choice(X_train.shape[0], CONFIG["background_samples"], replace=False)]
    if len(X_test) > CONFIG["test_samples"]:
        X_test_subset = X_test[np.random.choice(X_test.shape[0], CONFIG["test_samples"], replace=False)]
    else:
        X_test_subset = X_test
    explainer = shap.GradientExplainer(wrapped_model, background_data)
    shap_values = explainer.shap_values(X_test_subset)
    if isinstance(shap_values, list): shap_values = shap_values[0]
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1: shap_values = shap_values.squeeze(-1)
    return shap_values, X_test_subset.cpu().numpy()

# ==============================================================================
# 4. 可视化函数 (Visualization Function) - ✨【核心修改处】✨
# ==============================================================================

def plot_feature_importance(shap_values, feature_names, dataset_name, scale_name, current_config):
    """
    生成具有放大元素、优化配色、使用新罗马字体并符合英文科技论文风格的特征重要性棒棒糖图。
    """
    print(f"--- 生成优化的特征重要性图: {dataset_name} Dataset, {scale_name} Prediction ---")

    if shap_values.ndim >= 2:
        axes_to_agg = tuple(i for i in range(shap_values.ndim) if i != shap_values.ndim - 2)
        global_importance = np.abs(shap_values).mean(axis=axes_to_agg)
    else:
        print(f"SHAP value shape ({shap_values.shape}) not supported for plotting.")
        return

    total_importance = np.sum(global_importance)
    percentages = (global_importance / total_importance) * 100 if total_importance > 0 else 0
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': global_importance, 'Percentage': percentages}).sort_values(by='Importance', ascending=True)

    # 设置新罗马字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14 # 增大默认字体大小

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制棒棒糖图，增大线条宽度和点的大小
    ax.hlines(
        y=importance_df['Feature'],
        xmin=0,
        xmax=importance_df['Importance'],
        color='#1B9E77', # 调整颜色 (更鲜明的蓝色)
        alpha=0.6,
        linewidth=5 # 增大线条宽度
    )
    ax.scatter(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        s=300, # 增大点的大小
        color='#1B9E77',
        alpha=1,
        zorder=3
    )

    # 添加精确的文本标注 (绝对值 + 百分比)，增大字体大小
    for index, row in importance_df.iterrows():
        ax.text(
            x=row['Importance'],
            y=row['Feature'],
            s=f"   {row['Importance']:.4f} ({row['Percentage']:.1f}%)",
            color='black',
            fontsize=18, # 增大标注字体大小
            fontweight='normal',
            verticalalignment='center'
        )

    scale_config = current_config["scale_config"]
    display_name = scale_config.get("display_name", scale_name)
    ax.set_title(f'Feature Importance Analysis for {dataset_name}', fontsize=24, pad=25, weight='bold')
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=22, weight='bold')
    ax.set_ylabel('Feature', fontsize=22, weight='bold')

    ax.set_xlim(0, importance_df['Importance'].max() * 1.1) # 留出更多空间给文本
    ax.tick_params(axis='y', length=0, labelsize=18) # 增大Y轴特征名字体
    ax.tick_params(axis='x', labelsize=18)

    # 更柔和的网格线
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    output_dir = CONFIG["output_dir"] / dataset_name
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'feature_importance_optimized_{scale_name}.pdf'
    fig.savefig(output_path, dpi=CONFIG["visualization"]["dpi"], bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved optimized feature importance plot to: {output_path}")

def plot_importance_stacked_barchart(shap_values, feature_names, dataset_name, scale_name, current_config):
    """
    生成一个特征重要性贡献度的100%堆叠条形图。
    """
    print(f"--- 生成堆叠条形图以供参考: {dataset_name}数据集 {scale_name}预测 ---")

    # 1. 聚合SHAP值，计算每个特征的全局重要性
    if shap_values.ndim >= 2:
        axes_to_agg = tuple(i for i in range(shap_values.ndim) if i != shap_values.ndim - 2)
        global_importance = np.abs(shap_values).mean(axis=axes_to_agg)
    else:
        print(f"SHAP值形状 ({shap_values.shape}) 不支持，无法生成图表。")
        return

    # 2. 计算贡献度百分比并排序
    total_importance = np.sum(global_importance)
    if total_importance == 0:
        print("总重要性为0，无法生成堆叠条形图。")
        return
        
    percentages = (global_importance / total_importance) * 100
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'percentage': percentages
    }).sort_values(by='percentage', ascending=False) # 按重要性降序排列

    # 3. 准备绘图数据和颜色
    labels = importance_df['feature']
    sizes = importance_df['percentage']
    num_features = len(labels)
    # 使用tab20这种多颜色colormap，适合区分多个类别
    colors = plt.cm.get_cmap('tab20', num_features)(range(num_features)) 

    # 4. 开始绘图
    fig, ax = plt.subplots(figsize=(15, 5))
    
    left = 0 # 每一块的起始位置
    for i, (label, size) in enumerate(zip(labels, sizes)):
        # 绘制单个色块
        ax.barh(y=0, width=size, height=0.5, left=left, color=colors[i], label=label, edgecolor='white')
        
        # 在色块中间添加文本标签
        text_color = 'white' if size > 7 else 'black' # 如果色块太小，文字放外面可能更好，这里简化处理
        ax.text(left + size/2, 0, f'{size:.1f}%', ha='center', va='center', color=text_color, fontsize=9, weight='bold')
        
        left += size # 更新下一个色块的起始位置

    # 5. 美化图表
    scale_config = current_config["scale_config"]
    display_name = scale_config.get("display_name", scale_name)
    ax.set_title(f'Feature Importance Composition - {dataset_name} ({display_name})', fontsize=16, pad=20, weight='bold')
    
    # 设置X轴
    ax.set_xlim(0, 100)
    ax.set_xlabel('Contribution Percentage (%)', fontsize=12)
    
    # 隐藏Y轴，因为它没有意义
    ax.get_yaxis().set_visible(False)
    
    # 移除边框
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)
        
    # 添加图例
    ax.legend(title='Features', loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    plt.tight_layout()

    # 6. 保存图表
    output_dir = CONFIG["output_dir"] / dataset_name
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'stacked_barchart_reference_{scale_name}.pdf'
    fig.savefig(output_path, dpi=CONFIG["visualization"]["dpi"], bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 已保存堆叠条形图(参考)到: {output_path}")
# ==============================================================================
# 5. 主执行函数 (Main Execution) - ✨【核心修改处】✨
# ==============================================================================
def main():
    """主执行流程 - 同时生成高质量棒棒糖图和堆叠条形图以供对比。"""
    print("🔬 开始SHAP特征重要性分析...")
    
    CONFIG["output_dir"].mkdir(exist_ok=True)
    target_datasets = [CONFIG["target_dataset"]] if isinstance(CONFIG["target_dataset"], str) else CONFIG["target_dataset"]

    for dataset_name in target_datasets:
        dataset_output_dir = CONFIG["output_dir"] / dataset_name
        dataset_output_dir.mkdir(exist_ok=True)
        print(f"\n{'='*80}\n📂 开始分析数据集: {dataset_name}\n{'='*80}")

        for scale_name in CONFIG["target_scales"]:
            try:
                print(f"\n--- 📈 处理尺度: {scale_name} ---")
                
                # 1. 加载数据和模型
                current_config = get_current_config(dataset_name, scale_name)
                X_train, X_test, _, wrapped_model, feature_names, _, _ = load_data_and_model(dataset_name, scale_name, current_config)

                # 2. 计算SHAP值
                shap_values, X_test_subset = calculate_shap_values(wrapped_model, X_train, X_test, current_config)

                # 3. 生成并保存两种图表
                
                # 3a. 生成主要的、推荐用于论文的棒棒糖图
                plot_feature_importance(
                    shap_values, feature_names, dataset_name, scale_name, current_config
                )

                # 3b. ✨【新增调用】✨ 生成堆叠条形图，作为对比参考
                # plot_importance_stacked_barchart(
                #     shap_values, feature_names, dataset_name, scale_name, current_config
                # )

            except FileNotFoundError as e:
                print(f"❌ 处理数据集 {dataset_name}-{scale_name} 时出错: {e}。跳过此配置。")
                continue
            except Exception as e:
                print(f"💥 处理数据集 {dataset_name}-{scale_name} 时发生意外错误: {e}。跳过此配置。")
                import traceback
                traceback.print_exc()
                continue

    print("\n✅ SHAP特征重要性分析全部完成!")
    print(f"📁 所有结果文件保存在: {CONFIG['output_dir']}")

if __name__ == '__main__':
    main()