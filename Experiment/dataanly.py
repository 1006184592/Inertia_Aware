import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity

# --- 数据加载和预处理部分 (与你原有的代码相同) ---
DATA_NAME = "fujian"
DATA_LENGTH = "24-0_1"

# 字号控制参数 - 可以通过修改这个值来调整整个图片的字号
BASE_FONT_SIZE = 20  # 基础字号，可以根据需要调整
"""
字号层级说明：
- 图表标题：BASE_FONT_SIZE + 4 （默认16）
- 坐标轴标签和图例：BASE_FONT_SIZE + 2 （默认14）  
- 刻度标签：BASE_FONT_SIZE （默认12）

使用示例：
- 要制作较小的图片，可设置 BASE_FONT_SIZE = 10
- 要制作较大的图片或演示用图，可设置 BASE_FONT_SIZE = 14 或 16
"""

# KDE 计算函数 (与你原有的代码相同，无需修改)
def get_kde_pair(data1, data2, bandwidth=5, kernel='gaussian'):
    eps = 1e-6
    data_min = min(data1.min(), data2.min())
    data_max = max(data1.max(), data2.max())
    step = (data_max - data_min) / 500
    # get kde
    kde1 = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(data1)
    kde2 = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(data2)
    # get boundary
    while True:
        p1, p2 = np.exp(kde1.score(np.array([data_min]).reshape(-1, 1))), np.exp(
            kde2.score(np.array([data_min]).reshape(-1, 1)))
        p = max(p1, p2)
        if p < eps:
            break
        else:
            data_min = data_min - step
    while True:
        p1, p2 = np.exp(kde1.score(np.array([data_max]).reshape(-1, 1))), np.exp(
            kde2.score(np.array([data_max]).reshape(-1, 1)))
        p = max(p1, p2)
        if p < eps:
            break
        else:
            data_max = data_max + step
    x = np.linspace(data_min, data_max, 500)
    # get curves
    kde1_curve = np.exp(kde1.score_samples(x.reshape(-1, 1)))
    kde2_curve = np.exp(kde2.score_samples(x.reshape(-1, 1)))
    return x, kde1_curve, kde2_curve

NUM_SAMPLES_PLOT = 500 # number of samples used to plot
# 假设数据已加载并处理好
# 为了可复现性，这里生成一些随机数据作为示例
# 在你的实际使用中，请替换回你自己的数据加载代码
script_dir = os.path.dirname(os.path.abspath(__file__)) 
data_dir = os.path.join(script_dir, f'../data/fujian')    
x_data = torch.tensor(np.load(os.path.join(data_dir, 'train_data24-0_1.npy'))).to(dtype=torch.float32)
y_data = torch.tensor(np.squeeze(np.load(os.path.join(data_dir, 'val_data24-0_1.npy'))[:, :, 0:1], axis=2)).to(dtype=torch.float32)

split_ratio = 0.8
split_index = int(len(x_data) * split_ratio)
X_train, X_test = x_data[0:split_index], x_data[split_index:]

X_train_full = np.stack([X_train[:, :, 0]], axis=0)
X_test_full = np.stack([X_test[:, :, 0]], axis=0)

X_train_full = torch.from_numpy(X_train_full).float()
X_test_full = torch.from_numpy(X_test_full).float()

X_train_full = X_train_full.transpose(1, 2).contiguous().view(-1, X_train.size(-2))
X_test_full = X_test_full.transpose(1, 2).contiguous().view(-1, X_test.size(-2))

# --- t-SNE 降维部分 (与你原有的代码相同) ---
train_random_indices = np.random.choice(X_train_full.size(0), NUM_SAMPLES_PLOT, replace=False)
test_random_indices = np.random.choice(X_test_full.size(0), NUM_SAMPLES_PLOT, replace=False)
train_data_sample = X_train_full[train_random_indices].numpy()
test_data_sample = X_test_full[test_random_indices].numpy()

all_data = np.concatenate([train_data_sample, test_data_sample], axis=0)

data_tsne = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=42).fit_transform(all_data)


train_tsne = data_tsne[:NUM_SAMPLES_PLOT]
test_tsne = data_tsne[NUM_SAMPLES_PLOT:2*NUM_SAMPLES_PLOT]

# --- 计算KDE曲线 (与你原有的代码相同) ---
up_x, up_kde1, up_kde2 = get_kde_pair(train_tsne[:, [0]], test_tsne[:, [0]])
right_x, right_kde1, right_kde2 = get_kde_pair(train_tsne[:, [1]], test_tsne[:, [1]])


# ==============================================================================
# --- 全新的、优化后的绘图代码 ---
# ==============================================================================

# 设置全局字体为新罗马
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也使用新罗马风格

# 创建Figure和子图网格
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                     left=0.1, right=0.9, bottom=0.1, top=0.9,
                     wspace=0.05, hspace=0.05)

# 分配子图位置
ax = fig.add_subplot(gs[1, 0])  # 主散点图
ax_kde_x = fig.add_subplot(gs[0, 0], sharex=ax)  # 上方KDE图
ax_kde_y = fig.add_subplot(gs[1, 1], sharey=ax)  # 右侧KDE图

# 使用更友好的颜色方案（蓝绿色+橙红色）
c_train = '#1B9E77'  # 蓝绿色
c_test = '#D95F02'   # 橙红色

# 绘制主散点图 - 放大标记和字体
ax.scatter(train_tsne[:, 0], train_tsne[:, 1], c=c_train, label='Training Data', 
           alpha=0.6, edgecolors='k', s=50, linewidths=0.5)
ax.scatter(test_tsne[:, 0], test_tsne[:, 1], c=c_test, label='Test Data', 
           alpha=0.6, edgecolors='k', s=50, linewidths=0.5)

# 设置主图属性 - 放大字体
ax.legend(title="Dataset", fontsize=BASE_FONT_SIZE+2, markerscale=1.5, title_fontsize=BASE_FONT_SIZE+2,
          frameon=True, fancybox=True)
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
ax.set_xlabel('t-SNE Component 1', fontsize=BASE_FONT_SIZE+2)
ax.set_ylabel('t-SNE Component 2', fontsize=BASE_FONT_SIZE+2)
ax.tick_params(axis='both', which='major', labelsize=BASE_FONT_SIZE)

# 绘制上方KDE图
ax_kde_x.plot(up_x, up_kde1, color=c_train, lw=2.5)
ax_kde_x.plot(up_x, up_kde2, color=c_test, lw=2.5)
ax_kde_x.fill_between(up_x, up_kde1, facecolor=c_train, alpha=0.3)
ax_kde_x.fill_between(up_x, up_kde2, facecolor=c_test, alpha=0.3)

# 设置上方KDE图属性
plt.setp(ax_kde_x.get_xticklabels(), visible=False)
ax_kde_x.set_yticks([])
ax_kde_x.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
ax_kde_x.tick_params(axis='y', which='major', labelsize=BASE_FONT_SIZE)

# 绘制右侧KDE图
ax_kde_y.plot(right_kde1, right_x, color=c_train, lw=2.5)
ax_kde_y.plot(right_kde2, right_x, color=c_test, lw=2.5)
ax_kde_y.fill_betweenx(right_x, right_kde1, facecolor=c_train, alpha=0.3)
ax_kde_y.fill_betweenx(right_x, right_kde2, facecolor=c_test, alpha=0.3)

# 设置右侧KDE图属性
plt.setp(ax_kde_y.get_yticklabels(), visible=False)
ax_kde_y.set_xticks([])
ax_kde_y.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
ax_kde_y.tick_params(axis='x', which='major', labelsize=BASE_FONT_SIZE)

# 添加整体标题
plt.suptitle(f't-SNE Distribution Comparison ({DATA_NAME} {DATA_LENGTH})', 
             fontsize=BASE_FONT_SIZE+4, y=0.95)

# 保存高质量的矢量图
plt.savefig(os.path.join(data_dir, f'kde_tsne_{DATA_NAME}_{DATA_LENGTH}.pdf'), 
            dpi=300, bbox_inches='tight')
plt.show()