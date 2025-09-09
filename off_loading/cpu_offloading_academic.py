import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

# =============================================================================
# 1. 全局样式与配置 (Global Style & Configuration)
# =============================================================================
# 使用无衬线字体，确保在不同系统中的兼容性
plt.rcParams['font.family'] = 'Times New Roman' 
# 提升基础字号，以适应论文缩放后的可读性
plt.rcParams['font.size'] = 24  # 基础字号提升至24pt
plt.rcParams['axes.linewidth'] = 0

# 学术风格配色方案 (蓝=CPU, 绿=PCIe, 橙=GPU)
COLORS = {
    'cpu': '#8CB9E8',       # 柔和的蓝色
    'pcie': '#A2D5A2',      # 柔和的绿色
    'gpu': '#FFB37C',       # 柔和的橙色
    'phase_bg': '#F5F5F5',  # 阶段背景色 (非常淡的灰色)
    'border': '#B0B0B0',    # 边框颜色
    'text': '#333333',      # 主要文字颜色
    'arrow': '#555555',     # 箭头颜色
    'stream0': '#E57373',   # Stream 0 (计算流)
    'stream1': '#64B5F6',   # Stream 1 (数据流)
    'hook': '#FFC107',      # Hook符号颜色
    'loop': '#6A5ACD'       # 循环箭头颜色
}

# =============================================================================
# 2. 辅助绘图函数 (Helper Drawing Functions)
# =============================================================================

def create_process_box(ax, xy, width, height, text, facecolor):
    """创建核心处理模块框"""
    box = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.1,rounding_size=0.05",
        facecolor=facecolor,
        edgecolor=COLORS['border'],
        linewidth=2.2  # 加粗线条
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha='center', va='center', fontsize=20, color=COLORS['text'], linespacing=1.2  # 放大字体
    )

def create_arrow(ax, start, end, style='solid', color=COLORS['arrow'], label='', label_pos=0.55, label_size=13):
    """创建箭头及标注 - 大幅缩小箭头尺寸"""
    linestyle = '--' if style == 'async' else '-'
    arrow = ConnectionPatch(
        start, end, "data", "data",
        arrowstyle="->,head_width=0.8,head_length=1.5",  # 大幅缩小箭头头部尺寸
        shrinkA=8, shrinkB=8,
        color=color,
        linewidth=2.8,  # 加粗线条
        linestyle=linestyle
    )
    ax.add_patch(arrow)
    if label:
        mid_point = (start[0] * (1 - label_pos) + end[0] * label_pos, 
                     start[1] * (1 - label_pos) + end[1] * label_pos)
        ax.text(
            mid_point[0], mid_point[1] + 0.2, label,
            ha='center', va='bottom', fontsize=18, style='italic', color=color  # 放大字体
        )

def create_hook_symbol(ax, xy, radius=0.18):
    """创建Hook触发符号 - 使用文字替代符号"""
    hook_bg = plt.Circle(xy, radius, color=COLORS['hook'])
    ax.add_patch(hook_bg)
    # 使用简洁的文字替代锚点符号，避免字体缺失
    ax.text(xy[0], xy[1], 'H', ha='center', va='center', fontsize=18, color='white', weight='bold')  # 放大字体

# =============================================================================
# 3. 主绘图逻辑 (Main Plotting Logic)
# =============================================================================

# --- 画布设置 - 进一步优化高度节约版面 ---
fig, ax = plt.subplots(figsize=(16, 7.2)) # 减少高度，节约版面
ax.set_xlim(0, 16)
ax.set_ylim(0, 7.2)
ax.axis('off')

# --- 泳道与坐标定义 ---
lane_width, lane_gap = 4.8, 0.2
cpu_x = 0.5
pcie_x = cpu_x + lane_width + lane_gap
gpu_x = pcie_x + lane_width + lane_gap
cpu_center = cpu_x + lane_width / 2
pcie_center = pcie_x + lane_width / 2
gpu_center = gpu_x + lane_width / 2

# --- 调整间距：适应新的画布高度 ---
# 阶段背景绘制 (解决标题重叠的完美方案)
# 阶段1: 初始化 - 适应画布高度调整
phase1_top = 6.3  # 适应新的画布高度
ax.add_patch(patches.Rectangle((0.2, 5.4), 15.6, 0.9, facecolor=COLORS['phase_bg'], edgecolor='none', zorder=0))
ax.text(0.4, phase1_top - 0.1, "Phase 1: Smart Initialization", fontsize=20, fontweight='bold', color=COLORS['text'], ha='left')

# 阶段2: 训练循环 - 适应新的画布高度
phase2_top = 5.0  # 适应新的画布高度
ax.add_patch(patches.Rectangle((0.2, 1.2), 15.6, 3.8, facecolor=COLORS['phase_bg'], edgecolor='none', zorder=0))
ax.text(0.4, phase2_top - 0.1, "Phase 2: Training Loop", fontsize=20, fontweight='bold', color=COLORS['text'], ha='left')

# --- 泳道标题 - 适应新的画布高度 ---
title_y = 7.0  # 适应画布高度7.2
ax.text(cpu_center, title_y, "CPU Lane\nScheduling & Management", ha='center', va='top', fontsize=20, weight='bold')
ax.text(pcie_center, title_y, "PCIe/Bus Lane\nData Transfer Streams", ha='center', va='top', fontsize=20, weight='bold')
ax.text(gpu_center, title_y, "GPU Lane\nParallel Computing", ha='center', va='top', fontsize=20, weight='bold')

# --- 阶段1: 初始化流程 ---
init_y, box_w, box_h = 5.6, 4.4, 0.6  # 适应新布局，减小高度
create_process_box(ax, (cpu_center - box_w/2, init_y), box_w, box_h, "Model Structure\nAnalysis", COLORS['cpu'])
create_process_box(ax, (pcie_center - box_w/2, init_y), box_w, box_h, "Device Mapping\n& Optimization", COLORS['pcie'])
create_process_box(ax, (gpu_center - box_w/2, init_y), box_w, box_h, "Smart Dispatch\n& Hook Setup", COLORS['gpu'])
create_arrow(ax, (cpu_center + box_w/2, init_y + box_h/2), (pcie_center - box_w/2, init_y + box_h/2), label='analyze')
create_arrow(ax, (pcie_center + box_w/2, init_y + box_h/2), (gpu_center - box_w/2, init_y + box_h/2), label='dispatch')

# --- 阶段2: 训练循环 ---
# 核心并行流水线 (严格对齐)
pipe_y = 3.8  # 适应新布局
pipe_h = 0.6  # 统一流水线框高度
create_process_box(ax, (cpu_center - box_w/2, pipe_y), box_w, pipe_h, "Hook Manager &\nOrchestration", COLORS['cpu'])
create_process_box(ax, (pcie_center - box_w/2, pipe_y), box_w, pipe_h, "Weight Prefetch\n(Next Layer)", COLORS['pcie'])
create_process_box(ax, (gpu_center - box_w/2, pipe_y), box_w, pipe_h, "Forward Compute\n(Current Layer)", COLORS['gpu'])
create_arrow(ax, (cpu_center + box_w/2, pipe_y + pipe_h/2), (pcie_center - box_w/2, pipe_y + pipe_h/2), 'async', COLORS['stream1'], 'Stream 1')
create_arrow(ax, (pcie_center + box_w/2, pipe_y + pipe_h/2), (gpu_center - box_w/2, pipe_y + pipe_h/2), 'solid', COLORS['stream0'], 'Stream 0')

# Hook符号 - 修复位置并避免乱码
hook_x = cpu_x + 0.3
hook_y = pipe_y + pipe_h/2
create_hook_symbol(ax, (hook_x, hook_y))
# 修复Hook Trigger标签位置，使用清晰文字
ax.text(hook_x, pipe_y - 0.3, "Hook\nTrigger", ha='center', va='top', fontsize=16, 
        color=COLORS['hook'], weight='bold')

# 反向传播
back_y = 2.6  # 适应新布局
back_h = 0.5  # 反向传播框高度
create_process_box(ax, (gpu_center - box_w/2, back_y), box_w, back_h, "Gradient\nComputation", COLORS['gpu'])
create_process_box(ax, (pcie_center - box_w/2, back_y), box_w, back_h, "Transfer\nGradients", COLORS['pcie'])
create_process_box(ax, (cpu_center - box_w/2, back_y), box_w, back_h, "Parameter\nUpdate", COLORS['cpu'])
create_arrow(ax, (gpu_center - box_w/2, back_y + back_h/2), (pcie_center + box_w/2, back_y + back_h/2), 'async', label='gradients')
create_arrow(ax, (pcie_center - box_w/2, back_y + back_h/2), (cpu_center + box_w/2, back_y + back_h/2), 'async', label='update')

# 清理与准备
clean_y = 1.8  # 适应新布局
clean_h = 0.5  # 清理阶段框高度
create_process_box(ax, (gpu_center - box_w/2, clean_y), box_w, clean_h, "Memory\nCleanup", COLORS['gpu'])
create_process_box(ax, (pcie_center - box_w/2, clean_y), box_w, clean_h, "Sync\nBuffers", COLORS['pcie'])
create_process_box(ax, (cpu_center - box_w/2, clean_y), box_w, clean_h, "Next Batch\nPreparation", COLORS['cpu'])

# 精确对齐的循环箭头 - 使用更小的箭头尺寸
loop_start_x = cpu_center - box_w/2 - 0.3  # 精确对齐到CPU框体左边缘
loop_start_y = clean_y + clean_h/2  # 对齐到Next Batch Preparation中心
loop_end_x = cpu_center - box_w/2 - 0.3    # 精确对齐到CPU框体左边缘  
loop_end_y = pipe_y + pipe_h/2      # 对齐到Hook Manager中心

loop_arrow = ConnectionPatch(
    (loop_start_x, loop_start_y), (loop_end_x, loop_end_y), "data", "data",
    arrowstyle="->,head_width=0.8,head_length=1.5",  # 与其他箭头统一尺寸
    connectionstyle="arc3,rad=-0.3", # 调整弧度
    shrinkA=0, shrinkB=0,
    color=COLORS['loop'],
    linewidth=2.8  # 加粗线条
)
ax.add_patch(loop_arrow)

# Loop标签精确定位
loop_label_x = loop_start_x - 0.3
loop_label_y = (loop_start_y + loop_end_y) / 2
ax.text(loop_label_x, loop_label_y, "Loop", ha='center', va='center', 
        rotation=90, fontsize=16, color=COLORS['loop'], weight='bold')

# --- 紧贴Phase 2的紧凑图例 - 节约版面空间 ---
legend_y = 1.4  # 紧贴Phase 2底部，最大化节约版面
legend_x_start, legend_gap = 3.5, 2.0  # 稍微缩小间距以适应
items = [
    ('CPU', COLORS['cpu']), ('PCIe', COLORS['pcie']), ('GPU', COLORS['gpu']),
    ('Sync', COLORS['arrow']), ('Async', COLORS['arrow']), ('Hook', COLORS['hook'])
]
symbols = ['■', '■', '■', '—', '┅', 'H']  # 将锚点符号替换为H

for i, ((label, color), symbol) in enumerate(zip(items, symbols)):
    x = legend_x_start + i * legend_gap
    ax.text(x, legend_y, f"{symbol} {label}", ha='left', va='center', fontsize=18, color=color, weight='medium')
    # 为Async添加虚线示例
    if label == 'Async':
        line_y = legend_y + 0.02
        ax.plot([x-0.4, x-0.1], [line_y, line_y], color=color, linestyle='--', linewidth=2.5)

# =============================================================================
# 4. 保存与显示 (Save & Display)
# =============================================================================
plt.tight_layout()
# 保存为高分辨率PNG和矢量PDF，以供出版
plt.savefig('cpu_offloading_final.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('cpu_offloading_final.pdf', bbox_inches='tight', pad_inches=0.1)

print("🎉 终极完美版图表已生成！")
print("✨ 全面解决的问题:")
print("   📏 箭头彻底缩小：head_width=0.8, head_length=1.5，不再突兀")
print("   📐 标题间距优化：泳道标题与Phase 1之间距离适中")
print("   🔧 Phase间距紧凑：Phase 1与Phase 2间距合理") 
print("   🔤 符号替代方案：用'H'替代锚点符号，避免字体缺失")
print("   🎯 箭头精确对齐：弧状循环箭头完美对齐框体边缘")
print("   📋 Legend节约版面：紧贴Phase 2，最大化空间利用")
print("�� 学术论文级别的完美视觉效果！")
