import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16

# 数据
sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
DCNN_results = [90.56, 93.02, 94.04, 95.15, 95.89]
M3DCNN_results = [87.78, 91.51, 91.78, 93.48, 94.92]
HybidSN_results = [91.93, 95.64, 97.08, 97.73, 97.48]
RNN_results = [83.31, 87.9, 91.24, 93.76, 92.62]
SSRN_results = [90.45, 95.26, 95.84, 97.64, 97.69]
GCN_results = [91.56, 94.38, 95.19, 96.17, 96.75]
F2HGNNss_results = [92.83, 95.2, 95.67, 97.15, 97.01]
EHGCN_results = [93.78, 95.98, 96.52, 97.87, 98.04]

models = ['2DCNN', 'M3DCNN', 'HybridSN', 'RNN', 'SSRN', 'GCN', 'F2HGNN', 'EHGCN']
node_shapes = ['o', '^', 'v', 'D', 's', 'p', 'X', '*']  # 分配形状
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#ff4136']  # 更新颜色
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # 不同的线条样式

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制折线图
for i, model_results in enumerate([DCNN_results, M3DCNN_results, HybidSN_results, RNN_results, SSRN_results, GCN_results, F2HGNNss_results, EHGCN_results]):
    plt.plot(sample_ratios, model_results,
             marker=node_shapes[i], markersize=10, markeredgewidth=1.5,
             color=colors[i], linestyle=linestyles[i], linewidth=2.5,
             label=models[i])

# 设置坐标轴范围
plt.xlim(0.05, 0.55)
plt.ylim(80, 100)

# 添加坐标轴标签
plt.xlabel('Training Label Ratio (%)', fontsize=18)
plt.ylabel('OA (%)', fontsize=18)

# 设置刻度间距
plt.xticks(sample_ratios)
plt.yticks(np.arange(80, 101, 2))

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)

# 调整图例位置和样式
# 将图例分为两列，第一列3个方法，第二列5个方法
legend = plt.legend(loc='lower right', fontsize=14, ncol=2,
                    frameon=True, shadow=True,
                    edgecolor='black', facecolor='white')
legend.get_frame().set_alpha(0.9)

# 调整边距
plt.tight_layout()

# 保存图形（提高DPI）
plt.savefig("Diff_label_EHGCN_Xuzhou.png", format='png', transparent=True, dpi=400)
plt.savefig("Diff_label_EHGCN_Xuzhou.svg", format='svg', transparent=True, dpi=400)

# 显示图形
plt.show()