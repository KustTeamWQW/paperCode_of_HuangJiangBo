import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# 设置全局字体样式
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['axes.labelsize'] = 18  # 调整为与 Xuzhou 图一致
rcParams['xtick.labelsize'] = 16  # 调整为与 Xuzhou 图一致
rcParams['ytick.labelsize'] = 16  # 调整为与 Xuzhou 图一致

# 数据
sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
DCNN_results = [71.82, 75.63, 80.57, 82.86, 84.33]
M3DCNN_results = [80.36, 83.4, 87.51, 88.56, 90.93]
HybidSN_results = [76.84, 87.18, 87.59, 92.21, 94.01]
RNN_results = [72.23, 76.23, 72.86, 78.01, 80.83]
SSRN_results = [88.54, 92.5, 93.04, 93.47, 97.01]
GCN_results = [86.44, 88.92, 90.66, 89.46, 91.25]
F2HGNNss_results = [89.2, 91.04, 90.89, 91.29, 94.26]
EHGCN_results = [92.02, 94.85, 96.82, 97.83, 98.13]

models = ['2DCNN', 'M3DCNN', 'HybridSN', 'RNN', 'SSRN', 'GCN', 'F2HGNN', 'EHGCN']
node_shapes = ['o', '^', 'v', 'D', 's', 'p', 'X', '*']  # 分配形状
colors = ['#2E86C1', '#E74C3C', '#229954', '#F1C40F', '#884EA0', '#BA4A00', '#1ABC9C', '#FF4136']  # 使用更美观的颜色，EHGCN 使用红色系
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
plt.ylim(70, 100)

# 添加坐标轴标签
plt.xlabel('Training Label Ratio (%)', fontsize=18, labelpad=10)
plt.ylabel('OA (%)', fontsize=18, labelpad=10)

# 设置刻度间距
plt.xticks(sample_ratios)
plt.yticks(np.arange(70, 101, 5))

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
plt.savefig("model_comparison_PV.png", format='png',
            transparent=True, dpi=400, bbox_inches='tight')

# 显示图形
plt.show()