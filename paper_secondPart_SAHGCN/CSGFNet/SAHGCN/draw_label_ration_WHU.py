import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['axes.labelsize'] = 18  # 调整为与 Xuzhou 图一致
rcParams['xtick.labelsize'] = 16  # 调整为与 Xuzhou 图一致
rcParams['ytick.labelsize'] = 16  # 调整为与 Xuzhou 图一致
# 数据
sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

M3DCNN_results = [79.49, 86.22, 88.1, 88.7, 90.68]
HybidSN_results = [89.13, 93.29, 95.75, 96.53, 96.86]
SSRN_results = [84.67, 88.47, 91.2, 89.29, 92.61]
GCN_results = [88.78, 91.88, 92.67, 93.57, 95.3]
F2HGNNss_results = [91.46, 93.17, 94.17, 93.87, 94.04]
CEGCN_results = [91.55, 94.75, 94.19, 96.45, 96.68]
MambaHSI_results = [87.12, 93.1, 94.39, 95.58, 96.7]
SAHGCN_results = [92.84, 95.48, 96.03, 96.54, 97.26]

models = ['M3DCNN', 'HybidSN', 'SSRN', 'GCN', 'F2HGNN', 'CEGCN', 'MambaHSI', 'SAHGCN']

# 颜色和形状
node_shapes = ['o', '^', 'v', 'D', 's', 'p', 'X', '*']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 'red']  # SAHGCN 使用红色

# 图表设置
plt.figure(figsize=(10, 6))  # 调整图表大小

# 绘制折线图
for i, model_results in enumerate([M3DCNN_results, HybidSN_results, SSRN_results, GCN_results, F2HGNNss_results, CEGCN_results, MambaHSI_results, SAHGCN_results]):
    plt.plot(sample_ratios, model_results,
             marker=node_shapes[i],
             label=models[i],
             color=colors[i],
             markersize=12,
             linewidth=2.5,
             linestyle='-')  # 统一线型为实线，调整宽度和大小

# 设置坐标轴标签和标题
plt.xlabel('Training Label Ratio (%)', fontsize=18, labelpad=10)
plt.ylabel('OA (%)', fontsize=18, labelpad=10)

# 设置刻度间距
plt.xticks(sample_ratios)
plt.yticks(np.arange(75, 101, 5))

# 添加仅横向虚线网格
plt.grid(axis='y', color='gray', linestyle='--', linewidth=1.0, alpha=0.8)  # 横向网格，颜色稍深，更清晰

# 设置图例
plt.legend(ncol=2, loc='lower right', fontsize=14, frameon=True, edgecolor='black', framealpha=0.9)

# 保存和显示图表
plt.savefig("Diff_label_SAHGCN_WHU.png", format='png', transparent=True, dpi=400)
plt.show()