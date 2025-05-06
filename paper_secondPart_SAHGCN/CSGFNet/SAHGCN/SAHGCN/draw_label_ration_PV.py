import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置全局字体和样式
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['axes.labelsize'] = 18  # 调整为与 Xuzhou 图一致
rcParams['xtick.labelsize'] = 16  # 调整为与 Xuzhou 图一致
rcParams['ytick.labelsize'] = 16  # 调整为与 Xuzhou 图一致

# 数据
sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

M3DCNN_results = [80.36, 83.4, 87.51, 88.56, 90.93]
HybidSN_results = [76.84, 87.18, 87.59, 92.21, 94.01]
SSRN_results = [88.54, 92.5, 93.04, 93.47, 97.01]
GCN_results = [86.44, 88.92, 90.66, 89.46, 91.25]
F2HGNNss_results = [89.2, 91.04, 90.89, 91.29, 94.26]
CEGCN_results = [90.47, 94.23, 95.27, 96.67, 97.03]
MambaHSI_results = [87.29, 91.49, 92.3, 93.83, 94.92]
SAHGCN_results = [93.78, 95.48, 97.04, 97.87, 98.26]

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
plt.savefig("Diff_label_SAHGCN_PV.png", format='png', transparent=True, dpi=400)
plt.show()