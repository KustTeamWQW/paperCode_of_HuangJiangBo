import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# 设置全局字体样式
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12

# 创建图形
plt.figure(figsize=(8, 6))

segsize = [50,100,150,200]
PaviaU_results = [92.02,88.92,86.49,84.41]
Xuzhou_results = [93.78,92.2,91.49,91.09]

# 绘制折线图（修改颜色和样式）
pavia_line, = plt.plot(segsize, PaviaU_results,
                      marker='*', markersize=10, markeredgewidth=1.5,
                      color='#2E86C1', linestyle='-', linewidth=2.5,
                      label='University of Pavia')

xuzhou_line, = plt.plot(segsize, Xuzhou_results,
                       marker='^', markersize=10, markeredgewidth=1.5,
                       color='#E74C3C', linestyle='-', linewidth=2.5,
                       label='Xuzhou')

# 设置坐标轴范围
plt.xlim(40, 210)
plt.ylim(82, 95)

# 添加坐标轴标签（包含λ符号）
plt.xlabel('Segmentation Scale (λ)', fontsize=14, labelpad=10)
plt.ylabel('OA (%)', fontsize=14, labelpad=10)

# 设置刻度间距
plt.xticks(np.arange(50, 210, 50))
plt.yticks(np.arange(82, 96, 2))

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)

# 优化图例（左上角，添加阴影效果）
legend = plt.legend(loc='upper right', fontsize=12,
                   frameon=True, shadow=True,
                   edgecolor='black', facecolor='white')
legend.get_frame().set_alpha(0.9)

# 调整边距
plt.tight_layout()

# 保存图形（提高DPI）
plt.savefig("seg_seize1.png", format='png',
           transparent=True, dpi=400, bbox_inches='tight')

# 显示图形
plt.show()