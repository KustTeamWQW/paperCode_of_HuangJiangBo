import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

# 模擬一些實驗結果數據（這裡的數據是假設的，請根據實際數據進行替換）
sample_ratios = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
cnn_results = [74.97,	78.62,	86.73,	89.06,	91.26,	91.73,	91.81,	92.27,	93.31,	93.43]
hgcn_results = [85.53,	86.25,	86.73,	89.07,	91.15,	91.43,	91.46,	92.37,	92.01,	92.88]
gcn_results = [80.86,	82.76,	83.8,	85.59,	87.6,	89.06,	90.92,	90.89,	90.74,	91.02]

# 畫出折綫圖
plt.plot(sample_ratios, cnn_results, marker='*', color=(46/255, 117/255, 182 / 255),linestyle='--',linewidth=3, markersize=10,label='CNN')
plt.plot(sample_ratios, gcn_results, marker='d',color=(235 / 255, 129/ 255, 152/255), linestyle='--',linewidth=3, markersize=8,label='GCN')
plt.plot(sample_ratios, hgcn_results, marker='o', color=(240/255, 16/255,17 / 255),linestyle='--',linewidth=3, markersize=8,label='HGCN')
# 添加標題和軸標籤
# plt.title('實驗結果折綫圖')
plt.xlabel('Training Label Rationing')
plt.ylabel('Overall Accuracy(%)')
plt.xticks(sample_ratios)
# 添加圖例
plt.legend()
plt.savefig("CNN_GCN_HGCN"+ '.png', format='png', transparent=True, dpi=400)
# 顯示折綫圖
plt.show()